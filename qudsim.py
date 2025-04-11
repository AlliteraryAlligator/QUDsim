
from preprocessing import number
from qud_generation import pipeline
from qud_alignment import align
import schema
import argparse
import pandas as pd
import numpy as np
import config
from tools import openai
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='qudsim.log', level=logging.INFO)


def _check_columns(columns, required_cols, superset_cols):
    # No extraneous fields outside of those specified by /schema.py can be included
    if not set(columns).issubset(set(superset_cols)):
        logger.error("Extraneous fields exist in the dataframe")
        return False
    
    # Required fields (specified) must all be included
    if not set(required_cols).issubset(set(columns)):
        logger.error("Required fields missing")
        return False

    return True

def _save_data(data, file) -> bool:
    try:
        data.to_json(file)
        return True
    except Exception as e:
        logger.error(e)
    return False

def reduce_df(extended_df):
    required_cols = [schema.PairColumns.DOC1, 
                     schema.PairColumns.DOC2, 
                     schema.PairColumns.LEVEL,
                     schema.PairColumns.MODEL1,
                     schema.PairColumns.MODEL2,
                     schema.PairColumns.QUDS1,
                     schema.PairColumns.QUDS2,
                     schema.PairColumns.HARMONIC_SCORE,
                     schema.PairColumns.ALIGNED_SEGMENT_TEXT]
    
    df_cols = [col_name for (const, col_name) in vars(schema.PairColumns).items() if const.isupper()]

    if not _check_columns(extended_df.columns, required_cols, df_cols):
        logger.error("Failed to create display dataframe, missing or extraneous fields in provided dataframe.")
        return None

    display_df = pd.DataFrame()
    display_df[schema.PairColumns.DOC1] = extended_df[schema.PairColumns.DOC1]
    display_df[schema.PairColumns.DOC2] = extended_df[schema.PairColumns.DOC2]
    display_df[schema.PairColumns.LEVEL] = extended_df[schema.PairColumns.LEVEL]
    display_df[schema.PairColumns.MODEL1] = extended_df[schema.PairColumns.MODEL1]
    display_df[schema.PairColumns.MODEL2] = extended_df[schema.PairColumns.MODEL2]
    display_df[schema.PairColumns.QUDS1] = extended_df[schema.PairColumns.QUDS1]
    display_df[schema.PairColumns.QUDS2] = extended_df[schema.PairColumns.QUDS2]
    display_df[schema.PairColumns.HARMONIC_SCORE] = extended_df[schema.PairColumns.HARMONIC_SCORE]
    display_df[schema.PairColumns.ALIGNED_SEGMENTS] = extended_df[schema.PairColumns.ALIGNED_SEGMENTS]
    display_df[schema.PairColumns.ALIGNED_SEGMENT_TEXT] = extended_df[schema.PairColumns.ALIGNED_SEGMENT_TEXT]
    return display_df

def id_documents(with_replacement: bool, level:int, query_df):
    # check to make sure df has all the appropriate columns and column names
    pair_df_cols = [col_name for (const, col_name) in vars(schema.PairColumns).items() if const.isupper()]

    if not _check_columns(query_df.columns, schema.REQUIRED_FIELDS, pair_df_cols):
        logger.error("Please provide a dataframe with the required columns: ", schema.REQUIRED_FIELDS)
        return
    
    document_data = []

    # id the docs
    if with_replacement:
        # reuse the same qud list if the same doc appears in other rows
        unique_docs = []
        models = []
        for index, row in query_df.iterrows():
            if row[schema.PairColumns.DOC1] not in unique_docs:
                unique_docs.append(row[schema.PairColumns.DOC1])
                models.append(row[schema.PairColumns.MODEL1])
            
            if row[schema.PairColumns.DOC2] not in unique_docs:
                unique_docs.append(row[schema.PairColumns.DOC2])
                models.append(row[schema.PairColumns.MODEL2])
            
        num_docs = len(unique_docs)
        if num_docs > (99999-10000):
            logger.error("Too many documents to handle at once.")
            return
        
        # create unique ids for each doc
        try:
            ids = np.random.choice(range(10000, 99999), num_docs, replace=False)
        except Exception as e:
            logger.error(e)

        # create document df for qg step
        document_data = [(id, level, model, doc) for (id, doc, model) in zip(ids, unique_docs, models)]
        df = pd.DataFrame(document_data, columns=[schema.DocumentColumns.ID, 
                                                  schema.DocumentColumns.LEVEL,
                                                  schema.DocumentColumns.MODEL,
                                                  schema.DocumentColumns.DOCUMENT])

        # update input query df with ids
        ids1 = []
        ids2 = []
        for index, row in query_df.iterrows():
            id1 = df[df[schema.DocumentColumns.DOCUMENT]==row[schema.PairColumns.DOC1]][schema.DocumentColumns.ID]
            id2 = df[df[schema.DocumentColumns.DOCUMENT]==row[schema.PairColumns.DOC2]][schema.DocumentColumns.ID]
            ids1.append(id1.item())
            ids2.append(id2.item())
        query_df[schema.PairColumns.ID1] = ids1
        query_df[schema.PairColumns.ID2] = ids2

    else:
        # treat each occurrence of the doc as a separate instance and generate new quds
        num_docs = len(query_df)*2
        if num_docs > (99999-10000):
            logger.error("Too many documents to handle at once.")
            return
        
        # create unique ids for each doc
        try:
            ids = np.random.choice(range(10000, 99999), num_docs, replace=False)
        except Exception as e:
            logger.error(e)

        
        # update input query df with ids
        query_df[schema.PairColumns.ID1] = ids[:len(ids)/2]
        query_df[schema.PairColumns.ID2] = ids[len(ids)/2:]

        # create document df for qg step
        document_data = [(id, level, mod1, doc) for (id, doc, mod1) in zip(query_df[schema.PairColumns.ID1], query_df[schema.PairColumns.DOC1], query_df[schema.PairColumns.MODEL1])]
        document_data.extend([(id, level, mod2, doc) for (id, doc, mod2) in zip(query_df[schema.PairColumns.ID2], query_df[schema.PairColumns.DOC2], query_df[schema.PairColumns.MODEL2])])

        df = pd.DataFrame(document_data, columns=[schema.DocumentColumns.ID, 
                                                  schema.DocumentColumns.LEVEL,
                                                  schema.DocumentColumns.MODEL,
                                                  schema.DocumentColumns.DOCUMENT])
    return df

def build_pairs_metadata(input_df, document_df):
    pairs_data = []
    pair_df_cols = [col_name for (const, col_name) in vars(schema.PairColumns).items() if const.isupper()]
    for index, row in input_df.iterrows():
        addable = True
        document1_row = document_df[document_df[schema.DocumentColumns.ID]==row[schema.PairColumns.ID1]]
        document2_row = document_df[document_df[schema.DocumentColumns.ID]==row[schema.PairColumns.ID2]]

        # check if any part is None - if so don't add it here, but log it
        if document1_row[schema.DocumentColumns.NUMBERED_TEXT].item() is None:
            logger.error("Numbered document is none for id %d" % row[schema.PairColumns.ID1])
            addable = False

        if document2_row[schema.DocumentColumns.NUMBERED_TEXT].item() is None:
            logger.error("Numbered document is none for id %d" % row[schema.PairColumns.ID2])
            addable = False

        if document1_row[schema.DocumentColumns.NUMBER_TO_SENTENCE].item() is None:
            logger.error("Number to sentence dictionary is none for id %d" % row[schema.PairColumns.ID1])
            addable = False

        if document2_row[schema.DocumentColumns.NUMBER_TO_SENTENCE].item() is None:
            logger.error("Number to sentence dictionary is none for id %d" % row[schema.PairColumns.ID2])
            addable = False

        if document1_row[schema.DocumentColumns.SEGMENTS].item() is None:
            logger.error("Segments are none for id %d" % row[schema.PairColumns.ID1])
            addable = False

        if document2_row[schema.DocumentColumns.SEGMENTS].item() is None:
            logger.error("Segments are none for id %d" % row[schema.PairColumns.ID2])
            addable = False

        if document1_row[schema.DocumentColumns.ABSTRACTED_ENTITIES].item() is None:
            logger.error("Abstracted segments are none for id %d" % row[schema.PairColumns.ID1])
            addable = False

        if document2_row[schema.DocumentColumns.ABSTRACTED_ENTITIES].item() is None:
            logger.error("Abstracted segments are none for id %d" % row[schema.PairColumns.ID2])
            addable = False

        if document1_row[schema.DocumentColumns.QUDS].item() is None:
            logger.error("QUDs are none for id %d" % row[schema.PairColumns.ID1])
            addable = False

        if document2_row[schema.DocumentColumns.QUDS].item() is None:
            logger.error("QUDs are none for id %d" % row[schema.PairColumns.ID2])
            addable = False

        if document1_row[schema.DocumentColumns.QUD_TO_SEGMENT].item() is None:
            logger.error("QUD to sentence dict is none for id %d" % row[schema.PairColumns.ID1])
            addable = False

        if document2_row[schema.DocumentColumns.QUD_TO_SEGMENT].item() is None:
            logger.error("QUD to sentence dict is none for id %d" % row[schema.PairColumns.ID2])
            addable = False

        if document1_row[schema.DocumentColumns.SEGMENT_TO_QUD].item() is None:
            logger.error("Segment to QUD dict is none for id %d" % row[schema.PairColumns.ID1])
            addable = False

        if document2_row[schema.DocumentColumns.SEGMENT_TO_QUD].item() is None:
            logger.error("Segment to QUD dict is none for id %d" % row[schema.PairColumns.ID2])
            addable = False


        if not addable:
            continue
        
        pairs_data.append((row[schema.PairColumns.ID1],
                            row[schema.PairColumns.ID2],
                            document1_row[schema.DocumentColumns.LEVEL].item(),
                            document1_row[schema.DocumentColumns.MODEL].item(),
                            document2_row[schema.DocumentColumns.MODEL].item(),
                            row[schema.PairColumns.DOC1],
                            row[schema.PairColumns.DOC2],
                            document1_row[schema.DocumentColumns.NUMBERED_TEXT].item(),
                            document2_row[schema.DocumentColumns.NUMBERED_TEXT].item(),
                            document1_row[schema.DocumentColumns.NUMBER_TO_SENTENCE].item(),
                            document2_row[schema.DocumentColumns.NUMBER_TO_SENTENCE].item(),
                            document1_row[schema.DocumentColumns.SEGMENTS].item(),
                            document2_row[schema.DocumentColumns.SEGMENTS].item(),
                            document1_row[schema.DocumentColumns.ABSTRACTED_ENTITIES].item(),
                            document2_row[schema.DocumentColumns.ABSTRACTED_ENTITIES].item(),
                            document1_row[schema.DocumentColumns.QUDS].item(),
                            document2_row[schema.DocumentColumns.QUDS].item(),
                            document1_row[schema.DocumentColumns.QUD_TO_SEGMENT].item(),
                            document2_row[schema.DocumentColumns.QUD_TO_SEGMENT].item(),
                            document1_row[schema.DocumentColumns.SEGMENT_TO_QUD].item(),
                            document2_row[schema.DocumentColumns.SEGMENT_TO_QUD].item()))
        
    pairs_df = pd.DataFrame(pairs_data, columns=pair_df_cols[:-5])
    return pairs_df

def preprocess_docs(documents_df):
    # df must have: id, doc, model and ldevel
    required_cols = [schema.DocumentColumns.ID, 
                     schema.DocumentColumns.LEVEL,
                     schema.DocumentColumns.MODEL,
                     schema.DocumentColumns.DOCUMENT]
    document_df_cols = [col_name for (const, col_name) in vars(schema.DocumentColumns).items() if const.isupper()]

    if not _check_columns(documents_df.columns, required_cols, document_df_cols):
        logger.error("Failed to preprocess documents, missing or extraneous fields in provided dataframe.")
        return None
    
    # sentence-tokenize and number the sentences
    preprocessed_data = []
    for index, row in tqdm(documents_df.iterrows(), total=len(documents_df), desc='Preprocessing'):
        numbered_text, number_sentence_dict = number.number_text(row[schema.DocumentColumns.DOCUMENT])

        if not numbered_text or not number_sentence_dict:
            logger.error("Could not preprocess document: ", row[schema.DocumentColumns.ID])

        preprocessed_data.append((row[schema.DocumentColumns.ID],
                                  row[schema.DocumentColumns.LEVEL],
                                  row[schema.DocumentColumns.MODEL],
                                  row[schema.DocumentColumns.DOCUMENT],
                                  numbered_text,
                                  number_sentence_dict))
        
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=[schema.DocumentColumns.ID, 
                                                               schema.DocumentColumns.LEVEL,
                                                               schema.DocumentColumns.MODEL,
                                                               schema.DocumentColumns.DOCUMENT,
                                                               schema.DocumentColumns.NUMBERED_TEXT,
                                                               schema.DocumentColumns.NUMBER_TO_SENTENCE])
    return preprocessed_df


def generate_quds(qg_gpt_model, documents_df: pd.core.frame.DataFrame):
    # df must have have the following columns; preprocessing step must have been successful
    required_cols = [schema.DocumentColumns.ID, 
                     schema.DocumentColumns.LEVEL,
                     schema.DocumentColumns.MODEL,
                     schema.DocumentColumns.DOCUMENT, 
                     schema.DocumentColumns.NUMBERED_TEXT,
                     schema.DocumentColumns.NUMBER_TO_SENTENCE]
    
    document_df_cols = [col_name for (const, col_name) in vars(schema.DocumentColumns).items() if const.isupper()]
    if not _check_columns(documents_df.columns, required_cols, document_df_cols):
        logger.error("Failed to generate QUDs, missing or extraneous fields in provided dataframe.")
        return None

    # generate quds for each document in df
    new_document_data = []
    for index, row in tqdm(documents_df.iterrows(), total=len(documents_df), desc='QUD Generation'):
        numbered_document = row[schema.DocumentColumns.NUMBERED_TEXT]
        level = row[schema.DocumentColumns.LEVEL]
        sentence_to_num_dict = row[schema.DocumentColumns.NUMBER_TO_SENTENCE]
        
        qg_item = pipeline.get_quds(qg_gpt_model, numbered_document, sentence_to_num_dict, level)
        if qg_item is None:
            logger.error("Could not segment, abstract or generate quds")
            segments = None
            entity_abstracted_segments = None
            quds = None
            segment_qud_dict = None
            qud_segment_dict = None
        else:
            segments = qg_item['segments']
            entity_abstracted_segments = qg_item['entity_abstracted_segments']
            quds = qg_item['quds']
            segment_qud_dict = qg_item['segment_qud_dict']
            qud_segment_dict = qg_item['qud_segment_dict']

        new_document_data.append((row[schema.DocumentColumns.ID],
                                  row[schema.DocumentColumns.LEVEL],
                                  row[schema.DocumentColumns.MODEL],
                                  row[schema.DocumentColumns.DOCUMENT],
                                  row[schema.DocumentColumns.NUMBERED_TEXT],
                                  row[schema.DocumentColumns.NUMBER_TO_SENTENCE],
                                  segments,
                                  entity_abstracted_segments,
                                  quds,
                                  qud_segment_dict,
                                  segment_qud_dict))
    
    df = pd.DataFrame(new_document_data, columns=document_df_cols)
    return df

def align_documents(qa_gpt_model, pairs_df):
    required_cols = [schema.PairColumns.ID1,
                     schema.PairColumns.ID2,
                     schema.PairColumns.LEVEL,
                     schema.PairColumns.MODEL1,
                     schema.PairColumns.MODEL2,
                     schema.PairColumns.DOC1,
                     schema.PairColumns.DOC2,
                     schema.PairColumns.NUMBERED_TEXT1,
                     schema.PairColumns.NUMBERED_TEXT2,
                     schema.PairColumns.NUMBER_TO_SENTENCE1,
                     schema.PairColumns.NUMBER_TO_SENTENCE2,
                     schema.PairColumns.SEGMENTS1,
                     schema.PairColumns.SEGMENTS2,
                     schema.PairColumns.ABSTRACTED_ENTITIES1,
                     schema.PairColumns.ABSTRACTED_ENTITIES2,
                     schema.PairColumns.QUDS1,
                     schema.PairColumns.QUDS2,
                     schema.PairColumns.QUD_TO_SEGMENT1,
                     schema.PairColumns.QUD_TO_SEGMENT2,
                     schema.PairColumns.SEGMENT_TO_QUD1,
                     schema.PairColumns.SEGMENT_TO_QUD2]
                     
    pair_df_cols = [col_name for (const, col_name) in vars(schema.PairColumns).items() if const.isupper()]


    if not _check_columns(pairs_df.columns, required_cols, pair_df_cols):
        logger.error("Failed to align documents, missing or extraneous fields in provided dataframe.")
        return None
    
    source_qud_ans_list = []
    target_qud_ans_list = []
    harmonic_mean_list = []
    aligned_segment_list = []
    aligned_segment_text_list = []
    for index, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc='Document Alignment'):
        num_source_segments = len(row[schema.PairColumns.SEGMENT_TO_QUD1])
        num_target_segments = len(row[schema.PairColumns.SEGMENT_TO_QUD2])
        num_source_sentences = len(row[schema.PairColumns.NUMBER_TO_SENTENCE1])
        num_target_sentences = len(row[schema.PairColumns.NUMBER_TO_SENTENCE2])
        source_segments = row[schema.PairColumns.SEGMENTS1]
        target_segments = row[schema.PairColumns.SEGMENTS2]
        
        source_qud_answers, target_qud_answers, harmonic_mean_scores, aligned_segments = align.align(qa_gpt_model,
                                                                                                     row[schema.PairColumns.QUDS1],
                                                                                                     row[schema.PairColumns.NUMBERED_TEXT2],
                                                                                                     row[schema.PairColumns.QUDS2],
                                                                                                     row[schema.PairColumns.NUMBERED_TEXT1],
                                                                                                     num_source_segments,
                                                                                                     num_target_segments,
                                                                                                     row[schema.PairColumns.QUD_TO_SEGMENT1],
                                                                                                     row[schema.PairColumns.QUD_TO_SEGMENT2],
                                                                                                     row[schema.PairColumns.SEGMENTS1],
                                                                                                     row[schema.PairColumns.SEGMENTS2],
                                                                                                     num_source_sentences,
                                                                                                     num_target_sentences)


        aligned_segment_text = []
        for i, src in enumerate(aligned_segments):
            for j, tgt in enumerate(src):
                if tgt>0:
                    # alignment exists
                    source_sentences = eval(source_segments)['segmentation'][i]['sentences']
                    try:
                        source_text = [row[schema.PairColumns.NUMBER_TO_SENTENCE1][str(num)] for num in source_sentences]
                    except:
                        try:
                            source_text = [row[schema.PairColumns.NUMBER_TO_SENTENCE1][num] for num in source_sentences]
                        except Exception as e:
                            logger.error(e)

                    target_sentences = eval(target_segments)['segmentation'][j]['sentences']
                    try:
                        target_text = [row[schema.PairColumns.NUMBER_TO_SENTENCE2][str(num)] for num in target_sentences]
                    except:
                        try:
                            target_text = [row[schema.PairColumns.NUMBER_TO_SENTENCE2][num] for num in target_sentences]
                        except Exception as e:
                            logger.error(e)
                    
                    aligned_segment_text.append((" ".join(source_text), " ".join(target_text)))
    
        source_qud_ans_list.append(source_qud_answers)
        target_qud_ans_list.append(target_qud_answers)
        harmonic_mean_list.append(harmonic_mean_scores)
        aligned_segment_list.append(aligned_segments)
        aligned_segment_text_list.append(aligned_segment_text)

    pairs_df[schema.PairColumns.D1_TO_D2_QUD_ANSWERS] = source_qud_ans_list
    pairs_df[schema.PairColumns.D2_TO_D1_QUD_ANSWERS] = target_qud_ans_list
    pairs_df[schema.PairColumns.HARMONIC_SCORE] = harmonic_mean_list
    pairs_df[schema.PairColumns.ALIGNED_SEGMENTS] = aligned_segment_list
    pairs_df[schema.PairColumns.ALIGNED_SEGMENT_TEXT] = aligned_segment_text_list

    return pairs_df

def main(args):
    # models for qud generation and answering (defaults to a particular model of GPT-4o)
    qg_gpt_model_name = args.qg_gpt_model
    try:
        qg_gpt_model = openai.GPT(qg_gpt_model_name)
    except:
        logger.error("could not instantiate GPT client with provided model name: %s", qg_gpt_model_name)
        return

    qa_gpt_model_name = args.qa_gpt_model
    try:
        qa_gpt_model = openai.GPT(qa_gpt_model_name)
    except:
        logger.error("could not instantiate GPT client with provided model name: %s", qa_gpt_model_name)
        return

    # level of abstraction of QUDs, with 0 being highly specific and 1 being abstractive
    level = args.level
    
    if level!=0 and level!=1:
        logger.error("Levels 0 and 1 are supported, 0 being specific and 1 being abstract. Value passed was an unsupported level.")
        return
    

    if config.THRESHOLD < 0 or config.THRESHOLD > 1:
        logger.error("Threshold value is outside the valid range: ", config.THRESHOLD)
        return

    # contains pairs of documents whose discourses are being aligned
    query_file_path = args.query_file
    try:
        try:
            query_df = pd.read_json(query_file_path)
        except:
            query_df = pd.read_csv(query_file_path)
    except:
        logger.error("Please provide input dataframe file in a json or csv format.")
        return

    with_replacement = args.with_replacement

    # id documents according to the replacement flag (unique ids v. unique documents)
    df = id_documents(with_replacement, level, query_df)

    # preprocess the docs
    preprocessed_df = preprocess_docs(df)
    if not preprocess_docs:
        logger.error("Could not preprocess data")
        return
    df = preprocessed_df
    logger.info("Completed preprocessing documents.")

    # generate quds
    extended_df = generate_quds(qg_gpt_model, df)
    
    df = extended_df
    logger.info("Completed generating QUDs.")

    success = _save_data(df, 'output/document_data.json')
    if not success:
        logger.error("Could not save document dataframe")
    
    # create pairs_df
    pairs_df = build_pairs_metadata(query_df, df)
    
    # align documents
    aligned_df = align_documents(qa_gpt_model, pairs_df)
    pairs_df = aligned_df

    logger.info("Completed aligning documents.")

    # save dataframe
    success = _save_data(pairs_df, 'output/qudsim.json')
    if not success:
        logger.error("Could not save qud and sim dataframe")

    display_df = reduce_df(pairs_df)
    success = _save_data(display_df, 'output/reduced_qudsim.json')
    if not success:
        logger.error("Could not save qud and sim dataframe (reduced, with no metadata)")



if __name__ == '__main__':
    gpt_client = openai.GPT(gpt_model=config.GPT_MODEL)
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=config.LEVEL, required=False, help='specifies level of abstraction as 0 or 1')
    parser.add_argument('--query_file', type=str, default='example_query.json', required=True)
    parser.add_argument('--qg_gpt_model', type=str, default=config.GPT_MODEL, required=False)
    parser.add_argument('--qa_gpt_model', type=str, default=config.GPT_MODEL, required=False)
    parser.add_argument('--unique', type=bool, default=config.WITH_REPLACEMENT, required=False, help="specifies if quds should be regenerated for every instance of a given document")
    args = parser.parse_args()
    main(args)
