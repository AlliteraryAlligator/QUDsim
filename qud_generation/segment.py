
from tools import openai
import sys
sys.path.append('../')
import config
import logging
logger = logging.getLogger(__name__)

system_prompt = "You will be given text with numbered sentences and your task is to redraw the paragraph boundaries such that each chunk is about one atomic topic. Each segment cannot be about multiple topics or about a complex topic. You may not change the text or change the order of the sentences. For each segment, provide the list of sentence numbers that belong to that segment."

class Segment(openai.BaseModel):
    sentences: list[int]

class Answer(openai.BaseModel):
    segmentation: list[Segment]

def segment(gpt_model, passage, sentence_dict):
    text = " ".join(passage.split("\n\n"))

    document_sentences = [k for k,v in sentence_dict.items()]
    document_sentences = [int(num) for num in document_sentences if type(num)!=int]


    for i in range(config.MAX_TRIES):
        try:
            segments = gpt_model.call_gpt_format(text, system_prompt, Answer)

            if not segments:
                logger.error("Segmentation was unsuccesful, try #%d" % i)
                continue            

            # all sentences in document must be accounted for
            segmented_sentences = set()
            for segment in segments.segmentation:
                for sentence_num in segment.sentences:
                    segmented_sentences.add(sentence_num)
            if len(set(document_sentences).difference(set(segmented_sentences)))!=0:
                # extraneous or missing sentences
                logger.error("Segmentation was unsuccesful, sentence number mismatch, try #%d" % i)
                continue


            # all sentences in each segments must be in document
            successful_segmentation = True
            segmented_text = []
            for segment in segments.segmentation:
                segment_text = ""
                for sentence_num in segment.sentences:
                    try:
                        sentence = sentence_dict[str(sentence_num)]
                        segment_text += sentence +  " "
                    except:
                        try:
                            sentence = sentence_dict[sentence_num]
                            segment_text += sentence +  " "
                        except:
                            logger.error("Extraneous sentence number, try #%d"%i)
                            successful_segmentation = False
                segmented_text.append(segment_text)

            if successful_segmentation:
                return segments, segmented_text
            
        except Exception as e:
            logger.error(e)
            continue
    
    logger.error("Segmentation was unsuccessful, tried 3 times.")
    return None, None

    

    

    