from qud_generation import decontextualize
from qud_generation import qud
from qud_generation import segment

def _get_qud_dict(quds):
    qud_segment_dict = {}
    segment_qud_dict = {}
    num_quds = 0

    for i, source_qud in enumerate(quds):
        qud_idx_list = []
        for q in eval(source_qud)['quds']:
            qud_segment_dict[num_quds] = i
            qud_idx_list.append(num_quds)
            num_quds+=1
        segment_qud_dict[i] = qud_idx_list
            
    return segment_qud_dict, qud_segment_dict

def get_quds(gpt_model, text, sentence_num_dict, level):
    # SEGMENTATION
    segments, segmented_text = segment.segment(gpt_model, text, sentence_num_dict)
    if segments is None or segmented_text is None:
        return None
    
    segments_json = segments.model_dump_json()
    
    # DICTIONARY: SENTENCE --> SEGMENT
    segment_dict = {}
    for i, s in enumerate(segments.segmentation):
        for sentence_num in s.sentences:
            segment_dict[sentence_num] = i

    if level==1:
        # ENTITY ABSTRACTION
        numbered_segment_text = "\n\n".join(["[" + str(i)+ "] " + seg for i, seg in enumerate(segmented_text)])
        decontextualized_segments = decontextualize.decontextualize(gpt_model, numbered_segment_text, len(segmented_text))
        if decontextualized_segments is None:
            return None
        
        decontextualized_segments_json = decontextualized_segments.model_dump_json()

        # # QUD GENERATION - Level 1
        quds = [(qud.generate_quds(gpt_model, seg.para)).model_dump_json() 
                        for seg in decontextualized_segments.decontextualized_paragraphs]
        if quds is None:
            return None
    else:
        decontextualized_segments_json = None

        # QUD GENERATION - Level 0
        quds = [(qud.generate_quds(gpt_model, seg)).model_dump_json() for seg in segmented_text]
        if quds is None:
            return None

    segment_qud_dict, qud_segment_dict = _get_qud_dict(quds)
    qg_output_item = {"segments": segments_json,
                      "segment_dict": segment_dict,
                      "entity_abstracted_segments": decontextualized_segments_json,
                      "quds": quds,
                      "segment_qud_dict": segment_qud_dict,
                      "qud_segment_dict": qud_segment_dict}

    return qg_output_item 