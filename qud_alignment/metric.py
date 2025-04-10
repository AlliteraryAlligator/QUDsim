import numpy as np
import logging
logger = logging.getLogger(__name__)

class SimilarityMetric():
    def _calculate_document_similarity(self, segment_scores):
        max_sim_scores = np.max(segment_scores, axis=1)
        overall_similarity = np.average(max_sim_scores)
        return  overall_similarity
    
class FrequencyBasedSimilarity(SimilarityMetric):
    def _consolidate_sentences(self, qud_answers, 
                              source_qud_src_dict):
        
        src_seg_to_ans_sentences = {}
        for i, ans in enumerate(qud_answers['excerpts']):
            try:
                source_segment = source_qud_src_dict[str(i)]
            except:
                try:
                    source_segment = source_qud_src_dict[i]
                except Exception as e:
                    logger.error(e)

            sentence_nums = set(ans['sentence_nums'])

            if source_segment not in src_seg_to_ans_sentences:
                src_seg_to_ans_sentences[source_segment] = set()
            
            old_sentence_list = src_seg_to_ans_sentences[source_segment]
            src_seg_to_ans_sentences[source_segment] = old_sentence_list.union(sentence_nums)  

        return src_seg_to_ans_sentences
    
    def _count_sentences(self, num_source_segments:int, 
                        num_target_segments:int, 
                        src_seg_to_ans_sentences, 
                        target_segments):
        
        arr = np.zeros((num_source_segments, num_target_segments))
        for i, source in enumerate(src_seg_to_ans_sentences):
            answer_sentences = src_seg_to_ans_sentences[source]
            if len(answer_sentences)==0:
                continue
            for j, target_segment in enumerate(eval(target_segments)['segmentation']):
                target_segment_sentences = set(target_segment['sentences'])
                intersection = target_segment_sentences.intersection(answer_sentences)
                arr[i][j] = len(intersection)

        return arr

    def _get_segment_scores(self, sentence_count_map):
        num_tgt_answers_per_src = sentence_count_map.sum(axis=1)

        # to prevent division by 0, turn denominators into 1 if no source segments have answers in the target
        # scores calculated will still be zero, and reflect the original sentence counts
        num_tgt_answers_per_src[num_tgt_answers_per_src == 0] = 1   

        segment_scores_transposed = np.transpose(sentence_count_map)/num_tgt_answers_per_src
        segment_scores = np.transpose(segment_scores_transposed)

        return segment_scores

    def calculate_similarity(self, num_target_segments:int, 
                             num_source_segments:int, 
                             qud_answers:list, 
                             source_qud_src_dict, 
                             target_segments): 
        
        src_seg_to_ans_sentences = self._consolidate_sentences(qud_answers, 
                                                              source_qud_src_dict)
        
        sentence_count_map = self._count_sentences(num_source_segments, 
                                                  num_target_segments, 
                                                  src_seg_to_ans_sentences, 
                                                  target_segments)
        
        segment_scores = self._get_segment_scores(sentence_count_map)
        overall_similarity = self._calculate_document_similarity(segment_scores)

        return segment_scores, overall_similarity
    
