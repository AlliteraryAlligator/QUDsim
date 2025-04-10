from tools import openai
import sys
sys.path.append('../')
import config
import logging
logger = logging.getLogger(__name__)

system_prompt = "You are an expert reading comprehension agent. You will be given a passage with numbered sentences and a series of questions. For each question, your task is to extract all sentences that directly help answer it. You must return the question and a list of sentence numbers and sentences that answer it. The question may not always be answerable. In that case, return an empty list. Do NOT overgenerate. Do not modify the original text."

class Answer(openai.BaseModel):
    question: str
    sentence_nums: list[int]
    sentences: list[str]

class Response(openai.BaseModel):
    excerpts: list[Answer]


def get_answer(gpt_model, numbered_segments: str, qud_list: str, num_quds: int, num_target_sentences):
    for i in range(config.MAX_TRIES):
        try:
            prompt = "Passage: %s\nQuestions:\n%s" % (numbered_segments, qud_list)
            answer = gpt_model.call_gpt_format(prompt, system_prompt, Response)
            # print(len(answer.excerpts))
            if len(answer.excerpts)!=num_quds:
                logger.error("Mismatch between QUDs asked and answered, try #%d" % i)
                continue

            for ans in answer.excerpts:
                for sentence in ans.sentence_nums:
                    if sentence<0 or sentence>num_target_sentences:
                        logger.error("Sentence numbers out of bounds, try #%d" % i)
                        continue

            return answer.model_dump_json()
        except Exception as e:
            logger.error(e)
            continue

    logger.error("QUD answering was unsuccessful, tried 3 times.")
    return None