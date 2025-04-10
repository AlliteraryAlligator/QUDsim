from tools import openai
import sys
sys.path.append('../')
import config
import logging
logger = logging.getLogger(__name__)

system_prompt = "You will be given a paragraph. We are interested in forming unique, high-level, abstract QUDs with minimal details such that when they are answered, we understand the main themes of the paragraph. First answer the minimum number of QUD(s) required. Then list the QUDs. Do not use conjunctions in the QUDs."
system_prompt_high_level = "You will be given a paragraph. We are interested in forming unique, high-level, abstract QUDs with minimal details such that when they are answered, we understand the main themes of the paragraph. Details specific to the content should be omitted. QUDs should like: What were the individual's greatest accomplishments? What legacy did the individual leave behind?. First answer the minimum number of QUD(s) required. Then list the QUDs. Do not use conjunctions in the QUDs."

class QUD(openai.BaseModel):
    qud: str

class Answer(openai.BaseModel):
    num_quds: int
    quds: list[QUD]

def generate_quds(gpt_model, segment, abstract=False):
    for i in range(config.MAX_TRIES):
        try:
            if abstract:
                quds = gpt_model.call_gpt_format(segment, system_prompt_high_level, Answer)
            else:
                quds = gpt_model.call_gpt_format(segment, system_prompt, Answer)

            if not quds or len(quds.quds)==0:
                logger.error("No QUDs were generated, try #%d" % i)
                continue

            return quds
        except Exception as e:
            logger.error(e)
            continue
    
    logger.error("QUD generation was unsuccessful, tried 3 times.")
    return None
            
