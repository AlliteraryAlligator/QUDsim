from tools import openai
import sys
sys.path.append('../')
import config
import logging
logger = logging.getLogger(__name__)

system_prompt = "You will be given several numbered paragraphs. Decontextualize each paragraph such that the paragraph's general plot is captured. Names, places, extraneous details and descriptive language should all be abstracted away."

class Paragraph(openai.BaseModel):
    para_num: int
    para: str

class Answer(openai.BaseModel):
    decontextualized_paragraphs: list[Paragraph]


def decontextualize(gpt_model, text, num_segments):
    for i in range(config.MAX_TRIES):
        try:
            decontextualized = gpt_model.call_gpt_format(text, system_prompt, Answer)

            if not decontextualized:
                logger.error("Entity abstraction was unsuccessful, try #%d"%i)
                continue

            if len(decontextualized.decontextualized_paragraphs)!=num_segments:
                logger.error("Mismatch between number of segments and number of abstracted segments, try #%d"%i)
                continue

            return decontextualized
        except Exception as e:
            logger.error(e)
            continue

    logger.error("Entity abstraction was unsuccessful, tried 3 times.")
    return None

    

