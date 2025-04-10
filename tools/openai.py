import openai
from pydantic import BaseModel
import os

class GPT():
    def __init__(self, gpt_model):
        self.model = gpt_model

        if not openai.api_key:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI()

    def call(self, prompt, system_prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        msg = response.choices[0].message
        return msg.content


    def call_gpt_format(self, prompt, system_prompt, format):
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content":prompt}
                ],
                response_format=format,
            )

            answer = completion.choices[0].message.parsed
            return answer
        except:
            raise TypeError
        