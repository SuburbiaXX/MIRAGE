from openai import OpenAI
import re
BASE_URL = ""
API_KEY = ""
assert API_KEY is not None, "OPENAI_API_KEY is not set"

class GPT():
    def __init__(self, model_name, base_url=BASE_URL, api_key=API_KEY, max_output_tokens=10240):
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(base_url=base_url ,api_key=api_key)
        self.model_name = model_name
    
    def query(self, msg, temperature=0.1, max_tokens=10240, top_p=None):
        try:
            api_params = {
                "model": self.model_name,
                "temperature": temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.max_output_tokens,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ],
            }
            
            if top_p is not None:
                api_params["top_p"] = top_p
            
            completion = self.client.chat.completions.create(**api_params)
            response = completion.choices[0].message.content
            response = response.strip()

        except Exception as e:
            print(e)
            response = ""

        return response
    


class EmbeddingModel():
    def __init__(self, model_name):
        # self.max_output_tokens = 10240
        self.client = OpenAI(base_url=BASE_URL ,api_key=API_KEY)
        self.model_name = model_name
    
    def query(self, msg):
        try:
            completion = self.client.embeddings.create(
                model=self.model_name,
                input=msg
            )
            # response = completion.data[0].embedding
            response = [item.embedding for item in completion.data]

        except Exception as e:
            print(e)
            response = ""

        return response
