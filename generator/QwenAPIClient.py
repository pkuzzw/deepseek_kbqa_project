import requests
import re
from openai import OpenAI
from generator.prompt import SYSTEM_PROMPT
from generator.prompt import NOT_FOUND_MESSAGE
import time

class QwenAPIClient:
    def __init__(self, api_key, api_url):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_answer(self, question, contexts):
        client = OpenAI(api_key="sk-ooscmypgomjmlrzejjhcidzhzegvmirvvweonaoacfmpcrrc", 
                        base_url="https://api.siliconflow.cn/v1")
        try:
            max_seq_len = 9000
        
            context_tokens = "<s/> ".join(contexts)  # Join contexts into a single string

            if len(context_tokens) > max_seq_len:
                context_tokens = context_tokens[:max_seq_len]

            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",  
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Background Information： {context_tokens}"},
                    {"role": "user", "content": f"Question: {question}"},
                ],
            )
            message = response.choices[0].message.content
            print(f"Assistant: {message}")
            return message
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            #sleep for 1 second to avoid rate limit
            time.sleep(60)
            return NOT_FOUND_MESSAGE

