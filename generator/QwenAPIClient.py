import requests
from openai import OpenAI
from generator.prompt import SYSTEM_PROMPT

class QwenAPIClient:
    def __init__(self, api_key, api_url):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_answer(self, question, context):
        client = OpenAI(api_key="sk-ooscmypgomjmlrzejjhcidzhzegvmirvvweonaoacfmpcrrc", 
                        base_url="https://api.siliconflow.cn/v1")
        try:
            print("Backgroud Information: ")
            print(context)
            response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",  
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Background Information：{context}"},
                {"role": "user", "content": question}
            ],
            )
            message = response.choices[0].message.content
            print(f"Assistant: {message}")
            return message
        except Exception as e:
            message = f"An error occurred: {str(e)}"


