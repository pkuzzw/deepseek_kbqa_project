import requests
import json
from config.api_config import QWEN_API_CONFIG

class QwenAPIGenerator:
    def __init__(self):
        self.api_url = QWEN_API_CONFIG["endpoint"]
        self.headers = {
            "Authorization": f"Bearer {QWEN_API_CONFIG['api_key']}",
            "Content-Type": "application/json"
        }

    def generate_answer(self, question, contexts):
        prompt = self._build_prompt(question, contexts)
        
        payload = {
            "model": "Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"API Error: {str(e)}")
            return "Sorry, I encountered an error generating the answer."

    def _build_prompt(self, question, contexts):
        return f"""基于以下背景信息回答问题，如果答案不在信息中，请说明。

背景信息：
{" ".join(contexts)}

问题：{question}
答案："""