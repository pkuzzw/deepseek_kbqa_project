import json
import requests
from config.paths import API_CONFIG

class QwenAnswerGenerator:
    def __init__(self):
        self.api_url = API_CONFIG["qwen_endpoint"]
        self.headers = {
            "Authorization": f"Bearer {API_CONFIG['api_key']}",
            "Content-Type": "application/json"
        }

    def generate_answer(self, question, contexts):
        prompt = self._build_prompt(question, contexts)
        payload = {
            "model": "Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API Error: {e}")
            return "Sorry, I couldn't generate an answer."

    def _build_prompt(self, question, contexts):
        context_str = "\n".join([f"Document {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        return f"""Based on the following documents, answer the question. If the answer isn't found, say so.

Documents:
{context_str}

Question: {question}
Answer:"""