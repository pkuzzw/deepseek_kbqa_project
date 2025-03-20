from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams  # 使用vllm加速推理
import torch

class QwenLocalGenerator:
    def __init__(self, model_path="Qwen/Qwen2.5-7B-Instruct"):
        # 使用vLLM加速推理
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=32768
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512
        )

    def generate_answer(self, question, contexts):
        prompt = self._build_prompt(question, contexts)
        
        # 使用vLLM批量生成
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

    def _build_prompt(self, question, contexts):
        context_str = "\n".join(
            [f"[Context {i+1}]: {ctx[:2000]}..." for i, ctx in enumerate(contexts)]
        )
        return f"""<|im_start|>system
You are an expert QA system. Answer the question using ONLY the given contexts.
If the answer isn't in the contexts, say "I don't know".<|im_end|>
<|im_start|>contexts
{context_str}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""