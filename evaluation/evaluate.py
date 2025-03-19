import json
from tqdm import tqdm
from retrieval.bm25_retriever import BM25Retriever
from generator.qwen_generator import QwenAnswerGenerator

class Evaluator:
    def __init__(self):
        self.retriever = BM25Retriever()
        self.generator = QwenAnswerGenerator()

    def evaluate_val(self):
        with open(DATA_PATHS["val"], "r") as f:
            samples = [json.loads(line) for line in f]
        
        results = []
        for sample in tqdm(samples):
            doc_ids = self.retriever.retrieve(sample["question"])
            contexts = [self.retriever.documents[doc_id] for doc_id in doc_ids]
            answer = self.generator.generate_answer(sample["question"], contexts)
            
            results.append({
                "question": sample["question"],
                "answer": answer,
                "document_id": doc_ids
            })
        
        # Save results
        with open("val_predictions.jsonl", "w") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")

        print("Evaluation complete. Results saved to val_predictions.jsonl")

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate_val()