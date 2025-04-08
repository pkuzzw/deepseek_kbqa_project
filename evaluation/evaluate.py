import json
from tqdm import tqdm
from retrieval.bm25_retriever import BM25Retriever
from retrieval.document_store import DocumentStore
from retrieval.glove_retriever import GloVeRetriever
from generator.QwenAPIClient import QwenAPIClient
from config.paths import DATA_PATHS

class Evaluator:
    def __init__(self):
        document_store = DocumentStore("data/documents.jsonl")
        BM25_save_path = "bm25_data.pkl"
        glove_save_path = "glove_data.pkl"
        api_key = "sk-dwpxzqylfqdijatcxhmaqgjrjruxveypcziudbxfbzfwmaju"
        base_url = "https://api.siliconflow.cn"

        self.retriever = BM25Retriever(document_store, BM25_save_path)
        #self.retriever = GloVeRetriever(document_store, "glove/glove.6B.300d.w2vformat.txt",glove_save_path)
        self.generator = QwenAPIClient(api_key, base_url)

    def evaluate_val(self):
        with open(DATA_PATHS["val"], "r") as f:
            samples = [json.loads(line) for line in f]
        
        results = []

        i = 0
        for sample in tqdm(samples[:5]):
            # Retrieve top k documents
            docRetrievalResponse = self.retriever.retrieve(sample["question"], 5)
            doc_ids = docRetrievalResponse.topk_doc_id
            doc_chunks = docRetrievalResponse.topk_doc_chunk

            contexts = [doc_chunk.chunk_text for doc_chunk in doc_chunks]
            answer = self.generator.generate_answer(sample["question"], contexts)
            results.append({
                "question": sample["question"],
                "answer": answer,
                "document_id": doc_ids
            })
            i += 1
            print(f"Question {i}: {sample['question']}")
            print(f"Document IDs: {doc_ids}")
        
        # Save results
        with open(DATA_PATHS["val_result"], "w") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")

        print("Evaluation complete. Results saved to val_predictions.jsonl")

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate_val()