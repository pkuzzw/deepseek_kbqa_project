from rank_bm25 import BM25Okapi
import json
try:
    from config.paths import DATA_PATHS
except ImportError:
    raise ImportError("The module 'config.paths' or 'DATA_PATHS' is not found. Ensure it exists and is correctly defined.")
import numpy as np
from nltk.tokenize import word_tokenize

class BM25Retriever:
    def __init__(self):
        self.documents = self._load_documents()
        self.tokenized_docs = [word_tokenize(doc["document_text"].lower()) 
                              for doc in self.documents.values()]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _load_documents(self):
        documents = {}
        with open(DATA_PATHS["documents"], "r", encoding='utf-8') as f:
            print(DATA_PATHS["documents"])
            for line in f:
                doc = json.loads(line)
                documents[doc["document_id"]] = doc["document_text"]
        print(documents.shape)
        return documents

    def retrieve(self, question, top_k=5):
        tokenized_query = word_tokenize(question.lower())
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[-top_k:][::-1]
        return [list(self.documents.keys())[i] for i in top_indices]