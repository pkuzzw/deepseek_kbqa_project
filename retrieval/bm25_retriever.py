# retrieval/bm25_retriever.py
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from .document_store import DocumentStore

class BM25Retriever:
    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store
        self.doc_ids = list(document_store.documents.keys())
        
        # 预处理文档
        self.tokenized_docs = [
            self._tokenize(document_store.documents[doc_id])
            for doc_id in self.doc_ids
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _tokenize(self, text: str) -> list:
        return word_tokenize(text.lower())

    def retrieve(self, query: str, top_k: int = 5) -> list:
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # 获取top_k文档ID
        sorted_indices = sorted(
            range(len(doc_scores)),
            key=lambda i: doc_scores[i],
            reverse=True
        )[:top_k]
        
        return [self.doc_ids[i] for i in sorted_indices]