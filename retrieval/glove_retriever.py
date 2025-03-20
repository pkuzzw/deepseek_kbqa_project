# retrieval/glove_retriever.py
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from .document_store import DocumentStore

class GloVeRetriever:
    def __init__(self, document_store: DocumentStore, glove_path: str):
        self.document_store = document_store
        self.doc_ids = list(document_store.documents.keys())
        
        # 加载GloVe模型
        self.glove = KeyedVectors.load_word2vec_format(glove_path, binary=False)
        
        # 预计算文档向量
        self.doc_vectors = np.array([
            self._document_to_vector(document_store.documents[doc_id])
            for doc_id in self.doc_ids
        ])

    def _document_to_vector(self, text: str) -> np.ndarray:
        vectors = []
        for word in text.split():
            if word in self.glove:
                vectors.append(self.glove[word])
        return np.mean(vectors, axis=0) if vectors else np.zeros(300)

    def retrieve(self, query: str, top_k: int = 5) -> list:
        # 查询向量化
        query_vector = self._document_to_vector(query)
        
        # 计算相似度
        similarities = cosine_similarity([query_vector], self.doc_vectors)[0]
        
        # 获取top_k文档ID
        sorted_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.doc_ids[i] for i in sorted_indices]