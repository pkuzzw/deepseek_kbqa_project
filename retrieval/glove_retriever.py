import numpy as np
import joblib
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from .document_store import DocumentStore


class GloVeRetriever:
    def __init__(self, document_store: DocumentStore, glove_path: str, save_path=None):
        print("Initializing GloVeRetriever...")
        self.document_store = document_store
        self.doc_ids = list(document_store.documents.keys())

        # 加载 GloVe 模型
        self.glove = KeyedVectors.load_word2vec_format(glove_path, binary=False)

        if save_path:
            try:
                # 尝试加载保存的数据
                self.load_data(save_path)
            except FileNotFoundError:
                print("未找到保存的数据，重新计算...")
                self._precompute_doc_vectors()
                self.save_data(save_path)
        else:
            self._precompute_doc_vectors()

    def _precompute_doc_vectors(self):
        # 预计算文档向量
        self.doc_vectors = np.array([
            self._document_to_vector(self.document_store.documents[doc_id])
            for doc_id in self.doc_ids
        ])

    def _document_to_vector(self, text: str) -> np.ndarray:
        vectors = []
        for word in text.split():
            if word in self.glove:
                vectors.append(self.glove[word])
        return np.mean(vectors, axis=0) if vectors else np.zeros(300)

    def retrieve(self, query: str, top_k: int = 5) -> list: 
        print("Retrieving...for...+"+query)
        # 查询向量化
        query_vector = self._document_to_vector(query)
        # 计算相似度
        similarities = cosine_similarity([query_vector], self.doc_vectors)[0]
        # 获取 top_k 文档 ID
        sorted_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.doc_ids[i] for i in sorted_indices]

    def save_data(self, save_path):
        data = {
            "doc_vectors": self.doc_vectors
        }
        joblib.dump(data, save_path)
        print(f"数据已保存到 {save_path}")

    def load_data(self, save_path):
        data = joblib.load(save_path)
        self.doc_vectors = data["doc_vectors"]
        print(f"数据已从 {save_path} 加载")
    