import joblib
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from .document_store import DocumentStore


class BM25Retriever:
    def __init__(self, document_store: DocumentStore, save_path=None):
        print("Initializing BM25Retriever...")
        self.document_store = document_store
        self.doc_ids = list(document_store.documents.keys())

        if save_path:
            try:
                # 尝试加载保存的数据
                self.load_data(save_path)
            except FileNotFoundError:
                print("未找到保存的数据，重新计算...")
                self._initialize_bm25()
                self.save_data(save_path)
        else:
            self._initialize_bm25()

    def _initialize_bm25(self):
        # 预处理文档
        self.tokenized_docs = [
            self._tokenize(self.document_store.documents[doc_id])
            for doc_id in self.doc_ids
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _tokenize(self, text: str) -> list:
        return word_tokenize(text.lower())

    def retrieve(self, query: str, top_k: int = 5) -> list:
        print("Retrieving...for...+"+query)
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)

        # 获取 top_k 文档 ID
        sorted_indices = sorted(
            range(len(doc_scores)),
            key=lambda i: doc_scores[i],
            reverse=True
        )[:top_k]

        return [self.doc_ids[i] for i in sorted_indices]

    def save_data(self, save_path):
        data = {
            "tokenized_docs": self.tokenized_docs,
            "bm25": self.bm25
        }
        joblib.dump(data, save_path)
        print(f"数据已保存到 {save_path}")

    def load_data(self, save_path):
        data = joblib.load(save_path)
        self.tokenized_docs = data["tokenized_docs"]
        self.bm25 = data["bm25"]
        print(f"数据已从 {save_path} 加载")
    