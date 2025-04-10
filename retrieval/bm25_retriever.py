import joblib
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from .document_store import DocRetrievalResponse, DocumentChunk, DocumentStore


class BM25Retriever:
    def __init__(self, document_store: DocumentStore, save_path=None):
        print("Initializing BM25Retriever...")
        self.document_store = document_store
        self.doc_ids = list(document_store.documents.keys())
        self.doc_chunk_ids = list(document_store.document_chunks.keys())

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
        self.tokenized_chunks = [
            self._tokenize(self.document_store.get_chunk(doc_chunk_id).chunk_text)
            for doc_chunk_id in self.doc_chunk_ids
        ]

        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def _tokenize(self, text: str) -> list:
        return word_tokenize(text.lower())

    def retrieve(self, query: str, top_k: int = 5) -> DocRetrievalResponse:
        print("Retrieving...for...+"+query)
        tokenized_query = self._tokenize(query)
        chunk_scores = self.bm25.get_scores(tokenized_query)

        # 获取 top_k chunk ID
        sorted_indices = sorted(
            range(len(chunk_scores)),
            key=lambda i: chunk_scores[i],
            reverse=True
        )[:top_k*10]

        sorted_chunk_ids = [self.doc_chunk_ids[i] for i in sorted_indices]

        return self.document_store.get_doc_retrieval_response(sorted_chunk_ids, top_k)

    def save_data(self, save_path):
        data = {
            "tokenized_chunks": self.tokenized_chunks,
            "bm25": self.bm25
        }
        joblib.dump(data, save_path)
        print(f"数据已保存到 {save_path}")

    def load_data(self, save_path):
        data = joblib.load(save_path)
        self.tokenized_chunks = data["tokenized_chunks"]
        self.bm25 = data["bm25"]

        print(f"数据已从 {save_path} 加载")
    