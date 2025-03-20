import torch
from colbert.infra import Run, RunConfig
from colbert import Searcher, Indexer
from faiss_indexer import FaissIndexer  # 自定义FAISS索引管理
import time

class ColBERTRetriever:
    def __init__(self, document_store, index_path="colbert_index"):
        print("Initializing ColBERTRetriever...")
        self.doc_store = document_store
        self.index_path = index_path
        self.searcher = None
        self.faiss_indexer = None
        
        # 初始化模型和索引
        self._initialize_model()
        self._build_or_load_index()

    def _initialize_model(self):
        with Run().context(RunConfig(nranks=1)):
            self.searcher = Searcher(
                index=self.index_path,
                checkpoint="colbert-ir/colbertv2.0"
            )
        
        # 初始化FAISS索引器
        self.faiss_indexer = FaissIndexer(
            dimension=128,  # ColBERTv2压缩维度
            nlist=100       # 聚类中心数
        )

    def _build_or_load_index(self):
        if not self.searcher.check_index_exists():
            self._create_index()
        
        # 加载FAISS索引
        self.faiss_indexer.load(f"{self.index_path}/faiss_index.bin")

    def _create_index(self):
        print("Building ColBERT index...")
        start_time = time.time()
        
        # 转换文档格式
        documents = [
            {"id": did, "text": text}
            for did, text in self.doc_store.documents.items()
        ]
        
        # 使用ColBERT原生索引
        with Run().context(RunConfig(nranks=1)):
            indexer = Indexer(
                checkpoint="colbert-ir/colbertv2.0",
                index_root=self.index_path
            )
            indexer.index(name="nq_index", collection=documents)
        
        # 生成FAISS索引
        self._build_faiss_index()
        print(f"Index built in {time.time()-start_time:.2f}s")

    def _build_faiss_index(self):
        """将ColBERT原生索引转换为FAISS格式"""
        all_embeddings = []
        for doc_id in self.doc_store.documents.keys():
            embedding = self.searcher.encode(self.doc_store.documents[doc_id])
            all_embeddings.append(embedding.mean(axis=0))  # 取平均表示
        
        self.faiss_indexer.train(all_embeddings)
        self.faiss_indexer.add(all_embeddings)

    def retrieve(self, query, top_k=5):
        # 两步检索：FAISS粗筛 + ColBERT精排
        start_time = time.time()
        
        # 步骤1：FAISS快速召回
        query_embedding = self.searcher.encode(query).mean(axis=0)
        candidate_ids = self.faiss_indexer.search([query_embedding], top_k*10)[0]
        
        # 步骤2：ColBERT精确重排
        results = self.searcher.search(
            query, 
            k=top_k*10, 
            candidates=candidate_ids
        )
        
        # 返回top_k结果
        return [doc_id for doc_id, _ in results[:top_k]]