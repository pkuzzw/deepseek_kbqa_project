import torch
from transformers import DPRQuestionEncoder, DPRContextEncoder
from faiss_indexer import FaissIndexer
import numpy as np
import os

class DPRRetriever:
    def __init__(self, document_store, 
                 q_model="facebook/dpr-question_encoder-single-nq-base",
                 ctx_model="facebook/dpr-ctx_encoder-single-nq-base"):
        self.doc_store = document_store
        self.q_encoder = DPRQuestionEncoder.from_pretrained(q_model)
        self.ctx_encoder = DPRContextEncoder.from_pretrained(ctx_model)
        self.faiss_indexer = FaissIndexer(dimension=768)  # DPR输出维度768
        
        # 加载或构建索引
        index_path = "dpr_faiss_index.bin"
        if os.path.exists(index_path):
            self.faiss_indexer.load(index_path)
        else:
            self._build_index()
    
    def _build_index(self):
        """预处理文档生成向量索引"""
        all_embeddings = []
        for doc_id, text in self.doc_store.documents.items():
            inputs = self.ctx_encoder.tokenizer(
                text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            embedding = self.ctx_encoder(**inputs).pooler_output.detach().numpy()
            all_embeddings.append(embedding)
        
        # 转换为numpy数组并构建索引
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        self.faiss_indexer.train(all_embeddings)
        self.faiss_indexer.add(all_embeddings)
        self.faiss_indexer.save("dpr_faiss_index.bin")
    
    def retrieve(self, query, top_k=5):
        """检索流程"""
        # 编码问题
        inputs = self.q_encoder.tokenizer(
            query, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        query_embedding = self.q_encoder(**inputs).pooler_output.detach().numpy()
        
        # FAISS检索
        distances, indices = self.faiss_indexer.index.search(query_embedding, top_k)
        doc_ids = [list(self.doc_store.documents.keys())[i] for i in indices[0]]
        return doc_ids