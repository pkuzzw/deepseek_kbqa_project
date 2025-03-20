class HybridRetriever:
    def __init__(self, retrievers):
        self.retrievers = retrievers
        
    def retrieve(self, query, top_k=5):
        all_results = []
        for retriever in self.retrievers:
            results = retriever.retrieve(query, top_k*2)
            all_results.extend(results)
            
        # 基于频次的排序
        doc_scores = {}
        for doc_id in all_results:
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1
            
        sorted_docs = sorted(doc_scores.items(), 
                            key=lambda x: x[1], 
                            reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]