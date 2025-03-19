import streamlit as st
from retrieval.bm25_retriever import BM25Retriever
from retrieval.glove_retriever import GloVeRetriever
from generator.qwen_generator import QwenAnswerGenerator

class KBQAInterface:
    def __init__(self):
        self.retrievers = {
            "BM25": BM25Retriever(),
            "GloVe": GloVeRetriever()
        }
        self.generator = QwenAnswerGenerator()

    def run(self):
        st.title("KBQA System")
        
        # User input
        question = st.text_input("Enter your question:")
        retriever_type = st.selectbox("Select retrieval method:", list(self.retrievers.keys()))
        
        if question:
            # Retrieve documents
            retriever = self.retrievers[retriever_type]
            doc_ids = retriever.retrieve(question)
            contexts = [retriever.documents[doc_id] for doc_id in doc_ids]
            
            # Generate answer
            answer = self.generator.generate_answer(question, contexts)
            
            # Display results
            st.subheader("Retrieved Documents")
            for doc_id, context in zip(doc_ids, contexts):
                st.write(f"**Document {doc_id}**: {context[:500]}...")
            
            st.subheader("Generated Answer")
            st.markdown(f"**Answer**: {answer}")

if __name__ == "__main__":
    interface = KBQAInterface()
    interface.run()