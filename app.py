import streamlit as st
from retrieval.bm25_retriever import BM25Retriever
from retrieval.glove_retriever import GloVeRetriever
from generator.qwen_generator import QwenAPIGenerator
from retrieval.document_store import DocumentStore
from retrieval.dpr_retriever import DPRRetriever


# 初始化组件（带缓存）
@st.cache_resource
def init_system():
    print("Initializing system...")
    document_store = DocumentStore("data/documents.jsonl")
    return {
        "bm25": BM25Retriever(document_store),
        "DPR": DPRRetriever(document_store),
        "glove": GloVeRetriever(document_store, "glove/glove.6B.300d.w2vformat.txt"),
        "generator": QwenAPIGenerator(),
        "docs": document_store
    }

def main():
    st.set_page_config(page_title="KBQA System", layout="wide")
    
    # 系统初始化
    system = init_system()
    
    # 页面标题
    st.title("📚 Knowledge Base Question Answering System")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("Configuration")
        retriever_type = st.selectbox(
            "Retrieval Method",
            ["BM25", "DPR", "GloVe"],
            index=0
        )
        top_k = st.slider("Number of Documents to Retrieve", 1, 10, 5)
    
    # 主界面
    question = st.text_input("Ask your question:", placeholder="When was iOS 11 released?")
    
    if question:
        # 执行检索
        with st.spinner("Searching knowledge base..."):
            if retriever_type == "BM25":
                doc_ids = system["bm25"].retrieve(question, top_k)
            else:
                doc_ids = system["glove"].retrieve(question, top_k)
            
            contexts = [system["docs"].get_document(did) for did in doc_ids]
        
        # 显示检索结果
        st.subheader("🔍 Retrieved Documents")
        for i, (doc_id, context) in enumerate(zip(doc_ids, contexts)):
            with st.expander(f"Document {i+1} (ID: {doc_id})"):
                st.text(context[:1000] + "...")  # 限制显示长度
        
        # 生成答案
        with st.spinner("Generating answer..."):
            answer = system["generator"].generate_answer(question, contexts)
        
        # 显示答案
        st.subheader("💡 Answer")
        st.markdown(f"```\n{answer}\n```")

if __name__ == "__main__":
    print("Running app.py...")
    main()