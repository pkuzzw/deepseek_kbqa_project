import streamlit as st
from retrieval.bm25_retriever import BM25Retriever
from retrieval.glove_retriever import GloVeRetriever
from generator.QwenAPIClient import QwenAPIClient
from retrieval.document_store import DocumentStore
from generator.prompt import NOT_FOUND_MESSAGE


# Initialize components (with caching)
@st.cache_resource
def init_system():
    # Initialize the system components and cache them for performance
    print("Initializing system...")
    document_store = DocumentStore("data/documents.jsonl")
    BM25_save_path = "bm25_data.pkl"
    glove_save_path = "glove_data.pkl"
    api_key = "sk-ooscmypgomjmlrzejjhcidzhzegvmirvvweonaoacfmpcrrc"
    base_url = "https://api.siliconflow.cn"
    # Initialize the system components
    return {
        "bm25": BM25Retriever(document_store,BM25_save_path),  # BM25 retriever for document retrieval
      #  "DPR": DPRRetriever(document_store),  # Uncomment to add DPR retriever
        "glove": GloVeRetriever(document_store, "glove/glove.6B.300d.w2vformat.txt",glove_save_path),  # GloVe retriever
        "generator": QwenAPIClient(api_key, base_url) ,  # Answer generator using Qwen API
        "docs": document_store  # Document store for storing and retrieving documents
    }

def main():
    # Set up the Streamlit page configuration
    st.set_page_config(page_title="KBQA System", layout="wide")
    
    # System initialization
    system = init_system()  # Initialize the system components
    
    # Page title
    st.title("📚 Knowledge Base Question Answering System")  # Display the app title
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")  # Sidebar header
        retriever_type = st.selectbox(
            "Retrieval Method",  # Dropdown to select retrieval method
            ["BM25", "GloVe"],
            index=0
        )
        top_k = st.slider("Number of Documents to Retrieve", 1, 10, 5)  # Slider to select number of documents
    
    # Main interface
    question = st.text_input("Ask your question:", placeholder="When was iOS 11 released?")  # Input field for user question
    
    if question:
        # Perform retrieval
        with st.spinner("Searching knowledge base..."):  # Show spinner while retrieving documents
            if retriever_type == "BM25":
                doc_ids = system["bm25"].retrieve(question, top_k)  # Retrieve documents using BM25
            else:
                doc_ids = system["glove"].retrieve(question, top_k)  # Retrieve documents using GloVe
            
            contexts = [system["docs"].get_document(did) for did in doc_ids]  # Get document contents
        
        # Display retrieval results
        st.subheader("🔍 Retrieved Documents")  # Display retrieved documents
        selected_context = contexts[0] if contexts else None  # Default to the first context if available
        selected_id = doc_ids[0] if doc_ids else None  # Default to the first document ID if available

        for i, (doc_id, context) in enumerate(zip(doc_ids, contexts)):
            if st.button(f"Use Document {i+1} (ID: {doc_id})"):  # Button to select a document
                selected_context = context  # Set the selected context
                selected_id = doc_id  # Set the selected context document ID
        
        # Generate answer
        if selected_context:
            with st.spinner("Generating answer..."):  # Show spinner while generating answer
                answer = system["generator"].generate_answer(question, [selected_context])  # Generate answer using selected context
        else:
            st.warning("Please select a document to generate an answer.")  # Warning if no document is selected

        
        # Display answer
        st.subheader("💡 Answer")  # Display the generated answer
        st.text_area("Generated Answer", value=answer, height=5)  # Use a text area to display the answer

        if selected_context:
            st.subheader(f"📄 Selected Document ID: {selected_id}")
            st.text_area("Document Content", value=selected_context, height=500)  # Display the selected document content

if __name__ == "__main__":
    print("Running app.py...")  # Log message when the app starts
    main()  # Run the main function