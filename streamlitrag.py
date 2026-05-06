import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="RAG Chat with Llama3",
    page_icon="🦙",
    layout="wide"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False

# Initialize models
@st.cache_resource
def initialize_models():
    """Initialize Ollama LLM and embeddings"""
    try:
        llm = Ollama(
            model="llama3:8b",
            temperature=0.7,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        embeddings = OllamaEmbeddings(model="llama3:8b")
        return llm, embeddings
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        st.info("Make sure Ollama is running and llama3 model is installed.")
        return None, None

def process_pdf(uploaded_file, embeddings):
    """Process uploaded PDF and create vector store"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        
        # Create in-memory Qdrant vector store
        vectorstore = Qdrant.from_documents(
            splits,
            embeddings,
            location=":memory:",
            collection_name="pdf_collection"
        )
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return vectorstore, len(splits)
    
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None, 0

def create_qa_chain(llm, vectorstore):
    """Create retrieval QA chain"""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

# Main UI
st.title("🦙 RAG Chat with Llama3")
st.markdown("Upload a PDF and chat with your document using Llama3 and Qdrant!")

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=['pdf'],
        help="Upload a PDF document to chat with"
    )
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Initializing models..."):
                llm, embeddings = initialize_models()
            
            if llm and embeddings:
                with st.spinner("Processing PDF and creating vector store..."):
                    vectorstore, num_chunks = process_pdf(uploaded_file, embeddings)
                    
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.qa_chain = create_qa_chain(llm, vectorstore)
                        st.session_state.document_processed = True
                        st.success(f"✅ Document processed! Created {num_chunks} chunks.")
                    else:
                        st.error("Failed to process document.")
    
    if st.session_state.document_processed:
        st.success("✅ Document ready for questions!")
        if st.button("Clear Document"):
            st.session_state.vectorstore = None
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.session_state.document_processed = False
            st.rerun()
    
    st.divider()
    st.markdown("### ℹ️ Instructions")
    st.markdown("""
    1. Upload a PDF document
    2. Click 'Process Document'
    3. Ask questions about the document
    4. Get AI-powered answers with sources
    """)

# Main chat interface
if st.session_state.document_processed:
    st.header("💬 Chat with Your Document")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source)
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain({"query": prompt})
                    answer = response['result']
                    source_docs = response['source_documents']
                    
                    st.markdown(answer)
                    
                    # Extract source texts
                    sources = [doc.page_content for doc in source_docs]
                    
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(source)
                            st.divider()
                    
                    # Add assistant response to chat
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                
                except Exception as e:
                    st.error(f"Error getting response: {e}")

else:
    # Show instructions when no document is loaded
    st.info("👈 Please upload and process a PDF document from the sidebar to start chatting!")
    
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    This application uses:
    - **Ollama Llama3** for language understanding and generation
    - **Qdrant (in-memory)** for vector storage
    - **Llama3 embeddings** for document vectorization
    
    Make sure you have:
    1. Ollama installed and running
    2. Llama3 model pulled (`ollama pull llama3`)
    3. Required Python packages installed
    """)
    
    with st.expander("📦 Required Packages"):
        st.code("""
pip install streamlit
pip install langchain
pip install langchain-community
pip install qdrant-client
pip install pypdf
pip install ollama
        """)

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with Streamlit, LangChain, Llama3, and Qdrant</div>",
    unsafe_allow_html=True
)