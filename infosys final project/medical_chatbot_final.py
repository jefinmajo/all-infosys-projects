import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import requests
import os
import uuid
import re
import hashlib
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state with chat format
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_source_type' not in st.session_state:
    st.session_state.current_source_type = 'pdf'
if 'current_source_name' not in st.session_state:
    st.session_state.current_source_name = 'Default Medical Encyclopedia'
if 'current_file_hash' not in st.session_state:
    st.session_state.current_file_hash = None

# Cache Pinecone client initialization
@st.cache_resource(show_spinner=False)
def init_pinecone():
    try:
        return Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            timeout=30
        )
    except Exception as e:
        st.error(f"❌ Pinecone initialization failed: {str(e)}")
        st.stop()

pc = init_pinecone()
INDEX_NAME = "hospital"

# Cache index creation
@st.cache_resource(show_spinner=False)
def ensure_pinecone_index():
    try:
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
    except Exception as e:
        st.error(f"❌ Pinecone index operation failed: {str(e)}")
        st.stop()

ensure_pinecone_index()

# Optimized text splitter configuration
@st.cache_resource(show_spinner=False)
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n##", "\n\n", "\n", " "],
        length_function=len
    )

def get_file_hash(uploaded_file):
    """Generate MD5 hash of file content"""
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()

def create_safe_namespace(source_name):
    """Create a Pinecone-safe namespace with UUID suffix"""
    cleaned = re.sub(r'\W+', '', source_name)
    uuid_part = uuid.uuid4().hex[:8]
    return f"{cleaned[:50]}-{uuid_part}"[:64]

def update_source(source_type, source_name):
    """Update current source and reset chat history if source changed"""
    prev_source = (st.session_state.current_source_type, st.session_state.current_source_name)
    new_source = (source_type, source_name)
    
    if prev_source != new_source:
        st.session_state.chat_history = []
        st.session_state.current_source_type = source_type
        st.session_state.current_source_name = source_name

# Cached document processing
@st.cache_resource(show_spinner=False)
def load_and_process_pdf(pdf_path):
    """Load and process PDF with optimized splitting"""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return get_text_splitter().split_documents(documents)
    except Exception as e:
        st.error(f"📄 PDF processing failed: {str(e)}")
        raise

@st.cache_resource(show_spinner=False)
def load_and_process_url(url):
    """Load and process URL content with optimized splitting"""
    try:
        if not url.startswith(('http://', 'https://')):
            raise ValueError("Invalid URL format")
            
        loader = UnstructuredURLLoader(urls=[url])
        documents = loader.load()
        return get_text_splitter().split_documents(documents)
    except Exception as e:
        st.error(f"🌐 URL processing failed: {str(e)}")
        raise

# Dedicated default PDF initialization
@st.cache_resource(show_spinner=False)
def initialize_default_qa_system():
    """Persistent cache for default PDF"""
    try:
        default_pdf_path = os.getenv("DEFAULT_PDF_PATH")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        text_chunks = load_and_process_pdf(default_pdf_path)
        return PineconeVectorStore.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace="default_medical"  # Fixed namespace
        )
    except Exception as e:
        st.error(f"🔧 Default system initialization failed: {str(e)}")
        st.stop()

@st.cache_resource(show_spinner=False, hash_funcs={st.runtime.uploaded_file_manager.UploadedFile: get_file_hash})
def initialize_qa_system(source, is_url=False, is_upload=False):
    """Universal initializer with content-based caching"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        if is_upload:
            # Handle uploaded file bytes directly
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(source.getbuffer())
                temp_path = tmp.name
            
            try:
                text_chunks = load_and_process_pdf(temp_path)
            finally:
                # Ensure file cleanup even if processing fails
                os.unlink(temp_path)
        elif is_url:
            text_chunks = load_and_process_url(source)
        else:
            text_chunks = load_and_process_pdf(source)
        
        return PineconeVectorStore.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=create_safe_namespace(st.session_state.current_source_name)
        )
    except Exception as e:
        st.error(f"🔧 QA system initialization failed: {str(e)}")
        st.stop()

def get_relevant_chunks(vectorstore, question, k=3):
    """Retrieve relevant chunks from Pinecone"""
    try:
        docs = vectorstore.similarity_search(question, k=k)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        st.error(f"🔍 Context retrieval failed: {str(e)}")
        raise

def query_llama_api(question, context, model="llama3.2"):
    """Query the LLaMA model with proper error handling"""
    url = "http://localhost:11434/api/generate"
    
    prompt = f"""
    You are a medical information assistant. Use the provided context to answer the question.
    If unsure, say "I don't have enough information to answer that."

    Context:
    {context}

    Question: {question}
    
    Answer based on the context:
    """
    
    try:
        response = requests.post(url, json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }, timeout=30)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"⚠️ Error querying LLM: {str(e)}"

def main():
    st.set_page_config(page_title="Medical AI Chat", page_icon="🏥")
    st.title("🏥 Medical AI - Medical Knowledge Assistant")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.divider()
        st.caption(f"Current Knowledge Source: {st.session_state.current_source_name}")
        
        option = st.radio(
            "Select knowledge base:",
            ("Default Medical Encyclopedia", "Upload PDF", "Enter URL"),
            index=0
        )
        
        # Source handling
        if option == "Upload PDF":
            pdf_file = st.file_uploader("Choose PDF file", type=["pdf"])
            if pdf_file:
                file_hash = get_file_hash(pdf_file)
                if st.session_state.current_file_hash != file_hash:
                    with st.spinner("Processing PDF..."):
                        try:
                            st.session_state.vectorstore = initialize_qa_system(
                                source=pdf_file,
                                is_upload=True
                            )
                            st.session_state.current_file_hash = file_hash
                            update_source('pdf', pdf_file.name)
                            st.success("PDF processed successfully!")
                        except Exception as e:
                            st.error(f"Processing error: {str(e)}")

        elif option == "Enter URL":
            url = st.text_input("Enter website URL:")
            if st.button("Process URL") and url:
                with st.spinner("Analyzing website content..."):
                    try:
                        st.session_state.vectorstore = initialize_qa_system(url, is_url=True)
                        update_source('url', url)
                        st.success("Website content loaded!")
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")

        # Handle default source
        elif option == "Default Medical Encyclopedia":
            if st.session_state.current_source_name != 'Default Medical Encyclopedia':
                with st.spinner("🔄 Switching to default..."):
                    try:
                        st.session_state.vectorstore = initialize_default_qa_system()
                        update_source('pdf', 'Default Medical Encyclopedia')
                        st.success("✅ Default source loaded!")
                    except Exception as e:
                        st.error(f"❌ Load failed: {str(e)}")

    # Main chat interface
    col1, col2 = st.columns([6, 1])
    with col1:
        st.subheader("Chat with Medical AI")
    with col2:
        debug_mode = st.toggle("🐞", help="Enable debug mode")

    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(name=message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                st.caption(f"Source: {message['source']}")

    # Initialize default knowledge base
    if 'vectorstore' not in st.session_state:
        with st.spinner("🚀 Loading medical knowledge base..."):
            try:
                st.session_state.vectorstore = initialize_default_qa_system()
                update_source('pdf', 'Default Medical Encyclopedia')
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
                return

    # Chat input and processing
    if prompt := st.chat_input("Ask a medical question..."):
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "avatar": "👤"
        })
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Process query
        with st.spinner("🔍 Analyzing..."):
            try:
                context = get_relevant_chunks(st.session_state.vectorstore, prompt)
                
                if debug_mode:
                    with st.expander("Debug Context"):
                        st.write(context)
                
                response = query_llama_api(prompt, context)
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "avatar": "🤖",
                    "source": st.session_state.current_source_name
                })
                
                # Display assistant response
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(response)
                    st.caption(f"Source: {st.session_state.current_source_name}")

            except Exception as e:
                st.error(f"❌ Query failed: {str(e)}")

if __name__ == "__main__":
    main()