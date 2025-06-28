import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
import os
import hashlib

# --- UI SETUP ---
st.set_page_config(page_title="AI-Powered Webpage Chat (RAG+LLM)", page_icon="🌐", layout="centered")
st.title("AI-Powered Webpage Chat 🌐")
st.caption("Ask questions about any webpage—powered by local Llama 3.2 (Ollama) and Retrieval-Augmented Generation.")

# --- OLLAMA CONFIG ---
OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.2"

# --- Directory Helper ---
def url_to_dir(url, base_dir="chroma_store"):
    """Generate a unique persistent directory for a given URL."""
    os.makedirs(base_dir, exist_ok=True)
    url_hash = hashlib.sha1(url.encode()).hexdigest()
    return os.path.join(base_dir, url_hash)

# --- STATE MANAGEMENT ---
if "last_url" not in st.session_state:
    st.session_state.last_url = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "docs" not in st.session_state:
    st.session_state.docs = None

# --- USER INPUT ---
webpage_url = st.text_input("Enter Webpage URL", type="default")

def safe_load_and_index(url):
    """Load, split, embed, and store webpage in a persistent directory (Windows-safe)."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        if not docs or len(docs) == 0 or len(docs[0].page_content.strip()) == 0:
            st.error("Webpage loaded but appears empty.")
            return None, None, None
        # Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
        splits = text_splitter.split_documents(docs)
        # Persistent directory per URL
        persist_dir = url_to_dir(url)
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_ENDPOINT)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name="rag-webpage"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        return vectorstore, retriever, docs
    except Exception as e:
        st.error(f"Failed to load/index webpage: {e}")
        return None, None, None

if webpage_url:
    if webpage_url != st.session_state.last_url:
        st.info("Loading and indexing webpage. Please wait…")
        vectorstore, retriever, docs = safe_load_and_index(webpage_url)
        if vectorstore and retriever and docs:
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
            st.session_state.docs = docs
            st.session_state.last_url = webpage_url
            st.success(f"Loaded and indexed: {webpage_url}")
        else:
            st.stop()
    elif st.session_state.vectorstore is not None:
        st.success(f"Webpage ready: {webpage_url}")
    else:
        st.warning("No valid webpage indexed yet.")
        st.stop()
else:
    st.info("Enter a webpage URL to get started.")
    st.stop()

# --- LLM INFERENCE (RAG) ---
ollama = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_ENDPOINT)

def ollama_llm(question, context):
    """Send prompt to Ollama Llama3, return answer (with robust error handling)."""
    try:
        prompt = f"Answer the following question based only on the context below.\n\nQuestion: {question}\n\nContext:\n{context}"
        response = ollama.invoke([('human', prompt)])
        return response.content.strip() if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"Ollama inference failed: {e}"

def combine_docs(docs):
    """Concatenate document content for prompt context."""
    if not docs: return ""
    return "\n\n".join(doc.page_content for doc in docs if hasattr(doc, "page_content"))

def rag_chain(question):
    """RAG: Retrieve context, answer with LLM."""
    try:
        docs = st.session_state.retriever.invoke(question)
        formatted_context = combine_docs(docs)
        if not formatted_context:
            return "No relevant content retrieved from the webpage for this question."
        return ollama_llm(question, formatted_context)
    except Exception as e:
        return f"RAG failed: {e}"

# --- INTERACTIVE CHAT ---
with st.form("question_form"):
    prompt = st.text_input("Ask any question about the webpage", key="user_question", placeholder="e.g., What is this page about?")
    submitted = st.form_submit_button("Ask")
    if submitted and prompt.strip():
        with st.spinner("Thinking..."):
            result = rag_chain(prompt.strip())
        st.markdown("**Answer:**")
        st.write(result)
    elif submitted:
        st.warning("Please enter a non-empty question.")

# --- UI EXTRAS ---
with st.expander("Show indexed chunks (for debugging)", expanded=False):
    if st.session_state.docs:
        for idx, doc in enumerate(st.session_state.docs):
            st.markdown(f"**Chunk {idx+1}:**")
            st.code(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

st.markdown("---")
st.caption("Built with [Streamlit](https://streamlit.io/), [LangChain](https://langchain.dev/), [ChromaDB](https://www.trychroma.com/), and [Ollama](https://ollama.com/) Llama 3.2.")
