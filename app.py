"""
HealthDoc Assistant – A domain-specific RAG chatbot using Ollama
Simplified version - no PyTorch/HuggingFace embeddings needed
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Ollama Configuration
OLLAMA_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"

TOP_K_DOCS = 4

# ---------------------------------------------------------------------------
# Compliance system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a specialized HealthDoc Knowledge Assistant.
Your sole purpose is to answer questions based ONLY on the health documents provided.

STRICT RULES:
1. Use ONLY the provided context to answer questions.
2. Do NOT use any outside knowledge.
3. If the answer is not found in the context, respond with:
   "I am sorry, that information is not in my database."
4. Never speculate, guess, or fabricate information.
5. Keep answers clear, concise, and factual.

Context:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


# ---------------------------------------------------------------------------
# Helper – format retrieved documents
# ---------------------------------------------------------------------------
def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Building knowledge base…")
def load_vector_store():
    """Load docs → chunk → embed → FAISS index."""
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    if not documents:
        st.error("❌ No documents found in the docs/ folder.")
        st.info("📁 Please add .txt files to the docs/ folder first.")
        st.stop()

    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n",
    )
    chunks = splitter.split_documents(documents)

    # Use Ollama embeddings instead of HuggingFace
    embeddings = OllamaEmbeddings(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


@st.cache_resource(show_spinner="Connecting to Ollama…")
def load_llm():
    """Load Ollama LLM"""
    try:
        llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            num_predict=256,
        )
        
        # Test connection
        test_response = llm.invoke("test")
        if not test_response:
            raise Exception("No response from Ollama")
        
        return llm
        
    except Exception as e:
        st.error(f"❌ Failed to connect to Ollama")
        st.error(f"Error: {str(e)}")
        st.warning(
            f"""
            **Setup Instructions:**
            
            1. **Make sure Ollama is running**
            2. **Download the model:**
               ```bash
               ollama pull {OLLAMA_MODEL}
               ```
            3. **Verify connection:**
               ```bash
               ollama list
               ```
            4. **Restart the app**
            """
        )
        st.stop()


def build_rag_chain(vector_store, llm):
    """Compose the RAG chain."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_DOCS},
    )
    
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough()
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    
    return chain


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="HealthDoc Assistant",
        page_icon="🏥",
        layout="wide",
    )

    # ---- Sidebar -----------------------------------------------------------
    with st.sidebar:
        st.title("🏥 HealthDoc Assistant")
        st.caption("Health chatbot powered by Ollama")
        st.divider()

        st.subheader("⚙️ Settings")
        
        with st.expander("🤖 Model Information", expanded=False):
            st.success(f"✅ Model: {OLLAMA_MODEL}")
            st.info(
                f"""
**Model**: {OLLAMA_MODEL}

**Type**: Ollama (Local)

**URL**: {OLLAMA_BASE_URL}

**Privacy**: 🔒 Data stays local

**Cost**: 💰 Free
                """
            )
        
        if st.button("🧪 Test Connection", use_container_width=True):
            try:
                test_llm = OllamaLLM(
                    model=OLLAMA_MODEL,
                    base_url=OLLAMA_BASE_URL,
                )
                response = test_llm.invoke("Say 'Working'")
                st.success(f"✅ Connected! Response: {response}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        st.subheader("📁 Upload Docs")
        uploaded_files = st.file_uploader(
            "Select .txt files",
            type=["txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        
        if uploaded_files:
            if st.button("📥 Add to Knowledge Base", use_container_width=True):
                for uf in uploaded_files:
                    dest = os.path.join(DOCS_DIR, uf.name)
                    with open(dest, "wb") as fh:
                        fh.write(uf.getvalue())
                
                load_vector_store.clear()
                load_llm.clear()
                st.success(f"✅ Added {len(uploaded_files)} document(s)")
                st.rerun()

        st.divider()
        st.info("💡 Answers questions only from health documents.")

    # ---- Main area ---------------------------------------------------------
    st.title("🏥 HealthDoc Assistant")
    st.markdown("_Document-grounded health chatbot (Ollama)_")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    try:
        vector_store = load_vector_store()
        llm = load_llm()
        rag_chain = build_rag_chain(vector_store, llm)
    except Exception as exc:
        st.error(f"Failed to initialize: {exc}")
        return

    if user_input := st.chat_input("Ask a health question…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching…"):
                try:
                    response = rag_chain.invoke(user_input)
                except Exception as exc:
                    response = f"Error: {str(exc)}"
            
            st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )


if __name__ == "__main__":
    main()
