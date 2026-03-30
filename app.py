"""
HealthDoc Assistant – A domain-specific RAG chatbot that answers questions
using only the documents stored in the docs/ folder.

Architecture:
  - Document loading + 500-char chunking  (LangChain)
  - HuggingFace sentence-transformer embeddings stored in a FAISS index
  - Compliance prompt with few-shot examples prevents hallucinations
  - LCEL chain: retriever → prompt → LLM → response
  - Streamlit UI with chat history, clear-chat, and upload-new-docs
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_DOCS = 4

# ---------------------------------------------------------------------------
# Compliance system prompt with few-shot examples
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a specialized HealthDoc Knowledge Assistant. \
Your sole purpose is to answer questions based on the health documents \
provided to you in the context below.

STRICT RULES:
1. Use ONLY the provided context to answer. Do NOT use outside knowledge.
2. If the answer is not found in the context, respond exactly with:
   "I am sorry, that information is not in my database."
3. Never speculate, guess, or fabricate information.
4. Keep answers clear, concise, and factual.

--- FEW-SHOT EXAMPLES ---

Example 1 – Good Answer:
User: How much water should an adult drink per day?
Assistant: Adults should drink approximately 8 glasses (about 2 liters) of \
water per day to maintain proper hydration, according to the health \
guidelines in the knowledge base.

Example 2 – Proper Refusal:
User: What is the current stock price of Johnson & Johnson?
Assistant: I am sorry, that information is not in my database.

--- END EXAMPLES ---

Context:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


# ---------------------------------------------------------------------------
# Helper – format retrieved documents into a single string
# ---------------------------------------------------------------------------
def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Cached resources (rebuilt only when the cache is cleared)
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
        st.error("No documents found in the docs/ folder.")
        st.stop()

    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n",
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


@st.cache_resource(show_spinner="Connecting to language model…")
def load_llm():
    """Return a ChatOpenAI instance, reading the key from env or st.secrets."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except (KeyError, FileNotFoundError):
            api_key = ""
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=api_key,
    )


def build_rag_chain(vector_store, llm):
    """Compose the LCEL retrieval-augmented-generation chain."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_DOCS},
    )
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
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
        st.caption("A compliant, document-grounded health chatbot.")
        st.divider()

        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # Upload new documents
        st.subheader("📁 Upload New Docs")
        uploaded_files = st.file_uploader(
            "Add .txt files to the knowledge base",
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
                # Clear caches so the new docs are indexed
                load_vector_store.clear()
                load_llm.clear()
                st.success(f"✅ Added {len(uploaded_files)} document(s). Rebuilding index…")
                st.rerun()

        st.divider()
        st.info(
            "💡 This assistant answers questions **only** from its health "
            "document library. Off-topic queries will be politely declined."
        )

    # ---- Main area ---------------------------------------------------------
    st.title("🏥 HealthDoc Assistant")
    st.markdown(
        "_Your specialized, document-grounded health knowledge assistant._"
    )

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Load resources
    try:
        vector_store = load_vector_store()
        llm = load_llm()
        rag_chain = build_rag_chain(vector_store, llm)
    except Exception as exc:
        st.error(f"Failed to initialize the knowledge base: {exc}")
        return

    # Chat input
    if user_input := st.chat_input("Ask a health-related question…"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base…"):
                try:
                    response = rag_chain.invoke(user_input)
                except Exception as exc:
                    response = f"⚠️ Error generating response: {exc}"
            st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )


if __name__ == "__main__":
    main()
