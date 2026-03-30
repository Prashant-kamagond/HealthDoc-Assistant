# 🏥 HealthDoc Assistant

A **document-grounded health chatbot** powered by **Ollama** and **LangChain**. Ask health questions and get answers directly from your custom health documents!

---

## ✨ Features

- 📚 **Document-Based**: Answers questions ONLY from your health documents
- 🔒 **Privacy First**: All data stays local on your computer
- 💰 **Completely Free**: No API costs, no subscriptions
- ⚡ **Fast & Local**: Ollama models run locally for instant responses
- 🎯 **Accurate**: RAG (Retrieval-Augmented Generation) pipeline
- 🌐 **Easy UI**: Clean Streamlit interface
- 📁 **Easy Document Upload**: Add health documents via sidebar

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM (Language Model)** | Ollama (Mistral) |
| **Embeddings** | Ollama Embeddings |
| **Vector Database** | FAISS |
| **Framework** | LangChain |
| **UI Framework** | Streamlit |
| **Document Loader** | LangChain Document Loaders |
| **Text Splitter** | LangChain Character Text Splitter |

---

## 📋 Requirements

### System Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB+ for Ollama models
- **Python**: 3.8 or higher

### Software
- Ollama installed
- Python 3.8+
- pip (Python package manager)

---

## 🚀 Quick Start

### Step 1: Install Ollama

**Windows:**
```powershell
irm https://ollama.com/install.ps1 | iex
Step 2: Clone Repository
bash
git clone https://github.com/Prashant-kamagond/HealthDoc-Assistant.git
cd HealthDoc-Assistant
Step 3: Install Python Dependencies
bash
pip install -r requirements.txt
Step 4: Download Ollama Model
bash
ollama pull mistral
This downloads the Mistral model (~4GB). Takes a few minutes.

Step 5: Run the Application
bash
streamlit run app.py
The app will open at: http://localhost:8501
