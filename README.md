<<<<<<< HEAD
# 💹 AI-Powered Financial Summarizer  

## 📝 Description  
**AI-Powered Financial Summarizer** is a Streamlit-based app designed to extract, summarize, and analyze financial reports (PDFs) using NLP and AI models. It helps users quickly obtain key insights, such as financial summaries, revenue/net income/EPS metrics, sentiment analysis, and answer specific questions about financial documents.  

The app integrates **HuggingFace models, LangChain (RAG), FinBERT to provide a robust financial document processing solution. It also features interactive visualizations and a clean UI for a seamless experience.  

---

## ✨ Features  
- 📄 **PDF Extraction & Cleaning** – Extract text from financial reports using `pdfplumber`.  
- 🤖 **AI Summarization** – Summarize reports using HuggingFace BART (`facebook/bart-large-cnn`).  
- ❓ **Query & Q&A** – Ask context-based questions with `LangChain` and `ChromaDB` (RAG).  
- 💰 **Financial Metrics Extraction** – Extract & normalize key metrics (Revenue, Net Income, EPS).  
- 📊 **Visualizations** – Generate financial charts with `matplotlib`.  
- 🎨 **Streamlit UI** – Multi-tab interface with sidebar navigation & instructions.  

---

## 🔗 Links  
- **[GitHub Repository](https://github.com/S-Shipra/AI-Powered-Financial-Summarizer)**  

---

## 🤖 Tech Stack  
### Frontend / UI  
- Streamlit (Python-based web app framework)  
- Matplotlib (Charts & Visualization)  

### Backend / ML & NLP  
- HuggingFace Transformers (`facebook/bart-large-cnn` for summarization)  
- LangChain + ChromaDB (RAG for Q&A)  
- PDFPlumber (Text extraction)  

### Core Libraries  
- Python 3.10+  
- Torch (PyTorch backend for HuggingFace models)  

---

## 📈 Progress  
### ✅ Fully Implemented  
- PDF Extraction & Cleaning  
- Financial Summarization (BART)  
- Q&A via LangChain + ChromaDB  
- Metrics Extraction & Normalization
- Sentiment Analysis (FinBERT)  
- Interactive Visualizations (matplotlib)  
- Streamlit Tabs & Sidebar UI  

### 🔧 Partially Implemented  
- Persistent ChromaDB storage (currently session-only on Streamlit Cloud)  

---

## 🔮 Future Scope  
- 📊 Multi-report comparison to track KPI trends.  
- 🔍 Advanced financial NER for deeper entity tagging.  
- 📤 Integration with EDGAR/stock APIs for auto-fetching reports.  
- 📱 Mobile-friendly Streamlit UI.  

---

## 💸 Applications  
- **Investors & Analysts** – Quickly summarize and extract key KPIs.  
- **Finance Teams** – Automate quarterly report analysis.  
- **Students & Researchers** – Learn NLP in finance.  
- **Journalists** – Speed up financial report review.  

---

## 🛠 Project Setup  
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/S-Shipra/AI-Powered-Financial-Summarizer.git
cd AI-Powered-Financial-Summarizer

=======
# 💹 AI-Powered Financial Summarizer  

## 📝 Description  
**AI-Powered Financial Summarizer** is a Streamlit-based app designed to extract, summarize, and analyze financial reports (PDFs) using NLP and AI models. It helps users quickly obtain key insights, such as financial summaries, revenue/net income/EPS metrics, sentiment analysis, and answer specific questions about financial documents.  

The app integrates **HuggingFace models, LangChain (RAG), FinBERT to provide a robust financial document processing solution. It also features interactive visualizations and a clean UI for a seamless experience.  

---

## ✨ Features  
- 📄 **PDF Extraction & Cleaning** – Extract text from financial reports using `pdfplumber`.  
- 🤖 **AI Summarization** – Summarize reports using HuggingFace BART (`facebook/bart-large-cnn`).  
- ❓ **Query & Q&A** – Ask context-based questions with `LangChain` and `ChromaDB` (RAG).  
- 💰 **Financial Metrics Extraction** – Extract & normalize key metrics (Revenue, Net Income, EPS).  
- 📊 **Visualizations** – Generate financial charts with `matplotlib`.  
- 🎨 **Streamlit UI** – Multi-tab interface with sidebar navigation & instructions.  

---

## 🔗 Links  
- **[GitHub Repository](https://github.com/S-Shipra/AI-Powered-Financial-Summarizer)**  
- **Screenshots**: _[Add Google Drive link here]_  

---

## 🤖 Tech Stack  
### Frontend / UI  
- Streamlit (Python-based web app framework)  
- Matplotlib (Charts & Visualization)  

### Backend / ML & NLP  
- HuggingFace Transformers (`facebook/bart-large-cnn` for summarization)  
- LangChain + ChromaDB (RAG for Q&A)  
- PDFPlumber (Text extraction)  

### Core Libraries  
- Python 3.10+  
- Torch (PyTorch backend for HuggingFace models)  
- SentencePiece, Protobuf  

---

## 📈 Progress  
### ✅ Fully Implemented  
- PDF Extraction & Cleaning  
- Financial Summarization (BART)  
- Q&A via LangChain + ChromaDB  
- Metrics Extraction & Normalization
- Sentiment Analysis (FinBERT)  
- Interactive Visualizations (matplotlib)  
- Streamlit Tabs & Sidebar UI  

### 🔧 Partially Implemented  
- Persistent ChromaDB storage (currently session-only on Streamlit Cloud)  

---

## 🔮 Future Scope  
- 📊 Multi-report comparison to track KPI trends.  
- 🔍 Advanced financial NER for deeper entity tagging.  
- 📤 Integration with EDGAR/stock APIs for auto-fetching reports.  
- 📱 Mobile-friendly Streamlit UI.  

---

## 💸 Applications  
- **Investors & Analysts** – Quickly summarize and extract key KPIs.  
- **Finance Teams** – Automate quarterly report analysis.  
- **Students & Researchers** – Learn NLP in finance.  
- **Journalists** – Speed up financial report review.  

---

## 🛠 Project Setup  
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/S-Shipra/AI-Powered-Financial-Summarizer.git
cd AI-Powered-Financial-Summarizer

>>>>>>> 2d0a8ac (Initial project setup with app.py, requirements, and .gitignore)
