import streamlit as st
import pdfplumber
import re
import os
import matplotlib.pyplot as plt
import torch
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------ SETUP ------------------ #
st.set_page_config(page_title="AI Financial Summarizer", layout="wide")
import spacy
try:
    nlp = spacy.load("en_core_web_trf")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_trf")
    nlp = spacy.load("en_core_web_trf")

torch.set_num_threads(1)  # Resource limit for Streamlit Cloud

# ------------------ HELPERS ------------------ #
def extract_text_pdf(file, skip_pages=3):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            if page_num >= skip_pages:
                text += page.extract_text()
    return text

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'Page \d+ of \d+', '', text)
    return text.strip()

def normalize_metric(value, unit):
    if unit.lower().startswith('b'):  # Billion
        return value * 1e9
    elif unit.lower().startswith('m'):  # Million
        return value * 1e6
    return value

def extract_metrics(text):
    metrics = {}
    # Regex for Revenue/Net Income/EPS
    patterns = {
        "Revenue": r"Revenue[\s:]+\$?([\d,.]+)\s?([MB]?)",
        "Net Income": r"Net Income[\s:]+\$?([\d,.]+)\s?([MB]?)",
        "EPS": r"EPS[\s:]+\$?([\d,.]+)"
    }
    for metric, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1).replace(",", ""))
            unit = match.group(2) if len(match.groups()) > 1 else ""
            metrics[metric] = normalize_metric(value, unit) if metric != "EPS" else value
    return metrics

def plot_metrics(metrics):
    if not metrics:
        return None
    fig, ax = plt.subplots()
    names = list(metrics.keys())
    values = list(metrics.values())
    ax.bar(names, values)
    ax.set_ylabel("Value (scaled)")
    ax.set_title("Extracted Financial Metrics")
    return fig

# ------------------ LOAD MODELS ------------------ #
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    return summarizer, qa_model, qa_tokenizer, finbert_model, finbert_tokenizer

summarizer, qa_model, qa_tokenizer, finbert_model, finbert_tokenizer = load_models()

def analyze_sentiment(text):
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = finbert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["Negative", "Neutral", "Positive"]
    return labels[probs.argmax()], probs.max().item()

# ------------------ UI ------------------ #
with st.sidebar:
    st.title("⚡ AI Financial Summarizer")
    st.markdown("Upload a PDF and extract insights.")
    st.markdown("### Tabs:")
    st.markdown("- 📄 Summarization\n- ❓ Q&A\n- 📊 Metrics\n- 📈 Sentiment Analysis")

st.title("💹 AI-Powered Financial Summarizer")

uploaded_file = st.file_uploader("Upload a financial report (PDF):", type="pdf")

if uploaded_file:
    text = clean_text(extract_text_pdf(uploaded_file))

    tabs = st.tabs(["📄 Summarization", "❓ Q&A", "📊 Metrics", "📈 Sentiment Analysis"])

    # ---- Summarization ----
    with tabs[0]:
        st.subheader("Summary")
        summary = summarizer(text, max_length=300, min_length=100, do_sample=False)[0]['summary_text']
        st.write(summary)
        st.download_button("📥 Download Summary", summary, file_name="summary.txt")

    # ---- Q&A ----
    with tabs[1]:
        st.subheader("Ask Questions")
        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            inputs = qa_tokenizer(question, text, return_tensors="pt")
            outputs = qa_model(**inputs)
            start = torch.argmax(outputs.start_logits)
            end = torch.argmax(outputs.end_logits) + 1
            answer = qa_tokenizer.convert_tokens_to_string(
                qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start:end])
            )
            st.write(f"**Answer:** {answer}")

    # ---- Metrics ----
    with tabs[2]:
        st.subheader("Extracted Metrics")
        metrics = extract_metrics(text)
        if metrics:
            st.write(metrics)
            fig = plot_metrics(metrics)
            if fig:
                st.pyplot(fig)
        else:
            st.warning("No metrics found.")

    # ---- Sentiment ----
    with tabs[3]:
        st.subheader("Sentiment Analysis (FinBERT)")
        custom_text = st.text_area("Enter financial text for sentiment analysis:", value=summary)
        if st.button("Analyze Sentiment"):
            sentiment, confidence = analyze_sentiment(custom_text)
            st.write(f"**Sentiment:** {sentiment} ({confidence*100:.2f}% confidence)")
