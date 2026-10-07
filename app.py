import os
from pathlib import Path


import streamlit as st
from docx import Document
from openai import OpenAI
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="RAG AI Chatbot", page_icon="💬", layout="centered")


st.markdown("""
<style>
.block-container {
    max-width: 760px;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.chat-title, .chat-description { text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="chat-title">RAG AI Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="chat-description">Ask a question about the documents and get a clear, sourced answer.</p>', unsafe_allow_html=True)

@st.cache_data
def load_chunks():
    chunks = []
    docs_dir = next(
        (folder for folder in (Path("docs"), Path("Docs")) if folder.is_dir()),
        Path("docs"),
    )

