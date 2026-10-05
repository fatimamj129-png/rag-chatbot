import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("My RAG Chatbot")
st.write("Ask questions about your notes. ")

@st.cache_data
def load_chunks():
    chunks = []

    for file_path in Path("docs").glob("*.pdf"):
        reader = PdfReader(file_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for start in range(0, len(text), 700):
                piece = text[start:start + 800].strip()
                if piece:
                    chunks.append({
                        "text": piece,
                        "source": f"{file_path.name}, page {page_number}"
                    })

    for file_path in Path("docs").glob("*.docx"):
        document = Document(file_path)
        text = "\n".join(p.text for p in document.paragraphs)
        for start in range(0, len(text), 700):
            piece = text[start:start + 800].strip()
            if piece:
                chunks.append({
                    "text": piece,
                    "source": file_path.name
                })

    return chunks

chunks = load_chunks()

if not chunks:
    st.warning("No PDF or DOCX files found in the docs folder.")
    st.stop()

question = st.text_input("Ask your question")

if question:
    vectorizer = TfidfVectorizer()
    document_vectors = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, document_vectors)[0]
    best_indexes = scores.argsort()[-3:][::-1]

    found = False
    for index in best_indexes:
        if scores[index] > 0:
            found = True
            st.write(chunks[index]["text"])
            st.caption("Source: " + chunks[index]["source"])

    if not found:
        st.write("I couldn't find relevant information in your notes. ")