import os
import re
import pickle
import pdfplumber
import requests
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- CONFIG ---
COHERE_API_KEY = "XlkUMR07gD8HtNbXB2fiAa9jBbrfB04c2ffLX89F"
st.set_page_config(page_title="🧠 Question Analyzer AI", layout="wide")


# --- FUNCTIONS ---
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_questions(text):
    lines = text.split("\n")
    return [line.strip() for line in lines if re.search(r'[?]', line)]


def get_cohere_answer(question):
    url = "https://api.cohere.ai/v1/chat"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "command-r",
        "message": question
    }
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 200:
        return res.json().get("text", "⚠️ No answer received.")
    return f"⚠️ Error: {res.json().get('message', 'Something went wrong.')}"


# --- SIDEBAR UPLOAD ---
st.sidebar.header("📂 Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload one or more question paper PDFs", type="pdf", accept_multiple_files=True)

# --- TITLE ---
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>📄 Question Analyzer with AI Support</h1>
    <p style='text-align: center;'>Upload science papers, detect similar questions, and get AI-powered answers.</p>
""", unsafe_allow_html=True)

if uploaded_files:
    all_questions = []
    for file in uploaded_files:
        st.sidebar.success(f"✅ Uploaded: {file.name}")
        text = extract_text_from_pdf(file)
        extracted = extract_questions(text)
        all_questions.extend(extracted)

    all_questions = list(set(all_questions))
    st.success(f"✅ Extracted {len(all_questions)} unique questions.")

    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_questions)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # --- DUPLICATE DETECTION ---
    threshold = 0.4
    duplicates = []
    for i in range(len(all_questions)):
        for j in range(i + 1, len(all_questions)):
            if similarity_matrix[i, j] > threshold:
                duplicates.append((all_questions[i], all_questions[j], round(similarity_matrix[i, j], 2)))

    if duplicates:
        st.subheader("🔁 Detected Duplicate Questions")
        dup_df = pd.DataFrame(duplicates, columns=["Question 1", "Question 2", "Similarity"])
        st.dataframe(dup_df, use_container_width=True)

        # --- MODEL TRAINING ---
        labels = [q1 for q1, _, _ in duplicates]
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                vectorizer.transform(labels).toarray(), y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            with open("repeated_questions_model.pkl", "wb") as f:
                pickle.dump(model, f)
            with open("tfidf_vectorizer.pkl", "wb") as f:
                pickle.dump(vectorizer, f)

            st.success("✅ Model trained to recognize future repeated questions.")

        except Exception as e:
            st.warning(f"⚠️ Model training skipped: {e}")

        # --- AI Q&A SECTION ---
        st.subheader("🤖 Ask AI for an Answer")
        ai_questions = sorted(set([q1 for q1, _, _ in duplicates]))
        selected_question = st.selectbox("Choose a question to get an AI-generated answer:", ai_questions)

        if st.button("💡 Get AI Answer"):
            with st.spinner("Thinking..."):
                answer = get_cohere_answer(selected_question)
                st.success("✅ Answer Generated:")
                with st.expander(f"📘 {selected_question}"):
                    st.write(answer)

    else:
        st.warning("⚠️ No duplicate questions were detected.")

else:
    st.info("👈 Please upload one or more PDF files to begin.")

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ using Streamlit + Cohere</center>", unsafe_allow_html=True)
