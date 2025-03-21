import streamlit as st
import fitz  # PyMuPDF for PDF processing
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to calculate match score
def calculate_match_score(resume_text, job_description):
    if not resume_text or not job_description:
        return 0.0

    # Vectorize the text
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(vectors)
    match_score = round(similarity_matrix[0][1] * 100, 2)
    
    return match_score

# Streamlit UI
st.title("🎯 Resume Screening & Matching")
st.write("Upload a resume and enter a job description to calculate the match score.")

# File upload
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# Job description input
job_description = st.text_area("Paste Job Description Here", height=200)

if uploaded_file and job_description:
    st.success("✅ File uploaded and description provided!")

    # Extract text from PDF
    resume_text = extract_text_from_pdf(uploaded_file)

    # Calculate match score
    match_score = calculate_match_score(resume_text, job_description)

    # Display results
    st.subheader("📄 Parsed Resume Text")
    st.text_area("Resume Content", resume_text, height=300)

    st.subheader("🎯 Match Score")
    st.write(f"**Match Score:** {match_score}%")
