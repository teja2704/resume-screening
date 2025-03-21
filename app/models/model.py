# app/models/model.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def match_candidate(resume_text, job_description):
    """
    Match candidate resume with job description using TF-IDF and cosine similarity.
    """
    documents = [resume_text, job_description]

    # Vectorize the documents
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Calculate cosine similarity
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    
    # Convert score to percentage
    match_percentage = round(similarity_score * 100, 2)
    return match_percentage
