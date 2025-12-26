import streamlit as st
import pandas as pd
import os




st.set_page_config(
    page_title="Government Information QA System",
    page_icon="🏛️",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    input {
        border-radius: 10px !important;
        padding: 12px !important;
        font-size: 16px !important;
    }

    button {
        border-radius: 10px !important;
        background-color: #1f6feb !important;
        color: white !important;
        font-weight: 600 !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
        font-size: 15px;
    }

    .answer-box {
        background-color: #111827;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
        border-left: 4px solid #1f6feb;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.title("🏛️ Government Information Question Answering System")

st.markdown(
    "An AI-powered system designed to retrieve concise and relevant answers "
    "from large-scale public sector information repositories."
)

def is_small_talk(query):
    small_talk = [
        "how are you",
        "hello",
        "hi",
        "hey",
        "good morning",
        "good evening",
        "what's up"
    ]
    q = query.lower().strip()
    return any(phrase in q for phrase in small_talk)








#Thresholds 
RETRIEVAL_MIN_SCORE = 0.25
QA_MIN_CONFIDENCE = 0.25


import re

def clean_answer_text(text):
    import re

    # Remove minister headers
    text = re.sub(r"THE MINISTER.*?:", "", text, flags=re.I)

    # Remove brackets
    text = re.sub(r"\[.*?\]", "", text)

    # Remove leading uppercase noise
    text = re.sub(r"^[A-Z\s\(\)\.]{20,}", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "government_qa_clean.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="latin1")

df = load_data()

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

EMB_PATH = os.path.join(BASE_DIR, "data", "processed", "question_embeddings.npy")
FAISS_PATH = os.path.join(BASE_DIR, "data", "processed", "faiss_index.index")

st.success(f"Dataset loaded successfully! Total records: {len(df)}")

@st.cache_resource
def load_models():
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    qa = pipeline(
        "question-answering",
        model="deepset/bert-base-cased-squad2"
    )
    return sbert, qa


@st.cache_resource
def load_faiss():
    embeddings = np.load(EMB_PATH)
    faiss.normalize_L2(embeddings)
    index = faiss.read_index(FAISS_PATH)
    return embeddings, index

sbert_model, qa_reader = load_models()
question_embeddings, index = load_faiss()

st.success("Models and index loaded successfully")

def answer_question(query, top_k=3):

    if is_small_talk(query):
        return {
            "type": "no_answer",
            "message": "This system is designed to answer informational questions only."
        }

    # 1️⃣ Encode query
    query_embedding = sbert_model.encode([query])
    faiss.normalize_L2(query_embedding)

    # 2️⃣ FAISS search
    scores, indices = index.search(query_embedding, top_k)
    max_score = scores[0][0]

    # 3️⃣ Retrieval gate
    if max_score < RETRIEVAL_MIN_SCORE:
        return {
            "type": "no_answer",
            "message": "No relevant information found."
        }

    # 4️⃣ Build compact context
    contexts = []
    for idx in indices[0]:
        text = str(df.iloc[idx]["answer"]).replace("\n", " ")
        contexts.append(text[:400])

    combined_context = " ".join(contexts)

    # 5️⃣ Extract answer (BERT)
    qa_result = qa_reader(
        question=query,
        context=combined_context
    )

    
    # 6️⃣ QA confidence fallback
    if qa_result["score"] < QA_MIN_CONFIDENCE:
        fallback_answers = []

    for idx in indices[0]:
        raw = str(df.iloc[idx]["answer"])
        clean = clean_answer_text(raw)

        if len(clean) > 50:
            # keep first 2–3 lines worth of content
            clean = " ".join(clean.split()[:45])
            fallback_answers.append(clean)

        return {
        "type": "partial",
        "message": "Relevant information found. Showing summarized answers.",
        "confidence": round(max_score, 3),
        "answers": fallback_answers[:3]
        }


    # 7️⃣ Final clean answer (2–3 lines)
    answer = clean_answer_text(qa_result["answer"])
    answer = " ".join(answer.split()[:50])

    return {
        "type": "answer",
        "answer": answer,
        "confidence": round(qa_result["score"], 3)
    }


test_query = "How can I apply for passport?"
response = answer_question(test_query)

st.subheader("Ask a Question")

query = st.text_input("Enter your question here", key="user_query")


if st.button("Get Answer"):

    if not query.strip():
        st.warning("Please enter a question.")
    else:
        response = answer_question(query)

        if response["type"] == "answer":
            st.success("✅ Answer found")
            st.caption(f"Model confidence: {response['confidence']}")
            st.markdown(
                f"<div class='answer-box'>{response['answer']}</div>",
                unsafe_allow_html=True
            )

        elif response["type"] == "partial":
            st.warning("⚠️ Relevant information found")
            st.caption(f"Retrieval confidence: {response['confidence']}")

            for i, ans in enumerate(response["answers"], 1):
                st.markdown(
                    f"<div class='answer-box'><b>{i}.</b> {ans}</div>",
                    unsafe_allow_html=True
                )

        else:
            st.error("❌ No relevant information found")


st.markdown("---")
st.caption("© Academic Demonstration Project | AI-based Question Answering System")

