import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_qa():
    return pipeline(
        "question-answering", 
        model="bert-large-uncased-whole-word-masking-finetuned-squad", 
        tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad",
        topk=3  # Get top 3 answers
    )
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://i.imgur.com/0XkS5zX.png" alt="AskGenie AI" width="120"/>
        <h1 style="color: #4a148c;">🧞 AskGenie AI</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f7fa, #e1bee7);
        background-attachment: fixed;
    }
    .css-18e3th9 {
        background: transparent !important;
    }
    h1, h2, h3 {
        color: #4a148c;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 AI-QA App")

question = st.text_input("❓ Question")
context = st.text_area("📄 Context")

if context and question:
    qa = load_qa()
    results = qa(question=question, context=context)

    st.markdown("### 🔍 Top Answers")
    for i, result in enumerate(results):
        st.write(f"**Answer {i+1}:** {result['answer']}  \n**Score:** {result['score']:.4f}")
