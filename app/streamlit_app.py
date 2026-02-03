import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from rag_pipeline import rag_answer

st.set_page_config(
    page_title="Doc-Buddy",
    layout="centered"
)

st.title("Doc-Buddy : Agent to learn syntax")
st.caption("Hybrid Retrieval (BM25 + FAISS + RRF) + LLM")

query = st.text_input(
    "Ask a programming question:",
    placeholder="e.g. Difference between ArrayList and LinkedList"
)

if st.button("Get Answer") and query.strip():
    with st.spinner("Thinking..."):
        result = rag_answer(query)

    if isinstance(result, str):
        st.markdown("### Answer")
        st.write(result)
        st.stop()

    if result["is_comparative"]:
        st.info("Comparative query detected.")

    st.markdown("### Answer")
    st.write(result["answer"])

else:
    st.info("Enter a question and click **Get Answer**.")
