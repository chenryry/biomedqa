import streamlit as st
import os
import tempfile
from util import load_model_and_index, embed_new_doc, get_context_and_answer, generate_answer_rag


st.set_page_config(page_title="Biomedical Research Assistant")

st.title("Biomedical Chatbot")

st.sidebar.header("Upload Document (optional)")
uploaded_file = st.sidebar.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        doc_path = tmp_file.name
    embed_new_doc(doc_path)
    st.sidebar.success("Document embedded and added to the knowledge base.")


model, tokenizer, index, embedder, contexts = load_model_and_index()

# Chat UI
query = st.text_input("Ask a biomedical research question:")

if query:
    with st.spinner("Generating answer..."):
        answer = get_context_and_answer(query, model, tokenizer, index, embedder, contexts)
        st.markdown(f"**Answer:** {answer}")
