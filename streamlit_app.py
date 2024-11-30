import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.title("PDF Question Answering System")

# File upload section
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    files = {'file': uploaded_file}
    response = requests.post('http://localhost:5000/upload', files=files)
    if response.status_code == 200:
        st.success("PDF uploaded and processed successfully!")

# Chat section
user_question = st.text_input("Ask a question about your PDF:")
if user_question:
    payload = {'message': user_question}
    response = requests.post('http://localhost:5000/chat', json=payload)
    if response.status_code == 200:
        st.write("Answer:", response.json()['response'])
