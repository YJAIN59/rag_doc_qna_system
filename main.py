from flask import Flask, request, jsonify
from transformers import TableTransformerForObjectDetection
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import pymupdf  # PyMuPDF
import fitz
import pandas as pd
import numpy as np
import torch
import os
from dotenv import load_dotenv
from io import StringIO
from flask_cors import CORS
# from flask import render_template

app = Flask(__name__)
CORS(app)

# @app.route('/')
# def index():
#     return render_template('index.html')


os.makedirs('DB', exist_ok=True)
# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')



# Initialize models and components
table_transformer = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"}, cache_folder="embeddings_cache", encode_kwargs={"normalize_embeddings": True})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

class PDFProcessor:
    def __init__(self):
        self.vector_store = None
        
    def extract_text(self, pdf_path):
        doc = pymupdf.open(pdf_path)
        text_content = []
        for page in doc:
            text_content.append(page.get_text())
        return "\n".join(text_content)
    
    from io import StringIO

    def extract_tables(self, pdf_path):
        doc = pymupdf.open(pdf_path)
        tables = []
        for page in doc:
            pix = page.get_pixmap()
            img_array = np.array(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n), copy=True)
            
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
            img_tensor = img_tensor / 255.0
            
            outputs = table_transformer(img_tensor)
            
            for pred in outputs['pred_boxes'][0]:
                box = pred.detach().cpu().numpy()
                if len(box) >= 4:
                    x1, y1, x2, y2 = map(float, box[:4])
                    table_rect = fitz.Rect(x1, y1, x2, y2)
                    table_text = page.get_text("table", clip=table_rect)
                    try:
                        # Using StringIO to wrap the table text
                        table_df = pd.read_html(StringIO(table_text))[0]
                        tables.append(table_df)
                    except:
                        continue
                        
        return tables
        
    def process_pdf(self, pdf_path):
        # Extract text and tables
        text_content = self.extract_text(pdf_path)
        tables = self.extract_tables(pdf_path)
        
        # Generate embeddings
        text_embeddings = embeddings.embed_query(text_content)
        table_embeddings = []
        for table in tables:
            table_text = table.to_string()
            table_embeddings.append(embeddings.embed_query(table_text))
            
        # Store in VectorDB (FAISS)
        self.vector_store = FAISS.from_texts(
            texts=[text_content] + [t.to_string() for t in tables],
            embedding=embeddings
        )
        
        # Save with allow_dangerous_deserialization flag
        self.vector_store.save_local("DB/vector_store")
        
        return {"status": "success", "message": "PDF processed and vector store saved successfully"}
    
    def load_vector_store(self):
        if os.path.exists("DB/vector_store"):
            self.vector_store = FAISS.load_local(
                "DB/vector_store", 
                embeddings
            )
            return True
        return False

# Initialize RAG model
def initialize_rag():
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use the following context to answer the question:
        Context: {context}
        Question: {question}
        Answer: """
    )
    
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=pdf_processor.vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )
    return qa_chain

# Initialize PDFProcessor
pdf_processor = PDFProcessor()


@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        result = pdf_processor.process_pdf(file_path)
        return jsonify(result)
    
    return jsonify({"error": "Invalid file format"}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    
    if not pdf_processor.vector_store:
        if not pdf_processor.load_vector_store():
            return jsonify({"error": "Please upload a PDF first"}), 400
    
    qa_chain = initialize_rag()
    response = qa_chain.invoke({"question": data['message']})
    
    return jsonify({
        "response": response['answer'],
        #"sources": response.get('source_documents', [])
    })

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('DB', exist_ok=True)
    app.run(debug=True, host='127.0.0.1', port=5000)

