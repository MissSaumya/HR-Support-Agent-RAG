from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)

# 1. LOAD HR DATA
loader = DirectoryLoader('./data/', glob="**/*.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 2. VECTOR STORE
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embeddings)
retriever = vector_db.as_retriever()

# 3. UPDATED SYSTEM PROMPT
llm = ChatOllama(model="qwen3:4b", temperature=0.2) 

template = """You are a professional Corporate HR Assistant. 
Use the following context to answer the user's question.

STRICT INSTRUCTIONS:
- If the answer is not contained within the context below, strictly respond with: "I'm sorry, I can only answer questions based on the uploaded company policies. That information is not currently in the system."
- Provide the answer in a clear, point-wise format using bullet points.
- Use bold text for key terms or policy names.
- Do not include document IDs or metadata in your final answer.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    query = request.json.get('query')
    try:
        response = rag_chain.invoke(query)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)