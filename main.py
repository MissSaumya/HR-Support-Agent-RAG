import os
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- 1. SET UP THE LLM ---
# Point to our local server
# Note: "api_key" can be anything, it's not used by LM Studio
os.environ["OPENAI_API_KEY"] = "sk-11111111111111111111111111111111"
llm = OpenAI(base_url="http://localhost:1234/v1", temperature=0.7)

print("✅ LLM Connected")

# --- 2. LOAD OUR HR DATA ---
# Load documents from the 'data' directory
loader = DirectoryLoader('./data/', glob="**/*.txt") # Use PyPDFLoader for PDFs
documents = loader.load()

print(f"✅ Loaded {len(documents)} documents.")

# --- 3. CHUNK THE DOCUMENTS ---
# Split the documents into smaller chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

print(f"✅ Split into {len(texts)} chunks.")

# --- 4. CREATE EMBEDDINGS & VECTOR STORE ---
# Use a local, open-source model to create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a FAISS vector store to enable fast similarity searches
# This is our "knowledge base" index
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

print("✅ Vector Store Created")

# --- 5. CREATE THE RAG CHAIN ---
# LangChain provides a simple chain to handle the RAG process
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

print("✅ RAG Chain Created. The HR Agent is ready! 🚀")

# --- 6. INTERACTIVE CHAT LOOP ---
while True:
    query = input("\nAsk a question about HR policies (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    if query:
        try:
            # Run the query through the RAG chain
            answer = qa_chain.run(query)
            print("\nAnswer:")
            print(answer)
        except Exception as e:
            print(f"An error occurred: {e}")

print("\nGoodbye!")