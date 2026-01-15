# Interview Preparation Guide: HR Support Agent (RAG)

This document contains a comprehensive list of interview questions and answers tailored to your **HR Support Agent** project. It covers core concepts, technical implementation, and architectural decisions.

---

## 🚀 Project Overview (Quick Recap)
*   **Goal:** An AI-powered HR Assistant that answers policy questions using company-specific documents.
*   **Architecture:** Retrieval-Augmented Generation (RAG).
*   **Tech Stack:** 
    *   **Backend:** Python, Flask
    *   **LLM Orchestration:** LangChain (LCEL)
    *   **Model:** Qwen (4B) running locally via **Ollama**.
    *   **Vector DB:** **FAISS** (Facebook AI Similarity Search).
    *   **Embeddings:** HuggingFace `all-MiniLM-L6-v2`.
    *   **Text Splitting:** `RecursiveCharacterTextSplitter` (Chunk size: 500, Overlap: 50).

---

## 🧠 Section 1: General RAG Concepts

### 1. What is RAG and why did you use it instead of just fine-tuning the model?
**Answer:** RAG stands for **Retrieval-Augmented Generation**. It combines a retrieval system (finding relevant documents) with a generation system (LLM). 
*   **Why RAG?** 
    1.  **Up-to-date Info:** Fine-tuning is static. RAG allows you to update policies by just adding a text file.
    2.  **Reduced Hallucinations:** The model is forced to answer based on provided context rather than its training data.
    3.  **Citations/Transparency:** You can verify where the answer came from.
    4.  **Cost:** RAG is significantly cheaper and faster to implement than fine-tuning a model on company policies.

### 2. Explain the end-to-end flow of your HR Agent.
**Answer:** 
1.  **Data Ingestion:** HR documents (.txt) are loaded and split into small chunks.
2.  **Indexing:** Chunks are converted into numerical vectors (embeddings) and stored in **FAISS**.
3.  **Retrieval:** When a user asks a question, the question is also embedded, and FAISS finds the most similar chunks (semantic search).
4.  **Augmentation:** The retrieved chunks are added to a "System Prompt" alongside the user query.
5.  **Generation:** The LLM (Qwen) reads the prompt and context to generate a formatted HR response.

---

## 🛠️ Section 2: Retrieval & Vector Databases

### 3. Why did you choose FAISS as your vector store?
**Answer:** **FAISS** is a lightweight, high-performance library for similarity search. 
*   **Pros:** It’s extremely fast, runs locally (no cloud cost), and is perfect for mid-sized datasets like company policy manuals. 
*   **Alternatives:** I could have used ChromaDB or Pinecone (cloud), but FAISS is more efficient for this scale of local deployment.

### 4. What is the role of `all-MiniLM-L6-v2` in your project?
**Answer:** This is the **Embedding Model**. It converts human text into 384-dimensional vectors. 
*   **Why this model?** It strike a perfect balance between **speed** and **accuracy**. It is small (approx 80MB), runs quickly on CPU, and is highly effective at capturing semantic meaning for short-to-medium text chunks.

### 5. Why did you use `RecursiveCharacterTextSplitter` with a chunk size of 500 and 50 overlap?
**Answer:** 
*   **Recursive Splitting:** Unlike a simple split, it tries to keep paragraphs and sentences together by splitting on `\n\n`, `\n`, and then spaces. This preserves the **context** of the HR policy.
*   **Chunk Size (500):** Ensures the LLM gets enough detail but doesn't hit context window limits or get overwhelmed by irrelevant info.
*   **Overlap (50):** Prevents losing context that might be cut off at the edge of a chunk (e.g., a sentence split in half).

---

## 🤖 Section 3: LLM & Generation

### 6. How did you implement "Safety Filters" or prevent hallucinations?
**Answer:** I used **Prompt Engineering**. In my system prompt, I gave a **STRICT INSTRUCTION**: *"If the answer is not contained within the context below, strictly respond with: 'I'm sorry, I can only answer questions based on the uploaded company policies...'"* 
This constrains the LLM to only use the provided HR data and prevents it from making up general facts.

### 7. Why use Ollama to run the model locally?
**Answer:** 
1.  **Data Privacy:** HR policies are sensitive. Running the model locally ensures no data ever leaves the local environment.
2.  **Cost:** Free to run once the hardware is available.
3.  **Speed:** Low latency for inference compared to calling a cloud API over the internet.

### 8. What is LCEL (LangChain Expression Language)?
**Answer:** LCEL is a declarative way to compose chains. In my code, I used the pipe operator `|` to link the retriever, prompt, model, and output parser. 
*   **Benefits:** It handles async calls, streaming, and parallelization automatically, making the code much cleaner and production-ready.

---

## 💻 Section 4: Implementation & Flask

### 9. How does the Flask backend handle the request?
**Answer:** 
1.  The user sends a query via a POST request to `/ask`.
2.  The backend calls `rag_chain.invoke(query)`.
3.  The response is returned as JSON to the frontend.
4.  Error handling is implemented with a `try-except` block to catch LLM or Vector Store failures.

### 10. How would you scale this project for 10,000 documents?
**Answer:** 
1.  **Vector Store:** Migrate from FAISS (local file) to a managed vector database like **Weaviate** or **Milvus** for horizontal scaling.
2.  **Retrieval:** Implement **Hybrid Search** (Keyword + Semantic) to catch specific policy codes.
3.  **Caching:** Use Redis to cache common questions (e.g., "What is the leave policy?").
4.  **Re-ranking:** Use a Cross-Encoder to re-rank the top 10 retrieved chunks for better accuracy.

---

## 📈 Section 5: Evaluation & Performance

### 11. How do you evaluate if your RAG system is actually good?
**Answer:** I would use the **RAGAS** framework or the **RAG Triad**:
1.  **Context Relevance:** Is the retrieved context actually useful for the question?
2.  **Faithfulness:** Is the answer derived *only* from the context?
3.  **Answer Relevance:** Does the answer directly address the user's query?

### 12. What was the biggest challenge you faced?
**Answer:** (Sample Answer) *Fine-tuning the chunk size. Initially, I used a large chunk size, which led to the model getting confused by too much info. Reducing it to 500 with overlap improved the precision of the answers significantly.*

---

## 💡 Quick-Fire Tips for the Interview
*   **Be Clear on "Semantic Search":** Explain that it finds meaning, not just keywords (e.g., "vacation" will find policies about "leaves").
*   **Mention Hardware:** If asked, mention you ran it on your local GPU/CPU using Ollama.
*   **Focus on Business Value:** Emphasize that this saves HR departments hours of manual work answering repetitive questions.

---
**Created by Antigravity AI**
