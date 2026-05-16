# Local RAG Pipeline — A Beginner's Exploration

This project is a beginner-friendly exploration of **Retrieval-Augmented Generation (RAG)** — a technique that lets you chat with your own documents using a local LLM. Everything runs locally: no OpenAI, no cloud APIs, just your machine.

---

## What is RAG?

A standard LLM only knows what it was trained on. RAG solves a simple problem: *what if you want the model to answer questions about your own documents?*

The idea is straightforward:
1. Break your documents into chunks
2. Convert those chunks into vector embeddings and store them
3. When a user asks a question, find the most relevant chunks
4. Feed those chunks as context to the LLM and let it answer

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pypdf` | Extract text from PDF files |
| `ollama` | Run models locally (embeddings + chat) |
| `nomic-embed-text` | Embedding model (via Ollama) |
| `llama3.1:8b` | Chat/LLM model (via Ollama) |
| `chromadb` | Vector database to store and query embeddings |

---

## Pipeline Walkthrough

### Stage 1 — Load PDFs

```python
def load_pdfs(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents.append({"filename": filename, "text": text})
    return documents
```

All PDFs from the `docs/` folder are read page by page, and their full text is concatenated into a single string per document. Each document is stored as a dict with its filename and text.

---

### Stage 2 — Chunking

LLMs have a limited context window, and embedding a 50-page PDF as one block is impractical. So the text is split into overlapping chunks:

```python
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = start + (chunk_size - overlap)
    return chunks
```

- **`chunk_size=1000`** — each chunk is ~1000 characters
- **`overlap=200`** — consecutive chunks share 200 characters

The overlap is important. Without it, a sentence split across a chunk boundary would lose meaning. The overlap ensures context isn't lost at the edges.

```
Chunk 1: [0     →     1000]
Chunk 2:       [800   →   1800]
Chunk 3:             [1600 → 2600]
```

---

### Stage 3 — Embedding

Each chunk is converted into a vector (a list of numbers) that captures its semantic meaning. Similar chunks will have vectors that are close together in vector space.

```python
def embed_chunks(chunks):
    response = ollama.embed(
        model='nomic-embed-text',
        input=chunks
    )
    return response['embeddings']
```

`nomic-embed-text` is a lightweight, high-quality embedding model that runs entirely on your machine via Ollama.

---

### Stage 4 — Storing in ChromaDB

ChromaDB is a local vector database. The chunks, their embeddings, and metadata (which PDF they came from) are all stored here.

```python
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="research_papers")

if collection.count() == 0:
    for doc in documents:
        chunks = chunk_text(doc["text"])
        embeddings = embed_chunks(chunks)
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": doc["filename"]} for _ in chunks],
            ids=[str(uuid.uuid4()) for _ in chunks]
        )
```

The `if collection.count() == 0` check skips re-ingesting documents if the database already has data. The data is persisted to disk in the `chroma_db/` folder, so it survives between runs.

---

### Stage 5 — Querying

When the user asks a question, it goes through the same embedding model to produce a query vector. ChromaDB then finds the 5 most semantically similar chunks using vector similarity search.

```python
question = input("Ask a query: ")

question_embeddings = embed_chunks([question])[0]
results = collection.query(
    query_embeddings=[question_embeddings],
    n_results=5,
)

context = "\n\n".join(results["documents"][0])
```

The key insight: the question and the relevant chunks end up *close together in vector space*, even if they don't share exact keywords. This is what makes semantic search powerful compared to plain keyword search.

---

### Stage 6 — Generating the Answer

The retrieved chunks are assembled into a context string and passed to `llama3.1:8b` along with a system prompt that constrains the model to only use the provided context.

```python
system_instruction = """You are a helpful research assistant. 
Answer the user's question using ONLY the provided context.
Explain clearly and thoroughly, as if teaching a beginner.
Stay strictly focused on what the user asked — do not mix information from unrelated topics.
If the context doesn't contain enough information to answer, say so clearly."""

response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
    ],
)

print(response["message"]["content"])
```

The system prompt is important — it tells the model to stay grounded in the retrieved documents and not hallucinate answers from its training data.

---

## How to Run

**Prerequisites:** [Ollama](https://ollama.com) installed and running, with these models pulled:
```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

**Install dependencies:**
```bash
pip install pypdf ollama chromadb
```

**Add your PDFs** to the `docs/` folder, then run:
```bash
python rag.py
```

Type your question when prompted and the pipeline will retrieve relevant context from your documents and generate an answer.

---

## What I Learned

This project was built to understand the fundamentals of a RAG pipeline from scratch, without any high-level frameworks:

- How raw text gets transformed into vectors that encode *meaning*, not just keywords
- Why chunking with overlap matters for preserving context at boundaries
- How vector similarity search retrieves relevant content semantically
- How to ground an LLM's responses to a specific set of documents using a system prompt
- How to wire all these pieces together into a working end-to-end pipeline

It's intentionally minimal — no LangChain, no LlamaIndex — which makes it easier to see exactly what's happening at each stage. Great starting point before moving to more complex RAG architectures.
