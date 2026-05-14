import os
from pypdf import PdfReader

#load all content from pdf file into a document variable
def load_pdfs(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents.append({"filename":filename, "text":text})
            print(f"Loaded: {filename}")
    return documents

docs_folder = r"C:/Users/Asus/OneDrive/Desktop/rag-project/docs"
documents = load_pdfs(docs_folder)
print(f"\nTotal documents loaded: {len(documents)}")

#--------------------------------------------------------------------

#text chunking
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0 #first text character
    while start < len(text):
        end = start + chunk_size # 0+500 = 500 characters || 450+500 = 950 characters etc.
        chunks.append(text[start:end])
        start = start + (chunk_size - overlap)  #0+500-50 = 450 characters
    return chunks

for doc in documents:
    chunks = chunk_text(doc["text"])
    print(f"{doc['filename']}: {len(chunks)} chunks")

#--------------------------------------------------------------------

#now, embedding text chunks
import ollama

def embed_chunks(chunks):
    response = ollama.embed(
        model='nomic-embed-text', 
        input=chunks
        )
    
    return response['embeddings']

test_embeddings = embed_chunks(chunks[:3])
print(f"\nNumber of embeddings: {len(test_embeddings)}")
print(f"Embedding size: {len(test_embeddings[0])}")
    