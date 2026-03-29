# rag/vectorstore.py

import chromadb

# Use persistent storage so embeddings survive restarts.
client = chromadb.PersistentClient(path="db/chroma")
collection = client.get_or_create_collection(name="documents")


def get_collection():
    return collection

def add_to_db(chunks, embeddings, ids, metadatas=None):
    if metadatas is None:
        metadatas = [{} for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    # PersistentClient writes to disk automatically; no explicit persist() needed.


def query_db(query_embedding, k=3):
    col = get_collection()
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    return results

def clear_db():
    # Drop and recreate the collection to clear all entries
    global collection
    client.delete_collection(name="documents")
    collection = client.get_or_create_collection(name="documents")