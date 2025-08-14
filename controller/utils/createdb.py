'''
Creates a Vector Database of family and intent info provided in collection jsons
'''

import json
import chromadb
from sentence_transformers import SentenceTransformer

DB_NAME = "assist_db"
COLLECTIONS = ["family_facts", "intent_info"]

# Initialize ChromaDB in a local directory
chroma_client = chromadb.PersistentClient(path=f"./{DB_NAME}")

# Use a popular embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load facts from JSON file
for collection_name in COLLECTIONS:
    with open(f"{collection_name}.json", "r") as f:
        data = json.load(f)

    # Prepare embeddings and metadata
    texts = [item["text"] for item in data]
    metas = [item["metadata"] for item in data]
    ids = [f"{i:03d}" for i in range(1, len(metas)+1)]
    embeddings = model.encode(texts).tolist()

    # Create or get Chroma collection
    collection = chroma_client.get_or_create_collection(collection_name)

    # Add facts with embeddings
    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"metadata": m} for m in metas],
        ids=[f"id-{i}" for i in ids]
    )

    print(f"{collection_name} saved to ChromaDB.")
