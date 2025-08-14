import os

# Ensure script uses CPU for embeddings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import re
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from utils.intent_catch import catchAll

DB_NAME = "assist_db"

# Connect to persistent ChromaDB
settings = Settings(anonymized_telemetry=False)
chroma_client = chromadb.PersistentClient(
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assist_db'),
    settings=settings)

# Setup collections
collections = {"chat": chroma_client.get_collection("family_facts"), 
               "intent": chroma_client.get_collection("intent_info")}

# Get location info
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'location_info.json'), "r") as f:
        loc_data = json.load(f)
location = loc_data['location']
address = loc_data['address']

# Get intent info
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'intent_info.json'), "r") as f:
        intent_data = json.load(f)

# Default parameters
SCORE_CUTOFF = 1.3
N_RESULTS = 3

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_facts(user_message, top_k, type):
    query_embedding = model.encode([user_message]).tolist()[0]
    results = []
    results.append(collections[type].query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
        ))
    return results

def augmentUserMessage(user_message, n_results=N_RESULTS, score_cutoff=SCORE_CUTOFF, type="chat"):
    if type=="intent":
        caught = catchAll(user_message)
        if caught is not None:
            return caught
    results = retrieve_facts(user_message, top_k=n_results, type=type)
    retrieved_facts = []
    retrieved_metas = []
    for result in results:
        res_score = result.get("distances", [[]])[0]
        res_des = result.get("documents", [[]])[0]
        res_meta = result.get("metadatas", [[]])[0]
        for i, score in enumerate(res_score): 
            print(f"{score} --> {res_des[i]}")
            if score < score_cutoff:
                retrieved_facts = [res_des[i]]+retrieved_facts
                retrieved_metas = [res_meta[i]]+retrieved_metas
        today = datetime.now().strftime("%B %d, %Y")
    if type=="chat":
        augmented = f"""
You are a helpful home assistant.
You have access to the following retrieved context from our knowledge base:

{'. '.join(retrieved_facts)}

Today is {today}.
Your location is {location}.
Your address is {address}.

Use the provided information only if it is relevant to the user's question. Otherwise answer using your current knowledge and do not reference the provided context.
Now answer the user's question accurately and succinctly, keeping your response to under three sentences.

User question: 
{user_message}
        """
    else:
        examples = ""
        for i, meta in enumerate(retrieved_metas):
            for entry in intent_data:
                if meta['metadata'] == entry['metadata']:
                    examples += f"{retrieved_facts[i]} Examples: {entry['examples']}. "
        # augmented = f"{examples} User: {user_message}" # show examples in chat or just use system prompt
        augmented = f"User: {user_message}"
    return augmented

if __name__ == "__main__":
    text = "whats dads age"
    print(text)
    results = retrieve_facts(text, type="chat")
    for result in results:
        for doc, meta, score in zip(
            result["documents"][0], 
            result["metadatas"][0], 
            min(result["distances"])
        ):
            print(f"Fact: {doc} | Meta: {meta.get('metadata', 'Unknown')} | Distance: {score:.4f}")
    print(augmentUserMessage(text, type="chat"))


    text = "whats the weather forecast"
    print(text)
    results = retrieve_facts(text, type="intent")
    for result in results:
        for doc, meta, score in zip(
            result["documents"][0], 
            result["metadatas"][0], 
            min(result["distances"])
        ):
            print(f"Fact: {doc} | Meta: {meta.get('metadata', 'Unknown')} | Distance: {score:.4f}")
    print(augmentUserMessage(text, type="intent"))