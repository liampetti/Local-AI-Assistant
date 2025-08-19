import os    

# Ensure script uses CPU for embeddings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from utils.intent_catch import catchAll
from config import config

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
# TODO: Not currently used for intent check, just system prompt is used to avoid confusion and speed up request for now
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'intent_info.json'), "r") as f:
        intent_data = json.load(f)

# Default parameters
SCORE_CUTOFF = 1.3
N_RESULTS = 3

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_facts(user_message, top_k=3, type="chat"):
    query_embedding = model.encode([user_message]).tolist()[0]
    results = []
    results.append(collections[type].query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
        ))
    return results

def augmentUserMessage(user_message, n_results=3, score_cutoff=1.2, type="chat"):
    """
    Augments a users query with useful information and prompting for the LLM chat, otherwise just passes user query through for intent check
    """
    if type == "chat":
        results = retrieve_facts(user_message, top_k=n_results, type=type)
        retrieved_facts = []
        for result in results:
            res_score = result.get("distances", [[]])[0]
            res_des = result.get("documents", [[]])[0]
            for i, score in enumerate(res_score): 
                if score < score_cutoff:
                    retrieved_facts = [res_des[i]]+retrieved_facts        
        today = datetime.now().strftime("%B %d, %Y")
        augmented = f"""
You are a helpful home assistant.
You have access to the following retrieved facts from our knowledge base:

{chr(10).join(retrieved_facts)}

Today is {today}.
Your location is {location}.
Your address is {address}.

Use the provided information only if it is relevant to the user's question. Otherwise answer using your current knowledge and do not reference the provided context.
Now answer the user's question accurately and succinctly, keeping your response to under three sentences.

User question: 
{user_message}
        """
    else:
        # TODO: Review if relevant fact retrieval examples from the intent database improves the intent checking ability beyond what is already in the intent system prompt
        augmented = f"User: {user_message}"
    return augmented

if __name__ == "__main__":
    text = "whats dads age"
    print(text)
    print(augmentUserMessage(text, type="chat"))


    text = "whats the weather forecast"
    print(text)
    print(augmentUserMessage(text, type="intent"))