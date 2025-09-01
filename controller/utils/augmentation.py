import os    

# Ensure script uses CPU for embeddings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from utils.intent_catch import catchAll # Optional Regex Intent Catch
from config import config

DB_NAME = "assist_db"

# Connect to persistent ChromaDB
settings = Settings(anonymized_telemetry=False)
chroma_client = chromadb.PersistentClient(
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assist_db'),
    settings=settings)

# Setup collection
collection = chroma_client.get_collection("family_facts")

# Get location info
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'location_info.json'), "r") as f:
        loc_data = json.load(f)
location = loc_data['location']
address = loc_data['address']

# Default parameters
SCORE_CUTOFF = 1.0
N_RESULTS = 3

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_facts(user_message, top_k=3):
    query_embedding = model.encode([user_message]).tolist()[0]
    results = []
    results.append(collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
        ))
    return results

def augmentUserMessage(user_message, n_results=N_RESULTS, score_cutoff=SCORE_CUTOFF, type="chat"):
    """
    Augments a users query with useful information and prompting for the LLM chat, otherwise just passes user query through for intent check
    """
    if type == "chat":
        results = retrieve_facts(user_message, top_k=n_results)
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
        # OPTIONAL: Run regex intent catch to capture intents before sending to LLM
        augmented = catchAll(user_message)
    return augmented
