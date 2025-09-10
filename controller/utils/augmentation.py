import os    

# Ensure script uses CPU for embeddings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
from datetime import datetime
from utils.intent_catch import catchAll # Optional Regex Intent Catch
from utils.ai_knowledge import KnowledgeHandler
from config import config

# Get location info
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'location_info.json'), "r") as f:
        loc_data = json.load(f)
location = loc_data['location']
address = loc_data['address']

# Get knowledge graph
kg_handler = KnowledgeHandler()
kg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'family_kg.json')
G = kg_handler.load_graph_json(kg_file)

def augmentUserMessage(user_message, type="chat"):
    """
    Augments a users query with useful information and prompting for the LLM chat, otherwise just passes user query through for intent check
    """
    if type == "chat":
        retrieved_facts = kg_handler.handle_user_query(G, config.model.intent_model, config.service.ollama_intent_url, user_message)
        today = datetime.now().strftime("%B %d, %Y")
        augmented = f"""
You are a helpful home assistant.
You have access to the following retrieved facts from your knowledge base:

Knowledge Graph Facts:
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
