"""
Improved prompt system using the tool registry.

This module automatically generates prompts based on the available tools,
eliminating duplication and ensuring consistency.
"""

from typing import Dict, Any, List
from tools.tool_registry import tool_registry
from .intents import intent_handler
import logging


class PromptGenerator:
    """Automated prompt generator using the tool registry."""
    
    def __init__(self):
        self.logger = logging.getLogger("PromptGenerator")
    
    def generate_intent_prompt(self) -> str:
        """Generate intent detection prompt automatically from available tools."""
        function_descriptions = intent_handler.get_function_descriptions()
        
        prompt = f"""
Given a user's natural language query, generate a JSON response matching one of the following intents and argument patterns.  
IMPORTANT: Only output a JSON response if the user's query clearly matches one of the listed intents; otherwise, return an empty string ("").  
Never "guess" an intent for general information queries or when the user's question is out of scope of the listed intents.  
Do not provide explanations, apologies, or additional text. Output only valid JSON or an empty string.

Available Intents and their required arguments:
{function_descriptions}
"""\
"""

Instructions:
- Only select an intent if there is a clear, unambiguous match between the user's query and one of the available intents.
- If the user's query is not directly related to one of the above intents (such as general knowledge, facts, device control not listed, or unrelated questions), output an empty string: ""
- Do NOT match generic WH-questions (e.g., “what”, “why”, “how”, “when”) to "get_current_time" or any other intent unless the query explicitly asks for the current time or otherwise matches an intent exactly.
- Extract and place any required arguments in the order shown above.
- Output valid JSON only, or an empty string if not applicable. Never explain, comment, or add words outside the JSON or empty string.

Examples:

- User: "Whats the distance to the sun"
  Output:  
  ""

- User: "Whats the time"
  Output:
  {"intent": "get_current_time", "args": []}

- User: "Whats the weather tomorrow"
  Output:
  {"intent": "get_weather_forecast", "args": []}

- User: "Whats the weather forecast"
  Output:
  {"intent": "get_weather_forecast", "args": []}

- User: "Can you order pizza"
  Output:  
  ""
  
- User: "Turn on the lights"
  Output:
  {"intent": "turn_on_lights", "args": []}

- User: "Turn on the kitchen lights"
  Output:
  {"intent": "turn_on_lights", "args": ["kitchen"]}

- User: "Set brightness to 50 in the living room"
  Output:
  {"intent": "set_brightness", "args": ["50", "living room"]}

- User: "Play a song from the Beatles"
  Output:
  {"intent": "play_song", "args": ["the Beatles"]}

- User: "Play some music"
  Output:
  {"intent": "play_song", "args": []}

- User: "Who is president of the united states"
  Output:  
  ""

- User: "Set office temperature to 19 degrees Celcius."
  Output:
  {"intent": "set_temperature", "args": [19, "Office"]}

- User: "What is the temperature upstairs"
  Output:
  {"intent": "get_temperature", "args": ["Upstairs"]}
"""
        return prompt
    
    def generate_chat_prompt(self) -> str:
        """Generate chat prompt"""        
        prompt = f"""
You are a helpful, friendly, and engaging home assistant.

You can answer questions, chat, and help the family in a way that is friendly and appropriate for their ages. 
Be encouraging with the children, responsible and respectful with the parents, and remember the pet's presence when relevant (for example, offer fun pet facts or reminders).
Do not comment on any typos or errors in the query.

Always answer naturally and conversationally. If something is unsafe or not appropriate for children, gently defer or suggest asking a parent. 
Prioritize clarity, positivity, and practical help for all family members. Mention family members by name when suitable, and keep things fun and useful for the whole household.

Keep final answer length to three sentences or less, unless the user specifically asks for more detail.
"""
        return prompt


# Global prompt generator instance
prompt_generator = PromptGenerator()


def getIntentSystemPrompt():
    """Get the intent detection system prompt."""
    return prompt_generator.generate_intent_prompt()


def getChatSystemPrompt():
    """Get the chat system prompt with function calling."""
    return prompt_generator.generate_chat_prompt()



if __name__ == "__main__":
    print("Testing Prompt Generator")
    print("=" * 50)
    
    print("\nIntent Detection Prompt:")
    print("-" * 30)
    print(getIntentSystemPrompt())
    
    print("\nChat Prompt:")
    print("-" * 30)
    print(getChatSystemPrompt())
    