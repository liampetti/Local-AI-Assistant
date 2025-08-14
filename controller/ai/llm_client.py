"""
LLM client module for the voice assistant.

This module handles interactions with language models for intent detection
and chat responses.
"""

import asyncio
import json
import re
import requests
from typing import Dict, List, Optional, Any
from collections import deque
import logging

from config import config
from utils.system_prompts import getChatSystemPrompt, getIntentSystemPrompt
from utils.augmentation import augmentUserMessage
import utils.intents as intents


class LLMClient:
    """Handles interactions with language models."""
    
    def __init__(self):
        self.logger = config.get_logger("LLMClient")
        self.model_config = config.model
        
        # Chat history
        self.chat_history = deque(maxlen=6)
        
    async def send_text_to_ollama(
        self,
        text: str,
        buffer_out: bool = True,
        break_callback: Optional[callable] = None
    ):
        """
        Send text to Ollama for intent detection and chat response.
        
        Args:
            text: Input text to process
            buffer_out: Whether to buffer output for streaming
            break_callback: Callback function to check for breaks
            
        Yields:
            Response chunks from the language model
        """
        text = text.lower().strip()
        thinking = False
        
        payloads = [
            {
                "name": "intent",
                "payload": {
                    "model": self.model_config.intent_model,
                    "stream": True,
                    "think": True, # OPTIONAL: Turn off thinking to improve speed, results generally pretty bad on the smaller models
                    "system": getIntentSystemPrompt(),
                    "prompt": augmentUserMessage(text, type="intent")
                }
            },
            {
                "name": "chat",
                "payload": {
                    "model": self.model_config.chat_model,
                    "stream": True,
                    "system": getChatSystemPrompt()
                }
            }
        ]

        for load_map in payloads:
            name = load_map['name']
            payload = load_map['payload']
            
            self.logger.debug(f"Checking for {name} with Ollama: {text!r}")
            
            try:
                if name == "intent":
                    self.logger.debug(f"Ollama {name} Payload: {payload}")
                    if isinstance(payload['prompt'], dict):
                        self.logger.debug(f"Caught intent, loading --> {payload['prompt']}")
                        intent_response = intents.handle_intent(payload['prompt'])
                        yield intent_response
                        return
                    
                    response = requests.post(
                        config.service.ollama_intent_url,
                        json=payload,
                        stream=True
                    )
                else:
                    self.chat_history.extend([
                        {"role": "user", "content": augmentUserMessage(text, type="chat")}
                    ])
                    payload['messages'] = list(self.chat_history)
                    self.logger.debug(f"Ollama {name} Payload: {payload}")
                    response = requests.post(
                        config.service.ollama_chat_url,
                        json=payload,
                        stream=True
                    )

                full_response = ""
                textout_buffer = ""
                
                for line in response.iter_lines():
                    # Check for wakeword interruption
                    if break_callback and break_callback():
                        self.chat_history.extend([
                            {"role": "assistant", "content": "User cancelled request"}
                        ])
                        return
                    
                    if line:
                        data = json.loads(line.decode("utf-8"))
                        
                        if name == "intent":
                            full_response += data.get("response", "")
                        else:
                            token = data.get("message", {}).get("content")
                            full_response += token
                            
                            # Think check
                            thinking = (
                                bool(re.search(r'<think>', full_response)) and 
                                not re.search(r'</think>', full_response)
                            )
                            
                            if buffer_out:
                                textout_buffer += token
                                # Pause and send audio stream on comma, full stop except if number preceding or thinking
                                if ((token == ",") or (token == ".")) and not (textout_buffer[-2].isdigit()):
                                    if (thinking and self.model_config.chat_think) or not thinking:
                                        if not self.model_config.chat_think:
                                            textout_buffer = re.sub(
                                                r'^.*?</think>',
                                                '',
                                                textout_buffer,
                                                flags=re.DOTALL
                                            )
                                        # Return buffer for processing
                                        yield textout_buffer
                                    textout_buffer = ""

                # Clean, strip and remove thinking before checking intent or saving to chat history
                final_response = re.sub(
                    r'^.*?</think>',
                    '',
                    full_response,
                    flags=re.DOTALL
                )
                final_response = final_response.strip()

                if name == "intent":
                    matches = re.findall(r'\{[^}]*\}', final_response)
                    if len(matches) > 0:
                        for match in matches:
                            try:
                                self.logger.debug(f"Loading intent --> {match}")
                                intent_response = intents.handle_intent(match)
                                if isinstance(intent_response, asyncio.Task) or isinstance(intent_response, asyncio.Future):
                                    self.logger.debug(f"Awaiting response from intent --> {match}")
                                    # Wait for response if needed
                                    intent_response = await intent_response
                                yield intent_response
                            except Exception as e:
                                self.logger.debug(f"{match} intent not found")
                        return
                    else:
                        self.logger.debug("No intent given")
                else:
                    if buffer_out:
                        # Send the remaining buffer text
                        yield textout_buffer
                    else:
                        # Send final response if not buffering
                        yield final_response

                    self.chat_history.extend([
                        {"role": "assistant", "content": final_response}
                    ])
            except Exception as e:
                self.logger.exception(f"Error in send_text_to_ollama: {e}")
            
    
    def clear_chat_history(self) -> None:
        """Clear the chat history."""
        self.chat_history.clear()
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the current chat history."""
        return list(self.chat_history) 