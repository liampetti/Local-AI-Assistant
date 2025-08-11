"""
Distributed configuration for the voice assistant.

This configuration is for when Ollama services run on a separate computer
in the same local network.
"""

from config import Config, ServiceConfig

class DistributedConfig(Config):
    """Configuration for distributed setup with remote Ollama services."""
    
    def __init__(self, remote_host: str = "192.168.1.100"):
        """
        Initialize distributed configuration.
        
        Args:
            remote_host: IP address of the computer running Ollama services
        """
        super().__init__()
        
        # Override service configuration to point to remote Ollama services
        self.service = ServiceConfig(
            wakeword_uri="tcp://localhost:10400",
            whisper_uri="tcp://localhost:10300",
            piper_uri="tcp://localhost:10200",
            ollama_intent_url=f"http://{remote_host}:11434/api/generate",
            ollama_chat_url=f"http://{remote_host}:11435/api/chat"
        )

# Example usage:
# Replace the global config instance in config.py with:
# config = DistributedConfig(remote_host="192.168.1.100") 