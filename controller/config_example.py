"""
Example configuration file for the voice assistant.

This file shows how to customize the voice assistant configuration.
Copy this file to config.py and modify the values as needed.
"""

from config import Config, AudioConfig, ServiceConfig, ModelConfig, SilenceConfig

# Create a custom configuration
class CustomConfig(Config):
    """Custom configuration with modified settings."""
    
    def __init__(self):
        # Override audio settings for your hardware
        self.audio = AudioConfig(
            chunk_size=2048,  # Larger chunks for better performance
            input_sample_rate=44100,  # Higher quality audio
            output_sample_rate=44100,
            channels=2,  # Stereo audio
            buffer_seconds=20  # Longer buffer for complex queries
        )
        
        # Override service URLs if using different ports
        self.service = ServiceConfig(
            wakeword_uri="tcp://localhost:10400",
            whisper_uri="tcp://localhost:10300",
            piper_uri="tcp://localhost:10200",
            ollama_intent_url="http://localhost:11434/api/generate",
            ollama_chat_url="http://localhost:11435/api/chat"
        )
        
        # Override model settings
        self.model = ModelConfig(
            intent_model="qwen3:0.6b",
            chat_model="qwen3:1.7b",
            chat_think=True  # Enable thinking for reasoning models
        )
        
        # Override silence detection settings
        self.silence = SilenceConfig(
            silence_seconds=2.0,  # Longer silence detection
            threshold=0.7,  # More sensitive voice detection
            chunk_size=1024  # Larger chunks for VAD
        )
        
        # Calculate derived values
        self.bytes_per_second = (
            self.audio.input_sample_rate * 
            self.audio.channels * 
            self.audio.sample_width
        )
        self.buffer_size = self.audio.buffer_seconds * self.audio.input_sample_rate
        
        # Setup logging
        self._setup_logging()

# Example usage:
# Replace the global config instance in config.py with:
# config = CustomConfig()

"""
Configuration Tips:

1. Audio Settings:
   - chunk_size: Smaller values = lower latency, larger values = better performance
   - sample_rate: Higher rates = better quality, but more CPU usage
   - channels: 1 for mono, 2 for stereo
   - buffer_seconds: Longer buffers handle longer queries but use more memory

2. Service URLs:
   - Update these if your Wyoming servers run on different ports
   - Ensure Ollama is running on the specified URLs

3. Model Settings:
   - Use smaller models for faster responses
   - Enable chat_think for models that support reasoning
   - Adjust based on your hardware capabilities

4. Silence Detection:
   - silence_seconds: How long to wait before stopping transcription
   - threshold: Lower values = more sensitive to speech
   - chunk_size: Should match your VAD requirements

5. Performance Tuning:
   - For Raspberry Pi: Use smaller chunk sizes and lower sample rates
   - For powerful machines: Can use higher quality settings
   - Monitor CPU and memory usage to find optimal settings
""" 