"""
Configuration module for the voice assistant.

This module contains all configuration constants and settings used throughout
the application, organized in a centralized location for easy maintenance.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any
import os


@dataclass
class AudioConfig:
    """Audio configuration settings."""
    # Basic audio settings
    chunk_size: int = 1024  # Balanced size for processing efficiency
    input_sample_rate: int = 16000
    piper_sample_rate: int = 22050
    output_sample_rate: int = 22050
    channels: int = 1
    sample_width: int = 2
    
    # Buffer settings
    buffer_seconds: int = 15
    min_buffer_chunks: int = 3  # Minimum chunks to maintain
    target_buffer_chunks: int = 6  # Target buffer size in chunks
    max_buffer_chunks: int = 12  # Maximum chunks before dropping
    
    # Playback settings
    playback_buffer_size: int = 32768  # 32KB buffer for smooth playback
    prebuffer_size: int = 4096  # Amount to prebuffer before starting playback
    
    # Latency settings
    target_latency: str = 'low'  # Options: 'low', 'high', 'medium'
    underrun_threshold: float = 0.8  # Buffer level that triggers prebuffering

    # Echo cancellation settings
    filter_length: int = 4096  # Adjust based on room characteristics
    echo_buffer: int = 2 # Reference signal buffer for echo cancellation, 2 seconds buffer
    enable_echo_cancel: bool = True # Echo cancellation state
    
    def __post_init__(self):
        """Calculate derived values after initialization."""
        self.bytes_per_chunk = self.chunk_size * self.channels * self.sample_width
        self.chunks_per_second = self.output_sample_rate // self.chunk_size
        self.min_buffer_size = self.min_buffer_chunks * self.chunk_size
        self.target_buffer_size = self.target_buffer_chunks * self.chunk_size


@dataclass
class ServiceConfig:
    """External service configuration."""
    wakeword_uri: str = "tcp://localhost:10400"
    whisper_uri: str = "tcp://localhost:10300"
    piper_uri: str = "tcp://localhost:10200"
    ollama_intent_url: str = "http://localhost:11434/api/generate"
    ollama_chat_url: str = "http://localhost:11435/api/chat"


@dataclass
class ModelConfig:
    """AI model configuration."""
    intent_model: str = "qwen3:0.6b"
    chat_model: str = "qwen3:4b"
    chat_think: bool = False


@dataclass
class SilenceConfig:
    """Silence detection configuration."""
    silence_seconds: float = 1.5
    target_sample_rate: int = 16000
    threshold: float = 0.8
    chunk_size: int = 512


class Config:
    """Main configuration class that aggregates all settings."""
    
    def __init__(self):
        self.audio = AudioConfig()
        self.service = ServiceConfig()
        self.model = ModelConfig()
        self.silence = SilenceConfig()
        
        # Calculate derived values
        self.bytes_per_second = (
            self.audio.input_sample_rate * 
            self.audio.channels * 
            self.audio.sample_width
        )
        # Calculate buffer sizes based on audio settings
        self.buffer_size = (
            self.audio.buffer_seconds * 
            self.audio.output_sample_rate
        )
        
        # Playback buffer size (in samples)
        self.playback_buffer_size = (
            self.audio.playback_buffer_size // 
            (self.audio.channels * self.audio.sample_width)
        )
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        logging.getLogger('numba').setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance with the specified name."""
        return logging.getLogger(name)


# Global configuration instance
config = Config()