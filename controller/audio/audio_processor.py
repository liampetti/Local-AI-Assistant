"""
Audio processing utilities for the voice assistant.

This module contains functions for audio resampling, conversion, and processing.
"""

import numpy as np
from typing import Union, List
import logging
import os
import sys

# Add parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

try:
    from config import config
except ImportError:
    # Fall back to absolute import if relative fails
    from controller.config import config


class AudioProcessor:
    """Handles audio processing operations."""
    
    def __init__(self):
        self.logger = config.get_logger("AudioProcessor")
    
    @staticmethod
    def resample_audio(
        audio_data: np.ndarray,
        src_sr: int,
        target_sr: int,
        channels: int = 1
    ) -> np.ndarray:
        """
        Resample audio data to a different sample rate.
        
        Args:
            audio_data: Input audio data as numpy array
            src_sr: Source sample rate
            target_sr: Target sample rate
            channels: Number of audio channels
            
        Returns:
            Resampled audio data
        """
        if src_sr == target_sr:
            return audio_data
        
        # Simple linear interpolation for resampling
        ratio = target_sr / src_sr
        new_length = int(len(audio_data) * ratio)
        
        # Create new time indices
        old_indices = np.arange(len(audio_data))
        new_indices = np.linspace(0, len(audio_data) - 1, new_length)
        
        # Interpolate
        resampled = np.interp(new_indices, old_indices, audio_data)

        # Ensure output is int16 and clipped to valid range
        resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
        return resampled
    
    @staticmethod
    def downsample_audio(
        audio_data: np.ndarray,
        orig_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """
        Downsample audio data to a lower sample rate.
        
        Args:
            audio_data: Input audio data
            orig_rate: Original sample rate
            target_rate: Target sample rate
            
        Returns:
            Downsampled audio data
        """
        if orig_rate <= target_rate:
            return audio_data
        
        # Calculate decimation factor
        decimation_factor = orig_rate // target_rate
        
        # Simple decimation (take every nth sample)
        downsampled = audio_data[::decimation_factor]
        
        return downsampled
    
    @staticmethod
    def bytes_to_wav(audio_bytes: bytes) -> np.ndarray:
        """
        Convert audio bytes to numpy array.
        
        Args:
            audio_bytes: Raw audio bytes
            
        Returns:
            Audio data as numpy array
        """
        return np.frombuffer(audio_bytes, dtype=np.int16)
    
    @staticmethod
    def wav_to_bytes(audio_data: np.ndarray) -> bytes:
        """
        Convert numpy array to audio bytes.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Raw audio bytes
        """
        return audio_data.tobytes()
    
    @staticmethod
    def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio data to prevent clipping.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Normalized audio data
        """
        if audio_data.dtype == np.int16:
            # Convert to float for processing
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)
        
        # Normalize
        max_val = np.max(np.abs(audio_float))
        if max_val > 0:
            audio_float = audio_float / max_val * 0.95
        
        # Convert back to int16
        return (audio_float * 32767).astype(np.int16)
    
    @staticmethod
    def chunk_audio_pad(
        audio_data: np.ndarray,
        chunk_size: int
    ) -> List[np.ndarray]:
        """
        Split audio data into chunks with padding.
        
        Args:
            audio_data: Input audio data
            chunk_size: Size of each chunk
            
        Returns:
            List of audio chunks
        """
        chunks = []
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            chunks.append(chunk)
        return chunks