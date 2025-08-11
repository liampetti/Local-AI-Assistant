"""
Speech package for the voice assistant.

This package contains modules for speech recognition, silence detection,
and audio transcription.
"""

from .silence_detector import SilenceDetector
from .transcription import TranscriptionManager

__all__ = ['SilenceDetector', 'TranscriptionManager'] 