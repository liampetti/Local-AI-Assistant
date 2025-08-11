"""
Silence detection module for the voice assistant.

This module handles voice activity detection and silence detection for
controlling audio transcription sessions.
"""

import asyncio
import numpy as np
import time
import math
from typing import Optional
import logging

from silero_vad_lite import SileroVAD
from wyoming.audio import AudioStop
from config import config
from audio.audio_processor import AudioProcessor


class SilenceDetector:
    """Handles silence detection using Silero VAD."""
    
    def __init__(self):
        self.logger = config.get_logger("SilenceDetector")
        self.silence_config = config.silence
        
        # Initialize VAD
        self.vad = SileroVAD(self.silence_config.target_sample_rate)
        self.audio_processor = AudioProcessor()
    
    async def listen_silence(
        self,
        client,
        audio_buffer,
        sample_rate: int,
        silence_seconds: Optional[float] = None
    ) -> str:
        """
        Monitor audio buffer for silence and send stop event when detected.
        
        Args:
            client: Wyoming client for sending events
            audio_buffer: Audio buffer to monitor
            sample_rate: Sample rate of the audio
            silence_seconds: Duration of silence to detect (defaults to config)
            
        Returns:
            Empty string when silence is detected
        """
        if silence_seconds is None:
            silence_seconds = self.silence_config.silence_seconds
            
        silent_start = None
        
        while True:
            # Retrieve required audio from buffer for silence detection
            audio_float = (
                np.array(
                    list(audio_buffer)[-math.ceil(silence_seconds * sample_rate):],
                    dtype=np.int16
                ).astype(np.float32) / 32768.0
            )
            
            if sample_rate != self.silence_config.target_sample_rate:
                # Downsampling for SileroVAD
                audio_float = self.audio_processor.downsample_audio(
                    audio_float,
                    orig_rate=sample_rate,
                    target_rate=self.silence_config.target_sample_rate
                )

            # Set chunk size for SileroVAD and get list of chunks
            audio_chunked = self.audio_processor.chunk_audio_pad(
                audio_float,
                self.silence_config.chunk_size
            )

            # Get speech probabilities on each chunk
            results = []
            for chunk in audio_chunked:
                chunk = chunk.astype(np.float32)  # ensure dtype is correct
                result = self.vad.process(chunk)
                results.append(result)

            # Send event stop to client if silence, otherwise check for follow up question
            if client is not None:
                if max(results) < self.silence_config.threshold:
                    if silent_start is None:
                        silent_start = time.monotonic()
                    elif time.monotonic() - silent_start >= silence_seconds:
                        # Send AudioStop event to whisper client if silence
                        await client.write_event(AudioStop().event())
                        return ""
                else:
                    silent_start = None
            else:
                if max(results) < self.silence_config.threshold:
                    return True  # If silence, return True --> Wakeword is required
                else:
                    return False

            await asyncio.sleep(
                self.silence_config.chunk_size / self.silence_config.target_sample_rate
            ) 