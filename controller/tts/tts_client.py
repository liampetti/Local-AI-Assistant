"""
Text-to-speech client module for the voice assistant.

This module handles text-to-speech synthesis using Piper and
audio streaming for playback.
"""

import asyncio
import re
from typing import Optional
import logging
import numpy as np

from wyoming.client import AsyncClient
from wyoming.event import Event
from wyoming.tts import SynthesizeStop
from config import config
from audio.audio_processor import AudioProcessor


class TTSClient:
    """Handles text-to-speech synthesis."""
    
    def __init__(self):
        self.logger = config.get_logger("TTSClient")
        self.audio_config = config.audio
        self.audio_processor = AudioProcessor()
    
    async def send_text_to_piper(
        self,
        text: str,
        audio_manager,
        break_callback: Optional[callable] = None
    ) -> None:
        """
        Send text to Piper for synthesis and stream to audio output.
        
        Args:
            text: Text to synthesize
            audio_manager: Audio manager for playback
            break_callback: Callback to check for breaks
        """
        # Remove special characters from response
        text = re.sub(r"[^a-zA-Z0-9\s.,!?'\"]+", '', text)
        self.logger.debug(f"Sending text to Piper for synthesis: {text!r}")
        
        try:
            audio_manager.out_stream.start()  # Begin audio output stream
            
            async with AsyncClient.from_uri(config.service.piper_uri) as client:
                await client.write_event(Event("synthesize", data={"text": text}))
                
                while True:
                    # Check for wakeword interruption
                    if break_callback and break_callback():
                        await client.write_event(SynthesizeStop().event())
                        break
                        
                    event = await client.read_event()
                    if event is None:
                        self.logger.warning("Piper client connection closed.")
                        break
                        
                    # self.logger.debug(f"Received event from piper: {event.type}")
                    
                    if event.type == "audio-chunk":
                        # Convert bytes to np.int16 array
                        audio_array = np.frombuffer(event.payload, dtype=np.int16)
                        sampled_chunk = self.audio_processor.resample_audio(
                            audio_array,
                            src_sr=config.audio.piper_sample_rate,
                            target_sr=config.audio.output_sample_rate,
                            channels=1
                        )
                        audio_manager.add_to_playback_buffer(sampled_chunk)
                        
                    if event.type == "audio-stop":
                        break
                        
        except Exception as e:
            self.logger.exception(f"Error in send_text_to_piper: {e}")