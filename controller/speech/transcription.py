"""
Transcription module for the voice assistant.

This module handles speech-to-text transcription using Whisper and
audio streaming to transcription services.
"""

import asyncio
import numpy as np
from typing import Optional, List
import logging

from wyoming.client import AsyncClient
from wyoming.event import Event
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from config import config
from audio.audio_processor import AudioProcessor


class TranscriptionManager:
    """Manages speech-to-text transcription."""
    
    def __init__(self):
        self.logger = config.get_logger("TranscriptionManager")
        self.audio_config = config.audio
        self.audio_processor = AudioProcessor()
        from audio.beep_manager import BeepManager
        self.beep_manager = BeepManager()
    
    async def stream_mic(
        self,
        client: AsyncClient,
        audio_manager,  # Pass the whole audio_manager, not just buffer
        transcribe: bool = False
    ) -> None:
        """
        Stream microphone audio to a client.
        
        Args:
            client: Wyoming client to stream to
            audio_manager: Audio manager instance
            transcribe: Whether this is for transcription (affects buffer size)
        """
        last_sent_counter = 0

        while True:
            buffer_copy = list(audio_manager.mic_buffer)  # snapshot to avoid race conditions
            current_counter = audio_manager.mic_sample_counter  # Use the global counter

            buffer_start_counter = current_counter - len(buffer_copy)
            relative_start_index = max(0, last_sent_counter - buffer_start_counter)
            new_data = buffer_copy[relative_start_index:]

            if new_data:
                if transcribe:
                    # Retrieve all new buffered for transcription
                    mic_bytes = np.array(new_data, dtype=np.int16).tobytes()
                else:
                    # Retrieve only last 2 seconds from buffer for wakeword detection
                    mic_bytes = np.array(
                        new_data[-2 * self.audio_config.input_sample_rate:],
                        dtype=np.int16
                    ).tobytes()
            else:
                await asyncio.sleep(
                    self.audio_config.chunk_size / self.audio_config.input_sample_rate
                )
                continue

            mic_event = AudioChunk(
                audio=mic_bytes,
                rate=self.audio_config.input_sample_rate,
                width=self.audio_config.sample_width,
                channels=self.audio_config.channels
            ).event()

            try:
                await client.write_event(mic_event)
            except Exception as e:
                self.logger.debug(f"Mic stream ended: {e}")
                break

            last_sent_counter = current_counter  # Update last sent position

            await asyncio.sleep(
                self.audio_config.chunk_size / self.audio_config.input_sample_rate
            )
    
    async def receive_events(
        self,
        client: AsyncClient,
        break_callback: Optional[callable] = None
    ) -> Optional[str]:
        """
        Receive events from a transcription client.
        
        Args:
            client: Wyoming client to receive from
            break_callback: Callback to check for breaks
            
        Returns:
            Transcription result or None
        """
        while True:
            event = await client.read_event()
            if event is None:
                break
                
            self.logger.info(f"Received: {event}")
            
            if event.type == "detection":
                self.logger.debug("Wake word detected!")
                self.beep_manager.play_beep()
                return "WAKE_DETECTED"
                
            if event.type == "error":
                self.logger.debug("Error from server:", event.data)
                break
                
            if event.type == "transcript":
                transcript = event.data.get("text", "")
                self.logger.debug(f"Transcription result: {transcript}")
                return transcript

            await asyncio.sleep(0)
        
        return None
    
    async def get_audio_stream_event(
        self,
        client: AsyncClient,
        audio_manager,
        silence_detector,
        transcribe: bool = False
    ) -> str:
        """
        Get audio stream event with optional silence detection.
        
        Args:
            client: Wyoming client
            audio_manager: Audio buffer and sample counter needed for streaming
            silence_detector: Silence detector instance
            transcribe: Whether this is for transcription
            
        Returns:
            Transcription result
        """
        tasks = [
            asyncio.create_task(self.stream_mic(client, audio_manager, transcribe)),
            asyncio.create_task(self.receive_events(client))
        ]
        
        if transcribe:
            # Prepend silence checker
            tasks = [
                asyncio.create_task(
                    silence_detector.listen_silence(
                        client,
                        audio_manager.mic_buffer,
                        sample_rate=self.audio_config.input_sample_rate
                    )
                )
            ] + tasks
            
            completed = []
            for coro in asyncio.as_completed(tasks):
                result = await coro
                completed.append(result)
                if len(completed) == 2:
                    result = "".join(completed)
                    break

            # Cancel the remaining task(s)
            for t in tasks:
                if not t.done():
                    t.cancel()
                    try:
                        await t  # Await to suppress warnings
                    except asyncio.CancelledError:
                        pass
        else:
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
            for d in done:
                result = await d
            # Cancel remaining tasks
            for p in pending:
                p.cancel()
        
        return result