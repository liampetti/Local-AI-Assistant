"""
Main application module for the voice assistant.

This module orchestrates all components of the voice assistant including
audio management, speech recognition, AI processing, and text-to-speech.
"""

import asyncio
import re
import logging
from typing import Optional

from wyoming.client import AsyncClient
from wyoming.audio import AudioStart

from config import config
from audio.audio_manager import AudioManager
from audio.system_volume import VolumeManager
from speech.silence_detector import SilenceDetector
from speech.transcription import TranscriptionManager
from ai.llm_client import LLMClient
from tts.tts_client import TTSClient
from utils.wakeclean import clean_transcript

from tools.spotify import pause, resume, is_playing


class VoiceAssistant:
    """Main voice assistant application class."""
    
    def __init__(self):
        self.logger = config.get_logger("VoiceAssistant")
        
        # Initialize components
        self.audio_manager = AudioManager()
        self.volume_manager = VolumeManager()
        self.silence_detector = SilenceDetector()
        self.transcription_manager = TranscriptionManager()
        self.llm_client = LLMClient()
        self.tts_client = TTSClient()
        
        # Allow for follow up questions
        self.follow_up = True
        
    async def start(self) -> None:
        """Start the voice assistant."""
        try:
            self.audio_manager.start()
            self.logger.info("Voice assistant started successfully.")
            
            # Main processing loop
            await self._main_loop()
            
        except Exception as e:
            self.logger.exception(f"Fatal error in voice assistant: {e}")
        finally:
            self.audio_manager.stop()
    
    async def _main_loop(self) -> None:
        """Main processing loop for the voice assistant."""
        while True:
            try:
                # Handle wakeword detection and transcription
                transcript = await self._handle_wakeword_and_transcription()
                
                if not transcript:
                    self.logger.warning("No transcription result, skipping cycle.")
                    continue

                # Create tasks for concurrency
                async with AsyncClient.from_uri(config.service.wakeword_uri) as wake_client:
                    # Wakeword interruption monitoring
                    task_wakeword = asyncio.create_task(
                        self.transcription_manager.get_audio_stream_event(
                            wake_client,
                            self.audio_manager,
                            self.silence_detector
                        )
                    )
                    
                    # Process transcript with AI
                    task_process_transcript = asyncio.create_task(self._send_text_to_ollama_with_tts(transcript))

                    # Wait for either task to complete first
                    done, pending = await asyncio.wait(
                        [task_wakeword, task_process_transcript],
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    # Cancel any remaining pending tasks (e.g., if wakeword happens first)
                    for task in pending:
                        task.cancel()

                    if task_wakeword in done:
                        # Reset audio manager if wakeword interrupt
                        self.audio_manager.stop()
                        self.audio_manager.clear_buffers()
                        self.audio_manager.start()
                        self.logger.debug("Voice assistant interrupted and restarted.")

            except Exception as e:
                self.logger.exception(f"Error in main loop: {e}")
                await asyncio.sleep(1)
    
    async def _handle_wakeword_and_transcription(self) -> Optional[str]:
        """Handle wakeword detection and audio transcription."""
        async with AsyncClient.from_uri(config.service.whisper_uri) as whisp_client:
            # Check if wakeword is needed or follow-up question
            wakeword = True
            if self.follow_up:
                await asyncio.sleep(3)  # Pause to check for follow-up
                wakeword = await self.silence_detector.listen_silence(
                    None,
                    self.audio_manager.mic_buffer,
                    sample_rate=config.audio.input_sample_rate
                )
            
            if wakeword:
                self.logger.info("Listening for wakeword events...")

                async with AsyncClient.from_uri(config.service.wakeword_uri) as wake_client:
                    # Waiting for Wakeword
                    await self.transcription_manager.get_audio_stream_event(
                        wake_client,
                        self.audio_manager,
                        self.silence_detector
                    )

            # Check Spotify 
            spotify_on = is_playing()
            if spotify_on:
                pause() # Pause any playing music

            # Short pause before buffer clear, allow beep sound to finish
            await asyncio.sleep(0.5) 

            # Clear audio buffers to remove wakeword and any previous speech
            self.audio_manager.clear_buffers()

            # Beginning Transcription
            self.audio_manager.set_break_flag(False)  # Reset break switch

            # Reduce audio output volume
            self.volume_manager.set_master_volume(50)

            # Short pause before transcription start
            await asyncio.sleep(0.5) 

            self.logger.debug("Whisper transcribing audio now...")
            # Send AudioStart for transcription, silence triggers stop detection
            await whisp_client.write_event(
                AudioStart(
                    rate=config.audio.input_sample_rate,
                    width=config.audio.sample_width,
                    channels=config.audio.channels
                ).event()
            )
            
            transcript = await self.transcription_manager.get_audio_stream_event(
                whisp_client,
                self.audio_manager,
                self.silence_detector,
                transcribe=True
            )

            # Clear audio buffers
            self.audio_manager.clear_buffers()

            # Clean up transcription
            transcript = clean_transcript(transcript)  # Remove wakeword
            transcript = re.sub(
                r"[^a-zA-Z0-9\s.,!?'\"]+",
                '',
                transcript
            )  # Remove special characters
            
            self.logger.debug(f"Cleaned up transcription: {transcript}")
            
            # Cleanly disconnect from Whisper so other tasks can take over
            try:
                await asyncio.wait_for(whisp_client.disconnect(), timeout=2)
                self.logger.debug("WHISPER disconnected successfully")
            except asyncio.TimeoutError:
                self.logger.warning("WHISPER disconnect timed out, forcing socket close")
                writer = getattr(whisp_client, "_writer", None)
                if writer is not None:
                    writer.close()
                    await writer.wait_closed()
                else:
                    self.logger.warning("Client writer is None — cannot force close socket")

            # Turn volume back up after transcribing
            self.volume_manager.set_master_volume(100)
            if spotify_on:
                resume() # Resume any previously playing music
               
            return transcript
    
    async def _send_text_to_ollama_with_tts(self, text: str) -> Optional[str]:
        """Send text to Ollama and handle TTS response."""
        try:
            tts_tasks = []

            # Check Spotify 
            spotify_on = is_playing()
            if spotify_on:
                pause() # Pause any playing music

            # Get response from LLM
            async for response_chunk in self.llm_client.send_text_to_ollama(
                text,
                buffer_out=True, 
                break_callback=self.audio_manager.is_break_requested
            ):
                if response_chunk:
                    task = asyncio.create_task(
                        self.tts_client.send_text_to_piper(
                            response_chunk,
                            self.audio_manager,
                            break_callback=self.audio_manager.is_break_requested
                        )
                    )
                    tts_tasks.append(task)

            # Ensure all TTS tasks are complete before returning
            if tts_tasks:
                await asyncio.gather(*tts_tasks)

            if spotify_on:
                resume() # Resume any previously playing music
            
            return "Response completed"
        except Exception as e:
            self.logger.exception(f"Error in send_text_to_ollama_with_tts: {e}")
            return None


def main():
    """Main entry point for the voice assistant."""
    try:
        assistant = VoiceAssistant()
        asyncio.run(assistant.start())
    except KeyboardInterrupt:
        logging.info("Voice assistant stopped by user.")
    except Exception as e:
        logging.exception(f"Fatal error in main: {e}")


if __name__ == "__main__":
    main()
