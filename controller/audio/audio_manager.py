"""
Audio management module for the voice assistant.

This module handles audio input/output streams, buffering, and audio processing
operations in a thread-safe manner.
"""

import sounddevice as sd
import numpy as np
import threading
import asyncio
from queue import Queue, Empty
from typing import List, Optional, Callable
import logging
from collections import deque
import os
import sys
import time

# Set both input and output explicitly
# sd.default.device = 12

# Add parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

try:
    from config import config
except ImportError:
    # Fall back to absolute import if relative fails
    from controller.config import config


class MaxLenQueue(Queue):
    """Thread-safe extendable queue with maximum length and iteration support."""
    
    def __init__(self, maxlen: int = 0):
        super().__init__(maxlen)
        self.buffer_size = maxlen
    
    def extend(self, iterable) -> None:
        """Add multiple items to the queue, removing oldest if full."""
        for item in iterable:
            # If queue is full, remove items from the front
            while self.full():
                try:
                    self.get_nowait()
                except Exception:
                    pass
            self.put(item)

    def __iter__(self):
        """Iterate over queue items."""
        with self.mutex:
            return iter(list(self.queue))
        
    def __len__(self) -> int:
        """Get current queue size."""
        return self.qsize()
    
    def get_fill_level(self) -> float:
        """Get buffer fill level as percentage."""
        with self.mutex:
            if self.buffer_size:
                return (self.qsize() / self.buffer_size) * 100
            return 0.0


class AudioManager:
    """Manages audio input/output streams and processing."""
    
    def __init__(self):
        self.logger = config.get_logger("AudioManager")
        self.config = config.audio
        
        # Initialize audio streams
        self.mic_stream = None
        self.out_stream = None
        self.playback_buffer = MaxLenQueue()
        
        # Audio buffers
        self.mic_buffer = MaxLenQueue(maxlen=config.buffer_size)
        # self.mic_buffer = deque(maxlen=config.buffer_size)
        self.sample_counter = 0
        
        # Control flags
        self.break_flag = False
        self._running = False
        
        # Threads
        self._playback_thread = None
        self._buffer_thread = None
        
    def start(self) -> None:
        """Initialize and start audio streams."""
        try:
            self._running = True

            # Initialize microphone stream
            self.mic_stream = sd.InputStream(
                samplerate=self.config.input_sample_rate,
                channels=self.config.channels,
                blocksize=self.config.chunk_size,
                dtype='int16'
            )
            self.mic_stream.start()
            
            # Initialize output stream
            self.out_stream = sd.OutputStream(
                samplerate=self.config.output_sample_rate,
                channels=self.config.channels,
                blocksize=self.config.chunk_size,  # Match chunk size
                dtype='int16',
                latency='low',  # Request low latency
                prime_output_buffers_using_stream_callback=True  # Help prevent initial stuttering
            )
            
            # Start worker threads
            self._start_playback_worker()
            self._start_buffer_worker()
            
            self.logger.info("Audio manager started successfully")
        except KeyboardInterrupt:
            # Graceful: context manager will stop/close stream
            pass     
        except Exception as e:
            self.logger.error(f"Failed to start audio manager: {e}")
            raise
    
    def stop(self) -> None:
        """Stop audio streams and cleanup."""
        self._running = False
        self.break_flag = True
        
        if self.mic_stream and self.mic_stream.active:
            self.mic_stream.stop()
            self.mic_stream.close()
        
        if self.out_stream and self.out_stream.active:
            self.out_stream.stop()
            self.out_stream.close()
        
        self.logger.info("Audio manager stopped")
    
    def _start_playback_worker(self) -> None:
        """Start the audio playback worker thread."""
        self._playback_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True
        )
        self._playback_thread.start()
    
    def _start_buffer_worker(self) -> None:
        """Start the microphone buffer worker thread."""
        self._buffer_thread = threading.Thread(
            target=self._buffer_worker,
            daemon=True
        )
        self._buffer_thread.start()
    
    def _playback_worker(self) -> None:
        """Worker thread for audio playback."""
        buffer_threshold = self.config.chunk_size * 3  # Maintain 3 chunks minimum
        prev_chunk = None
        
        while self._running:
            try:
                # Pre-buffer audio if buffer is too low
                if self.playback_buffer.qsize() < buffer_threshold:
                    while self.playback_buffer.qsize() < buffer_threshold * 2 and self._running:
                        time.sleep(0.01)  # Small sleep to prevent CPU spinning
                
                # Get audio chunk with timeout
                try:
                    out_chunk = self.playback_buffer.get(timeout=0.1)
                    prev_chunk = out_chunk
                except Empty:
                    # Graceful underrun handling
                    if prev_chunk is not None:
                        out_chunk = prev_chunk  # Repeat last chunk
                    else:
                        out_chunk = np.zeros(self.config.chunk_size, dtype=np.int16)
                    self.logger.debug("Playback buffer underrun detected")
                
                if self.out_stream and self.out_stream.active:
                    self.out_stream.write(out_chunk)
                    
            except Exception as e:
                self.logger.error(f"Playback error: {e}")
                time.sleep(0.1)  # Prevent tight error loops
    
    def _buffer_worker(self) -> None:
        """Worker thread for microphone buffering."""
        while self._running:
            try:
                frames, _ = self.mic_stream.read(self.config.chunk_size)
                mic_chunk = frames[:, 0] if self.config.channels == 1 else frames
                mic_chunk = mic_chunk.astype(np.int16)  # Ensure correct dtype
                self.mic_buffer.extend(mic_chunk)
                self.sample_counter += len(mic_chunk)
            except Exception as e:
                self.logger.error(f"Error in buffer worker: {e}")
                break
    
    def clear_buffers(self) -> None:
        """Clear all audio buffers."""
        self._clear_queue(self.playback_buffer)
        self._clear_queue(self.mic_buffer)
        self.sample_counter = 0
    
    def _clear_queue(self, queue_to_clear: Queue) -> None:
        """Clear a queue in a thread-safe manner."""
        with queue_to_clear.mutex:
            queue_to_clear.queue.clear()
            queue_to_clear.all_tasks_done.notify_all()
            queue_to_clear.unfinished_tasks = 0
    
    def get_mic_buffer_snapshot(self) -> List[np.ndarray]:
        """Get a snapshot of the microphone buffer."""
        return list(self.mic_buffer)
    
    def get_buffer_info(self) -> tuple:
        """Get current buffer information."""
        buffer_copy = list(self.mic_buffer)
        current_counter = self.sample_counter
        buffer_start_counter = current_counter - len(buffer_copy)
        relative_start_index = max(0, 0 - buffer_start_counter)
        return buffer_copy, current_counter, relative_start_index
    
    def add_to_playback_buffer(self, audio_chunk: np.ndarray) -> None:
        """Add audio chunk to playback buffer."""
        self.playback_buffer.extend(audio_chunk)
    
    def set_break_flag(self, value: bool = True) -> None:
        """Set the break flag for interrupting operations."""
        self.break_flag = value
    
    def is_break_requested(self) -> bool:
        """Check if a break has been requested."""
        return self.break_flag
    
    @property
    def mic_sample_counter(self):
        return self.sample_counter
    
    def get_playback_health(self) -> dict:
        """Get playback buffer health metrics."""
        return {
            'buffer_size': self.playback_buffer.qsize(),
            'fill_level': self.playback_buffer.get_fill_level(),
            'is_active': self.out_stream and self.out_stream.active
        }