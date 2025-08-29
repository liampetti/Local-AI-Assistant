"""

Audio management module for the voice assistant with Acoustic Echo Cancellation.

This module handles audio input/output streams, buffering, and audio processing

operations in a thread-safe manner, including real-time echo cancellation.

"""

import sounddevice as sd
import numpy as np
import threading
import asyncio
from queue import Queue, Empty
from typing import List, Optional, Callable, Tuple
import logging
from collections import deque
import os
import sys
import time
from scipy import signal
from scipy.signal import correlate, correlation_lags

# Set both input and output explicitly
# sd.default.device = 12
# Set input explicitly
# sd.default.device = ('hw:1,0', None)

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

class CircularBuffer:
    """Efficient circular buffer for real-time audio processing."""

    def __init__(self, size: int, dtype=np.float32):
        self.size = size
        self.buffer = np.zeros(size, dtype=dtype)
        self.write_pos = 0
        self.is_full = False
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        """Write data to the circular buffer."""
        with self._lock:
            data_len = len(data)

            if data_len >= self.size:
                # If data is larger than buffer, take the last 'size' samples
                self.buffer[:] = data[-self.size:]
                self.write_pos = 0
                self.is_full = True
            else:
                # Calculate how much space is available before wrapping
                space_to_end = self.size - self.write_pos

                if data_len <= space_to_end:
                    # Data fits without wrapping
                    self.buffer[self.write_pos:self.write_pos + data_len] = data
                else:
                    # Data needs to wrap around
                    self.buffer[self.write_pos:] = data[:space_to_end]
                    wrap_amount = data_len - space_to_end
                    self.buffer[:wrap_amount] = data[space_to_end:]

                self.write_pos = (self.write_pos + data_len) % self.size
                if not self.is_full and self.write_pos == 0:
                    self.is_full = True

    def read(self, length: int, delay: int = 0) -> np.ndarray:
        """Read data from the circular buffer with optional delay."""
        with self._lock:
            if not self.is_full and self.write_pos < length + delay:
                # Not enough data available
                return np.zeros(length, dtype=self.buffer.dtype)

            start_pos = (self.write_pos - length - delay) % self.size

            if start_pos + length <= self.size:
                # No wrap around needed
                return self.buffer[start_pos:start_pos + length].copy()
            else:
                # Wrap around needed
                first_part = self.buffer[start_pos:].copy()
                second_part = self.buffer[:length - len(first_part)].copy()
                return np.concatenate([first_part, second_part])

    def get_latest(self, length: int) -> np.ndarray:
        """Get the latest 'length' samples from the buffer."""
        return self.read(length, delay=0)

class NLMSFilter:
    """Normalized Least Mean Squares adaptive filter for echo cancellation."""

    def __init__(self, filter_length: int, mu: float = 0.5, regularization: float = 1e-6):
        """
        Initialize NLMS filter.

        Args:
            filter_length: Number of filter taps
            mu: Step size (0 < mu < 2), typically 0.1 to 1.0
            regularization: Small constant to prevent division by zero
        """
        self.filter_length = filter_length
        self.mu = mu
        self.regularization = regularization

        # Filter coefficients (weights)
        self.weights = np.zeros(filter_length, dtype=np.float32)

        # Input buffer for reference signal
        self.input_buffer = np.zeros(filter_length, dtype=np.float32)

        self._lock = threading.Lock()

    def update(self, reference_sample: float, desired_sample: float) -> Tuple[float, float]:
        """
        Update the adaptive filter with new samples.

        Args:
            reference_sample: Reference signal sample (far-end audio)
            desired_sample: Desired signal sample (microphone input)

        Returns:
            Tuple of (filtered_output, error_signal)
        """
        with self._lock:
            # Shift input buffer and add new reference sample
            self.input_buffer[1:] = self.input_buffer[:-1]
            self.input_buffer[0] = reference_sample

            # Calculate filter output (estimated echo)
            estimated_echo = np.dot(self.weights, self.input_buffer)

            # Calculate error (echo-cancelled signal)
            error = desired_sample - estimated_echo

            # Calculate input power for normalization
            input_power = np.sum(self.input_buffer ** 2) + self.regularization

            # Normalized step size
            normalized_mu = self.mu / input_power

            # Update filter weights using NLMS algorithm
            self.weights += normalized_mu * error * self.input_buffer

            return estimated_echo, error

    def process_block(self, reference_block: np.ndarray, desired_block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a block of samples.

        Args:
            reference_block: Block of reference signal samples
            desired_block: Block of desired signal samples

        Returns:
            Tuple of (estimated_echo_block, error_block)
        """
        estimated_echo = np.zeros_like(desired_block)
        error_block = np.zeros_like(desired_block)

        for i in range(len(reference_block)):
            estimated_echo[i], error_block[i] = self.update(reference_block[i], desired_block[i])

        return estimated_echo, error_block

    def reset(self):
        """Reset the filter to initial state."""
        with self._lock:
            self.weights.fill(0.0)
            self.input_buffer.fill(0.0)

class DelayEstimator:
    """Estimates delay between reference and microphone signals using cross-correlation."""

    def __init__(self, max_delay_samples: int = 8000, correlation_length: int = 4096):
        """
        Initialize delay estimator.

        Args:
            max_delay_samples: Maximum delay to search for in samples
            correlation_length: Length of signal blocks for correlation
        """
        self.max_delay_samples = max_delay_samples
        self.correlation_length = correlation_length
        self.estimated_delay = 0
        self.confidence = 0.0
        self._lock = threading.Lock()

    def estimate_delay(self, reference: np.ndarray, microphone: np.ndarray) -> int:
        """
        Estimate delay between reference and microphone signals.

        Args:
            reference: Reference signal block
            microphone: Microphone signal block

        Returns:
            Estimated delay in samples
        """
        with self._lock:
            if len(reference) < self.correlation_length or len(microphone) < self.correlation_length:
                return self.estimated_delay

            # Use the last correlation_length samples
            ref_block = reference[-self.correlation_length:]
            mic_block = microphone[-self.correlation_length:]

            # Normalize signals to prevent numerical issues
            ref_norm = ref_block / (np.std(ref_block) + 1e-8)
            mic_norm = mic_block / (np.std(mic_block) + 1e-8)

            # Compute cross-correlation
            correlation = correlate(mic_norm, ref_norm, mode='full')
            lags = correlation_lags(len(mic_norm), len(ref_norm))

            # Find delay within valid range
            valid_indices = np.where((lags >= 0) & (lags <= self.max_delay_samples))[0]

            if len(valid_indices) > 0:
                valid_corr = correlation[valid_indices]
                valid_lags = lags[valid_indices]

                # Find peak correlation
                peak_idx = np.argmax(np.abs(valid_corr))
                estimated_delay = valid_lags[peak_idx]

                # Calculate confidence based on peak prominence
                peak_value = np.abs(valid_corr[peak_idx])
                mean_value = np.mean(np.abs(valid_corr))
                self.confidence = peak_value / (mean_value + 1e-8)

                # Update estimated delay if confidence is high enough
                if self.confidence > 1.5:  # Threshold for accepting new delay estimate
                    self.estimated_delay = estimated_delay

            return self.estimated_delay

class DoubleTalkDetector:
    """Detects when both near-end and far-end speakers are active."""

    def __init__(self, threshold: float = 0.3, smoothing: float = 0.9):
        """
        Initialize double-talk detector.

        Args:
            threshold: Detection threshold for double-talk
            smoothing: Smoothing factor for energy estimates
        """
        self.threshold = threshold
        self.smoothing = smoothing

        self.far_end_energy = 0.0
        self.near_end_energy = 0.0
        self.echo_energy = 0.0

        self.noise_floor_far = 1e-6   # Start with a low value, will adapt
        self.noise_floor_near = 1e-6
        self.noise_floor_alpha = 0.001  # Adaptation speed (smaller = slower adaptation)
        self.silence_margin = 5.0      # Silence threshold relative to noise floor

        self._lock = threading.Lock()

    def detect(self, far_end_signal: np.ndarray, near_end_signal: np.ndarray,
           echo_estimate: np.ndarray) -> bool:
        """
        Adaptive double-talk detection, with silence thresholding based on learned noise floor.
        """
        with self._lock:
            far_energy = np.mean(far_end_signal ** 2)
            near_energy = np.mean(near_end_signal ** 2)
            echo_energy = np.mean(echo_estimate ** 2)

            # Smooth energy estimates
            self.far_end_energy = self.smoothing * self.far_end_energy + (1 - self.smoothing) * far_energy
            self.near_end_energy = self.smoothing * self.near_end_energy + (1 - self.smoothing) * near_energy
            self.echo_energy = self.smoothing * self.echo_energy + (1 - self.smoothing) * echo_energy

            # Update adaptive noise floor (minimum observed energy with slow adaptation)
            self.noise_floor_far = min(self.noise_floor_far, far_energy)
            self.noise_floor_far += self.noise_floor_alpha * (far_energy - self.noise_floor_far)
            self.noise_floor_near = min(self.noise_floor_near, near_energy)
            self.noise_floor_near += self.noise_floor_alpha * (near_energy - self.noise_floor_near)

            # Calculate adaptive silence thresholds
            silence_threshold_far = self.silence_margin * self.noise_floor_far
            silence_threshold_near = self.silence_margin * self.noise_floor_near

            # Silence gate: Only proceed if signals exceed adaptive noise floor
            if (self.far_end_energy < silence_threshold_far and
                self.near_end_energy < silence_threshold_near):
                return False  # No significant activity above environmental noise floor

            # Double-talk ratio logic
            if self.echo_energy > 1e-8:
                energy_ratio = self.near_end_energy / self.echo_energy
                return (energy_ratio > self.threshold)

            return False


class EchoCanceller:
    """Complete acoustic echo cancellation system."""

    def __init__(self, sample_rate: int = 16000, filter_length: int = 8192):
        """
        Initialize echo canceller.

        Args:
            sample_rate: Audio sample rate
            filter_length: Length of adaptive filter
        """
        self.sample_rate = sample_rate
        self.filter_length = filter_length

        # Components
        self.adaptive_filter = NLMSFilter(filter_length, mu=0.3)
        self.delay_estimator = DelayEstimator(max_delay_samples=sample_rate//2)  # 0.5 sec max delay
        self.double_talk_detector = DoubleTalkDetector()

        # Reference signal buffer (for delay compensation)
        self.reference_buffer = CircularBuffer(sample_rate * 2)  # 2 seconds of audio

        # State
        self.adaptation_enabled = True
        self.current_delay = 0

        self.logger = logging.getLogger("EchoCanceller")

    def process(self, reference_signal: np.ndarray, microphone_signal: np.ndarray) -> np.ndarray:
        """
        Process audio signals to cancel echo.

        Args:
            reference_signal: Far-end signal (audio being played)
            microphone_signal: Microphone input signal

        Returns:
            Echo-cancelled microphone signal
        """
        # Add reference signal to buffer
        self.reference_buffer.write(reference_signal)

        # Get delayed reference signal for adaptive filtering
        delayed_reference = self.reference_buffer.read(len(microphone_signal), self.current_delay)

        # Apply adaptive filter
        estimated_echo, echo_cancelled = self.adaptive_filter.process_block(
            delayed_reference, microphone_signal
        )

        # Detect double-talk
        double_talk = self.double_talk_detector.detect(
            reference_signal, microphone_signal, estimated_echo
        )

        # Enable/disable adaptation based on double-talk detection
        if double_talk:
            self.adaptation_enabled = False
            # self.logger.debug("Double-talk detected - adaptation disabled")
        else:
            self.adaptation_enabled = True

        # Periodically update delay estimate (every 100 blocks)
        if hasattr(self, '_delay_counter'):
            self._delay_counter += 1
        else:
            self._delay_counter = 0

        if self._delay_counter % 100 == 0 and not double_talk:
            # Update delay estimate using larger signal blocks
            ref_history = self.reference_buffer.get_latest(8192)
            if len(microphone_signal) >= 1024:
                new_delay = self.delay_estimator.estimate_delay(ref_history, microphone_signal)
                if abs(new_delay - self.current_delay) > 10:  # Only update if significant change
                    self.current_delay = new_delay
                    self.logger.debug(f"Updated delay estimate: {self.current_delay} samples")

        return echo_cancelled

class AudioManager:
    """Manages audio input/output streams and processing with echo cancellation."""

    def __init__(self):
        self.logger = config.get_logger("AudioManager")
        self.config = config.audio

        # Initialize audio streams
        self.mic_stream = None
        self.out_stream = None
        self.playback_buffer = MaxLenQueue()

        # Audio buffers
        self.mic_buffer = MaxLenQueue(maxlen=config.buffer_size)
        self.sample_counter = 0

        # Control flags
        self.break_flag = False
        self._running = False

        # Threads
        self._playback_thread = None
        self._buffer_thread = None

        # Echo cancellation
        self.echo_canceller = EchoCanceller(
            sample_rate=self.config.input_sample_rate,
            filter_length=self.config.filter_length
        )

        # Reference signal buffer for echo cancellation
        self.reference_signal_buffer = CircularBuffer(
            size=self.config.input_sample_rate * self.config.echo_buffer,  
            dtype=np.int16
        )

        # Echo cancellation state
        self.enable_echo_cancellation = self.config.enable_echo_cancel

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
                blocksize=self.config.chunk_size, # Match chunk size
                dtype='int16',
                latency='low', # Request low latency
                prime_output_buffers_using_stream_callback=True # Help prevent initial stuttering
            )

            # Start worker threads
            self._start_playback_worker()
            self._start_buffer_worker()

            self.logger.info("Audio manager with echo cancellation started successfully")

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
        """Worker thread for audio playback with reference signal capture."""
        buffer_threshold = self.config.chunk_size * 3 # Maintain 3 chunks minimum
        prev_chunk = None

        while self._running:
            try:
                # Pre-buffer audio if buffer is too low
                if self.playback_buffer.qsize() < buffer_threshold:
                    while self.playback_buffer.qsize() < buffer_threshold * 2 and self._running:
                        time.sleep(0.01) # Small sleep to prevent CPU spinning

                # Get audio chunk with timeout
                try:
                    out_chunk = self.playback_buffer.get(timeout=0.1)
                    prev_chunk = out_chunk
                except Empty:
                    # Graceful underrun handling
                    if prev_chunk is not None:
                        out_chunk = prev_chunk # Repeat last chunk
                    else:
                        out_chunk = np.zeros(self.config.chunk_size, dtype=np.int16)
                    self.logger.debug("Playback buffer underrun detected")

                if self.out_stream and self.out_stream.active:
                    self.out_stream.write(out_chunk)

                    # Capture reference signal for echo cancellation
                    if self.enable_echo_cancellation and out_chunk is not None:
                        self.reference_signal_buffer.write(out_chunk.flatten())

            except Exception as e:
                self.logger.error(f"Playback error: {e}")
                time.sleep(0.1) # Prevent tight error loops

    def _buffer_worker(self) -> None:
        """Worker thread for microphone buffering with echo cancellation."""
        while self._running:
            try:
                frames, _ = self.mic_stream.read(self.config.chunk_size)
                mic_chunk = frames[:, 0] if self.config.channels == 1 else frames
                mic_chunk = mic_chunk.astype(np.int16) # Ensure correct dtype

                # Apply echo cancellation if enabled
                if self.enable_echo_cancellation:
                    # Get reference signal for this chunk
                    reference_chunk = self.reference_signal_buffer.get_latest(len(mic_chunk))

                    # Convert to float for processing
                    mic_float = mic_chunk.astype(np.float32) / 32768.0
                    ref_float = reference_chunk.astype(np.float32) / 32768.0

                    # Apply echo cancellation
                    echo_cancelled = self.echo_canceller.process(ref_float, mic_float)

                    # Convert back to int16
                    mic_chunk = (echo_cancelled * 32768.0).astype(np.int16)

                    # Clip to prevent overflow
                    mic_chunk = np.clip(mic_chunk, -32768, 32767)

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

        # Reset echo cancellation
        if hasattr(self, 'echo_canceller'):
            self.echo_canceller.adaptive_filter.reset()

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

    def get_echo_cancellation_stats(self) -> dict:
        """Get echo cancellation performance statistics."""
        if hasattr(self, 'echo_canceller'):
            return {
                'enabled': self.enable_echo_cancellation,
                'current_delay': self.echo_canceller.current_delay,
                'adaptation_enabled': self.echo_canceller.adaptation_enabled,
                'delay_confidence': getattr(self.echo_canceller.delay_estimator, 'confidence', 0.0),
                'filter_length': self.echo_canceller.filter_length
            }
        return {'enabled': False}

    def set_echo_cancellation(self, enabled: bool) -> None:
        """Enable or disable echo cancellation."""
        self.enable_echo_cancellation = enabled
        if enabled:
            self.logger.info("Echo cancellation enabled")
        else:
            self.logger.info("Echo cancellation disabled")
