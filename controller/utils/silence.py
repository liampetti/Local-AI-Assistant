import asyncio
import numpy as np
import time
import math
from silero_vad_lite import SileroVAD
from wyoming.audio import AudioStop
from utils.resample import downsampleAudio

SILENCE_SECONDS = 1.5
TARGET_SAMPLE_RATE = 16000 # SileroVAD sample rate must be 16000 or 8000
THRESHOLD = 0.8 # Speech probability threshhold
CHUNK_SIZE = 512 # 32 ms @ 16 kHz, required by SileroVAD
vad = SileroVAD(TARGET_SAMPLE_RATE)

def chunk_audio_pad(audio_data, chunk_size=CHUNK_SIZE):
    chunks = []
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        chunks.append(chunk)
    return chunks

async def listenSilence(client, audio_buffer, sample_rate, silence_seconds=SILENCE_SECONDS):
    silent_start = None
    while True:
        # Retrieve only required from buffer for silence detection
        audio_float = (np.array(list(audio_buffer)[-math.ceil(silence_seconds*sample_rate):], dtype=np.int16).astype(np.float32) / 32768.0)
        if sample_rate != TARGET_SAMPLE_RATE:
            # Downsampling for SileroVAD
            audio_float = downsampleAudio(audio_float, orig_rate=sample_rate, target_rate=TARGET_SAMPLE_RATE)

        # Set chunk size for SileroVAD and get list of chunks
        audio_chunked = chunk_audio_pad(audio_float)

        # Get speech probabilities on each chunk
        results = []
        for chunk in audio_chunked:
            chunk = chunk.astype(np.float32)  # ensure dtype is correct
            result = vad.process(chunk)
            results.append(result)

        # Send event stop to client if silence, otherwise check for follow up question
        if client is not None:
            if max(results) < THRESHOLD:
                    if silent_start is None:
                        silent_start = time.monotonic()
                        # print(f"Threshold: {max(results)}, time: {time.monotonic() - silent_start}")
                    elif time.monotonic() - silent_start >= silence_seconds:
                        # Send AudioStop event to whisper client if silence
                        await client.write_event(AudioStop().event())
                        return ""
            else:
                silent_start = None
        else:
            if max(results) < THRESHOLD:
                return True # If silence, return True --> Wakeword is required
            else:
                return False

        await asyncio.sleep(CHUNK_SIZE / TARGET_SAMPLE_RATE)