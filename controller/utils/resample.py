from scipy.signal import resample_poly
import numpy as np
import resampy

def downsampleAudio(audio_data, orig_rate, target_rate):
    """Example: Downsample PCM audio from 44100 Hz to 16000 Hz."""
    gcd = np.gcd(orig_rate, target_rate)
    up = target_rate // gcd
    down = orig_rate // gcd
    return resample_poly(audio_data, up, down)

def resampleAudio(payload: bytes, src_sr=22050, target_sr=48000, channels=1):
    # Step 1: Convert bytes to int16 numpy array
    audio = np.frombuffer(payload, dtype=np.int16)

    # Step 2: Reshape for mono or stereo
    audio = audio.reshape(-1, channels)

    # Step 3: Convert to float32 for resampy
    audio = audio.astype(np.float32) / 32768.0  # normalize from int16

    # Step 4: Resample (axis=0)
    resampled = resampy.resample(audio.T, src_sr, target_sr).T

    # Step 5: Convert back to int16
    resampled_int16 = np.clip(resampled * 32768.0, -32768, 32767).astype(np.int16)

    return resampled_int16