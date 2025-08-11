import wave

def bytesToWav(audio_bytes, filename, channels, sampwidth, framerate):
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(channels)      # Mono audio
        wav_file.setsampwidth(sampwidth)     # 2 bytes = 16 bits per sample
        wav_file.setframerate(framerate)     # Sample rate
        wav_file.writeframes(audio_bytes)