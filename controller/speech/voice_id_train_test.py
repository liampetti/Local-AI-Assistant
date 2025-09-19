#!/usr/bin/env python3
"""
Train 4 speakers from WAV files and test recognition using Wyoming AudioChunk.

Adjust paths in TRAIN_FILES and TEST_FILES before running.
"""

import os
import glob
import wave
from typing import Dict, List
from voice_id_recognition import VoiceIDRecognizer, RecognizerConfig 
from wyoming.audio import AudioChunk 


# --------- helper: build a Wyoming AudioChunk from a WAV file ----------
def wav_to_audio_chunk(wav_path: str) -> AudioChunk:
    """
    Read a WAV file, extract raw PCM bytes and header, and wrap in a Wyoming AudioChunk.

    - rate: sample rate in Hz (int)
    - width: sample width in bytes (int) -> e.g., 2 for 16-bit PCM
    - channels: number of channels (int)
    - audio: raw PCM bytes payload
    """
    with wave.open(wav_path, "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        width = wf.getsampwidth()  # bytes per sample (e.g., 2 for 16-bit)
        nframes = wf.getnframes()
        pcm = wf.readframes(nframes)

    # Construct the Wyoming AudioChunk (mirrors audio-chunk event content)
    return AudioChunk(audio=pcm, rate=rate, width=width, channels=channels)


# --------- configure training and test sets ----------

def glob_wavs(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(files)

TRAIN_FILES: Dict[str, List[str]] = {
    "jane": glob_wavs(["data/train/jane/*.wav"]),
    "bob": glob_wavs(["data/train/bob/*.wav"]),
    "lucy": glob_wavs(["data/train/lucy/*.wav"])
}

TEST_FILES: Dict[str, List[str]] = {
    "unknown": glob_wavs(["data/test/unknown/*.wav"]),
    "bob": glob_wavs(["data/test/bob/*.wav"]),
    "jane": glob_wavs(["data/test/jane/*.wav"]),
    "lucy": glob_wavs(["data/test/lucy/*.wav"])
}


def main() -> None:
    # OPTIONAL: Custom configuration
    # cfg = RecognizerConfig(
    #     model_dir="data/voice_models",
    #     target_sr=16000,
    #     use_neural=True,          # set False to disable ECAPA embeddings
    #     confidence_threshold=0.4,
    #     margin_threshold=0.05,
    #     abs_match_threshold=0.3,
    #     weights_neural_mfcc=(0.3, 0.7),
    #     prior_temperature=2.0,
    #     conf_temperature=0.8,
    # )

    # recognizer = VoiceIDRecognizer(cfg)

    # Default
    recognizer = VoiceIDRecognizer()

    # ------- training -------
    print("Training speakers...")
    for speaker, paths in TRAIN_FILES.items():
        if not paths:
            print(f"  [SKIP] No training files for {speaker}")
            continue
        ok = recognizer.train_from_files(speaker, paths)
        print(f"  [{ 'OK' if ok else 'FAIL' }] {speaker}: {len(paths)} files")

    print("\nTrained speakers:", recognizer.list_speakers())

    # ------- testing with Wyoming AudioChunk -------
    print("\nTesting on held-out files using Wyoming AudioChunk:")
    total, correct = 0, 0
    for true_speaker, paths in TEST_FILES.items():
        for path in paths:
            total += 1
            try:
                chunk = wav_to_audio_chunk(path)  # build a protocol-accurate AudioChunk
                pred, conf = recognizer.recognize(chunk)
                match = ((pred or 'unknown') == true_speaker)
                if match:
                    correct += 1
                print(f"  file={os.path.basename(path)} true={true_speaker} pred={pred or 'unknown'} conf={conf:.3f} match={match}")
            except Exception as e:
                print(f"  [ERR] {path}: {e}")

    if total > 0:
        acc = 100.0 * correct / total
        print(f"\nAccuracy: {correct}/{total} = {acc:.2f}%")
    else:
        print("\nNo test files found.")


if __name__ == "__main__":
    main()
