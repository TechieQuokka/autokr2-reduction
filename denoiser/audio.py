"""
Audio I/O utilities for noise reduction processing
"""

import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple


# DeepFilterNet works best with 48kHz audio
TARGET_SAMPLE_RATE = 48000


def load_audio(path: str, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load audio file and prepare for processing.

    Args:
        path: Path to audio file
        target_sr: Target sample rate (default: 48000 Hz)

    Returns:
        Tuple of (audio_data, sample_rate)
        - audio_data: numpy array, shape (samples,) for mono or (samples, channels) for stereo
        - sample_rate: sample rate of the audio

    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If audio loading fails
    """
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        # Load audio with soundfile
        audio, sr = sf.read(str(audio_path), dtype='float32')

        # Handle stereo to mono conversion if needed
        if audio.ndim == 2:
            # Convert stereo to mono by averaging channels
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != target_sr:
            audio = _resample(audio, sr, target_sr)
            sr = target_sr

        return audio, sr

    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {path}: {str(e)}")


def save_audio(path: str, audio: np.ndarray, sample_rate: int):
    """
    Save processed audio to file.

    Args:
        path: Output file path
        audio: Audio data (numpy array)
        sample_rate: Sample rate of the audio

    Raises:
        RuntimeError: If audio saving fails
    """
    try:
        # Ensure audio is in valid range [-1, 1]
        audio = np.clip(audio, -1.0, 1.0)

        # Save as WAV file
        sf.write(path, audio, sample_rate, subtype='PCM_16')

    except Exception as e:
        raise RuntimeError(f"Failed to save audio file {path}: {str(e)}")


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Simple resampling using linear interpolation.

    For production use, consider using librosa.resample or torchaudio.transforms.Resample
    for better quality resampling.

    Args:
        audio: Input audio
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    # Calculate the ratio and new length
    ratio = target_sr / orig_sr
    new_length = int(len(audio) * ratio)

    # Create new time axis
    old_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)

    # Linear interpolation
    resampled = np.interp(new_indices, old_indices, audio)

    return resampled.astype(np.float32)


def get_audio_info(path: str) -> dict:
    """
    Get audio file information without loading the entire file.

    Args:
        path: Path to audio file

    Returns:
        Dictionary with audio information (sample_rate, channels, duration, etc.)
    """
    info = sf.info(path)
    return {
        'sample_rate': info.samplerate,
        'channels': info.channels,
        'duration': info.duration,
        'frames': info.frames,
        'format': info.format,
        'subtype': info.subtype
    }
