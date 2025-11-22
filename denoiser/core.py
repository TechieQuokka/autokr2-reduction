"""
DeepFilterNet3 noise reduction core implementation
"""

import torch
import numpy as np
from typing import Optional
import warnings


class Denoiser:
    """
    DeepFilterNet3-based noise reduction with optional noise gate.

    This class wraps DeepFilterNet3 for audio denoising and provides
    additional post-processing capabilities like noise gating.
    """

    def __init__(self, device: str = 'auto', model_path: Optional[str] = None):
        """
        Initialize DeepFilterNet3 model.

        Args:
            device: Device to use ('auto', 'cpu', or 'cuda')
            model_path: Optional path to custom model checkpoint

        Raises:
            RuntimeError: If model loading fails
        """
        self.device = self._select_device(device)
        self.model = None
        self.df_state = None

        try:
            from df.enhance import enhance, init_df
            from df.io import resample

            self.enhance = enhance
            self.init_df = init_df
            self.resample = resample

            # Initialize DeepFilterNet (device is handled internally)
            self.model, self.df_state, _ = init_df(
                model_base_dir=model_path,
                post_filter=True
            )

            # Move model to selected device
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)

            print(f"✓ DeepFilterNet3 loaded on {self.device}")

        except ImportError as e:
            raise RuntimeError(
                "DeepFilterNet not installed. "
                "Install with: pip install deepfilternet"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load DeepFilterNet3: {str(e)}") from e

    def _select_device(self, device: str) -> str:
        """Select computation device (CPU or CUDA)."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available, falling back to CPU")
            return 'cpu'
        return device

    def denoise(
        self,
        audio: np.ndarray,
        sample_rate: int,
        strength: float = 0.7,
        chunk_size_sec: float = 10.0
    ) -> np.ndarray:
        """
        Apply DeepFilterNet3 noise reduction with chunk processing for memory efficiency.

        Args:
            audio: Input audio (numpy array, mono)
            sample_rate: Sample rate of the audio
            strength: Noise reduction strength (0.0 to 1.0)
                     Higher values = more aggressive noise reduction
            chunk_size_sec: Size of each processing chunk in seconds (default: 10.0)

        Returns:
            Denoised audio (numpy array)

        Raises:
            ValueError: If strength is out of range
        """
        if not (0.0 <= strength <= 1.0):
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {strength}")

        # Calculate chunk size in samples
        chunk_size = int(chunk_size_sec * sample_rate)
        total_samples = len(audio)

        # Process short audio in one go
        if total_samples <= chunk_size:
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            with torch.no_grad():
                # Conservative mode for already-clean audio
                # Reduce aggressiveness to preserve voice quality
                atten_lim = strength * 40  # Changed from 60 to 40 (less aggressive)
                enhanced = self.enhance(
                    self.model,
                    self.df_state,
                    audio_tensor,
                    atten_lim_db=atten_lim
                )
            return enhanced.squeeze().cpu().numpy()

        # Process long audio in chunks
        denoised = np.zeros_like(audio)
        num_chunks = (total_samples + chunk_size - 1) // chunk_size

        print(f"  Processing {num_chunks} chunks ({chunk_size_sec}s each)...")

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total_samples)
            chunk = audio[start:end]

            # Progress indicator
            progress = (i + 1) / num_chunks * 100
            print(f"  → Chunk {i+1}/{num_chunks} ({progress:.1f}%)", end='\r', flush=True)

            # Process chunk
            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
            with torch.no_grad():
                # Conservative mode for already-clean audio
                atten_lim = strength * 40  # Changed from 60 to 40 (less aggressive)
                enhanced_chunk = self.enhance(
                    self.model,
                    self.df_state,
                    chunk_tensor,
                    atten_lim_db=atten_lim
                )

            # Store result
            denoised[start:end] = enhanced_chunk.squeeze().cpu().numpy()

        print()  # New line after progress

        return denoised

    def apply_noise_gate(
        self,
        audio: np.ndarray,
        threshold_db: float = -40.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
        sample_rate: int = 48000
    ) -> np.ndarray:
        """
        Apply RMS-based noise gate with proper envelope following.

        Args:
            audio: Input audio (numpy array)
            threshold_db: Gate threshold in dB (signals below this are suppressed)
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds
            sample_rate: Sample rate of the audio

        Returns:
            Gated audio (numpy array)
        """
        # Convert to torch tensor and move to GPU
        audio_tensor = torch.from_numpy(audio).float().to(self.device)

        # Convert dB threshold to linear RMS amplitude
        threshold_linear = 10 ** (threshold_db / 20)

        # Calculate RMS envelope with proper window size (20ms for speech)
        rms_window_ms = 20.0
        rms_window_size = int(rms_window_ms * sample_rate / 1000)
        if rms_window_size % 2 == 0:
            rms_window_size += 1

        # Compute RMS envelope using efficient convolution
        audio_squared = audio_tensor ** 2
        rms_kernel = torch.ones(1, 1, rms_window_size, device=self.device) / rms_window_size

        # Pad for RMS calculation
        pad_size = rms_window_size // 2
        audio_sq_padded = torch.nn.functional.pad(
            audio_squared.unsqueeze(0).unsqueeze(0),
            (pad_size, pad_size),
            mode='replicate'
        )

        # Calculate RMS
        rms_squared = torch.nn.functional.conv1d(audio_sq_padded, rms_kernel, padding=0)
        rms = torch.sqrt(torch.clamp(rms_squared, min=1e-10)).squeeze()[:len(audio_tensor)]

        # Create gate signal based on RMS
        gate_open = (rms > threshold_linear).float()

        # Apply hysteresis to prevent gate fluttering
        # Once gate opens, it stays open longer (prevents choppy audio)
        hold_samples = int(50 * sample_rate / 1000)  # 50ms hold time
        if hold_samples > 1:
            hold_kernel = torch.ones(1, 1, hold_samples, device=self.device)
            gate_padded = torch.nn.functional.pad(
                gate_open.unsqueeze(0).unsqueeze(0),
                (hold_samples // 2, hold_samples // 2),
                mode='replicate'
            )
            gate_held = torch.nn.functional.conv1d(gate_padded, hold_kernel, padding=0)
            gate_open = (gate_held.squeeze()[:len(audio_tensor)] > 0).float()

        # Apply exponential smoothing for attack/release
        # This creates smooth transitions without loops
        attack_coef = 1.0 - torch.exp(torch.tensor(-2.2 / (attack_ms * sample_rate / 1000), device=self.device))
        release_coef = 1.0 - torch.exp(torch.tensor(-2.2 / (release_ms * sample_rate / 1000), device=self.device))

        # Use a simple IIR-like smoothing (vectorized approximation)
        # Create smoothed gain using cascaded moving averages (approximates exponential)
        smooth_window = max(3, int(release_ms * sample_rate / 1000))
        if smooth_window % 2 == 0:
            smooth_window += 1

        gain = gate_open
        # Apply smoothing passes (cascading approximates exponential)
        for _ in range(2):
            smooth_kernel = torch.ones(1, 1, smooth_window, device=self.device) / smooth_window
            gain_padded = torch.nn.functional.pad(
                gain.unsqueeze(0).unsqueeze(0),
                (smooth_window // 2, smooth_window // 2),
                mode='replicate'
            )
            gain = torch.nn.functional.conv1d(gain_padded, smooth_kernel, padding=0).squeeze()[:len(audio_tensor)]

        # Apply gain to audio
        gated = audio_tensor * gain

        # Convert back to numpy
        return gated.cpu().numpy()

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        strength: float = 0.7,
        apply_gate: bool = True,
        gate_threshold: float = -40.0,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Complete noise reduction pipeline.

        Args:
            audio: Input audio
            sample_rate: Sample rate
            strength: Noise reduction strength
            apply_gate: Whether to apply noise gate
            gate_threshold: Noise gate threshold in dB
            verbose: Print processing steps

        Returns:
            Processed audio
        """
        if verbose:
            print(f"→ Applying DeepFilterNet3 (strength={strength:.2f})...")

        # Step 1: DeepFilterNet noise reduction
        denoised = self.denoise(audio, sample_rate, strength)

        # Step 2: Optional noise gate
        if apply_gate:
            if verbose:
                print(f"→ Applying noise gate (threshold={gate_threshold:.1f} dB)...")
            denoised = self.apply_noise_gate(
                denoised,
                threshold_db=gate_threshold,
                sample_rate=sample_rate
            )

        if verbose:
            print("✓ Processing complete")

        return denoised
