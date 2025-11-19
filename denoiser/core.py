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

            # Initialize DeepFilterNet
            self.model, self.df_state, _ = init_df(
                model_base_dir=model_path,
                post_filter=True,
                device=self.device
            )

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
        strength: float = 0.7
    ) -> np.ndarray:
        """
        Apply DeepFilterNet3 noise reduction.

        Args:
            audio: Input audio (numpy array, mono)
            sample_rate: Sample rate of the audio
            strength: Noise reduction strength (0.0 to 1.0)
                     Higher values = more aggressive noise reduction

        Returns:
            Denoised audio (numpy array)

        Raises:
            ValueError: If strength is out of range
        """
        if not (0.0 <= strength <= 1.0):
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {strength}")

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # Add batch dimension

        # Apply DeepFilterNet enhancement
        with torch.no_grad():
            enhanced = self.enhance(
                self.model,
                self.df_state,
                audio_tensor,
                atten_lim_db=strength * 60  # Scale strength to attenuation limit
            )

        # Convert back to numpy
        denoised = enhanced.squeeze().cpu().numpy()

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
        Apply amplitude-based noise gate to suppress low-level noise.

        Args:
            audio: Input audio (numpy array)
            threshold_db: Gate threshold in dB (signals below this are suppressed)
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds
            sample_rate: Sample rate of the audio

        Returns:
            Gated audio (numpy array)
        """
        # Convert dB threshold to linear amplitude
        threshold_linear = 10 ** (threshold_db / 20)

        # Calculate attack and release coefficients
        attack_samples = int(attack_ms * sample_rate / 1000)
        release_samples = int(release_ms * sample_rate / 1000)

        # Compute envelope (absolute value with smoothing)
        envelope = np.abs(audio)

        # Apply gate
        gain = np.ones_like(audio)
        for i in range(len(audio)):
            if envelope[i] < threshold_linear:
                # Below threshold: reduce gain
                if i > 0:
                    # Smooth release
                    gain[i] = max(0.0, gain[i-1] - (1.0 / release_samples))
                else:
                    gain[i] = 0.0
            else:
                # Above threshold: increase gain
                if i > 0:
                    # Smooth attack
                    gain[i] = min(1.0, gain[i-1] + (1.0 / attack_samples))
                else:
                    gain[i] = 1.0

        # Apply gain to audio
        gated = audio * gain

        return gated

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
