#!/usr/bin/env python3
"""
DeepFilterNet3 Noise Reduction CLI

Usage:
    python denoise.py --input input.wav --config config.json --output output.wav
"""

import argparse
import sys
from pathlib import Path

from denoiser import NoiseReductionConfig, Denoiser, load_audio, save_audio


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DeepFilterNet3 Noise Reduction Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (use JSON settings)
  python denoise.py --input vocals.wav --config settings.json --output denoised.wav

  # Override strength
  python denoise.py --input vocals.wav --config settings.json --output denoised.wav --strength 0.8

  # Disable noise gate
  python denoise.py --input vocals.wav --config settings.json --output denoised.wav --no-gate

  # Use CPU instead of GPU
  python denoise.py --input vocals.wav --config settings.json --output denoised.wav --device cpu
        """
    )

    # Required arguments
    parser.add_argument(
        '--input',
        required=True,
        help='Input audio file (WAV format recommended)'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='JSON configuration file with noise_reduction settings'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output audio file path'
    )

    # Optional overrides
    parser.add_argument(
        '--strength',
        type=float,
        metavar='FLOAT',
        help='Override noise reduction strength (0.0 to 1.0)'
    )
    parser.add_argument(
        '--gate-threshold',
        type=float,
        metavar='DB',
        help='Override noise gate threshold in dB (e.g., -40)'
    )
    parser.add_argument(
        '--no-gate',
        action='store_true',
        help='Disable noise gate post-processing'
    )

    # Execution options
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Computation device (default: auto)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed processing information'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    try:
        # Load JSON configuration
        if args.verbose:
            print(f"Loading configuration from {args.config}...")

        config = NoiseReductionConfig(args.config)

        if args.verbose:
            print(f"  {config}")

        # Resolve settings (CLI args override JSON)
        strength = args.strength if args.strength is not None else config.get_strength()
        gate_threshold = (args.gate_threshold if args.gate_threshold is not None
                         else config.get_gate_threshold())
        apply_gate = not args.no_gate and config.is_gate_enabled()

        # Validate overridden strength
        if strength < 0.0 or strength > 1.0:
            print(f"Error: --strength must be between 0.0 and 1.0, got {strength}")
            sys.exit(1)

        if args.verbose:
            print(f"\nProcessing settings:")
            print(f"  Strength: {strength:.2f}")
            print(f"  Noise gate: {'enabled' if apply_gate else 'disabled'}")
            if apply_gate:
                print(f"  Gate threshold: {gate_threshold:.1f} dB")

        # Load audio
        if args.verbose:
            print(f"\nLoading audio from {args.input}...")

        audio, sample_rate = load_audio(args.input)

        if args.verbose:
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Duration: {len(audio) / sample_rate:.2f} seconds")
            print(f"  Samples: {len(audio)}")

        # Initialize denoiser
        if args.verbose:
            print(f"\nInitializing DeepFilterNet3 on {args.device}...")

        denoiser = Denoiser(device=args.device)

        # Process audio
        if args.verbose:
            print("\n" + "="*50)
            print("Processing audio...")
            print("="*50)

        processed = denoiser.process(
            audio=audio,
            sample_rate=sample_rate,
            strength=strength,
            apply_gate=apply_gate,
            gate_threshold=gate_threshold,
            verbose=args.verbose
        )

        # Save result
        if args.verbose:
            print(f"\nSaving denoised audio to {args.output}...")

        save_audio(args.output, processed, sample_rate)

        if args.verbose:
            print("="*50)
            print("✓ Noise reduction complete!")
            print("="*50)
        else:
            print(f"✓ Denoised audio saved to {args.output}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Processing error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
