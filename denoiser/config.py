"""
JSON configuration parser for noise reduction settings
"""

import json
from pathlib import Path
from typing import Dict, Any


class NoiseReductionConfig:
    """
    Parse and validate noise reduction settings from JSON config file.

    Expected JSON structure:
    {
        "noise_reduction": {
            "strength": 0.5,
            "apply_gate": true,
            "gate_threshold": -40,
            "reason": "..."
        }
    }
    """

    def __init__(self, json_path: str):
        """
        Load and parse JSON configuration.

        Args:
            json_path: Path to JSON configuration file

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            ValueError: If JSON is invalid or missing required fields
        """
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")

        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self._validate()

    def _validate(self):
        """Validate configuration structure and values."""
        if "noise_reduction" not in self.config:
            raise ValueError("JSON must contain 'noise_reduction' section")

        nr_config = self.config["noise_reduction"]

        # Validate required fields
        if "strength" not in nr_config:
            raise ValueError("Missing required field: 'strength'")

        # Validate strength range
        strength = nr_config["strength"]
        if not (0.0 <= strength <= 1.0):
            raise ValueError(f"'strength' must be between 0.0 and 1.0, got {strength}")

        # Validate optional gate settings
        if "apply_gate" in nr_config and nr_config["apply_gate"]:
            if "gate_threshold" not in nr_config:
                raise ValueError("'gate_threshold' required when 'apply_gate' is true")

    def get_strength(self) -> float:
        """Get noise reduction strength (0.0 to 1.0)."""
        return self.config["noise_reduction"]["strength"]

    def get_gate_threshold(self) -> float:
        """Get noise gate threshold in dB."""
        return self.config["noise_reduction"].get("gate_threshold", -40.0)

    def is_gate_enabled(self) -> bool:
        """Check if noise gate should be applied."""
        return self.config["noise_reduction"].get("apply_gate", True)

    def get_reason(self) -> str:
        """Get processing reason/note (optional)."""
        return self.config["noise_reduction"].get("reason", "")

    def to_dict(self) -> Dict[str, Any]:
        """Return noise_reduction config as dictionary."""
        return self.config["noise_reduction"].copy()

    def __repr__(self) -> str:
        return (f"NoiseReductionConfig(strength={self.get_strength()}, "
                f"gate_enabled={self.is_gate_enabled()}, "
                f"gate_threshold={self.get_gate_threshold()})")
