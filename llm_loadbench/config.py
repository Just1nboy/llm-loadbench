import os
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class ExperimentConfig:
    """Central configuration for all experiments."""
    models: List[str] = field(default_factory=lambda: [
        "EleutherAI/gpt-neo-125m",
    ])
    warmup_iterations: int = 3
    measured_iterations: int = 10
    input_prompt: str = "The future of artificial intelligence is"
    max_new_tokens: int = 50
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    output_dir: str = "results"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
