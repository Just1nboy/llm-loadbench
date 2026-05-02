import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseLoader


class StandardLoader(BaseLoader):
    """Baseline: default HuggingFace model loading."""

    strategy_name = "standard"

    def load_model(self):
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        load_time = time.perf_counter() - t0
        return self.model, self.tokenizer, load_time
