import time

import torch

from ..metrics import MetricsCollector


class BaseLoader:
    """Base class for all loading strategies."""

    strategy_name: str = "base"

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        raise NotImplementedError

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        MetricsCollector.clear_state()

    def generate(self, prompt: str, max_new_tokens: int = 50):
        assert self.model is not None, "Model not loaded."

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # TTFT
        t0 = time.perf_counter()
        with torch.no_grad():
            self.model.generate(**inputs, max_new_tokens=1, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = time.perf_counter() - t0

        # Full generation
        t1 = time.perf_counter()
        with torch.no_grad():
            full_out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.perf_counter() - t1

        generated_tokens = full_out.shape[1] - input_len
        throughput = generated_tokens / total_time if total_time > 0 else 0
        output_text = self.tokenizer.decode(full_out[0], skip_special_tokens=True)

        return output_text, ttft, throughput, total_time
