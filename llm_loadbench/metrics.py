import gc
import os
from dataclasses import dataclass

import psutil
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM


@dataclass
class BenchmarkMetrics:
    """Stores metrics for a single benchmark run."""
    strategy: str
    model_name: str
    load_time_s: float = 0.0
    peak_memory_mb: float = 0.0
    ttft_s: float = 0.0
    throughput_tps: float = 0.0
    total_inference_s: float = 0.0
    initial_memory_mb: float = 0.0
    memory_delta_mb: float = 0.0


class MetricsCollector:
    """Collects and tracks system metrics during experiments."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self._peak_memory = 0
        self._tracking = False

    def get_memory_mb(self) -> float:
        return self.process.memory_info().rss / (1024 ** 2)

    def start_tracking(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        self._peak_memory = self.get_memory_mb()
        self._tracking = True

    def update_peak(self):
        if self._tracking:
            current = self.get_memory_mb()
            self._peak_memory = max(self._peak_memory, current)

    def get_peak_memory_mb(self) -> float:
        self.update_peak()
        return self._peak_memory

    @staticmethod
    def clear_state():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def resolve_weight_path(model_name: str):
    """Find the cached weight file, return (path, is_safetensors)."""
    try:
        path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
        return path, True
    except Exception:
        path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        return path, False


def load_state_dict_from_file(path: str, is_safetensors: bool) -> dict:
    """Load a state dict from file (safetensors or pytorch bin)."""
    if is_safetensors:
        from safetensors.torch import load_file
        return load_file(path)  # mmap by default
    else:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def init_model_on_meta(model_name: str):
    """
    Initialize model on meta device (zero memory for parameters).
    Returns (model, config).
    """
    model_config = AutoConfig.from_pretrained(model_name)
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(model_config)
    return model, model_config


def load_and_tie(model, state_dict, device="cpu"):
    """
    Load state dict into a meta-initialized model using assign=True,
    then tie weights and move to device.
    """
    model.load_state_dict(state_dict, strict=False, assign=True)

    if hasattr(model, 'lm_head') and hasattr(model, 'transformer'):
        if hasattr(model.transformer, 'wte'):
            model.lm_head.weight = model.transformer.wte.weight

    model.to(device)
    model.eval()
    return model
