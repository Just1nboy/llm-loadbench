import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..metrics import init_model_on_meta
from .base import BaseLoader


class LazyLoader(BaseLoader):
    """
    Lazy loading: init with empty weights,
    load real weights on first forward pass.
    """

    strategy_name = "lazy"

    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__(model_name, device)
        self._weights_loaded = False
        self._hooks = []

    def _load_weights_on_first_use(self, module, input):
        if not self._weights_loaded:
            self._weights_loaded = True
            real_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float32,
            )
            self.model.load_state_dict(real_model.state_dict(), assign=True)
            self.model.to(self.device)
            del real_model
            gc.collect()

    def load_model(self):
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model, _ = init_model_on_meta(self.model_name)
        self.model = self.model.to_empty(device='cpu')
        self.model.eval()

        self._weights_loaded = False
        hook = self.model.register_forward_pre_hook(self._load_weights_on_first_use)
        self._hooks.append(hook)

        load_time = time.perf_counter() - t0
        return self.model, self.tokenizer, load_time

    def cleanup(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._weights_loaded = False
        super().cleanup()
