import time
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..metrics import init_model_on_meta, load_and_tie
from .base import BaseLoader


class CachedLoader(BaseLoader):
    """
    Cached loading with class-level state dict cache.

    MISS: standard from_pretrained, then cache state dict.
    HIT: meta init + load_state_dict(cached_sd, assign=True).
    """

    strategy_name = "cached"
    _cache: Dict[str, dict] = {}
    hits: int = 0
    misses: int = 0

    def load_model(self):
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.model_name in self._cache:
            CachedLoader.hits += 1
            self.model, _ = init_model_on_meta(self.model_name)
            load_and_tie(self.model, self._cache[self.model_name], self.device)
        else:
            CachedLoader.misses += 1
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float32,
            )
            self.model.to(self.device)
            self.model.eval()

            self._cache[self.model_name] = {
                k: v.clone().cpu() for k, v in self.model.state_dict().items()
            }

        load_time = time.perf_counter() - t0
        return self.model, self.tokenizer, load_time

    @classmethod
    def reset_cache(cls):
        cls._cache.clear()
        cls.hits = 0
        cls.misses = 0

    @classmethod
    def cache_stats(cls) -> dict:
        total = cls.hits + cls.misses
        return {
            "hits": cls.hits, "misses": cls.misses,
            "hit_rate": f"{cls.hits/total:.1%}" if total > 0 else "N/A",
        }
