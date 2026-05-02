import time

from transformers import AutoTokenizer

from ..metrics import (
    init_model_on_meta,
    load_and_tie,
    load_state_dict_from_file,
    resolve_weight_path,
)
from .base import BaseLoader


class MmapLoader(BaseLoader):
    """
    Memory-mapped loading.

    Meta-device init (zero alloc) -> load_state_dict(assign=True)
    lets the OS page cache serve repeated loads efficiently.
    """

    strategy_name = "mmap"

    def load_model(self):
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model, _ = init_model_on_meta(self.model_name)

        weight_path, is_safetensors = resolve_weight_path(self.model_name)
        state_dict = load_state_dict_from_file(weight_path, is_safetensors)

        load_and_tie(self.model, state_dict, self.device)

        load_time = time.perf_counter() - t0
        return self.model, self.tokenizer, load_time
