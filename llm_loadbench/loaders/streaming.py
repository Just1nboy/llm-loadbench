import gc
import re
import time

import torch
from transformers import AutoTokenizer

from ..metrics import (
    init_model_on_meta,
    load_state_dict_from_file,
    resolve_weight_path,
)
from .base import BaseLoader


class StreamingLoader(BaseLoader):
    """
    Weight streaming: load transformer blocks one at a time,
    release after computation.
    """

    strategy_name = "streaming"

    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__(model_name, device)
        self._weight_path = None
        self._is_safetensors = False
        self._hooks = []

    def _get_layer_weights(self, layer_prefix: str) -> dict:
        """Load only weights for a specific layer from disk."""
        if self._is_safetensors:
            from safetensors import safe_open
            tensors = {}
            with safe_open(self._weight_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith(layer_prefix):
                        tensors[key] = f.get_tensor(key)
            return tensors
        else:
            full = torch.load(
                self._weight_path, map_location="cpu",
                mmap=True, weights_only=True,
            )
            return {k: v for k, v in full.items() if k.startswith(layer_prefix)}

    def load_model(self):
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self._weight_path, self._is_safetensors = resolve_weight_path(self.model_name)

        self.model, _ = init_model_on_meta(self.model_name)

        full_sd = load_state_dict_from_file(self._weight_path, self._is_safetensors)
        block_pattern = re.compile(r'(?:transformer\.h|model\.layers)\.\d+\.')
        non_block_sd = {k: v for k, v in full_sd.items() if not block_pattern.search(k)}
        del full_sd

        self.model.load_state_dict(non_block_sd, strict=False, assign=True)

        if hasattr(self.model, 'lm_head') and hasattr(self.model, 'transformer'):
            if hasattr(self.model.transformer, 'wte'):
                self.model.lm_head.weight = self.model.transformer.wte.weight

        for name, param in list(self.model.named_parameters()):
            if block_pattern.search(name) and param.device == torch.device('meta'):
                parts = name.split('.')
                module = self.model
                for part in parts[:-1]:
                    module = getattr(module, part)
                setattr(module, parts[-1], torch.nn.Parameter(
                    torch.empty(0, device='cpu'), requires_grad=False
                ))

        self._register_streaming_hooks()
        self.model.eval()

        load_time = time.perf_counter() - t0
        return self.model, self.tokenizer, load_time

    def _register_streaming_hooks(self):
        blocks = None
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            blocks = self.model.transformer.h
            prefix_template = "transformer.h.{idx}."
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            blocks = self.model.model.layers
            prefix_template = "model.layers.{idx}."

        if blocks is None:
            print("WARNING: Could not find transformer blocks.")
            return

        for idx, block in enumerate(blocks):
            prefix = prefix_template.format(idx=idx)

            def make_pre_hook(p):
                def hook(module, inputs):
                    weights = self._get_layer_weights(p)
                    local_weights = {k[len(p):]: v for k, v in weights.items()}
                    module.load_state_dict(local_weights, strict=False, assign=True)
                    module.to(self.device)
                return hook

            def make_post_hook(p):
                def hook(module, inputs, outputs):
                    for pname, param in module.named_parameters(recurse=True):
                        parts = pname.split('.')
                        target = module
                        for part in parts[:-1]:
                            target = getattr(target, part)
                        setattr(target, parts[-1], torch.nn.Parameter(
                            torch.empty(0, device='cpu'), requires_grad=False
                        ))
                    gc.collect()
                return hook

            self._hooks.append(block.register_forward_pre_hook(make_pre_hook(prefix)))
            self._hooks.append(block.register_forward_hook(make_post_hook(prefix)))

    def generate(self, prompt: str, max_new_tokens: int = 50):
        """Manual token-by-token generation for streaming."""
        assert self.model is not None, "Model not loaded."

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        # TTFT
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(input_ids)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        ttft = time.perf_counter() - t0

        # Remaining tokens
        t1 = time.perf_counter()
        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                outputs = self.model(input_ids)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        total_time = time.perf_counter() - t1 + ttft

        generated_tokens = input_ids.shape[1] - input_len
        throughput = generated_tokens / total_time if total_time > 0 else 0
        output_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        return output_text, ttft, throughput, total_time

    def cleanup(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        super().cleanup()
