"""One copy of the base weights for vLLM generation and LoRA training (vLLM env, one GPU).

vLLM runs in-process; an HF model is built on the meta device and every frozen parameter is bound to a view of
vLLM's tensor (q/k/v are row slices of the fused qkv_proj, gate/up of gate_up_proj), so training allocates only the
LoRA and its optimizer state. After an update, `AdapterPublisher.publish` saves the PEFT adapter to /dev/shm and
returns a new LoRARequest for vLLM. Checked by scripts/shared_vllm_proto.py (bit-identical weights, matching
log-probs, adapter updates visible to vLLM).

Importing this module sets VLLM_ENABLE_V1_MULTIPROCESSING=0; import it before anything imports vllm.
"""
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import shutil

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from vllm import LLM
from vllm.lora.request import LoRARequest

GB = 2 ** 30


def start_vllm(model: str, kv_gb: float = 2.0, max_model_len: int = 6144, max_lora_rank: int = 8, seed: int = 0,
               eager: bool = False) -> LLM:
    """In-process vLLM with LoRA enabled and a fixed KV-cache size (the rest of the GPU is left for training)."""
    return LLM(model=model, enable_lora=True, max_lora_rank=max_lora_rank, max_loras=2, max_model_len=max_model_len,
               seed=seed, kv_cache_memory_bytes=int(kv_gb * GB), enable_prefix_caching=True, enforce_eager=eager)


def bind_hf(llm: LLM, model: str, attn_implementation: str = "sdpa"):
    """An HF CausalLM (Qwen2 layout) whose parameters are views of vLLM's weights, all frozen."""
    vmodel = llm.apply_model(lambda m: m)[0]  # in-process: the module itself
    # plain tensors: vLLM's Parameter subclasses cannot be re-wrapped; LoRA wrapping adds ".base_layer"
    vparams = {n.replace(".base_layer", ""): p.data for n, p in vmodel.named_parameters()}
    cfg = AutoConfig.from_pretrained(model)
    H, nh, nkv, I = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.intermediate_size
    hd = getattr(cfg, "head_dim", None) or H // nh
    with torch.device("meta"):
        hf = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation=attn_implementation)

    def view(name):
        if name in ("lm_head.weight", "model.embed_tokens.weight"):  # vLLM may pad the vocabulary
            return vparams[name][: cfg.vocab_size]
        for proj, (lo, hi) in (("q_proj", (0, nh * hd)), ("k_proj", (nh * hd, (nh + nkv) * hd)),
                               ("v_proj", ((nh + nkv) * hd, (nh + 2 * nkv) * hd))):
            if f".self_attn.{proj}." in name:
                return vparams[name.replace(proj, "qkv_proj")][lo:hi]
        for proj, (lo, hi) in (("gate_proj", (0, I)), ("up_proj", (I, 2 * I))):
            if f".mlp.{proj}." in name:
                return vparams[name.replace(proj, "gate_up_proj")][lo:hi]
        return vparams[name]

    for name, p in list(hf.named_parameters()):
        v = view(name)
        assert tuple(v.shape) == tuple(p.shape), (name, tuple(v.shape), tuple(p.shape))
        parent, leaf = name.rsplit(".", 1)
        hf.get_submodule(parent)._parameters[leaf] = torch.nn.Parameter(v, requires_grad=False)
    device = next(iter(vparams.values())).device
    with torch.device(device):  # non-persistent buffers (rotary inv_freq) were created on meta
        hf.model.rotary_emb = type(hf.model.rotary_emb)(config=cfg)
    assert not any(t.is_meta for t in [*hf.parameters(), *hf.buffers()])
    return hf


class AdapterPublisher:
    """Save the PEFT adapter under /dev/shm and hand vLLM a new LoRARequest; keeps only the latest two versions."""

    def __init__(self, root: str):
        self.root, self.version = root, 0
        shutil.rmtree(root, ignore_errors=True)

    def publish(self, peft_model) -> LoRARequest:
        self.version += 1
        path = os.path.join(self.root, f"v{self.version}")
        peft_model.save_pretrained(path)
        shutil.rmtree(os.path.join(self.root, f"v{self.version - 2}"), ignore_errors=True)
        return LoRARequest(f"v{self.version}", self.version, path)

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)
