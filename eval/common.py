import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def load(model_name: str, dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="cuda").eval()
    return tok, model


def to_legacy(pkv):
    return pkv.to_legacy_cache() if hasattr(pkv, "to_legacy_cache") else tuple(pkv)


def fresh_cache(legacy):
    """DynamicCache.update() concatenates (new tensors), so legacy tensors are never mutated."""
    return DynamicCache.from_legacy_cache(tuple(legacy))


def wikitext_ids(tok, min_tokens: int, split="test", name="wikitext-2-raw-v1"):
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("wikitext", name, split=split)["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    assert ids.numel() >= min_tokens, f"corpus has {ids.numel()} tokens, need {min_tokens}"
    return ids


@torch.no_grad()
def greedy(model, tok, legacy_past, suffix_ids, n_new=16):
    out = model(suffix_ids, past_key_values=fresh_cache(legacy_past), use_cache=True)
    gen = []
    for _ in range(n_new):
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        gen.append(nxt)
        out = model(nxt, past_key_values=out.past_key_values, use_cache=True)
    return tok.decode(torch.cat(gen, 1)[0], skip_special_tokens=True)
