"""tierquant: precision-descending KV-cache tiering.

Torch-free modules (block, policy, breakeven, sim, workloads) import without torch.
Torch-dependent modules (quant, manager) are imported explicitly by the caller.
"""
__version__ = "0.1.0"
