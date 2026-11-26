import pytest

torch = pytest.importorskip("torch")

from tierquant.policy import PrecisionMap, bytes_ratio  # noqa: E402
from tierquant.quant import (dequantize_groups, encode_block, encode_layer, quantize_groups,  # noqa: E402
                             rel_error)


def kv(H=4, T=256, D=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(H, T, D, generator=g).half(), torch.randn(H, T, D, generator=g).half())


@pytest.mark.parametrize("bits,tol", [(8, 0.01), (4, 0.15)])
@pytest.mark.parametrize("axis", [-1, -2])
def test_group_roundtrip_error(bits, tol, axis):
    x = kv()[0]
    y = dequantize_groups(quantize_groups(x, bits, axis, 64))
    assert y.shape == x.shape and rel_error(x, y) < tol


def test_int4_pack_unpack_exact_on_grid():
    # values on the 16-level grid must be reproduced (nearly) exactly
    x = (torch.arange(16).float().repeat(4, 8, 2)).half()      # [4,8,32]
    y = dequantize_groups(quantize_groups(x, 4, -1, 32))
    assert torch.allclose(x.float(), y.float(), atol=1e-2)


def test_padding_for_odd_lengths():
    x = torch.randn(2, 100, 32).half()                          # 100 tokens, group 64 -> padded
    y = dequantize_groups(quantize_groups(x, 4, -2, 64))
    assert y.shape == x.shape


def test_sizes_match_policy_ratio():
    k, v = kv(H=8, T=256, D=128)
    raw = (k.numel() + v.numel()) * 2
    for prec in ("fp8", "int8", "int4"):
        enc = encode_layer(k, v, prec, 64, 64)
        assert enc.nbytes() / raw == pytest.approx(bytes_ratio(prec, 64), rel=0.02)


def test_fp16_lossless_and_sinks_exact():
    k, v = kv()
    dk, dv = encode_layer(k, v, "fp16").decode()
    assert torch.equal(dk, k) and torch.equal(dv, v)
    dk, dv = encode_layer(k, v, "int4", sink_tokens=4).decode()
    assert torch.equal(dk[:, :4], k[:, :4]) and torch.equal(dv[:, :4], v[:, :4])
    assert not torch.equal(dk[:, 4:], k[:, 4:])


def test_protected_layers_exact_in_block():
    ks, vs = zip(*[kv(seed=i) for i in range(4)])
    pm = PrecisionMap(protect_first_layers=2)
    enc = encode_block(ks, vs, pm.layer_precisions("nvme", 4), sink_tokens=4)
    dk, _ = enc.decode()
    assert torch.equal(dk[0], ks[0]) and torch.equal(dk[1], ks[1])
    assert rel_error(ks[2], dk[2]) > 0


from tierquant.manager import TieredKVManager  # noqa: E402


def mk(tmp_path, requant, caps=None, n_layers=4):
    pm = PrecisionMap(tiers={"hbm": "fp16", "dram": "fp8", "nvme": "int4"}, protect_first_layers=1)
    return TieredKVManager(caps or {"hbm": 1, "dram": 1, "nvme": 10**9}, pm, n_layers, device="cpu",
                           nvme_dir=str(tmp_path), requant=requant, measure_error=True)


def block(n_layers=4, seed=0):
    ks, vs = zip(*[kv(H=2, T=64, D=32, seed=seed * 10 + i) for i in range(n_layers)])
    return list(ks), list(vs)


def fp16_bytes(n_layers=4):
    return n_layers * 2 * 2 * 64 * 32 * 2


def test_cascade_demotes_hbm_to_dram_to_nvme(tmp_path):
    fb = fp16_bytes()
    m = mk(tmp_path, "stacked", {"hbm": fb, "dram": fb, "nvme": 10 * fb})
    for i in range(4):
        ks, vs = block(seed=i)
        m.put(f"b{i}", ks, vs, start_token=64)
    tiers = [m.contains(f"b{i}") for i in range(4)]
    assert tiers[3] == "hbm" and "nvme" in tiers
    assert all(t is not None for t in tiers)


def test_single_shot_never_requantises(tmp_path):
    fb = fp16_bytes()
    m = mk(tmp_path, "single_shot", {"hbm": fb, "dram": fb, "nvme": 10 * fb})
    for i in range(3):
        ks, vs = block(seed=i)
        m.put(f"b{i}", ks, vs, start_token=64)
    for t in m.tiers[1:]:
        for e in m.store[t].values():
            assert e.meta.n_lossy == 1


def test_stacked_compounds_error(tmp_path):
    fb = fp16_bytes()
    m = mk(tmp_path, "stacked", {"hbm": fb, "dram": fb // 2, "nvme": 10 * fb})
    for i in range(4):
        ks, vs = block(seed=i)
        m.put(f"b{i}", ks, vs, start_token=64)
    lossy = [e.meta.n_lossy for e in m.store["nvme"].values()]
    assert lossy and max(lossy) == 2


def test_get_roundtrip_and_promotion(tmp_path):
    fb = fp16_bytes()
    m = mk(tmp_path, "single_shot", {"hbm": fb, "dram": 10 * fb, "nvme": 10 * fb})
    ks0, vs0 = block(seed=0)
    m.put("a", ks0, vs0, start_token=0)
    ks1, vs1 = block(seed=1)
    m.put("b", ks1, vs1, start_token=64)           # evicts "a" from HBM
    assert m.contains("a") in ("dram", "nvme")
    out = m.get("a")
    assert out is not None
    dk, dv, meta = out
    assert meta.n_lossy == 1
    assert torch.equal(dk[0], ks0[0])                      # protected layer exact
    assert torch.equal(dk[1][:, :4], ks0[1][:, :4])        # sink tokens exact (start_token == 0)
    assert rel_error(ks0[1], dk[1]) < 0.1
    assert m.stats["hits"]["dram"] + m.stats["hits"]["nvme"] == 1


def test_miss_counted(tmp_path):
    m = mk(tmp_path, "single_shot")
    assert m.get("nope") is None and m.stats["misses"] == 1
