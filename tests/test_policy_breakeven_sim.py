import pytest

from tierquant import workloads
from tierquant.block import prefix_block_ids
from tierquant.breakeven import (HardwareProfile, breakeven_tokens, example_profile, load_time,
                                 recompute_time, should_load)
from tierquant.policy import PrecisionMap, QualityBudgetController, bytes_ratio
from tierquant.sim import HierarchySimulator, run_trace


def test_bytes_ratio_values():
    assert bytes_ratio("fp16") == 1.0
    assert bytes_ratio("fp8") == 0.5
    assert bytes_ratio("int4", 64) == pytest.approx(0.28125)
    assert bytes_ratio("int8", 64) == pytest.approx(0.53125)


def test_precision_must_not_increase_down_hierarchy():
    with pytest.raises(ValueError):
        PrecisionMap(tiers={"hbm": "fp16", "dram": "int4", "nvme": "fp8"})
    with pytest.raises(ValueError):
        PrecisionMap(tiers={"hbm": "fp8", "dram": "fp8"})


def test_protected_layers_stay_fp16():
    pm = PrecisionMap(protect_first_layers=2)
    assert pm.layer_precisions("nvme", 4) == ["fp16", "fp16", "int4", "int4"]
    assert 0.28 < pm.ratio("nvme") < 0.29
    assert pm.ratio("nvme", n_layers=32) > pm.ratio("nvme")


def test_controller_respects_budget_and_maximises_capacity():
    delta = {"fp16": 0.0, "fp8": 0.1, "int8": 0.05, "int4": 1.0}
    caps = {"dram": 100e9, "nvme": 1000e9}
    share = {"dram": 0.4, "nvme": 0.4}
    loose = QualityBudgetController(delta, budget=10.0).choose(share, caps)
    assert loose.tiers["nvme"] == "int4"            # nothing binds -> most compression
    tight = QualityBudgetController(delta, budget=0.2).choose(share, caps)
    assert tight.tiers["nvme"] != "int4"            # int4 would cost 0.4 > 0.2
    ctl = QualityBudgetController(delta, budget=0.2)
    assert ctl.expected_degradation(tight, share) <= 0.2
    zero = QualityBudgetController(delta, budget=0.0).choose(share, caps)
    assert zero.tiers == {"hbm": "fp16", "dram": "fp16", "nvme": "fp16"}


def test_controller_prefers_compressing_where_hits_are_rare():
    delta = {"fp16": 0.0, "fp8": 0.3, "int8": 0.3, "int4": 1.0}
    caps = {"dram": 100e9, "nvme": 1000e9}
    ctl = QualityBudgetController(delta, budget=0.35)
    pm = ctl.choose({"dram": 0.05, "nvme": 0.9}, caps)   # most hits come from nvme
    assert ctl.expected_degradation(pm, {"dram": 0.05, "nvme": 0.9}) <= 0.35


def test_breakeven_monotone_in_ratio_and_consistent_with_should_load():
    p = example_profile()
    n_full = breakeven_tokens(p, "redis", 1.0)
    n_q = breakeven_tokens(p, "redis", bytes_ratio("int4"))
    assert n_q < n_full                               # fewer bytes -> loading wins earlier
    n = breakeven_tokens(p, "redis", 1.0)
    assert should_load(p, "redis", n, 1.0) and not should_load(p, "redis", n - 1, 1.0)
    assert load_time(p, "nvme", 8192, 0.28) < load_time(p, "nvme", 8192, 1.0) + 8192 * p.kv_bytes_per_token / p.dequant_Bps


def test_breakeven_never_when_load_slower_than_prefill():
    p = HardwareProfile(tier_bw={"slow": 1e6}, tier_latency={"slow": 0.0}, prefill_c=0.0)
    assert breakeven_tokens(p, "slow", 1.0, max_tokens=100000) is None


def test_prefix_block_ids_chain():
    a = prefix_block_ids(list(range(1024)), 256)
    b = prefix_block_ids(list(range(512)) + [9] * 512, 256)
    assert len(a) == 4 and a[:2] == b[:2] and a[2] != b[2]


def test_workloads_deterministic_and_zipf_skew():
    t1 = workloads.zipf_rag(n_requests=500, n_docs=100, seed=3)
    t2 = workloads.zipf_rag(n_requests=500, n_docs=100, seed=3)
    assert [r.prefix_key for r in t1] == [r.prefix_key for r in t2]
    from collections import Counter
    c = Counter(r.prefix_key for r in t1).most_common()
    assert c[0][1] > 5 * c[-1][1]


def test_multi_turn_prefix_grows():
    reqs = [r for r in workloads.multi_turn(n_sessions=3, turns=4, seed=1) if r.prefix_key == "sess0"]
    pref = [r.prefix_tokens for r in sorted(reqs, key=lambda r: r.t)]
    assert pref == sorted(pref) and pref[0] == 0


def _sim(pm, caps, trace):
    return run_trace(trace, capacities_bytes=caps, pmap=pm, profile=example_profile())


def test_quantised_tiers_never_hurt_hit_rate_and_help_under_pressure():
    trace = workloads.zipf_rag(n_requests=3000, n_docs=1500, seed=0)
    caps = {"hbm": 8e9, "dram": 24e9, "nvme": 100e9}
    base = _sim(PrecisionMap(tiers={"hbm": "fp16", "dram": "fp16", "nvme": "fp16"}), caps, trace)
    q = _sim(PrecisionMap(tiers={"hbm": "fp16", "dram": "fp8", "nvme": "int4"}), caps, trace)
    assert q["capacity_blocks"] > 1.5 * base["capacity_blocks"]
    assert q["prefix_hit_rate"] >= base["prefix_hit_rate"]
    assert q["prefix_hit_rate"] > base["prefix_hit_rate"] + 0.03


def test_sim_exclusive_and_capacity_respected():
    pm = PrecisionMap(tiers={"hbm": "fp16", "dram": "fp8", "nvme": "int4"})
    sim = HierarchySimulator({"hbm": 1e9, "dram": 2e9, "nvme": 4e9}, pm)
    for i in range(200):
        sim.request(f"d{i % 40}", 4096, 64, 4096)
    for t in sim.tiers:
        assert sim.used[t] <= sim.cap[t] + 1e-6
    keys = [k for t in sim.tiers for k in sim.store[t]]
    assert len(keys) == len(set(keys))               # exclusive hierarchy


def test_fallback_triggers_on_slow_tier():
    p = HardwareProfile(tier_bw={"dram": 1e6, "nvme": 1e6}, tier_latency={"dram": 0, "nvme": 0})
    pm = PrecisionMap(tiers={"hbm": "fp16", "dram": "fp16", "nvme": "fp16"})
    sim = HierarchySimulator({"hbm": 0.5e9, "dram": 20e9, "nvme": 20e9}, pm, profile=p)
    sim.request("a", 8192, 64, 8192)
    for i in range(20):
        sim.request(f"x{i}", 8192, 64, 8192)          # push "a" down to dram/nvme
    sim.request("a", 8192, 64, 8192)
    assert sim.stats["fallback_tokens"] > 0
