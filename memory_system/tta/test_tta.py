"""Tests for the TTA DPO v1 (CPU by default; GPU smoke only with --gpu).

Run:  python memory_system/tta/test_tta.py            # CPU tests
      python memory_system/tta/test_tta.py --gpu      # + GPU optimizer-step smoke

The GPU smoke is WRITTEN but has NOT been executed yet (2026-09-09: GPU time
was reserved for validation only; running the smoke is the first thing to do
once GPU authorization covers it). It performs ONE throwaway optimizer step
and saves nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# MUST run before the imports below: they transitively import huggingface_hub,
# whose cache/offline config freezes at import time (review finding 3).
from memory_system.tta.model import setup_offline_hf_cache  # noqa: E402

setup_offline_hf_cache()

from memory_system.tta.dataset import CHOSEN, PreferenceDataset  # noqa: E402
from memory_system.tta.model import (  # noqa: E402
    TTADPOModel,
    dpo_loss_from_errors,
    load_policy_model,
    share_noise_across_pairs,
)

CKPT_DIR = Path("/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B")
EPISODE_DIR = REPO / "LIBERO-Cosmos-Policy" / "all_episodes"
_FIXTURE_FAIL = sorted(EPISODE_DIR.glob("episode_data--*--success=False--regen_demo.hdf5"))[0]
_FIXTURE_OK = sorted(EPISODE_DIR.glob("episode_data--*--success=True--regen_demo.hdf5"))[0]
_STATS = str(CKPT_DIR / "libero_dataset_statistics.json")
_T5 = str(CKPT_DIR / "libero_t5_embeddings.pkl")

# Golden captured from git-HEAD LIBERODataset on 2026-09-09 (sample 0/1 of the
# fixture episode, aug off, normalization off) — guards the shared assembly.
_GOLDEN = {
    0: {"video_sha": "729936d1cec9b147", "actions_sha": "c32dcc2fbb9960d3"},
    1: {"video_sha": "44fb29f6af12f8fd", "actions_sha": "e139e16eacb4dd91"},
}
_FIXTURE_FAIL = _FIXTURE_FAIL  # rejected side
_FIXTURE_OK = _FIXTURE_OK      # chosen side


def _hash(t) -> str:
    if isinstance(t, torch.Tensor):
        if t.dtype == torch.bfloat16:
            t = t.float()
        t = t.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest()[:16]


def _fixture_pair_dataset(strict_labels: bool = True) -> PreferenceDataset:
    """Real pair: a SUCCEEDING episode as chosen and a FAILING one as rejected
    (different episodes -> deltas do not cancel; strict label check active)."""
    manifest = Path("/tmp/tta_test_pair_manifest.json")  # test-owned file, safe to (re)write
    manifest.write_text(json.dumps(
        {"pairs": [{"pair_id": "TEST", "chosen_path": str(_FIXTURE_OK),
                    "rejected_path": str(_FIXTURE_FAIL),
                    "chosen_success": True, "rejected_success": False}]}
    ))
    return PreferenceDataset(
        manifest_path=str(manifest),
        t5_text_embeddings_path=_T5,
        dataset_stats_path=_STATS,
        normalize_actions=False,
        normalize_proprio=False,
        strict_labels=strict_labels,
    )


# ------------------------------------------------------------------ CPU tests
def test_loss_sign_swap_and_log2():
    # deltas: chosen fits better (lower error) than rejected -> margin > 0
    delta_c = torch.tensor([0.5, 1.5])
    delta_r = torch.tensor([1.0, 2.0])
    loss, margin = dpo_loss_from_errors(delta_c, delta_r, beta=1.0)
    assert margin > 0 and float(loss) < float(torch.log(torch.tensor(2.0)))
    _, margin_swapped = dpo_loss_from_errors(delta_r, delta_c, beta=1.0)
    assert abs(float(margin) + float(margin_swapped)) < 1e-6
    zero = torch.zeros(2, requires_grad=True)
    loss0, margin0 = dpo_loss_from_errors(zero, zero.clone().detach(), beta=0.1)
    assert abs(float(margin0)) < 1e-6 and abs(float(loss0) - 0.693147) < 1e-4
    loss0.backward()  # gradients must flow at zero margin
    assert zero.grad is not None and float(zero.grad.abs().sum()) > 0


def test_share_noise_across_pairs():
    eps = torch.arange(8, dtype=torch.float32).reshape(4, 2)      # 2 pairs x 2 sides
    sig = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    eps2, sig2 = share_noise_across_pairs(eps.clone(), sig.clone())
    assert torch.equal(eps2[1], eps2[0]) and torch.equal(eps2[2], eps2[3])
    assert torch.equal(sig2[1], sig2[0]) and torch.equal(sig2[3], sig2[2])
    assert torch.equal(eps2[0], eps[0]) and torch.equal(eps2[2], eps[2])  # chosen slot untouched


def test_dataset_pair_item_shapes():
    ds = _fixture_pair_dataset()
    item = ds[0]
    assert item["video"].shape[0] == 2 and item["video"].dtype == torch.uint8
    assert item["video"].shape[1] == 3 and item["video"].shape[2] == 33  # C, T=1+8*4
    assert item["actions"].shape == (2, 16, 7)
    assert list(item["action_latent_idx"]) == [4, 4]
    assert list(item["rollout_data_mask"]) == [0, 0]  # policy-mode forward
    assert item["side"] == [CHOSEN, 1 - CHOSEN]
    assert isinstance(item["pair_id"], list) and item["pair_id"] == ["TEST", "TEST"]

    # The chosen side must be byte-identical to what LIBERODataset produces at
    # the same episode step (both go through build_action_chunk_sample).
    from cosmos_policy.datasets.libero_dataset import LIBERODataset

    tmp = Path(tempfile.mkdtemp(prefix="tta_pair_ref_"))
    try:
        (tmp / "rollouts").mkdir()
        (tmp / "demos").mkdir()
        os.symlink(_FIXTURE_OK, tmp / "rollouts" / _FIXTURE_OK.name)
        shutil.copy(CKPT_DIR / "libero_dataset_statistics.json", tmp / "demos" / "dataset_statistics.json")
        ref = LIBERODataset(
            data_dir=str(tmp / "demos"), t5_text_embeddings_path=_T5, chunk_size=16,
            use_image_aug=False, use_stronger_image_aug=False, normalize_actions=False,
            normalize_proprio=False, rollout_data_dir=str(tmp / "rollouts"),
            demonstration_sampling_prob=0.0, success_rollout_sampling_prob=1.0,
        )
        start = item["chunk_start"][0]
        ref_sample = ref[start]
        assert torch.equal(item["video"][0], ref_sample["video"]), "chosen video != LIBERODataset at same step"
        assert np.allclose(item["actions"][0], ref_sample["actions"]), "chosen actions != LIBERODataset"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_golden_libero_assembly():
    """Compact lock on the shared build_action_chunk_sample via LIBERODataset."""
    from cosmos_policy.datasets.libero_dataset import LIBERODataset

    tmp = Path(tempfile.mkdtemp(prefix="tta_golden_"))
    try:
        (tmp / "rollouts").mkdir()
        (tmp / "demos").mkdir()
        os.symlink(_FIXTURE_FAIL, tmp / "rollouts" / _FIXTURE_FAIL.name)
        shutil.copy(CKPT_DIR / "libero_dataset_statistics.json", tmp / "demos" / "dataset_statistics.json")
        ref = LIBERODataset(
            data_dir=str(tmp / "demos"), t5_text_embeddings_path=_T5, chunk_size=16,
            use_image_aug=False, use_stronger_image_aug=False, normalize_actions=False,
            normalize_proprio=False, rollout_data_dir=str(tmp / "rollouts"),
            demonstration_sampling_prob=0.0, success_rollout_sampling_prob=0.0,
        )
        for idx, expected in _GOLDEN.items():
            s = ref[idx]
            assert _hash(s["video"]) == expected["video_sha"], f"video changed at idx={idx}"
            assert _hash(s["actions"]) == expected["actions_sha"], f"actions changed at idx={idx}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ GPU smoke
def gpu_smoke() -> None:
    """WRITTEN, NOT YET EXECUTED. One throwaway optimizer step; saves nothing
    (the save/load check writes to /tmp and deletes immediately).

    Checks: LoRA-only optimizer; base frozen with grad None; base tensors'
    _version unchanged (no in-place base mutation, no copying); >=1 LoRA
    parameter changes; shared noise + adapter toggle non-vacuous (perturbed
    lora_B changes policy but never the reference); adapter save -> destroy ->
    reload -> identical loss under the SAME noise; peak memory reported.
    """
    from memory_system.tta.dpo_train import pick_gpu

    gpu = pick_gpu(20.0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # before first CUDA use
    print(f"[gpu-smoke] GPU {gpu} (least used)")

    inner = load_policy_model(str(CKPT_DIR / "Cosmos-Policy-LIBERO-Predict2-2B.pt"))
    model = TTADPOModel(inner, beta=0.1, base_checkpoint=str(CKPT_DIR / "Cosmos-Policy-LIBERO-Predict2-2B.pt"))
    model.model.train()

    ds = _fixture_pair_dataset()
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=1)
    batch = model.batch_to_device(next(iter(loader)))
    prepared = model.prepare_inputs(batch)  # ONE draw, reused everywhere below

    def perturb_lora(std: float) -> None:
        with torch.no_grad():
            for _n, m in model.model.net.named_modules():
                if hasattr(m, "lora_B"):
                    for b in m.lora_B.values():
                        b.weight.normal_(0.0, std)

    # zero adapter: policy == reference under the same noise
    e_pol0, e_ref0 = model.errors(batch, prepared=prepared)
    assert torch.allclose(e_pol0, e_ref0, rtol=1e-3, atol=1e-4), "zero adapter: policy != reference"

    # perturbed adapter: policy changes, reference is blind to adapter weights
    perturb_lora(0.02)
    e_pol1, e_ref1 = model.errors(batch, prepared=prepared)
    assert not torch.allclose(e_pol1, e_pol0, rtol=1e-3, atol=1e-4), "perturbed adapter must change policy"
    assert torch.allclose(e_ref1, e_ref0, rtol=1e-3, atol=1e-4), "reference must not see adapter weights"
    assert not torch.allclose(e_pol1, e_ref1, rtol=1e-3, atol=1e-4), "deltas must be nonzero now"
    print("[gpu-smoke] shared-noise toggle/perturb: OK")

    # DPO forward + backward (margin nonzero BECAUSE the adapter is perturbed)
    loss, margin, deltas = model.dpo_forward(batch, prepared=prepared)
    assert abs(float(margin)) > 0, "perturbed adapter must give a nonzero margin"
    loss.backward()
    lora_params = model.lora_parameters()
    assert lora_params and all(p.grad is not None for p in lora_params), "LoRA params must have grads"
    base_params = [p for n, p in model._iter_params() if not p.requires_grad]
    assert all(p.grad is None for p in base_params), "base params must have no grads"
    assert all(not p.requires_grad for p in base_params)

    base_versions = {n: p._version for n, p in model._iter_params() if not p.requires_grad}
    lora_before = {n: p.detach().clone() for n, p in model._iter_params() if p.requires_grad}
    optimizer = torch.optim.AdamW(lora_params, lr=1e-5)
    assert all(p.requires_grad for p in sum((g["params"] for g in optimizer.param_groups), [])), \
        "optimizer must contain only LoRA parameters"
    optimizer.step()
    changed = any(not torch.equal(lora_before[n], p.detach()) for n, p in model._iter_params() if p.requires_grad)
    assert changed, "optimizer step must change at least one LoRA parameter"
    assert all(model.model.net.state_dict(keep_vars=True)[n]._version == v for n, v in base_versions.items()), \
        "base parameters must be untouched by the optimizer step"
    print(f"[gpu-smoke] step: loss={float(loss):.4f} margin={float(margin):.4f} "
          f"peak={torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

    # adapter save -> destroy -> reload -> identical loss under the same noise
    loss_saved = float(model.dpo_forward(batch, prepared=prepared)[0])
    tmp_adapter = Path("/tmp/tta_smoke_adapter.pt")
    model.save_adapter(str(tmp_adapter), extra_meta={"step": 1})
    perturb_lora(0.5)
    model.load_adapter(str(tmp_adapter))
    loss_reloaded = float(model.dpo_forward(batch, prepared=prepared)[0])
    assert abs(loss_saved - loss_reloaded) < 1e-6, "adapter reload must restore identical loss"
    os.remove(tmp_adapter)
    print("[gpu-smoke] save/reload: OK (nothing persisted)")


if __name__ == "__main__":
    test_loss_sign_swap_and_log2()
    print("PASS test_loss_sign_swap_and_log2")
    test_dataset_pair_item_shapes()
    print("PASS test_dataset_pair_item_shapes")
    test_golden_libero_assembly()
    print("PASS test_golden_libero_assembly")
    if "--gpu" in sys.argv:
        assert torch.cuda.is_available(), "--gpu requested but CUDA is unavailable"
        gpu_smoke()
    else:
        print("(GPU smoke skipped: pass --gpu to run it — written, not yet executed)")
    print("ALL CPU TESTS PASSED")
