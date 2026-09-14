#!/usr/bin/env python
"""L1 mechanism probe: reference(base) vs base+DPO-adapter, PER-SIDE
(chosen / rejected) teacher-forcing action errors on the DPO pair manifest.

Design: ONE loaded model whose adapters are toggled (TTADPOModel), never two
loaded copies. prepare_inputs() draws ONE sigma/epsilon shared across each
pair's two sides, and errors() returns policy and reference errors for the
SAME draw — a paired comparison with minimal variance. Replaces the earlier
SFT base-vs-merged probe (git history keeps that version).

Registered L1 criterion (TTA_DPO_LAUNCH.md §5):
  chosen   delta = policy - reference  → expect < 0  (adapter improved the
                                        successful side)
  rejected delta = policy - reference  → expect >= chosen delta (worsened or
                                        less improved on the failed side)
  chosen-only improvement = overall drift, NOT contrastive learning → stop.

Without --adapter the model runs with its fresh (B=0) adapter: policy must
EQUAL reference, so every delta must be ~0 — the self-check mode.

Usage:
  python tests/tta/compare_base_vs_merged_train_loss.py --adapter <adapter.pt>
  python tests/tta/compare_base_vs_merged_train_loss.py            # self-check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from memory_system.tta.model import setup_offline_hf_cache  # noqa: E402

setup_offline_hf_cache()

import torch  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from memory_system.tta.dataset import PreferenceDataset  # noqa: E402
from memory_system.tta.dpo_train import pick_gpu  # noqa: E402
from memory_system.tta.model import TTADPOModel, load_policy_model  # noqa: E402

DEFAULT_BASE = "/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt"
DEFAULT_MANIFEST = str(REPO / "memory_system/tta/results/manifests/dpo_pairs_v3.json")
DEFAULT_META = str(REPO / "training/tta_sft_metadata")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", default="", help="adapter .pt; omit = fresh B=0 self-check")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--metadata-dir", default=DEFAULT_META)
    p.add_argument("--epochs", type=int, default=3, help="chunk draws per pair (set_epoch)")
    p.add_argument("--max-pairs", type=int, default=0, help="0 = all pairs in the manifest")
    p.add_argument("--out", default="", help="json record path (default: alongside adapter)")
    return p.parse_args()


def main():
    args = parse_args()
    gpu = pick_gpu(12.0)
    # load_policy_model hardcodes .to("cuda"): pin the card via env BEFORE the
    # first CUDA call instead of re-.to()-ing a loaded model.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[probe] GPU {gpu}")

    meta = {}
    if args.adapter:
        meta = torch.load(args.adapter, map_location="cpu")["meta"]
    rank = meta.get("lora_rank", 8)
    alpha = meta.get("lora_alpha", 16)
    beta = meta.get("beta", 0.1)
    targets = meta.get("target_modules",
                       "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2")

    model = load_policy_model(args.base)
    ttm = TTADPOModel(model, beta=beta, lora_rank=rank, lora_alpha=alpha,
                      target_modules=targets, base_checkpoint=args.base)
    if args.adapter:
        ttm.load_adapter(args.adapter)
        mode = f"adapter={Path(args.adapter).name}"
    else:
        mode = "SELF-CHECK (fresh B=0 adapter, deltas must be ~0)"
    print(f"[probe] {mode}")

    ds = PreferenceDataset(
        manifest_path=args.manifest,
        t5_text_embeddings_path=str(Path(args.metadata_dir) / "t5_embeddings.pkl"),
        dataset_stats_path=str(Path(args.metadata_dir) / "dataset_statistics.json"),
        strict_labels=True,
    )
    n_pairs = len(ds) if args.max_pairs <= 0 else min(args.max_pairs, len(ds))
    print(f"[probe] manifest pairs={len(ds)} evaluating={n_pairs} epochs={args.epochs}")

    rec = {"mode": mode, "adapter": args.adapter, "manifest": args.manifest,
           "epochs": args.epochs, "pairs": n_pairs, "beta": beta,
           "created": datetime.now().isoformat(),
           "per_pair": [], "d_chosen": [], "d_rejected": [],
           "e_policy_chosen": [], "e_policy_rejected": [],
           "e_ref_chosen": [], "e_ref_rejected": []}

    with torch.no_grad():
        for epoch in range(args.epochs):
            ds.set_epoch(epoch)
            for idx in range(n_pairs):
                batch = TTADPOModel.batch_to_device(default_collate([ds[idx]]))
                prepared = ttm.prepare_inputs(batch)
                e_policy, e_reference = ttm.errors(batch, prepared=prepared)
                d = (e_policy - e_reference).tolist()
                rec["d_chosen"].append(d[0])
                rec["d_rejected"].append(d[1])
                rec["e_policy_chosen"].append(float(e_policy[0]))
                rec["e_policy_rejected"].append(float(e_policy[1]))
                rec["e_ref_chosen"].append(float(e_reference[0]))
                rec["e_ref_rejected"].append(float(e_reference[1]))
                rec["per_pair"].append({"epoch": epoch, "pair": idx,
                                        "d_chosen": d[0], "d_rejected": d[1]})
            print(f"[probe] epoch {epoch} done "
                  f"(running mean d_chosen={sum(rec['d_chosen'])/len(rec['d_chosen']):+.5f} "
                  f"d_rejected={sum(rec['d_rejected'])/len(rec['d_rejected']):+.5f})")

    def stats(xs):
        xs = sorted(xs)
        n = len(xs)
        mean = sum(xs) / n
        median = xs[n // 2]
        return {"mean": mean, "median": median, "min": xs[0], "max": xs[-1],
                "frac_lt_0": sum(1 for x in xs if x < 0) / n}

    s_ch, s_re = stats(rec["d_chosen"]), stats(rec["d_rejected"])
    rec["summary"] = {"d_chosen": s_ch, "d_rejected": s_re}

    print("\n========== L1 readout ==========")
    print(f"d_chosen   (policy-ref): mean={s_ch['mean']:+.5f} median={s_ch['median']:+.5f} "
          f"improved={s_ch['frac_lt_0']:.0%}   (expect < 0)")
    print(f"d_rejected (policy-ref): mean={s_re['mean']:+.5f} median={s_re['median']:+.5f} "
          f"improved={s_re['frac_lt_0']:.0%}   (expect >= chosen delta)")
    margin_pol = beta * (sum(rec["e_policy_rejected"]) - sum(rec["e_policy_chosen"])) / len(rec["d_chosen"])
    margin_ref = beta * (sum(rec["e_ref_rejected"]) - sum(rec["e_ref_chosen"])) / len(rec["d_chosen"])
    print(f"margin  policy={margin_pol:+.5f}  reference={margin_ref:+.5f}")
    if "SELF-CHECK" in mode:
        ok = abs(s_ch["mean"]) < 5e-3 and abs(s_re["mean"]) < 5e-3
        print(f"SELF-CHECK {'PASS' if ok else 'FAIL'} (|mean deltas| must be < 5e-3)")
    else:
        verdict = "PASS" if (s_ch["mean"] < 0 and s_re["mean"] >= s_ch["mean"]) else "FAIL"
        print(f"L1 verdict: {verdict}")

    out = args.out or (str(Path(args.adapter).with_suffix("")) + "_l1_probe.json"
                       if args.adapter else str(REPO / "experiments/tta/redo68/probe_selfcheck.json"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rec, indent=1))
    print(f"[probe] record -> {out}")


if __name__ == "__main__":
    main()
