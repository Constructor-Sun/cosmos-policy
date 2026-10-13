"""Attention-location capture for Cosmos-Policy (LIBERO DiT).

Patches ``Attention.compute_attention`` (class level).  The stock TE / flash /
SDPA kernels never materialize attention weights, so the patch recomputes the
softmax itself on the post-RoPE q/k that reach ``compute_attention``; outputs
still come from the original op, so model behaviour is unchanged.

Only the self-attention of the last N DiT blocks is recorded, and only the
query rows of the *action* latent frame (LIBERO 9-frame layout: frame 4).  The
buffer is keep-latest, so when ``get_action()`` returns it holds the final
denoising step.  Per control step, the action-rows x obs-frame-columns block is
reduced to a square token grid per camera and saved as a JET overlay PNG.

Installed via smoke_entry.py (SMOKE_PYTHON_SCRIPT override); see README.md.
"""

import functools
import os
from pathlib import Path

import cv2
import numpy as np
import torch

ST = {
    "installed": False,
    "n_blocks": 4,
    "targets": {},          # id(self_attn module) -> block index
    "cross_targets": {},    # id(cross_attn module) -> block index
    "orig": None,           # original Attention.compute_attention
    "t_total": 9,           # latent frames in the DiT sequence (state_t)
    "action_t": 4,          # action latent frame index
    "obs_frames": (2, 3),   # (wrist, primary) frame indices
    "buf": {},              # block idx -> (n_tokens_per_frame, S) fp32, keep-latest
    "lang_buf": {},         # block idx -> (S,) fp32, cross-attn text-grounding, keep-latest
    "step": 0,              # control-step counter (one per get_action call)
    "every": 1,
    "out_dir": Path("outputs/attn_viz"),
    "warned": set(),
    # intervention: "white"|"yellow"|"blank"|None; action-row attention mixed with
    # a uniform distribution over the region's key tokens (lambda weight).
    "intervene": None,
    "v_scale": 3.0,         # value-boost factor when intervene == "value"
    "region2_bbox": None,   # partner region for vswap
    "region2_tokens": None,
    "lam": 0.5,
    "region_bbox": None,    # "x1,y1,x2,y2" in model-orientation pixels
    "region_tokens": None,  # flat key-token indices of the steered region (computed lazily)
    "proprio_src": None,     # dir with stepNNNN_proprio.npy for the proprio swap
}


def _note_once(msg):
    if msg not in ST["warned"]:
        ST["warned"].add(msg)
        print(f"[attn_viz] {msg}", flush=True)


def _patched_compute_attention(self, q, k, v, *args, **kwargs):
    orig = ST["orig"]
    try:
        idx = ST["targets"].get(id(self))
        if idx is not None:
            B, S, Hh, D = q.shape
            n = S // ST["t_total"]
            if S % ST["t_total"] == 0 and n > 0:
                if ST["intervene"] and ST["region_tokens"] is None and ST["region_bbox"]:

                    def _region_tokens(bbox):
                        x1, y1, x2, y2 = bbox
                        return [
                            ST["obs_frames"][1] * n + r * side + c
                            for r in range(side)
                            for c in range(side)
                            if x1 - 4 <= (6 + (c + 0.5) * 16 * 212 / 224) * 256 / 224 <= x2 + 4
                            and y1 - 4 <= (6 + (r + 0.5) * 16 * 212 / 224) * 256 / 224 <= y2 + 4
                        ]

                    side = int(round(n**0.5))
                    if side * side == n:
                        ST["region_tokens"] = _region_tokens(ST["region_bbox"])
                        if ST.get("region2_bbox"):
                            ST["region2_tokens"] = _region_tokens(ST["region2_bbox"])
                        _note_once(
                            f"intervene '{ST['intervene']}': {len(ST['region_tokens'])} tokens "
                            f"(+{len(ST.get('region2_tokens') or [])} partner), lam={ST['lam']}"
                        )
                t = ST["action_t"]
                rows = slice(t * n, (t + 1) * n)
                qf = q[:, rows].transpose(1, 2).float()                       # (B,H,n,D)
                kf = k.transpose(1, 2).transpose(-1, -2).float()               # (B,H,D,S)
                attn = torch.softmax(qf @ kf * (D**-0.5), dim=-1)              # (B,H,n,S)
                if ST["intervene"] and ST["region_tokens"]:
                    if ST["intervene"] == "value":
                        # V-level boost: amplify the region tokens' value content for
                        # ALL readers (every query row, every head) — attention
                        # weights untouched.  Discriminates content-pathway vs
                        # readout-steering.
                        v2 = v.clone()
                        v2[:, ST["region_tokens"]] = v2[:, ST["region_tokens"]] * ST["v_scale"]
                        return orig(self, q, k, v2, *args, **kwargs)
                    if ST["intervene"] == "vswap":
                        # swap V content between two regions (positions unchanged;
                        # RoPE lives on Q/K): tests content-vs-position as the
                        # carrier of target selection.  Pair tokens index-wise up
                        # to min(len) to avoid shape mismatch.
                        if ST["region2_tokens"]:
                            r1 = ST["region_tokens"]
                            r2 = ST["region2_tokens"]
                            n_pair = min(len(r1), len(r2))
                            v2 = v.clone()
                            v2[:, r1[:n_pair]] = v[:, r2[:n_pair]]
                            v2[:, r2[:n_pair]] = v[:, r1[:n_pair]]
                            return orig(self, q, k, v2, *args, **kwargs)
                    tgt = torch.zeros_like(attn)
                    tgt[..., ST["region_tokens"]] = 1.0 / len(ST["region_tokens"])
                    attn = (1.0 - ST["lam"]) * attn + ST["lam"] * tgt
                    # splice: recompute action-row outputs with the mixed attention.
                    # output_proj is linear (single bias), so splicing projected rows
                    # of a convex combination is exact.
                    new_h = (attn @ v.transpose(1, 2).float())                 # (B,H,n,D)
                    new_h = new_h.transpose(1, 2).reshape(B, n, Hh * D)
                    new_out = self.output_proj(new_h.to(v.dtype))
                    out = orig(self, q, k, v, *args, **kwargs)
                    out[:, rows] = new_out.to(out.dtype)
                    ST["buf"][idx] = attn[0].mean(0).cpu().numpy()
                    return out
                ST["buf"][idx] = attn[0].mean(0).cpu().numpy()                # head avg
        xidx = ST["cross_targets"].get(id(self))
        if xidx is not None:
            # cross-attn: q from all tokens, k from T5 text. Per query token take
            # the max attention over text keys (robust to padding), then head avg.
            B, Sx, Hh, D = q.shape
            qf = q.transpose(1, 2).float()                                       # (B,H,Sx,D)
            kf = k.transpose(1, 2).transpose(-1, -2).float()                     # (B,H,D,Stext)
            attn = torch.softmax(qf @ kf * (D**-0.5), dim=-1)                    # (B,H,Sx,Stext)
            ST["lang_buf"][xidx] = attn[0].max(dim=-1).values.mean(0).cpu().numpy()  # (Sx,)
    except Exception as exc:  # never break the eval
        try:
            shapes = f"q={tuple(q.shape)} k={tuple(k.shape)}"
        except Exception:
            shapes = "?"
        _note_once(f"capture failed ({type(exc).__name__}: {exc}) [{shapes}]")
    return orig(self, q, k, v, *args, **kwargs)


def _ensure_installed(model):
    if ST["installed"]:
        return
    from cosmos_policy._src.predict2.networks.minimal_v4_dit import Attention, MiniTrainDIT

    dit = next((m for m in model.modules() if isinstance(m, MiniTrainDIT)), None)
    if dit is None:
        return  # model not built yet; retry on the next get_action call
    try:
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_latent_indices_from_model_config,
        )

        _cs, ce, fs, _fe = get_latent_indices_from_model_config(model)
        ST["t_total"] = int(model.config.state_t)
        ST["action_t"] = int(fs) - 1
        ST["obs_frames"] = (int(ce) - 1, int(ce))  # layout: [.., proprio, wrist, primary]
    except Exception as exc:
        _note_once(f"frame-index discovery failed ({exc}); using LIBERO defaults")
    blocks = list(dit.blocks)
    sel = ST.get("block_sel", "last")
    n = ST["n_blocks"]
    if sel == "all":
        chosen = blocks
    elif sel == "first":
        chosen = blocks[:n]
    else:
        chosen = blocks[-n:]
    ST["targets"] = {id(b.self_attn): bi for bi, b in enumerate(chosen)}
    ST["cross_targets"] = {id(b.cross_attn): bi for bi, b in enumerate(chosen)}
    ST["orig"] = Attention.compute_attention
    Attention.compute_attention = _patched_compute_attention
    ST["installed"] = True
    print(
        f"[attn_viz] installed on {len(ST['targets'])} blocks: "
        f"t_total={ST['t_total']} action_t={ST['action_t']} obs_frames={ST['obs_frames']}",
        flush=True,
    )


def _norm(m):
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < 1e-8:
        return np.zeros_like(m, dtype=np.float32)
    return ((m - lo) / (hi - lo)).astype(np.float32)


def _overlay(img_rgb, heat01, alpha=0.5):
    h, w = img_rgb.shape[:2]
    small = cv2.resize(heat01, (w, h), interpolation=cv2.INTER_LINEAR)
    jet = cv2.cvtColor(
        cv2.applyColorMap((small * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    out = img_rgb.astype(np.float32) * (1 - alpha) + jet.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _render(obs):
    if not ST["buf"] or (ST["step"] - 1) % ST["every"] != 0:
        _note_once(f"render skipped: buf_blocks={len(ST['buf'])} step={ST['step']} every={ST['every']}")
        return
    _note_once(f"render inputs: obs_keys={sorted(obs.keys()) if hasattr(obs, 'keys') else type(obs)}")
    m = np.stack([ST["buf"][i] for i in sorted(ST["buf"])], axis=0).mean(axis=0)  # (n, S)
    n, S = m.shape
    side = int(round(n**0.5))
    if side * side != n:
        _note_once(f"non-square token grid (n={n}); render skipped")
        return
    cams = {"wrist": ST["obs_frames"][0], "primary": ST["obs_frames"][1]}
    for name, t in cams.items():
        img = obs.get(f"{name}_image")
        if img is None:
            continue
        grid = m[:, t * n : (t + 1) * n].mean(axis=0).reshape(side, side)
        raw_grid = grid.astype(np.float32)  # model orientation (same as rollout videos)
        np.savez(ST["out_dir"] / f"step{ST['step'] - 1:04d}_{name}_grid.npz", grid=raw_grid)
        grid = _norm(raw_grid)
        img = _to_numpy(img).astype(np.uint8)
        overlay = _overlay(img, grid)
        ST["out_dir"].mkdir(parents=True, exist_ok=True)
        out = ST["out_dir"] / f"step{ST['step'] - 1:04d}_{name}.png"
        cv2.imwrite(str(out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    if ST["lang_buf"]:
        try:
            lm = np.stack([ST["lang_buf"][i] for i in sorted(ST["lang_buf"])], axis=0).mean(axis=0)  # (S,)
            n2 = lm.shape[0] // ST["t_total"]
            side2 = int(round(n2**0.5))
            if side2 * side2 == n2:
                for name, t in cams.items():
                    img = obs.get(f"{name}_image")
                    if img is None:
                        continue
                    lg = lm[t * n2 : (t + 1) * n2].reshape(side2, side2).astype(np.float32)
                    np.savez(ST["out_dir"] / f"step{ST['step'] - 1:04d}_{name}_lang.npz", grid=lg)
                    disp = cv2.GaussianBlur(lg, (5, 5), 1.0)  # display-only smoothing
                    lo, hi = np.percentile(disp, [2, 98])
                    disp = np.clip((disp - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32)
                    img_u = _to_numpy(img).astype(np.uint8)
                    overlay = _overlay(img_u, disp)
                    cv2.imwrite(
                        str(ST["out_dir"] / f"step{ST['step'] - 1:04d}_{name}_lang.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                    )
        except Exception as exc:
            _note_once(f"lang render failed ({type(exc).__name__}: {exc})")


def wrap_get_action(every=1, out_dir="outputs/attn_viz", n_blocks=4):
    """Wrap cosmos_utils.get_action in place.  Must run BEFORE run_libero_eval
    is imported, because it binds the name via ``from cosmos_utils import``."""
    import cosmos_policy.experiments.robot.cosmos_utils as cu

    if getattr(cu.get_action, "_attn_viz", False):
        return
    ST["n_blocks"] = int(n_blocks)
    ST["every"] = int(every)
    ST["out_dir"] = Path(out_dir)
    ST["intervene"] = os.environ.get("ATTN_VIZ_INTERVENE") or None
    ST["lam"] = float(os.environ.get("ATTN_VIZ_LAMBDA", "0.5"))
    ST["v_scale"] = float(os.environ.get("ATTN_VIZ_VSCALE", "3.0"))
    ST["block_sel"] = os.environ.get("ATTN_VIZ_BLOCK_SEL", "last")
    ST["proprio_src"] = (
        Path(os.environ["ATTN_VIZ_PROPRIO_SRC"]) if os.environ.get("ATTN_VIZ_PROPRIO_SRC") else None
    )
    ST["region_bbox"] = (
        tuple(float(x) for x in os.environ["ATTN_VIZ_BBOX"].split(","))
        if os.environ.get("ATTN_VIZ_BBOX")
        else None
    )
    ST["region2_bbox"] = (
        tuple(float(x) for x in os.environ["ATTN_VIZ_BBOX2"].split(","))
        if os.environ.get("ATTN_VIZ_BBOX2")
        else None
    )
    orig = cu.get_action

    @functools.wraps(orig)
    def wrapped(cfg, model, dataset_stats, obs, task_label_or_embedding, *a, **kw):
        _ensure_installed(model)
        if ST["intervene"] == "proprio" and ST["proprio_src"] is not None:
            fn = ST["proprio_src"] / f"step{ST['step']:04d}_proprio.npy"
            if fn.exists():
                obs = dict(obs)
                obs["proprio"] = np.load(fn)
                _note_once(f"proprio swapped from {ST['proprio_src'].name} (step>=0)")
        ST["buf"].clear()
        ST["lang_buf"].clear()
        try:
            if "proprio" in obs:
                ST["out_dir"].mkdir(parents=True, exist_ok=True)
                np.save(
                    ST["out_dir"] / f"step{ST['step']:04d}_proprio.npy",
                    np.asarray(obs["proprio"]),
                )
        except Exception as exc:
            _note_once(f"proprio save failed ({type(exc).__name__}: {exc})")
        out = orig(cfg, model, dataset_stats, obs, task_label_or_embedding, *a, **kw)
        try:
            if isinstance(out, dict) and "actions" in out:
                ST["out_dir"].mkdir(parents=True, exist_ok=True)
                np.save(
                    ST["out_dir"] / f"step{ST['step']:04d}_actions.npy",
                    np.asarray(out["actions"]),
                )
        except Exception as exc:
            _note_once(f"action save failed ({type(exc).__name__}: {exc})")
        ST["step"] += 1
        try:
            _render(obs)
        except Exception as exc:
            _note_once(f"render failed ({type(exc).__name__}: {exc})")
        return out

    wrapped._attn_viz = True
    cu.get_action = wrapped
    print(f"[attn_viz] get_action wrapped -> {ST['out_dir']}", flush=True)
