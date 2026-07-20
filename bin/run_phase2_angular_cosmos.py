#!/usr/bin/env python3
"""Capture Cosmos Policy paired hidden/action metrics for angular analysis."""

from __future__ import annotations

import argparse, json, os, pathlib, pickle, site, sys, time
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cosmospolicy-numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmospolicy-matplotlib")
os.environ.setdefault("DETERMINISTIC", "True")

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBERO_PLUS = pathlib.Path(os.environ.get("LIBERO_PLUS_PATH", str(ROOT.parent / "LIBERO-plus")))
for item in (str(LIBERO_PLUS), str(ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)
user_site = site.getusersitepackages()
if user_site in sys.path:
    sys.path.remove(user_site)

import numpy as np
import torch
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from cosmos_policy.experiments.robot import cosmos_utils
from cosmos_policy.experiments.robot.cosmos_utils import get_action, get_model, load_dataset_stats
from cosmos_policy.utils.utils import set_seed_everywhere

CHECKPOINT_ROOT = ROOT.parent.parent / "checkpoints"
BASE_MODEL_DIR = CHECKPOINT_ROOT / "Cosmos-Predict2-2B-Video2World"
POLICY_DIR = CHECKPOINT_ROOT / "Cosmos-Policy-LIBERO-Predict2-2B"
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
EPS = 1e-8


def patch_checkpoint_db(policy_dir: pathlib.Path) -> None:
    from cosmos_policy._src.imaginaire.utils import checkpoint_db

    orig = checkpoint_db.get_checkpoint_path
    base_prefix, aloha_uri = (
        "hf://nvidia/Cosmos-Predict2-2B-Video2World/",
        "hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt",
    )

    def get_checkpoint_path_offline(uri):
        uri = str(uri).rstrip("/")
        if uri.startswith(base_prefix): return str(BASE_MODEL_DIR / uri[len(base_prefix):])
        if uri == aloha_uri: return str(policy_dir / "Cosmos-Policy-LIBERO-Predict2-2B.pt")
        return orig(uri)

    checkpoint_db.get_checkpoint_path = get_checkpoint_path_offline


def make_cfg(args) -> SimpleNamespace:
    return SimpleNamespace(
        suite="libero",
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=str(args.policy_dir / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
        config_file="cosmos_policy/config/config.py",
        use_third_person_image=True,
        num_third_person_images=1,
        use_wrist_image=True,
        num_wrist_images=1,
        use_proprio=True,
        flip_images=args.flip_images,
        use_variance_scale=False,
        use_jpeg_compression=True,
        num_denoising_steps_action=args.num_denoising_steps,
        unnormalize_actions=True,
        normalize_proprio=True,
        dataset_stats_path=str(args.policy_dir / "libero_dataset_statistics.json"),
        t5_text_embeddings_path=str(args.policy_dir / "libero_t5_embeddings.pkl"),
        trained_with_image_aug=True,
        chunk_size=16,
        randomize_seed=False,
    )


def load_extra_t5(path: str) -> None:
    if not path:
        return
    if not pathlib.Path(path).exists():
        print(f"Extra T5 embeddings not found, skipping: {path}")
        return
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)
    cosmos_utils.t5_text_embeddings_cache.update(data)
    print(f"Loaded {len(data)} extra T5 embeddings from {path}")


def load_json(path: pathlib.Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path: str, summary_dir: pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    if p.is_absolute():
        return p
    for base in (ROOT, summary_dir, summary_dir.parent):
        candidate = base / p
        if candidate.exists():
            return candidate
    return ROOT / p


def load_episodes(path: pathlib.Path) -> dict[int, dict[str, Any]]:
    return {int(item["episode"]): item for item in load_json(path)}


def discover_pairs(summary_path: pathlib.Path, conditions: set[str], groups: set[str]) -> tuple[dict, list[dict]]:
    summary = load_json(summary_path)
    by_condition = {item["condition"]: item for item in summary["conditions"]}
    pairs = []
    for condition, info in by_condition.items():
        if condition == "clean" or (conditions and condition not in conditions):
            continue
        source_root = pathlib.Path(info.get("source_root", summary_path.parent))
        clean_path = source_root / "clean/episodes.json"
        if not clean_path.exists():
            clean_path = resolve_path(by_condition["clean"]["episodes_path"], summary_path.parent)
        clean_eps = load_episodes(clean_path)
        pert_eps = load_episodes(resolve_path(info["episodes_path"], summary_path.parent))
        for ep, clean in clean_eps.items():
            pert = pert_eps.get(ep)
            if pert is None:
                continue
            group = pair_group(clean["success"], pert["success"])
            if group not in groups:
                continue
            pairs.append({"condition": condition, "group": group, "clean": clean, "pert": pert})
    return summary, pairs


def pair_group(clean_success: bool, pert_success: bool) -> str:
    if clean_success and pert_success:
        return "preserved"
    if clean_success and not pert_success:
        return "flipped"
    if not clean_success and pert_success:
        return "recovery"
    return "both_fail"


def make_env(task_name: str, suite: str, resolution: int) -> OffScreenRenderEnv:
    bddl = pathlib.Path(get_libero_path("bddl_files")) / suite / f"{task_name}.bddl"
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution)
    env.seed(0)
    return env


def load_init_states(task_name: str, suite: str):
    init_root = pathlib.Path(get_libero_path("init_states"))
    init_file = f"{task_name}.pruned_init"
    path = init_root / suite / init_file
    is_newobj = "_add_" in init_file or "_level" in init_file
    if is_newobj and not path.exists():
        path = init_root / "libero_newobj" / suite / init_file
    states = torch.load(str(path), weights_only=False)
    if is_newobj:
        states = states.reshape(1, -1)
    return states


def first_observation(ep: dict[str, Any], args) -> dict[str, np.ndarray]:
    env = make_env(ep["task_name"], ep["suite"], args.env_resolution)
    try:
        if ep.get("deterministic_reset", True):
            set_seed_everywhere(int(ep.get("deterministic_reset_seed", args.reset_seed)))
        env.reset()
        init_task = ep["task_name"] if ep.get("condition") == "objects_layout" else ep.get("base_task", ep["task_name"])
        states = load_init_states(init_task, ep["suite"])
        obs = env.set_init_state(states[int(ep.get("init_state_index", ep["episode"])) % len(states)])
        for _ in range(args.num_warmup):
            obs, _, _, _ = env.step(DUMMY_ACTION)
        primary = obs["agentview_image"]
        wrist = obs["robot0_eye_in_hand_image"]
        if args.flip_images:
            primary = np.flipud(primary)
            wrist = np.flipud(wrist)
        proprio = np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]))
        return {"primary_image": primary, "wrist_image": wrist, "proprio": proprio}
    finally:
        env.close()


def latent_indices_for_libero() -> tuple[list[int], int]:
    return [2, 3], 4


class HiddenCapture:
    def __init__(self, model, layers: list[int], video_indices: list[int], action_index: int):
        self.layers = set(layers)
        self.video_indices = video_indices
        self.action_index = action_index
        self.pass_idx = -1
        self.video: dict[str, torch.Tensor] = {}
        self.action: dict[str, torch.Tensor] = {}
        self.handles = []
        blocks = model.net.blocks
        hook_layers = sorted(self.layers | {0})
        for layer in hook_layers:
            self.handles.append(blocks[layer].register_forward_hook(self._hook(layer)))

    def _hook(self, layer: int):
        def hook(_module, _inputs, output):
            if layer == 0:
                self.pass_idx += 1
            if layer not in self.layers or self.pass_idx % 2 != 0:
                return
            x = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(x) or x.ndim != 5:
                return
            self.video[str(layer)] = x[:, self.video_indices].detach().float().cpu()
            self.action[str(layer)] = x[:, [self.action_index]].detach().float().cpu()
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def parse_layers(spec: list[str], n_layers: int) -> list[int]:
    if spec == ["all"]:
        return list(range(n_layers))
    out = []
    for item in spec:
        if item == "mid":
            out.append(n_layers // 2)
        elif item == "last":
            out.append(n_layers - 1)
        else:
            out.append(int(item))
    return sorted({layer for layer in out if 0 <= layer < n_layers})


def model_forward(cfg, model, stats, obs, instruction: str, seed: int, layers: list[int], args):
    video_idx, action_idx = latent_indices_for_libero()
    cap = HiddenCapture(model, layers, video_idx, action_idx)
    try:
        set_seed_everywhere(args.reset_seed)
        out = get_action(
            cfg, model, stats, obs, instruction, seed=seed,
            randomize_seed=False, num_denoising_steps_action=args.num_denoising_steps,
            generate_future_state_and_value_in_parallel=False,
        )
    finally:
        cap.close()
    return np.asarray(out["actions"], dtype=np.float32), cap.video, cap.action


def flat_cos_angle(a: torch.Tensor, b: torch.Tensor) -> tuple[float | None, float | None]:
    aa = a.float().reshape(-1)
    bb = b.float().reshape(-1)
    denom = torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb)
    if float(denom) <= EPS:
        return None, None
    cos = float(torch.dot(aa, bb).div(denom))
    cos = max(-1.0, min(1.0, cos))
    return cos, float(np.degrees(np.arccos(cos)))


def compute_metrics(clean_action, pert_action, clean_video, pert_video, clean_act_hid, pert_act_hid):
    metrics: dict[str, Any] = {
        "cos_video": {}, "angle_video_deg": {},
        "cos_action_hidden": {}, "angle_action_hidden_deg": {},
    }
    for layer in sorted(set(clean_video) & set(pert_video), key=int):
        cos, angle = flat_cos_angle(clean_video[layer], pert_video[layer])
        metrics["cos_video"][layer] = cos
        metrics["angle_video_deg"][layer] = angle
    for layer in sorted(set(clean_act_hid) & set(pert_act_hid), key=int):
        cos, angle = flat_cos_angle(clean_act_hid[layer], pert_act_hid[layer])
        metrics["cos_action_hidden"][layer] = cos
        metrics["angle_action_hidden_deg"][layer] = angle
    ca = clean_action.reshape(-1).astype(np.float64)
    pa = pert_action.reshape(-1).astype(np.float64)
    metrics["action_error"] = float(np.linalg.norm(pa - ca) / (np.linalg.norm(ca) + EPS))
    metrics["action_mse"] = float(np.mean((pa - ca) ** 2))
    cos = float(np.dot(ca, pa) / ((np.linalg.norm(ca) * np.linalg.norm(pa)) + EPS))
    cos = max(-1.0, min(1.0, cos))
    metrics["action_cos"] = cos
    metrics["action_angle_deg"] = float(np.degrees(np.arccos(cos)))
    return metrics


def save_pair(out_dir: pathlib.Path, clean, pert, clean_action, pert_action, cv, pv, ca, pa, metrics) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "action_clean.npy", clean_action)
    np.save(out_dir / "action_pert.npy", pert_action)
    torch.save(cv, out_dir / "hidden_video_clean.pt")
    torch.save(pv, out_dir / "hidden_video_pert.pt")
    torch.save(ca, out_dir / "hidden_action_clean.pt")
    torch.save(pa, out_dir / "hidden_action_pert.pt")
    payload = {
        **metrics,
        "condition": pert["condition"],
        "group": pair_group(clean["success"], pert["success"]),
        "episode": int(pert["episode"]),
        "init_state_index_clean": int(clean.get("init_state_index", clean["episode"])),
        "init_state_index_pert": int(pert.get("init_state_index", pert["episode"])),
        "clean_success": bool(clean["success"]),
        "pert_success": bool(pert["success"]),
        "clean_task_name": clean["task_name"],
        "pert_task_name": pert["task_name"],
        "clean_instruction": clean["language"],
        "pert_instruction": pert["language"],
        "instruction_mode": pert.get("instruction_mode", "task"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    args.policy_dir = pathlib.Path(args.policy_dir)
    patch_checkpoint_db(args.policy_dir)
    summary_path = pathlib.Path(args.summary)
    summary, pairs = discover_pairs(summary_path, set(args.conditions or []), set(args.groups))
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    print(f"Discovered {len(pairs)} pairs from {summary_path}")
    if not pairs:
        print("No pairs selected; exiting before model load.")
        return

    cfg = make_cfg(args)
    set_seed_everywhere(args.seed)
    cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    load_extra_t5(args.t5_extra_embeddings)
    stats = load_dataset_stats(cfg.dataset_stats_path)
    model, _ = get_model(cfg)
    layers = parse_layers(args.layers, len(model.net.blocks))
    print(f"Capturing layers: {layers}")

    out_root = pathlib.Path(args.output_dir)
    records = []
    start = time.time()
    for idx, pair in enumerate(pairs, 1):
        clean, pert = pair["clean"], pair["pert"]
        cond, ep = pair["condition"], int(pert["episode"])
        print(f"[{idx}/{len(pairs)}] {cond}/ep{ep:02d} {pair['group']}")
        clean_obs = first_observation(clean, args)
        pert_obs = clean_obs if cond == "language_instructions" else first_observation(pert, args)
        clean_instr = clean["language"]
        pert_instr = pert["language"] if pert.get("instruction_mode") == "strict" else clean_instr
        clean_a, clean_v, clean_h = model_forward(cfg, model, stats, clean_obs, clean_instr, args.seed, layers, args)
        pert_a, pert_v, pert_h = model_forward(cfg, model, stats, pert_obs, pert_instr, args.seed, layers, args)
        metrics = compute_metrics(clean_a, pert_a, clean_v, pert_v, clean_h, pert_h)
        pair_dir = out_root / cond / f"ep{ep:02d}"
        save_pair(pair_dir, clean, pert, clean_a, pert_a, clean_v, pert_v, clean_h, pert_h, metrics)
        records.append({"condition": cond, "episode": ep, "group": pair["group"], **metrics})
        torch.cuda.empty_cache()

    overall = {
        "suite": summary["suite"],
        "base_task": summary["base_task"],
        "summary": str(summary_path),
        "layers": layers,
        "num_pairs": len(records),
        "elapsed_s": time.time() - start,
        "pairs": records,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(f"Saved: {out_root / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    default_summary = ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it__all_conditions__20pair_combined_summary.json"
    p.add_argument("--summary", default=str(default_summary))
    p.add_argument("--output-dir", default=str(ROOT / "experiments/phase2_angular_cosmos/kitchen_scene4_seed7_all_20case"))
    p.add_argument("--policy-dir", default=str(POLICY_DIR))
    p.add_argument("--t5-extra-embeddings", default=str(ROOT / "experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_language_t5.pkl"))
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--groups", nargs="*", default=["preserved", "flipped", "recovery"])
    p.add_argument("--layers", nargs="+", default=["0", "mid", "last"])
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reset-seed", type=int, default=0)
    p.add_argument("--num-warmup", type=int, default=10)
    p.add_argument("--num-denoising-steps", type=int, default=5)
    p.add_argument("--env-resolution", type=int, default=256)
    p.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    return p


if __name__ == "__main__":
    main()
