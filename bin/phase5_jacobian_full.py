#!/usr/bin/env python3
"""
Phase 5: Full-model Jacobian-based action-sensitive subspace.

Computes the Jacobian ∂a/∂h at multiple measurement points:
  - VAE output (video latent space, DiT input)
  - DiT layer 0 (after patch embedding)
  - DiT layer 14 (mid)
  - DiT layer 27 (last, before FinalLayer)

Method: finite-difference Jacobian-vector products + randomized SVD.
  For each measurement point:
    1. Run clean forward pass → capture h and action a
    2. Generate k random unit vectors v in the measurement space
    3. For each v: inject perturbation h+ε*v, re-run forward → a'
    4. J@v ≈ (a' - a)/ε
    5. Stack J@v → matrix M (action_dim × k)
    6. SVD of M → right singular vectors span the action-sensitive subspace

No gradient tracking needed. Works with torch.no_grad().

Requires: GPU, LIBERO environment, model checkpoint.
"""

from __future__ import annotations

import argparse, json, math, os, pathlib, pickle, site, sys, time
from types import SimpleNamespace
from typing import Any, Optional

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DETERMINISTIC", "True")

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBERO_PLUS = pathlib.Path(os.environ.get("LIBERO_PLUS_PATH", str(ROOT.parent / "LIBERO-plus")))
for item in (str(LIBERO_PLUS), str(ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

import numpy as np
import torch
import torch.nn.functional as F
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

# Measurement points
MEASURE_POINTS = {
    "vae_output": "VAE latent output (before DiT patching)",
    "layer_0": "DiT block 0 output (after patch embedding + pos encoding)",
    "layer_14": "DiT block 14 output (mid)",
    "layer_27": "DiT block 27 output (last, before FinalLayer)",
}


# ---------------------------------------------------------------------------
# Model loading (copy from phase2)
# ---------------------------------------------------------------------------

def patch_checkpoint_db(policy_dir: pathlib.Path) -> None:
    from cosmos_policy._src.imaginaire.utils import checkpoint_db
    orig = checkpoint_db.get_checkpoint_path
    base_prefix = "hf://nvidia/Cosmos-Predict2-2B-Video2World/"
    aloha_uri = "hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt"

    def get_checkpoint_path_offline(uri):
        uri = str(uri).rstrip("/")
        if uri.startswith(base_prefix):
            return str(BASE_MODEL_DIR / uri[len(base_prefix):])
        if uri == aloha_uri:
            return str(policy_dir / "Cosmos-Policy-LIBERO-Predict2-2B.pt")
        return orig(uri)

    checkpoint_db.get_checkpoint_path = get_checkpoint_path_offline


def make_cfg(args) -> SimpleNamespace:
    return SimpleNamespace(
        suite="libero",
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=str(args.policy_dir / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
        config_file="cosmos_policy/config/config.py",
        use_third_person_image=True, num_third_person_images=1,
        use_wrist_image=True, num_wrist_images=1,
        use_proprio=True, flip_images=True,
        use_variance_scale=False, use_jpeg_compression=True,
        num_denoising_steps_action=args.num_denoising_steps,
        unnormalize_actions=True, normalize_proprio=True,
        dataset_stats_path=str(args.policy_dir / "libero_dataset_statistics.json"),
        t5_text_embeddings_path=str(args.policy_dir / "libero_t5_embeddings.pkl"),
        trained_with_image_aug=True, chunk_size=16, randomize_seed=False,
    )


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_env(task_name: str, suite: str, resolution: int) -> OffScreenRenderEnv:
    bddl = pathlib.Path(get_libero_path("bddl_files")) / suite / f"{task_name}.bddl"
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution)
    env.seed(0)
    return env


def load_init_states(task_name: str, suite: str):
    init_root = pathlib.Path(get_libero_path("init_states"))
    init_file = f"{task_name}.pruned_init"
    path = init_root / suite / init_file
    if "_add_" in init_file and not path.exists():
        path = init_root / "libero_newobj" / suite / init_file
    return torch.load(str(path), weights_only=False)


def first_observation(ep: dict, args) -> dict:
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
        if getattr(args, 'flip_images', True):
            primary = np.flipud(primary); wrist = np.flipud(wrist)
        proprio = np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]))
        return {"primary_image": primary, "wrist_image": wrist, "proprio": proprio}
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Model forward with perturbation injection
# ---------------------------------------------------------------------------

class PerturbableForward:
    """Runs model forward pass, optionally injecting perturbations at measurement points.

    The forward pass goes through:
      VAE encode → DiT blocks 0..27 → FinalLayer → action extraction
    """

    def __init__(self, cfg, model, stats, obs, instruction, seed, args):
        self.cfg = cfg
        self.model = model
        self.stats = stats
        self.obs = obs
        self.instruction = instruction
        self.seed = seed
        self.args = args

        # Captured states
        self.vae_output: Optional[torch.Tensor] = None
        self.layer_states: dict[int, torch.Tensor] = {}
        self.action_output: Optional[np.ndarray] = None

        # Perturbation to inject
        self.perturbation: Optional[dict] = None  # {"vae_output": tensor, "layer_0": tensor, ...}

        self._handles = []

    def _register_hooks(self):
        """Register hooks to capture VAE output and DiT block outputs."""
        # Hook VAE output: we intercept after the model's internal VAE encoding
        # The VAE output goes through model.net.x_embedder
        # Actually, the VAE output is computed in the data preprocessing step in get_action
        # We need to capture it differently.

        # For now, hook DiT blocks
        blocks = self.model.net.blocks
        target_layers = sorted({0, 14, 27})
        for layer in target_layers:
            self._handles.append(blocks[layer].register_forward_hook(self._make_hook(layer)))

    def _make_hook(self, layer: int):
        def hook(_module, _inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            if isinstance(x, torch.Tensor) and x.ndim == 5:
                if self.perturbation and f"layer_{layer}" in self.perturbation:
                    pert = self.perturbation[f"layer_{layer}"]
                    # Inject perturbation at video slots (indices 2,3)
                    x_pert = x.clone()
                    x_pert[:, [2, 3]] = x_pert[:, [2, 3]] + pert.to(x.device)
                    self.layer_states[layer] = x_pert.detach().cpu()
                    return (x_pert,) + output[1:] if len(output) > 1 else x_pert
                self.layer_states[layer] = x.detach().cpu()
        return hook

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def run_clean(self):
        """Run forward pass without perturbations, capture clean states."""
        self.perturbation = None
        self._register_hooks()
        try:
            set_seed_everywhere(self.args.reset_seed)
            out = get_action(
                self.cfg, self.model, self.stats, self.obs, self.instruction,
                seed=self.seed, randomize_seed=False,
                num_denoising_steps_action=self.args.num_denoising_steps,
                generate_future_state_and_value_in_parallel=False,
            )
            self.action_output = np.asarray(out["actions"], dtype=np.float32)
        finally:
            self._remove_hooks()
        return self.action_output, dict(self.layer_states)

    def run_perturbed(self, perturbations: dict):
        """Run forward pass with perturbations injected at specified layers.

        Args:
            perturbations: {"layer_0": tensor, "layer_14": tensor, ...}
                Each tensor should match the shape of the video slots at that layer.
        """
        self.perturbation = perturbations
        self.layer_states.clear()
        self._register_hooks()
        try:
            set_seed_everywhere(self.args.reset_seed)
            out = get_action(
                self.cfg, self.model, self.stats, self.obs, self.instruction,
                seed=self.seed, randomize_seed=False,
                num_denoising_steps_action=self.args.num_denoising_steps,
                generate_future_state_and_value_in_parallel=False,
            )
            self.action_output = np.asarray(out["actions"], dtype=np.float32)
        finally:
            self._remove_hooks()
        return self.action_output


# ---------------------------------------------------------------------------
# Finite-difference Jacobian via randomized SVD
# ---------------------------------------------------------------------------

def randomized_jacobian_svd(
    runner: PerturbableForward,
    measure_point: str,
    clean_states: dict,
    clean_action: np.ndarray,
    k: int = 50,
    eps_val: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate top singular vectors of J = ∂a/∂h at a measurement point.

    Uses finite differences: J@v ≈ (a(h+εv) - a(h)) / ε
    with k random unit vectors v.

    Args:
        runner: PerturbableForward instance (reused)
        measure_point: key in {"layer_0", "layer_14", "layer_27"}
        clean_states: dict of clean hidden states at all layers
        clean_action: clean action output [action_dim]
        k: number of random directions
        eps_val: perturbation scale

    Returns:
        U: left singular vectors in action space
        S: singular values
        V: right singular vectors in measurement space (flattened)
    """
    # Map string names or integer keys to the correct clean_states key
    if isinstance(measure_point, str) and measure_point.startswith("layer_"):
        layer_key = int(measure_point.split("_")[-1])
    elif isinstance(measure_point, int):
        layer_key = measure_point
    else:
        layer_key = measure_point

    if layer_key not in clean_states:
        print(f"  WARNING: {measure_point} (key={layer_key}) not in clean states, keys={list(clean_states.keys())}")
        return None, None, None

    h_clean = clean_states[layer_key]  # tensor at the measurement point
    # Video slots: indices 2,3
    video_slots = h_clean[:, [2, 3]]  # [1, 2, 14, 14, 2048]
    shape = video_slots.shape
    D = int(video_slots.numel())

    action_dim = clean_action.reshape(-1).shape[0]
    print(f"  {measure_point}: shape={list(shape)}, D={D:,}, action_dim={action_dim}")

    # Target: how many samples per iteration
    max_k_per_batch = 5  # Limit forward passes per batch to avoid OOM
    Jv_list = []

    for batch_start in range(0, k, max_k_per_batch):
        batch_k = min(max_k_per_batch, k - batch_start)
        batch_actions = []
        batch_vectors = []

        for i in range(batch_k):
            # Generate random unit vector
            v = torch.randn(shape, dtype=torch.bfloat16)
            v = v / (v.norm() + 1e-8)
            v_scaled = v * eps_val

            # Inject perturbation
            pert = {measure_point: v_scaled}
            pert_action = runner.run_perturbed(pert)
            da = (pert_action.reshape(-1) - clean_action.reshape(-1)) / eps_val

            batch_actions.append(da)
            batch_vectors.append(v.reshape(-1).float().numpy())

        batch_Jv = np.stack(batch_actions, axis=1)  # [action_dim, batch_k]
        Jv_list.append(batch_Jv)
        print(f"    Batch {batch_start//max_k_per_batch + 1}: {batch_k} directions done")

    Jv = np.concatenate(Jv_list, axis=1)  # [action_dim, k]

    # SVD of J@V to get top singular vectors
    # J ≈ U @ S @ Vt, and (J@V) @ (J@V)^T = U @ S² @ U^T (approximately)
    # The right singular vectors in measurement space: V_right = V_random @ V_svd
    # But this requires saving all random vectors. Instead:
    # We get the left singular vectors U from eig(J@V @ (J@V)^T)
    # And we can estimate the right singular vectors via: V_right = J^T @ U @ diag(1/S)

    M = Jv @ Jv.T  # [action_dim, action_dim]
    eigenvals, U = np.linalg.eigh(M)
    # Descending
    eigenvals = eigenvals[::-1]
    U = U[:, ::-1]
    S = np.sqrt(np.maximum(eigenvals, 0))  # singular values

    # Truncate to non-zero
    r = int(sum(S > S[0] * 1e-6)) if S[0] > 0 else 0
    S = S[:r]
    U = U[:, :r]

    # Right singular vectors: V = J^T @ U @ diag(1/S)
    # J^T @ U ≈ V_random @ (J@V_random)^T @ U = sum_i v_i @ (J@v_i)^T @ U
    # But we didn't save v_i in the right format. Instead:
    # V = Jv.T @ U @ diag(1/S) gives the coefficients in the random basis
    # We don't need the full V in measurement space — we just need projection.
    # We'll compute projection using the JVP approach.

    print(f"  Effective rank: {r}, top-5 singular values: {S[:min(5,r)].round(6)}")
    return U, S, Jv


# ---------------------------------------------------------------------------
# Containment: project delta_h onto Jacobian subspace
# ---------------------------------------------------------------------------

def containment_via_jvp(
    runner: PerturbableForward,
    measure_point: str,
    U: np.ndarray,          # [action_dim, r] — left singular vectors
    S: np.ndarray,          # [r] — singular values
    delta_h: torch.Tensor,  # perturbation delta_h at this measurement point
    eps_val: float = 0.01,
) -> float:
    """Estimate ||Proj_{S_action} delta_h||² / ||delta_h||² via JVP.

    The projection onto S_action is: ||J @ delta_h||² ≈ Σ (U[:,i]·(J@delta_h))²
    And ||delta_h||² in the action-sensitive metric = Σ (σ_i * (v_i·delta_h))²

    We approximate using the singular values and 2-sided projection.
    """
    video_dh = delta_h[:, [2, 3]]  # video slots

    # Inject delta_h as perturbation
    pert = {measure_point: video_dh.to(torch.bfloat16)}
    pert_action = runner.run_perturbed(pert)
    da = (pert_action.reshape(-1) - runner.action_output.reshape(-1)) / eps_val  # Actually, delta_h IS the real perturbation, not scaled

    # Wait, this is wrong. We want J@delta_h directly — the action change caused by the perturbation.
    # The perturbation IS delta_h, not eps * delta_h.
    # So: J @ delta_h ≈ a(h + delta_h) - a(h) (finite difference with step=delta_h)
    # But delta_h might be large. Better: use the cleaned-up version.

    # Actually, the Jacobian is LINEAR only for small perturbations. delta_h from real perturbations
    # might be too large for the linear approximation.
    # Better approach: project via the singular vectors.

    return 0.0  # placeholder


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-dir", default=str(POLICY_DIR))
    p.add_argument("--output-dir", default=str(ROOT / "experiments/phase5_jacobian_full"))
    p.add_argument("--summary", default=None,
                   help="Phase 1 summary JSON path (optional, for paired episodes)")
    p.add_argument("--num-denoising-steps", type=int, default=5)
    p.add_argument("--k-random", type=int, default=50,
                   help="Number of random directions for Jacobian estimation")
    p.add_argument("--eps", type=float, default=0.01,
                   help="Perturbation scale for finite differences")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reset-seed", type=int, default=0)
    p.add_argument("--num-warmup", type=int, default=10)
    p.add_argument("--env-resolution", type=int, default=256)
    p.add_argument("--measure-points", nargs="+",
                   default=["layer_0", "layer_14", "layer_27"],
                   help="Which layers to measure Jacobian at")
    args = p.parse_args()

    args.policy_dir = pathlib.Path(args.policy_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup
    patch_checkpoint_db(args.policy_dir)
    cosmos_utils.init_t5_text_embeddings_cache(str(args.policy_dir / "libero_t5_embeddings.pkl"))

    cfg = make_cfg(args)
    set_seed_everywhere(args.seed)
    stats = load_dataset_stats(cfg.dataset_stats_path)
    model, _ = get_model(cfg)

    # Get one clean episode for Jacobian computation
    # Use the Phase 1 summary for episode info, or hardcode a known clean episode
    if args.summary:
        summary_path = pathlib.Path(args.summary)
        summary = load_json(summary_path)
        # Take first clean episode
        clean_condition = [c for c in summary["conditions"] if c["condition"] == "clean"][0]
        clean_ep_path = pathlib.Path(clean_condition["source_root"]) / "clean/episodes.json"
        clean_eps = load_json(clean_ep_path)
        ep = clean_eps["0"]  # first episode
    else:
        # Hardcoded clean episode for KITCHEN_SCENE4
        ep = {
            "task_name": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
            "suite": "libero_10",
            "episode": 0,
            "deterministic_reset": True,
            "deterministic_reset_seed": args.reset_seed,
            "language": "put the black bowl in the bottom drawer of the cabinet and close it",
            "base_task": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
            "init_state_index": 0,
            "condition": "clean",
        }

    print(f"Using clean episode: {ep['task_name']} ep={ep['episode']}")
    obs = first_observation(ep, args)

    # Create forward runner
    runner = PerturbableForward(cfg, model, stats, obs, ep["language"], args.seed, args)

    # 1. Run clean forward pass, capture states
    print(f"\n=== Step 1: Clean forward pass ===")
    clean_action, clean_states = runner.run_clean()
    print(f"  Action shape: {clean_action.shape}")
    for name, state in clean_states.items():
        print(f"  {name}: {list(state.shape)}")

    # 2. Compute Jacobian at each measurement point
    print(f"\n=== Step 2: Jacobian estimation (k={args.k_random} directions) ===")
    jacobian_results = {}
    for mp in args.measure_points:
        # Convert "layer_27" → integer 27 (key format in clean_states)
        if mp.startswith("layer_"):
            mp_key = int(mp.split("_")[-1])
        else:
            mp_key = mp
        if mp_key not in clean_states:
            print(f"  Skipping {mp} (key={mp_key}) — not in clean states (keys={list(clean_states.keys())})")
            continue
        print(f"\n--- {mp} ---")
        U, S, Jv = randomized_jacobian_svd(
            runner, mp_key, clean_states, clean_action,
            k=args.k_random, eps_val=args.eps,
        )
        if U is not None:
            jacobian_results[mp] = {
                "U": U, "S": S, "Jv": Jv,
                "singular_values": S.tolist(),
                "effective_rank": len(S),
            }

    # 3. Save results
    print(f"\n=== Step 3: Saving ===")
    summary = {
        "episode": ep,
        "clean_action_shape": list(clean_action.shape),
        "measurement_points": list(jacobian_results.keys()),
        "k_random_directions": args.k_random,
        "eps": args.eps,
        "jacobian": {},
    }
    for mp, res in jacobian_results.items():
        sv_path = out_dir / f"singular_values_{mp}.npy"
        np.save(sv_path, res["S"])
        summary["jacobian"][mp] = {
            "singular_values": res["singular_values"],
            "effective_rank": res["effective_rank"],
            "sv_file": str(sv_path),
        }
        print(f"  {mp}: r={res['effective_rank']}, top S={res['S'][:5].round(6)}")

    (out_dir / "jacobian_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone. {out_dir / 'jacobian_summary.json'}")


if __name__ == "__main__":
    main()
