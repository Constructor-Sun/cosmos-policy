#!/usr/bin/env python
"""Export replay videos for saved phase-check cases (no policy inference).

Replays the actions stored in each case's episode.h5 in the same 256-resolution
depth env the rules were recorded with (physics is identical to the original
rollout: replayed proprio matches the stored stream bitwise), renders the
agentview camera after every action, and writes an mp4 with a step counter and
a green frame border once the step reaches the oracle's Pick confirmation
(when known from phase_timing.json).
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "LIBERO-plus")]
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("COSMOS_SKILL_COMPLETION_SHADOW", "1")
from check_oracle_failures import DEFAULT_ROOTS, load_cases  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

# The Pick late-confirm / rule-missed cases under review (TMP discussion).
DEFAULT_CASES = [
    "tta_phase_check_repro_68/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it/init007",
    "tta_phase_check_repro_68/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it/init010",
    "tta_phase_check_repro_68/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it/init012",
    "tta_phase_check_repro_68/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it/init013",
    "tta_phase_check_repro_68/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it/init015",
    "tta_phase_check_repro_68/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it/init017",
    "tta_phase_check_repro_68/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it/init021",
    "tta_phase_check_repro_68/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket/init017",
    "tta_phase_check_bg5_correct/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it/init008",
    "tta_phase_check_bg5_correct/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it/init012",
]
ORACLE_PICK_TIMING = ROOT / "tests/phase/phase_timing.json"


def oracle_pick_steps(path: Path) -> dict[str, int | None]:
    """Map <dataset>/<task>/<init> -> oracle Pick confirm step, if known."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    out = {}
    for case in payload["cases"]:
        parts = Path(case["case"]).parts
        idx = next(i for i, p in enumerate(parts) if p.startswith("tta_phase_check_"))
        key = "/".join(parts[idx:idx + 3])
        for phase in case["phases"]:
            if phase["skill"] == "Pick" and phase["oracle_confirm"] is not None:
                out[key] = int(phase["oracle_confirm"])
    return out


def annotate(frame: np.ndarray, lines, oracle_step, step,
             t_star=None) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text((4, 4), "\n".join(lines), fill=(255, 255, 255))
    if oracle_step is not None and step >= oracle_step:
        draw.rectangle([0, 0, img.width - 1, img.height - 1], outline=(80, 220, 80), width=3)
    if t_star is not None and step >= t_star:
        draw.rectangle([3, 3, img.width - 4, img.height - 4], outline=(230, 60, 60), width=3)
    return np.asarray(img)


def export_video(env, init_state, actions, out_path: Path, label: str,
                 oracle_step: int | None, t_star: int | None = None) -> None:
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    total = len(actions)
    import imageio
    with imageio.get_writer(out_path, fps=20) as writer:
        frame = np.flipud(np.asarray(obs["agentview_image"]))
        writer.append_data(annotate(frame, [label, "step 0 (settled)"],
                                    oracle_step, 0, t_star))
        for step, action in enumerate(np.asarray(actions, dtype=np.float64)):
            obs, _, done, _ = env.step(action.tolist())
            frame = np.flipud(np.asarray(obs["agentview_image"]))
            note = " success!" if done else ""
            writer.append_data(annotate(
                frame, [label, f"step {step + 1}/{total}{note}"], oracle_step,
                step + 1, t_star))
            if done:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES,
                        help="<dataset>/<task>/<init> relative case selectors")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/tta/phase_check_videos")
    args = parser.parse_args()

    wanted = {c.strip("/") for c in args.cases}
    selected = []
    for dataset, case_dir, record in load_cases(args.roots):
        key = f"{dataset}/{case_dir.parent.name}/{case_dir.name}"
        if key in wanted:
            selected.append((key, case_dir, record))
            wanted.discard(key)
    missing = sorted(wanted)
    if missing:
        print(f"WARNING: no phase_record.json found for: {missing}", flush=True)

    oracle_steps = oracle_pick_steps(ORACLE_PICK_TIMING)
    grouped = defaultdict(list)
    for key, case_dir, record in selected:
        grouped[record["suite_task_name"]].append((key, case_dir, record))

    from replay_phase_rules import make_replay_context
    for name, cases in grouped.items():
        env, init_states, _, _ = make_replay_context(name)
        try:
            for key, case_dir, record in cases:
                with h5py.File(case_dir / "episode.h5") as h5:
                    actions = h5["actions"][:]
                out_path = args.output / f"{key}.mp4"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                oracle_step = oracle_steps.get(key)
                cand = record.get("candidate")
                t_star = None if cand is None else cand.get("t_star")
                label = f"{key.split('/')[-2][:40]}/{key.split('/')[-1]}"
                if oracle_step is not None:
                    label += f"  oracle Pick@{oracle_step}"
                if t_star is not None:
                    cand_txt = "none" if cand is None else (
                        f"{cand['span']['skill']}#{cand['span']['phase_index']} "
                        f"{cand['status']}")
                    label += f"  t*={t_star} ({cand_txt})"
                export_video(env, init_states[record["census_abs_init"]],
                             actions, out_path, label, oracle_step, t_star)
                print(f"wrote {out_path} ({len(actions)} actions, "
                      f"oracle Pick@{oracle_step}, t*={t_star})", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
