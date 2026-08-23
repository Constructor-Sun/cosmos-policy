"""Stage 1 visualization: render RGB + metric depth (raw and flipped).

Test-only helper.  Produces a 2x2 figure per task:

    [agentview RGB          | metric depth (viridis + colorbar)
     agentview RGB (flipped)| metric depth (flipped)             ]

Usage (cosmospolicy env):

    python tests/stage1_depth/visualize_depth.py \
        --task KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
        --out tests/stage1_depth/artifacts
"""
from __future__ import annotations

import argparse

import numpy as np

from harness import RESOLUTION, create_env, flip_depth, load_task_names, metric_depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None, help="LIBERO-10 task name (default: first manifest task)")
    parser.add_argument("--out", default="tests/stage1_depth/artifacts", help="output directory")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task = args.task or load_task_names()[0]
    env = create_env(task)
    obs = env.reset()

    rgb = np.asarray(obs["agentview_image"])
    d = metric_depth(env, obs)[..., 0]
    rgb_f = np.flipud(rgb)
    d_f = flip_depth(d)

    vmin, vmax = float(np.percentile(d, 1)), float(np.percentile(d, 99))
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    titles = ["agentview RGB", "metric depth (m)", "agentview RGB (flipped)", "metric depth (m, flipped)"]
    ims = [rgb, d, rgb_f, d_f]
    for ax, im, title in zip(axes.ravel(), ims, titles):
        if im.ndim == 3 and im.shape[-1] == 3:
            ax.imshow(im)
        else:
            show = ax.imshow(im, cmap="viridis", vmin=vmin, vmax=vmax)
            fig.colorbar(show, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{task}\ndepth range [{vmin:.3f}, {vmax:.3f}] m (1-99 pct)", fontsize=11)
    fig.tight_layout()

    import os
    from pathlib import Path

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = task.replace("/", "_")
    out_path = out_dir / f"depth_{slug}.png"
    fig.savefig(out_path, dpi=110)
    print(f"saved {out_path}")
    env.close()


if __name__ == "__main__":
    main()
