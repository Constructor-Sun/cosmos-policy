#!/usr/bin/env python
"""Compare cuRobo-sphere and URDF-depth robot removal on LIBERO frames.

This is a standalone diagnostic.  MuJoCo segmentation is used only as an
evaluation oracle; neither filter receives segmentation as input.  For each
requested initial state the script reports:

* robot pixels correctly removed;
* robot pixels left in the planner point cloud;
* non-robot pixels removed by mistake.

It also writes a CSV and an RGB overlay figure.  No production code is changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

logging.disable(logging.CRITICAL)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from memory_system.offline.build_targets import patch_numpy2_segmentation

patch_numpy2_segmentation()

from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import _make_main_depth
from libero.libero import benchmark
from memory_system.execute.curobo_planner import CuroboPlanner
from memory_system.execute.urdf_depth_filter import (
    UrdfDepthFilter,
    UrdfDepthFilterConfig,
)
from memory_system.geometry import camera_params as build_camera_params
from memory_system.geometry import pixel_to_world


DEFAULT_TASK = (
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
    "_view_0_0_100_0_0_initstate_274"
)
DEFAULT_EPISODES_JSON = (
    "scripts/experiments/libero10_robotinit_single/robotinit/"
    "robot_initial_states/episodes.json"
)
DEFAULT_IMAGE = (
    "tests/initial_alignment_investigation/"
    "urdf_vs_spheres_robot_filter_ep1_ep2.png"
)
DEFAULT_3D_IMAGE = (
    "tests/initial_alignment_investigation/"
    "urdf_vs_spheres_robot_filter_ep1_ep2_3d.png"
)
DEFAULT_CSV = (
    "tests/initial_alignment_investigation/"
    "urdf_vs_spheres_robot_filter_ep1_ep2.csv"
)
ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
DEFAULT_TARGET = np.array(
    [-0.0588200204, -0.0857331678, 0.9660174251], dtype=np.float64
)
CATEGORY_COLORS = {
    "robot": "#d62728",
    "bowl": "#ff7f0e",
    "cabinet": "#2ca02c",
    "table": "#9467bd",
    "floor": "#8c564b",
    "other": "#7f7f7f",
}


@dataclass(frozen=True)
class FilterMetrics:
    robot_total: int
    robot_removed: int
    robot_remaining: int
    nonrobot_total: int
    environment_removed: int
    removed_total: int

    @property
    def recall(self) -> float:
        return self.robot_removed / self.robot_total if self.robot_total else float("nan")

    @property
    def precision(self) -> float:
        return self.robot_removed / self.removed_total if self.removed_total else float("nan")


@dataclass(frozen=True)
class EpisodeComparison:
    init_state_index: int
    success: bool | None
    rgb: np.ndarray
    oracle_robot_mask: np.ndarray
    valid_depth_mask: np.ndarray
    sphere_mask: np.ndarray
    urdf_mask: np.ndarray
    sphere_metrics: FilterMetrics
    urdf_metrics: FilterMetrics
    points_world: np.ndarray
    point_categories: np.ndarray


def default_urdf_path() -> Path:
    import curobo

    return (
        Path(curobo.__file__).resolve().parent
        / "content/assets/robot/franka_description/franka_panda.urdf"
    )


def classify_robot(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return (
        "robot" in lowered
        or "gripper" in lowered
        or lowered.startswith("panda")
    )


def classify_category(name: str | None) -> str:
    if classify_robot(name):
        return "robot"
    if not name:
        return "other"
    lowered = name.lower()
    if "bowl" in lowered:
        return "bowl"
    if "cabinet" in lowered:
        return "cabinet"
    if "table" in lowered:
        return "table"
    if "floor" in lowered or "wall" in lowered or "visual" in lowered:
        return "floor"
    return "other"


def segmentation_labels(
    segmentation: np.ndarray, sim
) -> tuple[np.ndarray, np.ndarray]:
    """Convert canonical MuJoCo segmentation into robot and category labels."""

    flat = np.asarray(segmentation).reshape(-1, 2)
    robot = np.zeros(len(flat), dtype=bool)
    categories = np.empty(len(flat), dtype=object)
    for index, (object_type, object_id) in enumerate(flat):
        object_type = int(object_type)
        object_id = int(object_id)
        if object_type == 5:
            name = sim.model.geom_id2name(object_id)
        elif object_type == 6 and hasattr(sim.model, "site_id2name"):
            name = sim.model.site_id2name(object_id)
        else:
            name = None
        category = classify_category(name)
        categories[index] = category
        robot[index] = category == "robot"
    shape = segmentation.shape[:2]
    return robot.reshape(shape), categories.reshape(shape)


def metrics(
    removal_mask: np.ndarray,
    oracle_robot_mask: np.ndarray,
    valid_depth_mask: np.ndarray,
) -> FilterMetrics:
    evaluated_robot = oracle_robot_mask & valid_depth_mask
    evaluated_nonrobot = ~oracle_robot_mask & valid_depth_mask
    removed = removal_mask & valid_depth_mask
    robot_removed = int(np.count_nonzero(removed & evaluated_robot))
    robot_total = int(np.count_nonzero(evaluated_robot))
    return FilterMetrics(
        robot_total=robot_total,
        robot_removed=robot_removed,
        robot_remaining=robot_total - robot_removed,
        nonrobot_total=int(np.count_nonzero(evaluated_nonrobot)),
        environment_removed=int(np.count_nonzero(removed & evaluated_nonrobot)),
        removed_total=int(np.count_nonzero(removed)),
    )


def sphere_removal_mask(
    *,
    depth: np.ndarray,
    camera,
    robot_base_pose: np.ndarray,
    joint_positions: np.ndarray,
    planner: CuroboPlanner,
    motion_planner,
) -> np.ndarray:
    """Reproduce the exact per-depth-pixel sphere filter used by the planner."""

    import torch
    from curobo._src.state.state_joint import JointState

    height, width = depth.shape
    rows, cols = np.meshgrid(
        np.arange(height), np.arange(width), indexing="ij"
    )
    pixels_all = np.stack([rows.ravel(), cols.ravel()], axis=-1)
    valid_flat = (np.isfinite(depth) & (depth > 0.0)).ravel()
    pixels = pixels_all[valid_flat]
    raw_world = pixel_to_world(pixels, depth, camera)

    base_pos = np.asarray(robot_base_pose[:3], dtype=np.float64)
    base_rotation = Rotation.from_quat(
        np.asarray(robot_base_pose[3:], dtype=np.float64)
    ).as_matrix()
    raw_base = (base_rotation.T @ (raw_world - base_pos).T).T

    start = JointState.from_position(
        torch.as_tensor(
            joint_positions, device=planner.device, dtype=torch.float32
        ).reshape(1, -1),
        joint_names=motion_planner.joint_names,
    )
    spheres = (
        motion_planner.compute_kinematics(start)
        .robot_spheres.detach()
        .cpu()
        .numpy()
        .reshape(-1, 4)
    )
    clearance = np.linalg.norm(
        raw_base[:, None, :] - spheres[None, :, :3], axis=2
    ) - spheres[None, :, 3]
    removed_valid = np.any(clearance <= float(planner.robot_padding), axis=1)

    removal_flat = np.zeros(height * width, dtype=bool)
    removal_flat[np.flatnonzero(valid_flat)] = removed_valid
    return removal_flat.reshape(height, width)


def urdf_joint_configuration(obs: dict) -> dict[str, float]:
    arm = np.asarray(obs["robot0_joint_pos"], dtype=np.float64).reshape(-1)
    if len(arm) != len(ARM_JOINT_NAMES):
        raise ValueError(f"expected 7 Franka arm joints, got {len(arm)}")
    configuration = {
        name: float(value) for name, value in zip(ARM_JOINT_NAMES, arm)
    }

    # LIBERO uses opposite signs for the two finger slides; the URDF encodes
    # that direction in each joint axis and expects both displacements positive.
    gripper = np.abs(
        np.asarray(obs.get("robot0_gripper_qpos", [0.04, 0.04]), dtype=np.float64)
    ).reshape(-1)
    if len(gripper) >= 2:
        configuration["panda_finger_joint1"] = float(np.clip(gripper[0], 0.0, 0.04))
        configuration["panda_finger_joint2"] = float(np.clip(gripper[1], 0.0, 0.04))
    return configuration


def load_success_labels(path: str) -> dict[int, bool]:
    episodes_path = Path(path)
    if not episodes_path.exists():
        return {}
    episodes = json.loads(episodes_path.read_text())
    return {
        int(episode["init_state_index"]): bool(episode["success"])
        for episode in episodes
    }


def compare_episode(
    *,
    env,
    state: np.ndarray,
    init_state_index: int,
    success: bool | None,
    planner: CuroboPlanner,
    motion_planner,
    urdf_filter: UrdfDepthFilter,
) -> EpisodeComparison:
    obs = env.set_init_state(state)
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))

    rgb = np.flipud(obs["agentview_image"]).copy()
    height, width = rgb.shape[:2]
    camera = build_camera_params(env.sim, "agentview", height, width)
    depth = _make_main_depth(obs, camera, flip_images=True)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    valid_depth = np.isfinite(depth) & (depth > 0.0)

    segmentation_render, _ = env.sim.render(
        height,
        width,
        camera_name="agentview",
        depth=True,
        segmentation=True,
    )
    oracle_robot, categories = segmentation_labels(
        np.flipud(segmentation_render), env.sim
    )

    rows, cols = np.meshgrid(
        np.arange(height), np.arange(width), indexing="ij"
    )
    pixels_all = np.stack([rows.ravel(), cols.ravel()], axis=-1)
    valid_flat = valid_depth.ravel()
    points_world = pixel_to_world(pixels_all[valid_flat], depth, camera)
    point_categories = categories.ravel()[valid_flat]

    robot_base_pose = np.concatenate(
        [env.robots[0].base_pos, env.robots[0].base_ori]
    ).astype(np.float64)
    joint_positions = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
    spheres = sphere_removal_mask(
        depth=depth,
        camera=camera,
        robot_base_pose=robot_base_pose,
        joint_positions=joint_positions,
        planner=planner,
        motion_planner=motion_planner,
    )
    urdf = urdf_filter.filter(
        depth,
        camera_params=camera,
        joint_positions=urdf_joint_configuration(obs),
        robot_base_pose=robot_base_pose,
    ).robot_mask

    return EpisodeComparison(
        init_state_index=init_state_index,
        success=success,
        rgb=rgb,
        oracle_robot_mask=oracle_robot,
        valid_depth_mask=valid_depth,
        sphere_mask=spheres,
        urdf_mask=urdf,
        sphere_metrics=metrics(spheres, oracle_robot, valid_depth),
        urdf_metrics=metrics(urdf, oracle_robot, valid_depth),
        points_world=points_world,
        point_categories=point_categories,
    )


def error_overlay(
    rgb: np.ndarray,
    removal_mask: np.ndarray,
    oracle_robot_mask: np.ndarray,
    valid_depth_mask: np.ndarray,
) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    overlay = image.copy()
    true_positive = removal_mask & oracle_robot_mask & valid_depth_mask
    false_negative = ~removal_mask & oracle_robot_mask & valid_depth_mask
    false_positive = removal_mask & ~oracle_robot_mask & valid_depth_mask
    alpha = 0.72
    for mask, color in (
        (true_positive, np.array([0.1, 0.9, 0.2])),
        (false_negative, np.array([1.0, 0.05, 0.05])),
        (false_positive, np.array([1.0, 0.85, 0.0])),
    ):
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
    return np.clip(overlay, 0.0, 1.0)


def oracle_overlay(
    rgb: np.ndarray,
    oracle_robot_mask: np.ndarray,
    valid_depth_mask: np.ndarray,
) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    overlay = image.copy()
    mask = oracle_robot_mask & valid_depth_mask
    overlay[mask] = 0.30 * overlay[mask] + 0.70 * np.array([0.9, 0.2, 0.9])
    return np.clip(overlay, 0.0, 1.0)


def metric_title(name: str, value: FilterMetrics) -> str:
    return (
        f"{name}: removed robot={value.robot_removed}/{value.robot_total} "
        f"({100.0 * value.recall:.1f}%)\n"
        f"robot remaining={value.robot_remaining}, environment removed={value.environment_removed}, "
        f"precision={100.0 * value.precision:.1f}%"
    )


def save_figure(comparisons: list[EpisodeComparison], output: Path) -> None:
    figure, axes = plt.subplots(
        len(comparisons),
        3,
        figsize=(18, 5.5 * len(comparisons)),
        squeeze=False,
    )
    for row, comparison in enumerate(comparisons):
        episode = comparison.init_state_index + 1
        success = comparison.success if comparison.success is not None else "?"
        axes[row, 0].imshow(
            oracle_overlay(
                comparison.rgb,
                comparison.oracle_robot_mask,
                comparison.valid_depth_mask,
            )
        )
        oracle_count = int(
            np.count_nonzero(
                comparison.oracle_robot_mask & comparison.valid_depth_mask
            )
        )
        axes[row, 0].set_title(
            f"episode {episode} (init={comparison.init_state_index}, success={success})\n"
            f"MuJoCo robot oracle={oracle_count} depth pixels"
        )

        axes[row, 1].imshow(
            error_overlay(
                comparison.rgb,
                comparison.sphere_mask,
                comparison.oracle_robot_mask,
                comparison.valid_depth_mask,
            )
        )
        axes[row, 1].set_title(metric_title("cuRobo spheres", comparison.sphere_metrics))

        axes[row, 2].imshow(
            error_overlay(
                comparison.rgb,
                comparison.urdf_mask,
                comparison.oracle_robot_mask,
                comparison.valid_depth_mask,
            )
        )
        axes[row, 2].set_title(metric_title("URDF depth", comparison.urdf_metrics))
        for axis in axes[row]:
            axis.axis("off")

    figure.legend(
        handles=[
            Patch(color="#e633e6", label="MuJoCo robot oracle"),
            Patch(color="#1ae633", label="correctly removed robot"),
            Patch(color="#ff0d0d", label="robot left in point cloud"),
            Patch(color="#ffd900", label="non-robot removed by mistake"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Robot removal comparison (segmentation is evaluation-only)", fontsize=16
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_3d_cloud(
    axis,
    comparison: EpisodeComparison,
    keep_points: np.ndarray,
    *,
    title: str,
    limits: tuple[np.ndarray, np.ndarray],
    seed: int,
    max_points_per_category: int = 12_000,
) -> None:
    rng = np.random.default_rng(seed)
    for category, color in CATEGORY_COLORS.items():
        indices = np.flatnonzero(
            keep_points & (comparison.point_categories == category)
        )
        if len(indices) > max_points_per_category:
            indices = rng.choice(indices, max_points_per_category, replace=False)
        if not len(indices):
            continue
        points = comparison.points_world[indices]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=1.4 if category == "robot" else 0.45,
            c=color,
            alpha=0.9 if category == "robot" else 0.55,
            linewidths=0,
            depthshade=False,
        )
    axis.scatter(
        DEFAULT_TARGET[0],
        DEFAULT_TARGET[1],
        DEFAULT_TARGET[2],
        s=110,
        c="black",
        marker="*",
        depthshade=False,
    )
    minimum, maximum = limits
    axis.set_xlim(minimum[0], maximum[0])
    axis.set_ylim(minimum[1], maximum[1])
    axis.set_zlim(minimum[2], maximum[2])
    axis.set_box_aspect(np.maximum(maximum - minimum, 1e-3))
    axis.view_init(elev=30, azim=-60)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_title(title)


def save_3d_figure(
    comparisons: list[EpisodeComparison], output: Path
) -> None:
    figure = plt.figure(figsize=(21, 7 * len(comparisons)))
    for row, comparison in enumerate(comparisons):
        points = comparison.points_world
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        padding = np.maximum((maximum - minimum) * 0.025, 0.01)
        limits = (minimum - padding, maximum + padding)
        valid_flat = comparison.valid_depth_mask.ravel()
        sphere_keep = ~comparison.sphere_mask.ravel()[valid_flat]
        urdf_keep = ~comparison.urdf_mask.ravel()[valid_flat]
        all_points = np.ones(len(points), dtype=bool)
        episode = comparison.init_state_index + 1
        success = comparison.success if comparison.success is not None else "?"

        original_axis = figure.add_subplot(
            len(comparisons), 3, row * 3 + 1, projection="3d"
        )
        plot_3d_cloud(
            original_axis,
            comparison,
            all_points,
            title=(
                f"episode {episode} original (success={success})\n"
                f"robot points={comparison.sphere_metrics.robot_total}"
            ),
            limits=limits,
            seed=episode * 10,
        )

        sphere_axis = figure.add_subplot(
            len(comparisons), 3, row * 3 + 2, projection="3d"
        )
        plot_3d_cloud(
            sphere_axis,
            comparison,
            sphere_keep,
            title=(
                "after cuRobo spheres\n"
                f"robot remaining={comparison.sphere_metrics.robot_remaining}, "
                f"environment removed={comparison.sphere_metrics.environment_removed}"
            ),
            limits=limits,
            seed=episode * 10 + 1,
        )

        urdf_axis = figure.add_subplot(
            len(comparisons), 3, row * 3 + 3, projection="3d"
        )
        plot_3d_cloud(
            urdf_axis,
            comparison,
            urdf_keep,
            title=(
                "after URDF depth\n"
                f"robot remaining={comparison.urdf_metrics.robot_remaining}, "
                f"environment removed={comparison.urdf_metrics.environment_removed}"
            ),
            limits=limits,
            seed=episode * 10 + 2,
        )

    legend = [
        Patch(color=color, label=category)
        for category, color in CATEGORY_COLORS.items()
    ]
    legend.append(
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            linestyle="None",
            markersize=12,
            label="memory target",
        )
    )
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=len(legend),
        frameon=False,
    )
    figure.suptitle(
        "3D point cloud before and after robot removal", fontsize=17
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def metric_rows(comparisons: list[EpisodeComparison]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for comparison in comparisons:
        for method, value in (
            ("curobo_spheres", comparison.sphere_metrics),
            ("urdf_depth", comparison.urdf_metrics),
        ):
            rows.append(
                {
                    "episode": comparison.init_state_index + 1,
                    "init_state_index": comparison.init_state_index,
                    "success": comparison.success,
                    "method": method,
                    "robot_total": value.robot_total,
                    "robot_removed": value.robot_removed,
                    "robot_remaining": value.robot_remaining,
                    "robot_recall": value.recall,
                    "nonrobot_total": value.nonrobot_total,
                    "environment_removed": value.environment_removed,
                    "removed_total": value.removed_total,
                    "removal_precision": value.precision,
                }
            )
    return rows


def save_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default=DEFAULT_TASK)
    parser.add_argument("--init-state-indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--episodes-json", default=DEFAULT_EPISODES_JSON)
    parser.add_argument("--urdf-path", default=None)
    parser.add_argument("--geometry", choices=("visual", "collision"), default="visual")
    parser.add_argument("--abs-tolerance", type=float, default=0.008)
    parser.add_argument("--relative-tolerance", type=float, default=0.005)
    parser.add_argument("--output-image", default=DEFAULT_IMAGE)
    parser.add_argument("--output-3d", default=DEFAULT_3D_IMAGE)
    parser.add_argument("--output-csv", default=DEFAULT_CSV)
    args = parser.parse_args()

    suite = benchmark.get_benchmark_dict()["libero_10"](
        category_value="Robot Initial States"
    )
    matches = [
        (index, task)
        for index, task in enumerate(suite.tasks)
        if task.name == args.task_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"task not found or ambiguous: {args.task_name}")
    task_index, task = matches[0]
    states = suite.get_task_init_states(task_index)
    for index in args.init_state_indices:
        if index < 0 or index >= len(states):
            raise IndexError(f"init state index out of range: {index}")

    urdf_path = Path(args.urdf_path) if args.urdf_path else default_urdf_path()
    if not urdf_path.exists():
        raise FileNotFoundError(f"Franka URDF not found: {urdf_path}")
    urdf_filter = UrdfDepthFilter(
        UrdfDepthFilterConfig(
            urdf_path=str(urdf_path),
            abs_tolerance=args.abs_tolerance,
            relative_tolerance=args.relative_tolerance,
            use_collision_geometry=args.geometry == "collision",
        ),
        default_joint_positions={
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        },
    )
    success_labels = load_success_labels(args.episodes_json)

    env, _ = get_libero_env(
        task,
        "cosmos",
        resolution=256,
        camera_depths=[True, False],
    )
    planner = CuroboPlanner(joint_execution=False)
    comparisons: list[EpisodeComparison] = []
    try:
        import torch
        from curobo._src.geom.types import SceneCfg

        env.reset()
        motion_planner = planner._ensure_planner(SceneCfg())
        print(f"URDF: {urdf_path}")
        print(f"URDF geometry: {args.geometry}")
        print(f"cuRobo joints: {motion_planner.joint_names}")
        print(
            "depth agreement: "
            f"abs={args.abs_tolerance:.4f}m rel={args.relative_tolerance:.4f}"
        )
        for init_state_index in args.init_state_indices:
            comparison = compare_episode(
                env=env,
                state=states[init_state_index],
                init_state_index=init_state_index,
                success=success_labels.get(init_state_index),
                planner=planner,
                motion_planner=motion_planner,
                urdf_filter=urdf_filter,
            )
            comparisons.append(comparison)
            for method, value in (
                ("spheres", comparison.sphere_metrics),
                ("urdf", comparison.urdf_metrics),
            ):
                print(
                    f"episode={init_state_index + 1} init={init_state_index} "
                    f"success={comparison.success} method={method:7s} "
                    f"robot_total={value.robot_total:5d} "
                    f"robot_removed={value.robot_removed:5d} "
                    f"robot_remaining={value.robot_remaining:5d} "
                    f"environment_removed={value.environment_removed:5d} "
                    f"recall={value.recall:.4f} precision={value.precision:.4f}"
                )
        # Keep torch referenced until planning resources have been initialized;
        # this also makes an unavailable torch import fail before any output.
        _ = torch
    finally:
        env.close()

    image_path = Path(args.output_image)
    image_3d_path = Path(args.output_3d)
    csv_path = Path(args.output_csv)
    save_figure(comparisons, image_path)
    save_3d_figure(comparisons, image_3d_path)
    rows = metric_rows(comparisons)
    save_csv(rows, csv_path)
    print(f"saved figure to {image_path}")
    print(f"saved 3D figure to {image_3d_path}")
    print(f"saved metrics to {csv_path}")


if __name__ == "__main__":
    main()
