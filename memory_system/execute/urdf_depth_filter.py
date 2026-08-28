"""Remove visible robot pixels from metric depth using a posed URDF model.

The memory system stores depth in canonical image space (vertically flipped
from the renderer), while ``CameraParams`` is expressed in render space.  This
module renders in render space and flips the resulting robot depth before it is
compared with the observed depth.

The implementation intentionally has no simulator dependency.  It uses
``yourdfpy`` for URDF forward kinematics and ``trimesh``/``rtree`` for CPU ray
intersection, which makes the same code usable with LIBERO and a calibrated
real camera.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from memory_system.types import CameraParams


class UrdfDepthFilterError(RuntimeError):
    """Raised when the URDF depth filter cannot be constructed or rendered."""


@dataclass(frozen=True)
class UrdfDepthFilterConfig:
    """Configuration for URDF-based robot removal.

    ``abs_tolerance`` and ``relative_tolerance`` define the depth agreement
    test ``abs(observed - rendered) <= abs_tol + rel_tol * rendered``.
    Distances are in metres.
    """

    urdf_path: str
    abs_tolerance: float = 0.008
    relative_tolerance: float = 0.005
    invalid_depth: float = 0.0
    use_collision_geometry: bool = False
    ray_chunk_size: int = 16_384

    def __post_init__(self) -> None:
        if not self.urdf_path:
            raise ValueError("urdf_path must not be empty")
        if self.abs_tolerance < 0.0:
            raise ValueError("abs_tolerance must be non-negative")
        if self.relative_tolerance < 0.0:
            raise ValueError("relative_tolerance must be non-negative")
        if self.ray_chunk_size <= 0:
            raise ValueError("ray_chunk_size must be positive")


@dataclass(frozen=True)
class UrdfDepthFilterResult:
    """Depth filtering output in canonical image space."""

    filtered_depth: np.ndarray
    robot_depth: np.ndarray
    robot_mask: np.ndarray
    removed_pixel_count: int


def _depth_image(depth: Any, name: str) -> np.ndarray:
    array = np.asarray(depth)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (H, W) or (H, W, 1)")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric depth values")
    return array


def build_robot_depth_mask(
    observed_depth: Any,
    robot_depth: Any,
    *,
    abs_tolerance: float = 0.008,
    relative_tolerance: float = 0.005,
) -> np.ndarray:
    """Return pixels whose observed depth agrees with rendered robot depth.

    Both inputs are canonical-space metric depth images.  Invalid values
    (non-finite or non-positive) never become robot pixels.  In particular, an
    object visibly in front of the robot is preserved whenever its depth differs
    from the rendered robot by more than the configured tolerance.
    """

    observed = _depth_image(observed_depth, "observed_depth").astype(
        np.float64, copy=False
    )
    rendered = _depth_image(robot_depth, "robot_depth").astype(
        np.float64, copy=False
    )
    if observed.shape != rendered.shape:
        raise ValueError(
            "observed_depth and robot_depth must have the same shape; "
            f"got {observed.shape} and {rendered.shape}"
        )
    if abs_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("depth tolerances must be non-negative")

    valid_observed = np.isfinite(observed) & (observed > 0.0)
    valid_robot = np.isfinite(rendered) & (rendered > 0.0)
    tolerance = float(abs_tolerance) + float(relative_tolerance) * np.abs(rendered)
    return valid_observed & valid_robot & (np.abs(observed - rendered) <= tolerance)


def filter_depth_with_robot_depth(
    observed_depth: Any,
    robot_depth: Any,
    *,
    abs_tolerance: float = 0.008,
    relative_tolerance: float = 0.005,
    invalid_depth: float = 0.0,
) -> UrdfDepthFilterResult:
    """Invalidate observed pixels that match a rendered robot depth image."""

    observed = _depth_image(observed_depth, "observed_depth")
    rendered = _depth_image(robot_depth, "robot_depth")
    robot_mask = build_robot_depth_mask(
        observed,
        rendered,
        abs_tolerance=abs_tolerance,
        relative_tolerance=relative_tolerance,
    )
    filtered = np.array(observed, copy=True)
    filtered[robot_mask] = invalid_depth
    return UrdfDepthFilterResult(
        filtered_depth=filtered,
        robot_depth=np.array(rendered, copy=True),
        robot_mask=robot_mask,
        removed_pixel_count=int(np.count_nonzero(robot_mask)),
    )


def _pose_matrix(pose: Any | None) -> np.ndarray:
    """Convert identity, a 4x4 transform, or xyz+xyzw into a transform."""

    if pose is None:
        return np.eye(4, dtype=np.float64)
    value = np.asarray(pose, dtype=np.float64)
    if value.shape == (4, 4):
        if not np.all(np.isfinite(value)):
            raise ValueError("robot_base_pose contains non-finite values")
        return value.copy()
    if value.size != 7:
        raise ValueError("robot_base_pose must have shape (4, 4) or xyz+xyzw")

    x, y, z, qx, qy, qz, qw = value.reshape(7)
    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("robot_base_pose quaternion must be finite and non-zero")
    qx, qy, qz, qw = quaternion / norm
    rotation = np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [x, y, z]
    return transform


class UrdfDepthFilter:
    """Render a posed URDF and remove matching pixels from observed depth.

    The object loads the URDF once and can be reused across frames.  It is not
    thread-safe because ``yourdfpy`` updates its scene graph in place.
    """

    def __init__(
        self,
        config: UrdfDepthFilterConfig,
        *,
        default_joint_positions: Mapping[str, float] | None = None,
        filename_handler: Callable[[str], str] | None = None,
    ) -> None:
        self.config = config
        urdf_path = Path(config.urdf_path).expanduser()
        if not urdf_path.is_file():
            raise UrdfDepthFilterError(f"URDF file does not exist: {urdf_path}")

        try:
            from yourdfpy import URDF
        except ImportError as exc:
            raise UrdfDepthFilterError(
                "URDF filtering requires yourdfpy, trimesh, and rtree"
            ) from exc

        load_options: dict[str, Any] = {
            "build_scene_graph": not config.use_collision_geometry,
            "build_collision_scene_graph": config.use_collision_geometry,
            "load_meshes": not config.use_collision_geometry,
            "load_collision_meshes": config.use_collision_geometry,
            "force_mesh": True,
            "force_collision_mesh": True,
        }
        if filename_handler is not None:
            load_options["filename_handler"] = filename_handler
        try:
            self._urdf = URDF.load(str(urdf_path), **load_options)
        except Exception as exc:
            raise UrdfDepthFilterError(
                f"failed to load URDF or its meshes: {urdf_path}"
            ) from exc

        self._joint_names = tuple(self._urdf.actuated_joint_names)
        defaults = dict(default_joint_positions or {})
        unknown = sorted(set(defaults) - set(self._joint_names))
        if unknown:
            raise ValueError(f"unknown default actuated joints: {unknown}")
        self._default_joint_positions = {
            name: float(defaults.get(name, 0.0)) for name in self._joint_names
        }

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Actuated joint order reported by the loaded URDF."""

        return self._joint_names

    def _joint_configuration(
        self,
        joint_positions: Mapping[str, float] | Sequence[float] | np.ndarray,
        joint_names: Sequence[str] | None,
    ) -> dict[str, float]:
        configuration = dict(self._default_joint_positions)
        if isinstance(joint_positions, Mapping):
            supplied = {str(name): float(value) for name, value in joint_positions.items()}
        else:
            values = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
            names = tuple(joint_names) if joint_names is not None else self._joint_names
            if len(values) != len(names):
                raise ValueError(
                    f"received {len(values)} joint values for {len(names)} joint names"
                )
            supplied = {str(name): float(value) for name, value in zip(names, values)}

        unknown = sorted(set(supplied) - set(self._joint_names))
        if unknown:
            raise ValueError(f"joint names are not actuated by this URDF: {unknown}")
        if not all(np.isfinite(value) for value in supplied.values()):
            raise ValueError("joint_positions contains non-finite values")
        configuration.update(supplied)
        return configuration

    def _posed_mesh(
        self,
        joint_positions: Mapping[str, float] | Sequence[float] | np.ndarray,
        joint_names: Sequence[str] | None,
    ) -> Any:
        configuration = self._joint_configuration(joint_positions, joint_names)
        self._urdf.update_cfg(configuration)
        scene = (
            self._urdf.scene_collision
            if self.config.use_collision_geometry
            else self._urdf.scene
        )
        if scene is None or not scene.geometry:
            geometry_kind = "collision" if self.config.use_collision_geometry else "visual"
            raise UrdfDepthFilterError(f"URDF has no loaded {geometry_kind} geometry")
        mesh = scene.to_mesh()
        if mesh is None or len(mesh.faces) == 0:
            raise UrdfDepthFilterError("URDF scene did not produce a triangle mesh")
        return mesh

    @staticmethod
    def _camera_rays(camera: CameraParams) -> tuple[np.ndarray, np.ndarray]:
        height, width = int(camera.height), int(camera.width)
        if height <= 0 or width <= 0:
            raise ValueError("camera height and width must be positive")
        intrinsic = np.asarray(camera.K, dtype=np.float64)
        camera_to_world = np.asarray(camera.T_c2w, dtype=np.float64)
        if intrinsic.shape != (3, 3) or camera_to_world.shape != (4, 4):
            raise ValueError("camera K and T_c2w must have shapes (3, 3) and (4, 4)")
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")

        rows, cols = np.indices((height, width), dtype=np.float64)
        directions = np.stack(
            [
                (cols.ravel() - intrinsic[0, 2]) / fx,
                (rows.ravel() - intrinsic[1, 2]) / fy,
                np.ones(height * width, dtype=np.float64),
            ],
            axis=-1,
        )
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return directions, camera_to_world

    def render_robot_depth(
        self,
        *,
        camera_params: CameraParams,
        joint_positions: Mapping[str, float] | Sequence[float] | np.ndarray,
        joint_names: Sequence[str] | None = None,
        robot_base_pose: Any | None = None,
    ) -> np.ndarray:
        """Render canonical-space metric z-depth for the posed robot.

        ``robot_base_pose`` is the world pose of the URDF base, represented as
        a 4x4 transform or ``[x, y, z, qx, qy, qz, qw]``.  Identity is used when
        it is omitted.
        """

        try:
            import trimesh
        except ImportError as exc:
            raise UrdfDepthFilterError(
                "URDF filtering requires yourdfpy, trimesh, and rtree"
            ) from exc

        mesh = self._posed_mesh(joint_positions, joint_names)
        directions_camera, camera_to_world = self._camera_rays(camera_params)
        world_to_base = np.linalg.inv(_pose_matrix(robot_base_pose))
        camera_to_base = world_to_base @ camera_to_world
        ray_origin = camera_to_base[:3, 3]
        directions_base = directions_camera @ camera_to_base[:3, :3].T

        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        depth_render = np.zeros(
            int(camera_params.height) * int(camera_params.width), dtype=np.float32
        )
        chunk_size = self.config.ray_chunk_size
        try:
            for start in range(0, len(directions_base), chunk_size):
                stop = min(start + chunk_size, len(directions_base))
                chunk_directions = directions_base[start:stop]
                chunk_origins = np.broadcast_to(ray_origin, chunk_directions.shape)
                locations, ray_indices, _ = intersector.intersects_location(
                    ray_origins=chunk_origins,
                    ray_directions=chunk_directions,
                    multiple_hits=False,
                )
                if len(ray_indices) == 0:
                    continue
                global_indices = start + ray_indices
                distances = np.einsum(
                    "ij,ij->i",
                    locations - chunk_origins[ray_indices],
                    chunk_directions[ray_indices],
                )
                z_depth = distances * directions_camera[global_indices, 2]
                valid = (
                    np.isfinite(z_depth)
                    & (z_depth >= float(camera_params.near))
                    & (z_depth <= float(camera_params.far))
                )
                depth_render[global_indices[valid]] = z_depth[valid].astype(np.float32)
        except ModuleNotFoundError as exc:
            if exc.name == "rtree":
                raise UrdfDepthFilterError(
                    "trimesh CPU ray intersection requires the rtree package"
                ) from exc
            raise
        except Exception as exc:
            raise UrdfDepthFilterError("failed to ray-render the posed URDF") from exc

        depth_render = depth_render.reshape(
            int(camera_params.height), int(camera_params.width)
        )
        return np.flipud(depth_render).copy()

    def filter(
        self,
        observed_depth: Any,
        *,
        camera_params: CameraParams,
        joint_positions: Mapping[str, float] | Sequence[float] | np.ndarray,
        joint_names: Sequence[str] | None = None,
        robot_base_pose: Any | None = None,
    ) -> UrdfDepthFilterResult:
        """Render the robot and invalidate matching observed depth pixels."""

        observed = _depth_image(observed_depth, "observed_depth")
        expected_shape = (int(camera_params.height), int(camera_params.width))
        if observed.shape != expected_shape:
            raise ValueError(
                f"observed_depth shape {observed.shape} does not match camera {expected_shape}"
            )
        robot_depth = self.render_robot_depth(
            camera_params=camera_params,
            joint_positions=joint_positions,
            joint_names=joint_names,
            robot_base_pose=robot_base_pose,
        )
        return filter_depth_with_robot_depth(
            observed,
            robot_depth,
            abs_tolerance=self.config.abs_tolerance,
            relative_tolerance=self.config.relative_tolerance,
            invalid_depth=self.config.invalid_depth,
        )


__all__ = [
    "UrdfDepthFilter",
    "UrdfDepthFilterConfig",
    "UrdfDepthFilterError",
    "UrdfDepthFilterResult",
    "build_robot_depth_mask",
    "filter_depth_with_robot_depth",
]
