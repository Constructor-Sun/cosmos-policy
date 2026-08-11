#!/usr/bin/env python3
"""Build an auditable LIBERO-Plus catalog without training on registered variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CATEGORIES = {
    "camera": "Camera Viewpoints",
    "background": "Background Textures",
    "light": "Light Conditions",
    "noise": "Sensor Noise",
    "language": "Language Instructions",
}
CAMERA_RE = re.compile(
    r"^(?P<base>.+)_view_(?P<h>\d+)_(?P<v>\d+)_(?P<s>\d+)_"
    r"(?P<yaw>\d+)_(?P<pitch>\d+)_initstate_(?P<init>\d+)$"
)
NOISE_RE = re.compile(r"^(?P<base>.+)_view_0_0_100_0_0_initstate_0_noise_(?P<id>\d+)$")


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def language_from_bddl(path: pathlib.Path) -> str:
    match = re.search(r"\(:language\s+([^)]+)\)", path.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"missing language in {path}")
    return match.group(1).strip()


def official_base(item: dict[str, Any]) -> str | None:
    name, category = item["name"], item["category"]
    if category in {CATEGORIES["camera"], "Robot Initial States"}:
        match = CAMERA_RE.match(name)
        return match.group("base") if match else None
    if category == CATEGORIES["noise"]:
        match = NOISE_RE.match(name)
        return match.group("base") if match else None
    if category == CATEGORIES["language"] and "_language_" in name:
        return name.split("_language_", 1)[0]
    for marker in ("_table_", "_tb_", "_light_", "_add_", "_level"):
        if marker in name:
            return name.split(marker, 1)[0]
    return None


def base_tasks(items: list[dict[str, Any]], bddl_root: pathlib.Path) -> dict[str, dict[str, str]]:
    names = sorted({base for item in items if (base := official_base(item))})
    out = {}
    for name in names:
        path = bddl_root / f"{name}.bddl"
        if path.is_file():
            out[name] = {"bddl_file": path.name, "instruction": language_from_bddl(path)}
    return out


def make_record(
    *, condition: str, base: str, task_name: str, bddl_file: str | None,
    instruction: str, official: dict[str, Any] | None, parameters: dict[str, Any],
) -> dict[str, Any]:
    digest = hashlib.sha256(f"{condition}:{task_name}".encode()).hexdigest()[:12]
    return {
        "variant_id": f"{condition}-{digest}",
        "condition": condition,
        "base_task": base,
        "task_name": task_name,
        "bddl_file": bddl_file,
        "instruction": instruction,
        "official": official is not None,
        "classification_id": official.get("id") if official else None,
        "difficulty_level": official.get("difficulty_level") if official else None,
        "parameters": parameters,
    }


def parse_camera(name: str) -> tuple[str, dict[str, int]]:
    match = CAMERA_RE.match(name)
    if match is None:
        raise ValueError(f"invalid camera task: {name}")
    values = {key: int(match.group(key)) for key in ("h", "v", "s", "yaw", "pitch", "init")}
    return match.group("base"), values


def build(args: argparse.Namespace) -> dict[str, Any]:
    libero = pathlib.Path(args.libero_plus).expanduser().resolve()
    bddl_root = libero / "libero/libero/bddl_files" / args.suite
    classification_path = libero / "libero/libero/benchmark/task_classification.json"
    items = read_json(classification_path)[args.suite]
    tasks = base_tasks(items, bddl_root)
    official_by_name = {item["name"]: item for item in items}
    variants = {task: {condition: [] for condition in CATEGORIES} for task in tasks}

    # Registered camera variants plus non-registered 5-degree training views.
    for item in items:
        if item["category"] != CATEGORIES["camera"]:
            continue
        base, p = parse_camera(item["name"])
        if base not in tasks or p["init"] != 0:
            continue
        variants[base]["camera"].append(make_record(
            condition="camera", base=base, task_name=item["name"], bddl_file=None,
            instruction=tasks[base]["instruction"], official=item, parameters=p,
        ))
        train_p = dict(p)
        train_p["v"] = (train_p["v"] + args.camera_train_offset_deg) % 360
        view = "_".join(str(train_p[key]) for key in ("h", "v", "s", "yaw", "pitch"))
        train_name = f"{base}_view_{view}_initstate_0"
        if train_name not in official_by_name:
            variants[base]["camera"].append(make_record(
                condition="camera", base=base, task_name=train_name, bddl_file=None,
                instruction=tasks[base]["instruction"], official=None, parameters=train_p,
            ))

    # BDDL-backed variants: every resource is cataloged; classification decides official status.
    suffixes = {
        "background": re.compile(r"^(?P<base>.+)_(?P<kind>table|tb)_(?P<id>\d+)$"),
        "light": re.compile(r"^(?P<base>.+)_light_(?P<id>\d+)$"),
        "language": re.compile(r"^(?P<base>.+)_language_(?P<id>\d+)$"),
    }
    official_language = {
        name.split("_view_", 1)[0]: item
        for name, item in official_by_name.items()
        if item["category"] == CATEGORIES["language"] and "_view_" in name
    }
    for path in sorted(bddl_root.glob("*.bddl")):
        for condition, pattern in suffixes.items():
            match = pattern.match(path.stem)
            if match is None or match.group("base") not in tasks:
                continue
            base = match.group("base")
            official = official_language.get(path.stem) if condition == "language" else official_by_name.get(path.stem)
            params = {f"{condition}_id": int(match.group("id"))}
            if condition == "background":
                params["kind"] = match.group("kind")
            task_name = next(
                (name for name, item in official_by_name.items() if condition == "language" and name.startswith(path.stem + "_view_") and item == official),
                path.stem,
            )
            variants[base][condition].append(make_record(
                condition=condition, base=base, task_name=task_name, bddl_file=path.name,
                instruction=language_from_bddl(path), official=official, parameters=params,
            ))
            break

    # Noise filenames are virtual; all levels are supported by the official wrapper.
    official_noise: dict[str, dict[int, dict[str, Any]]] = {task: {} for task in tasks}
    for item in items:
        if item["category"] != CATEGORIES["noise"]:
            continue
        match = NOISE_RE.match(item["name"])
        if match and match.group("base") in tasks:
            official_noise[match.group("base")][int(match.group("id"))] = item
    for base in tasks:
        for noise_id in range(1, 51):
            item = official_noise[base].get(noise_id)
            name = f"{base}_view_0_0_100_0_0_initstate_0_noise_{noise_id}"
            family = ("motion", "gaussian", "zoom", "fog", "glass")[(noise_id - 1) // 10]
            variants[base]["noise"].append(make_record(
                condition="noise", base=base, task_name=name, bddl_file=tasks[base]["bddl_file"],
                instruction=tasks[base]["instruction"], official=item,
                parameters={"noise_id": noise_id, "family": family, "severity": (noise_id - 1) % 10 + 1, "cameras": ["front"]},
            ))

    for task_variants in variants.values():
        for records in task_variants.values():
            records.sort(key=lambda record: record["variant_id"])
    return {
        "schema_version": 1,
        "suite": args.suite,
        "source": {"libero_plus": str(libero), "classification": str(classification_path)},
        "policy": {"official_variants_reserved_for_external_test": True},
        "tasks": {task: {**tasks[task], "variants": variants[task]} for task in sorted(tasks)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-plus", default=str(ROOT.parent / "LIBERO-plus"))
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--camera-train-offset-deg", type=int, default=5)
    parser.add_argument("--output", default=str(ROOT / "configs/libero_plus_variant_catalog.json"))
    args = parser.parse_args()
    catalog = build(args)
    write_json(pathlib.Path(args.output), catalog)
    counts = {
        condition: sum(len(task["variants"][condition]) for task in catalog["tasks"].values())
        for condition in CATEGORIES
    }
    print(json.dumps({"tasks": len(catalog["tasks"]), "variants": counts}, indent=2))


if __name__ == "__main__":
    main()
