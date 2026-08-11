import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "configs/libero_plus_variant_catalog.json").read_text())
SPLITS = json.loads((ROOT / "configs/libero_plus_variant_splits.json").read_text())


def test_catalog_has_all_conditions_for_ten_tasks():
    assert len(CATALOG["tasks"]) == 10
    for task in CATALOG["tasks"].values():
        assert set(task["variants"]) == {"camera", "background", "light", "noise", "language"}
        assert all(task["variants"].values())


def test_official_variants_are_external_only():
    records = {
        record["variant_id"]: record
        for task in CATALOG["tasks"].values()
        for variants in task["variants"].values()
        for record in variants
    }
    for task in SPLITS["tasks"].values():
        for split in task.values():
            assert all(not records[variant]["official"] for variant in split["train"] + split["val"])
            assert all(records[variant]["official"] for variant in split["external_test"])


def test_noise_matches_official_front_only_contract():
    for task in CATALOG["tasks"].values():
        noise = task["variants"]["noise"]
        assert {record["parameters"]["noise_id"] for record in noise} == set(range(1, 51))
        assert all(record["parameters"]["cameras"] == ["front"] for record in noise)


def test_language_nonofficial_resources_have_distinct_instructions():
    for task in CATALOG["tasks"].values():
        candidates = [record for record in task["variants"]["language"] if not record["official"]]
        assert candidates
        assert all(record["instruction"] != task["instruction"] for record in candidates)
