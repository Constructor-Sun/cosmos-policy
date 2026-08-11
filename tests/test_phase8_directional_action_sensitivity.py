from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from libero_plus_task_index import index_variants_by_base


def test_variant_index_uses_authoritative_base_names():
    base = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
    variant = SimpleNamespace(name=f"{base}_light_32")

    assert index_variants_by_base([base], [variant]) == {base: [variant]}


def test_variant_index_rejects_unclassified_base():
    variant = SimpleNamespace(name="unknown_task_light_1")

    try:
        index_variants_by_base(["known_task"], [variant])
    except RuntimeError as error:
        assert "has no clean base task" in str(error)
    else:
        raise AssertionError("expected an unmapped variant to fail")
