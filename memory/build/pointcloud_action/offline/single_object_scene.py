"""Generate a minimal single-surface + single-object BDDL for local Pick tests.

The generated BDDL keeps only:
  - one support fixture (table, floor, etc.),
  - one target object,
  - the placement region/init needed by that object,
  - a trivial goal that is safe for LIBERO's parser.

It removes all other objects, containers, and extra fixtures from the template.
"""
from __future__ import annotations

import re
from pathlib import Path


def _split_top_level_blocks(text: str) -> list[str]:
    """Split text at top-level balanced parentheses."""
    blocks: list[str] = []
    start: int | None = None
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            if depth == 0:
                start = index
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(text[start : index + 1])
                start = None
    return blocks


_TOP_LEVEL_SECTIONS = (
    "regions",
    "fixtures",
    "objects",
    "obj_of_interest",
    "init",
    "goal",
)


def _section_text(text: str, header: str) -> str:
    """Return text belonging to a top-level BDDL section, e.g. regions/fixtures."""
    # Top-level sections can be indented, so do not anchor to column 0.  We only
    # stop before one of the known top-level section headers, not nested ones.
    next_section = "|".join(_TOP_LEVEL_SECTIONS)
    pattern = re.compile(
        rf"\(\:{header}\b(.*?)(?=\(\:(?:{next_section})\b|\Z)",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"cannot find (: {header}) section in template")
    return match.group(1)


def _parse_section_entries(text: str) -> list[str]:
    """Return top-level parenthesised entries inside a section."""
    return _split_top_level_blocks(text)


def _region_name(block: str) -> str:
    match = re.match(r"\(\s*([\w]+)", block)
    if not match:
        raise ValueError(f"cannot parse region block: {block[:80]!r}")
    return match.group(1)


def _region_target(block: str) -> str | None:
    match = re.search(r"\(\s*:target\s+([\w]+)\s*\)", block)
    return match.group(1) if match else None


def _find_object_init(text: str, object_name: str) -> str:
    """Return the full ``(On object_name table_region)`` init expression."""
    init_text = _section_text(text, "init")
    for block in _parse_section_entries(init_text):
        # Example: (On moka_pot_1 kitchen_table_moka_pot_init_region)
        m = re.match(rf"\(\s*On\s+{re.escape(object_name)}\s+([\w]+)\s*\)", block.strip())
        if m:
            return block.strip()
    raise ValueError(f"cannot find (On {object_name} ...) in template init section")


def _extract_region_for_object(text: str, object_name: str, surface_fixture: str) -> str:
    """Extract the exact region block referenced by the object's init clause."""
    init = _find_object_init(text, object_name)
    full_region = init.split()[2].rstrip(")")
    region_text = _section_text(text, "regions")
    for block in _parse_section_entries(region_text):
        name = _region_name(block)
        # full_region is usually <surface>_<region_name>.
        if full_region.endswith(f"_{name}") or full_region == name:
            target = _region_target(block)
            if target == surface_fixture:
                return block
    # Fallback: if there is exactly one region targeting the surface whose name
    # contains the object type, use that.
    object_base = re.sub(r"_\d+$", "", object_name)
    for block in _parse_section_entries(region_text):
        name = _region_name(block)
        target = _region_target(block)
        if target == surface_fixture and object_base in name:
            return block
    raise ValueError(
        f"cannot find region for {object_name} on {surface_fixture} in template"
    )


def _fixtures(text: str) -> dict[str, str]:
    """Return mapping fixture_name -> fixture_type from the template."""
    fixtures: dict[str, str] = {}
    fixture_text = _section_text(text, "fixtures")
    for line in fixture_text.splitlines():
        line = line.strip()
        if " - " not in line:
            continue
        names, ftype = line.split(" - ", 1)
        for name in names.split():
            fixtures[name.strip()] = ftype.strip()
    return fixtures


def _object_type_from_template(text: str, object_name: str) -> str:
    """Infer object type from the template's (:objects ...) section."""
    object_text = _section_text(text, "objects")
    for line in object_text.splitlines():
        line = line.strip()
        if " - " not in line:
            continue
        names, otype = line.split(" - ", 1)
        if object_name in names.split():
            return otype.strip()
    raise ValueError(f"cannot find {object_name} in template objects section")


def generate_single_object_bddl(
    template_path: str | Path,
    object_name: str,
    object_type: str | None = None,
    surface_fixture: str | None = None,
    output_path: str | Path | None = None,
    language: str | None = None,
) -> str:
    """Generate a minimal one-surface/one-object BDDL string.

    Args:
        template_path: Existing BDDL file to use as scene/object template.
        object_name: Full object instance name, e.g. ``moka_pot_1``.
        object_type: Optional object type, e.g. ``moka_pot``. If not given, it
            is inferred from the template.
        surface_fixture: The support fixture for the object, e.g. ``kitchen_table``
            or ``floor``. If not given, it is inferred from the object's init
            region target.
        output_path: If provided, the generated BDDL is written here.
        language: Optional human-readable language string.
    """
    template = Path(template_path).read_text()

    if object_type is None:
        object_type = _object_type_from_template(template, object_name)

    if surface_fixture is None:
        # The init line references <surface>_<region>. Determine the support
        # fixture from the matching region's (:target ...).
        init = _find_object_init(template, object_name)
        full_region = init.split()[2].rstrip(")")
        region_text = _section_text(template, "regions")
        for block in _parse_section_entries(region_text):
            if _region_name(block) in full_region or full_region.endswith(
                "_" + _region_name(block)
            ):
                target = _region_target(block)
                if target is not None:
                    surface_fixture = target
                    break
        if surface_fixture is None:
            # Fall back to a fixture whose name appears in the full region token.
            for name in _fixtures(template):
                if name in full_region:
                    surface_fixture = name
                    break
        if surface_fixture is None:
            fixtures = _fixtures(template)
            if len(fixtures) == 1:
                surface_fixture = next(iter(fixtures))
        if surface_fixture is None:
            raise ValueError("cannot infer support fixture from template")

    region_block = _extract_region_for_object(
        template, object_name, surface_fixture
    )

    if language is None:
        language = f"pick up the {object_type.replace('_', ' ')}"

    # Keep only the support fixture, preserving its original fixture type
    # (e.g. ``main_table - table``, not ``main_table - main_table``).
    fixture_type = _fixtures(template).get(surface_fixture, surface_fixture)
    fixture_lines = [f"    {surface_fixture} - {fixture_type}"]
    fixtures_block = "  (:fixtures\n" + "\n".join(fixture_lines) + "\n  )"

    objects_block = f"  (:objects\n    {object_name} - {object_type}\n  )"

    init_expr = _find_object_init(template, object_name)
    init_block = "  (:init\n" + f"    {init_expr}\n" + "  )"

    # Use a predicate that is always valid/true. This BDDL is only used to
    # instantiate the local Pick scene; success is judged by the Pick evaluator,
    # not by LIBERO's task goal.
    goal_block = f"  (:goal\n    (And (true {object_name}))\n  )"

    # Keep the original problem/domain header so LIBERO maps the BDDL to the
    # correct environment class (study/kitchen/living-room etc.).  Only the
    # language is replaced.
    problem_match = re.match(r"\(define\s+\(problem\s+([\w]+)\)", template)
    domain_match = re.search(r"\(\:domain\s+([\w]+)\)", template)
    domain = domain_match.group(1) if domain_match else "robosuite"
    problem_name = problem_match.group(1) if problem_match else f"SingleObjectPick_{object_name}"
    header = f"(define (problem {problem_name})\n  (:domain {domain})"


    bddl = (
        header
        + "\n"
        + f"  (:language {language})"
        + "\n"
        + "    (:regions\n"
        + f"      {region_block}\n"
        + "    )\n"
        + "\n"
        + fixtures_block
        + "\n\n"
        + objects_block
        + "\n\n"
        + f"  (:obj_of_interest\n    {object_name}\n  )\n\n"
        + init_block
        + "\n\n"
        + goal_block
        + "\n)"
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(bddl)

    return bddl
