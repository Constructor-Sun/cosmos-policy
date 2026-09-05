"""Analyze LIBERO-Object Pick sweep results for retrieval confusion.

Separates seen objects (present in LIBERO-90 memory) from unseen/novel objects
(not present in memory, where nearest-neighbor retrieval is expected).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO = Path("/data1/liu/exp/counterfactual/external/cosmos-policy")
MEMORY = REPO / "memory_system/pointcloud_action/pointcloud_action_memory.pt"


def load_memory_id_to_item(memory_path: Path) -> tuple[dict[str, str], set[str]]:
    payload = torch.load(memory_path, map_location="cpu", weights_only=False)
    id2item = {
        record["memory_id"]: str(record.get("arguments", {}).get("item"))
        for record in payload.get("records", [])
    }
    memory_items = set(id2item.values())
    return id2item, memory_items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO / "rollouts/libero_object_pick_sweep_video_results.jsonl",
    )
    parser.add_argument("--memory", type=Path, default=MEMORY)
    args = parser.parse_args()

    id2item, memory_items = load_memory_id_to_item(args.memory)
    rows = []
    with open(args.results) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"results file: {args.results}")
    print(f"total cases : {len(rows)}")
    print()

    # Categories:
    # seen_correct       : query item exists in memory and retrieved same item
    # seen_wrong         : query item exists in memory but retrieved another item
    # seen_no_retrieval  : query item exists in memory but no memory retrieved
    # unseen_retrieved   : query item NOT in memory, retrieved some nearest memory
    # unseen_no_retrieval: query item NOT in memory and no memory retrieved
    counters = Counter()
    success_by_category = Counter()
    total_by_category = Counter()
    per_object = defaultdict(lambda: [0, 0])
    confusion = Counter()
    unseen_fallbacks = []

    for row in rows:
        query_item = str(row.get("item"))
        memory_id = row.get("memory_id")
        retrieved_item = id2item.get(memory_id) if memory_id else None
        success = bool(row.get("success"))
        per_object[query_item][0] += 1
        per_object[query_item][1] += success

        if query_item not in memory_items:
            if retrieved_item is None:
                category = "unseen_no_retrieval"
            else:
                category = "unseen_retrieved"
                unseen_fallbacks.append((query_item, retrieved_item))
        else:
            if retrieved_item is None:
                category = "seen_no_retrieval"
            elif retrieved_item == query_item:
                category = "seen_correct"
            else:
                category = "seen_wrong"
                confusion[(query_item, retrieved_item)] += 1

        total_by_category[category] += 1
        success_by_category[category] += success

    print("=== Memory coverage ===")
    print(f"query objects in memory: {sum(1 for r in rows if str(r.get('item')) in memory_items)}/{len(rows)}")
    print(f"query objects NOT in memory: {sum(1 for r in rows if str(r.get('item')) not in memory_items)}/{len(rows)}")
    print()

    print("=== Category summary ===")
    for category in [
        "seen_correct",
        "seen_wrong",
        "seen_no_retrieval",
        "unseen_retrieved",
        "unseen_no_retrieval",
    ]:
        total = total_by_category[category]
        succ = success_by_category[category]
        rate = succ / total if total else float("nan")
        print(f"{category:20s}: {succ}/{total} = {rate:.2%}")
    print(f"{'overall':20s}: {sum(success_by_category.values())}/{len(rows)} = {sum(success_by_category.values()) / len(rows):.2%}")
    print()

    print("=== Seen-object retrieval ===")
    seen_total = total_by_category["seen_correct"] + total_by_category["seen_wrong"] + total_by_category["seen_no_retrieval"]
    seen_succ = success_by_category["seen_correct"] + success_by_category["seen_wrong"] + success_by_category["seen_no_retrieval"]
    correct = total_by_category["seen_correct"]
    wrong = total_by_category["seen_wrong"]
    no_ret = total_by_category["seen_no_retrieval"]
    print(f"seen total            : {seen_total}")
    print(f"seen correct retrieval: {correct}")
    print(f"seen wrong retrieval  : {wrong}")
    print(f"seen no retrieval     : {no_ret}")
    if seen_total:
        print(f"seen success rate     : {seen_succ}/{seen_total} = {seen_succ / seen_total:.2%}")
    print()

    print("=== Unseen/novel fallback (expected nearest-neighbor) ===")
    for query_item, retrieved_item in unseen_fallbacks:
        print(f"{query_item:24s} -> {retrieved_item:24s}")
    print()

    print("=== Per-object success ===")
    for obj, (total, succ) in sorted(per_object.items()):
        mark = "" if obj in memory_items else "  [unseen]"
        print(f"{obj:24s}: {succ}/{total} = {succ / total:.2%}{mark}")
    print()

    print("=== Seen-object retrieval confusion (query -> retrieved) ===")
    for (query_item, retrieved_item), count in sorted(confusion.items(), key=lambda x: -x[1]):
        print(f"{query_item:24s} -> {retrieved_item:24s} x{count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
