#!/usr/bin/env python
"""Re-encode repaired_full.mp4 for rollouts recorded as prefix+rerun concats.

The paired repair runs for the five Place-skill interventions recorded the
baseline prefix and the repaired re-run back to back in one hdf5
(frame_indices resets to ~10 at the seam). The concat seam looks like a
violent jump in the encoded video, so for these cases the video is rebuilt
from the repaired re-run segment only, labelled with its own frame indices.
"""
import io
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = (ROOT / "experiments/tta/phase_check_videos/repair_failures"
           / "summary.json")
ROLLOUT_DIRS = {
    "repro68": ROOT / "experiments/tta/tta_repair_work_repro_68/results",
    "bg5": ROOT / "experiments/tta/tta_repair_bg5_tstar_correct/results",
}
sys.path[:0] = [str(ROOT / "tests/phase")]
from export_case_videos import annotate  # noqa: E402


def main():
    import imageio
    summary = json.loads(SUMMARY.read_text())
    for entry in summary:
        tag = entry["tag"]
        rollout = sorted(
            (ROLLOUT_DIRS[entry["set"]] / tag / "logs" / "rollout_data")
            .glob("episode_data--*.hdf5"))
        if not rollout:
            print(f"{tag}: no rollout file, skipped", flush=True)
            continue
        with h5py.File(rollout[0]) as h5:
            indices = h5["frame_indices"][:]
            dataset = h5["primary_images_jpeg"]
            n = len(dataset)
            jpegs = [dataset[k] for k in range(n)]
        seams = [i + 1 for i in range(n - 1) if indices[i + 1] != indices[i] + 1]
        out_path = Path(entry["artifacts"][-1])
        assert out_path.name == "repaired_full.mp4", out_path
        success = entry.get("repaired_rollout_success")
        success_text = "unknown" if success is None else str(success)
        if seams:
            start = seams[0]
            label = (f"{tag} REPAIRED rollout success={success_text} "
                     f"(re-run from initial state; prefix excluded)")
        else:
            start = 0
            label = f"{tag} REPAIRED rollout success={success_text}"
        with imageio.get_writer(out_path, fps=20) as w:
            for k in range(start, n):
                frame = np.asarray(PILImage.open(io.BytesIO(jpegs[k])))
                w.append_data(annotate(
                    frame, [label, f"rollout step {int(indices[k])}"],
                    None, int(indices[k]), None))
        if seams:
            entry["repaired_rollout_seam"] = {
                "prefix_frames": int(start),
                "rerun_first_frame_index": int(indices[start]),
                "note": ("recording concatenates the baseline prefix and a "
                         "fresh repaired re-run; video shows the re-run only")}
        print(f"reencoded {tag}: seams={seams or 'none'}", flush=True)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
