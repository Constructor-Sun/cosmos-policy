# Plan: Update memory ready_frame only

- Goal: only update memory data files; do not modify existing source code.
- Add one standalone script: `update_memory_ready_by_distance.py`.
- Selection rule per segment, search frames `[end-32, end]`:
  - Pick: target distance ≈ 7 cm
  - Place: target distance ≈ 14 cm
  - If none in range: fallback to ≈ 16 cm
- Update files:
  - `feasible_recovery_targets.pt`
  - `ready3d_targets.pt`
  - optionally sync `ready_frame` in segment manifest
- Do NOT modify:
  - `build_recovery.py`
  - `build_targets.py`
  - `label_boundaries.py`
  - recovery/execution code
