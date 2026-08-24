"""Thin CLI entry point for building the 3D ready-distance memory."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_system.offline.build_ready3d import main  # noqa: E402

if __name__ == "__main__":
    main()
