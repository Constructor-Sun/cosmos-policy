"""SMOKE_PYTHON_SCRIPT override for scripts/run_libero_smoke_test.sh.

Wraps cosmos_utils.get_action for attention capture BEFORE the smoke-test
script runs (run_libero_eval binds the name via ``from cosmos_utils import``),
then executes the original script as __main__ unchanged.

Env knobs (see attn_capture.py): ATTN_VIZ_DIR, ATTN_VIZ_EVERY, ATTN_VIZ_BLOCKS.
"""

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(HERE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from attn_capture import wrap_get_action

wrap_get_action(
    every=int(os.environ.get("ATTN_VIZ_EVERY", "1")),
    out_dir=os.environ.get("ATTN_VIZ_DIR", str(REPO / "outputs/attn_viz/default")),
    n_blocks=int(os.environ.get("ATTN_VIZ_BLOCKS", "4")),
)


def install_object_dump(substrings):
    """Dump ground-truth object bboxes from the eval's own env at episode start.

    Wraps run_libero_eval.run_episode; on the first set_init_state it reads the
    matching bodies' world poses from MuJoCo, projects them onto the agentview
    image via memory_system.geometry, and writes objects.json into ATTN_VIZ_DIR.
    """
    import json

    import numpy as np

    import cosmos_policy.experiments.robot.libero.run_libero_eval as rle
    from memory_system.geometry import camera_params, world_to_pixel

    orig_run_episode = rle.run_episode
    done = {"v": False}

    def dump(env):
        if done["v"]:
            return
        done["v"] = True
        sim = env.sim
        try:
            cam = camera_params(sim, "agentview", 256, 256)
        except Exception as exc:
            print(f"[attn_viz] object dump: no agentview camera ({exc})", flush=True)
            return
        f = float(cam.K[0, 0])
        out = {}
        for i in range(sim.model.nbody):
            name = sim.model.body_id2name(i)
            if not name or not any(s.lower() in name.lower() for s in substrings):
                continue
            gadr = sim.model.body_geomadr[i]
            gnum = sim.model.body_geomnum[i]
            rows, cols, rmax = [], [], 6.0
            for gi in range(gadr, gadr + gnum):
                gw = np.asarray(sim.data.geom_xpos[gi], dtype=np.float64)
                size = float(np.max(sim.model.geom_size[gi]))
                z = (cam.T_w2c[:3, :3] @ gw + cam.T_w2c[:3, 3])[2]
                rmax = max(rmax, f * size / max(abs(z), 1e-6))
                r, c = world_to_pixel(gw, cam, flip=True)
                rows.append(float(r))
                cols.append(float(c))
            row_m = 255.0 - float(np.mean(rows))
            col_m = float(np.mean(cols))
            r0 = int(round(rmax))
            out[name] = {
                "bbox_model_orient": [
                    int(round(col_m - r0)), int(round(row_m - r0)),
                    int(round(col_m + r0)), int(round(row_m + r0)),
                ],
                "world_xyz": [float(x) for x in sim.data.body_xpos[i]],
            }
        path = Path(os.environ.get("ATTN_VIZ_DIR", ".")) / "objects.json"
        path.write_text(json.dumps(out, indent=1))
        print(f"[attn_viz] object dump -> {path}: {list(out)}", flush=True)

    def wrapped_run_episode(cfg, env, *a, **kw):
        orig_set = env.set_init_state

        def set_and_dump(state):
            orig_set(state)
            dump(env)

        env.set_init_state = set_and_dump
        return orig_run_episode(cfg, env, *a, **kw)

    rle.run_episode = wrapped_run_episode


_dump = os.environ.get("ATTN_VIZ_DUMP_OBJECTS", "")
if _dump:
    install_object_dump(_dump.split(","))

target = str(REPO / "scripts" / "run_libero_smoke_test.py")
sys.argv[0] = target
runpy.run_path(target, run_name="__main__")
