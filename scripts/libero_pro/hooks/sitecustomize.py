"""Debug probe: append one line per sim step for a chosen body and the gripper.

原为 /tmp/hook_site/sitecustomize.py。把它所在的目录放进 PYTHONPATH 就会生效
（Python 启动时自动 import sitecustomize）。每个 step 追加一行：

    step  <obj_z>  <eef_z>  <gripper_cmd>

环境变量
    PROBE_TAG     输出文件名后缀，默认 'x' -> probe_x.txt
    PROBE_DIR     输出目录，默认 /tmp
    PROBE_OBJECT  被探测 body 的名字片段，默认 'moka_pot'（跑 pan 时设成 chefmate_8_frypan）
    PROBE_FROM / PROBE_TO  记录步数窗口，默认 150..600

只用于单 case 调试（scripts/libero_pro/run_single_case.py 会把它加进 PYTHONPATH），
任何异常都被吞掉，不影响主流程。
"""
import os


def _install():
    try:
        from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
    except Exception:
        return
    tag = os.environ.get("PROBE_TAG", "x")
    probe_dir = os.environ.get("PROBE_DIR", "/tmp")
    needle = os.environ.get("PROBE_OBJECT", "moka_pot")
    lo = int(os.environ.get("PROBE_FROM", "150"))
    hi = int(os.environ.get("PROBE_TO", "600"))
    out_path = os.path.join(probe_dir, "probe_%s.txt" % tag)
    orig = BDDLBaseDomain.step
    n = {"i": 0}

    def step(self, action):
        out = orig(self, action)
        n["i"] += 1
        try:
            import numpy as np
            sim = self.sim
            if lo <= n["i"] <= hi:
                bodies = {sim.model.body_id2name(i): i for i in range(sim.model.nbody)}
                pid = next((v for k, v in bodies.items() if k and needle in k), None)
                pz = float(sim.data.body_xpos[pid][2]) if pid is not None else float("nan")
                site = getattr(self.robots[0], "eef_site_id", None)
                ez = float(sim.data.site_xpos[site][2]) if site is not None else float("nan")
                with open(out_path, "a") as fh:
                    fh.write("%d %.4f %.4f %.2f\n" % (n["i"], pz, ez, float(np.asarray(action).reshape(-1)[-1])))
        except Exception:
            pass
        return out

    BDDLBaseDomain.step = step


_install()
