"""Goal-predicate probe: log every step the BDDL goal predicates plus the numbers
that decide them.  Enabled only when PROBE_GOAL_TAG is set."""

import os


def install():
    tag = os.environ.get("PROBE_GOAL_TAG")
    if not tag:
        return
    try:
        from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
    except Exception:
        return
    out = os.path.join(os.environ.get("PROBE_DIR", "/tmp"), "probe_goal_%s.txt" % tag)
    orig = BDDLBaseDomain.step
    n = {"i": 0}

    def line(env):
        import numpy as np
        gs = []
        try:
            gs = env.parsed_problem.get("goal_state") or []
        except Exception:
            pass
        bits = []
        for st in gs:
            st = tuple(st)
            try:
                ok = env._eval_predicate(st)
            except Exception as exc:
                ok = "ERR:%s" % type(exc).__name__
            parts = []
            for name in st[1:]:
                try:
                    pos = env.object_states_dict[name].get_geom_state()["pos"]
                    parts.append("%s(%.4f,%.4f,%.4f)" % (name, pos[0], pos[1], pos[2]))
                except Exception:
                    parts.append("%s(?)" % name)
            extra = ""
            if st[0] == "On" and len(st) == 3:
                a, b = st[1], st[2]
                try:
                    pa = np.asarray(env.object_states_dict[a].get_geom_state()["pos"])
                    pb = np.asarray(env.object_states_dict[b].get_geom_state()["pos"])
                    d = float(np.linalg.norm(pa[:2] - pb[:2]))
                    extra += " dxy=%.4f dz=%.4f" % (d, float(pb[2] - pa[2]))
                except Exception:
                    pass
                try:
                    extra += " contact=%s" % bool(
                        env.check_contact(env.get_object(a), env.get_object(b)))
                except Exception:
                    pass
                if b in getattr(env, "object_sites_dict", {}):
                    site = env.object_sites_dict[b]
                    extra += " under=%s" % hasattr(site, "under")
                    pn = getattr(site, "parent_name", None)
                    if pn:
                        try:
                            extra += " contact(%s,%s)=%s" % (
                                pn, a, bool(env.check_contact(env.get_object(pn), env.get_object(a))))
                        except Exception:
                            pass
            bits.append("%s%s=%s{%s%s}" % (st[0], tuple(st[1:]), ok, " ".join(parts), extra))
        try:
            succ = env._check_success()
        except Exception:
            succ = "ERR"
        return "succ=%s %s" % (succ, " | ".join(bits))

    def step(self, action):
        r = orig(self, action)
        n["i"] += 1
        try:
            with open(out, "a") as fh:
                fh.write("%d %s\n" % (n["i"], line(self)))
        except Exception:
            pass
        return r

    BDDLBaseDomain.step = step
