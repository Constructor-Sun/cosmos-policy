"""Replay collected LIBERO-PRO rollouts in the simulator and measure how well
the phase-completion rule matches simulator ground truth.

Two stages, run in this order:

  --mode fidelity
      Replay the recorded action sequence and compare per-frame proprio
      against the recorded ``proprio``.  Bit-exact means the replay is 1:1 and
      every later conclusion is sound.  One-time gate, not a per-run ritual.

  --mode measure
      Additionally render agentview depth+segmentation, run the rule and the
      simulator truth per frame, then slice by truth-defined phase windows.

Two independent signal channels:

  truth  simulator-privileged -- is_grasping (contact) + body pose for Pick,
         BDDL predicates for the rest.  Evaluation label ONLY; never used as
         a runtime trigger, so real-robot transferability is unaffected.
  rule   observable-only -- the phase machine's own completion rules, reading
         the visible point cloud + proprio, i.e. what a real robot has.

Windows come from the truth, never from the rule output, so scoring a phase
can never gate itself.  A phase that never happened is not scored at all.

Env: needs LIBERO_CONFIG_PATH / PYTHONPATH / MUJOCO_GL set (as run_swap.sh).
For --mode measure also export COSMOS_SKILL_COMPLETION_SHADOW=1 so that
get_libero_env builds a SegmentationRenderEnv with instance segmentation.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np

NUM_STEPS_WAIT = 10
MODEL_FAMILY = "cosmos"
ENV_IMG_RES = 256
LIFT_THRESHOLD = 0.02       # mirrors label_segments.step_succeeded
DEBOUNCE_FRAMES = 5         # consecutive frames that count as a real state

PROPRIO_SLICES = {
    "gripper_qpos": slice(0, 2),
    "eef_pos": slice(2, 5),
    "eef_quat": slice(5, 9),
}
OBS_KEYS = {
    "gripper_qpos": "robot0_gripper_qpos",
    "eef_pos": "robot0_eef_pos",
    "eef_quat": "robot0_eef_quat",
}
PREDICATE_SKILLS = {
    "PlaceIn": "in", "PlaceOn": "on",
    "Close": "close", "Open": "open",
    "TurnOn": "turnon", "TurnOff": "turnoff",
}

EP_RE = re.compile(
    r"--task=(?P<task_id>\d+)--ep=(?P<ep>\d+)--success=(?P<success>\w+)--(?P<note>.+)\.hdf5$"
)


# ------------------------------------------------------------------ episode

def parse_episode_path(path):
    m = EP_RE.search(Path(path).name)
    if m is None:
        raise ValueError(f"cannot parse episode filename: {path}")
    return {
        "task_id": int(m["task_id"]),
        "ep": int(m["ep"]),
        "success": m["success"].lower() == "true",
        "note": m["note"],
    }


def load_episode(path):
    with h5py.File(path, "r") as f:
        out = {
            "actions": np.asarray(f["actions"], dtype=np.float64),
            "proprio": np.asarray(f["proprio"], dtype=np.float64),
        }
        # frame_indices is metadata only (never used for the replay) and is
        # absent from some collections, e.g. the regen_demo set.
        if "frame_indices" in f:
            out["frame_indices"] = np.asarray(f["frame_indices"], dtype=np.int64)
        else:
            out["frame_indices"] = np.arange(len(out["actions"]), dtype=np.int64)
    return out


def build_env(suite_name, task_id, seed):
    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
    from cosmos_policy.utils.utils import set_seed_everywhere

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    set_seed_everywhere(seed)
    env, _ = get_libero_env(
        task, MODEL_FAMILY, resolution=ENV_IMG_RES, camera_depths=[True, False]
    )
    return env, task, suite.get_task_init_states(task_id)


def replay(env, initial_state, actions, record_fn=None):
    """Reset, settle, then execute the recorded actions.

    ``record_fn(t, obs, action)`` is called BEFORE each env.step with the state
    that ``proprio[t]`` corresponds to (the eval loop records proprio before
    executing the action).
    """
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_dummy_action

    env.reset()
    obs = env.set_init_state(initial_state)
    dummy = get_libero_dummy_action(MODEL_FAMILY)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(dummy)

    for t, action in enumerate(actions):
        if record_fn is not None:
            record_fn(t, obs, action)
        obs, reward, done, info = env.step(np.asarray(action, dtype=np.float64).tolist())
    return obs


# ---------------------------------------------------------------- stage 1

def check_fidelity(env, initial_state, episode, tolerance=1e-6):
    recorded = episode["proprio"]
    n = len(episode["actions"])
    trace = {k: np.full((n, v.stop - v.start), np.nan) for k, v in PROPRIO_SLICES.items()}

    def record(t, obs, action):
        for name, key in OBS_KEYS.items():
            trace[name][t] = np.asarray(obs[key], dtype=np.float64)

    replay(env, initial_state, episode["actions"], record_fn=record)

    report = {}
    for name, sl in PROPRIO_SLICES.items():
        got, want = trace[name], recorded[:, sl]
        finite = np.isfinite(got).all(axis=1)
        diff = np.abs(got[finite] - want[finite])
        worst = float(diff.max()) if diff.size else float("nan")
        bad = np.nonzero(diff.max(axis=1) > tolerance)[0] if diff.size else np.array([], int)
        report[name] = {
            "max_abs_diff": worst,
            "first_bad_index": int(bad[0]) if len(bad) else None,
            "n_bad": int(len(bad)),
            "n_frames": int(finite.sum()),
        }
    return report


# ---------------------------------------------------------------- stage 2

def _base(env):
    return getattr(env, "env", env)


def pick_truth(env, item, start_pos, lift_threshold=LIFT_THRESHOLD):
    """Simulator truth for Pick: contact AND displacement from body pose."""
    from memory_system.offline.label_segments import is_grasping, object_position

    base = _base(env)
    displacement = object_position(base, item) - start_pos
    contact_lift = is_grasping(base, item) and np.linalg.norm(displacement) >= lift_threshold
    return bool(contact_lift or displacement[2] >= lift_threshold)


def relation_truth(env, predicate, expr_args):
    try:
        return bool(_base(env)._eval_predicate([predicate, *expr_args]))
    except Exception:
        return False


def rising_edge(trace, k=DEBOUNCE_FRAMES):
    run = 0
    for i, value in enumerate(trace):
        run = run + 1 if value else 0
        if run >= k:
            return i - k + 1
    return None


def falling_edge(trace, start, k=DEBOUNCE_FRAMES):
    if start is None:
        return None
    run = 0
    for i in range(start, len(trace)):
        run = run + 1 if not trace[i] else 0
        if run >= k:
            return i - k + 1
    return None


def phases_for_task(env, task):
    """Nominal phase sequence from the BDDL goals (task spec, not world state)."""
    from memory_system.offline.planner import (
        build_skeleton, infer_openables, normalized_goals, resolve_bddl,
    )
    from libero.libero.envs.bddl_utils import robosuite_parse_problem

    parsed = getattr(_base(env), "parsed_problem", None)
    if parsed is None:
        parsed = robosuite_parse_problem(str(resolve_bddl("libero_10", task.name)))
    goals = normalized_goals(parsed["goal_state"])
    steps = build_skeleton(goals, task.language, infer_openables(parsed))
    return [
        {"step_id": s.step_id, "skill": s.skill, "arguments": dict(s.arguments),
         "execution_mode": s.execution_mode}
        for s in steps
    ]


def make_recorder(env, task, n_frames, dump_traces=False, open_frames=16):
    """Build (record, finalize) for one trajectory of ``n_frames`` frames.

    Source-agnostic: the caller drives the frames, whether by replaying
    actions or by restoring simulator states.  ``record(t, obs, action)`` is
    called once per frame with the state that frame ``t`` corresponds to.
    """
    from memory_system.execute.skill_completion.pick import PickCompletionChecker
    from memory_system.execute.skill_completion.shadow import (
        PickTargetPointCloud, resolve_target_instance,
    )
    from memory_system.offline.label_segments import object_position

    base = _base(env)
    n = n_frames
    phases = phases_for_task(env, task)

    specs = []
    for p in phases:
        skill, a = p["skill"], p["arguments"]
        item, target = a.get("item"), a.get("target")
        specs.append({
            "key": f"{skill}|{item or ''}|{target or ''}",
            "skill": skill, "item": item, "target": target,
            "arguments": dict(a),
            # Resolved in setup(), i.e. after env.reset() has populated
            # instance_to_id -- resolving earlier silently yields None and
            # skips both the truth and the rule.
            "item_instance": None,
            "has_rule": skill in {"Pick", "PlaceIn", "PlaceOn", "TurnOn"},
        })

    truth = {s["key"]: np.zeros(n, dtype=bool) for s in specs}
    fired = {s["key"]: np.zeros(n, dtype=bool) for s in specs if s["has_rule"]}
    diag = {s["key"]: {"rigid_error": [], "confirmation_count": [],
                       "vertical_progress": [], "gripper_gap": [],
                       "n_points": []} for s in specs if s["skill"] == "Pick"}

    # Per-frame gripper signals, kept so the stateful Place rule can be
    # started at the phase's true start rather than at frame 0.
    gq = np.full((n, 2), np.nan)
    gc = np.zeros(n, dtype=bool)
    eq = np.full((n, 4), np.nan)

    clouds, checkers, start_pos = {}, {}, {}
    state = {"ready": False}

    def setup(obs):
        for s in specs:
            if s["item_instance"] is None and s["arguments"]:
                s["item_instance"] = resolve_target_instance(env, s["arguments"], s["skill"])
        for s in specs:
            if s["item_instance"] and s["skill"] == "Pick":
                clouds[s["key"]] = PickTargetPointCloud(env, ENV_IMG_RES)
                checkers[s["key"]] = PickCompletionChecker()
        for s in specs:
            if s["item_instance"]:
                start_pos[s["key"]] = np.asarray(
                    object_position(base, s["item_instance"]), dtype=np.float64
                ).copy()
        unresolved = [s["key"] for s in specs if s["item"] and s["item_instance"] is None]
        if unresolved:
            print(f"  !! WARNING unresolved instances (skipped): {unresolved}",
                  file=sys.stderr, flush=True)
        state["ready"] = True

    def record(t, obs, action):
        if not state["ready"]:
            setup(obs)
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        gripper_closed = bool(action_array[-1] > 0.0) if len(action_array) else False
        qpos = obs.get("robot0_gripper_qpos")
        gc[t] = gripper_closed
        if qpos is not None:
            gq[t] = np.asarray(qpos, dtype=np.float64)
        eef_pos = obs.get("robot0_eef_pos")
        eef_quat = obs.get("robot0_eef_quat")
        if eef_quat is not None:
            eq[t] = np.asarray(eef_quat, dtype=np.float64)

        for s in specs:
            # ---- truth (all skills)
            if s["skill"] == "Pick" and s["item_instance"]:
                truth[s["key"]][t] = pick_truth(env, s["item_instance"], start_pos[s["key"]])
            elif s["skill"] in PREDICATE_SKILLS:
                pred = PREDICATE_SKILLS[s["skill"]]
                if s["skill"] in {"PlaceIn", "PlaceOn"}:
                    expr = [s["item"], s["target"]]
                else:
                    expr = [s["target"] or "ensure"]
                truth[s["key"]][t] = relation_truth(env, pred, expr)

            # ---- rule (Pick only for now)
            if s["has_rule"] and s["key"] in checkers:
                pts = clouds[s["key"]].points(obs, s["item_instance"])
                done = checkers[s["key"]].update(
                    target_points=pts, eef_pos=eef_pos, eef_quat=eef_quat,
                    gripper_closed=gripper_closed, gripper_qpos=qpos, frame=t,
                )
                fired[s["key"]][t] = done
                d = diag[s["key"]]
                d["rigid_error"].append(checkers[s["key"]].last_rigid_error)
                d["confirmation_count"].append(checkers[s["key"]].confirmation_count)
                d["vertical_progress"].append(checkers[s["key"]].vertical_progress)
                d["gripper_gap"].append(checkers[s["key"]].last_gripper_gap)
                d["n_points"].append(0 if pts is None else int(len(pts)))

    def finalize():
        out = _summarize(specs, checkers, truth, fired, diag, dump_traces)
        _score_place(specs, out, gq, gc, n, open_frames)
        _score_turnon(specs, out, eq, gc, n)
        return out
    return record, finalize


def _score_place(specs, out, gq, gc, n, open_frames=16):
    """Score the stateful Place rule, started where the phase truly starts.

    ReleaseSkillCompletion detects the gripper opening, while the truth is
    "item is in/on the target", which only becomes true once the item has
    settled.  A negative delta is therefore expected and correct.
    """
    from memory_system.execute.skill_completion.place import ReleaseSkillCompletion

    by_key = {row["key"]: row for row in out}
    for i, spec in enumerate(specs):
        if spec["skill"] not in {"PlaceIn", "PlaceOn"}:
            continue
        row = by_key[spec["key"]]
        row["rule_active"] = True
        if not spec["item_instance"]:
            row["rule_fired"] = None
            continue

        # The phase starts when the preceding phase truly completed; that is
        # the only non-circular boundary available (it comes from the truth).
        start = 0
        if i > 0:
            prev_rise = by_key[specs[i - 1]["key"]].get("truth_rise")
            if prev_rise is not None:
                start = int(prev_rise)

        # closed_confirmed=True mirrors the phase machine, which starts a
        # Place phase only after the preceding Pick reported completion, i.e.
        # with the item already held.  With False the baseline gets sampled
        # while the fingers are still closing and is far too wide.
        checker = ReleaseSkillCompletion(closed_confirmed=True,
                                         required_open_frames=int(open_frames))
        fire = None
        for t in range(start, n):
            if not np.isfinite(gq[t]).all():
                continue
            decision = checker.observe_frame(
                gripper_closed=bool(gc[t]), gripper_qpos=gq[t]
            )
            if decision.advance:
                fire = t
                break
        row["rule_fired"] = fire
        rise = row.get("truth_rise")
        if rise is not None and fire is not None:
            row["delta"] = int(fire - rise)


def _score_turnon(specs, out, eq, gc, n):
    """Score the TurnOn rule, which anchors its baseline on the grasp.

    The rule measures the end-effector rotation since the first closed-gripper
    frame, so it has to be fed from before the grasp -- starting at the phase
    boundary is what the real phase machine does.
    """
    from memory_system.execute.skill_completion.turnon import TurnOnCompletion

    by_key = {row["key"]: row for row in out}
    for i, spec in enumerate(specs):
        if spec["skill"] != "TurnOn":
            continue
        row = by_key[spec["key"]]
        row["rule_active"] = True

        start = 0
        if i > 0:
            prev_rise = by_key[specs[i - 1]["key"]].get("truth_rise")
            if prev_rise is not None:
                start = int(prev_rise)

        checker = TurnOnCompletion()
        fire = None
        for t in range(start, n):
            if not np.isfinite(eq[t]).all():
                continue
            decision = checker.observe_frame(
                eef_quat=eq[t], gripper_closed=bool(gc[t])
            )
            if decision.advance:
                fire = t
                break
        row["rule_fired"] = fire
        rise = row.get("truth_rise")
        if rise is not None and fire is not None:
            row["delta"] = int(fire - rise)


def _summarize(specs, checkers, truth, fired, diag, dump_traces):
    out_phases = []
    for s in specs:
        key = s["key"]
        tt = truth[key]
        row = {
            "key": key, "skill": s["skill"],
            "item": s["item"], "target": s["target"],
            "item_instance": s["item_instance"],
            "truth_frames": int(tt.sum()),
            "rule_active": key in checkers,
        }
        if s["has_rule"]:
            rise = rising_edge(tt)
            row["truth_rise"] = rise
            row["truth_fall"] = falling_edge(tt, rise)
            ft = fired[key]
            row["rule_fired"] = rising_edge(ft, k=1)
            row["rule_frames_true"] = int(ft.sum())
            if rise is not None and row["rule_fired"] is not None:
                row["delta"] = int(row["rule_fired"] - rise)
            if dump_traces:
                row["traces"] = {k: v for k, v in diag.get(key, {}).items()}
        else:
            # no rule exists for this skill (TimeoutOnlySkillCompletion)
            row["truth_rise"] = rising_edge(tt)
            row["rule_fired"] = None
        out_phases.append(row)

    return out_phases


# -------------------------------------------------------------------- main

DEMO_RE = re.compile(r"^(?P<task>.+)_demo\.hdf5$")


def load_demos(path):
    """LIBERO demo hdf5: data/demo_N/{actions, states} with full sim states."""
    with h5py.File(path, "r") as f:
        d = f["data"]
        names = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
        return [
            {
                "actions": np.asarray(d[n]["actions"], dtype=np.float64),
                "states": np.asarray(d[n]["states"], dtype=np.float64),
            }
            for n in names
        ]


def print_phases(phases):
    for row in phases:
        line = (f"        {row['key']:48s} inst={row['item_instance']} "
                f"truth_rise={row['truth_rise']} rule_fired={row['rule_fired']}")
        if "delta" in row:
            line += f" delta={row['delta']}"
        print(line)


def run_demo(args):
    """Restore each demo frame from its recorded sim state (no replay)."""
    from libero.libero import benchmark
    from memory_system.pointcloud_action.offline.demo_state_restore import (
        restore_demo_frame,
    )

    if args.demo_file:
        files = [args.demo_file]
    elif args.demo_dir:
        files = sorted(glob.glob(os.path.join(args.demo_dir, "*_demo.hdf5")))
    else:
        print("need --demo-dir or --demo-file", file=sys.stderr)
        return 1
    if not files:
        print(f"no demo files under {args.demo_dir}", file=sys.stderr)
        return 1

    suite = benchmark.get_benchmark_dict()[args.suite]()
    name_to_id = {suite.get_task(i).name: i for i in range(suite.n_tasks)}

    results = []
    for path in files:
        task_name = DEMO_RE.match(Path(path).name).group("task")
        if task_name not in name_to_id:
            print(f"skip {task_name}: not in suite {args.suite}", file=sys.stderr)
            continue
        task_id = name_to_id[task_name]
        demos = load_demos(path)
        if args.limit:
            demos = demos[: args.limit]
        elif args.n_demos:
            demos = demos[: args.n_demos]

        env, task, _ = build_env(args.suite, task_id, args.seed)
        try:
            env.reset()
            for di, demo in enumerate(demos):
                n = len(demo["actions"])
                record, finalize = make_recorder(env, task, n,
                                                 dump_traces=args.dump_traces,
                                                 open_frames=args.open_frames)
                for t in range(n):
                    obs = restore_demo_frame(env, demo["states"][t],
                                             demo["actions"], t)
                    record(t, obs, demo["actions"][t])
                phases = finalize()
                # Every demo is success_only, so the BDDL goal must hold on the
                # final frame.  This validates the truth channel independently
                # of any rule: a goal predicate that reads False here means our
                # truth expression for that skill is wrong.
                if args.validate_truth:
                    last = n - 1
                    restore_demo_frame(env, demo["states"][last],
                                       demo["actions"], last)
                    goal = getattr(_base(env), "parsed_problem", {}).get(
                        "goal_state", [])
                    checks = {}
                    for g in goal:
                        label = " ".join(str(x) for x in g)
                        try:
                            checks[label] = bool(_base(env)._eval_predicate(list(g)))
                        except Exception as exc:
                            checks[label] = type(exc).__name__
                    try:
                        checks["<check_success>"] = bool(_base(env)._check_success())
                    except Exception as exc:
                        checks["<check_success>"] = type(exc).__name__
                    print(f"        GOAL {task_name} demo={di}: {checks}")
                    phases_goal = checks
                else:
                    phases_goal = None
                results.append({
                    "goal_check": phases_goal,
                    "file": Path(path).name, "task": task_name,
                    "task_id": task_id, "demo": di, "n_frames": n,
                    "phases": phases,
                })
                print(f"[---] {task_name} demo={di} frames={n}")
                print_phases(phases)
                sys.stdout.flush()
        finally:
            env.close()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}  ({len(results)} demos)")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=["fidelity", "measure", "demo"],
                    default="fidelity")
    ap.add_argument("--hdf5", default="")
    ap.add_argument("--hdf5-glob", default="")
    ap.add_argument("--demo-dir", default="")
    ap.add_argument("--demo-file", default="")
    ap.add_argument("--n-demos", type=int, default=10)
    ap.add_argument("--open-frames", type=int, default=5,
                    help="consecutive open frames the Place rule requires")
    ap.add_argument("--suite", default="libero_10_swap")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump-traces", action="store_true")
    ap.add_argument("--validate-truth", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.mode == "demo":
        if not (args.demo_dir or args.demo_file):
            ap.error("--mode demo needs --demo-dir or --demo-file")
        return run_demo(args)

    if args.hdf5:
        paths = [args.hdf5]
    elif args.hdf5_glob:
        paths = sorted(glob.glob(args.hdf5_glob, recursive=True))
    else:
        ap.error("need --hdf5 or --hdf5-glob")
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("no episode files matched", file=sys.stderr)
        return 1

    results = []
    for path in paths:
        meta = parse_episode_path(path)
        episode = load_episode(path)
        env, task, init_states = build_env(args.suite, meta["task_id"], args.seed)
        # ep is a GLOBAL episode counter in some runs (e.g. the regen_demo set
        # numbers task=1 as 51..100), so wrap it onto the per-task init-state
        # list instead of using it directly.  Reduces to ep-1 for runs whose
        # counter is per-task.
        episode_idx = (meta["ep"] - 1) % len(init_states)
        try:
            init = init_states[episode_idx]
            if args.mode == "fidelity":
                payload = check_fidelity(env, init, episode,
                                         tolerance=args.tolerance)
                ok = all(v["n_bad"] == 0 for v in payload.values())
                print(f"[{'OK ' if ok else 'BAD'}] {Path(path).name}")
                for name, v in payload.items():
                    print(f"        {name:14s} max|d|={v['max_abs_diff']:.3e} "
                          f"bad={v['n_bad']}/{v['n_frames']} "
                          f"first_bad={v['first_bad_index']}")
                record = {"fidelity": payload}
            else:
                n = len(episode["actions"])
                rec, finalize = make_recorder(env, task, n,
                                              dump_traces=args.dump_traces,
                                              open_frames=args.open_frames)
                replay(env, init, episode["actions"], record_fn=rec)
                phases = finalize()
                print(f"[---] {Path(path).name}")
                print_phases(phases)
                record = {"n_frames": n, "phases": phases}
        finally:
            env.close()

        results.append({"file": Path(path).name, "task_id": meta["task_id"],
                        "ep": meta["ep"], "success": meta["success"], **record})
        sys.stdout.flush()

    if args.mode == "fidelity":
        n_ok = sum(1 for r in results
                   if all(v["n_bad"] == 0 for v in r["fidelity"].values()))
        print(f"\n=== {n_ok}/{len(results)} episodes replay 1:1 ===")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
