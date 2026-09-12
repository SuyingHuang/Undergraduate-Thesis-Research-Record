"""Run the frozen old-BS Phase-B experiment as a resumable pipeline.

The controller finishes the 2048-frame screen, audits every artifact, and
starts the 4096-frame confirmation only if all 30 budgeted runs pass the
pre-registered strict screen.  It intentionally contains no mechanism for
changing the selected budget after seeing Phase-B data.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "analysis" / "run_old_bs_generalization.py"
RESULT_PARENT = ROOT / "results" / "old_bs_generalization"

LOADS = (10.0, 12.0, 16.0)
TASK_STD_MBIT = 3.0
USERS_PER_BS = 10
SCENARIO_SEEDS = (155921, 196613, 238919, 275015, 314159)
POLICY_SEEDS = (456, 789)
BUDGET_FRACTION = 0.75
TREATMENTS = ("legacy", "budgeted_0p75")
SHORT_FRAMES = 2048
LONG_FRAMES = 4096
KEY_METRICS = (
    "PAoI",
    "Q_Mbit_per_user",
    "E_BS_J_per_node",
    "max_energy_queue_slope_J_per_frame",
    "Q_slope_Mbit_per_user_per_frame",
)


class PipelineInterrupted(Exception):
    """Raised after an external stop request has been recorded."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def case_name(load):
    return f"L{float(load):g}_std{TASK_STD_MBIT:g}_J{USERS_PER_BS}"


def expected_keys():
    return {
        (case_name(load), treatment, environment, policy)
        for load in LOADS
        for environment in SCENARIO_SEEDS
        for policy in POLICY_SEEDS
        for treatment in TREATMENTS
    }


def artifact_stem(directory, key):
    case, treatment, environment, policy = key
    return directory / f"{case}_{treatment}_env{environment}_policy{policy}"


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_errors(directory, frames):
    path = directory / "manifest.json"
    if not path.is_file():
        return ["missing manifest.json"]
    try:
        arguments = _read_json(path)["arguments"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"invalid manifest.json: {exc}"]
    expected = {
        "frames": frames,
        "loads_mbit": list(LOADS),
        "task_std_mbit": TASK_STD_MBIT,
        "users_per_bs": [USERS_PER_BS],
        "scenario_seeds": list(SCENARIO_SEEDS),
        "policy_seeds": list(POLICY_SEEDS),
        "budget_fractions": [BUDGET_FRACTION],
    }
    return [
        f"manifest {name}={arguments.get(name)!r}, expected {value!r}"
        for name, value in expected.items()
        if arguments.get(name) != value
    ]


def audit_directory(directory, frames):
    """Audit a stage without treating an unfinished stage as corrupt."""
    directory = Path(directory).resolve()
    expected = expected_keys()
    errors = _manifest_errors(directory, frames)
    rows = {}
    missing = []

    for key in sorted(expected):
        stem = artifact_stem(directory, key)
        json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
        if not json_path.is_file() or not npz_path.is_file():
            missing.append("/".join(map(str, key)))
            continue
        try:
            if not zipfile.is_zipfile(npz_path):
                raise ValueError(f"invalid NPZ archive {npz_path.name}")
            with zipfile.ZipFile(npz_path) as archive:
                if not any(name.endswith(".npy") for name in archive.namelist()):
                    raise ValueError(f"empty NPZ archive {npz_path.name}")
            row = _read_json(json_path)
            actual = (
                row["case"], row["treatment"], row["scenario_seed"],
                row["policy_seed"],
            )
            if actual != key:
                raise ValueError(f"identity {actual!r} does not match filename")
            if row["frames"] != frames:
                raise ValueError(f"frames={row['frames']}, expected {frames}")
            for metric in KEY_METRICS:
                value = float(row["metrics"][metric])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite metric {metric}")
            if not isinstance(row["metrics"].get("screen_pass"), bool):
                raise ValueError("screen_pass is not boolean")
            rows[key] = row
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{json_path.name}: {exc}")

    expected_json = {
        artifact_stem(directory, key).with_suffix(".json").name
        for key in expected
    }
    actual_json = {
        path.name for path in directory.glob("*_env*_policy*.json")
    }
    for name in sorted(actual_json - expected_json):
        errors.append(f"unexpected run artifact {name}")

    # A completed matched block must use one environment workload realization.
    blocks = {}
    for key, row in rows.items():
        block = (key[0], key[2], key[3])
        blocks.setdefault(block, set()).add(row.get("scenario_hash"))
    for block, hashes in blocks.items():
        present = [key for key in rows if (key[0], key[2], key[3]) == block]
        if len(present) == len(TREATMENTS) and (None in hashes or len(hashes) != 1):
            errors.append(f"unmatched workload hashes for block {block!r}")

    summary_path = directory / "summary.json"
    if summary_path.is_file():
        try:
            failures = _read_json(summary_path).get("failures", [])
            if failures:
                errors.append(f"runner recorded {len(failures)} failure(s)")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"invalid summary.json: {exc}")

    budget_rows = {
        key: row for key, row in rows.items() if key[1] == "budgeted_0p75"
    }
    passes_by_load = {}
    for load in LOADS:
        case = case_name(load)
        selected = [row for key, row in budget_rows.items() if key[0] == case]
        passes_by_load[f"{load:g}"] = {
            "passed": sum(row["metrics"]["screen_pass"] for row in selected),
            "completed": len(selected),
            "expected": len(SCENARIO_SEEDS) * len(POLICY_SEEDS),
        }

    complete = not missing and not errors and len(rows) == len(expected)
    gate_pass = complete and all(
        row["metrics"]["screen_pass"] for row in budget_rows.values()
    ) and len(budget_rows) == len(expected) // 2
    return {
        "directory": str(directory),
        "frames": frames,
        "expected_runs": len(expected),
        "completed_runs": len(rows),
        "missing_runs": len(missing),
        "missing_examples": missing[:5],
        "errors": errors,
        "complete": complete,
        "budgeted_completed": len(budget_rows),
        "budgeted_passed": sum(
            row["metrics"]["screen_pass"] for row in budget_rows.values()),
        "passes_by_load": passes_by_load,
        "strict_gate_pass": gate_pass,
        "rows": rows,
    }


def _paired_effects(rows):
    effects = {}
    for load in LOADS:
        case = case_name(load)
        environment_differences = []
        for environment in SCENARIO_SEEDS:
            differences = []
            for policy in POLICY_SEEDS:
                budget = rows.get((case, "budgeted_0p75", environment, policy))
                legacy = rows.get((case, "legacy", environment, policy))
                if budget is not None and legacy is not None:
                    differences.append(
                        budget["metrics"]["PAoI"] - legacy["metrics"]["PAoI"]
                    )
            if len(differences) == len(POLICY_SEEDS):
                environment_differences.append(statistics.mean(differences))
        effects[f"{load:g}"] = environment_differences
    return effects


def public_audit(audit):
    return {key: value for key, value in audit.items() if key != "rows"}


def render_report(state, short=None, long=None):
    lines = [
        "# 旧 BS 阶段 B 自动流水线报告",
        "",
        f"- 状态：`{state['status']}`",
        f"- 更新时间（UTC）：`{state['updated_at']}`",
        "- 决策规则：2048 帧的 30 个 budgeted-75% 运行必须全部严格通过，才允许启动 4096 帧。",
        "",
    ]
    for label, audit in (("2048 帧筛选", short), ("4096 帧确认", long)):
        if audit is None:
            continue
        lines.extend([
            f"## {label}", "",
            f"结果目录：`{audit['directory']}`", "",
            f"完整运行：{audit['completed_runs']}/{audit['expected_runs']}；"
            f"budgeted 通过：{audit['budgeted_passed']}/{audit['budgeted_completed']}；"
            f"严格门槛：`{audit['strict_gate_pass']}`。", "",
            "| L (Mbit) | budgeted 通过/完成 | 计划 |", "|---:|---:|---:|",
        ])
        for load, counts in audit["passes_by_load"].items():
            lines.append(
                f"| {load} | {counts['passed']}/{counts['completed']} | {counts['expected']} |"
            )
        if audit["errors"]:
            lines.extend(["", "审计错误："])
            lines.extend(f"- {error}" for error in audit["errors"])

        failed = [
            row for key, row in audit["rows"].items()
            if key[1] == "budgeted_0p75" and not row["metrics"]["screen_pass"]
        ]
        if failed:
            lines.extend([
                "", "严格失败的连续指标：", "",
                "| L | 环境 | 策略 | 能量队列斜率 | 物理队列斜率 |",
                "|---:|---:|---:|---:|---:|",
            ])
            for row in sorted(failed, key=lambda item: (
                    item["load_mbit"], item["scenario_seed"], item["policy_seed"])):
                metrics = row["metrics"]
                lines.append(
                    f"| {row['load_mbit']:g} | {row['scenario_seed']} | {row['policy_seed']} | "
                    f"{metrics['max_energy_queue_slope_J_per_frame']:.8g} | "
                    f"{metrics['Q_slope_Mbit_per_user_per_frame']:.8g} |"
                )

        effects = _paired_effects(audit["rows"])
        available = {load: values for load, values in effects.items() if values}
        if available:
            lines.extend([
                "", "PAoI 配对差（budgeted − legacy；先在环境内平均策略种子）：", "",
                "| L | 完整环境数 | 均值 | 中位数 | 最小值 | 最大值 |",
                "|---:|---:|---:|---:|---:|---:|",
            ])
            for load, values in available.items():
                lines.append(
                    f"| {load} | {len(values)} | {statistics.mean(values):.8g} | "
                    f"{statistics.median(values):.8g} | {min(values):.8g} | {max(values):.8g} |"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def atomic_write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save(control_dir, state, short=None, long=None):
    state["updated_at"] = utc_now()
    atomic_write_json(control_dir / "state.json", state)
    (control_dir / "report.md").write_text(
        render_report(state, short, long), encoding="utf-8"
    )


def runner_command(frames, workers, device, resume=None):
    command = [
        sys.executable, str(RUNNER), "--frames", str(frames),
        "--loads-mbit", *(f"{value:g}" for value in LOADS),
        "--task-std-mbit", f"{TASK_STD_MBIT:g}",
        "--users-per-bs", str(USERS_PER_BS),
        "--scenario-seeds", *(str(value) for value in SCENARIO_SEEDS),
        "--policy-seeds", *(str(value) for value in POLICY_SEEDS),
        "--budget-fractions", f"{BUDGET_FRACTION:g}",
        "--workers", str(workers), "--device", device,
    ]
    if resume is not None:
        command.extend(("--resume", str(Path(resume).resolve())))
    return command


def _stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=10)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_stage(stage, frames, control_dir, state, workers, device, resume=None,
              report_short=None, report_long=None):
    command = runner_command(frames, workers, device, resume)
    log_path = control_dir / f"{stage}.log"
    state.update({"status": f"{stage}_running", "active_stage": stage,
                  "last_command": command})
    if resume is not None:
        state[f"{stage}_directory"] = str(Path(resume).resolve())
    save(control_dir, state, report_short, report_long)

    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{utc_now()}] {' '.join(command)}\n")
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                candidate = Path(line.strip())
                if not state.get(f"{stage}_directory") and candidate.is_absolute():
                    try:
                        if candidate.parent.resolve() == RESULT_PARENT.resolve():
                            state[f"{stage}_directory"] = str(candidate.resolve())
                            save(control_dir, state, report_short, report_long)
                    except OSError:
                        pass
            return_code = process.wait()
        except (KeyboardInterrupt, PipelineInterrupted):
            _stop_process(process)
            raise
    if return_code:
        raise RuntimeError(f"{stage} runner exited with status {return_code}")
    directory = state.get(f"{stage}_directory")
    if not directory:
        raise RuntimeError(f"{stage} runner did not report its result directory")
    return Path(directory)


@contextmanager
def exclusive_lock(control_dir):
    path = control_dir / "pipeline.lock"
    with path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"pipeline already active for {control_dir}") from exc
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        yield


def audit_known_stage(state, key, frames):
    directory = state.get(key)
    if not directory:
        return None
    try:
        return audit_directory(directory, frames)
    except Exception:
        return None


def load_or_create_state(control_dir, short_directory=None, long_directory=None):
    path = control_dir / "state.json"
    if path.is_file():
        state = _read_json(path)
    else:
        state = {"created_at": utc_now(), "status": "new"}
    supplied = {
        "short_directory": short_directory,
        "long_directory": long_directory,
    }
    for key, value in supplied.items():
        if value is None:
            continue
        resolved = str(Path(value).resolve())
        if state.get(key) not in (None, resolved):
            raise ValueError(f"{key} conflicts with existing pipeline state")
        state[key] = resolved
    return state


def execute(args):
    if args.control_dir:
        control_dir = args.control_dir.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        control_dir = ROOT / "results" / "old_bs_pipeline" / stamp
    control_dir.mkdir(parents=True, exist_ok=True)

    with exclusive_lock(control_dir):
        state = load_or_create_state(
            control_dir, args.resume_short, args.resume_long
        )
        short = None
        long = None
        try:
            short_dir = state.get("short_directory")
            if args.evaluate_only:
                if short_dir:
                    short = audit_directory(short_dir, SHORT_FRAMES)
                if state.get("long_directory"):
                    long = audit_directory(state["long_directory"], LONG_FRAMES)
                state["status"] = "evaluated_only"
                save(control_dir, state, short, long)
                return control_dir

            if not short_dir or not audit_directory(short_dir, SHORT_FRAMES)["complete"]:
                short_dir = run_stage(
                    "short", SHORT_FRAMES, control_dir, state,
                    args.workers, args.device, short_dir,
                )
            short = audit_directory(short_dir, SHORT_FRAMES)
            state["short_audit"] = public_audit(short)
            if not short["complete"]:
                state["status"] = "short_incomplete_or_invalid"
                save(control_dir, state, short)
                return control_dir
            if not short["strict_gate_pass"]:
                state["status"] = "stopped_strict_gate_failed_2048"
                save(control_dir, state, short)
                return control_dir

            long_dir = state.get("long_directory")
            if not long_dir or not audit_directory(long_dir, LONG_FRAMES)["complete"]:
                long_dir = run_stage(
                    "long", LONG_FRAMES, control_dir, state,
                    args.workers, args.device, long_dir,
                    report_short=short,
                )
            long = audit_directory(long_dir, LONG_FRAMES)
            state["long_audit"] = public_audit(long)
            if not long["complete"]:
                state["status"] = "long_incomplete_or_invalid"
            elif long["strict_gate_pass"]:
                state["status"] = "complete_confirmed"
            else:
                state["status"] = "complete_long_gate_failed"
            save(control_dir, state, short, long)
            return control_dir
        except (KeyboardInterrupt, PipelineInterrupted):
            short = audit_known_stage(state, "short_directory", SHORT_FRAMES)
            long = audit_known_stage(state, "long_directory", LONG_FRAMES)
            state["status"] = "paused"
            save(control_dir, state, short, long)
            return control_dir
        except Exception as exc:
            short = audit_known_stage(state, "short_directory", SHORT_FRAMES)
            long = audit_known_stage(state, "long_directory", LONG_FRAMES)
            state["status"] = "runner_or_pipeline_failed"
            state["error"] = repr(exc)
            save(control_dir, state, short, long)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=Path)
    parser.add_argument("--resume-short", type=Path)
    parser.add_argument("--resume-long", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="audit known result directories without launching simulations",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")

    previous_handlers = {}
    def request_stop(signum, _frame):
        raise PipelineInterrupted(f"received signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    try:
        directory = execute(args)
        print(f"Pipeline control directory: {directory}")
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
