import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import run_old_bs_pipeline as pipeline


class OldBsPipelineTests(unittest.TestCase):
    def make_stage(self, root, frames, failed_key=None, limit=None):
        directory = Path(root) / "stage"
        directory.mkdir()
        manifest = {
            "arguments": {
                "frames": frames,
                "loads_mbit": list(pipeline.LOADS),
                "task_std_mbit": pipeline.TASK_STD_MBIT,
                "users_per_bs": [pipeline.USERS_PER_BS],
                "scenario_seeds": list(pipeline.SCENARIO_SEEDS),
                "policy_seeds": list(pipeline.POLICY_SEEDS),
                "budget_fractions": [pipeline.BUDGET_FRACTION],
                "workers": 8,
                "device": "cpu",
            }
        }
        (directory / "manifest.json").write_text(json.dumps(manifest))
        keys = sorted(pipeline.expected_keys())
        if limit is not None:
            keys = keys[:limit]
        for key in keys:
            case, treatment, environment, policy = key
            passed = key != failed_key
            row = {
                "case": case,
                "load_mbit": float(case.split("_")[0][1:]),
                "treatment": treatment,
                "scenario_seed": environment,
                "policy_seed": policy,
                "scenario_hash": f"hash-{case}-{environment}-{policy}",
                "frames": frames,
                "metrics": {
                    "PAoI": 2.0 if treatment == "legacy" else 1.5,
                    "Q_Mbit_per_user": 0.1,
                    "E_BS_J_per_node": 50.0,
                    "max_energy_queue_slope_J_per_frame": 0.001,
                    "Q_slope_Mbit_per_user_per_frame": -0.001 if passed else 1e-5,
                    "screen_pass": passed,
                },
            }
            stem = pipeline.artifact_stem(directory, key)
            stem.with_suffix(".json").write_text(json.dumps(row))
            with zipfile.ZipFile(stem.with_suffix(".npz"), "w") as archive:
                archive.writestr("workload.npy", b"artifact")
        (directory / "summary.json").write_text(json.dumps({"failures": []}))
        return directory

    def test_complete_stage_advances_only_when_all_budgeted_runs_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.make_stage(temporary, pipeline.SHORT_FRAMES)
            audit = pipeline.audit_directory(directory, pipeline.SHORT_FRAMES)
        self.assertTrue(audit["complete"])
        self.assertTrue(audit["strict_gate_pass"])
        self.assertEqual(audit["budgeted_passed"], 30)

    def test_one_budgeted_failure_stops_strict_gate(self):
        failed = (
            pipeline.case_name(10), "budgeted_0p75",
            pipeline.SCENARIO_SEEDS[0], pipeline.POLICY_SEEDS[0],
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.make_stage(
                temporary, pipeline.SHORT_FRAMES, failed_key=failed
            )
            audit = pipeline.audit_directory(directory, pipeline.SHORT_FRAMES)
        self.assertTrue(audit["complete"])
        self.assertFalse(audit["strict_gate_pass"])
        self.assertEqual(audit["budgeted_passed"], 29)

    def test_incomplete_stage_never_passes_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.make_stage(
                temporary, pipeline.SHORT_FRAMES, limit=1
            )
            audit = pipeline.audit_directory(directory, pipeline.SHORT_FRAMES)
        self.assertFalse(audit["complete"])
        self.assertFalse(audit["strict_gate_pass"])
        self.assertEqual(audit["completed_runs"], 1)
        self.assertEqual(audit["missing_runs"], 59)

    def test_runner_command_preserves_frozen_design_and_resume(self):
        resume = Path("relative-results")
        command = pipeline.runner_command(2048, 8, "cpu", resume)
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], str(pipeline.RUNNER))
        self.assertEqual(command[command.index("--frames") + 1], "2048")
        self.assertEqual(
            command[command.index("--loads-mbit") + 1:
                    command.index("--task-std-mbit")],
            ["10", "12", "16"],
        )
        self.assertEqual(command[-2], "--resume")
        self.assertEqual(command[-1], str(resume.resolve()))


if __name__ == "__main__":
    unittest.main()
