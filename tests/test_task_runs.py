# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

"""
Unit tests for validating that tasks run successfully and produce expected output files.

These tests run the main scripts and verify that output files are created.
"""

import json
import pathlib
import subprocess
import tempfile
import unittest

# Shared temporary directory for all tests
_SHARED_TEMP_DIR = None
_SHARED_TEMP_PATH = None
_TASK_RUNS_PASSED = {}


def setUpModule():
    """Create a shared temporary directory for all tests in this module."""
    global _SHARED_TEMP_DIR, _SHARED_TEMP_PATH
    _SHARED_TEMP_DIR = tempfile.mkdtemp()
    _SHARED_TEMP_PATH = pathlib.Path(_SHARED_TEMP_DIR)


def tearDownModule():
    """Clean up the shared temporary directory."""
    global _SHARED_TEMP_DIR
    import shutil

    if _SHARED_TEMP_DIR:
        shutil.rmtree(_SHARED_TEMP_DIR, ignore_errors=True)


class TestTaskRuns(unittest.TestCase):
    """Test that task runs complete successfully and produce output files."""

    @classmethod
    def setUpClass(cls):
        """Use the shared temporary directory."""
        cls.temp_dir = _SHARED_TEMP_DIR
        cls.temp_path = _SHARED_TEMP_PATH

    def _run_task(
        self, config_name: str, output_file: str, time_steps: int = 1
    ) -> tuple[bool, str]:
        """
        Run a task with the given configuration.

        Args:
            config_name: Name of the config file (without .yaml)
            output_file: Name of the output JSON file
            time_steps: Number of time steps (default: 1)

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        cmd = [
            "python",
            "-m",
            "scripts.run",
            "--config-name",
            f"test/{config_name}",  # Use test configs
            f"output_dir={self.temp_dir}",
            f"output_file={output_file}",
            f"time_steps={time_steps}",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=pathlib.Path(__file__).parent.parent,
            )

            output_path = self.temp_path / output_file
            file_exists = output_path.exists()

            if not file_exists:
                return (
                    False,
                    f"Output file {output_file} not created. Return code: {result.returncode}",
                )

            # Verify the JSON file is valid
            try:
                with open(output_path, "r") as f:
                    data = json.load(f)
                # Check for essential keys
                if "task_name" not in data:
                    return False, f"Output file {output_file} missing 'task_name' key"
                return True, ""
            except json.JSONDecodeError as e:
                return False, f"Output file {output_file} is not valid JSON: {e}"

        except subprocess.TimeoutExpired:
            return False, f"Task {config_name} timed out after 300 seconds"
        except Exception as e:
            return False, f"Task {config_name} failed with exception: {e}"

    # Text-to-Image Tasks (T=1)
    def test_t2i_pos_objects(self):
        """Test text-to-image position objects task."""
        success, error = self._run_task(
            "t2i_pos_objects", "t2i_pos_objects.json", time_steps=1
        )
        _TASK_RUNS_PASSED["t2i_pos_objects"] = success
        self.assertTrue(success, error)

    def test_t2i_num_objects(self):
        """Test text-to-image number of objects task."""
        success, error = self._run_task(
            "t2i_num_objects", "t2i_num_objects.json", time_steps=1
        )
        _TASK_RUNS_PASSED["t2i_num_objects"] = success
        self.assertTrue(success, error)

    def test_t2i_saturation(self):
        """Test text-to-image saturation task."""
        success, error = self._run_task(
            "t2i_saturation", "t2i_saturation.json", time_steps=1
        )
        _TASK_RUNS_PASSED["t2i_saturation"] = success
        self.assertTrue(success, error)

    # LLM Tasks (T=2)
    def test_llm_even_odd(self):
        """Test LLM even/odd task with dialogue."""
        success, error = self._run_task(
            "llm_even_odd", "llm_even_odd.json", time_steps=2
        )
        _TASK_RUNS_PASSED["llm_even_odd"] = success
        self.assertTrue(success, error)

    def test_llm_num_chars(self):
        """Test LLM number of characters task with dialogue."""
        success, error = self._run_task(
            "llm_num_chars", "llm_num_chars.json", time_steps=2
        )
        _TASK_RUNS_PASSED["llm_num_chars"] = success
        self.assertTrue(success, error)

    def test_llm_avg_word_length(self):
        """Test LLM average word length task with dialogue."""
        success, error = self._run_task(
            "llm_avg_word_length", "llm_avg_word_length.json", time_steps=2
        )
        _TASK_RUNS_PASSED["llm_avg_word_length"] = success
        self.assertTrue(success, error)

    def test_llm_formality(self):
        """Test LLM formality task with dialogue."""
        success, error = self._run_task(
            "llm_formality", "llm_formality.json", time_steps=2
        )
        _TASK_RUNS_PASSED["llm_formality"] = success
        self.assertTrue(success, error)


class TestMetricsPlotting(unittest.TestCase):
    """Test that metrics plotting works correctly."""

    @classmethod
    def setUpClass(cls):
        """Use the shared temporary directory."""
        cls.temp_dir = _SHARED_TEMP_DIR
        cls.temp_path = _SHARED_TEMP_PATH

    def _check_task_passed(self, task_name: str):
        """Check if the corresponding task run test passed."""
        if task_name not in _TASK_RUNS_PASSED or not _TASK_RUNS_PASSED[task_name]:
            self.skipTest(
                f"Skipping because task run test for {task_name} did not pass"
            )

    def _run_plot_metrics(
        self, json_file: pathlib.Path, output_file: str, time_step: int = 0
    ) -> tuple[bool, str]:
        """
        Run plot_metrics.py script.

        Args:
            json_file: Path to the input JSON file
            output_file: Name of the output PNG file
            time_step: Time step to plot

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        cmd = [
            "python",
            "scripts/plots/plot_metrics.py",
            "--json",
            str(json_file),
            "--outfile",
            str(self.temp_path / output_file),
            "--time-step",
            str(time_step),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=pathlib.Path(__file__).parent.parent,
            )

            output_path = self.temp_path / output_file
            if not output_path.exists():
                return (
                    False,
                    f"Plot file {output_file} not created. Return code: {result.returncode}\nStderr: {result.stderr}",
                )

            # Check file is not empty
            if output_path.stat().st_size == 0:
                return False, f"Plot file {output_file} is empty"

            return True, ""

        except subprocess.TimeoutExpired:
            return False, f"Plot script timed out after 60 seconds"
        except Exception as e:
            return False, f"Plot script failed with exception: {e}"

    def _run_plot_trajectories(
        self, json_file: pathlib.Path, output_file: str
    ) -> tuple[bool, str]:
        """
        Run plot_trajectories.py script.

        Args:
            json_file: Path to the input JSON file
            output_file: Name of the output PNG file

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        cmd = [
            "python",
            "scripts/plots/plot_trajectories.py",
            "--json",
            str(json_file),
            "--outfile",
            str(self.temp_path / output_file),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=pathlib.Path(__file__).parent.parent,
            )

            output_path = self.temp_path / output_file
            if not output_path.exists():
                return (
                    False,
                    f"Plot file {output_file} not created. Return code: {result.returncode}\nStderr: {result.stderr}",
                )

            # Check file is not empty
            if output_path.stat().st_size == 0:
                return False, f"Plot file {output_file} is empty"

            return True, ""

        except subprocess.TimeoutExpired:
            return False, f"Plot script timed out after 60 seconds"
        except Exception as e:
            return False, f"Plot script failed with exception: {e}"

    # T2I Metrics Tests
    def test_plot_metrics_t2i_pos_objects(self):
        """Test plotting metrics for t2i_pos_objects."""
        self._check_task_passed("t2i_pos_objects")
        json_file = self.temp_path / "t2i_pos_objects.json"
        success, error = self._run_plot_metrics(
            json_file, "metrics_t2i_pos_objects.png", time_step=0
        )
        self.assertTrue(success, error)

    def test_plot_metrics_t2i_num_objects(self):
        """Test plotting metrics for t2i_num_objects."""
        self._check_task_passed("t2i_num_objects")
        json_file = self.temp_path / "t2i_num_objects.json"
        success, error = self._run_plot_metrics(
            json_file, "metrics_t2i_num_objects.png", time_step=0
        )
        self.assertTrue(success, error)

    def test_plot_metrics_t2i_saturation(self):
        """Test plotting metrics for t2i_saturation."""
        self._check_task_passed("t2i_saturation")
        json_file = self.temp_path / "t2i_saturation.json"
        success, error = self._run_plot_metrics(
            json_file, "metrics_t2i_saturation.png", time_step=0
        )
        self.assertTrue(success, error)

    # LLM Metrics Tests (T=2)
    def test_plot_metrics_llm_even_odd(self):
        """Test plotting metrics for llm_even_odd."""
        self._check_task_passed("llm_even_odd")
        json_file = self.temp_path / "llm_even_odd.json"
        success, error = self._run_plot_metrics(
            json_file, "metrics_llm_even_odd.png", time_step=2
        )
        self.assertTrue(success, error)

    def test_plot_metrics_llm_num_chars(self):
        """Test plotting metrics for llm_num_chars."""
        self._check_task_passed("llm_num_chars")
        json_file = self.temp_path / "llm_num_chars.json"
        success, error = self._run_plot_metrics(
            json_file, "metrics_llm_num_chars.png", time_step=2
        )
        self.assertTrue(success, error)

    def test_plot_metrics_llm_avg_word_length(self):
        """Test plotting metrics for llm_avg_word_length."""
        self._check_task_passed("llm_avg_word_length")
        json_file = self.temp_path / "llm_avg_word_length.json"
        success, error = self._run_plot_metrics(
            json_file, "metrics_llm_avg_word_length.png", time_step=2
        )
        self.assertTrue(success, error)

    def test_plot_metrics_llm_formality(self):
        """Test plotting metrics for llm_formality."""
        self._check_task_passed("llm_formality")
        json_file = self.temp_path / "llm_formality.json"
        success, error = self._run_plot_metrics(
            json_file, "metrics_llm_formality.png", time_step=2
        )
        self.assertTrue(success, error)

    # Trajectory Tests
    def test_plot_trajectories_llm_even_odd(self):
        """Test plotting trajectories for llm_even_odd."""
        self._check_task_passed("llm_even_odd")
        json_file = self.temp_path / "llm_even_odd.json"
        success, error = self._run_plot_trajectories(
            json_file, "trajectories_llm_even_odd.png"
        )
        self.assertTrue(success, error)

    def test_plot_trajectories_llm_num_chars(self):
        """Test plotting trajectories for llm_num_chars."""
        self._check_task_passed("llm_num_chars")
        json_file = self.temp_path / "llm_num_chars.json"
        success, error = self._run_plot_trajectories(
            json_file, "trajectories_llm_num_chars.png"
        )
        self.assertTrue(success, error)

    def test_plot_trajectories_llm_avg_word_length(self):
        """Test plotting trajectories for llm_avg_word_length."""
        self._check_task_passed("llm_avg_word_length")
        json_file = self.temp_path / "llm_avg_word_length.json"
        success, error = self._run_plot_trajectories(
            json_file, "trajectories_llm_avg_word_length.png"
        )
        self.assertTrue(success, error)

    def test_plot_trajectories_llm_formality(self):
        """Test plotting trajectories for llm_formality."""
        self._check_task_passed("llm_formality")
        json_file = self.temp_path / "llm_formality.json"
        success, error = self._run_plot_trajectories(
            json_file, "trajectories_llm_formality.png"
        )
        self.assertTrue(success, error)


if __name__ == "__main__":
    unittest.main()
