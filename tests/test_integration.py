"""Integration tests - full pipeline with generate_measurement_data().

These tests verify the complete workflow. The provided helper function generates
the simulated experiment data, then used to construct ``MeasurementRecord``,
which is saved to disk, then loaded back, finally verify the data and metadata integrity
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from device_data_pipeline import (
    load_record,
    save_record,
)
from device_data_pipeline.adapters import build_record
from device_data_pipeline.generate_measurement import generate_measurement_data

# ----------------------------------------------------------------------------------------
#  Full Pipeline Integration tests
# ----------------------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end tests with helper generate_measurement_data()."""

    def test_roundtrip_with_generated_data(self, tmp_path: Path) -> None:
        np.random.seed(42)
        raw = generate_measurement_data()
        original_record = build_record(raw=raw)

        path = save_record(record=original_record, directory=tmp_path)
        restored_record = load_record(path=path)

        npt.assert_allclose(
            original_record.measured_data, restored_record.measured_data
        )
        assert restored_record.metadata.experiment_type == raw["mode"]
        assert len(restored_record.sweep_data) == len(original_record.sweep_data)
        for orig_sweep, restored_sweep in zip(
            original_record.sweep_data, restored_record.sweep_data
        ):
            npt.assert_allclose(orig_sweep, restored_sweep)

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_multiple_random_experiments(self, tmp_path: Path, seed: int) -> None:
        """Verify roundtrip correctness across multiple randomly generated experiments."""
        np.random.seed(seed=seed)
        raw = generate_measurement_data()
        original_record = build_record(raw=raw)

        path = save_record(record=original_record, directory=tmp_path)
        restored_record = load_record(path=path)

        npt.assert_allclose(
            original_record.measured_data, restored_record.measured_data
        )
        assert restored_record.metadata.experiment_type == raw["mode"]
        assert len(restored_record.sweep_data) == len(original_record.sweep_data)

    def test_csv_row_count_matches_data(self, tmp_path: Path) -> None:
        """Verify the row counts in file saved on disk"""
        np.random.seed(42)
        raw = generate_measurement_data()
        original_record = build_record(raw=raw)

        path = save_record(record=original_record, directory=tmp_path)
        csv_file = path.parent / f"{path.name}_data.csv"
        lines_csv = csv_file.read_text().strip().split("\n")

        # Uses .size instead of len() because we flattened N-dimensional elements
        assert len(lines_csv) == original_record.measured_data.size + 1

    def test_json_metadata_fields(self, tmp_path: Path) -> None:
        """Verify the fields from raw data matches data saved on disk"""
        np.random.seed(42)
        raw = generate_measurement_data()
        original_record = build_record(raw=raw)

        path = save_record(record=original_record, directory=tmp_path)
        json_file = path.parent / f"{path.name}_metadata.json"
        with open(json_file, encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["experiment_type"] == raw["mode"]
        assert "start_time" in meta
        assert "end_time" in meta
        assert meta["pc_name"] == "integration-test-pc"
        assert "sweep_parameters" in meta

    def test_time_trace_produces_no_sweep_axis(self, tmp_path: Path) -> None:
        raw = {
            "mode": "time-trace",
            "measured": np.random.normal(size=50),
            "measured_params": "current",
            "started_at": "2026-01-01T01:00:00",
            "ended_at": "2026-01-01T01:00:02",
        }

        record = build_record(raw=raw)

        assert len(record.sweep_data) == 0, "time-trace sweep data must be empty"
        assert len(record.metadata.sweep_parameters) == 0

        path = save_record(record=record, directory=tmp_path)
        restored_record = load_record(path=path)

        assert len(restored_record.sweep_data) == 0

    def test_missing_measured_key_raises_clear_error(self) -> None:
        bad_raw = {
            "mode": "1D",
            # 'measured' key intentionally missing
            "started_at": "2026-01-01T01:00:00",
            "ended_at": "2026-01-01T01:00:02",
            "swept_params": "voltage",
        }
        with pytest.raises(KeyError, match="'measured' key missing"):
            build_record(bad_raw)
