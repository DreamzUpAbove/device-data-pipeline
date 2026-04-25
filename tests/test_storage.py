"""Tests for device_data_pipeline.storage roundtrip save/load with edge cases"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from device_data_pipeline import (
    MeasurementMetaData,
    MeasurementRecord,
    SweepParameter,
    load_record,
    save_record,
)

# ----------------------------------------------------------------------------------
# Tests for roundtrips
# ----------------------------------------------------------------------------------


class TestSaveLoadRecords:
    """Verify the save->load produces identical data and metadata."""

    def test_1d_sweep_roundtrip(
        self, tmp_path: Path, sample_1d_record: MeasurementRecord
    ) -> None:
        path = save_record(sample_1d_record, directory=tmp_path, name="test_1d")
        restored_record = load_record(path=path)

        npt.assert_allclose(
            restored_record.measured_data, sample_1d_record.measured_data
        )
        assert len(restored_record.sweep_data) == 1
        npt.assert_allclose(
            restored_record.sweep_data[0], sample_1d_record.sweep_data[0]
        )
        assert restored_record.metadata.experiment_type == "1D"
        assert restored_record.metadata.measured_parameter == "current"
        assert restored_record.metadata.sweep_parameters[0].name == "frequency"
        assert restored_record.metadata.sweep_parameters[0].unit == "Hz"
        assert restored_record.metadata.pc_name == "lab-pc-101"

    def test_timetrace_roundtrip(
        self, tmp_path: Path, sample_time_trace_record: MeasurementRecord
    ) -> None:
        path = save_record(sample_time_trace_record, directory=tmp_path, name="test_tt")
        restored_record = load_record(path=path)

        npt.assert_allclose(
            restored_record.measured_data, sample_time_trace_record.measured_data
        )
        assert len(restored_record.sweep_data) == 0
        assert len(restored_record.metadata.sweep_parameters) == 0
        assert restored_record.metadata.experiment_type == "time-trace"

    def test_linescan_roundtrip(
        self, tmp_path: Path, simple_linescan_record: MeasurementRecord
    ) -> None:
        path = save_record(simple_linescan_record, directory=tmp_path, name="test_linescan")
        restored_record = load_record(path=path)

        npt.assert_allclose(
            restored_record.measured_data, simple_linescan_record.measured_data
        )
        assert len(restored_record.sweep_data) == 2
        npt.assert_allclose(
            restored_record.sweep_data[0], simple_linescan_record.sweep_data[0]
        )
        npt.assert_allclose(
            restored_record.sweep_data[1], simple_linescan_record.sweep_data[1]
        )
        assert restored_record.metadata.experiment_type == "line-scan"
        assert restored_record.metadata.measured_parameter == "current"
        assert restored_record.metadata.sweep_parameters[0].name == "pin1"
        assert restored_record.metadata.sweep_parameters[1].name == "pin2"

    def test_timestamp_preserved(
        self, tmp_path: Path, sample_1d_record: MeasurementRecord
    ) -> None:
        path = save_record(sample_1d_record, directory=tmp_path, name="test_ts")
        restored_record = load_record(path=path)

        assert (
            restored_record.metadata.start_time == sample_1d_record.metadata.start_time
        )
        assert restored_record.metadata.end_time == sample_1d_record.metadata.end_time

    def test_user_metadata_preserved(self, tmp_path: Path) -> None:
        start = datetime(2026, 1, 1, 1, 0, 0)
        end = start + timedelta(hours=1)
        user_metadata = {
            "device": "quantum_chip_v2",
            "lab": "London lab_01",
            "temperature": 22.6,
            "tags": ["calibration", "night_run"],
        }
        meta = MeasurementMetaData(
            experiment_type="1D",
            measured_parameter="current",
            sweep_parameters=[SweepParameter(name="frequency", unit="Hz")],
            start_time=start,
            end_time=end,
            pc_name="lab-pc-101",
            user_metadata=user_metadata,
        )
        original_record = MeasurementRecord(
            metadata=meta,
            measured_data=np.array([1.0, 2.0, 3.0]),
            sweep_data=[np.array([0.0, 0.5, 1.0])],
        )
        path = save_record(
            record=original_record, directory=tmp_path, name="test_user_meta"
        )
        restored_record = load_record(path=path)
        assert restored_record.metadata == original_record.metadata
        assert restored_record.metadata.user_metadata["tags"] == [
            "calibration",
            "night_run",
        ]


# ------------------------------------------------------------------------------------------------------------
# Tests for file creation
# ------------------------------------------------------------------------------------------------------------


class TestFileManagement:
    """Verifies the ``save_record`` autogenerates unique timestamped filenames."""

    def test_auto_name_includes_experiment_type(
        self, tmp_path: Path, sample_1d_record: MeasurementRecord
    ) -> None:
        path = save_record(sample_1d_record, directory=tmp_path)
        assert "1D" in path.name

    def test_auto_name_create_csv_and_json(
        self, tmp_path: Path, sample_1d_record: MeasurementRecord
    ) -> None:
        path = save_record(sample_1d_record, directory=tmp_path)
        csv_file = path.parent / f"{path.name}_data.csv"
        json_file = path.parent / f"{path.name}_metadata.json"

        assert csv_file.exists()
        assert json_file.exists()

    def test_custom_name_creates_expected_files(
        self, tmp_path: Path, sample_1d_record: MeasurementRecord
    ) -> None:
        path = save_record(
            sample_1d_record, directory=tmp_path, name="custom_experiment"
        )

        assert "custom_experiment" == path.name
        assert (tmp_path / "custom_experiment_data.csv").exists()
        assert (tmp_path / "custom_experiment_metadata.json").exists()

    def test_save_record_creates_directory_if_missing(
        self, tmp_path: Path, sample_1d_record: MeasurementRecord
    ) -> None:
        nested = tmp_path / "sub" / "dir"
        path = save_record(sample_1d_record, directory=nested, name="deep")
        assert path.parent.exists()
        assert (nested / "deep_data.csv").exists()

    def test_save_record_returns_loadable_path(
        self, tmp_path: Path, sample_1d_record: MeasurementRecord
    ) -> None:
        path = save_record(sample_1d_record, directory=tmp_path)
        restored_record = load_record(path=path)
        assert restored_record.metadata.experiment_type == "1D"

    def test_rapid_saves_produce_distinct_files(
        self, tmp_path: Path, sample_1d_record: MeasurementRecord
    ) -> None:
        paths = [save_record(sample_1d_record, directory=tmp_path) for _ in range(10)]
        assert len(set(paths)) == 10

    def test_save_record_returns_base_path_without_suffix(
        self, tmp_path: Path, sample_1d_record: MeasurementRecord
    ) -> None:
        path = save_record(sample_1d_record, directory=tmp_path)
        assert not str(path).endswith(".csv")
        assert not str(path).endswith(".json")


# -------------------------------------------------------------------------------------------
# Tests for Data Integrity
# -------------------------------------------------------------------------------------------


class TestDataIntegrity:
    """Verify that CSV text format preserves data without corruption."""

    def test_dtype_is_float64(self, tmp_path: Path) -> None:
        start = datetime(2026, 1, 1, 1, 0, 0)
        end = start + timedelta(minutes=1)
        data = np.array([1.23456789012345], dtype=np.float64)
        meta = MeasurementMetaData(
            experiment_type="1D",
            measured_parameter="current",
            sweep_parameters=[],
            start_time=start,
            end_time=end,
            pc_name="lab-pc-101",
        )
        record = MeasurementRecord(measured_data=data, metadata=meta)

        path = save_record(record=record, directory=tmp_path, name="dtype64")
        restored_record = load_record(path=path)

        assert restored_record.measured_data.dtype == np.float64
        npt.assert_allclose(restored_record.measured_data, data)

    def test_large_array(self, tmp_path: Path) -> None:
        """Verifies that a 10,000 point array saves and loads correctly."""

        np.random.seed(42)
        start = datetime(2026, 1, 1, 1, 0, 0)
        end = start + timedelta(hours=1)
        data = np.random.normal(0, 1, 10_000)
        sweep_data = np.linspace(0, 100, 10_000)
        meta = MeasurementMetaData(
            experiment_type="1D",
            measured_parameter="current",
            sweep_parameters=[SweepParameter(name="frequency", unit="Hz")],
            start_time=start,
            end_time=end,
            pc_name="lab-pc-01",
            user_metadata={"operator": "Dr. John", "temperature": 22.6},
        )
        record = MeasurementRecord(
            measured_data=data, metadata=meta, sweep_data=[sweep_data]
        )

        path = save_record(record=record, directory=tmp_path, name="test_large")
        restored_record = load_record(path=path)

        npt.assert_allclose(restored_record.measured_data, data)
        npt.assert_allclose(restored_record.sweep_data[0], sweep_data)

    def test_multidimensional_array_roundtrip(self, tmp_path: Path) -> None:
        """A 2-Dimensional matrix should flatten naturally on save and restore on load."""

        np.random.seed(42)
        start = datetime(2026, 1, 1, 1, 0, 0)
        end = start + timedelta(hours=20)
        data = np.random.normal(0, 1, size=(5, 5))
        sweep_data = np.linspace(0, 100, 25).reshape(5, 5)
        meta = MeasurementMetaData(
            experiment_type="2D-grid",
            measured_parameter="intensity",
            sweep_parameters=[SweepParameter(name="grid_position")],
            start_time=start,
            end_time=end,
            pc_name="lab-pc-01",
        )
        record = MeasurementRecord(
            measured_data=data, metadata=meta, sweep_data=[sweep_data]
        )

        path = save_record(record=record, directory=tmp_path, name="test_2d")
        restored_record = load_record(path=path)

        assert restored_record.measured_data.shape == (5, 5)
        assert restored_record.sweep_data[0].shape == (5, 5)
        npt.assert_allclose(restored_record.measured_data, data)
        npt.assert_allclose(restored_record.sweep_data[0], sweep_data)

    def test_nan_and_infinity_values_roundtrip_safely(
        self, tmp_path: Path, sample_1d_record: MeasurementRecord
    ) -> None:
        """NaN and Inf values logically persist transparently."""
        # make a copy of the data to avoid side effects
        sample_1d_record.measured_data = sample_1d_record.measured_data.copy()
        sample_1d_record.measured_data[0] = np.nan
        sample_1d_record.measured_data[1] = np.inf
        sample_1d_record.measured_data[2] = -np.inf

        path = save_record(sample_1d_record, directory=tmp_path, name="test_bounds")
        restored_record = load_record(path=path)

        assert np.isnan(restored_record.measured_data[0])
        assert np.isposinf(restored_record.measured_data[1])
        assert np.isneginf(restored_record.measured_data[2])


# ---------------------------------------------------------------------------------------------
# Tests Error Handling
# ---------------------------------------------------------------------------------------------


class TestErrorHandling:
    """Verify missing files produce clear errors."""

    def test_load_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_record(tmp_path / "nonexistent")

    def test_load_corrupt_json_raises_value_error(
        self, tmp_path: Path, sample_1d_record: MeasurementRecord
    ) -> None:
        path = save_record(sample_1d_record, directory=tmp_path)
        metadata_path = path.parent / f"{path.name}_metadata.json"

        # corrupt JSON explicitly mapping bad formatting
        with open(metadata_path, "w") as f:
            f.write("Invalid json formatting \n")

        with pytest.raises(ValueError, match="Corrupted or invalid JSON"):
            load_record(path=path)

    def test_load_with_wrong_suffix_fails_cleanly(
        self, tmp_path: Path, sample_1d_record: MeasurementRecord
    ) -> None:
        """If user passes data.csv into load_record() instead of stem, it fails."""
        path = save_record(sample_1d_record, directory=tmp_path, name="bad_suffix")
        csv_path = path.parent / f"{path.name}_data.csv"

        with pytest.raises(FileNotFoundError):
            load_record(csv_path)
