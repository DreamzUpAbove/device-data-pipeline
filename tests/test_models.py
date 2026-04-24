"""Tests for device_data_pipeline.models validation and serialisation"""

from datetime import datetime, timedelta

from device_data_pipeline import SweepParameter
import pytest
import numpy as np
from device_data_pipeline.models import MeasurementMetaData, MeasurementRecord

# ------------------------------------------------------------------------
# SweepParameter Tests
# ------------------------------------------------------------------------

class TestSweepParameter:
    """Tests for SweepParameter dataclass."""

    def test_creation_with_name_and_unit(self) -> None:
        sp = SweepParameter(name="voltage", unit="V")
        assert sp.name == "voltage"
        assert sp.unit == "V"

    def test_unit_defaults_to_empty_string(self) -> None:
        sp = SweepParameter(name="frequency")
        assert sp.unit == ""

    def test_empty_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            SweepParameter(name="")

    def test_whitespace_only_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            SweepParameter(name=" ")

    def test_frozen_prevents_mutation(self) -> None:
        sp = SweepParameter(name="voltage", unit="V")
        with pytest.raises(AttributeError):
            sp.name = "current"  # type: ignore


# -------------------------------------------------------------------------------------
# MeasurementMetadata Tests
# -------------------------------------------------------------------------------------

class TestMeasurementMetadata:
    """Tests for MeasurementMetadata dataclass."""

    def test_create_smoke(self) -> None:
        start = datetime(2026, 1, 1, 1, 0, 0)
        end = start + timedelta(hours=1)
        meta = MeasurementMetaData(
            experiment_type="1D",
            measured_parameter="current",
            sweep_parameters=[SweepParameter(name="voltage")],
            start_time=start,
            end_time=end,
            pc_name="lab-pc-101",
            user_metadata={"device": "chip_v3"},
        )
        assert meta.experiment_type == "1D"
        assert meta.measured_parameter == "current"
        assert len(meta.sweep_parameters) == 1
        assert meta.pc_name == "lab-pc-101"
        assert meta.user_metadata == {"device": "chip_v3"}

    def test_end_before_start_raises_error(self) -> None:
        start = datetime(2026, 1, 1, 1, 0, 0)
        end = start - timedelta(seconds=1)

        with pytest.raises(ValueError, match="end_time"):
            MeasurementMetaData(
                experiment_type="1D",
                measured_parameter="current",
                sweep_parameters=[],
                start_time=start,
                end_time=end,
            )

    def test_pc_name_auto_populated(self) -> None:
        """When pc name not provided, it should get from socket.gethostname()."""
        import socket

        start = datetime(2026, 1, 1, 1, 0, 0)
        end = start + timedelta(seconds=1)
        meta = MeasurementMetaData(
            experiment_type="1D",
            measured_parameter="current",
            sweep_parameters=[],
            start_time=start,
            end_time=end,
        )
        assert meta.pc_name == socket.gethostname()

    def test_user_metadata_defaults_to_empty_dict(self) -> None:
        start = datetime(2026, 1, 1, 1, 0, 0)
        end = start + timedelta(seconds=1)
        meta = MeasurementMetaData(
            experiment_type="1D",
            measured_parameter="current",
            sweep_parameters=[],
            start_time=start,
            end_time=end,
        )
        assert meta.user_metadata == {}

    def test_dict_produces_serialisable_output(self) -> None:
        import json

        start = datetime(2026, 1, 1, 1, 0, 0)
        end = start + timedelta(seconds=1)
        meta = MeasurementMetaData(
            experiment_type="line-scan",
            measured_parameter="current",
            sweep_parameters=[
                SweepParameter(name="pin1", unit="V"),
                SweepParameter(name="pin2", unit="V"),
            ],
            start_time=start,
            end_time=end,
            pc_name="test-pc",
            user_metadata={"lab": "B2", "device": "chip_v4"},
        )
        d = meta.to_dict()
        json_str = json.dumps(d)

        # check serialisable without errors
        assert isinstance(json_str, str)

        # verify individual fields
        assert d["experiment_type"] == "line-scan"
        assert d["start_time"] == "2026-01-01T01:00:00"
        assert len(d["sweep_parameters"]) == 2
        assert d["sweep_parameters"][0]["name"] == "pin1"

    def test_from_dict_roundtrip(self) -> None:
        """to_dict -> from_dict should produce equivalent object."""
        start = datetime(2026, 1, 1, 1, 0, 0)
        end = start + timedelta(hours=1)
        original_record = MeasurementMetaData(
            experiment_type="1D",
            measured_parameter="current",
            sweep_parameters=[SweepParameter(name="frequency", unit="Hz")],
            start_time=start,
            end_time=end,
            pc_name="lab-pc-101",
            user_metadata={"operator": "Dr. John", "temperature": 22.6},
        )
        restored_record = MeasurementMetaData.from_dict(original_record.to_dict())

        assert restored_record == original_record

    def test_sweep_parameter_unit_roundtrip(self) -> None:
        """Explicitly verifies the empty vs custom units survives JSON roundtrip"""

        start = datetime(2026, 1, 1, 1, 0, 0)
        end = start + timedelta(seconds=1)
        original_record = MeasurementMetaData(
            experiment_type="line-scan",
            measured_parameter="current",
            sweep_parameters=[
                SweepParameter(name="pin3", unit="mV"),
                SweepParameter(name="pin4"),
            ],
            start_time=start,
            end_time=end,
        )
        d = original_record.to_dict()
        assert original_record.sweep_parameters[0].unit == "mV"
        assert original_record.sweep_parameters[1].unit == ""

        restored_record = MeasurementMetaData.from_dict(d)
        assert restored_record.sweep_parameters[0].unit == "mV"
        assert restored_record.sweep_parameters[1].unit == ""

    def test_raises_value_error_on_missing_keys(self) -> None:
        """Verify the structural wrap exceptions hides dictionary mechanics securely"""
        bad_data = {
            "experiment_type": "1D",
            # internal 'measured_parameter' missing
            "start_time": "2026-01-01T01:00:00",
            "end_time": "2026-01-01T01:00:02",
        }
        with pytest.raises(ValueError, match="Invalid metadata dictionary format"):
            MeasurementMetaData.from_dict(bad_data)


# -------------------------------------------------------------------------------------------------------------
# MeasurementRecords Tests
# -------------------------------------------------------------------------------------------------------------

class TestMeasurementRecord:
    """Tests for MeasurementRecord dataclass."""

    def test_creation_with_valid_data(
        self, sample_1d_record: MeasurementRecord
    ) -> None:
        assert isinstance(sample_1d_record.measured_data, np.ndarray)
        assert len(sample_1d_record.sweep_data) == 1
        assert sample_1d_record.metadata.experiment_type == "1D"

    def test_non_array_raises_type_error(self) -> None:
        start = datetime(2026, 1, 1, 1, 0, 0)
        end = start + timedelta(seconds=1)

        meta = MeasurementMetaData(
            experiment_type="1D",
            measured_parameter="current",
            sweep_parameters=[],
            start_time=start,
            end_time=end,
        )
        with pytest.raises(TypeError, match="numpy array"):
            MeasurementRecord(
                metadata=meta,
                measured_data=[1, 2, 3],  # type: ignore
            )

    def test_mismatched_sweep_count_raises_value_error(self) -> None:
        start = datetime(2026, 1, 1, 1, 0, 0)
        end = start + timedelta(hours=1)
        meta = MeasurementMetaData(
            experiment_type="1D",
            measured_parameter="current",
            sweep_parameters=[SweepParameter(name="voltage")],
            start_time=start,
            end_time=end,
            pc_name="lab-pc-101",
            user_metadata={"device": "chip_v3"},
        )

        with pytest.raises(ValueError, match="sweep arrays"):
            MeasurementRecord(
                metadata=meta, measured_data=np.array([1.0, 2.6, 3.1]), sweep_data=[]
            )

    def test_zero_sweep_parameters_allowed(
        self, sample_time_trace_record: MeasurementRecord
    ) -> None:
        """time-trace experiments have sweep parameters zero - this must work"""

        assert len(sample_time_trace_record.metadata.sweep_parameters) == 0
        assert len(sample_time_trace_record.sweep_data) == 0

    def test_multiple_sweep_params_allowed(
        self, simple_linescan_record: MeasurementRecord
    ) -> None:
        """Line-scan experiments has 2+ sweep parameters."""
        assert len(simple_linescan_record.sweep_data) == 2
        assert len(simple_linescan_record.metadata.sweep_parameters) == 2

    def test_mismatched_array_sizes_raises_value_error(self) -> None:
        start = datetime(2026, 1, 1, 1, 0, 0)
        end = start + timedelta(hours=1)
        meta = MeasurementMetaData(
            experiment_type="1D",
            measured_parameter="current",
            sweep_parameters=[SweepParameter(name="voltage")],
            start_time=start,
            end_time=end,
        )

        with pytest.raises(
            ValueError, match="Size mismatch: measured_data has 3 elements"
        ):
            MeasurementRecord(
                metadata=meta,
                measured_data=np.array([1.0, 2.0, 3.0]),
                sweep_data=[np.array([0.0, 1.0])],
            )

    def test_identical_object_reference_raises_value_error(self) -> None:
        start = datetime(2026, 1, 1, 1, 0, 0)
        end = start + timedelta(minutes=1)
        meta = MeasurementMetaData(
            experiment_type="1D",
            measured_parameter="current",
            sweep_parameters=[SweepParameter(name="voltage")],
            start_time=start,
            end_time=end,
        )
        sample_array = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="exact same object reference"):
            MeasurementRecord(
                metadata=meta, measured_data=sample_array, sweep_data=[sample_array]
            )
