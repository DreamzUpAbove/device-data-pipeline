"""Shared tests fixures for device_data_pipeline tests."""

from datetime import datetime, timedelta
import numpy as np
import pytest

from device_data_pipeline import (
    MeasurementMetaData,
    MeasurementRecord,
    SweepParameter,
)


@pytest.fixture
def fixed_time() -> tuple[datetime, datetime]:
    """Deterministic start and end time for reproducible tests."""
    start = datetime(2026, 1, 1, 1, 0, 0)
    end = start + timedelta(minutes=5)
    return start, end


@pytest.fixture
def sample_1d_record(fixed_time: tuple[datetime, datetime]) -> MeasurementRecord:
    start, end = fixed_time
    np.random.seed(42)

    voltages = np.linspace(0, 5, 100)
    currents = np.sin(voltages) + np.random.normal(0, 0.1, 100)

    metadata = MeasurementMetaData(
        experiment_type="1D",
        measured_parameter="current",
        sweep_parameters=[SweepParameter(name="frequency", unit="Hz")],
        start_time=start,
        end_time=end,
        pc_name="lab-pc-101",
    )

    return MeasurementRecord(
        metadata=metadata, measured_data=currents, sweep_data=[voltages]
    )


@pytest.fixture
def sample_time_trace_record(
    fixed_time: tuple[datetime, datetime],
) -> MeasurementRecord:
    start, end = fixed_time
    np.random.seed(42)

    currents = np.random.normal(0, 0.1, 500)

    metadata = MeasurementMetaData(
        experiment_type="time-trace",
        measured_parameter="current",
        sweep_parameters=[],
        start_time=start,
        end_time=end,
        pc_name="lab-pc-01",
    )
    return MeasurementRecord(metadata=metadata, measured_data=currents, sweep_data=[])


@pytest.fixture
def simple_linescan_record(fixed_time: tuple[datetime, datetime]) -> MeasurementRecord:

    start, end = fixed_time
    np.random.seed(42)

    voltage_01 = np.linspace(0, 1, 50)
    voltage_02 = np.linspace(2, 3, 50)
    measured_values = (
        np.sin(voltage_01) + 2 * np.sin(voltage_02) + np.random.normal(0, 0.1, 50)
    )
    metadata = MeasurementMetaData(
        experiment_type="line-scan",
        measured_parameter="current",
        sweep_parameters=[
            SweepParameter(name="pin1", unit="V"),
            SweepParameter(name="pin2", unit="V"),
        ],
        start_time=start,
        end_time=end,
        pc_name="test-pc-02",
    )

    return MeasurementRecord(
        metadata=metadata,
        measured_data=measured_values,
        sweep_data=[voltage_01, voltage_02],
    )
