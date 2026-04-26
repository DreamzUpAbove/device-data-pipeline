"""Adapters for translating external data formats into domain models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from .models import MeasurementMetaData, MeasurementRecord, SweepParameter


def build_record(raw: dict[str, Any]) -> MeasurementRecord:
    """Adapt raw dictionary output from external source into MeasurementRecord.

    Args:
        raw: Dictionary containing raw experimental data.

    Returns:
        A validated MeasurementRecord.

    Raises:
        KeyError: If essential keys are missing.
    """
    required_keys = ["mode", "measured", "started_at", "ended_at"]
    for key in required_keys:
        if key not in raw:
            raise KeyError(
                f"'{key}' key missing from raw data dict. "
                "Expected output from generate_measurement_data()."
            )
    mode = raw["mode"]
    swept_params_raw = raw.get("swept_params", [])
    measured_param = raw.get("measured_params", "unknown")

    sweep_labels: list[SweepParameter] = []
    sweep_arrays: list[np.ndarray] = []

    if mode != "time-trace" and swept_params_raw:
        if isinstance(swept_params_raw, str):
            sweep_labels = [SweepParameter(name=swept_params_raw)]
        elif isinstance(swept_params_raw, list):
            sweep_labels = [SweepParameter(name=s) for s in swept_params_raw]
        else:
            sweep_labels = []

        if "swept" in raw:
            sweep_arrays.append(raw["swept"])
        elif "swept1" in raw:
            sweep_arrays.append(raw["swept1"])
            if "swept2" in raw:
                sweep_arrays.append(raw["swept2"])

    metadata = MeasurementMetaData(
        experiment_type=mode,
        measured_parameter=measured_param,
        sweep_parameters=sweep_labels,
        start_time=datetime.fromisoformat(raw["started_at"]),
        end_time=datetime.fromisoformat(raw["ended_at"]),
        pc_name="integration-test-pc",
        user_metadata={"source": "generate_measurement_data()"},
    )

    return MeasurementRecord(
        metadata=metadata,
        measured_data=raw["measured"],
        sweep_data=sweep_arrays,
    )
