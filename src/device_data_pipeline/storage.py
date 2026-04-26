"""Persistence layer for saving and loading measurement records.

File Format
------------
Each measurement is stored as two files sharing a common stem:

- ``{stem}_data.csv`` - Comma-separated values with a header row.
   Columns: ``measured``, ``sweep_0``, ``sweep_1``, etc.

- ``{stem}_metadata.json`` - Human-readable JSON file containing all
    metadata (experiment type, time stamps, sweep labels, user extras).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np


from .models import MeasurementMetaData, MeasurementRecord

# 18 decimal places in scientific notations preserves full float64 precision.
_FLOAT_FMT = "%.18e"


def _make_stem(experiment_type: str) -> str:
    """Create a timestamped, filesystem-safe stem string"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if experiment_type:
        safe_type = experiment_type.replace(" ", "_").replace("/", "_")
        return f"{timestamp}_{safe_type}"
    return timestamp


def save_record(
    record: MeasurementRecord,
    directory: str | Path = ".",
    name: str | None = None,
) -> Path:
    """Save a ``MeasurementRecord`` to disk as CSV (data) and JSON (metadata)


    Creates two files:
        - ``{directory}/{name}_data.csv``
        - ``{directory}/{name}_metadata.json``

    Args:
        record: The measurement record to save.
        directory: Directory to save into, created automatically
        if it does not exist
        name: Custom filename to be used. If ``None`` (default),
        a unique timestamp name is generated automatically.
        Note: If a custom name is provided, and the file exists, will be overwritten.

    Returns:
        The full base path (stem), that was used so caller knows
        where file was saved. Pass this to :func:`load_record` later.

    Raises:
        OSError: If files cannot be written into the disk.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    orig_name = name
    if name is None:
        name = _make_stem(record.metadata.experiment_type)

    base_path = directory / name
    csv_path = directory / f"{name}_data.csv"

    if orig_name is None:
        counter = 1
        while csv_path.exists():
            name = f"{base_path.name}_{counter}"
            csv_path = directory / f"{name}_data.csv"
            counter += 1
        base_path = directory / name

    metadata_path = directory / f"{name}_metadata.json"

    # ---- write data into CSV ----
    # flatten all arrays to 1D for tabular CSV storage
    columns: list[np.ndarray] = [record.measured_data.ravel()]
    header_parts: list[str] = ["measured"]

    for i, sweep_arr in enumerate(record.sweep_data):
        columns.append(sweep_arr.ravel())
        header_parts.append(f"sweep_{i}")

    header = ",".join(header_parts)

    if len(columns) == 1:
        data_matrix = columns[0].reshape(-1, 1)
    else:
        data_matrix = np.column_stack(columns)

    np.savetxt(
        csv_path, data_matrix, delimiter=",", header=header, fmt=_FLOAT_FMT, comments=""
    )

    # ---- write metadata as JSON file -----
    meta_dict = record.metadata.to_dict()

    # store the structural shape payload at top level of JSON metadata
    meta_dict["_array_shapes_internal"] = {
        "measured": record.measured_data.shape,
        "sweeps": [arr.shape for arr in record.sweep_data],
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)

    return base_path


def load_record(path: str | Path) -> MeasurementRecord:
    """Load a MeasurementRecord from the disk.

    Expects two files created by :func:`save_record`
        - ``{path}_data.csv``
        - ``{path}_metadata.json``

    Warning:
        Loads entire csv into RAM via `numpy.loadtxt`.
        For substantial scientific datasets, this will trigger `MemoryError`.

    Args:
        path: Base file path as returned by :func:`save_record`.

    Returns:
        A fully reconstructed ``MeasurementRecord``.

    Raises:
        FileNotFoundError: If either CSV or JSON file is missing.
        RuntimeError: If data arrays cannot construct successfully parallel to shapes.
        ValueError: If JSON metadata is structurally invalid.
    """
    path = Path(path)
    csv_path = path.parent / f"{path.name}_data.csv"
    metadata_path = path.parent / f"{path.name}_metadata.json"

    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found at {csv_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    # Load metadata
    try:
        with open(metadata_path, encoding="utf-8") as f:
            meta_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Corrupted or invalid JSON metadata at {metadata_path}: {e}"
        ) from e

    shapes = meta_dict.pop("_array_shapes_internal", None)
    metadata = MeasurementMetaData.from_dict(meta_dict)

    # Load data
    try:
        data_matrix = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    except OSError as e:
        raise RuntimeError(
            f"Failed to parse CSV data matrix accurately from {csv_path}: {e}"
        ) from e

    measured_data = data_matrix[:, 0]
    sweep_data = [data_matrix[:, i] for i in range(1, data_matrix.shape[1])]

    # reshape arrays back to their original multidimensional shapes if possible
    if shapes:
        measured_data = measured_data.reshape(shapes["measured"])
        for i, shape in enumerate(shapes["sweeps"]):
            sweep_data[i] = sweep_data[i].reshape(shape)

    return MeasurementRecord(
        measured_data=measured_data, metadata=metadata, sweep_data=sweep_data
    )
