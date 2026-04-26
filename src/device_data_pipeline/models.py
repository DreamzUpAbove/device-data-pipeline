"""
Data models for scientific measurement recording and management.

Provides dataclasses that represent complete structure
of an experiment measurement.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import socket
from typing import Any
import numpy as np


@dataclass(frozen=True, slots=True)
class SweepParameter:
    """A label for a swept axis (name + optional unit)

    Attributes:
        name: Human-readable parameter name.
        unit: Optional physical unit.

    Raises:
        ValueError: if ``name`` is empty or contains only whitespace.

    Example::

        >>> voltage = SweepParameter(name="voltage", unit="V")
        >>> frequency = SweepParameter(name="frequency") # unit default to ""
    """

    name: str
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Sweep Parameter cannot be empty")


@dataclass(slots=True)
class MeasurementMetaData:
    """Contextual information about an experimental measurement.

    Captures everything needed to understand, reproduce, and catalogue
    a measurement - except the raw numerical data itself.

    Attributes:
        experiment_type: Label describing the experiment kind.
        measured_parameter: Name of the quantity that was measured.
        sweep_parameters: Ordered list of ``SweepParameter`` labels.
            Empty for time-traces, one entry for 1D sweeps, two or more
            for line-scans or higher-dimensional experiments.
        start_time: When measurement began.
        end_time: When measurement ended.
        pc_name: Hostname of the machine that performed the measurement.
            Auto-populated from ``socket.gethostname()`` if not provided.
        user_metadata: Arbitrary user supplied key-value pairs.
            Must be JSON-serialisable.

    Raises:
        ValueError: If ``end_time`` is before ``start_time``
    """

    experiment_type: str
    measured_parameter: str
    sweep_parameters: list[SweepParameter]
    start_time: datetime
    end_time: datetime
    pc_name: str = field(default_factory=socket.gethostname)
    user_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end_time < self.start_time:
            raise ValueError(
                f"end_time ({self.end_time.isoformat()}) must be >= "
                f"start_time ({self.start_time.isoformat()})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise metadata into a Json friendly dictionary.

        Datetimes are converted into ISO 8601 strings.
        SweptParameters are converted to ``{"name":..., "unit":...}`` dicts.

        Returns:
            A plain dictionary suitable for ``json.dump()``.
        """
        return {
            "experiment_type": self.experiment_type,
            "measured_parameter": self.measured_parameter,
            "sweep_parameters": [
                {"name": sp.name, "unit": sp.unit} for sp in self.sweep_parameters
            ],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "pc_name": self.pc_name,
            "user_metadata": self.user_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeasurementMetaData:
        """Reconstruct MeasurementMetaData from a dictionary.

        This is inverse of the :meth: `to_dict` and is used during loading to restore metadata from a JSON file.

        Args:
            data: dictionary as produced by :meth:`to_dict`.

        Returns:
            A fully reconstructed ``MeasurementMetadata`` instance.
        """
        try:
            return cls(
                experiment_type=data["experiment_type"],
                measured_parameter=data["measured_parameter"],
                sweep_parameters=[
                    SweepParameter(name=sp["name"], unit=sp.get("unit", ""))
                    for sp in data["sweep_parameters"]
                ],
                start_time=datetime.fromisoformat(data["start_time"]),
                end_time=datetime.fromisoformat(data["end_time"]),
                pc_name=data["pc_name"],
                user_metadata=data["user_metadata"],
            )
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid metadata dictionary format {e}") from e


@dataclass(slots=True)
class MeasurementRecord:
    """A complete experimental Measurement: data arrays plus metadata.

    Attributes:
        metadata: contextual information about the measurement.
        measured_data: the dependent-variable as a numpy array.
        sweep_data: Ordered list of independent-variables as numpy arrays.
            Must have same length as sweep parameters in metadata.

    Raises:
        TypeError: If ``measured_data`` not a numpy array.
        ValueError: If the number of sweep arrays doesn't match
        the parameters number of sweep parameters in metadata.

    Example::

        record = MeasurementRecord(
            metadata=metadata,
            measured_data=np.array([1.0, 2.0, 3.0]),
            sweep_data=[np.linspace(0, 1, 3)]
        )
    """

    metadata: MeasurementMetaData
    measured_data: np.ndarray
    sweep_data: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.measured_data, np.ndarray):
            raise TypeError(
                f"Measured data must be a numpy array, "
                f"got {type(self.measured_data).__name__}"
            )
        if len(self.sweep_data) != len(self.metadata.sweep_parameters):
            raise ValueError(
                f"Number of sweep arrays ({len(self.sweep_data)}) must match "
                f"number of sweep parameters "
                f"({len(self.metadata.sweep_parameters)})"
            )
        for i, sweep in enumerate(self.sweep_data):
            if not isinstance(sweep, np.ndarray):
                raise TypeError(
                    f"sweep_data[{i}] must be a numpy array, got {type(sweep).__name__}"
                )

            if self.measured_data is sweep:
                raise ValueError(
                    f"measured_data and sweep_data[{i}] are the exact same object reference. "
                    f"Please duplicate the array if the measurement values perfectly mirror the swept values explicitly."
                )
            if self.measured_data.size != sweep.size:
                raise ValueError(
                    f"Size mismatch: measured_data has {self.measured_data.size} elements, "
                    f"but sweep_data[{i}] has {sweep.size} elements. "
                    f"All elements must contain the same total number of the elements to stack."
                )
