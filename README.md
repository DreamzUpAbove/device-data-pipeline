# Device Data Pipeline

A lightweight, robust Python module for accepting, validating, and storing diverse scientific measurement data and metadata. 

## Overview 
Researchers conduct semi-automated experiments (eg., 1D sweeps, time-traces, N-dimensional line scans) for different hardware parameters like voltage or current. This module provides a reliable pipeline to handle array inputs, associate them with contextual metadata, serialise them to disk for future analysis or publication.  



## Project Structure

```
device_data_pipeline/
├── __init__.py              # public API
├── models.py                # core data models
├── storage.py               # save/load logic
├── adapters.py              # translates external formats into models
└── generate_measurement.py  # helper: simulates experiment data

tests/
├── conftest.py              # shared pytest fixtures
├── test_models.py           # unit tests for models
├── test_storage.py          # save/load roundtrip and edge cases
└── test_integration.py      # end-to-end pipeline tests

examples/
└── demo.py

pyproject.toml
README.md
.gitignore
```

## Installation

```bash
pip install -e .

# with dev dependencies (testing)
pip install -e ".[dev]"
```

## Experiment Types

| Type | Sweep axes | Example |
|---|---|---|
| `1D` | 1 | Voltage swept 0–5 V, current measured |
| `time-trace` | 0 | Current sampled every second for 5 minutes |
| `line-scan` | 2 | Two pins swept together, current measured |

## Usage

```python
from datetime import datetime
import numpy as np
from device_data_pipeline import (
    MeasurementMetaData,
    MeasurementRecord,
    SweepParameter,
    save_record,
    load_record,
)

voltage = np.linspace(0, 5, 100)
current = voltage / 1000

meta = MeasurementMetaData(
    experiment_type="1D",
    measured_parameter="current",
    sweep_parameters=[SweepParameter(name="voltage", unit="V")],
    start_time=datetime(2026, 4, 25, 9, 0, 0),
    end_time=datetime(2026, 4, 25, 9, 5, 0),
    user_metadata={"device": "resistor_1k", "lab": "B2"},
)

record = MeasurementRecord(
    metadata=meta,
    measured_data=current,
    sweep_data=[voltage],
)

path = save_record(record, directory="data/")
loaded = load_record(path)
```

### Using the adapter with `generate_measurement_data()`

```python
from device_data_pipeline.adapters import build_record
from device_data_pipeline.generate_measurement import generate_measurement_data
from device_data_pipeline import save_record

raw = generate_measurement_data()
record = build_record(raw)
path = save_record(record, directory="data/")
```
## Running Tests

```bash
# run all tests
pytest tests/

# run all tests with coverage
pytest --cov=device_data_pipeline tests/

# run a specific test file
pytest tests/test_storage.py -v
```

## Running the Demo

```bash
# simple demo
python examples/demo.py

# specify the output directory
python examples/demo.py --output_dir measurements/

```

## Design Notes & Considerations
**Time-trace and the time axis:**  
The task description states time-trace produces "no related swept parameters recorded."
However, the provided helper includes a time linspace as `swept_params="time"`.
This implementation follows the written specification (zero sweep axes) and discards
the time array in `build_record()`. To preserve the time axis, construct the MeasurementRecord directly 
instead of using build_record(), passing the time array explicitly 
in both sweep_parameters and sweep_data.

**Why CSV/JSON over other formats**  
Simplicity and accessibility. CSV and JSON can be opened in any text editor, 
or spreadsheets with no special software. For most simple lab experiments
producing thousands to low millions of size data points this would be sufficient.  

**When to consider alternatives**  
- Large continuous datasets(Multi-GB time traces) -> HDF5 via `h5py`
- High-frequency roundtrips -> Numpy `.npz`
- Structured relational data -> SQLite or Postgres

**Memory Warning**  
`load_record` loads the entire CSV into memory at once via `numpy.loadtxt()`. This will raise `MemoryError` on very large files. This was a design choice to avoid complexity vs scope trade off for simple research module. 

**Precision**  
Data is stored as 18-decimal places scientific notation (`%.18e`) preserving full float64 precision across save/load roundtrips.