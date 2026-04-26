"""Demo: Full pipeline using generate_measurement_data() -> save -> load.

Usage::

    python examples/demo.py
    python examples/demo.py --output_dir /path/to/output/directory
"""

import argparse
from pathlib import Path

import numpy as np

from device_data_pipeline import save_record, load_record
from device_data_pipeline.adapters import build_record
from device_data_pipeline.generate_measurement import generate_measurement_data


def main() -> None:
    """Run the full measurement recording pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data"),
        help="Directory to save the measurement data.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    np.random.seed(42)

    # Run 3 random experiments to show timestamped files
    for i in range(3):
        raw = generate_measurement_data()
        record = build_record(raw)

        # save() auto-generates a timestamped filename and returns the path
        path = save_record(record, directory=output_dir)

        print(f"\n--- Experiment {i + 1}: {raw['mode']} ---")
        print(f"  Saved to:  {path.name}_data.csv")
        print(f"             {path.name}_metadata.json")

        # Load it back to verify
        loaded = load_record(path)
        data_ok = np.allclose(loaded.measured_data, record.measured_data)

        print(f"  Type:       {loaded.metadata.experiment_type}")
        print(f"  Measured:   {loaded.metadata.measured_parameter}")
        print(f"  Sweeps:     {[sp.name for sp in loaded.metadata.sweep_parameters]}")
        print(f"  Data shape: {loaded.measured_data.shape}")
        print(f"  Roundtrip:  {'PASS' if data_ok else 'FAIL'}")

    # Show all saved files
    print(f"\n{'=' * 60}")
    print(f"All files in {output_dir.name}/:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
