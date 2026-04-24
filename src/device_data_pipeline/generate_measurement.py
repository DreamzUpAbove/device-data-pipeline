import numpy as np
import random
from datetime import datetime, timedelta


def generate_measurement_data():
    mode = random.choice(["1D", "line-scan", "time-trace"])
    length = random.choice([100, 1000, 10000])
    delay = random.choice([0.01, 0.1, 0.2])
    start = datetime.now()
    end = start + timedelta(seconds=length * delay)
    d = {"mode": mode, "started_at": start.isoformat(), "ended_at": end.isoformat()}
    d2 = {}
    match mode:
        case "1D":
            s = random.choice(["voltage", "current", "frequency"])
            m = random.choice(["voltage", "current"])
            start = 0
            if s == "frequency":
                start = 0.01
            x = np.linspace(start, 1, length)
            y = np.sin(x) + np.random.normal(0, 0.1, size=length)
            d2 = {"swept_params": s, "measured_params": m, "swept": x, "measured": y}
        case "line-scan":
            s = [
                random.choice(["voltage", "current", "frequency"]),
                random.choice(["voltage", "current", "frequency"]),
            ]
            m = random.choice(["voltage", "current"])
            start = 0
            if "frequency" in s:
                start = 0.01
            x1 = np.linspace(start, 1, length)
            x2 = np.linspace(start, 2, length)
            data = np.sin(x1) + 2 * np.sin(x2) + np.random.normal(0, 0.1, size=length)
            d2 = {
                "swept_params": s,
                "measured_params": m,
                "swept1": x1,
                "swept2": x2,
                "measured": data,
            }
        case "time-trace":
            s = "time"
            m = random.choice(["voltage", "current"])
            x = np.linspace(0, 100, length)
            y = np.sin(x) + np.random.normal(0, 0.1, size=length)
            d2 = {"swept_params": s, "measured_params": m, "swept": x, "measured": y}
    return d | d2


def main():
    print("Generating measurement data...\n")
    result = generate_measurement_data()

    print(f"Mode:        {result['mode']}")
    print(f"Started at:  {result['started_at']}")
    print(f"Ended at:    {result['ended_at']}")
    print(f"Swept param: {result['swept_params']}")
    print(f"Measured:    {result['measured_params']}")

    if result["mode"] == "line-scan":
        print(f"Swept1 shape: {result['swept1'].shape}")
        print(f"Swept2 shape: {result['swept2'].shape}")
    else:
        print(f"Swept shape:  {result['swept'].shape}")

    print(f"Data shape:  {result['measured'].shape}")
    print(
        f"Data range:  [{result['measured'].min():.4f}, {result['measured'].max():.4f}]"
    )


if __name__ == "__main__":
    main()
