from pathlib import Path
import xarray as xr
import numpy as np

ROOT = Path(r"D:\Research\Sen12Landslides\data\data_harmonized")

S1_PATH = ROOT / "s1asc" / "chimanimani_s1asc_1004.nc"
S2_PATH = ROOT / "s2" / "chimanimani_s2_1004.nc"

with xr.open_dataset(S1_PATH) as ds1, xr.open_dataset(S2_PATH) as ds2:

    s1_dates = ds1.time.values.astype("datetime64[D]")
    s2_dates = ds2.time.values.astype("datetime64[D]")

    print("=" * 75)
    print("SENTINEL-1 ASCENDING ACQUISITION DATES")
    print("=" * 75)

    for i, date in enumerate(s1_dates):
        print(f"S1[{i:02d}] : {date}")

    print("\n" + "=" * 75)
    print("SENTINEL-2 ACQUISITION DATES")
    print("=" * 75)

    for i, date in enumerate(s2_dates):
        print(f"S2[{i:02d}] : {date}")

    print("\n" + "=" * 75)
    print("NEAREST-DATE CORRESPONDENCE")
    print("=" * 75)

    for i, s1_date in enumerate(s1_dates):
        differences = np.abs(s2_dates - s1_date)
        j = int(np.argmin(differences))
        delta_days = int(differences[j] / np.timedelta64(1, "D"))

        print(
            f"S1[{i:02d}] {s1_date}  <->  "
            f"S2[{j:02d}] {s2_dates[j]}  "
            f"| Difference: {delta_days} days"
        )

    print("\n" + "=" * 75)
    print("TEMPORAL SUMMARY")
    print("=" * 75)

    print("S1 observations:", len(s1_dates))
    print("S2 observations:", len(s2_dates))

    print(
        "Exact common dates:",
        sorted(set(s1_dates.tolist()) & set(s2_dates.tolist()))
    )

    print(
        "S1 date range:",
        s1_dates.min(),
        "to",
        s1_dates.max()
    )

    print(
        "S2 date range:",
        s2_dates.min(),
        "to",
        s2_dates.max()
    )

    print("\nEvent date:", ds1.attrs.get("event_date"))