from pathlib import Path
import xarray as xr
import numpy as np

ROOT = Path(r"D:\Research\Sen12Landslides\data\data_harmonized")
PATH = ROOT / "s2" / "chimanimani_s2_1004.nc"

# Sentinel-2 Scene Classification Layer classes
SCL_CLASSES = {
    0: "No data",
    1: "Saturated / defective",
    2: "Dark area pixels",
    3: "Cloud shadows",
    4: "Vegetation",
    5: "Bare soils",
    6: "Water",
    7: "Unclassified",
    8: "Cloud medium probability",
    9: "Cloud high probability",
    10: "Thin cirrus",
    11: "Snow / ice",
}

with xr.open_dataset(PATH) as ds:

    print("Dataset dimensions:", dict(ds.sizes))
    print("Variables:", list(ds.data_vars))

    scl = ds["SCL"].values
    dates = ds.time.values.astype("datetime64[D]")

    print("\nSCL shape:", scl.shape)
    print("SCL dtype:", scl.dtype)

    for i, date in enumerate(dates):

        frame = scl[i]
        values, counts = np.unique(frame, return_counts=True)

        print(f"\nObservation {i:02d} | {date}")
        print("-" * 55)

        for value, count in zip(values, counts):
            label = SCL_CLASSES.get(int(value), "Unknown")
            percentage = 100 * count / frame.size

            print(
                f"Class {int(value):2d} ({label:24s}) "
                f"{count:6d} pixels | {percentage:6.2f}%"
            )

    print("\n" + "=" * 65)
    print("OPTICAL BAND VALUE RANGES")
    print("=" * 65)

    for band in [
        "B02", "B03", "B04", "B05", "B06",
        "B07", "B08", "B8A", "B11", "B12"
    ]:
        arr = ds[band].values

        print(
            f"{band}: min={np.nanmin(arr)}, "
            f"max={np.nanmax(arr)}, "
            f"mean={np.nanmean(arr):.2f}, "
            f"std={np.nanstd(arr):.2f}, "
            f"finite={np.isfinite(arr).mean():.4f}"
        )