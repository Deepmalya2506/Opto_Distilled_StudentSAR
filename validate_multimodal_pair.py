from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(r"D:\Research\Sen12Landslides\data\data_harmonized")

S1_PATH = ROOT / "s1asc" / "chimanimani_s1asc_1004.nc"
S2_PATH = ROOT / "s2" / "chimanimani_s2_1004.nc"

for path in [S1_PATH, S2_PATH]:
    if not path.exists():
        raise FileNotFoundError(path)

def inspect(ds, name):
    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    print("Dimensions:", dict(ds.sizes))
    print("Variables:", list(ds.data_vars))
    print("Coordinates:", list(ds.coords))
    print("Attributes:")
    for key in ["crs", "event_date", "annotated", "satellite"]:
        print(f"  {key}: {ds.attrs.get(key)}")

    for coord in ["x", "y", "time"]:
        if coord in ds.coords:
            values = np.asarray(ds[coord].values)
            print(f"\n{coord}:")
            print("  dtype:", values.dtype)
            print("  count:", len(values))
            print("  first:", values[0])
            print("  last:", values[-1])

    print("\nImage variables:")
    for variable in ds.data_vars:
        if variable not in {"MASK", "DEM", "SCL", "spatial_ref"}:
            da = ds[variable]
            values = np.asarray(da.values)
            finite = np.isfinite(values)

            print(
                variable,
                "| dims:", da.dims,
                "| shape:", values.shape,
                "| dtype:", values.dtype,
                "| finite fraction:", finite.mean()
            )

    if "MASK" in ds:
        mask = ds["MASK"]
        print("\nMask dimensions:", mask.dims)
        print("Mask shape:", mask.shape)
        print("Mask unique values:", np.unique(mask.values))

with xr.open_dataset(S1_PATH, engine="netcdf4") as s1, \
     xr.open_dataset(S2_PATH, engine="netcdf4") as s2:

    inspect(s1, "SENTINEL-1 ASCENDING")
    inspect(s2, "SENTINEL-2 OPTICAL")

    print("\n" + "=" * 65)
    print("CROSS-MODAL COMPARISON")
    print("=" * 65)

    for coord in ["x", "y"]:
        if coord in s1.coords and coord in s2.coords:
            a = np.asarray(s1[coord].values)
            b = np.asarray(s2[coord].values)

            print(f"\n{coord}:")
            print("  Same shape:", a.shape == b.shape)
            print("  Exact coordinate match:", np.array_equal(a, b))

            if a.shape == b.shape:
                print("  Maximum absolute difference:",
                      np.max(np.abs(a.astype(float) - b.astype(float))))

    print("\nCRS equality:",
          s1.attrs.get("crs") == s2.attrs.get("crs"))

    print("\nSpatial dimension equality:",
          s1.sizes.get("x") == s2.sizes.get("x") and
          s1.sizes.get("y") == s2.sizes.get("y"))

    if "MASK" in s1 and "MASK" in s2:
        m1 = np.asarray(s1["MASK"].values)
        m2 = np.asarray(s2["MASK"].values)

        print("\nMask shapes:", m1.shape, m2.shape)

        if m1.shape == m2.shape:
            m1 = m1 > 0
            m2 = m2 > 0

            intersection = np.logical_and(m1, m2).sum()
            union = np.logical_or(m1, m2).sum()

            iou = intersection / union if union else 1.0

            print("Mask exact equality:", np.array_equal(m1, m2))
            print("Mask IoU:", iou)
        else:
            print("Mask comparison skipped: shapes differ.")
    else:
        print("\nOne or both modalities lack MASK.")
