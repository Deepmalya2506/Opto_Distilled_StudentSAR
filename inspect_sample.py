import tarfile
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

archive = Path(
    r"D:\Research\Sen12Landslides\data_zipped\data_harmonized\s1asc\s1asc_part01.tar.gz"
)
target = "chimanimani_s1asc_1004.nc"

with tarfile.open(archive, "r:gz") as tar:
    member = next(
        (m for m in tar.getmembers()
         if m.isfile() and Path(m.name).name == target),
        None,
    )
    if member is None:
        raise FileNotFoundError(f"{target} not found in archive")

    with tempfile.TemporaryDirectory(prefix="sen12ls_inspect_") as tmp:
        sample_path = Path(tmp) / target

        with tar.extractfile(member) as src, sample_path.open("wb") as dst:
            dst.write(src.read())

        with xr.open_dataset(sample_path, engine="netcdf4") as ds:
            print("=" * 70)
            print("SAMPLE:", target)
            print("FILE SIZE:", sample_path.stat().st_size, "bytes")
            print("\nDATASET SUMMARY")
            print(ds)

            print("\nDIMENSIONS")
            for name, size in ds.sizes.items():
                print(f"  {name}: {size}")

            print("\nCOORDINATES")
            for name, coord in ds.coords.items():
                values = np.asarray(coord.values)
                preview = values[:5] if values.ndim == 1 else values.shape
                print(
                    f"  {name}: dims={coord.dims}, "
                    f"dtype={values.dtype}, preview={preview}"
                )

            print("\nVARIABLES IN STORED ORDER")
            for name, da in ds.data_vars.items():
                arr = np.asarray(da.values)
                finite = np.isfinite(arr)
                finite_values = arr[finite]

                print(f"\n  {name}")
                print(f"    dims: {da.dims}")
                print(f"    shape: {arr.shape}")
                print(f"    dtype: {arr.dtype}")

                if finite_values.size:
                    print(
                        f"    finite min/max: "
                        f"{finite_values.min()} / {finite_values.max()}"
                    )
                    print(
                        f"    non-finite fraction: "
                        f"{1 - finite.mean():.6f}"
                    )
                    print(
                        f"    -9999 fraction: "
                        f"{np.mean(arr == -9999):.6f}"
                    )
                else:
                    print("    No finite values")

                if name.upper() == "MASK":
                    values, counts = np.unique(arr, return_counts=True)
                    print("    mask value counts:")
                    for value, count in zip(values, counts):
                        print(
                            f"      {value}: {count} pixels "
                            f"({count / arr.size:.2%})"
                        )

                print(f"    attributes: {dict(da.attrs)}")

            print("\nGLOBAL ATTRIBUTES")
            for key, value in ds.attrs.items():
                print(f"  {key}: {value}")