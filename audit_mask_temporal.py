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
        m for m in tar.getmembers()
        if m.isfile() and Path(m.name).name == target
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / target
        with tar.extractfile(member) as src, path.open("wb") as dst:
            dst.write(src.read())

        with xr.open_dataset(path) as ds:
            masks = ds["MASK"].transpose("time", "y", "x").values > 0
            dates = ds["time"].values

            reference = masks[0]
            print("MASK TEMPORAL AUDIT")
            print("Shape:", masks.shape)
            print("Unique values:", np.unique(ds["MASK"].values))

            print("\nPer-timestep positive pixels:")
            for i, mask in enumerate(masks):
                intersection = np.logical_and(reference, mask).sum()
                union = np.logical_or(reference, mask).sum()
                iou = intersection / union if union else 1.0

                print(
                    f"{i:02d} | {str(dates[i])[:10]} | "
                    f"positive={mask.sum():5d} | "
                    f"changed_vs_t0={(mask != reference).sum():5d} | "
                    f"IoU_vs_t0={iou:.6f}"
                )

            print("\nAcross all timestamps:")
            print("Pixels positive at least once:", masks.any(axis=0).sum())
            print("Pixels positive at every timestamp:", masks.all(axis=0).sum())
            print("Pixels positive at first timestamp:", reference.sum())
            print(
                "Pixels positive at last timestamp:",
                masks[-1].sum()
            )
