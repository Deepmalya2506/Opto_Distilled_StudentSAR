from pathlib import Path
import sys

import numpy as np
import torch
import xarray as xr

from src.data.datasets import Sen12Landslides
from src.data.transforms import Normalize

UPSTREAM = Path(r"D:\Research\Sen12Landslides")

SAMPLE = (
    UPSTREAM
    / "data"
    / "data_harmonized"
    / "s1asc"
    / "chimanimani_s1asc_1004.nc"
)

NORM_JSON = (
    UPSTREAM
    / "tasks"
    / "S12LS-LD"
    / "harmonized"
    / "s1asc"
    / "norm.json"
)

assert SAMPLE.exists(), f"Sample missing: {SAMPLE}"
assert NORM_JSON.exists(), f"Normalization file missing: {NORM_JSON}"

print("=" * 65)
print("UPSTREAM LOADER SMOKE TEST")
print("=" * 65)

print("Python:", sys.executable)
print("Imported dataset module:", sys.modules["src.data.datasets"].__file__)
print("Sample:", SAMPLE)

# --------------------------------------------------
# 1. Load one sample using the official dataset class
# --------------------------------------------------

dataset = Sen12Landslides(
    files=[{
        "id": "chimanimani_1004",
        "s1asc": str(SAMPLE)
    }],
    modality="s1asc",
    min_date="2015-12-03",
    use_dem=False,
    transforms=None,
)

assert len(dataset) == 1

raw = dataset[0]

img = raw["img"]
dates = raw["dates"]
mask = raw["msk"]

print("\n--- RAW LOADER OUTPUT ---")
print("Image shape:", tuple(img.shape))
print("Image dtype:", img.dtype)
print("Dates shape:", tuple(dates.shape))
print("Mask shape:", tuple(mask.shape))
print("Mask dtype:", mask.dtype)

print("Dates (days from 2015-12-03):")
print(dates.tolist())

# --------------------------------------------------
# 2. Structural checks
# --------------------------------------------------

assert img.shape == (15, 2, 128, 128), (
    f"Unexpected image shape: {img.shape}"
)

assert dates.shape == (15,)
assert mask.shape == (128, 128)

assert img.dtype == torch.float32
assert mask.dtype == torch.int64

assert torch.all(dates[1:] > dates[:-1]), (
    "Dates are not strictly increasing"
)

assert set(torch.unique(mask).tolist()).issubset({0, 1}), (
    "Mask contains unexpected values"
)

assert torch.isfinite(img).all(), (
    "Image contains NaN or infinite values"
)

positive_pixels = int(mask.sum().item())

print("\n--- LABEL CHECK ---")
print("Unique mask values:", torch.unique(mask).tolist())
print("Positive mask pixels:", positive_pixels)
print("Negative mask pixels:", int((mask == 0).sum().item()))

# --------------------------------------------------
# 3. Verify actual channel order against NetCDF
# --------------------------------------------------

with xr.open_dataset(SAMPLE, engine="netcdf4") as ds:
    stored_bands = [
        name for name in ds.data_vars
        if name not in {"MASK", "DEM", "SCL", "spatial_ref"}
    ]

    print("\n--- CHANNEL ORDER CHECK ---")
    print("NetCDF image variable order:", stored_bands)

    assert stored_bands == ["VH", "VV"], (
        f"Unexpected band order: {stored_bands}"
    )

    expected_vh = torch.from_numpy(
        ds["VH"].isel(time=0).transpose("y", "x").values.copy()
    ).float()

    expected_vv = torch.from_numpy(
        ds["VV"].isel(time=0).transpose("y", "x").values.copy()
    ).float()

    expected_mask = torch.from_numpy(
        (ds["MASK"].isel(time=0).transpose("y", "x").values > 0)
        .astype(np.int64)
    )

    vh_match = torch.allclose(img[0, 0], expected_vh)
    vv_match = torch.allclose(img[0, 1], expected_vv)
    mask_match = torch.equal(mask, expected_mask)

    print("Channel 0 matches VH:", vh_match)
    print("Channel 1 matches VV:", vv_match)
    print("Loader mask matches NetCDF mask:", mask_match)

    assert vh_match and vv_match and mask_match, (
        "Loader output does not match source NetCDF variables"
    )

# --------------------------------------------------
# 4. Inspect raw SAR ranges
# --------------------------------------------------

print("\n--- RAW SAR VALUE RANGES ---")

for channel, name in enumerate(["VH", "VV"]):
    values = img[:, channel]

    print(
        f"{name}: min={values.min().item():.4f}, "
        f"max={values.max().item():.4f}, "
        f"mean={values.mean().item():.4f}, "
        f"std={values.std().item():.4f}"
    )

# --------------------------------------------------
# 5. Test official normalization transform
# --------------------------------------------------

normalizer = Normalize(
    norm_json=str(NORM_JSON),
    clip_data=False,
)

normalized = normalizer({
    "img": img.clone(),
    "dates": dates.clone(),
    "msk": mask.clone(),
})

normalized_img = normalized["img"]

print("\n--- NORMALIZATION CHECK ---")
print("Normalized shape:", tuple(normalized_img.shape))
print("Normalized dtype:", normalized_img.dtype)
print("All values finite:", torch.isfinite(normalized_img).all().item())

for channel, name in enumerate(["VH", "VV"]):
    values = normalized_img[:, channel]

    print(
        f"{name}: min={values.min().item():.4f}, "
        f"max={values.max().item():.4f}, "
        f"mean={values.mean().item():.4f}, "
        f"std={values.std().item():.4f}"
    )

assert normalized_img.shape == img.shape
assert torch.isfinite(normalized_img).all()
assert torch.equal(normalized["msk"], mask)

print("\n" + "=" * 65)
print("ALL SINGLE-SAMPLE LOADER CHECKS PASSED")
print("=" * 65)
