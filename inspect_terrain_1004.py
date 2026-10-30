
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path(r"D:\Research\Sen12Landslides\data\data_harmonized")

S1_PATH = ROOT / "s1asc" / "chimanimani_s1asc_1004.nc"
S2_PATH = ROOT / "s2" / "chimanimani_s2_1004.nc"

OUTPUT = (
    Path(__file__).resolve().parent
    / "outputs"
    / "visual_audit"
    / "sample_1004"
)

OUTPUT.mkdir(parents=True, exist_ok=True)


def frame(ds, variable):
    return (
        ds[variable]
        .isel(time=0)
        .transpose("y", "x")
        .values.astype(np.float32)
    )


with xr.open_dataset(S1_PATH) as s1, xr.open_dataset(S2_PATH) as s2:

    dem = frame(s1, "DEM")
    mask = frame(s1, "MASK")

    print("DEM shape:", dem.shape)
    print("DEM min:", np.nanmin(dem))
    print("DEM max:", np.nanmax(dem))
    print("DEM mean:", np.nanmean(dem))
    print("DEM std:", np.nanstd(dem))
    print("DEM unique count:", len(np.unique(dem)))
    print("DEM finite fraction:", np.isfinite(dem).mean())

    print("\nDEM attributes:", s1["DEM"].attrs)
    print("DEM encoding:", s1["DEM"].encoding)

    print("\nSpatial reference attributes:")
    print(s1["spatial_ref"].attrs)

    print("\nGlobal attributes:")
    print(s1.attrs)

    # Pixel spacing from coordinate arrays.
    x = s1.x.values
    y = s1.y.values

    print("\nPixel spacing:")
    print("X:", np.diff(x)[:5])
    print("Y:", np.diff(y)[:5])

    # Calculate approximate slope in degrees.
    # This assumes DEM values are in metres and x/y spacing is metres.
    dx = float(np.median(np.abs(np.diff(x))))
    dy = float(np.median(np.abs(np.diff(y))))

    dz_dy, dz_dx = np.gradient(dem, dy, dx)

    slope = np.degrees(
        np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    )

    # Display
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    im0 = axes[0, 0].imshow(dem, cmap="terrain", origin="upper")
    axes[0, 0].set_title("DEM / Elevation")
    plt.colorbar(im0, ax=axes[0, 0], label="Stored DEM units")

    im1 = axes[0, 1].imshow(slope, cmap="magma", origin="upper")
    axes[0, 1].set_title("Approximate Slope")
    plt.colorbar(im1, ax=axes[0, 1], label="Degrees")

    im2 = axes[1, 0].imshow(mask, cmap="gray", origin="upper")
    axes[1, 0].set_title("Landslide Mask")

    im3 = axes[1, 1].imshow(dem, cmap="terrain", origin="upper")
    axes[1, 1].contour(
        mask,
        levels=[0.5],
        colors="red",
        linewidths=0.8
    )
    axes[1, 1].set_title("DEM with Landslide Boundaries")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Chimanimani 1004 — Terrain Inspection")
    fig.tight_layout()

    output_path = OUTPUT / "06_terrain_inspection.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved:", output_path)
