
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


def get_dem_cube(ds):
    return (
        ds["DEM"]
        .transpose("time", "y", "x")
        .values.astype(np.float32)
    )


with xr.open_dataset(S1_PATH) as s1, xr.open_dataset(S2_PATH) as s2:

    dem_s1 = get_dem_cube(s1)
    dem_s2 = get_dem_cube(s2)

    mask = (
        s1["MASK"]
        .isel(time=0)
        .transpose("y", "x")
        .values > 0
    )

    x = s1.x.values
    y = s1.y.values

    # -------------------------------------------------
    # 1. DEM TEMPORAL CONSISTENCY
    # -------------------------------------------------

    print("=" * 65)
    print("SENTINEL-1 DEM TEMPORAL CONSISTENCY")
    print("=" * 65)

    for i in range(dem_s1.shape[0]):
        difference = np.abs(dem_s1[i] - dem_s1[0])

        print(
            f"Frame {i:02d}: "
            f"min={dem_s1[i].min():.2f}, "
            f"max={dem_s1[i].max():.2f}, "
            f"max difference from frame 0={difference.max():.4f}"
        )

    print(
        "\nS1 DEM identical over time:",
        np.array_equal(dem_s1, np.broadcast_to(
            dem_s1[0], dem_s1.shape
        ))
    )

    print("\n" + "=" * 65)
    print("SENTINEL-2 DEM TEMPORAL CONSISTENCY")
    print("=" * 65)

    for i in range(dem_s2.shape[0]):
        difference = np.abs(dem_s2[i] - dem_s2[0])

        print(
            f"Frame {i:02d}: "
            f"max difference from frame 0={difference.max():.4f}"
        )

    print(
        "\nS2 DEM identical over time:",
        np.array_equal(dem_s2, np.broadcast_to(
            dem_s2[0], dem_s2.shape
        ))
    )

    # -------------------------------------------------
    # 2. CROSS-MODAL DEM COMPARISON
    # -------------------------------------------------

    print("\n" + "=" * 65)
    print("CROSS-MODAL DEM COMPARISON")
    print("=" * 65)

    dem_difference = np.abs(dem_s1[0] - dem_s2[0])

    print("Same shape:", dem_s1[0].shape == dem_s2[0].shape)
    print("Exact equality:", np.array_equal(dem_s1[0], dem_s2[0]))
    print("Maximum absolute difference:", dem_difference.max())
    print("Mean absolute difference:", dem_difference.mean())

    # -------------------------------------------------
    # 3. ELEVATION NEIGHBOUR DIFFERENCES
    # -------------------------------------------------

    dem = dem_s1[0]

    dx_diff = np.abs(np.diff(dem, axis=1))
    dy_diff = np.abs(np.diff(dem, axis=0))

    print("\n" + "=" * 65)
    print("NEIGHBOURING PIXEL ELEVATION DIFFERENCES")
    print("=" * 65)

    for name, values in [
        ("Horizontal", dx_diff),
        ("Vertical", dy_diff),
    ]:
        print(f"\n{name}:")
        print("Mean:", np.mean(values))
        print("Median:", np.median(values))
        print("95th percentile:", np.percentile(values, 95))
        print("99th percentile:", np.percentile(values, 99))
        print("Maximum:", np.max(values))

    # -------------------------------------------------
    # 4. SLOPE USING ACTUAL COORDINATES
    # -------------------------------------------------

    dz_dy, dz_dx = np.gradient(dem, y, x)

    slope = np.degrees(
        np.arctan(np.hypot(dz_dx, dz_dy))
    )

    print("\n" + "=" * 65)
    print("SLOPE STATISTICS")
    print("=" * 65)

    for percentile in [0, 25, 50, 75, 90, 95, 99, 100]:
        print(
            f"Percentile {percentile:3d}: "
            f"{np.percentile(slope, percentile):.2f} degrees"
        )

    print("\nInside mask:")
    print("Mean elevation:", dem[mask].mean())
    print("Median elevation:", np.median(dem[mask]))
    print("Mean slope:", slope[mask].mean())
    print("Median slope:", np.median(slope[mask]))

    print("\nOutside mask:")
    print("Mean elevation:", dem[~mask].mean())
    print("Median elevation:", np.median(dem[~mask]))
    print("Mean slope:", slope[~mask].mean())
    print("Median slope:", np.median(slope[~mask]))

    # -------------------------------------------------
    # 5. DIAGNOSTIC FIGURE
    # -------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    im0 = axes[0, 0].imshow(dem, cmap="terrain", origin="upper")
    axes[0, 0].set_title("S1 DEM — Frame 0")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(slope, cmap="magma", origin="upper")
    axes[0, 1].set_title("Slope using actual x/y coordinates")
    fig.colorbar(im1, ax=axes[0, 1], label="Degrees")

    im2 = axes[1, 0].imshow(
        dem_difference, cmap="viridis", origin="upper"
    )
    axes[1, 0].set_title("Absolute S1–S2 DEM Difference")
    fig.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(
        np.maximum(dx_diff[:-1], dy_diff[:, :-1]),
        cmap="inferno",
        origin="upper",
    )
    axes[1, 1].set_title("Neighbouring Elevation Differences")
    fig.colorbar(im3, ax=axes[1, 1])

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("DEM Integrity Audit — Chimanimani 1004")
    fig.tight_layout()

    path = OUTPUT / "07_dem_integrity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved:", path.resolve())
