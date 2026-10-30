
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATA_ROOT = Path(
    r"D:\Research\Sen12Landslides\data\data_harmonized"
)

S1_PATH = DATA_ROOT / "s1asc" / "chimanimani_s1asc_1004.nc"
S2_PATH = DATA_ROOT / "s2" / "chimanimani_s2_1004.nc"

OUTPUT = Path("outputs/visual_audit/sample_1004")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def get_frame(ds, variable, index):
    """Return a frame consistently arranged as (y, x)."""
    return (
        ds[variable]
        .isel(time=index)
        .transpose("y", "x")
        .values
    )


def shared_limits(ds, variable, low=2, high=98):
    """One common display stretch across all dates."""
    values = ds[variable].values.astype(np.float32)
    return (
        float(np.nanpercentile(values, low)),
        float(np.nanpercentile(values, high)),
    )


def rgb_composite(ds, bands, index, limits):
    """Build an RGB composite with shared per-band stretches."""
    channels = []

    for band in bands:
        image = get_frame(ds, band, index).astype(np.float32)
        lo, hi = limits[band]

        image = np.clip((image - lo) / (hi - lo + 1e-8), 0, 1)
        channels.append(image)

    return np.stack(channels, axis=-1)


def save_atlas(ds, dates, plot_function, filename, title):
    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        plot_function(ax, i)
        ax.set_title(str(dates[i]), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = OUTPUT / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", path.resolve())


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

with xr.open_dataset(S1_PATH) as s1, xr.open_dataset(S2_PATH) as s2:

    dates_s1 = s1.time.values.astype("datetime64[D]")
    dates_s2 = s2.time.values.astype("datetime64[D]")

    mask = get_frame(s1, "MASK", 0)

    # Confirm that the mask is static in both files.
    s1_masks_static = all(
        np.array_equal(get_frame(s1, "MASK", 0),
                       get_frame(s1, "MASK", i))
        for i in range(s1.sizes["time"])
    )

    s2_masks_static = all(
        np.array_equal(get_frame(s2, "MASK", 0),
                       get_frame(s2, "MASK", i))
        for i in range(s2.sizes["time"])
    )

    print("S1 mask static:", s1_masks_static)
    print("S2 mask static:", s2_masks_static)
    print("Mask positive pixels:", int((mask > 0).sum()))

    # Inspect metadata before interpreting pixel values.
    print("\nSCL attributes:")
    print(s2["SCL"].attrs)
    print("\nSCL encoding:")
    print(s2["SCL"].encoding)

    for band in ["B02", "B03", "B04", "B08", "SCL"]:
        print(f"\n{band} attributes:")
        print(s2[band].attrs)
        print(f"{band} encoding:")
        print(s2[band].encoding)

    # Shared display stretches across time.
    optical_limits = {
        band: shared_limits(s2, band)
        for band in ["B02", "B03", "B04", "B08"]
    }

    sar_limits = {
        band: shared_limits(s1, band)
        for band in ["VV", "VH"]
    }

    print("\nOptical display limits:", optical_limits)
    print("SAR display limits:", sar_limits)

    # -----------------------------------------------------
    # 1. TRUE COLOUR: B04, B03, B02
    # -----------------------------------------------------

    def plot_rgb(ax, i):
        rgb = rgb_composite(
            s2, ["B04", "B03", "B02"], i, optical_limits
        )
        ax.imshow(rgb, origin="upper")
        ax.contour(
            mask, levels=[0.5], colors="yellow", linewidths=0.7
        )

    save_atlas(
        s2, dates_s2, plot_rgb,
        "01_s2_true_colour.png",
        "Sentinel-2 True Colour (B04/B03/B02) — Yellow: Landslide Mask"
    )

    # -----------------------------------------------------
    # 2. FALSE COLOUR: NIR, RED, GREEN
    # -----------------------------------------------------

    def plot_false_colour(ax, i):
        rgb = rgb_composite(
            s2, ["B08", "B04", "B03"], i, optical_limits
        )
        ax.imshow(rgb, origin="upper")
        ax.contour(
            mask, levels=[0.5], colors="yellow", linewidths=0.7
        )

    save_atlas(
        s2, dates_s2, plot_false_colour,
        "02_s2_false_colour.png",
        "Sentinel-2 False Colour (B08/B04/B03) — Yellow: Landslide Mask"
    )

    # -----------------------------------------------------
    # 3. SCENE CLASSIFICATION LAYER
    # -----------------------------------------------------

    scl_cmap = plt.get_cmap("tab20", 12)
    scl_cmap.set_bad("lightgray")

    def plot_scl(ax, i):
        frame = get_frame(s2, "SCL", i)
        frame = np.ma.masked_where(frame == 255, frame)

        ax.imshow(
            frame,
            cmap=scl_cmap,
            vmin=0,
            vmax=11,
            origin="upper",
            interpolation="nearest",
        )
        ax.contour(
            mask, levels=[0.5], colors="black", linewidths=0.6
        )

    save_atlas(
        s2, dates_s2, plot_scl,
        "03_s2_scl.png",
        "Sentinel-2 SCL — Grey: Stored Value 255"
    )

    # -----------------------------------------------------
    # 4. SENTINEL-1 VV
    # -----------------------------------------------------

    def plot_vv(ax, i):
        frame = get_frame(s1, "VV", i)
        ax.imshow(
            frame,
            cmap="gray",
            vmin=sar_limits["VV"][0],
            vmax=sar_limits["VV"][1],
            origin="upper",
        )
        ax.contour(
            mask, levels=[0.5], colors="red", linewidths=0.7
        )

    save_atlas(
        s1, dates_s1, plot_vv,
        "04_s1_vv.png",
        "Sentinel-1 VV Backscatter — Red: Landslide Mask"
    )

    # -----------------------------------------------------
    # 5. SENTINEL-1 VH
    # -----------------------------------------------------

    def plot_vh(ax, i):
        frame = get_frame(s1, "VH", i)
        ax.imshow(
            frame,
            cmap="gray",
            vmin=sar_limits["VH"][0],
            vmax=sar_limits["VH"][1],
            origin="upper",
        )
        ax.contour(
            mask, levels=[0.5], colors="red", linewidths=0.7
        )

    save_atlas(
        s1, dates_s1, plot_vh,
        "05_s1_vh.png",
        "Sentinel-1 VH Backscatter — Red: Landslide Mask"
    )

print("\nVisual atlas generation completed.")
