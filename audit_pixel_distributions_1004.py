
from pathlib import Path
import csv

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(r"D:\Research\Sen12Landslides\data\data_harmonized")

S1_PATH = ROOT / "s1asc" / "chimanimani_s1asc_1004.nc"
S2_PATH = ROOT / "s2" / "chimanimani_s2_1004.nc"

TARGET_DATE = "2019-02-28"

OUTPUT = (
    Path(__file__).resolve().parent
    / "outputs"
    / "visual_audit"
    / "sample_1004"
    / "distribution_2019-02-28"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

OPTICAL_BANDS = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
]

SAR_BANDS = ["VV", "VH"]


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def get_index(ds, date):
    dates = ds.time.values.astype("datetime64[D]")
    matches = np.flatnonzero(dates == np.datetime64(date))

    if len(matches) != 1:
        raise ValueError(
            f"Expected one observation on {date}, "
            f"found {len(matches)}."
        )

    return int(matches[0])


def get_frame(ds, variable, index):
    return (
        ds[variable]
        .isel(time=index)
        .transpose("y", "x")
        .values.astype(np.float64)
    )


def calculate_statistics(values):
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError("No finite values to analyse.")

    return {
        "count": len(values),
        "mean": np.mean(values),
        "std": np.std(values, ddof=1),
        "min": np.min(values),
        "p10": np.percentile(values, 10),
        "p25": np.percentile(values, 25),
        "median": np.median(values),
        "p75": np.percentile(values, 75),
        "p90": np.percentile(values, 90),
        "max": np.max(values),
    }


# ---------------------------------------------------------
# LOAD AND VALIDATE
# ---------------------------------------------------------

with xr.open_dataset(S1_PATH) as s1, xr.open_dataset(S2_PATH) as s2:

    i1 = get_index(s1, TARGET_DATE)
    i2 = get_index(s2, TARGET_DATE)

    mask_s1 = get_frame(s1, "MASK", 0) > 0
    mask_s2 = get_frame(s2, "MASK", 0) > 0

    if not np.array_equal(mask_s1, mask_s2):
        raise ValueError("S1 and S2 masks are different.")

    if not np.array_equal(s1.x.values, s2.x.values):
        raise ValueError("X coordinates differ.")

    if not np.array_equal(s1.y.values, s2.y.values):
        raise ValueError("Y coordinates differ.")

    mask = mask_s1
    background = ~mask

    print("=" * 65)
    print("SAMPLE AND DATE VALIDATION")
    print("=" * 65)

    print("Date:", TARGET_DATE)
    print("S1 index:", i1)
    print("S2 index:", i2)
    print("Landslide pixels:", mask.sum())
    print("Background pixels:", background.sum())
    print("Positive pixel percentage:", 100 * mask.mean())

    # -----------------------------------------------------
    # COLLECT VARIABLES
    # -----------------------------------------------------

    variables = {}

    for band in SAR_BANDS:
        variables[band] = get_frame(s1, band, i1)

    for band in OPTICAL_BANDS:
        variables[band] = get_frame(s2, band, i2)

    # Difference in dB:
    # 10 log10(PVV/PVH) = VV_dB - VH_dB
    variables["VV_minus_VH_dB"] = (
        variables["VV"] - variables["VH"]
    )

    # -----------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # -----------------------------------------------------

    rows = []

    print("\n" + "=" * 65)
    print("LANDSLIDE VS BACKGROUND DISTRIBUTIONS")
    print("=" * 65)

    for name, image in variables.items():

        if image.shape != mask.shape:
            raise ValueError(f"Unexpected shape for {name}")

        if not np.isfinite(image).all():
            print(f"Warning: non-finite values found in {name}")

        positive = image[mask]
        negative = image[background]

        pos_stats = calculate_statistics(positive)
        neg_stats = calculate_statistics(negative)

        pooled_std = np.sqrt(
            (pos_stats["std"] ** 2 + neg_stats["std"] ** 2) / 2
        )

        smd = (
            (pos_stats["mean"] - neg_stats["mean"]) / pooled_std
            if pooled_std > 0 else np.nan
        )

        median_difference = (
            pos_stats["median"] - neg_stats["median"]
        )

        print(f"\n{name}")
        print("-" * 45)

        print(
            f"Landslide: mean={pos_stats['mean']:.3f}, "
            f"median={pos_stats['median']:.3f}, "
            f"std={pos_stats['std']:.3f}"
        )

        print(
            f"Background: mean={neg_stats['mean']:.3f}, "
            f"median={neg_stats['median']:.3f}, "
            f"std={neg_stats['std']:.3f}"
        )

        print(f"Median difference: {median_difference:.3f}")
        print(f"Standardised mean difference: {smd:.3f}")

        for group, stats in [
            ("landslide", pos_stats),
            ("background", neg_stats),
        ]:
            row = {"variable": name, "group": group}
            row.update(stats)
            rows.append(row)

    # Save full statistics
    csv_path = OUTPUT / "pixel_statistics.csv"

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved statistics:", csv_path)

    # -----------------------------------------------------
    # HISTOGRAMS: SAR
    # -----------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, band in zip(axes, SAR_BANDS):

        ax.hist(
            variables[band][background],
            bins=40,
            density=True,
            alpha=0.55,
            label="Background",
        )

        ax.hist(
            variables[band][mask],
            bins=40,
            density=True,
            alpha=0.65,
            label="Landslide",
        )

        ax.set_title(band)
        ax.set_xlabel("Backscatter (dB)")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("SAR Distributions — 28 February 2019")
    fig.tight_layout()

    path = OUTPUT / "sar_distributions.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------
    # HISTOGRAMS: OPTICAL
    # -----------------------------------------------------

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))

    for ax, band in zip(axes.ravel(), OPTICAL_BANDS):

        ax.hist(
            variables[band][background],
            bins=35,
            density=True,
            alpha=0.55,
            label="Background",
        )

        ax.hist(
            variables[band][mask],
            bins=35,
            density=True,
            alpha=0.65,
            label="Landslide",
        )

        ax.set_title(band)
        ax.set_xlabel("Stored value")
        ax.set_ylabel("Density")

    axes[0, 0].legend()

    fig.suptitle("Sentinel-2 Distributions — 28 February 2019")
    fig.tight_layout()

    path = OUTPUT / "optical_distributions.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------
    # SCL CLASS COUNTS INSIDE AND OUTSIDE MASK
    # -----------------------------------------------------

    scl = get_frame(s2, "SCL", i2).astype(np.int32)

    print("\n" + "=" * 65)
    print("SCL CLASS DISTRIBUTION BY MASK")
    print("=" * 65)

    for value in np.unique(scl):

        count_inside = int(np.sum(scl[mask] == value))
        count_outside = int(np.sum(scl[background] == value))

        print(
            f"Class {value:3d}: "
            f"inside={count_inside:4d}, "
            f"outside={count_outside:5d}"
        )

print("\nDistribution audit completed.")
print("Output directory:", OUTPUT.resolve())


"""
- Histograms: Do the landslide and background distributions overlap heavily, or are they distinguishable?
- Median differences: Is there a shift in typical backscatter or reflectance?
- Standardised mean differences: How large is the difference relative to within-group variability?
- VV–VH difference: Does the relationship between the two polarisations differ across the annotated footprint?
- SCL by mask: Are annotated pixels disproportionately associated with bare-soil or vegetation classifications?

An important caution: a large difference in one band does not prove that the band identifies landslides. It could reflect terrain, vegetation, land-cover composition or the spatial distribution of the annotations.

"""