
from pathlib import Path

import numpy as np
import xarray as xr
import rasterio

from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import ColorInterp


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATA_ROOT = Path(
    r"D:\Research\Sen12Landslides\data\data_harmonized"
)

S1_PATH = DATA_ROOT / "s1asc" / "chimanimani_s1asc_1004.nc"
S2_PATH = DATA_ROOT / "s2" / "chimanimani_s2_1004.nc"

OUTPUT = (
    Path(__file__).resolve().parent
    / "outputs"
    / "visual_audit"
    / "sample_1004"
    / "qgis_geotiffs"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

TARGET_DATE = "2019-02-28"


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def get_crs_and_transform(ds):

    attrs = ds["spatial_ref"].attrs

    wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")

    if not wkt:
        raise ValueError("CRS WKT missing from spatial_ref.")

    crs = CRS.from_wkt(wkt)

    geotransform = attrs.get("GeoTransform")

    if not geotransform:
        raise ValueError("GeoTransform missing from spatial_ref.")

    values = tuple(float(v) for v in geotransform.split())

    if len(values) != 6:
        raise ValueError(f"Unexpected GeoTransform: {geotransform}")

    transform = Affine.from_gdal(*values)

    return crs, transform


def find_date_index(ds, date):

    dates = ds.time.values.astype("datetime64[D]")
    target = np.datetime64(date)

    indices = np.flatnonzero(dates == target)

    if len(indices) != 1:
        raise ValueError(
            f"Expected exactly one observation on {date}; "
            f"found {len(indices)}."
        )

    return int(indices[0])


def get_frame(ds, variable, index):

    return (
        ds[variable]
        .isel(time=index)
        .transpose("y", "x")
        .values
    )


def write_geotiff(
    filename,
    arrays,
    descriptions,
    crs,
    transform,
    colorinterp=None,
    tags=None,
):

    arrays = [np.asarray(a) for a in arrays]

    shapes = {a.shape for a in arrays}

    if len(shapes) != 1:
        raise ValueError(f"Different array shapes: {shapes}")

    stack = np.stack(arrays, axis=0)

    if stack.ndim != 3:
        raise ValueError(f"Unexpected raster shape: {stack.shape}")

    if len(descriptions) != stack.shape[0]:
        raise ValueError("Band descriptions do not match band count.")

    profile = {
        "driver": "GTiff",
        "height": stack.shape[1],
        "width": stack.shape[2],
        "count": stack.shape[0],
        "dtype": stack.dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }

    path = OUTPUT / filename

    with rasterio.open(path, "w", **profile) as dst:

        dst.write(stack)

        for i, description in enumerate(descriptions, start=1):
            dst.set_band_description(i, description)

        if colorinterp is not None:
            dst.colorinterp = colorinterp

        if tags:
            dst.update_tags(**tags)

    print("Saved:", path)


# ---------------------------------------------------------
# OPEN ORIGINAL NETCDF FILES
# ---------------------------------------------------------

with xr.open_dataset(S1_PATH) as s1, xr.open_dataset(S2_PATH) as s2:

    s1_crs, s1_transform = get_crs_and_transform(s1)
    s2_crs, s2_transform = get_crs_and_transform(s2)

    if s1_crs != s2_crs:
        raise ValueError("S1 and S2 CRS differ.")

    if s1_transform != s2_transform:
        raise ValueError("S1 and S2 GeoTransforms differ.")

    if not np.array_equal(s1.x.values, s2.x.values):
        raise ValueError("X coordinates differ.")

    if not np.array_equal(s1.y.values, s2.y.values):
        raise ValueError("Y coordinates differ.")

    # Exact common acquisition date
    s1_idx = find_date_index(s1, TARGET_DATE)
    s2_idx = find_date_index(s2, TARGET_DATE)

    print("S1 index:", s1_idx)
    print("S2 index:", s2_idx)
    print("CRS:", s1_crs)
    print("Transform:", s1_transform)

    # -----------------------------------------------------
    # 1. SENTINEL-2 TRUE COLOUR
    # -----------------------------------------------------

    true_colour = [
        get_frame(s2, band, s2_idx).astype(np.int16)
        for band in ["B04", "B03", "B02"]
    ]

    write_geotiff(
        "01_s2_true_colour_2019-02-28.tif",
        true_colour,
        ["Red - B04", "Green - B03", "Blue - B02"],
        s2_crs,
        s2_transform,
        colorinterp=(
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
        ),
        tags={"date": TARGET_DATE, "type": "true_colour"},
    )

    # -----------------------------------------------------
    # 2. SENTINEL-2 FALSE COLOUR
    # -----------------------------------------------------

    false_colour = [
        get_frame(s2, band, s2_idx).astype(np.int16)
        for band in ["B08", "B04", "B03"]
    ]

    write_geotiff(
        "02_s2_false_colour_2019-02-28.tif",
        false_colour,
        ["Red - B08 NIR", "Green - B04", "Blue - B03"],
        s2_crs,
        s2_transform,
        colorinterp=(
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
        ),
        tags={"date": TARGET_DATE, "type": "false_colour"},
    )

    # -----------------------------------------------------
    # 3. SENTINEL-2 SCL
    # -----------------------------------------------------

    scl = get_frame(s2, "SCL", s2_idx).astype(np.int16)

    write_geotiff(
        "03_s2_scl_2019-02-28.tif",
        [scl],
        ["SCL categorical classes"],
        s2_crs,
        s2_transform,
        tags={"date": TARGET_DATE, "type": "categorical"},
    )

    # -----------------------------------------------------
    # 4. SENTINEL-1 VV AND VH
    # -----------------------------------------------------

    vv = get_frame(s1, "VV", s1_idx).astype(np.float32)
    vh = get_frame(s1, "VH", s1_idx).astype(np.float32)

    write_geotiff(
        "04_s1_vv_2019-02-28.tif",
        [vv],
        ["VV backscatter dB"],
        s1_crs,
        s1_transform,
        tags={"date": TARGET_DATE, "units": "dB"},
    )

    write_geotiff(
        "05_s1_vh_2019-02-28.tif",
        [vh],
        ["VH backscatter dB"],
        s1_crs,
        s1_transform,
        tags={"date": TARGET_DATE, "units": "dB"},
    )

    # -----------------------------------------------------
    # 5. STATIC DEM AND MASK
    # -----------------------------------------------------

    dem = get_frame(s1, "DEM", 0).astype(np.int16)
    mask = (get_frame(s1, "MASK", 0) > 0).astype(np.uint8)

    write_geotiff(
        "06_dem_static.tif",
        [dem],
        ["DEM"],
        s1_crs,
        s1_transform,
        tags={"type": "static_elevation"},
    )

    write_geotiff(
        "07_mask_static.tif",
        [mask],
        ["Binary landslide inventory mask"],
        s1_crs,
        s1_transform,
        tags={"type": "binary_reference_mask"},
    )

print("\nAll QGIS visualisation exports completed.")
