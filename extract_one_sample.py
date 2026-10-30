import tarfile
from pathlib import Path
import xarray as xr

UPSTREAM = Path(r"D:\Research\Sen12Landslides")

ARCHIVE = (
    UPSTREAM
    / "data_zipped"
    / "data_harmonized"
    / "s1asc"
    / "s1asc_part01.tar.gz"
)

TARGET = "chimanimani_s1asc_1004.nc"

OUTPUT = (
    UPSTREAM
    / "data"
    / "data_harmonized"
    / "s1asc"
    / TARGET
)

if not ARCHIVE.exists():
    raise FileNotFoundError(f"Archive not found: {ARCHIVE}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT.exists():
    print(f"Already extracted: {OUTPUT}")
else:
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        matches = [
            m for m in tar.getmembers()
            if m.isfile() and Path(m.name).name == TARGET
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one archive member for {TARGET}; "
                f"found {len(matches)}"
            )

        member = matches[0]
        print("Archive member:", member.name)

        source = tar.extractfile(member)
        if source is None:
            raise RuntimeError("Could not read archive member")

        with source, OUTPUT.open("wb") as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)

print("\nPersistent sample:", OUTPUT)
print("File size:", OUTPUT.stat().st_size, "bytes")

with xr.open_dataset(OUTPUT, engine="netcdf4") as ds:
    print("\nNetCDF validation")
    print("Dimensions:", dict(ds.sizes))
    print("Variables:", list(ds.data_vars))
    print("Time observations:", ds.sizes.get("time"))
    print("CRS:", ds.attrs.get("crs"))
