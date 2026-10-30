import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

UPSTREAM = Path(r"D:\Research\Sen12Landslides")
TARGET = "chimanimani_s2_1004.nc"

ARCHIVE_REPO_PATH = "data_harmonized/s2/s2_part01.tar.gz"

OUTPUT = (
    UPSTREAM
    / "data"
    / "data_harmonized"
    / "s2"
    / TARGET
)

print("Downloading S2 archive part 01...")
archive = Path(
    hf_hub_download(
        repo_id="paulhoehn/Sen12Landslides",
        repo_type="dataset",
        filename=ARCHIVE_REPO_PATH,
        local_dir=str(UPSTREAM / "data_zipped"),
    )
)

print("Downloaded:", archive)
print("Archive size:", round(archive.stat().st_size / 1e9, 3), "GB")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

found = False

try:
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.isfile() and Path(member.name).name == TARGET:
                print("\nMATCH FOUND!")
                print("Archive member:", member.name)

                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError("Could not read matched member")

                with source, OUTPUT.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)

                found = True
                break

    if found:
        print("\nExtracted:", OUTPUT)
        print("File size:", OUTPUT.stat().st_size, "bytes")
    else:
        print(f"\n{TARGET} was NOT found in part 01.")

finally:
    if archive.exists():
        archive.unlink()
        print("Temporary archive removed:", archive)

print("\nRESULT:", "SUCCESS" if found else "NOT FOUND IN PART 01")
