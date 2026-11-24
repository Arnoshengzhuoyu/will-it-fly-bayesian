"""Download the OpenADMET/Octant data release (Zenodo record 21450795) into data/raw/ and check MD5s."""
import hashlib
import urllib.request
from pathlib import Path

RECORD = "21450795"
FILES = {
    "will_it_fly_in_mass_spec.tsv": "1faf98dd8fa3bd7fc4b12149d56303e1",
    "reactivity.tsv": "a6627aeaf900dba2825c484deac95f03",
    "reactivity_wells.tsv": "a7d12cb3a7c156000c31150f618bd0bb",
    "README.md": "2ae66359ba460dcaa692e3dbcd97bbed",
}
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        dest = RAW / name
        if dest.exists() and md5(dest) == expected:
            print(f"ok (cached)  {name}")
            continue
        url = f"https://zenodo.org/api/records/{RECORD}/files/{name}/content"
        urllib.request.urlretrieve(url, dest)
        got = md5(dest)
        if got != expected:
            raise RuntimeError(f"MD5 mismatch for {name}: {got} != {expected}")
        print(f"downloaded   {name}")


if __name__ == "__main__":
    main()
