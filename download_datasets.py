"""
download_datasets.py
Downloads ADBench Classical .npz datasets from GitHub into ./data/.

Datasets live at:
  https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/{N}_{name}.npz

Usage:
    python download_datasets.py            # downloads Tier 1-3 (E1-E4)
    python download_datasets.py --all      # also downloads Tier 4 (large scale)

No external dependencies — stdlib only.
"""

import sys
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

GITHUB_RAW = (
    "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/"
)

# Maps our canonical dataset name → ADBench Classical filename.
# Substitutions (no direct match in ADBench Classical):
#   arrhythmia → vertebral   (Tier 1, N≈267, similar small size)
#   kddcup99   → smtp        (Tier 3, N≈95K, network intrusion — same domain)
#   forestcov  → skin        (Tier 3, N≈245K, large tabular)
DATASET_MAP = {
    # Tier 1 (N < 1000)
    "breastw":    "4_breastw.npz",
    "glass":      "14_glass.npz",
    "ionosphere": "18_Ionosphere.npz",
    "lympho":     "21_Lymphography.npz",
    "vertebral":  "39_vertebral.npz",      # substituting arrhythmia

    # Tier 2 (1000 ≤ N ≤ 10000)
    "annthyroid": "2_annthyroid.npz",
    "letter":     "20_letter.npz",
    "musk":       "25_musk.npz",
    "optdigits":  "26_optdigits.npz",
    "pendigits":  "28_pendigits.npz",
    "satellite":  "30_satellite.npz",
    "thyroid":    "38_thyroid.npz",

    # Tier 3 (N > 10000)
    "covertype":  "10_cover.npz",
    "shuttle":    "32_shuttle.npz",
    "smtp":       "34_smtp.npz",           # substituting kddcup99
    "skin":       "33_skin.npz",           # substituting forestcov

    # Tier 4 (N > 100000, for E5)
    "http":       "16_http.npz",
    "mammography":"23_mammography.npz",
}

# Tier 1-3 used for E1-E4
TIER123 = [
    "breastw", "glass", "ionosphere", "lympho", "vertebral",
    "annthyroid", "letter", "musk", "optdigits", "pendigits", "satellite", "thyroid",
    "covertype", "shuttle", "smtp", "skin",
]

# Tier 4 used for E5
TIER4 = ["http", "mammography"]


def download_one(our_name: str, adbench_fname: str, dest_dir: Path) -> bool:
    dest = dest_dir / f"{our_name}.npz"
    if dest.exists():
        print(f"  already exists: {our_name}.npz ({dest.stat().st_size // 1024} KB)")
        return True

    url = GITHUB_RAW + adbench_fname
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        if len(data) < 200:
            print(f"  FAIL {our_name}: response too small ({len(data)} bytes)")
            return False
        dest.write_bytes(data)
        print(f"  OK   {our_name}.npz  ← {adbench_fname}  ({len(data) // 1024} KB)")
        return True
    except Exception as e:
        print(f"  FAIL {our_name}: {e}")
        return False


def main():
    download_all = "--all" in sys.argv
    targets = TIER123 + (TIER4 if download_all else [])

    DATA_DIR.mkdir(exist_ok=True)
    print(f"Downloading {len(targets)} datasets → {DATA_DIR}/\n")

    ok, failed = [], []
    for name in targets:
        fname = DATASET_MAP.get(name)
        if fname is None:
            print(f"  SKIP {name}: not in DATASET_MAP")
            failed.append(name)
            continue
        if download_one(name, fname, DATA_DIR):
            ok.append(name)
        else:
            failed.append(name)
        time.sleep(0.4)

    print(f"\nResult: {len(ok)} OK, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")
        print(f"Manual download: {GITHUB_RAW}")


if __name__ == "__main__":
    main()
