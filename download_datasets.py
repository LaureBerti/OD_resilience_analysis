"""
download_datasets.py
Downloads the 16 core ADBench Classical .npz datasets from GitHub into ./data/ and checks each file
against the SHA-256 of the copy used for the paper (ADBench may change its files on the main branch).

Datasets live at:
  https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/{N}_{name}.npz

Usage:
    python download_datasets.py

No external dependencies — stdlib only.
"""

import hashlib
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

GITHUB_RAW = (
    "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/"
)

DATASET_MAP = {
    "breastw":    "4_breastw.npz",
    "glass":      "14_glass.npz",
    "ionosphere": "18_Ionosphere.npz",
    "lympho":     "21_Lymphography.npz",
    "vertebral":  "39_vertebral.npz",

    "annthyroid": "2_annthyroid.npz",
    "letter":     "20_letter.npz",
    "musk":       "25_musk.npz",
    "optdigits":  "26_optdigits.npz",
    "pendigits":  "28_pendigits.npz",
    "satellite":  "30_satellite.npz",
    "thyroid":    "38_thyroid.npz",

    "covertype":  "10_cover.npz",
    "shuttle":    "32_shuttle.npz",
    "smtp":       "34_smtp.npz",
    "skin":       "33_skin.npz",
}

SHA256 = {
    "breastw": "4f37ce07dbc25ddad8178fdc6d664ab46d2d0000ee28c2787c3c7ff3c03ef9f6",
    "glass": "aa7a8db1d475ba01a10488af454bc15220a3d5951b010b03b7020567d8152cdc",
    "ionosphere": "bca46cb0cfd1bf9ddef6cde5ad2215d54fb0189508d0d4c4eb98e788cad69001",
    "lympho": "dad977b04f4ea65b78ccd0e9f76b45175bad605165b37e63f3cc177845e67037",
    "vertebral": "0816286f92b96fe6867664a41a031ac7228dcfbf1a5feefe50ca3883b2e9073d",
    "annthyroid": "e738f7c58a8740d2c4aac4c71ff49ce5e1d1aa16f44178135fc9c09df05fad1b",
    "letter": "f986d3da80cf1fa2aea83a5ebe73b7d4800e0d4c01fdd1ae074b58d20f10d039",
    "musk": "2ed3f4392197b76a1e59116ff880c7720a4b026cf4842073a53b56fdea265017",
    "optdigits": "60de17a375a278ba220a7849727d998aa02b5e2f9fd17732a004bba28de4656e",
    "pendigits": "c4936b89dcdad59d303020e492f763a0f54170080d787c77497609acbb9c2d79",
    "satellite": "1f048f5686f1b8c16a227fdfa3c81a62cd9a92715714a6b24c69b1c5ad96b46e",
    "thyroid": "8de161b3f235b7647bb0b39a836d983ff61d67bf234edac461da2d71e889cb87",
    "covertype": "945a097c3542ae81a551bf1c06dcad78a755d94f84fcaf1d60687e42d1a656df",
    "shuttle": "26900b0f6be45d9ddb7a402c9b2f956ceda42d23a8ca4cf4406b8bc1352823d1",
    "smtp": "d2569fc9bc12115419e3f5b37d19921e10e6c26e5ad1e61064cfb0fbc402dfdd",
    "skin": "d02aa01ffc1b6bf2b8b1ac250c90e8a8eacd0cefc962940ed5cc4bf5deac9c95",
}


def check(our_name: str, data: bytes) -> bool:
    if hashlib.sha256(data).hexdigest() == SHA256[our_name]:
        return True
    print(f"  WARN {our_name}.npz differs from the copy used for the paper (SHA-256 mismatch)")
    return False


def download_one(our_name: str, adbench_fname: str, dest_dir: Path) -> bool:
    dest = dest_dir / f"{our_name}.npz"
    if dest.exists():
        print(f"  already exists: {our_name}.npz ({dest.stat().st_size // 1024} KB)")
        return check(our_name, dest.read_bytes())

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
        return check(our_name, data)
    except Exception as e:
        print(f"  FAIL {our_name}: {e}")
        return False


def main():
    targets = list(SHA256)

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
