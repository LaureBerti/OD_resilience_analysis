"""download_extra_datasets.py — fetch the 13 extra E2 datasets (ODDS / ADBench, as distributed with the
ADOD code, https://github.com/Qian-Lily/ADOD, pinned commit) into data/extra/.

Run: .venv/bin/python revision/code/download_extra_datasets.py
"""
from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

PROJECT = Path(__file__).resolve().parents[2]
FILES = ["cardio_ODDS", "vowels_ODDS", "Waveform_ADBench", "speech_ODDS", "PageBlocks_ADBench",
         "satimage-2_ODDS", "mnist_ODDS", "mammography_ODDS", "magic_gamma_ADBench", "campaign_ADBench",
         "celeba_ADBench", "fraud_ADBench", "http_ODDS"]


@dataclass
class Cfg:
    commit: str = "dc76889e7237da15fdc327a89f3c1f716586587c"
    base_url: str = "https://raw.githubusercontent.com/Qian-Lily/ADOD"
    out_dir: str = "data/extra"
    files: list = field(default_factory=lambda: list(FILES))


ConfigStore.instance().store(name="dl", node=Cfg)


@hydra.main(version_base=None, config_name="dl")
def main(c: Cfg) -> None:
    out = PROJECT / c.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for f in c.files:
        dest = out / f"{f}.npz"
        if dest.exists():
            print("have", dest.name); continue
        url = f"{c.base_url}/{c.commit}/datasets/{f}.npz"
        with urllib.request.urlopen(url, timeout=300) as r:
            dest.write_bytes(r.read())
        print("got ", dest.name)


if __name__ == "__main__":
    sys.argv += ["hydra.run.dir=.", "hydra.output_subdir=null"]
    main()
