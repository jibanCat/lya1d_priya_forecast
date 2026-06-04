"""Strip a KODIAQ GP basedir down to the files needed at inference time.

Usage:
    python scripts/prep_kodiaq_gp.py \
        --source /nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48 \
        --dest data/kodiaq_gp

Drops alt-cut HDF5 variants, leave-one-out diagnostics, the temperature
emulator, and the kims_* test subdir. Keeps all 13 z-bin pickles so the
forecast can switch redshift without a re-strip.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

KEEP_TOP_LEVEL = {
    "emulator_params.json",
    "mf_emulator_flux_vectors_tau1000000.hdf5",
}

KEEP_HIRES = {
    "emulator_params.json",
    "mf_emulator_flux_vectors_tau1000000.hdf5",
}


def _copy_file(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return src.stat().st_size


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--dest", required=True, type=Path)
    p.add_argument(
        "--overwrite", action="store_true",
        help="Wipe --dest before writing (default: refuse if non-empty).",
    )
    args = p.parse_args()

    src: Path = args.source
    dst: Path = args.dest
    if not src.is_dir():
        raise SystemExit(f"--source not a directory: {src}")
    if dst.exists() and any(dst.iterdir()) and not args.overwrite:
        raise SystemExit(f"--dest exists and is non-empty: {dst}. Pass --overwrite.")
    if args.overwrite and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    total = 0
    for name in KEEP_TOP_LEVEL:
        s = src / name
        if not s.exists():
            raise SystemExit(f"Missing required file in source: {s}")
        total += _copy_file(s, dst / name)

    train_src = src / "trained_mf"
    train_dst = dst / "trained_mf"
    train_dst.mkdir(parents=True, exist_ok=True)
    for entry in sorted(train_src.iterdir()):
        if not entry.name.startswith("zbin"):
            continue
        total += _copy_file(entry, train_dst / entry.name)

    hires_src = src / "hires"
    hires_dst = dst / "hires"
    hires_dst.mkdir(parents=True, exist_ok=True)
    for name in KEEP_HIRES:
        s = hires_src / name
        if not s.exists():
            raise SystemExit(f"Missing required hires file: {s}")
        total += _copy_file(s, hires_dst / name)

    readme = dst / "README.md"
    readme.write_text(
        f"# kodiaq_gp/ — stripped GP basedir\n\n"
        f"Source: `{src}`\n\n"
        f"Stripped via `scripts/prep_kodiaq_gp.py`. Drops alt-cut HDF5 "
        f"variants, leave-one-out diagnostics, temperature emulator files, "
        f"and `kims_*/`. Keeps all 13 trained z-bins so the forecast can "
        f"switch redshift without a re-strip.\n\n"
        f"To refresh:\n\n"
        f"```bash\n"
        f"python scripts/prep_kodiaq_gp.py --source <SRC> --dest data/kodiaq_gp --overwrite\n"
        f"```\n",
        encoding="utf-8",
    )

    print(f"Wrote stripped GP basedir to {dst} ({total / 1024 / 1024:.1f} MB).")


if __name__ == "__main__":
    main()
