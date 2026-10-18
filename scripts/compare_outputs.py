#!/usr/bin/env python3
"""Compare QBiC-SELEX result files (or directories of them) with a float tolerance.

Text columns must match exactly; numeric columns must agree within --rtol/--atol,
with NaN in the same places. Used to check that code changes keep outputs
consistent with the golden baselines.

Usage:
    python scripts/compare_outputs.py BASELINE CANDIDATE [--rtol 1e-5] [--atol 1e-8]
        [--ignore-columns status]

BASELINE and CANDIDATE are both CSV files or both directories (matched by
relative path). Exits non-zero if anything differs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compare_frames(base: pd.DataFrame, cand: pd.DataFrame, rtol: float, atol: float,
                   ignore: set) -> list:
    problems = []
    base_cols = [c for c in base.columns if c not in ignore]
    cand_cols = [c for c in cand.columns if c not in ignore]
    if base_cols != cand_cols:
        problems.append(f"columns differ: {base_cols} vs {cand_cols}")
        return problems
    if len(base) != len(cand):
        problems.append(f"row count differs: {len(base)} vs {len(cand)}")
        return problems

    for col in base_cols:
        b, c = base[col], cand[col]
        if pd.api.types.is_numeric_dtype(b) and pd.api.types.is_numeric_dtype(c):
            b_arr, c_arr = b.to_numpy(dtype=float), c.to_numpy(dtype=float)
            nan_mismatch = np.isnan(b_arr) != np.isnan(c_arr)
            if nan_mismatch.any():
                problems.append(f"{col}: NaN mismatch in {int(nan_mismatch.sum())} rows "
                                f"(first row {int(np.flatnonzero(nan_mismatch)[0])})")
                continue
            ok = np.isclose(b_arr, c_arr, rtol=rtol, atol=atol, equal_nan=True)
            if not ok.all():
                bad = np.flatnonzero(~ok)
                denom = np.maximum(np.abs(b_arr[bad]), atol)
                max_rel = float(np.max(np.abs(b_arr[bad] - c_arr[bad]) / denom))
                problems.append(f"{col}: {len(bad)} values outside tolerance "
                                f"(max rel diff {max_rel:.3g}, first row {int(bad[0])}: "
                                f"{b_arr[bad[0]]!r} vs {c_arr[bad[0]]!r})")
        else:
            b_str = b.astype(str).where(b.notna(), "<NA>")
            c_str = c.astype(str).where(c.notna(), "<NA>")
            diff = (b_str != c_str).to_numpy()
            if diff.any():
                i = int(np.flatnonzero(diff)[0])
                problems.append(f"{col}: {int(diff.sum())} rows differ "
                                f"(first row {i}: {b_str.iloc[i]!r} vs {c_str.iloc[i]!r})")
    return problems


def compare_files(base_path: Path, cand_path: Path, rtol: float, atol: float, ignore: set) -> list:
    if not cand_path.exists():
        return ["missing in candidate"]
    base = pd.read_csv(base_path)
    cand = pd.read_csv(cand_path)
    return compare_frames(base, cand, rtol, atol, ignore)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--ignore-columns", nargs="*", default=[],
                        help="Columns to leave out of the comparison (e.g. new columns)")
    args = parser.parse_args()

    base, cand = Path(args.baseline), Path(args.candidate)
    ignore = set(args.ignore_columns)
    if base.is_dir():
        pairs = [(p, cand / p.relative_to(base)) for p in sorted(base.rglob("*.csv"))]
    else:
        pairs = [(base, cand)]

    n_bad = 0
    for b, c in pairs:
        problems = compare_files(b, c, args.rtol, args.atol, ignore)
        label = b.relative_to(base) if base.is_dir() else b.name
        if problems:
            n_bad += 1
            print(f"DIFF  {label}")
            for p in problems:
                print(f"        {p}")
        else:
            print(f"OK    {label}")

    print(f"\n{len(pairs) - n_bad}/{len(pairs)} files match")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
