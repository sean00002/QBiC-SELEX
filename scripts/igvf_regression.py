#!/usr/bin/env python3
"""Check that this version reproduces the QBiC-SELEX predictions submitted to the IGVF portal.

For each model, samples rows from the IGVF per-model result file
(results_by_model/model_<id>.csv, short-output format, p <= 1e-6), re-scores
those variants with the same settings as the IGVF run (--zero-based
--compute-stats --short-output, hg38, +/-10 bp), and compares the values at the
published 4-significant-digit precision.

Optionally (--input-variants) also samples variants from the IGVF input that are
*absent* from a model's results, and checks they score p > 1e-6 here too.

Example:
    python scripts/igvf_regression.py \\
        --results-dir igvf_prediction/results_by_model \\
        --model-map model_maps/igvf_primary_model_ids.csv \\
        --model-dir tf_models_primary_collection --cov-dir covs_primary \\
        --genome hg38.fa --models CTCF_FL_TG40NCTGTGG_YWL_jolma2025_0_3_7mer ... \\
        --input-variants Y2AVECombinedVariants_SNV_filtered.csv.gz
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from qbic_selex.cli import main as qbic_main

PVAL = 1e-6
KEY = ["chrom", "pos", "ref", "alt"]


def sample_input_pool(path, n: int, seed: int) -> pd.DataFrame:
    """A random sample of about n rows of the IGVF input variants (read once, in chunks)."""
    chunks = []
    for chunk in pd.read_csv(path, usecols=KEY, chunksize=2_000_000):
        chunks.append(chunk.sample(frac=0.01, random_state=seed + len(chunks)))
    pool = pd.concat(chunks, ignore_index=True)
    return pool.sample(n=min(n, len(pool)), random_state=seed)


def check_model(args, name: str, model_id: int, tmp: Path, pool) -> dict:
    published = pd.read_csv(args.results_dir / f"model_{model_id}.csv")
    sample = published.sample(n=min(args.n, len(published)), random_state=args.seed)
    variants = sample[KEY]
    if pool is not None:
        # input variants this model did not report as significant
        merged = pool.merge(published[KEY].drop_duplicates(), on=KEY, how="left", indicator=True)
        negatives = merged.loc[merged["_merge"] == "left_only", KEY]
        variants = pd.concat([variants, negatives], ignore_index=True)
    vfile = tmp / f"{name}.variants.csv"
    variants.to_csv(vfile, index=False)

    model_list = tmp / f"{name}.models.txt"
    model_list.write_text(name + "\n")
    # same flags as the IGVF run (sbatch_predict_gpu.sh), without the p-value filter
    out = tmp / f"{name}.short.csv"
    rc = qbic_main(["--quiet", "predict", "-v", str(vfile), "-m", str(model_list),
                    "--model-dir", str(args.model_dir), "-c", str(args.cov_dir), "--compute-stats",
                    "--zero-based", "--short-output", "--model-map", str(args.model_map),
                    "-g", str(args.genome), "--device", args.device, "-o", str(out)])
    if rc != 0:
        return {"model": name, "error": f"qbic-selex exited {rc}"}
    new = pd.read_csv(out).drop(columns="model")

    pos = sample.drop(columns="model").merge(new, on=KEY, how="left", suffixes=("_igvf", ""))
    result = {"model": name, "id": model_id, "n_published_checked": len(pos)}
    for col in ("predicted_effect", "z_score", "p_value"):
        a, b = pos[col + "_igvf"].to_numpy(float), pos[col].to_numpy(float)
        exact = np.isclose(a, b, rtol=0, atol=0) | (np.isnan(a) & np.isnan(b))
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-300)
        result[f"{col}_mismatch_4sig"] = int((~exact).sum())
        result[f"{col}_max_rel_diff"] = float(np.nanmax(rel)) if len(rel) else 0.0
    result["published_now_above_threshold"] = int((pos["p_value"] > PVAL).sum())

    if pool is not None:
        neg = negatives.merge(new, on=KEY, how="left")   # rows absent from `new` could not be scored
        result["n_negatives_checked"] = len(neg)
        result["negatives_now_significant"] = int((neg["p_value"] <= PVAL).sum())
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--model-map", type=Path, required=True, help="model_id,model_name used for the run")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--cov-dir", type=Path, required=True)
    p.add_argument("--genome", type=Path, required=True)
    p.add_argument("--models", nargs="+", required=True, help="Model names to check")
    p.add_argument("--input-variants", type=Path, help="The IGVF input variants (for the negative check)")
    p.add_argument("-n", type=int, default=5000, help="Published rows sampled per model")
    p.add_argument("--n-negative", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("-o", "--output", type=Path, help="Write the summary table here (CSV)")
    args = p.parse_args()

    ids = pd.read_csv(args.model_map).set_index("model_name")["model_id"]
    pool = sample_input_pool(args.input_variants, args.n_negative, args.seed) if args.input_variants else None
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in args.models:
            rows.append(check_model(args, name, int(ids[name]), Path(tmp), pool))
            print(pd.Series(rows[-1]).to_string(), "\n", flush=True)
    summary = pd.DataFrame(rows)
    if args.output:
        summary.to_csv(args.output, index=False)

    mismatch_cols = [c for c in summary.columns if c.endswith("_mismatch_4sig")]
    total_checked = summary["n_published_checked"].sum()
    total_mismatch = summary[mismatch_cols].to_numpy().sum()
    print(f"Checked {total_checked} published rows across {len(summary)} models; "
          f"{total_mismatch} value(s) differ at 4 significant digits")
    bad = (summary.get("error").notna().any() if "error" in summary else False) or \
        summary["published_now_above_threshold"].sum() > 0 or \
        summary.get("negatives_now_significant", pd.Series([0])).sum() > 0
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
