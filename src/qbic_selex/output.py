"""Output formatting: the compact --short-output format and model-ID maps."""

import csv
from pathlib import Path
from collections.abc import Iterable

import pandas as pd

from .models import model_name_from_path

# Canonical short-output column -> accepted input spellings
COLUMN_ALIASES = {
    "chrom": ["chrom", "Chromosome", "chr"],
    "pos": ["pos", "Position"],
    "ref": ["ref", "Ref"],
    "alt": ["alt", "Alt"],
}


def round_sig(series: pd.Series, sig: int = 4) -> pd.Series:
    """Round to ``sig`` significant digits, keeping NaN and zero as they are."""
    def _round(x):
        if pd.isna(x) or x == 0:
            return x
        return float(f"{x:.{sig - 1}e}")
    return series.map(_round)


def apply_short_output(df: pd.DataFrame, model_ids: dict[str, int], with_stats: bool) -> pd.DataFrame:
    """Compact output: ``chrom,pos,ref,alt,model,predicted_effect[,z_score,p_value]``.

    Model names become integer IDs and numbers are rounded to 4 significant digits.
    """
    result = df.copy()
    rename = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical not in result.columns:
            for alias in aliases:
                if alias in result.columns:
                    rename[alias] = canonical
                    break
    result = result.rename(columns=rename)
    result["model"] = result["model"].map(model_ids)
    result["predicted_effect"] = round_sig(result["predicted_effect"])
    cols = ["chrom", "pos", "ref", "alt", "model", "predicted_effect"]
    if with_stats:
        result["z_score"] = round_sig(result["z_score"])
        result["p_value"] = round_sig(result["p_value"])
        cols += ["z_score", "p_value"]
    return result[[c for c in cols if c in result.columns]]


def load_model_map(path) -> dict[str, int]:
    """Read a ``model_id,model_name[,...]`` CSV into {model_name: model_id}."""
    with open(path, newline="") as f:
        return {row["model_name"]: int(row["model_id"]) for row in csv.DictReader(f)}


def write_model_map(model_entries: Iterable[str], path) -> int:
    """Assign IDs 0..n-1 to the unique model names, sorted alphabetically. Returns n."""
    names = sorted({model_name_from_path(e) for e in model_entries})
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_id", "model_name"])
        for i, name in enumerate(names):
            writer.writerow([i, name])
    return len(names)


def error_report_dir(output: str, output_dir_mode: bool) -> Path:
    """Where to write the error report: the output directory, or next to the output file."""
    out = Path(output)
    return out if output_dir_mode else (out.parent if str(out.parent) else Path("."))
