"""Command-line interface: ``qbic-selex predict | extract | model-map``."""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from . import __version__
from . import sequences as seqs
from .models import ModelInputError, model_name_from_path, resolve_covariances, resolve_models
from .output import error_report_dir, load_model_map, write_model_map
from .predict import RunConfig, run_models, write_error_report
from .stats import gpu_available, gpu_name

log = logging.getLogger("qbic_selex")


def read_table(path: str) -> pd.DataFrame:
    """CSV, or tab-separated for .tsv/.txt; optionally gzip-compressed."""
    stem = path[:-3] if path.endswith(".gz") else path
    sep = "\t" if stem.endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep)


def load_variants_with_sequences(args) -> pd.DataFrame:
    log.info("Reading variants from %s", args.variants)
    variants = read_table(args.variants)
    log.info("Loaded %d rows", len(variants))
    if all(c in variants.columns for c in seqs.SEQUENCE_COLUMNS):
        log.info("Using ref_sequence/alt_sequence from the input")
        return variants
    log.info("Extracting +/-%d bp context from %s (%s coordinates)", args.context_length,
             args.genome, "0-based" if args.zero_based else "1-based")
    return seqs.extract_sequences(variants, args.genome, context_length=args.context_length,
                                  zero_based=args.zero_based)


# ----------------------------------------------------------------------------
# predict
# ----------------------------------------------------------------------------

def add_predict_args(p: argparse.ArgumentParser) -> None:
    io = p.add_argument_group("input/output")
    io.add_argument("-v", "--variants", required=True,
                    help="Variants (CSV, or TSV for .tsv/.txt; .gz ok) with columns chrom,pos,ref,alt "
                         "or ref_sequence,alt_sequence")
    io.add_argument("-m", "--models", "--model", dest="models", required=True,
                    help="A .weights.qbic model, a directory of models, or a text file listing "
                         "model paths (or bare model names, with --model-dir)")
    io.add_argument("--model-dir", help="Directory searched for model names given in a list file "
                                        "(e.g. the unpacked Zenodo model collection)")
    io.add_argument("-o", "--output", required=True, help="Output CSV (or directory with --output-dir)")
    io.add_argument("--output-dir", action="store_true", help="Write one CSV per model into --output")

    st = p.add_argument_group("statistics")
    st.add_argument("--compute-stats", "--stats", dest="compute_stats", action="store_true",
                    help="Add z_score and p_value columns (requires --covs)")
    st.add_argument("-c", "--covs", "--cov-file", dest="covs",
                    help="A covariance .npy (single model), a directory of *.cov_*.npy files, or a "
                         "text file listing covariance paths (or model_path,cov_path pairs)")
    st.add_argument("--pval", type=float, help="Only output rows with p_value <= PVAL "
                                               "(requires --compute-stats)")
    st.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                    help="Where to compute statistics (default: GPU if available)")
    st.add_argument("--use-gpu", action="store_const", const="auto", dest="device",
                    help=argparse.SUPPRESS)
    st.add_argument("--use-cpu", action="store_const", const="cpu", dest="device",
                    help=argparse.SUPPRESS)
    st.add_argument("--n-jobs", type=int, help=argparse.SUPPRESS)

    sq = p.add_argument_group("sequences")
    sq.add_argument("-g", "--genome", default="hg38",
                    help="Reference FASTA path, or a name (hg38, hg19) looked up as <name>.fa in "
                         "$QBIC_SELEX_GENOME_DIR or ./genome (default: hg38)")
    sq.add_argument("--context-length", type=int, default=10,
                    help="Reference bases on each side of the variant (default: 10)")
    sq.add_argument("--zero-based", action="store_true",
                    help="Positions are 0-based (BED-style); default is 1-based (VCF-style)")
    sq.add_argument("--wildcard", choices=["A", "C", "G", "T"],
                    help="Replace N with this base; by default rows containing N get NA")

    fm = p.add_argument_group("output format")
    fm.add_argument("--short-output", action="store_true",
                    help="Compact output: chrom,pos,ref,alt,model ID,predicted_effect[,z_score,p_value] "
                         "rounded to 4 significant digits (requires --model-map)")
    fm.add_argument("--model-map", help="model_id,model_name CSV for --short-output "
                                        "(from `qbic-selex model-map`)")
    fm.add_argument("--allow-failures", action="store_true",
                    help="Exit 0 even if some models fail (they are listed in the error report)")


def run_predict(args) -> int:
    start = time.time()
    if args.compute_stats and not args.covs:
        raise SystemExit("error: --compute-stats requires --covs")
    if args.pval is not None and not args.compute_stats:
        raise SystemExit("error: --pval requires --compute-stats")
    if args.short_output != bool(args.model_map):
        raise SystemExit("error: --short-output and --model-map must be used together")
    if args.n_jobs is not None:
        log.warning("--n-jobs has no effect: statistics use batched BLAS/GPU operations")
    out = Path(args.output)
    if args.output_dir and out.exists() and not out.is_dir():
        raise SystemExit(f"error: --output {out} exists and is not a directory")

    model_paths = resolve_models(args.models, args.model_dir)
    names = [model_name_from_path(p) for p in model_paths]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ModelInputError(f"model names appear more than once: {', '.join(dupes[:5])}")
    log.info("%d model(s) to run", len(model_paths))

    cov_paths = resolve_covariances(args.covs, names) if args.compute_stats else None

    model_ids = None
    if args.short_output:
        model_ids = load_model_map(args.model_map)
        unmapped = [n for n in names if n not in model_ids]
        if unmapped:
            raise ModelInputError(f"{len(unmapped)} model(s) are not in {args.model_map}: "
                                  f"{', '.join(unmapped[:5])}")

    use_gpu = False
    if args.compute_stats:
        has_gpu = gpu_available()
        if args.device == "gpu" and not has_gpu:
            raise SystemExit("error: --device gpu but no CUDA GPU is usable (is cupy installed? "
                             "pip install 'qbic-selex[gpu]')")
        use_gpu = args.device == "gpu" or (args.device == "auto" and has_gpu)
        log.info("Statistics on %s", f"GPU: {gpu_name()}" if use_gpu else "CPU")

    variants = load_variants_with_sequences(args)
    cfg = RunConfig(with_stats=args.compute_stats, use_gpu=use_gpu, pval=args.pval,
                    short_output=args.short_output, model_ids=model_ids,
                    output_dir_mode=args.output_dir, wildcard=args.wildcard)
    summary = run_models(variants, model_paths, args.output, cfg, cov_paths)

    if args.pval is not None:
        log.info("p-value filter (p <= %g): kept %d of %d rows", args.pval,
                 summary.rows_written, summary.rows_before_filter)
    log.info("Wrote %d rows for %d model(s) to %s in %.1f s", summary.rows_written,
             len(summary.models_ok), args.output, time.time() - start)
    report = write_error_report(summary, error_report_dir(args.output, args.output_dir))
    if summary.models_failed:
        log.error("%d of %d model(s) failed; details in %s", len(summary.models_failed),
                  len(model_paths), report)
        if not args.allow_failures:
            return 1
    return 0


# ----------------------------------------------------------------------------
# extract / model-map
# ----------------------------------------------------------------------------

def add_extract_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-v", "--variants", required=True, help="CSV/TSV with chrom,pos,ref,alt")
    p.add_argument("-o", "--output", required=True, help="Output CSV")
    p.add_argument("-g", "--genome", default="hg38", help="Reference FASTA path or name (default: hg38)")
    p.add_argument("--context-length", type=int, default=10)
    p.add_argument("--zero-based", action="store_true")


def run_extract(args) -> int:
    seqs_df = seqs.extract_sequences(read_table(args.variants), args.genome,
                                     context_length=args.context_length, zero_based=args.zero_based)
    seqs_df.to_csv(args.output, index=False)
    log.info("Wrote %d rows to %s", len(seqs_df), args.output)
    return 0


def add_model_map_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-m", "--models", required=True, help="Text file listing model paths or names")
    p.add_argument("-o", "--output", required=True, help="Output model_id,model_name CSV")


def run_model_map(args) -> int:
    with open(args.models) as f:
        entries = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not entries:
        raise SystemExit(f"error: no models listed in {args.models}")
    n = write_model_map(entries, args.output)
    log.info("Generated mapping for %d models: %s", n, args.output)
    return 0


# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qbic-selex",
        description="QBiC-SELEX: predict variant effects on transcription factor binding.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--quiet", action="store_true", help="Only warnings and errors")
    verbosity.add_argument("--verbose", action="store_true", help="Debug output")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, add, run, help_ in [
        ("predict", add_predict_args, run_predict, "Predict variant effects (main command)"),
        ("extract", add_extract_args, run_extract, "Only extract ref/alt context sequences"),
        ("model-map", add_model_map_args, run_model_map, "Make a model ID map for --short-output"),
    ]:
        p = sub.add_parser(name, help=help_, description=help_)
        add(p)
        p.set_defaults(func=run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)
    log.info("QBiC-SELEX %s", __version__)
    try:
        return args.func(args)
    except (ModelInputError, seqs.GenomeNotFoundError, ValueError, FileNotFoundError) as e:
        log.error("%s", e)
        return 2
    except KeyboardInterrupt:
        log.error("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
