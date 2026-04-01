# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QBiC-SELEX predicts how genetic variants affect transcription factor (TF) binding using k-mer-based machine learning models trained on SELEX data. It supports 1023+ TF models and outputs binding scores, z-scores, and p-values.

## Running the Tool

The main entry point is `qbic_predict.py` (executable). Key usage patterns:

**Pre-extracted sequences as input:**
```bash
./qbic_predict.py -v example_input_sequences.csv \
  -m example_models/ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer.weights.qbic \
  -o results.csv
```

**Variant coordinates as input (requires genome FASTA in `genome/`):**
```bash
./qbic_predict.py -v example_input_variants.csv \
  -m example_models_list.txt -c example_covs_list.txt \
  --compute-stats -o output_dir/ --output-dir
```

**CPU parallel stats (when no GPU):**
```bash
./qbic_predict.py -v example_input_variants.csv \
  -m example_models_list.txt -c example_covs_list.txt \
  --compute-stats --use-cpu --n-jobs 8 -o results.csv
```

## Environment Setup

```bash
conda env create -f env.yml
conda activate qbic-selex
```

Key dependencies: pandas 1.5.3, numpy 1.26.4, scipy 1.12.0, pysam 0.21.0. GPU acceleration (cupy, cudf, cuml) is optional but strongly recommended for `--compute-stats`.

## Architecture

### Core Files
- **`qbic_predict.py`** — Main script (~910 lines). Handles CLI, two input modes, single/batch model processing, GPU/CPU fallback for statistics, and output formatting.
- **`util_scripts/sequence_utils.py`** — DNA sequence utilities: k-mer generation, sliding windows, reverse complement, frequency analysis.
- **`util_scripts/extract_seq.py`** — Genome sequence extraction from FASTA (via pysam) for variant coordinate input mode.

### Two Input Modes
1. **Variant coordinates** (`chrom,pos,ref,alt`): Requires a genome FASTA in `genome/` (hg38.fa or hg19.fa). Context of ±10bp extracted automatically (`--context-length` to change).
2. **Pre-extracted sequences** (`ref_sequence,alt_sequence`): Sequences must be the same length; alt has the variant substitution in place.

### Model and Covariance Files
- Models: `.weights.qbic` format — one file per TF, stored in `example_models/` or a full collection
- Covariance matrices: `.cov_8192.npy` — required only for `--compute-stats`
- Batch mode: pass a `.txt` list file to `-m` and `-c` instead of a single file

### Statistics Computation
- GPU path uses cupy/cudf/cuml (auto-detected); CPU path uses joblib parallelism (`--n-jobs`)
- `--compute-stats` adds z-score and p-value columns to output
- `--use-cpu` forces CPU even if GPU is available

### Output
- Single CSV (default): all models concatenated with a `model` column
- Directory output (`--output-dir`): one CSV per model, named after the model

### N-handling
- Sequences with 'N' return NA by default
- `--wildcard A` (or C/G/T) replaces 'N' with the specified nucleotide

## Mapping Files
- `mapping_files/TableS5_model_to_tfs_mapping.csv` — which TFs each model targets
- `mapping_files/TableS6_tf_to_models_mapping.csv` — which models cover each TF, with validation metrics

## Error Logging
Errors write to `qbic_error_report_YYYYMMDD_HHMMSS.log` in the working directory.
