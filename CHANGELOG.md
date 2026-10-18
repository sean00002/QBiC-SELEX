# Changelog

## 1.0.0 (unreleased)

First packaged release, accompanying the QBiC-SELEX paper.

### Added
- `pip install qbic-selex` (GPU: `pip install "qbic-selex[gpu]"`) with a `qbic-selex` command:
  `predict`, `extract` (context sequences only) and `model-map`.
- `-m` and `-c` accept directories; list files may give bare model names, resolved with `--model-dir`.
- `--device auto|cpu|gpu`. `auto` checks that the GPU actually works and otherwise falls back to the CPU.
- `qbic_status` output column: why a row has NA (`ref_mismatch`, `chrom_not_found`, `out_of_bounds`,
  `invalid_allele`, `missing_value`, `contains_N`, `invalid_sequence`, `shorter_than_k`).
- `--allow-failures`; gzip-compressed input.
- Final paper tables S3–S6 in `mapping_files/`; collection lists in `model_lists/`
  (997 primary, 243 secondary); the model IDs used on the IGVF portal in `model_maps/`.
- MIT license, tests (unit, golden outputs, GPU), CI.

### Fixed
- A failed or timed-out statistics step no longer silently drops the `z_score`/`p_value` columns.
  Previously `--pval` then wrote unfiltered rows, and batch CSVs got misaligned columns.
  The model now fails, and the run exits 1. The 600 s statistics timeout is removed.
- `--wildcard` is applied before the statistics too. N-containing rows used to get an effect but NaN z/p.
- Lowercase input sequences are scored correctly (they used to score 0).
- Rows that cannot be scored are kept with NA and a reason, instead of being dropped.
- Positions too close to a contig end are reported as `out_of_bounds` instead of a reference mismatch.
- The k-mer count matrix is int16, so very long or repetitive contexts cannot overflow.
- The covariance size is checked against the model. One covariance file for several models is an error.
- Numeric chromosome names (`1`) no longer crash coordinate extraction.
- Error reports are written next to the output instead of the working directory.
- Model files are read with a restricted unpickler.

### Changed
- `qbic_predict.py` / `qbic_predict_faster.py` / `generate_model_map.py` are thin wrappers around the
  package and accept the old options. `--use-gpu`/`--use-cpu` map to `--device`; `--n-jobs` is
  accepted but has no effect.
- Statistics use only the batched implementation (formerly `qbic_predict_faster.py`), which matches
  the original per-variant computation up to float32 rounding.
- GPU support needs only CuPy (`cudf`, `cuml` and `torch` are no longer used). `env.yml` is removed.
- Unused training-side helpers in `util_scripts/sequence_utils.py` are removed.
- Missing models or covariance files are an error before the run starts. They used to be skipped
  with a warning.
