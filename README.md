# QBiC-SELEX

**Quantitative, bias-corrected prediction of variant effects on transcription factor binding.**

QBiC-SELEX predicts how non-coding genetic variants change the binding of
transcription factors (TFs). Its models are built from 4,362 HT-SELEX
experiments covering 997 human TFs, and correct for the systematic sequence
biases of SELEX assays. Each model is an interpretable k-mer regression, so
every variant gets a quantitative predicted effect and, optionally, a z-score and
p-value.

- **Primary collection:** 997 models, one per human TF, each cross-experiment validated (Table S3).
- **Secondary collection:** 243 models for 203 TFs, curated against SNP-SELEX data (Table S4).
- **Mapping files:** model → TF and TF → model, covering 1,111 TFs via CIS-BP (Tables S5 and S6).

## Install

```bash
pip install qbic-selex            # CPU
pip install "qbic-selex[gpu]"     # + CuPy with CUDA 12 libraries, for fast statistics on NVIDIA GPUs
```

Python ≥ 3.10. This installs the `qbic-selex` command, with two small example
models and example inputs for the [quick start](#quick-start). The mapping
tables and model lists are in this GitHub repository; clone it for those, or to
change the code (`pip install -e .` from the clone).

## Get the models

The models and covariance matrices are on
[Zenodo (10.5281/zenodo.23482275)](https://doi.org/10.5281/zenodo.23482275): exactly the
997 primary and 243 secondary models described in the paper. They are also on the
[IGVF portal](https://data.igvf.org/model-sets/IGVFDS0925IUTE/), one file per model.

`qbic-selex download` downloads a dataset, checks its checksum against Zenodo,
and **unpacks it for you**; there is nothing to unzip. Interrupted downloads
resume where they stopped, and the archive is deleted after unpacking
(`--keep-archive` keeps it). Files go into `./qbic_selex_data` (change it with `-o DIR`):

```bash
qbic-selex download primary-weights secondary-weights   # model files, 37 MB
qbic-selex download --list                              # all datasets and their sizes
```

| Dataset | Zenodo file(s) | Size | Unpacks to (inside `-o DIR`) |
|---|---|---|---|
| `primary-weights` | `models_primary.tar.zst` | 30 MB | `tf_models_primary_collection/`: the 997 primary models (`*.weights.qbic`) |
| `secondary-weights` | `models_secondary.tar.zst` | 6.5 MB | `tf_models_secondary_collection/`: the 243 secondary models |
| `primary-covs` | `covs_primary.part01`–`07.tar.zst` | 124 GB | `covs_primary/`: covariances for the primary models |
| `secondary-covs` | `covs_secondary.tar.zst` | 25 GB | `covs_secondary/`: covariances for the secondary models |
| `examples` | `example_covs.tar.zst` | 250 MB | everything for the [quick start](#quick-start): the two example models (ETV4, GABPA) in `example_models/` and the example inputs, which come with the package, plus their covariances in `example_covs/` |

Covariance matrices are only needed for z-scores and p-values
(`--compute-stats`). Each is about 130 MB, so you can fetch just the models you
need instead of whole archives; each file is checked against its md5:

```bash
qbic-selex download primary-covs --models CTCF_FL_TG40NCTGTGG_YWL_jolma2025_0_3_7mer
qbic-selex download primary-covs --model-list my_models.txt     # one model ID or path per line
```

## Quick start

`qbic-selex download examples` sets up a folder with two example models (ETV4
and GABPA, both from the primary collection), their covariance matrices and
example inputs:

```bash
qbic-selex download examples -o qbic_examples && cd qbic_examples
```

From variant coordinates, with z-scores and p-values (needs a reference genome, see below):

```bash
qbic-selex predict -v example_input_variants.csv -m example_models_list.txt \
    -c example_covs --compute-stats -g hg38.fa -o results.csv
```

From pre-extracted sequences, with and without statistics (no genome needed):

```bash
qbic-selex predict -v example_input_sequences.csv -m example_models_list.txt \
    -c example_covs --compute-stats -o results.csv
qbic-selex predict -v example_input_sequences.csv -m example_models_list.txt -o results.csv
```

Each command writes `results.csv` with 398 rows (199 variants × 2 models). If
they run, the installation works.

Reference genomes (UCSC): [hg38](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz),
[hg19](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz). Decompress with `gunzip` first.
`-g` takes a FASTA path, or a name (`hg38`) looked up as `hg38.fa` in
`$QBIC_SELEX_GENOME_DIR` or `./genome/`.

## Input

A CSV file, or a TSV for `.tsv`/`.txt` names; gzip-compressed is fine. It comes in one of two forms.

**Variant coordinates.** Required columns are `chrom,pos,ref,alt`:

```csv
chrom,pos,ref,alt
chr1,1019636,A,G
chr2,5000,G,C
```

- `pos` is **1-based** (VCF-style) by default. Use `--zero-based` for 0-based (BED-style) positions.
- `chrom` matches the genome with or without the `chr` prefix.
- Alleles are checked against the genome. A mismatch is reported in `qbic_status` and the row is not scored.

**Pre-extracted sequences.** Required columns are `ref_sequence,alt_sequence`. They take precedence when both forms are present.

```csv
ref_sequence,alt_sequence
ATCGATCGATCGATCGATCG,ATCGATCGACCGATCGATCG
```

Sequences are case-insensitive. A sequence containing `N` gets NA, unless you
set `--wildcard A|C|G|T` to substitute a base. All other input columns are
passed through to the output unchanged.

## Output

One row per input row per model:

| Column | Meaning |
|---|---|
| *(your columns)* | passed through |
| `ref_sequence`, `alt_sequence` | the sequences that were scored |
| `model` | model name |
| `predicted_effect` | change in predicted binding, alt minus ref (positive means stronger binding) |
| `z_score`, `p_value` | with `--compute-stats`: z = effect / SE, with a two-sided normal p-value |
| `qbic_status` | `ok`, or why the row has NA (see below) |

`qbic_status` values: `ref_mismatch`, `chrom_not_found`, `out_of_bounds` (too
close to a contig end), `invalid_allele`, `missing_value`, `contains_N`,
`invalid_sequence`, `shorter_than_k`.

Options:
- `--pval P` keeps only rows with `p_value <= P` (requires `--compute-stats`).
- `--output-dir` writes one `<model>.csv` per model into the `-o` directory.
- `--short-output --model-map MAP` is the compact format for bulk runs: `chrom,pos,ref,alt,model,predicted_effect[,z_score,p_value]`, with integer model IDs and 4 significant digits, about 78% smaller. It contains only scored rows. Make a map with `qbic-selex model-map -m models.txt -o map.csv`, which numbers models alphabetically from 0. [`model_maps/igvf_primary_model_ids.csv`](model_maps/igvf_primary_model_ids.csv) holds the IDs used in the files on the IGVF portal.

## Running a collection

After `qbic-selex download primary-weights primary-covs`, score your variants
with every primary model:

```bash
qbic-selex predict -v my_variants.csv \
    -m qbic_selex_data/tf_models_primary_collection \
    -c qbic_selex_data/covs_primary --compute-stats --pval 1e-6 \
    -g hg38.fa -o results.csv
```

| Flag | Meaning here |
|---|---|
| `-v my_variants.csv` | your variants (`chrom,pos,ref,alt`, or `ref_sequence,alt_sequence`) |
| `-m qbic_selex_data/tf_models_primary_collection` | the models to run: here a folder, so every model in it (all 997 primary models) |
| `-c qbic_selex_data/covs_primary` | where the covariance matrices are; each model's file is found by its name |
| `--compute-stats` | add `z_score` and `p_value` |
| `--pval 1e-6` | keep only rows with `p_value <= 1e-6` |
| `-g hg38.fa` | the reference genome, to get the sequence around each variant |
| `-o results.csv` | the output file (all models in one CSV; `--output-dir` for one file per model) |

`-m` also takes a single model file, or a text file listing model paths or
names. With names, `--model-dir` says where to find them, e.g.
`-m model_lists/secondary_collection.txt --model-dir qbic_selex_data/tf_models_secondary_collection`.
`-c` also takes a single covariance file (for one model), or a text file listing
covariance paths.

Models are processed one at a time, and results are written as each finishes.
For millions of variants, split the input into chunks and run them as a job
array, one chunk per task.

If a model fails, for example on a missing or mismatched covariance, it is left
out of the output and listed in `qbic_error_report_<time>.log` next to the
output. The run then exits with status 1; use `--allow-failures` to exit 0
instead.

**Which TFs does a variant affect?** Run all models, then map models to TFs with
[`mapping_files/TableS5_model_to_tfs_mapping.csv`](mapping_files/TableS5_model_to_tfs_mapping.csv).

**Which models cover my TF?** Look the TF up in
[`mapping_files/TableS6_tf_to_models_mapping.csv`](mapping_files/TableS6_tf_to_models_mapping.csv).
It lists each model's validation metrics.

## Commands

```text
qbic-selex predict    score variants (main command; `qbic-selex predict -h` for all options)
qbic-selex extract    only extract ref/alt context sequences from a genome
qbic-selex model-map  make a model ID map for --short-output
qbic-selex download   download and unpack model collections from Zenodo
```

The pre-1.0 scripts `qbic_predict.py` and `generate_model_map.py` still work as
wrappers around the same code; see [CHANGELOG.md](CHANGELOG.md).

## Citation

If you use QBiC-SELEX, please cite the paper (citation to be added on
publication) and the software release; see [CITATION.cff](CITATION.cff).

## License

MIT. Questions: Shengyu Li (shengyu.li@duke.edu), or open a GitHub issue.
