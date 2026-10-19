# QBiC-SELEX

**Quantitative, bias-corrected prediction of variant effects on transcription factor binding.**

QBiC-SELEX scores how a genetic variant changes the binding of a transcription
factor (TF), using k-mer models trained on SELEX data. For each variant and TF
model it reports the predicted change in binding (`predicted_effect`).
Optionally it also reports a z-score and a p-value, computed from the model's
coefficient covariance.

- **Primary collection:** 997 models, one per human TF, each cross-experiment validated (Table S3).
- **Secondary collection:** 243 models for 203 TFs, curated against SNP-SELEX data (Table S4).
- **Mapping files:** model → TF and TF → model, covering 1,111 TFs via CIS-BP (Tables S5 and S6).

## Install

```bash
pip install qbic-selex            # CPU
pip install "qbic-selex[gpu]"     # + CuPy with CUDA 12 libraries, for fast statistics on NVIDIA GPUs
```

Python ≥ 3.10. To run from a clone instead: `pip install -e .`. The scripts
`qbic_predict.py` and `generate_model_map.py` still work, without installing,
as wrappers around the same code.

## Get the models

The models, covariance matrices and example covariances are on
[Zenodo (10.5281/zenodo.15891511)](https://doi.org/10.5281/zenodo.15891511).
They are also on the [IGVF portal](https://data.igvf.org/model-sets/IGVFDS0925IUTE/).

Download them with `qbic-selex download`, which checks each file's checksum
against Zenodo, resumes interrupted downloads, and unpacks into
`./qbic_selex_data` (change it with `-o DIR`):

```bash
qbic-selex download primary secondary     # model files, 38 MB
qbic-selex download example-covs          # covariances for the two example models, 250 MB
qbic-selex download --list                # all datasets and their sizes
```

| Dataset | Zenodo file | Size | Contents |
|---|---|---|---|
| `primary` | `models_primary.tar.zst` | 31 MB | primary-collection model files (`*.weights.qbic`) |
| `secondary` | `models_secondary.tar.zst` | 7 MB | secondary-collection model files |
| `example-covs` | `example_covs.tar.zst` | 250 MB | covariances for the two example models in this repository |
| `primary-covs` | `covs_primary.tar.zst` | 127 GB | covariances for the primary models (only needed for z/p) |
| `secondary-covs` | `covs_secondary.tar.zst` | 26 GB | covariances for the secondary models |

By default it downloads from the Zenodo version this release was made with;
`--record latest` follows the newest version. Unpacking needs about twice the
archive size in free space, briefly (`--keep-archive` keeps the `.tar.zst`).

The archives also contain candidate models that were curated out of the final
collections. To run exactly the published collections, use the lists in
[`model_lists/`](model_lists) (see [Running a collection](#running-a-collection)).

Covariance matrices are only needed with `--compute-stats`. Each primary 7-mer
covariance is 128 MB. To get only some of them, download the archive from
Zenodo yourself and extract just those files, e.g.
`tar -m -I zstd -xf covs_primary.tar.zst covs_primary/CTCF_FL_TG40NCTGTGG_YWL_jolma2025_0_3_7mer.cov_8192.npy`.

## Quick start

The repository includes two example models (ETV4 and GABPA) and example inputs
(`git clone https://github.com/sean00002/QBiC-SELEX && cd QBiC-SELEX`):

```bash
# Effects only, from pre-extracted sequences
qbic-selex predict -v example_input_sequences.csv -m example_models_list.txt -o results.csv

# With z-scores and p-values (download and unpack example_covs.tar.zst first)
qbic-selex predict -v example_input_sequences.csv -m example_models_list.txt \
    -c example_covs --compute-stats -o results.csv

# From variant coordinates (needs a reference genome; see below)
qbic-selex predict -v example_input_variants.csv -m example_models_list.txt \
    -c example_covs --compute-stats -g hg38.fa -o results.csv
```

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
- The sequence scored is `--context-length` bases (default 10) on each side of the variant. Indels and multi-base substitutions are supported.
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
close to a contig end for the requested context), `invalid_allele`,
`missing_value`, `contains_N`, `invalid_sequence`, `shorter_than_k`.

Options:
- `--pval P` keeps only rows with `p_value <= P` (requires `--compute-stats`).
- `--output-dir` writes one `<model>.csv` per model into the `-o` directory.
- `--short-output --model-map MAP` is the compact format for bulk runs: `chrom,pos,ref,alt,model,predicted_effect[,z_score,p_value]`, with integer model IDs and 4 significant digits, about 78% smaller. It contains only scored rows. Make a map with `qbic-selex model-map -m models.txt -o map.csv`, which numbers models alphabetically from 0. [`model_maps/igvf_primary_model_ids.csv`](model_maps/igvf_primary_model_ids.csv) holds the IDs used in the files on the IGVF portal.

## Statistics

For a variant with k-mer count difference vector *c* (alt minus ref, over the
model's canonical k-mers), model coefficients *β* and coefficient covariance *Σ*:

- effect = *c*·*β*
- SE = √(*c*ᵀ *Σ* *c*)
- z = effect / SE
- p = 2(1 − Φ(|z|))

A variant whose effect is exactly 0 gets z = 0 and p = 1. The computation runs in
batches with float32 linear algebra, on the GPU when available (`--device
auto|cpu|gpu`). See [docs/statistics.md](docs/statistics.md) for details.

`--compute-stats` needs a covariance for every model. `-c` accepts:
- a directory of `*.cov_<n>.npy` files, matched to models by name;
- a single file, when running one model;
- a text file with one covariance path, or `model_path,cov_path`, per line.

## Running a collection

`-m` takes a model file, a directory of models, or a text file with one model
path per line. A list can also give bare model names, which `--model-dir`
resolves:

```bash
qbic-selex predict -v my_variants.csv \
    -m model_lists/primary_collection.txt --model-dir qbic_selex_data/tf_models_primary_collection \
    -c qbic_selex_data/covs_primary --compute-stats --pval 1e-6 \
    --short-output --model-map model_maps/igvf_primary_model_ids.csv \
    -g hg38.fa -o results.csv
```

Models are processed one at a time, and results are written as each finishes.
For millions of variants, split the input into chunks and run them as a job
array, one chunk per task. This is how the IGVF predictions (millions of variants × all
primary models) were made, on GPU nodes.

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

## Upgrading from pre-1.0 scripts

`qbic_predict.py` (and `qbic_predict_faster.py`) accept the old options.
`--use-gpu`/`--use-cpu` map to `--device`, and `--n-jobs` is accepted but has
no effect. Results are unchanged, apart from these fixes:

- With `--wildcard`, rows containing N now also get z/p. Previously they got NaN.
- Rows that cannot be scored stay in the output with a `qbic_status` reason. Previously they were dropped.
- Lowercase sequences are scored correctly. Previously they scored 0.
- A failed statistics step fails that model. Previously the z/p columns were silently dropped.

See [CHANGELOG.md](CHANGELOG.md).

## Citation

If you use QBiC-SELEX, please cite the paper (citation to be added on
publication) and the software release; see [CITATION.cff](CITATION.cff).

## License

MIT. Questions: Shengyu Li (shengyu.li@duke.edu), or open a GitHub issue.
