# QBIC-SELEX: Quantitative, Bias-Corrected Modeling of Variant Effects on Transcription Factor Binding

A Python script for predicting how genetic variants affect transcription factor binding using QBIC-SELEX models. Supports single model and batch processing with optional statistical testing.

The example models are in the `example_models` directory, and the corresponding [example covariance matrices](https://www.dropbox.com/scl/fo/iuyk869geskokbhbunrdj/AJr_1xNjqFZ0YIGNKmbU_Hc?rlkey=te8ienggqtsng8l134s8nxe7b&st=0hhr2yaq&dl=0) are available for download.
For complete QBiC-SELEX models and corresponding covariance matrices, we have two curated collections:

- **Primary Model Collection**: Independently cross-sample validated models for 1023 transcription factors, one model per TF. ([link](https://www.dropbox.com/scl/fi/x3q0ee2maq4g2lpovz9nv/models_primary_collection.zip?rlkey=05hjixpfjbs4rto7sbnlj5hnw&st=7j8hqxvi&dl=0))
- **Secondary Model Collection**: Models curated against SNP-SELEX data. ([link](https://www.dropbox.com/scl/fi/fq6lisnpadb07yj7m4a73/models_secondary_collection.zip?rlkey=uzpuovpjfnzreek0j38dc3vee&st=r6kwj5lq&dl=0))

We also provide models to TF and TF to models mapping files `TF_to_models.txt` and `models_to_TF.txt` based on CISBP database. Users can use these files to find the models that are for a given TF or the TFs that a given model is mapped to.

- **Scenario 1**: When users have a list of interested variants, they can use run all of the QBiC-SELEX models on them, and use the `models_to_TF.txt` file to find which TFs the variants affect the most.

- **Scenario 2**: When users have a list of interested TFs, they can use the `TF_to_models.txt` file to find which models are for them. 


## Quick Start

### Example Usage
```bash
# Predict effects for a single model
./qbic_predict.py -v example_sequence_input.csv \
-m example_models/ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer.weights.qbic \
-o results.csv

# Add statistical testing (p-values and z-scores). 
# We highly recommend using GPU (default) for this step or use multiple CPU cores for small datasets.
./qbic_predict.py -v example_input_variants.csv \
-m example_models/ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer.weights.qbic \
-c example_covs/ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer.cov.qbic \
--compute-stats \
-o results.csv

# Process multiple models without statistics computation
# example_models_list.txt is a text file with one model path per line
./qbic_predict.py -v example_input_sequences.csv \
-m example_models_list.txt \
-o results.csv

# Process multiple models with statistics computation
# example_models_list.txt and example_covs_list.txt are text files with one model or covariance matrix path per line
./qbic_predict.py -v example_input_variants.csv \
-m example_models_list.txt \
-c example_covs_list.txt \
--compute-stats \
-o results.csv

# Output individual files for each model in a directory
./qbic_predict.py -v example_input_variants.csv \
-m example_models_list.txt \
-o output_dir/ \
--output-dir
```
## Input Formats
 
Users can provide variants in two ways:

### Option 1: Variant Coordinates (example_input_variants.csv)
Provide chromosome, position, and alleles - context sequences will be extracted automatically (genome file is needed in the `genome/` directory in this case):

```csv
chrom,pos,ref,alt
chr1,1000,A,T
chr2,5000,G,C
```

### Option 2: Pre-extracted Context Sequences (example_input_sequences.csv)
If users already have context sequences, they can provide them directly:

```csv
ref_sequence,alt_sequence
ATCGATCGATCGATCGATCG,ATCGATCGACCGATCGATCG
GCTAGCTAGCTAGCTAGCTA,GCTAGCTAGATAGCTAGCTA
```

## Processing Modes

### Variants Effect Predictions Only (Default)
By default, the script computes the effect of variants on transcription factor binding only without any statistics computation.
- **CPU**: Single CPU by default (fast for most cases)
- **Parallel**: Use `--n-jobs 4` if users want to speed up large datasets

### Statistics Computation (Optional with `--compute-stats`)
If users want to compute the p-values and z-scores for the predictions, they can use the `--compute-stats` option, but it requires a covariance matrix file or a text file with one covariance matrix path per line with `-c`. 
- **GPU**: Highly recommended (much faster for matrix operations in p-value and z-score computation)
- **CPU**: Falls back to parallel CPU if GPU unavailable (do not recommend for large datasets)

### Batch Processing for Multiple Models 
Users can provide a text file with one model path per line with `-m`, and the script will process all the models in the text file.
If users want to process multiple models with statistics computation, they will also need to provide a text file with one covariance matrix path per line with `-c`. The model name must match the covariance matrix name, or match any of the covariance matrix names in the text file.
- **CPU**: Single CPU by default (fast for most cases)
- **GPU**: Highly recommended for statistics computation


## Command Line Options

### Required
- `-v, --variants`: Users' input file (CSV/TSV)
- `-m, --model`: Single model file OR text file with model paths
- `-o, --output`: Where to save results

### Optional
- `-c, --cov-file`: Covariance matrix file(s) for statistics
- `--compute-stats`: Add p-values and z-scores to output
- `--output-dir`: Output individual files for each model in a directory (works for both single and batch processing)
- `--use-cpu`: Force CPU usage (GPU is default for statistics)
- `--n-jobs`: Number of CPU cores (default: 1 for predictions, all cores for statistics)
- `-g, --genome`: Reference genome (default: hg38)
- `--context-length`: Sequence context around variants (default: 10bp)
- `--wildcard`: Replace 'N' with T/G/C/A (default: return NA for sequences with N)

## Output Formats

### Single File Output (Default)
When `--output-dir` is not specified, all results are saved to a single CSV file containing:
- All users' original columns
- `ref_sequence`, `alt_sequence`: The extracted context sequences
- `model`: Which model was used
- `predicted_effect`: How much the variant changes binding
- `z_score`, `p_value`: Statistical significance (if `--compute-stats` used)

### Directory Output (with `--output-dir`)
When `--output-dir` is specified, individual files are created for each model:
- **Single model processing**: Creates `{model_name}.csv` in the specified directory
- **Batch processing**: Creates separate `{model_name}.csv` files for each model in the specified directory
- Each file contains the same columns as single file output, but only for that specific model

## File Organization

### Model Files
- Extension: `.weights.qbic`
- Example: `ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer.weights.qbic`

### Covariance Files (Optional)
- Extension: `.cov.qbic`
- Must match model names: `ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer.cov.qbic`

### Batch Processing Files
Create text files with one path per line:

**models.txt:**
```
/path/to/model1.weights.qbic
/path/to/model2.weights.qbic
```

**covariance.txt:**
```
/path/to/model1.cov.qbic
/path/to/model2.cov.qbic
```

## Troubleshooting

### Common Issues

**"GPU not available"**
- Script automatically falls back to CPU
- No action needed unless you specifically need GPU

**"Covariance matrix not found"**
- Check that covariance file names match model names
- Make sure the file paths exist in input text files

**Slow performance**
- For predictions: try single CPU (`--n-jobs 1`)
- For statistics: ensure GPU is available or use parallel CPU

### Error Reports
The script creates detailed error logs when things go wrong:
- `qbic_error_report_YYYYMMDD_HHMMSS.log`

## Dependencies
Here are the dependencies for the script. Please see details in the `env.yml` file.

### Required
```
python>=3.10
pandas>=1.5.3
numpy>=1.26.4
scipy>=1.12.0
pysam>=0.21.0
```

### Optional (for GPU)
```
cupy>=13.0.0
cudf>=23.12.01
cuml>=23.12.00
pytorch>=2.4.0
```

### Reference Genome
Genome files are required in the `genome/` directory if you have variant coordinates as input.
The genome file should be `hg38.fa` or `hg19.fa`.

