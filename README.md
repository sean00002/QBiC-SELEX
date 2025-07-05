# QBIC-SELEX: Quantitative, Bias-Corrected Modeling of Variant Effects on Transcription Factor Binding

A Python script for predicting how genetic variants affect transcription factor binding using QBIC-SELEX models. Supports single model and batch processing with optional statistical testing.

## Quick Start

### Example Usage
```bash
# Predict effects for a single model
python qbic_predict.py -v example_sequence_input.csv \
-m example_models/ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer.weights.qbic \
-o results.csv

# Add statistical testing (p-values and z-scores). We highly recommend using GPU (default) for this step or use multiple CPU cores for small datasets.
python qbic_predict.py -v example_variant_input.csv \
-m example_models/ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer.weights.qbic \
-c example_covs/ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer.cov.qbic \
--compute-stats \
-o results.csv

# Process multiple models without statistics computation
python qbic_predict.py -v example_sequence_input.csv \
-m example_models_list.txt \
-o results.csv

# Process multiple models with statistics computation
python qbic_predict.py -v example_variant_input.csv \
-m example_models_list.txt \
-c example_covs_list.txt \
--compute-stats \
-o results.csv
```
## Input Formats

You can provide variants in two ways:

### Option 1: Variant Coordinates
Provide chromosome, position, and alleles - sequences will be extracted automatically (genome file is needed in the `genome/` directory):

```csv
chrom,pos,ref,alt
chr1,1000,A,T
chr2,5000,G,C
```

### Option 2: Pre-extracted Sequences
If you already have sequences, provide them directly:

```csv
ref_sequence,alt_sequence
ATCGATCGATCGATCGATCG,ATCGATCGACCGATCGATCG
GCTAGCTAGCTAGCTAGCTA,GCTAGCTAGATAGCTAGCTA
```
## Data Availability

We provide example models in the `example_models` directory. The example covariance matrices are available download from Zenodo (https://zenodo.org/xxxx).
For complete QBiC-SELEX models, we provide two collections (Zenodo link: https://zenodo.org/xxxx):

- **Primary Model Collection**: Independently cross-sample validated models for 1023 transcription factors, one model per TF.
- **Secondary Model Collection**: Models curated against SNP-SELEX data if primary models perform poorly.

We also provide models to TF and TF to models mapping files `TF_to_models.txt` and `models_to_TF.txt` based on CISBP database. You can use these files to find the models that are for a given TF or the TFs that a given model is mapped to.

- **Scenario 1**: When you have a list of interested variants, you can use run all of the QBiC-SELEX models on them, and use the `models_to_TF.txt` file to get the transcription factors that the models are for.

- **Scenario 2**: When you have a list of interested TFs, you can use the `TF_to_models.txt` file to get the models that are for them. 

## Processing Modes

### Variants Effect Predictions Only (Default)
- **CPU**: Single CPU by default (fast for most cases)
- **Parallel**: Use `--n-jobs 4` if you want to speed up large datasets

### Statistics Computation (Optional with `--compute-stats`)
- **GPU**: Highly recommended (much faster for matrix operations in p-value and z-score computation)
- **CPU**: Falls back to parallel CPU if GPU unavailable (do not recommend for large datasets)


## Command Line Options

### Required
- `-v, --variants`: Your input file (CSV/TSV)
- `-m, --model`: Single model file OR text file with model paths
- `-o, --output`: Where to save results

### Optional
- `-c, --cov-file`: Covariance matrix file(s) for statistics
- `--compute-stats`: Add p-values and z-scores to output
- `--use-cpu`: Force CPU usage (GPU is default for statistics)
- `--n-jobs`: Number of CPU cores (default: 1 for predictions, all cores for statistics)
- `-g, --genome`: Reference genome (default: hg38)
- `--context-length`: Sequence context around variants (default: 20bp)
- `--wildcard`: Replace 'N' with T/G/C/A (default: return NA for sequences with N)

## Output Format

The output CSV contains all your original columns plus:

- `ref_sequence`, `alt_sequence`: The extracted sequences
- `model`: Which model was used
- `predicted_effect`: How much the variant changes binding
- `z_score`, `p_value`: Statistical significance (if `--compute-stats` used)

## File Organization

### Model Files
- Extension: `.weights.qbic`
- Example: `ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer.weights.qbic`

### Covariance Files  
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
- Make sure files exist and are readable

**Slow performance**
- For predictions: try single CPU (`--n-jobs 1`)
- For statistics: ensure GPU is available or use parallel CPU

### Error Reports
The script creates detailed error logs when things go wrong:
- `qbic_error_report_YYYYMMDD_HHMMSS.log`
- Contains specific error details and suggestions

## Dependencies

### Required
```
pandas numpy scipy pysam
```

### Optional (for GPU)
```
cupy cudf cuml
```

### Reference Genome
Put genome files in `genome/`, the genome file should be `hg38.fa` or `hg19.fa`.

