#!/usr/bin/env python3
"""
QBIC-SELEX Variant Effect Prediction Script
==========================================

Predict variant effects using QBIC-SELEX models with optional statistical significance computation.
Supports both single model and batch processing with GPU/CPU acceleration.

Features:
- Single model and batch processing
- GPU/CPU parallel processing
- Sequence extraction from reference genomes
- Statistical significance computation (p-values and z-scores)
- Flexible N handling (NA by default, configurable replacement)
- Comprehensive error handling and logging

Author: Shengyu Li
"""

import pandas as pd
import numpy as np
import os
import pickle
import argparse
import scipy.stats as stats
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import traceback
from datetime import datetime
import signal
import re

# Optional dependencies
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    warnings.warn("tqdm not available. Progress bars will be disabled.")

try:
    import cupy as cp
    import cudf
    import cuml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    warnings.warn("GPU libraries not available. Using CPU only.")

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'util_scripts'))
import sequence_utils as utils
import extract_seq

# Global caches and error collector
_model_cache = {}
_cov_cache = {}
_error_collector = []

# ========================================
# UTILITY FUNCTIONS
# ========================================

def extract_model_name(file_path: str) -> str:
    """Extract model name from file path using first part before '.'"""
    return Path(file_path).name.split('.')[0]

def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Statistics computation timed out")

def run_with_timeout(func, timeout_seconds=300, *args, **kwargs):
    """Run a function with a timeout."""
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)
        return result
    finally:
        signal.signal(signal.SIGALRM, old_handler)

def get_gpu_info() -> str:
    """Get GPU information if available."""
    if not GPU_AVAILABLE:
        return "Not available"
    
    try:
        device = cp.cuda.Device(0)
        props = device.attributes
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
        memory_gb = props['TotalGlobalMemory'] / (1024**3)
        return f"{gpu_name} ({memory_gb:.1f}GB)"
    except Exception:
        return "Available (details unknown)"

@lru_cache(maxsize=128)
def sliding_window(seq: str, k: int) -> Tuple[str, ...]:
    """Create sliding window of k-mers from sequence. Cached for efficiency."""
    return tuple(seq[i:i+k] for i in range(len(seq) - k + 1))

# ========================================
# ERROR HANDLING
# ========================================

def save_error_log(error_type: str, details: str, model_name: str = None):
    """Save error information to global collector."""
    error_entry = {
        'timestamp': datetime.now().isoformat(),
        'type': error_type,
        'model': model_name,
        'details': details
    }
    _error_collector.append(error_entry)

def save_combined_error_log(output_dir: str = ".") -> Optional[str]:
    """Save all collected errors to a single combined log file."""
    if not _error_collector:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_error_file = os.path.join(output_dir, f"qbic_error_report_{timestamp}.log")
    
    try:
        with open(combined_error_file, 'w') as f:
            f.write("QBIC Combined Error Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total Errors: {len(_error_collector)}\n")
            f.write("=" * 60 + "\n\n")
            
            # Group errors by type
            error_groups = {}
            for error in _error_collector:
                error_type = error['type']
                if error_type not in error_groups:
                    error_groups[error_type] = []
                error_groups[error_type].append(error)
            
            # Write grouped errors
            for error_type, errors in error_groups.items():
                f.write(f"ERROR TYPE: {error_type}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Count: {len(errors)}\n\n")
                
                for i, error in enumerate(errors, 1):
                    f.write(f"Error {i}:\n")
                    f.write(f"  Timestamp: {error['timestamp']}\n")
                    if error['model']:
                        f.write(f"  Model: {error['model']}\n")
                    f.write(f"  Details:\n{error['details']}\n")
                    if i < len(errors):
                        f.write("-" * 30 + "\n")
                
                f.write("\n" + "=" * 60 + "\n\n")
        
        return combined_error_file
    except Exception as e:
        print(f"Could not save combined error log: {e}")
        return None

# ========================================
# DATA LOADING AND CACHING
# ========================================

def load_model_cached(model_path: str) -> Tuple[Dict[str, float], int, Dict[str, float]]:
    """Load QBIC model from file with caching."""
    if model_path in _model_cache:
        return _model_cache[model_path]
    
    with open(model_path, 'rb') as f:
        partial_dict = pickle.load(f)
    
    complete_dict = utils.recover_non_rc_dict(partial_dict)
    kmer_size = len(list(complete_dict.keys())[0])
    
    _model_cache[model_path] = (complete_dict, kmer_size, partial_dict)
    return complete_dict, kmer_size, partial_dict

def load_covariance_matrix_cached(cov_path: str) -> np.ndarray:
    """Load covariance matrix from file with caching."""
    if cov_path in _cov_cache:
        return _cov_cache[cov_path]
    
    # Load NPY format (new format)
    if cov_path.endswith('.npy'):
        try:
            match = re.search(r'.cov_(\d+)\.npy$', cov_path)
            if match:
                n = int(match.group(1))
                upper_half = np.load(cov_path)
                # Reconstruct full matrix
                cov_matrix = np.empty((n, n), dtype=upper_half.dtype)
                iu, ju = np.triu_indices(n)
                cov_matrix[iu, ju] = upper_half
                cov_matrix[ju, iu] = upper_half
                _cov_cache[cov_path] = cov_matrix
                return cov_matrix
            else:
                raise ValueError(f"Could not extract matrix size from filename: {cov_path}")
        except Exception as e:
            print(f"Error loading NPY format: {e}")
            raise
    
    # Fallback to pickle format (legacy)
    try:
        with open(cov_path, 'rb') as f:
            cov_matrix = pickle.load(f)
        _cov_cache[cov_path] = cov_matrix
        return cov_matrix
    except Exception as e:
        raise ValueError(f"Failed to load covariance matrix from {cov_path}: {e}")

def read_file_paths(file_path: str, file_type: str) -> Union[List[str], Dict[str, str]]:
    """Read file paths from text file, handling both model and covariance files."""
    paths = [] if file_type == 'model' else {}
    missing_files = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith('#'):
                if file_type == 'model':
                    if not Path(line).exists():
                        missing_files.append((line_num, line))
                    else:
                        paths.append(line)
                else:  # covariance
                    if ',' in line:
                        model_path, cov_path = line.split(',', 1)
                        model_name = extract_model_name(model_path.strip())
                        cov_path = cov_path.strip()
                    else:
                        cov_path = line
                        model_name = extract_model_name(cov_path)
                    
                    if not Path(cov_path).exists():
                        missing_files.append((line_num, cov_path))
                    else:
                        paths[model_name] = cov_path
    
    if missing_files:
        print(f"Warning: {len(missing_files)} {file_type} files not found (will be skipped)")
        missing_details = f"Missing {file_type} files in {file_path}:\n" + "\n".join([f"Line {line_num}: {file_path}" for line_num, file_path in missing_files])
        save_error_log(f"MISSING_{file_type.upper()}_FILES", missing_details)
    
    if not paths:
        raise ValueError(f"No valid {file_type} paths found in {file_path}")
    
    return paths

# ========================================
# SEQUENCE PROCESSING
# ========================================

def extract_sequences_from_variants(variants_df: pd.DataFrame, 
                                  context_length: int = 20,
                                  genome: str = "hg38") -> pd.DataFrame:
    """Extract reference and alternate sequences for variants."""
    if 'ref_sequence' in variants_df.columns and 'alt_sequence' in variants_df.columns:
        print("Sequences already present in input, skipping extraction")
        return variants_df
    
    required_cols = ['chrom', 'pos', 'ref', 'alt']
    if not all(col in variants_df.columns for col in required_cols):
        raise ValueError(f"Either sequences (ref_sequence, alt_sequence) or variant information ({', '.join(required_cols)}) must be provided")
    
    return extract_seq.batch_extract_sequences(
        variants_df,
        context_length=context_length,
        genome_file=genome
    )

def process_variants_vectorized(df: pd.DataFrame, k: int) -> Tuple[List[List[str]], List[List[str]]]:
    """Process variant data to extract k-mers for prediction."""
    if not all(col in df.columns for col in ['ref_sequence', 'alt_sequence']):
        raise ValueError("DataFrame must contain 'ref_sequence' and 'alt_sequence' columns")
    
    alt_list = [list(sliding_window(seq, k)) for seq in df['alt_sequence']]
    ref_list = [list(sliding_window(seq, k)) for seq in df['ref_sequence']]
    
    return alt_list, ref_list

def count_sequences_with_n(variants_df: pd.DataFrame) -> int:
    """Count sequences containing 'N' characters."""
    if 'ref_sequence' not in variants_df.columns or 'alt_sequence' not in variants_df.columns:
        return 0
    
    na_count = 0
    for _, row in variants_df.iterrows():
        if 'N' in row['ref_sequence'] or 'N' in row['alt_sequence']:
            na_count += 1
    
    return na_count

# ========================================
# PREDICTION FUNCTIONS
# ========================================

def qbic_predict_vectorized(model_dict: Dict[str, float], alt_seqs: List[List[str]], 
                           ref_seqs: List[List[str]], wildcard: Optional[str] = None) -> Tuple[np.ndarray, int]:
    """Predict variant effects using QBIC model."""
    na_count = 0
    if wildcard is None:
        for alt_seq, ref_seq in zip(alt_seqs, ref_seqs):
            if any('N' in kmer for kmer in alt_seq) or any('N' in kmer for kmer in ref_seq):
                na_count += 1
    
    replacement = wildcard if wildcard is not None else 'A'
    
    # Pre-compute model values for faster lookup
    model_values = np.array(list(model_dict.values()), dtype=np.float32)
    model_keys = list(model_dict.keys())
    key_to_idx = {key: idx for idx, key in enumerate(model_keys)}
    
    predictions = np.zeros(len(alt_seqs), dtype=np.float32)
    
    for i, (alt_seq, ref_seq) in enumerate(zip(alt_seqs, ref_seqs)):
        if wildcard is None and (any('N' in kmer for kmer in alt_seq) or any('N' in kmer for kmer in ref_seq)):
            predictions[i] = np.nan
            continue
        
        alt_sum = sum(model_values[key_to_idx[kmer.replace('N', replacement)]] 
                     for kmer in alt_seq if kmer.replace('N', replacement) in key_to_idx)
        ref_sum = sum(model_values[key_to_idx[kmer.replace('N', replacement)]] 
                     for kmer in ref_seq if kmer.replace('N', replacement) in key_to_idx)
        
        predictions[i] = alt_sum - ref_sum
    
    return predictions, na_count

# ========================================
# STATISTICAL COMPUTATION
# ========================================

def convert_kmers_to_c_vectorized(alt_kmers: List[str], ref_kmers: List[str], 
                                 partial_dict: Dict[str, float]) -> np.ndarray:
    """Convert k-mers to coefficient vector using partial dictionary."""
    partial_keys = list(partial_dict.keys())
    key_to_idx = {key: idx for idx, key in enumerate(partial_keys)}
    
    output = np.zeros(len(partial_dict), dtype=np.int8)
    
    for kmer in alt_kmers:
        if kmer in key_to_idx:
            output[key_to_idx[kmer]] += 1
        else:
            rc_kmer = utils.reverse_complement(kmer)
            if rc_kmer in key_to_idx:
                output[key_to_idx[rc_kmer]] += 1
    
    for kmer in ref_kmers:
        if kmer in key_to_idx:
            output[key_to_idx[kmer]] -= 1
        else:
            rc_kmer = utils.reverse_complement(kmer)
            if rc_kmer in key_to_idx:
                output[key_to_idx[rc_kmer]] -= 1
    
    return output

def compute_p_value_cpu(alt_kmers: List[str], ref_kmers: List[str], 
                       partial_dict: Dict[str, float], cov_matrix: np.ndarray) -> Tuple[float, float]:
    """Compute p-value and z-score using CPU."""
    c = convert_kmers_to_c_vectorized(alt_kmers, ref_kmers, partial_dict)
    lm_values = np.array(list(partial_dict.values()), dtype=np.float32)
    
    estimate = np.dot(c.T, lm_values)
    
    if estimate == 0:
        return 0.0, 1.0
    
    se_estimate = np.sqrt(np.dot(np.dot(c.T, cov_matrix), c))
    t_score = estimate / se_estimate
    p_value = 2 * stats.norm.sf(np.abs(t_score))
    
    return float(t_score), float(p_value)

def compute_statistics_gpu(alt_list: List[List[str]], ref_list: List[List[str]], 
                          partial_dict: Dict[str, float], cov_matrix: np.ndarray,
                          model_name: str = None) -> Tuple[List[float], List[float]]:
    """Compute statistics efficiently using GPU processing."""
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available for processing")
    
    lm_values_cp = cp.array(list(partial_dict.values()), dtype=cp.float32)
    cov_matrix_cp = cp.asarray(cov_matrix, dtype=cp.float32)
    
    t_scores = []
    p_values = []
    error_count = 0
    
    iterator = tqdm(zip(alt_list, ref_list), total=len(alt_list), desc="Computing statistics", unit="variant") if TQDM_AVAILABLE else zip(alt_list, ref_list)
    
    for i, (alt_kmers, ref_kmers) in enumerate(iterator):
        try:
            c = convert_kmers_to_c_vectorized(alt_kmers, ref_kmers, partial_dict)
            c_cp = cp.array(c, dtype=cp.float32)
            
            estimate = cp.dot(c_cp, lm_values_cp)
            estimate_np = float(cp.asnumpy(estimate))
            
            if estimate_np == 0:
                t_scores.append(0.0)
                p_values.append(1.0)
                continue
            
            se_estimate = cp.sqrt(cp.dot(c_cp, cp.dot(cov_matrix_cp, c_cp)))
            
            if se_estimate == 0 or cp.isnan(se_estimate) or cp.isinf(se_estimate):
                raise ValueError(f"Invalid standard error: {float(cp.asnumpy(se_estimate))}")
            
            t_score = estimate / se_estimate
            t_score_np = float(cp.asnumpy(t_score))
            
            if np.isnan(t_score_np) or np.isinf(t_score_np):
                raise ValueError(f"Invalid t-score: {t_score_np}")
            
            p_value = 2 * stats.norm.sf(np.abs(t_score_np))
            
            t_scores.append(t_score_np)
            p_values.append(p_value)
            
        except Exception as e:
            error_count += 1
            error_msg = f"Variant {i}: {type(e).__name__}: {str(e)}"
            save_error_log("GPU_PROCESSING_ERROR", error_msg, model_name=model_name)
            t_scores.append(np.nan)
            p_values.append(np.nan)
    
    # Clean up GPU memory
    del lm_values_cp, cov_matrix_cp
    cp.get_default_memory_pool().free_all_blocks()
    
    if error_count > 0:
        print(f"Warning: {error_count}/{len(alt_list)} variants failed during statistics computation")
    
    return t_scores, p_values

def compute_statistics_parallel(alt_list: List[List[str]], ref_list: List[List[str]], 
                              partial_dict: Dict[str, float], cov_matrix: np.ndarray,
                              use_gpu: bool = False, n_jobs: int = -1,
                              model_name: str = None) -> Tuple[List[float], List[float]]:
    """Compute statistics efficiently using GPU sequential processing or parallel CPU."""
    if use_gpu and GPU_AVAILABLE:
        try:
            return compute_statistics_gpu(alt_list, ref_list, partial_dict, cov_matrix, model_name)
        except Exception as e:
            error_details = f"GPU processing failed: {str(e)}\nModel: {model_name}"
            save_error_log("GPU_PROCESSING_ERROR", error_details, model_name=model_name)
            print(f"Error: GPU processing failed: {str(e)}")
            print("Use --use-cpu instead of --use-gpu")
            raise RuntimeError(f"GPU processing failed: {str(e)} - please use CPU instead")
    
    # CPU processing
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    print(f"Using {n_jobs} CPU cores for statistics computation")
    
    t_scores = []
    p_values = []
    
    if n_jobs == 1:
        # Sequential processing
        for alt_kmers, ref_kmers in zip(alt_list, ref_list):
            try:
                t_score, p_value = compute_p_value_cpu(alt_kmers, ref_kmers, partial_dict, cov_matrix)
                t_scores.append(t_score)
                p_values.append(p_value)
            except Exception as e:
                error_msg = f"CPU processing error: {str(e)}"
                save_error_log("STATS_COMPUTATION_ERROR", error_msg, model_name=model_name)
                t_scores.append(np.nan)
                p_values.append(np.nan)
    else:
        # Parallel processing
        def compute_stats_worker(args):
            i, alt_kmers, ref_kmers = args
            try:
                return i, compute_p_value_cpu(alt_kmers, ref_kmers, partial_dict, cov_matrix)
            except Exception as e:
                error_msg = f"Variant {i}: {type(e).__name__}: {str(e)}"
                save_error_log("STATS_COMPUTATION_ERROR", error_msg, model_name=model_name)
                return i, (np.nan, np.nan)
        
        args_list = [(i, alt_kmers, ref_kmers) for i, (alt_kmers, ref_kmers) in enumerate(zip(alt_list, ref_list))]
        t_scores = [np.nan] * len(alt_list)
        p_values = [np.nan] * len(alt_list)
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(compute_stats_worker, args) for args in args_list]
            
            iterator = tqdm(as_completed(futures), total=len(futures), desc="Computing statistics", unit="variant") if TQDM_AVAILABLE else as_completed(futures)
            
            for future in iterator:
                i, (t_score, p_value) = future.result()
                t_scores[i] = t_score
                p_values[i] = p_value
    
    return t_scores, p_values

# ========================================
# MAIN PREDICTION FUNCTIONS
# ========================================

def predict_single_model(model_path: str, variants_df: pd.DataFrame, 
                        compute_stats: bool = False, cov_file: Optional[str] = None,
                        use_gpu: bool = True, genome: str = "hg38", 
                        wildcard: Optional[str] = None, n_jobs: int = -1) -> pd.DataFrame:
    """Predict variant effects using a single model."""
    model_name = extract_model_name(model_path)
    
    print(f"Processing model: {model_name}")
    
    # Load model
    complete_dict, kmer_size, partial_dict = load_model_cached(model_path)
    
    # Process variants
    alt_list, ref_list = process_variants_vectorized(variants_df, kmer_size)
    
    # Make predictions
    print(f"Predicting variant effects...")
    predictions, na_count = qbic_predict_vectorized(complete_dict, alt_list, ref_list, wildcard)
    print(f"Predictions completed")
    
    # Create results DataFrame
    results = variants_df.copy()
    results['model'] = model_name
    results['predicted_effect'] = predictions
    
    # Ensure sequences are in output
    if 'ref_sequence' not in results.columns:
        results['ref_sequence'] = [''] * len(results)
    if 'alt_sequence' not in results.columns:
        results['alt_sequence'] = [''] * len(results)
    
    # Compute statistics if requested
    if compute_stats:
        print(f"Computing statistics...")
        if cov_file is None:
            raise ValueError(f"Covariance file is required when --compute-stats is used")
        
        # Determine covariance matrix path
        if (Path(cov_file).suffix == '.qbic' and 'cov' in Path(cov_file).name) or Path(cov_file).suffix == '.npy':
            cov_path = cov_file
            print(f"Using direct covariance matrix: {Path(cov_file).name}")
        else:
            # Text file with covariance paths
            print(f"Reading covariance paths from: {cov_file}")
            cov_paths = read_file_paths(cov_file, 'covariance')
            if model_name in cov_paths:
                cov_path = cov_paths[model_name]
                print(f"Found covariance matrix for {model_name}: {Path(cov_path).name}")
            else:
                raise ValueError(f"Covariance matrix not found for model {model_name} in {cov_file}")
        
        # Load covariance matrix
        print(f"Loading covariance matrix...")
        cov_matrix = load_covariance_matrix_cached(cov_path)
        print(f"Covariance matrix loaded: {cov_matrix.shape}")
        
        # Compute statistics
        try:
            print(f"Starting statistics computation...")
            t_scores, p_values = run_with_timeout(
                compute_statistics_parallel,
                600,  # 10 minutes timeout
                alt_list, ref_list, partial_dict, cov_matrix, 
                use_gpu, n_jobs, model_name
            )
            results['z_score'] = t_scores
            results['p_value'] = p_values
            print(f"Statistics completed")
        except TimeoutError as e:
            error_details = f"Statistics computation timed out for {model_name}\nError: {str(e)}"
            save_error_log("STATS_TIMEOUT_ERROR", error_details, model_name=model_name)
            print(f"Statistics computation timed out for {model_name} - continuing without statistics")
        except Exception as e:
            error_details = f"Statistics computation failed for {model_name}\nError: {str(e)}"
            save_error_log("STATS_COMPUTATION_ERROR", error_details, model_name=model_name)
            print(f"Statistics computation failed for {model_name} - continuing without statistics")
    
    print(f"Model {model_name} completed")
    print()
    
    return results

def predict_batch_models(models_file: str, variants_df: pd.DataFrame,
                        compute_stats: bool = False, cov_file: Optional[str] = None,
                        use_gpu: bool = True, genome: str = "hg38", 
                        wildcard: Optional[str] = None, n_jobs: int = -1) -> pd.DataFrame:
    """Predict variant effects using models specified in a text file."""
    print("\nVALIDATION")
    print("-" * 50)
    
    # Read model paths
    model_paths = read_file_paths(models_file, 'model')
            print(f"[OK] Found {len(model_paths)} models for batch processing")
    
    # Check sequences for N characters
    na_count = count_sequences_with_n(variants_df) if wildcard is None else 0
    if na_count > 0:
        print(f"[WARNING] Found {na_count} sequences containing 'N' characters")
        if wildcard is None:
            print("   These sequences will return NA predictions")
        else:
            print(f"   'N' will be replaced with '{wildcard}'")
    else:
        print("[OK] No sequences contain 'N' characters")
    
    # Read covariance paths if provided
    cov_paths = {}
    if compute_stats and cov_file is not None:
        if (Path(cov_file).suffix == '.qbic' and 'cov' in Path(cov_file).name) or Path(cov_file).suffix == '.npy':
            # Single covariance matrix file
            print(f"[OK] Using single covariance matrix: {Path(cov_file).name}")
            for model_path in model_paths:
                model_name = extract_model_name(model_path)
                cov_paths[model_name] = cov_file
        else:
            # Text file with covariance paths
            cov_paths = read_file_paths(cov_file, 'covariance')
            print(f"[OK] Loaded {len(cov_paths)} covariance matrix paths")
    
    # Validate model-covariance mapping
    if compute_stats and cov_file is not None:
        missing_cov_models = []
        for model_path in model_paths:
            model_name = extract_model_name(model_path)
            if model_name not in cov_paths:
                missing_cov_models.append(model_name)
        
        if missing_cov_models:
            error_msg = f"Error: {len(missing_cov_models)} models missing covariance matrices"
            error_details = f"Missing covariance matrices for models:\n" + "\n".join([f"  - {model}" for model in missing_cov_models])
            save_error_log("MISSING_COVARIANCE_MATRICES", error_details)
            print(f"\n{error_msg}")
            raise RuntimeError(error_msg)
        else:
            print(f"[OK] All {len(model_paths)} models have corresponding covariance matrices")
    
    print(f"\nProcessing {len(model_paths)} models...")
    print()
    
    # Process models
    all_results = []
    failed_models = []
    
    processing_mode = "GPU (sequential)" if use_gpu and GPU_AVAILABLE else f"CPU ({n_jobs if n_jobs != -1 else 'all'} cores)"
    print(f"Using {processing_mode} processing")
    print()
    
    for model_path in model_paths:
        model_name = extract_model_name(model_path)
        cov_path = cov_paths.get(model_name) if compute_stats and cov_file else None
        
        try:
            result = predict_single_model(
                model_path, variants_df, compute_stats, cov_path, 
                use_gpu, genome, wildcard, n_jobs
            )
            if result is not None:
                all_results.append(result)
            else:
                failed_models.append(model_name)
        except Exception as e:
            failed_models.append(model_name)
            error_details = f"Model {model_name}: {type(e).__name__}: {str(e)}"
            save_error_log("BATCH_PROCESSING_ERROR", error_details, model_name=model_name)
    
    if failed_models:
        print(f"Warning: {len(failed_models)} models failed out of {len(model_paths)}")
    
    if not all_results:
        raise RuntimeError("No models were successfully processed")
    
    print(f"Combining results from {len(all_results)} models...")
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Clear caches
    _model_cache.clear()
    _cov_cache.clear()
    
    print(f"Batch processing complete! Total predictions: {len(combined_results)}")
    return combined_results

# ========================================
# MAIN FUNCTION
# ========================================

def main():
    """Main function with comprehensive argument parsing and execution."""
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                    QBiC-SELEX Variant Effect Prediction                      ║")
    print("║              Quantitative, Bias-Corrected Modeling of TF Binding             ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    
    parser = argparse.ArgumentParser(
        description="Predict variant effects using QBIC models with efficient GPU/CPU processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model prediction
  python qbic_predict_final.py -v variants.csv -m model.weights.qbic -o results.csv
  
  # Single model with statistics (GPU default)
  python qbic_predict_final.py -v variants.csv -m model.weights.qbic -c model.cov.npy -o results.csv --compute-stats
  
  # Batch processing
  python qbic_predict_final.py -v variants.csv -m models.txt -o results.csv
  
  # Batch processing with statistics (CPU)
  python qbic_predict_final.py -v variants.csv -m models.txt -c covariance.txt -o results.csv --compute-stats --use-cpu --n-jobs 8

Input Formats:
  Variant coordinates: chrom,pos,ref,alt
  Pre-extracted sequences: ref_sequence,alt_sequence
        """
    )
    
    # Required arguments
    parser.add_argument("-v", "--variants", required=True, 
                       help="Input variants file (CSV/TSV)")
    parser.add_argument("-m", "--model", required=True, 
                       help="Path to single model file or text file with model paths")
    parser.add_argument("-o", "--output", required=True, 
                       help="Output file path or directory")
    
    # Optional arguments
    parser.add_argument("-c", "--cov-file", 
                       help="Covariance matrix file or text file with covariance paths")
    parser.add_argument("--compute-stats", action="store_true", 
                       help="Compute p-values and z-scores (requires --cov-file)")
    parser.add_argument("--output-dir", action="store_true",
                       help="Output individual files for each model")
    parser.add_argument("--use-gpu", action="store_true", default=None,
                       help="Use GPU for statistics (default: auto)")
    parser.add_argument("--use-cpu", action="store_true", 
                       help="Force CPU usage")
    parser.add_argument("-g", "--genome", default="hg38", 
                       help="Reference genome (default: hg38)")
    parser.add_argument("--context-length", type=int, default=10,
                       help="Context length around variants (default: 10)")
    parser.add_argument("--wildcard", choices=['T', 'G', 'C', 'A'], default=None,
                       help="Nucleotide to replace 'N' with")
    parser.add_argument("--n-jobs", type=int, default=None,
                       help="Number of parallel jobs (default: auto)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_cpu:
        args.use_gpu = False
    elif args.compute_stats and args.use_gpu is None:
        args.use_gpu = True
    elif not args.compute_stats:
        args.use_gpu = False
    
    if args.n_jobs is None:
        args.n_jobs = -1 if args.compute_stats else 1
    
    if args.use_gpu and not GPU_AVAILABLE:
        print("Warning: GPU not available, using CPU instead")
        args.use_gpu = False
    
    if args.compute_stats and args.cov_file is None:
        parser.error("--compute-stats requires --cov-file")
    
    # Determine if single model or batch
    model_path = args.model
    is_single_model = (Path(model_path).suffix == '.qbic' and 'weights' in Path(model_path).name)
    
    # Validate output directory
    if args.output_dir:
        output_path = Path(args.output)
        if output_path.exists() and not output_path.is_dir():
            parser.error(f"Output path '{args.output}' exists but is not a directory")
    
    start_time = time.time()
    
    try:
        # Input processing
        print("\nINPUT PROCESSING")
        print("-" * 50)
        
        print(f"Reading variants from: {args.variants}")
        if args.variants.endswith('.tsv') or args.variants.endswith('.txt'):
            variants_df = pd.read_csv(args.variants, sep='\t')
        else:
            variants_df = pd.read_csv(args.variants)
        
        print(f"Loaded {len(variants_df)} variants")
        
        # Configuration
        print("\nCONFIGURATION")
        print("-" * 50)
        print(f"   Statistics computation: {'Enabled' if args.compute_stats else 'Disabled'}")
        if args.compute_stats:
            if args.use_gpu:
                print(f"   Processing mode: GPU (sequential)")
                print(f"   GPU device: {get_gpu_info()}")
            else:
                print(f"   Processing mode: CPU (parallel)")
                print(f"   CPU cores: {args.n_jobs if args.n_jobs != -1 else 'All available'}")
        else:
            print(f"   Processing mode: CPU (predictions only)")
        print(f"   Output mode: {'Directory' if args.output_dir else 'Single file'}")
        print(f"   N handling: {'Replace with ' + args.wildcard if args.wildcard else 'Return NA'}")
        
        # Sequence processing
        print("\nSEQUENCE PROCESSING")
        print("-" * 50)
        
        has_sequences = ('ref_sequence' in variants_df.columns and 
                        'alt_sequence' in variants_df.columns)
        
        if not has_sequences:
            print(f"Extracting sequences from genome: {args.genome}")
            variants_df = extract_sequences_from_variants(
                variants_df, context_length=args.context_length, genome=args.genome
            )
            print("Sequence extraction completed")
        else:
            print("Using pre-extracted sequences from input")
        
        n_count = count_sequences_with_n(variants_df)
        if n_count > 0:
            print(f"Found {n_count} sequences containing 'N' characters")
        else:
            print("No sequences contain 'N' characters")
        
        # Prediction processing
        print("\nPREDICTION PROCESSING")
        print("-" * 50)
        
        if is_single_model:
            print(f"Processing single model: {Path(model_path).name}")
            results = predict_single_model(
                model_path, variants_df, args.compute_stats, 
                args.cov_file, args.use_gpu, args.genome, args.wildcard, args.n_jobs
            )
        else:
            print(f"Processing models from file: {model_path}")
            results = predict_batch_models(
                model_path, variants_df, args.compute_stats,
                args.cov_file, args.use_gpu, args.genome, args.wildcard, args.n_jobs
            )
        
        # Output processing
        print("\nOUTPUT PROCESSING")
        print("-" * 50)
        
        if args.output_dir:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if is_single_model:
                model_name = extract_model_name(model_path)
                output_file = output_dir / f"{model_name}.csv"
                results.to_csv(output_file, index=False)
                print(f"Saved results to: {output_file}")
            else:
                for model_name in results['model'].unique():
                    model_results = results[results['model'] == model_name]
                    output_file = output_dir / f"{model_name}.csv"
                    model_results.to_csv(output_file, index=False)
                    print(f"Saved {model_name}: {output_file.name}")
        else:
            print(f"Saving results to: {args.output}")
            results.to_csv(args.output, index=False)
        
        # Summary
        elapsed_time = time.time() - start_time
        print(f"Successfully processed {len(results)} predictions in {elapsed_time:.2f} seconds")
        
        if args.compute_stats and 'z_score' in results.columns:
            valid_stats = results['z_score'].notna().sum()
            print(f"Computed statistics for {valid_stats} variants")
        
        # Save error log
        if _error_collector:
            combined_error_file = save_combined_error_log()
            if combined_error_file:
                print(f"Combined error report saved to: {combined_error_file}")
        
        print("\n" + "╔══════════════════════════════════════════════════════════════════════════════╗")
        print("║                              Analysis Complete!                              ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝")
        print("For questions, contact Shengyu Li at shengyu.li@duke.edu")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        error_details = f"Main execution error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        save_error_log("MAIN_EXECUTION_ERROR", error_details)
        print(f"\nError: {str(e)}")
        print("Detailed error information saved to error report")
        sys.exit(1)

if __name__ == "__main__":
    main()
