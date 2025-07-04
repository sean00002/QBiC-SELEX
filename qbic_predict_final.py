#!/usr/bin/env python3
"""
QBIC-SELEX Variant Effect Prediction Script
==========================================

Predict variant effects using QBIC-SELEX models with optional statistical significance computation.
Supports both single model and batch processing.

Features:
- Support for multiple models
- Parallel processing for speed
- Automatic sequence extraction from reference genomes
- Optional p-value and z-score computation by GPU default, CPU fallback
- Flexible N handling (NA by default, configurable replacement)
- Caching for repeated operations

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
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import time
import traceback
from datetime import datetime
import signal

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    warnings.warn("tqdm not available. Progress bars will be disabled.")

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import sequence_utils as utils
import extract_seq

# Optional GPU imports
try:
    import cupy as cp
    import cudf
    import cuml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    warnings.warn("GPU libraries (cupy, cudf, cuml) not available. Using CPU only.")

# Global cache for frequently used data
_model_cache = {}
_cov_cache = {}

# Global error collector for combining all errors
_error_collector = []

def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Statistics computation timed out")

def run_with_timeout(func, timeout_seconds=300, *args, **kwargs):
    """
    Run a function with a timeout.
    
    Args:
        func: Function to run
        timeout_seconds: Timeout in seconds (default: 5 minutes)
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Result of the function
        
    Raises:
        TimeoutError: If the function times out
    """
    # Set up signal handler for timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Cancel the alarm
        return result
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # Restore original handler

def save_error_log(error_type: str, details: str, output_dir: str = ".", append: bool = False, model_name: str = None):
    """
    Save detailed error information to the global error collector for combined reporting.
    
    Args:
        error_type: Type of error (e.g., 'GPU_ERROR', 'MODEL_ERROR')
        details: Detailed error information
        output_dir: Directory to save error logs (not used for individual files)
        append: Whether to append to existing log or create new one (not used)
        model_name: Name of the model associated with this error
    """
    # Add to global error collector only
    error_entry = {
        'timestamp': datetime.now().isoformat(),
        'type': error_type,
        'model': model_name,
        'details': details
    }
    _error_collector.append(error_entry)

def save_combined_error_log(output_dir: str = "."):
    """
    Save all collected errors to a single combined log file.
    
    Args:
        output_dir: Directory to save the combined error log
    """
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

def combine_error_details(error_details: List[str]) -> str:
    """
    Combine error details to show counts and essential information only.
    
    Args:
        error_details: List of error detail strings
        
    Returns:
        Combined error report with counts and essential details
    """
    if not error_details:
        return "No errors"
    
    # Count errors by type
    error_counts = {}
    for error in error_details:
        if "Variant" in error:
            # Extract error type from variant errors
            parts = error.split(":", 2)
            if len(parts) >= 3:
                error_type = parts[1].strip()
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            else:
                error_counts["Unknown"] = error_counts.get("Unknown", 0) + 1
        elif "Model" in error:
            error_counts["Model Error"] = error_counts.get("Model Error", 0) + 1
        else:
            error_counts["Other"] = error_counts.get("Other", 0) + 1
    
    # Build report
    combined = []
    
    # Show error counts
    total_errors = len(error_details)
    combined.append(f"Total errors: {total_errors}")
    for error_type, count in error_counts.items():
        combined.append(f"  {error_type}: {count}")
    
    combined.append("")  # Empty line
    
    # Show essential error details only
    combined.append("Error Details:")
    combined.append("=" * 80)
    for i, error in enumerate(error_details, 1):
        combined.append(f"Case {i}:")
        combined.append(error)
        if i < len(error_details):
            combined.append("-" * 40)
    
    return "\n".join(combined)

def get_gpu_info() -> str:
    """Get GPU information if available."""
    if not GPU_AVAILABLE:
        return "Not available"
    
    try:
        # Try to get GPU device info
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

def convert_kmers_to_c_vectorized(alt_kmers: List[str], ref_kmers: List[str], 
                                 partial_dict: Dict[str, float]) -> np.ndarray:
    """
    Convert k-mers to coefficient vector using partial dictionary (non-reverse complement only).
    Handles k-mers not in partial dict by trying their reverse complements.
    
    Args:
        alt_kmers: List of alternate k-mers
        ref_kmers: List of reference k-mers  
        partial_dict: Partial model dictionary (non-reverse complement only)
        
    Returns:
        Coefficient vector for statistical testing
    """
    k = len(list(partial_dict.keys())[0])
    
    # Create lookup arrays for faster indexing
    partial_keys = list(partial_dict.keys())
    key_to_idx = {key: idx for idx, key in enumerate(partial_keys)}
    
    output = np.zeros(len(partial_dict), dtype=np.int8)
    
    # Process alternate k-mers
    for kmer in alt_kmers:
        # First try the k-mer as-is
        if kmer in key_to_idx:
            output[key_to_idx[kmer]] += 1
        else:
            # Try reverse complement
            rc_kmer = utils.reverse_complement(kmer)
            if rc_kmer in key_to_idx:
                output[key_to_idx[rc_kmer]] += 1
            # If neither exists, skip (this k-mer doesn't contribute to statistics)
    
    # Process reference k-mers
    for kmer in ref_kmers:
        # First try the k-mer as-is
        if kmer in key_to_idx:
            output[key_to_idx[kmer]] -= 1
        else:
            # Try reverse complement
            rc_kmer = utils.reverse_complement(kmer)
            if rc_kmer in key_to_idx:
                output[key_to_idx[rc_kmer]] -= 1
            # If neither exists, skip (this k-mer doesn't contribute to statistics)
    
    return output

def compute_p_value_cpu(alt_kmers: List[str], ref_kmers: List[str], 
                       partial_dict: Dict[str, float], cov_matrix: np.ndarray) -> Tuple[float, float]:
    """Compute p-value and z-score using CPU."""
    c = convert_kmers_to_c_vectorized(alt_kmers, ref_kmers, partial_dict)
    
    # Use pre-computed values for efficiency
    lm_values = np.array(list(partial_dict.values()), dtype=np.float32)
    
    estimate = np.dot(c.T, lm_values)
    
    # Check for zero effect (estimate = 0)
    if estimate == 0:
        # Zero effect: z-score = 0, p-value = 1
        # This avoids division by zero when standard error = 0
        # A zero effect means no difference between ref and alt sequences
        return 0.0, 1.0
    
    se_estimate = np.sqrt(np.dot(np.dot(c.T, cov_matrix), c))
    
    t_score = estimate / se_estimate
    p_value = 2 * stats.norm.sf(np.abs(t_score))
    
    return float(t_score), float(p_value)

def qbic_predict_vectorized(model_dict: Dict[str, float], alt_seqs: List[List[str]], 
                           ref_seqs: List[List[str]], wildcard: Optional[str] = None) -> Tuple[np.ndarray, int]:
    """
    Predict variant effects using QBIC model.
    
    Args:
        model_dict: Dictionary mapping k-mers to coefficients
        alt_seqs: List of alternate sequences
        ref_seqs: List of reference sequences
        wildcard: Nucleotide to replace 'N' with (T, G, C, A, or None for NA)
        
    Returns:
        Tuple of (array of predicted effect sizes, number of NA sequences)
    """
    # Check if sequences contain 'N' and wildcard is None
    na_count = 0
    if wildcard is None:
        for alt_seq, ref_seq in zip(alt_seqs, ref_seqs):
            if any('N' in kmer for kmer in alt_seq) or any('N' in kmer for kmer in ref_seq):
                na_count += 1
        
        if na_count > 0:
            # Return NA for all predictions if any sequence contains N
            return np.full(len(alt_seqs), np.nan), na_count
    
    # Use optimized approach while maintaining same results
    replacement = wildcard if wildcard is not None else 'A'
    
    # Pre-compute model values for faster lookup
    model_values = np.array(list(model_dict.values()), dtype=np.float32)
    model_keys = list(model_dict.keys())
    key_to_idx = {key: idx for idx, key in enumerate(model_keys)}
    
    # Compute predictions (silent processing)
    predictions = np.zeros(len(alt_seqs), dtype=np.float32)
    
    for i, (alt_seq, ref_seq) in enumerate(zip(alt_seqs, ref_seqs)):
        # Replace N and compute sums
        alt_sum = 0.0
        ref_sum = 0.0
        
        for kmer in alt_seq:
            clean_kmer = kmer.replace('N', replacement)
            if clean_kmer in key_to_idx:
                alt_sum += model_values[key_to_idx[clean_kmer]]
        
        for kmer in ref_seq:
            clean_kmer = kmer.replace('N', replacement)
            if clean_kmer in key_to_idx:
                ref_sum += model_values[key_to_idx[clean_kmer]]
        
        predictions[i] = alt_sum - ref_sum
    
    return predictions, na_count

def process_variants_vectorized(df: pd.DataFrame, k: int) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Process variant data to extract k-mers for prediction.
    
    Args:
        df: DataFrame with ref_sequence and alt_sequence columns
        k: k-mer size
        
    Returns:
        Tuple of (alt_kmer_lists, ref_kmer_lists)
    """
    if not all(col in df.columns for col in ['ref_sequence', 'alt_sequence']):
        raise ValueError("DataFrame must contain 'ref_sequence' and 'alt_sequence' columns")
    
    # Extract k-mers using cached sliding window (silent processing)
    alt_list = [list(sliding_window(seq, k)) for seq in df['alt_sequence']]
    ref_list = [list(sliding_window(seq, k)) for seq in df['ref_sequence']]
    
    return alt_list, ref_list

def load_model_cached(model_path: str) -> Tuple[Dict[str, float], int, Dict[str, float]]:
    """
    Load QBIC model from file with caching.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Tuple of (complete_dict, kmer_size, partial_dict)
        - complete_dict: Complete dictionary with reverse complements (for predictions)
        - kmer_size: Size of k-mers in the model
        - partial_dict: Partial dictionary with non-reverse complement k-mers (for statistics)
    """
    if model_path in _model_cache:
        return _model_cache[model_path]
    
    with open(model_path, 'rb') as f:
        partial_dict = pickle.load(f)
    
    # Recover full dictionary including reverse complements
    complete_dict = utils.recover_non_rc_dict(partial_dict)
    kmer_size = len(list(complete_dict.keys())[0])
    
    # Cache the result
    _model_cache[model_path] = (complete_dict, kmer_size, partial_dict)
    
    return complete_dict, kmer_size, partial_dict

def load_covariance_matrix_cached(cov_path: str) -> np.ndarray:
    """Load covariance matrix from file with caching."""
    if cov_path in _cov_cache:
        return _cov_cache[cov_path]
    
    with open(cov_path, 'rb') as f:
        cov_matrix = pickle.load(f)
    
    # Cache the result
    _cov_cache[cov_path] = cov_matrix
    
    return cov_matrix

def extract_sequences_from_variants(variants_df: pd.DataFrame, 
                                  context_length: int = 20,
                                  genome: str = "hg38") -> pd.DataFrame:
    """
    Extract reference and alternate sequences for variants.
    
    Args:
        variants_df: DataFrame with variant information (chrom, pos, ref, alt) OR sequences (ref_sequence, alt_sequence)
        context_length: Number of base pairs on each side of variant
        genome: Reference genome name or path
        
    Returns:
        DataFrame with added ref_sequence and alt_sequence columns
    """
    # Check if sequences are already present
    if 'ref_sequence' in variants_df.columns and 'alt_sequence' in variants_df.columns:
        print("Sequences already present in input, skipping extraction")
        return variants_df
    
    # Check if variant information is present for extraction
    required_cols = ['chrom', 'pos', 'ref', 'alt']
    if not all(col in variants_df.columns for col in required_cols):
        raise ValueError(f"Either sequences (ref_sequence, alt_sequence) or variant information ({', '.join(required_cols)}) must be provided")
    
    return extract_seq.batch_extract_sequences(
        variants_df,
        context_length=context_length,
        genome_file=genome
    )

def compute_stats_parallel(args_tuple):
    """
    Worker function for parallel statistical computation (CPU only).
    
    Args:
        args_tuple: Tuple containing (variant_idx, alt_kmers, ref_kmers, partial_dict, cov_matrix, use_gpu, variant_info, na_count)
        
    Returns:
        Tuple of (variant_idx, t_score, p_value)
    """
    variant_idx, alt_kmers, ref_kmers, partial_dict, cov_matrix, use_gpu, variant_info, na_count = args_tuple
    
    try:
        # Only use CPU processing in parallel workers
        t_score, p_value = compute_p_value_cpu(alt_kmers, ref_kmers, partial_dict, cov_matrix)
        return variant_idx, t_score, p_value
    except Exception as e:
        # Create concise error message with essential variant information
        variant_details = get_variant_details_simple(variant_idx, alt_kmers, ref_kmers, partial_dict, variant_info, na_count)
        error_msg = f"Variant {variant_idx}: {type(e).__name__}: {str(e)} | {variant_details}"
        save_error_log("STATS_COMPUTATION_ERROR", error_msg)
        return variant_idx, np.nan, np.nan

def compute_statistics_gpu(alt_list: List[List[str]], ref_list: List[List[str]], 
                          partial_dict: Dict[str, float], cov_matrix: np.ndarray,
                          model_name: str = None, variants_df: pd.DataFrame = None, na_count: int = 0) -> Tuple[List[float], List[float]]:
    """
    Compute statistics efficiently using GPU processing.
    
    Args:
        alt_list: List of alternate k-mer lists
        ref_list: List of reference k-mer lists
        partial_dict: Partial model dictionary (non-reverse complement only)
        cov_matrix: Covariance matrix (computed from partial dictionary)
        model_name: Name of the model being processed
        variants_df: DataFrame with variant information for error reporting
        na_count: Number of sequences with NA values
        
    Returns:
        Tuple of (t_scores, p_values)
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available for processing")
    
    # Pre-compute model values for efficiency
    lm_values = list(partial_dict.values())
    lm_values_cp = cp.array(lm_values, dtype=cp.float32)
    cov_matrix_cp = cp.asarray(cov_matrix, dtype=cp.float32)
    
    # Process all variants
    t_scores = []
    p_values = []
    error_count = 0
    zero_effect_count = 0
    error_details = []
    
    # Prepare variant information for error reporting
    variant_info = {}
    if variants_df is not None:
        for i in range(len(variants_df)):
            row = variants_df.iloc[i]
            info = {}
            if 'chrom' in variants_df.columns and 'pos' in variants_df.columns and 'ref' in variants_df.columns and 'alt' in variants_df.columns:
                info['variant'] = f"{row['chrom']}:{row['pos']} {row['ref']}>{row['alt']}"
            if 'ref_sequence' in variants_df.columns:
                info['ref_seq'] = row['ref_sequence']
            if 'alt_sequence' in variants_df.columns:
                info['alt_seq'] = row['alt_sequence']
            variant_info[i] = info
    
    if TQDM_AVAILABLE:
        iterator = tqdm(zip(alt_list, ref_list), total=len(alt_list), desc="Computing statistics", unit="variant")
    else:
        iterator = zip(alt_list, ref_list)
    
    for i, (alt_kmers, ref_kmers) in enumerate(iterator):
        try:
            # Convert to coefficient vector using partial dictionary
            c = convert_kmers_to_c_vectorized(alt_kmers, ref_kmers, partial_dict)
            
            c_cp = cp.array(c, dtype=cp.float32)
            
            # GPU computation with proper vector operations
            estimate = cp.dot(c_cp, lm_values_cp)
            se_estimate = cp.sqrt(cp.dot(c_cp, cp.dot(cov_matrix_cp, c_cp)))
            
            # Check for zero effect (estimate = 0)
            estimate_np = float(cp.asnumpy(estimate))
            if estimate_np == 0:
                # Zero effect: z-score = 0, p-value = 1
                # This avoids division by zero when standard error = 0
                # A zero effect means no difference between ref and alt sequences
                t_scores.append(0.0)
                p_values.append(1.0)
                zero_effect_count += 1
                continue
            
            # Check for division by zero or invalid values
            if se_estimate == 0 or cp.isnan(se_estimate) or cp.isinf(se_estimate):
                raise ValueError(f"Invalid standard error: {float(cp.asnumpy(se_estimate))}")
            
            t_score = estimate / se_estimate
            t_score_np = float(cp.asnumpy(t_score))
            
            # Check for invalid t-score
            if np.isnan(t_score_np) or np.isinf(t_score_np):
                raise ValueError(f"Invalid t-score: {t_score_np}")
            
            p_value = 2 * stats.norm.sf(np.abs(t_score_np))
            
            t_scores.append(t_score_np)
            p_values.append(p_value)
            
        except Exception as e:
            error_count += 1
            # Create concise error message with essential variant information
            variant_details = get_variant_details_simple(i, alt_kmers, ref_kmers, partial_dict, variant_info, na_count)
            error_msg = f"Variant {i}: {type(e).__name__}: {str(e)} | {variant_details}"
            error_details.append(error_msg)
            t_scores.append(np.nan)
            p_values.append(np.nan)
    
    # Clean up GPU memory
    del lm_values_cp, cov_matrix_cp
    cp.get_default_memory_pool().free_all_blocks()
    
    if error_count > 0:
        print(f"Warning: {error_count}/{len(alt_list)} variants failed during statistics computation")
        # Save concise error log
        error_log = f"GPU Processing Errors\n\nCombined error summary:\n" + combine_error_details(error_details)
        save_error_log("GPU_PROCESSING_ERROR", error_log, model_name=model_name)
    
    if zero_effect_count > 0:
        print(f"Note: {zero_effect_count} variants had zero predicted effects (z-score=0, p-value=1)")
    
    valid_count = len([x for x in t_scores if not np.isnan(x)])
    print(f"Statistics computed successfully: {valid_count}/{len(t_scores)} variants")
    
    return t_scores, p_values

def compute_statistics_parallel(alt_list: List[List[str]], ref_list: List[List[str]], 
                              partial_dict: Dict[str, float], cov_matrix: np.ndarray,
                              use_gpu: bool = False, n_jobs: int = -1,
                              model_name: str = None, variants_df: pd.DataFrame = None, na_count: int = 0) -> Tuple[List[float], List[float]]:
    """
    Compute statistics efficiently using GPU sequential processing or parallel CPU.
    
    Args:
        alt_list: List of alternate k-mer lists
        ref_list: List of reference k-mer lists
        partial_dict: Partial model dictionary (non-reverse complement only)
        cov_matrix: Covariance matrix (computed from partial dictionary)
        use_gpu: Whether to use GPU (sequential processing only)
        n_jobs: Number of parallel jobs (-1 for all cores, only used for CPU)
        model_name: Name of the model being processed
        variants_df: DataFrame with variant information for error reporting
        na_count: Number of sequences with NA values
        
    Returns:
        Tuple of (t_scores, p_values)
    """
    # Use GPU sequential processing when available
    if use_gpu and GPU_AVAILABLE:
        try:
            t_scores, p_values = compute_statistics_gpu(
                alt_list, ref_list, partial_dict, cov_matrix, model_name, variants_df, na_count
            )
            
            # Check if GPU processing returned valid results
            valid_count = len([x for x in t_scores if not np.isnan(x)])
            if valid_count == 0:
                error_msg = "GPU processing failed - no valid results obtained"
                save_error_log("GPU_NO_RESULTS", f"GPU processing returned all NaN values\nModel: {model_name}\nVariants: {len(variants_df)}", append=True, model_name=model_name)
                print(f"Error: {error_msg}")
                print("Use --use-cpu instead of --use-gpu")
                raise RuntimeError(error_msg)
            
            return t_scores, p_values
            
        except Exception as e:
            error_details = f"GPU processing failed: {str(e)}\nModel: {model_name}\nVariants: {len(variants_df)}"
            save_error_log("GPU_PROCESSING_ERROR", error_details, model_name=model_name)
            print(f"Error: GPU processing failed: {str(e)}")
            print("Use --use-cpu instead of --use-gpu")
            raise RuntimeError(f"GPU processing failed: {str(e)} - please use CPU instead")
    
    # CPU parallel processing
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    if n_jobs == 1:
        print(f"Using single CPU processing")
        print()  # Add gap
        
        # Sequential processing for single CPU
        t_scores = []
        p_values = []
        for i, (alt_kmers, ref_kmers) in enumerate(zip(alt_list, ref_list)):
            try:
                t_score, p_value = compute_p_value_cpu(alt_kmers, ref_kmers, partial_dict, cov_matrix)
                t_scores.append(t_score)
                p_values.append(p_value)
            except Exception as e:
                # Create concise error message with essential variant information
                variant_details = get_variant_details_simple(i, alt_kmers, ref_kmers, partial_dict, {}, na_count)
                error_msg = f"Variant {i}: {type(e).__name__}: {str(e)} | {variant_details}"
                save_error_log("STATS_COMPUTATION_ERROR", error_msg)
                t_scores.append(np.nan)
                p_values.append(np.nan)
    else:
        print(f"Using {n_jobs} parallel CPU processes")
        print()  # Add gap
        
        # Prepare variant information dictionary for parallel processing
        variant_info = {}
        if variants_df is not None:
            for i in range(len(variants_df)):
                row = variants_df.iloc[i]
                info = {}
                if 'chrom' in variants_df.columns and 'pos' in variants_df.columns and 'ref' in variants_df.columns and 'alt' in variants_df.columns:
                    info['variant'] = f"{row['chrom']}:{row['pos']} {row['ref']}>{row['alt']}"
                if 'ref_sequence' in variants_df.columns:
                    info['ref_seq'] = row['ref_sequence']
                if 'alt_sequence' in variants_df.columns:
                    info['alt_seq'] = row['alt_sequence']
                variant_info[i] = info
        
        # Prepare arguments for parallel processing
        args_list = []
        for i, (alt_kmers, ref_kmers) in enumerate(zip(alt_list, ref_list)):
            args_list.append((i, alt_kmers, ref_kmers, partial_dict, cov_matrix, False, variant_info, na_count))
        
        # Process in parallel
        t_scores = [np.nan] * len(alt_list)
        p_values = [np.nan] * len(alt_list)
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all jobs
            future_to_idx = {executor.submit(compute_stats_parallel, args): args[0] for args in args_list}
            
            # Collect results as they complete with progress bar
            if TQDM_AVAILABLE:
                futures_iterator = tqdm(as_completed(future_to_idx), total=len(args_list), desc="Computing statistics", unit="variant")
            else:
                futures_iterator = as_completed(future_to_idx)
            
            for future in futures_iterator:
                variant_idx, t_score, p_value = future.result()
                t_scores[variant_idx] = t_score
                p_values[variant_idx] = p_value
    
    return t_scores, p_values

def predict_single_model(model_path: str, variants_df: pd.DataFrame, 
                        compute_stats: bool = False, cov_file: Optional[str] = None,
                        use_gpu: bool = True, genome: str = "hg38", 
                        wildcard: Optional[str] = None, n_jobs: int = -1) -> pd.DataFrame:
    """
    Predict variant effects using a single model.
    
    Args:
        model_path: Path to model file
        variants_df: DataFrame with variant information
        compute_stats: Whether to compute p-values and z-scores
        cov_file: Path to covariance matrix file (.cov.qbic) or text file with covariance paths
        use_gpu: Whether to use GPU for statistical computations
        genome: Reference genome name or path
        wildcard: Nucleotide to replace 'N' with (T, G, C, A, or None for NA)
        n_jobs: Number of parallel jobs for statistical computations (-1 for all cores)
        
    Returns:
        DataFrame with predictions
    """
    model_name = Path(model_path).stem.replace('.weights', '')
    
    print(f"Processing model: {model_name}")
    
    # Load model with caching
    complete_dict, kmer_size, partial_dict = load_model_cached(model_path)
    
    # Process variants
    alt_list, ref_list = process_variants_vectorized(variants_df, kmer_size)
    
    # Make predictions
    print(f"Predicting variant effects...")
    predictions, na_count = qbic_predict_vectorized(complete_dict, alt_list, ref_list, wildcard)
    print(f"Predictions completed")
    
    # Create results DataFrame - always include sequences
    results = variants_df.copy()
    results['model'] = model_name
    results['predicted_effect'] = predictions
    
    # Ensure sequences are always in output (in case they were missing from input)
    if 'ref_sequence' not in results.columns:
        results['ref_sequence'] = [''] * len(results)  # Placeholder if not available
    if 'alt_sequence' not in results.columns:
        results['alt_sequence'] = [''] * len(results)  # Placeholder if not available
    
    # Compute statistics if requested
    if compute_stats:
        print(f"Computing statistics...")
        if cov_file is None:
            raise ValueError(f"Covariance file is required when --compute-stats is used")
        
        # Determine covariance matrix path
        cov_path = None
        if Path(cov_file).suffix == '.qbic' and 'cov' in Path(cov_file).name:
            # Direct covariance matrix file
            cov_path = cov_file
            print(f"Using direct covariance matrix: {Path(cov_file).name}")
        else:
            # Text file with covariance paths - find matching covariance matrix
            print(f"Reading covariance paths from: {cov_file}")
            cov_paths = read_covariance_paths_from_file(cov_file)
            if model_name in cov_paths:
                cov_path = cov_paths[model_name]
                print(f"Found covariance matrix for {model_name}: {Path(cov_path).name}")
            else:
                raise ValueError(f"Covariance matrix not found for model {model_name} in {cov_file}")
        
        # Load covariance matrix with caching
        print(f"Loading covariance matrix...")
        cov_matrix = load_covariance_matrix_cached(cov_path)
        print(f"Covariance matrix loaded: {cov_matrix.shape}")
        
        # Use the new function that handles zero effects properly
        try:
            print(f"Starting statistics computation...")
            # Use timeout wrapper to prevent getting stuck
            t_scores, p_values = run_with_timeout(
                compute_statistics_parallel,
                600,  # timeout_seconds: 10 minutes timeout
                alt_list, ref_list, partial_dict, cov_matrix, 
                use_gpu, n_jobs, model_name, variants_df, na_count
            )
            results['z_score'] = t_scores
            results['p_value'] = p_values
            print(f"Statistics completed")
        except TimeoutError as e:
            error_details = f"Statistics computation timed out for {model_name}\nError: {str(e)}\nModel: {model_path}\nVariants: {len(variants_df)}"
            save_error_log("STATS_TIMEOUT_ERROR", error_details, model_name=model_name)
            print(f"Statistics computation timed out for {model_name} - continuing without statistics")
            # Continue without statistics - predictions are still valid
        except Exception as e:
            error_details = f"Statistics computation failed for {model_name}\nError: {str(e)}\nModel: {model_path}\nVariants: {len(variants_df)}"
            save_error_log("STATS_COMPUTATION_ERROR", error_details, model_name=model_name)
            print(f"Statistics computation failed for {model_name} - continuing without statistics")
            # Continue without statistics - predictions are still valid
    
    print(f"Model {model_name} completed")
    print()  # Add gap between models
    
    return results

def process_single_model_parallel(args_tuple):
    """
    Worker function for parallel model processing.
    
    Args:
        args_tuple: Tuple containing (model_path, variants_df, compute_stats, cov_file, use_gpu, genome, wildcard, n_jobs)
        
    Returns:
        DataFrame with predictions for this model
    """
    model_path, variants_df, compute_stats, cov_file, use_gpu, genome, wildcard, n_jobs = args_tuple
    
    try:
        return predict_single_model(
            model_path, variants_df, compute_stats, cov_file, use_gpu, genome, wildcard, n_jobs
        )
    except Exception as e:
        model_name = Path(model_path).stem.replace('.weights', '')  # Use stem, not name
        print(f"Error processing model {model_name}: {e}")
        return None

def read_model_paths_from_file(models_file: str) -> List[str]:
    """
    Read model file paths from a text file.
    
    Args:
        models_file: Path to text file containing model paths (one per line)
        
    Returns:
        List of model file paths
    """
    model_paths = []
    missing_files = []
    
    with open(models_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                # Check if model file exists
                if not Path(line).exists():
                    missing_files.append((line_num, line))
                else:
                    model_paths.append(line)
    
    # Report missing files
    if missing_files:
        print(f"Warning: {len(missing_files)} model files not found (will be skipped)")
        # Save detailed missing files log
        missing_details = f"Missing model files in {models_file}:\n" + "\n".join([f"Line {line_num}: {file_path}" for line_num, file_path in missing_files])
        save_error_log("MISSING_MODEL_FILES", missing_details)
    
    if not model_paths:
        raise ValueError(f"No valid model paths found in {models_file}")
    
    return model_paths

def read_covariance_paths_from_file(cov_file: str) -> Dict[str, str]:
    """
    Read covariance matrix file paths from a text file.
    
    Args:
        cov_file: Path to text file containing covariance matrix paths (one per line)
        
    Returns:
        Dictionary mapping model names to covariance matrix paths
    """
    cov_paths = {}
    missing_files = []
    
    with open(cov_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                # Expect format: model_path,covariance_path or just covariance_path
                if ',' in line:
                    model_path, cov_path = line.split(',', 1)
                    model_name = Path(model_path.strip()).stem.replace('.weights', '')
                    cov_path = cov_path.strip()
                else:
                    # If no comma, assume it's just a covariance path
                    cov_path = line
                    model_name = Path(cov_path).stem.replace('.cov', '')
                
                # Check if covariance file exists
                if not Path(cov_path).exists():
                    missing_files.append((line_num, cov_path))
                else:
                    cov_paths[model_name] = cov_path
    
    # Report missing files
    if missing_files:
        print(f"Warning: {len(missing_files)} covariance matrix files not found (will be skipped)")
        # Save detailed missing files log
        missing_details = f"Missing covariance files in {cov_file}:\n" + "\n".join([f"Line {line_num}: {file_path}" for line_num, file_path in missing_files])
        save_error_log("MISSING_COVARIANCE_FILES", missing_details)
    
    return cov_paths

def check_sequences_for_n(variants_df: pd.DataFrame, wildcard: Optional[str] = None) -> int:
    """
    Pre-check sequences for 'N' characters and report summary.
    
    Args:
        variants_df: DataFrame with variant sequences
        wildcard: Nucleotide to replace 'N' with (T, G, C, A, or None for NA)
        
    Returns:
        Number of sequences containing 'N' characters
    """
    if wildcard is not None:
        # If wildcard is specified, no need to check for N
        return 0
    
    # Check if sequences are present
    if 'ref_sequence' not in variants_df.columns or 'alt_sequence' not in variants_df.columns:
        # If sequences not present yet, we can't check
        return 0
    
    na_count = 0
    for _, row in variants_df.iterrows():
        ref_seq = row['ref_sequence']
        alt_seq = row['alt_sequence']
        if 'N' in ref_seq or 'N' in alt_seq:
            na_count += 1
    
    return na_count

def validate_batch_inputs(models_file: str, variants_df: pd.DataFrame, 
                         compute_stats: bool = False, cov_file: Optional[str] = None,
                         wildcard: Optional[str] = None) -> Tuple[List[str], Dict[str, str]]:
    """
    Validate batch processing inputs and perform pre-checks.
    
    Args:
        models_file: Text file containing paths to model files
        variants_df: DataFrame with variant information
        compute_stats: Whether to compute p-values and z-scores
        cov_file: Path to covariance matrix file or text file with covariance paths
        wildcard: Nucleotide to replace 'N' with (T, G, C, A, or None for NA)
        
    Returns:
        Tuple of (model_paths, cov_paths)
    """
    print("\nVALIDATION")
    print("-" * 50)
    
    # Read model paths from file
    model_paths = read_model_paths_from_file(models_file)
    print(f"✓ Found {len(model_paths)} models for batch processing")
    
    # Check sequences for N characters
    na_count = check_sequences_for_n(variants_df, wildcard)
    if na_count > 0:
        print(f"⚠  Found {na_count} sequences containing 'N' characters")
        if wildcard is None:
            print("   These sequences will return NA predictions")
            print("   Use --wildcard T/G/C/A to replace 'N' with a specific nucleotide")
        else:
            print(f"   'N' will be replaced with '{wildcard}'")
    else:
        print("✓ No sequences contain 'N' characters")
    
    # Read covariance paths if provided
    cov_paths = {}
    if compute_stats and cov_file is not None:
        if Path(cov_file).suffix == '.qbic' and 'cov' in Path(cov_file).name:
            # Single covariance matrix file - use for all models
            print(f"✓ Using single covariance matrix: {Path(cov_file).name}")
            for model_path in model_paths:
                model_name = Path(model_path).stem.replace('.weights', '')
                cov_paths[model_name] = cov_file
        else:
            # Text file with covariance paths
            cov_paths = read_covariance_paths_from_file(cov_file)
            print(f"✓ Loaded {len(cov_paths)} covariance matrix paths")
    
    # Validate that all models have corresponding covariance matrices if stats are requested
    if compute_stats and cov_file is not None:
        missing_cov_models = []
        for model_path in model_paths:
            model_name = Path(model_path).stem.replace('.weights', '')
            if model_name not in cov_paths:
                missing_cov_models.append(model_name)
        
        if missing_cov_models:
            error_msg = f"Error: {len(missing_cov_models)} models missing covariance matrices for statistics computation"
            error_details = f"Missing covariance matrices for models:\n" + "\n".join([f"  - {model}" for model in missing_cov_models])
            save_error_log("MISSING_COVARIANCE_MATRICES", error_details)
            print(f"\n{error_msg}")
            print("Missing models:")
            for model in missing_cov_models:
                print(f"  - {model}")
            print("\nPlease ensure all models have corresponding covariance matrices in the covariance file.")
            print("Stopping execution.")
            raise RuntimeError(error_msg)
        else:
            print(f"✓ All {len(model_paths)} models have corresponding covariance matrices")
    
    print()  # Add gap after validation
    return model_paths, cov_paths

def predict_batch_models_parallel(models_file: str, variants_df: pd.DataFrame,
                                compute_stats: bool = False, cov_file: Optional[str] = None,
                                use_gpu: bool = True, genome: str = "hg38", 
                                wildcard: Optional[str] = None, n_jobs: int = -1) -> pd.DataFrame:
    """
    Predict variant effects using models specified in a text file with parallel processing.
    
    Args:
        models_file: Text file containing paths to model files (one per line)
        variants_df: DataFrame with variant information
        compute_stats: Whether to compute p-values and z-scores
        cov_file: Path to covariance matrix file (.cov.qbic) or text file with covariance paths
        use_gpu: Whether to use GPU for statistical computations
        genome: Reference genome name or path
        wildcard: Nucleotide to replace 'N' with (T, G, C, A, or None for NA)
        n_jobs: Number of parallel jobs (-1 for all cores)
        
    Returns:
        DataFrame with predictions from all models
    """
    # Validate inputs and perform pre-checks
    model_paths, cov_paths = validate_batch_inputs(
        models_file, variants_df, compute_stats, cov_file, wildcard
    )
    
    print(f"Processing {len(model_paths)} models...")
    print()  # Add gap before model processing starts
    
    # Prepare arguments for processing
    args_list = []
    
    for model_path in model_paths:
        # Determine covariance matrix path if needed
        cov_path = None
        if compute_stats and cov_file is not None:
            model_name = Path(model_path).stem.replace('.weights', '')
            cov_path = cov_paths[model_name]
        
        args_list.append((
            model_path, variants_df, compute_stats, cov_path, 
            use_gpu, genome, wildcard, n_jobs
        ))
    
    # Process models - use GPU sequential or CPU parallel
    all_results = []
    completed = 0
    failed_models = []
    error_details = []
    
    if use_gpu and GPU_AVAILABLE:
        # Sequential GPU processing for all models
        print(f"Using sequential GPU processing")
        print()  # Add gap
        
        for i, args in enumerate(args_list):
            model_path = args[0]
            model_name = Path(model_path).stem.replace('.weights', '')  # Use stem, not name
            try:
                result = predict_single_model(*args)
                if result is not None:
                    all_results.append(result)
                    completed += 1
                else:
                    failed_models.append(model_name)
                    error_details.append(f"Model {model_name}: Returned None result")
            except Exception as e:
                failed_models.append(model_name)
                error_details.append(f"Model {model_name}: {type(e).__name__}: {str(e)}")
    else:
        # Parallel CPU processing
        if n_jobs == -1:
            n_jobs = min(mp.cpu_count(), len(model_paths))
        
        if n_jobs == 1:
            print(f"Using single CPU processing")
            print()  # Add gap
            
            # Sequential processing for single CPU
            for i, args in enumerate(args_list):
                model_path = args[0]
                model_name = Path(model_path).stem.replace('.weights', '')  # Use stem, not name
                try:
                    result = predict_single_model(*args)
                    if result is not None:
                        all_results.append(result)
                        completed += 1
                    else:
                        failed_models.append(model_name)
                        error_details.append(f"Model {model_name}: Returned None result")
                except Exception as e:
                    failed_models.append(model_name)
                    error_details.append(f"Model {model_name}: {type(e).__name__}: {str(e)}")
        else:
            print(f"Using {n_jobs} parallel CPU processes")
            print()  # Add gap
            
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all jobs
                future_to_model = {executor.submit(process_single_model_parallel, args): args[0] for args in args_list}
                
                # Collect results as they complete with progress bar
                if TQDM_AVAILABLE:
                    futures_iterator = tqdm(as_completed(future_to_model), total=len(args_list), desc="Processing models", unit="model")
                else:
                    futures_iterator = as_completed(future_to_model)
                
                for future in futures_iterator:
                    model_path = future_to_model[future]
                    model_name = Path(model_path).stem.replace('.weights', '')  # Use stem, not name
                    try:
                        result = future.result()
                        if result is not None:
                            all_results.append(result)
                            completed += 1
                        else:
                            failed_models.append(model_name)
                            error_details.append(f"Model {model_name}: Returned None result")
                    except Exception as e:
                        failed_models.append(model_name)
                        error_details.append(f"Model {model_name}: {type(e).__name__}: {str(e)}")
    
    # Report summary
    if failed_models:
        print(f"Warning: {len(failed_models)} models failed out of {len(args_list)}")
        # Save concise error log
        error_log = f"Batch Processing Errors\n\nTotal models: {len(args_list)}\nFailed models: {len(failed_models)}\n\nCombined error summary:\n" + combine_error_details(error_details)
        save_error_log("BATCH_PROCESSING_ERROR", error_log)
    
    if not all_results:
        raise RuntimeError("No models were successfully processed")
    
    print(f"Combining results from {len(all_results)} models...")
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Clear caches to free memory
    _model_cache.clear()
    _cov_cache.clear()
    
    print(f"Batch processing complete! Total predictions: {len(combined_results)}")
    return combined_results

def get_variant_details_simple(variant_idx: int, alt_kmers: List[str], ref_kmers: List[str], 
                              partial_dict: Dict[str, float], variant_info: dict = None, na_count: int = 0) -> str:
    """
    Get essential information about a variant for error reporting.
    
    Args:
        variant_idx: Index of the variant
        alt_kmers: List of alternate k-mers
        ref_kmers: List of reference k-mers
        partial_dict: Partial model dictionary with k-mer weights
        variant_info: Dictionary with variant information
        na_count: Number of sequences with NA values
        
    Returns:
        Essential string with variant information
    """
    details = []
    
    # Add variant information if available
    if variant_info is not None and variant_idx in variant_info:
        info = variant_info[variant_idx]
        if 'variant' in info:
            details.append(f"Variant: {info['variant']}")
        if 'ref_seq' in info:
            details.append(f"Ref: {info['ref_seq']}")
        if 'alt_seq' in info:
            details.append(f"Alt: {info['alt_seq']}")
    
    # Add NA information if available
    if na_count > 0:
        details.append(f"NA sequences: {na_count}")
    
    result = " | ".join(details)
    
    return result

def main():
    """Main function with error handling."""
    
    # Print welcome message
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                    QBiC-SELEX Variant Effect Prediction                      ║")
    print("║              Quantitative, Bias-Corrected Modeling of TF Binding             ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    
    parser = argparse.ArgumentParser(
        description="Predict variant effects using QBIC models with efficient GPU/CPU processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Processing Modes:
  GPU Mode (default for statistics): Sequential processing using GPU for statistics computation
    - Default when --compute-stats is used (unless --use-cpu is specified)
    - Single model: All variants processed in one GPU batch
    - Multiple models: Models processed sequentially (one at a time)
    - Best for: Large datasets with statistics computation
    - Falls back to parallel CPU if GPU not available
  
  CPU Mode (default for predictions): Processing using CPU cores
    - Predictions only: Single CPU by default (fast and efficient)
    - Statistics computation: Parallel CPU when GPU not available or --use-cpu specified
    - Single model: Variants processed sequentially or in parallel
    - Multiple models: Models processed sequentially or in parallel
    - Best for: Predictions only, small datasets, shared computing environments

Examples:
  # Single model prediction with variant information (chrom,pos,ref,alt)
  python qbic_predict_final.py -v variants.csv -m model.weights.qbic -o results.csv
  
  # Single model prediction with pre-extracted sequences (ref_sequence,alt_sequence)
  python qbic_predict_final.py -v sequences.csv -m model.weights.qbic -o results.csv
  
  # Single model with statistics using GPU (default for statistics)
  python qbic_predict_final.py -v variants.csv -m model.weights.qbic -c model.cov.qbic -o results.csv --compute-stats
  
  # Single model with statistics using CPU (user specified)
  python qbic_predict_final.py -v variants.csv -m model.weights.qbic -c model.cov.qbic -o results.csv --compute-stats --use-cpu --n-jobs 4
  
  # Batch processing using models.txt file (single CPU by default for predictions)
  python qbic_predict_final.py -v variants.csv -m models.txt -o results.csv
  
  # Batch processing with parallel CPU (user specified)
  python qbic_predict_final.py -v variants.csv -m models.txt -o results.csv --n-jobs 4
  
  # Batch processing with statistics using GPU (default for statistics)
  python qbic_predict_final.py -v variants.csv -m models.txt -c covariance.txt -o results.csv --compute-stats
  
  # Batch processing with statistics using CPU (user specified)
  python qbic_predict_final.py -v variants.csv -m models.txt -c covariance.txt -o results.csv --compute-stats --use-cpu --n-jobs 8
  
  # Batch processing with statistics using single covariance file for all models
  python qbic_predict_final.py -v variants.csv -m models.txt -c model.cov.qbic -o results.csv --compute-stats --use-gpu
  
  # Force CPU usage with custom N replacement and 6 cores
  python qbic_predict_final.py -v variants.csv -m model.weights.qbic -c model.cov.qbic -o results.csv --compute-stats --use-cpu --wildcard T --n-jobs 6

Input Formats:
  Option 1: Variant information (chrom,pos,ref,alt) - sequences will be extracted
    chrom,pos,ref,alt
    chr1,1000,A,T
    chr2,5000,G,C

  Option 2: Pre-extracted sequences (ref_sequence,alt_sequence)
    ref_sequence,alt_sequence
    ATCGATCGATCGATCGATCG,TTCGATCGATCGATCGATCG
    GCTAGCTAGCTAGCTAGCTA,GCTAGCTAGCTAGCTAGCTA

File Formats:
  models.txt: One model path per line
    /path/to/model1.weights.qbic
    /path/to/model2.weights.qbic
    /path/to/model3.weights.qbic

  covariance.txt: One covariance path per line (optional: model_path,covariance_path)
    /path/to/model1.cov.qbic
    /path/to/model2.cov.qbic
    /path/to/model3.cov.qbic

Note: --cov-file is required when using --compute-stats. It can be either:
  - A direct covariance matrix file (.cov.qbic)
  - A text file containing covariance matrix paths
        """
    )
    
    # Input options - combined model argument that auto-detects
    parser.add_argument("-m", "--model", required=True, 
                       help="Path to single model file (.weights.qbic) or text file containing model paths (one per line)")
    
    # Required arguments
    parser.add_argument("-v", "--variants", required=True, 
                       help="Input variants file (CSV/TSV). Must contain either variant info (chrom,pos,ref,alt) or sequences (ref_sequence,alt_sequence)")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    
    # Optional arguments
    parser.add_argument("-c", "--cov-file", 
                       help="Path to covariance matrix file (.cov.qbic) or text file containing covariance matrix paths (required when --compute-stats)")
    parser.add_argument("--compute-stats", action="store_true", 
                       help="Compute p-values and z-scores (requires --cov-file)")
    parser.add_argument("--use-gpu", action="store_true", default=None,
                       help="Use GPU for statistical computations (default: True for statistics, False for predictions)")
    parser.add_argument("--use-cpu", action="store_true", 
                       help="Force CPU usage for statistical computations")
    parser.add_argument("-g", "--genome", default="hg38", 
                       help="Reference genome name or path (default: hg38)")
    parser.add_argument("--context-length", type=int, default=10,
                       help="Context length around variants in bp (default: 10)")
    parser.add_argument("--wildcard", choices=['T', 'G', 'C', 'A'], default=None,
                       help="Nucleotide to replace 'N' with in sequences (default: return NA for sequences with N)")
    parser.add_argument("--n-jobs", type=int, default=None,
                       help="Number of parallel jobs (None for single CPU, -1 for all cores, default: None for predictions, -1 for statistics)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_cpu:
        args.use_gpu = False
    elif not args.compute_stats:
        # Use CPU by default when compute-stats is not specified
        # GPU is only beneficial for statistics computation
        args.use_gpu = False
        print("Note: Using CPU processing (GPU is only beneficial for statistics computation)")
    elif args.compute_stats and args.use_gpu is None:
        # Use GPU by default for statistics computation
        args.use_gpu = True
        print("Note: Using GPU processing for statistics computation (use --use-cpu to force CPU)")
    
    # Set default n_jobs based on whether statistics are being computed
    if args.n_jobs is None:
        if args.compute_stats:
            # For statistics computation, use all cores by default
            args.n_jobs = -1
        else:
            # For predictions only, use single CPU by default
            args.n_jobs = 1
    
    if args.use_gpu and not GPU_AVAILABLE:
        print("Warning: GPU not available, using parallel CPU instead")
        args.use_gpu = False
    
    # Validate compute-stats requires cov-file
    if args.compute_stats and args.cov_file is None:
        parser.error("--compute-stats requires --cov-file to be specified")
    
    # Auto-detect if model input is single file or text file
    model_path = args.model
    if Path(model_path).suffix == '.qbic' and 'weights' in Path(model_path).name:
        # Single model file
        is_single_model = True
        models_file = None
    else:
        # Text file with model paths
        is_single_model = False
        models_file = model_path
    
    # Start timing
    start_time = time.time()
    
    try:
        print("\nINPUT PROCESSING")
        print("-" * 50)
        
        # Read variants file
        print(f"Reading variants from: {args.variants}")
        if args.variants.endswith('.tsv') or args.variants.endswith('.txt'):
            variants_df = pd.read_csv(args.variants, sep='\t')
        else:
            variants_df = pd.read_csv(args.variants)
        
        print(f"Loaded {len(variants_df)} variants")
        
        # Print configuration information
        print("\nCONFIGURATION")
        print("-" * 50)
        print(f"   Statistics computation: {'Enabled' if args.compute_stats else 'Disabled'}")
        if args.compute_stats:
            if args.use_gpu:
                print(f"   Processing mode: GPU (sequential)")
                gpu_info = get_gpu_info()
                print(f"   GPU device: {gpu_info}")
            else:
                print(f"   Processing mode: CPU (parallel)")
                print(f"   CPU cores: {args.n_jobs if args.n_jobs != -1 else 'All available'}")
        else:
            print(f"   Processing mode: CPU (predictions only)")
            if args.n_jobs == 1:
                print(f"   CPU cores: Single CPU")
            else:
                print(f"   CPU cores: {args.n_jobs if args.n_jobs != -1 else 'All available'}")
        if args.wildcard:
            print(f"   N replacement: {args.wildcard}")
        else:
            print(f"   N handling: Return NA for sequences with N")
        if TQDM_AVAILABLE:
            print(f"   Progress bars: Enabled")
        else:
            print(f"   Progress bars: Disabled (tqdm not available)")
        
        # Check if sequences are already present
        has_sequences = ('ref_sequence' in variants_df.columns and 
                        'alt_sequence' in variants_df.columns)
        
        print("\nSEQUENCE PROCESSING")
        print("-" * 50)
        if not has_sequences:
            print(f"Extracting sequences from genome: {args.genome}")
            variants_df = extract_sequences_from_variants(
                variants_df, context_length=args.context_length, genome=args.genome
            )
            print("Sequence extraction completed")
        else:
            print("Using pre-extracted sequences from input")
        
        # Make predictions
        print("\nPREDICTION PROCESSING")
        print("-" * 50)
        if is_single_model:
            # Single model prediction
            print(f"Processing single model: {Path(model_path).name}")
            results = predict_single_model(
                model_path, variants_df, args.compute_stats, 
                args.cov_file, args.use_gpu, args.genome, args.wildcard, args.n_jobs
            )
        else:
            # Batch model prediction
            print(f"Processing models from file: {models_file}")
            results = predict_batch_models_parallel(
                models_file, variants_df, args.compute_stats,
                args.cov_file, args.use_gpu, args.genome, args.wildcard, args.n_jobs
            )
        
        print()  # Add gap after prediction processing
        
        # Save results
        print("\nOUTPUT PROCESSING")
        print("-" * 50)
        print(f"Saving results to: {args.output}")
        results.to_csv(args.output, index=False)
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"Successfully processed {len(results)} predictions in {elapsed_time:.2f} seconds")
        
        if args.compute_stats:
            # Check if statistics columns exist (they might be missing if GPU failed)
            if 'z_score' in results.columns:
                valid_stats = results['z_score'].notna().sum()
                print(f"Computed statistics for {valid_stats} variants")
            else:
                print("Statistics columns not found - likely due to GPU processing failures")
                print("To get statistics, use --use-cpu instead of --use-gpu")
        
        # Save combined error log if any errors occurred
        if _error_collector:
            combined_error_file = save_combined_error_log()
            if combined_error_file:
                print(f"Combined error report saved to: {combined_error_file}")
        
        print("\n" + "╔══════════════════════════════════════════════════════════════════════════════╗")
        print("║                              Analysis Complete!                              ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝")
        print("Please refer to TF-to-model or model-to-TF tables for TF model mapping")
        print("For questions, contact Shengyu Li at shengyu.li@duke.edu")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        # Save detailed error log
        error_details = f"Main execution error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}\n\nArguments: {vars(args)}"
        save_error_log("MAIN_EXECUTION_ERROR", error_details)
        
        # Print brief error message
        print(f"\nError: {str(e)}")
        print("Detailed error information will be included in the combined error report")
        sys.exit(1)

if __name__ == "__main__":
    main()
