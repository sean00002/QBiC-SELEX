#!/usr/bin/env python3
"""
Extract reference and alternate sequences from genome FASTA files given variant information.
All sequences are returned in uppercase.
"""

import os
import pysam
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Union, Optional
import argparse

# Set default genome directory
DEFAULT_GENOME_DIR = "./genome"

def get_genome_path(genome_name: str) -> str:
    """
    Get the full path to a genome file.
    
    Args:
        genome_name: Name of the genome (e.g., 'hg38' or 'hg38.fa')
        
    Returns:
        Full path to the genome file
    """
    if os.path.exists(genome_name):
        return genome_name
        
    # Check if the full path is provided
    if not genome_name.endswith('.fa'):
        genome_name += '.fa'
        
    # Check in the default directory
    genome_path = os.path.join(DEFAULT_GENOME_DIR, genome_name)
    if os.path.exists(genome_path):
        return genome_path
        
    raise FileNotFoundError(f"Genome file not found: {genome_name} or {genome_path}")

def extract_sequence(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    context_length: int,
    genome_file: str
) -> Tuple[str, str]:
    """
    Extract reference and alternate sequences from a reference genome.
    All sequences are returned in uppercase.

    Args:
        chrom: Chromosome name (with or without 'chr' prefix)
        pos: 1-based position of the variant
        ref: Reference allele
        alt: Alternate allele
        context_length: Number of base pairs to include on each side of the variant
        genome_file: Path to the reference genome FASTA file or genome name (e.g., 'hg38')

    Returns:
        Tuple of (reference sequence, alternate sequence) in uppercase
    """
    # Get the full path to the genome file
    genome_path = get_genome_path(genome_file)
    
    # Open the FASTA file
    with pysam.FastaFile(genome_path) as fasta:
        # Get list of chromosome names in the reference
        contigs = fasta.references
        
        # Handle chromosome naming (with or without 'chr' prefix)
        orig_chrom = chrom
        if chrom not in contigs:
            if chrom.startswith('chr'):
                alt_chrom = chrom[3:]
            else:
                alt_chrom = f"chr{chrom}"
            
            if alt_chrom in contigs:
                chrom = alt_chrom
            else:
                raise ValueError(f"Chromosome {orig_chrom} not found in reference genome")
        
        # Convert to 0-based position for pysam
        pos_0 = pos - 1
        
        # Calculate the region to extract
        start = max(0, pos_0 - context_length)
        end = pos_0 + len(ref) + context_length
        
        # Extract the sequence
        region_seq = fasta.fetch(chrom, start, end).upper()
        
        # Verify the reference allele
        ref_in_genome = region_seq[context_length:context_length + len(ref)]
        if ref_in_genome != ref.upper():
            raise ValueError(
                f"Reference allele mismatch at {chrom}:{pos}. "
                f"Expected '{ref.upper()}', found '{ref_in_genome}' in genome."
            )
        
        # Create reference and alternate sequences
        ref_seq = region_seq
        alt_seq = region_seq[:context_length] + alt.upper() + region_seq[context_length + len(ref):]
        
        return ref_seq, alt_seq

def extract_sequence_range(
    chrom: str, 
    start: int, 
    end: int, 
    genome_file: str
) -> str:
    """
    Extract sequence from a reference genome for a specified range.
    Positions are 1-based, inclusive.
    
    Args:
        chrom: Chromosome name (with or without 'chr' prefix)
        start: Start position (1-based)
        end: End position (1-based, inclusive)
        genome_file: Path to the reference genome FASTA file or genome name (e.g., 'hg38')
        
    Returns:
        Sequence in the specified range in uppercase
    """
    # Get the full path to the genome file
    genome_path = get_genome_path(genome_file)
    
    # Open the FASTA file
    with pysam.FastaFile(genome_path) as fasta:
        # Get list of chromosome names in the reference
        contigs = fasta.references
        
        # Handle chromosome naming (with or without 'chr' prefix)
        orig_chrom = chrom
        if chrom not in contigs:
            if chrom.startswith('chr'):
                alt_chrom = chrom[3:]
            else:
                alt_chrom = f"chr{chrom}"
            
            if alt_chrom in contigs:
                chrom = alt_chrom
            else:
                raise ValueError(f"Chromosome {orig_chrom} not found in reference genome")
        
        # Convert to 0-based position for pysam (start is 0-based, end is exclusive)
        start_0 = start - 1
        end_0 = end  # pysam.fetch is exclusive of end
        
        # Extract the sequence
        sequence = fasta.fetch(chrom, start_0, end_0).upper()
        
        return sequence

def batch_extract_sequences(
    variants_df: pd.DataFrame,
    context_length: int = 10,
    genome_file: str = "hg38",
    chrom_col: str = 'chrom',
    pos_col: str = 'pos',
    ref_col: str = 'ref',
    alt_col: str = 'alt'
) -> pd.DataFrame:
    """
    Extract sequences for a batch of variants, optimized for performance.
    All sequences are returned in uppercase.
    
    Args:
        variants_df: DataFrame containing variant information
        context_length: Number of base pairs to include on each side of the variant
        genome_file: Path to the reference genome FASTA file or genome name (e.g., 'hg38')
        chrom_col: Column name for chromosome
        pos_col: Column name for position
        ref_col: Column name for reference allele
        alt_col: Column name for alternate allele
        
    Returns:
        DataFrame with added ref_sequence and alt_sequence columns in uppercase
    """
    # Make a copy to avoid modifying the original
    variants = variants_df.copy()
    
    # Ensure position is an integer
    variants[pos_col] = variants[pos_col].astype(int)
    
    # Add empty columns for sequences
    variants['ref_sequence'] = None
    variants['alt_sequence'] = None
    
    # Get the full path to the genome file
    genome_path = get_genome_path(genome_file)
    
    # Process variants by chromosome for efficiency
    with pysam.FastaFile(genome_path) as fasta:
        contigs = fasta.references
        chrom_map = {}  # Cache for chromosome name mapping
        
        # Group variants by chromosome
        for chrom, group_indices in variants.groupby(chrom_col).groups.items():
            # Handle chromosome naming
            if chrom not in contigs:
                if chrom.startswith('chr'):
                    alt_chrom = chrom[3:]
                else:
                    alt_chrom = f"chr{chrom}"
                
                if alt_chrom in contigs:
                    chrom_map[chrom] = alt_chrom
                    chrom = alt_chrom
                else:
                    print(f"Warning: Chromosome {chrom} not found in reference genome. Skipping.")
                    continue
            else:
                chrom_map[chrom] = chrom
            
            # Create arrays for results
            group_size = len(group_indices)
            ref_sequences = [None] * group_size
            alt_sequences = [None] * group_size
            
            # Process each variant in the chromosome group
            for i, idx in enumerate(group_indices):
                variant = variants.loc[idx]
                pos = int(variant[pos_col])
                ref = variant[ref_col].upper()
                alt = variant[alt_col].upper()
                
                try:
                    # Convert to 0-based position for pysam
                    pos_0 = pos - 1
                    
                    # Calculate the region to extract
                    start = max(0, pos_0 - context_length)
                    end = pos_0 + len(ref) + context_length
                    
                    # Extract the sequence and convert to uppercase
                    region_seq = fasta.fetch(chrom, start, end).upper()
                    
                    # Verify the reference allele
                    ref_in_genome = region_seq[context_length:context_length + len(ref)]
                    if ref_in_genome != ref:
                        print(f"Warning: Reference allele mismatch at {chrom}:{pos}. Expected '{ref}', found '{ref_in_genome}'.")
                        continue
                    
                    # Create reference and alternate sequences
                    ref_seq = region_seq
                    alt_seq = region_seq[:context_length] + alt + region_seq[context_length + len(ref):]
                    
                    ref_sequences[i] = ref_seq
                    alt_sequences[i] = alt_seq
                    
                except Exception as e:
                    print(f"Error processing variant {chrom}:{pos} {ref}>{alt}: {str(e)}")
            
            # Update the DataFrame efficiently
            for i, idx in enumerate(group_indices):
                if ref_sequences[i] is not None:
                    variants.at[idx, 'ref_sequence'] = ref_sequences[i]
                    variants.at[idx, 'alt_sequence'] = alt_sequences[i]
    
    return variants

def extract_sequences_from_df(
    df: pd.DataFrame,
    context_length: int = 10,
    genome: str = "hg38",
    chrom_col: str = 'chrom',
    pos_col: str = 'pos',
    ref_col: str = 'ref',
    alt_col: str = 'alt'
) -> pd.DataFrame:
    """
    Extract reference and alternate sequences for variants in a DataFrame.
    All sequences are returned in uppercase.
    
    Args:
        df: DataFrame containing variant information
        context_length: Number of base pairs to include on each side of the variant
        genome: Reference genome name (e.g., 'hg38') or path to FASTA file
        chrom_col: Column name for chromosome
        pos_col: Column name for position
        ref_col: Column name for reference allele
        alt_col: Column name for alternate allele
        
    Returns:
        DataFrame with added ref_sequence and alt_sequence columns in uppercase
    """
    return batch_extract_sequences(
        df,
        context_length=context_length,
        genome_file=genome,
        chrom_col=chrom_col,
        pos_col=pos_col,
        ref_col=ref_col,
        alt_col=alt_col
    )

def extract_sequences_from_range_df(
    df: pd.DataFrame,
    genome: str = "hg38",
    chrom_col: str = 'chrom',
    start_col: str = 'start',
    end_col: str = 'end'
) -> pd.DataFrame:
    """
    Extract sequences for genomic ranges in a DataFrame.
    All sequences are returned in uppercase.
    
    Args:
        df: DataFrame containing genomic range information
        genome: Reference genome name (e.g., 'hg38') or path to FASTA file
        chrom_col: Column name for chromosome
        start_col: Column name for start position (1-based)
        end_col: Column name for end position (1-based, inclusive)
        
    Returns:
        DataFrame with added sequence column in uppercase
    """
    # Make a copy to avoid modifying the original
    regions = df.copy()
    
    # Ensure positions are integers
    regions[start_col] = regions[start_col].astype(int)
    regions[end_col] = regions[end_col].astype(int)
    
    # Add empty column for sequence
    regions['sequence'] = None
    
    # Get the full path to the genome file
    genome_path = get_genome_path(genome)
    
    # Process regions by chromosome for efficiency
    with pysam.FastaFile(genome_path) as fasta:
        contigs = fasta.references
        chrom_map = {}  # Cache for chromosome name mapping
        
        # Group regions by chromosome
        for chrom, group_indices in regions.groupby(chrom_col).groups.items():
            # Handle chromosome naming
            if chrom not in contigs:
                if chrom.startswith('chr'):
                    alt_chrom = chrom[3:]
                else:
                    alt_chrom = f"chr{chrom}"
                
                if alt_chrom in contigs:
                    chrom_map[chrom] = alt_chrom
                    chrom = alt_chrom
                else:
                    print(f"Warning: Chromosome {chrom} not found in reference genome. Skipping.")
                    continue
            else:
                chrom_map[chrom] = chrom
            
            # Create arrays for results
            group_size = len(group_indices)
            sequences = [None] * group_size
            
            # Process each region in the chromosome group
            for i, idx in enumerate(group_indices):
                region = regions.loc[idx]
                start = int(region[start_col])
                end = int(region[end_col])
                
                try:
                    # Convert to 0-based position for pysam
                    start_0 = start - 1
                    end_0 = end  # pysam.fetch is exclusive of end
                    
                    # Extract the sequence
                    sequence = fasta.fetch(chrom, start_0, end_0).upper()
                    sequences[i] = sequence
                    
                except Exception as e:
                    print(f"Error processing region {chrom}:{start}-{end}: {str(e)}")
            
            # Update the DataFrame efficiently
            for i, idx in enumerate(group_indices):
                if sequences[i] is not None:
                    regions.at[idx, 'sequence'] = sequences[i]
    
    return regions

def main():
    parser = argparse.ArgumentParser(
        description='Extract sequences from a reference genome for variants or genomic ranges'
    )
    # Common parameters
    parser.add_argument('--chrom', help='Chromosome')
    parser.add_argument('--genome', default='hg38', help='Reference genome name or path')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Variant mode
    variant_parser = subparsers.add_parser('variant', help='Extract sequence for a single variant')
    variant_parser.add_argument('--pos', type=int, required=True, help='Position (1-based)')
    variant_parser.add_argument('--ref', required=True, help='Reference allele')
    variant_parser.add_argument('--alt', required=True, help='Alternate allele')
    variant_parser.add_argument('--context', type=int, default=10, help='Context length on each side')
    
    # Range mode
    range_parser = subparsers.add_parser('range', help='Extract sequence for a genomic range')
    range_parser.add_argument('--start', type=int, required=True, help='Start position (1-based)')
    range_parser.add_argument('--end', type=int, required=True, help='End position (1-based, inclusive)')
    
    # File mode
    file_parser = subparsers.add_parser('file', help='Process a file of variants or ranges')
    file_parser.add_argument('--input-file', required=True, help='Input file')
    file_parser.add_argument('--output-file', help='Output file')
    file_parser.add_argument('--mode', choices=['variant', 'range'], required=True, help='File processing mode')
    file_parser.add_argument('--context', type=int, default=10, help='Context length for variant mode')
    file_parser.add_argument('--chrom-col', default='chrom', help='Column name for chromosome')
    file_parser.add_argument('--pos-col', default='pos', help='Column name for position (variant mode)')
    file_parser.add_argument('--ref-col', default='ref', help='Column name for reference allele (variant mode)')
    file_parser.add_argument('--alt-col', default='alt', help='Column name for alternate allele (variant mode)')
    file_parser.add_argument('--start-col', default='start', help='Column name for start position (range mode)')
    file_parser.add_argument('--end-col', default='end', help='Column name for end position (range mode)')
    
    args = parser.parse_args()
    
    # Process based on the mode
    if args.mode == 'variant':
        if not all([args.chrom, args.pos, args.ref, args.alt]):
            parser.print_help()
            return
        
        try:
            ref_seq, alt_seq = extract_sequence(
                args.chrom, args.pos, args.ref, args.alt, args.context, args.genome
            )
            print(f"Reference sequence: {ref_seq}")
            print(f"Alternate sequence: {alt_seq}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    elif args.mode == 'range':
        if not all([args.chrom, args.start, args.end]):
            parser.print_help()
            return
        
        try:
            sequence = extract_sequence_range(args.chrom, args.start, args.end, args.genome)
            print(f"Sequence: {sequence}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    elif args.mode == 'file':
        # Read the input file
        if args.input_file.endswith('.tsv') or args.input_file.endswith('.txt'):
            df = pd.read_csv(args.input_file, sep='\t')
        else:
            df = pd.read_csv(args.input_file)
        
        # Process based on file mode
        if args.mode == 'variant':
            results = extract_sequences_from_df(
                df, 
                args.context, 
                args.genome,
                args.chrom_col,
                args.pos_col,
                args.ref_col,
                args.alt_col
            )
        else:  # range mode
            results = extract_sequences_from_range_df(
                df,
                args.genome,
                args.chrom_col,
                args.start_col,
                args.end_col
            )
        
        # Output the results
        if args.output_file:
            results.to_csv(args.output_file, index=False)
        else:
            print(results)
        
        print(f"Processed {len(results)} entries")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()



