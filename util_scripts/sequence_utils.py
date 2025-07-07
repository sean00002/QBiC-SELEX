# this script is for general functions to deal with sequence and kmer data 
import pandas as pd # type: ignore
import numpy as np # type: ignore
import itertools # type: ignore
import gzip # type: ignore
from Bio import SeqIO # type: ignore
from collections import Counter # type: ignore

nucleotides = ['A','C','G','T']


#######################################
####### sequence loading functions
#######################################
def fastq_to_counter(file):
    """
    This function takes a fastq file and returns a dataframe with the sequence and count
    """
    with gzip.open(f"{file}", 'rt') as file:
        sequences = [str(record.seq) for record in SeqIO.parse(file, 'fastq')]
    sequence_counts = Counter([seq for seq in sequences if 'N' not in seq])
    df = pd.DataFrame(sequence_counts.items(), columns=['sequence', 'count'])
    df = df.sort_values('count', ascending=False)
    return df

def fastq_to_list(file):
    """
    This function takes a fastq file and returns a list of sequences
    """
    with gzip.open(f"{file}", 'rt') as file:
        sequences = [str(record.seq) for record in SeqIO.parse(file, 'fastq')]
    return sequences

def fastq_to_unique_list(file):
    """
    This function takes a fastq file and returns a list of unique sequences
    """
    with gzip.open(f"{file}", 'rt') as file:
        sequences = [str(record.seq) for record in SeqIO.parse(file, 'fastq')]
    return list(set(sequences))

def load_sequences_from_counter(filepath, expand = False):
    """ Load DNA sequences from a CSV counter file.
    Args:
        filepath (str): The path to the CSV file.
        expand (bool): If True, expand the sequences based on the count column. Default is False.
    Returns:
        list: A list of DNA sequences.
    """
    if expand == False:
        return pd.read_csv(filepath, header=0)['sequence'].tolist()
    else:
        df = pd.read_csv(filepath, header=0)
        return df.loc[df.index.repeat(df['count'])]['sequence'].tolist()
    

#######################################
####### kmer processing functions
#######################################
def sliding_window(x, k):
    """Create a sliding window of DNA sequences.
    Args:
        x (str): A DNA sequence.
        k (int): The window size.
    Returns:
        list: A list of k-mer sequences.
    """
    return [x[i:i+k] for i in range(len(x)-(k-1))]

def reverse_complement(sequence):
    """Function to get the reverse complement of a DNA sequence"""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return "".join(complement[base] for base in reversed(sequence))

def get_complete_kmers(k):
    """Get all possible k-mers for a given k.
    Args:
        k (int): The k-mer size.
    Returns:
        list: A list of all possible k-mers.
    """
    all_kmer = [''.join(x) for x in itertools.product(nucleotides, repeat=k)]
    return all_kmer


def get_non_rc_kmer(k):
    """
    Get a dictionary of non-reverse complement k-mers.
    Args:
        k (int): The k-mer size.
    Returns:
        dict: A dictionary of non-reverse complement k-mers mapped to their reverse complements.
    """
    kmers = get_complete_kmers(k)
    rev_comp_map = {kmer: reverse_complement(kmer) for kmer in kmers}
    unique_pairs = set()
    palindromes = set()
    for kmer, rev_kmer in rev_comp_map.items():
        if kmer == rev_kmer:
            palindromes.add(kmer)
        else:
            unique_pairs.add(tuple(sorted([kmer, rev_kmer])))
    
    rc_dict = dict(zip([x[0] for x in list(unique_pairs)],[x[1] for x in list(unique_pairs)]))
    rc_dict.update(dict(zip([x for x in list(palindromes)],[x for x in list(palindromes)])))

    return rc_dict

def recover_non_rc_dict(d):
    """
    This function takes a dict of non_rc kmer's coefficients and returns a dict of all kmer's coefficients
    """
    non_rc_dict = get_non_rc_kmer(len(list(d.keys())[0]))
    rev_non_rc_dict = {value:key for key, value in non_rc_dict.items()}
    rev_d = {rev_non_rc_dict[key]:value for key, value in d.items()}
    all_kmer_dict = d|rev_d
    return all_kmer_dict


def get_kmer_frequencies(sequences, k, prob = False):
    """Get the frequencies of all k-mers of length k in a list of sequences."""
    kmers = (kmer for seq in sequences for kmer in sliding_window(seq, k))
    kmer_counts = Counter(kmers)

    if prob == False:
        return kmer_counts
    else:
        total_count = sum(kmer_counts.values())
        return {kmer: count / total_count for kmer, count in kmer_counts.items()}

def find_largest_k(sequences, rc = True):
    """
    Find the largest k-mer size for a list of sequences.
    Args:
        sequences (list): A list of DNA sequences.
        rc (bool): If True, include reverse complements. Default is True.
    Returns:
        int: The largest k-mer size.
    """
    if rc == True:
        for k in range(1,len(sequences[0]) + 1):
            kmer = set([x for sub in [sliding_window(x, k) for x in sequences] for x in sub])
            kmer_rc = set([reverse_complement(x) for x in kmer])
            kmer_set = kmer.union(kmer_rc)
            if len(kmer_set) < 4 ** k:
                return k - 1
    else: 
        for k in range(1,len(sequences[0]) + 1):
            kmer = set([x for sub in [sliding_window(x, k) for x in sequences] for x in sub])
            if len(kmer) < 4 ** k:
                return k - 1

#######################################
####### sequence processing functions
#######################################

def diff_position(seq1, seq2):
    """
    This function takes two sequences and returns the position of difference between them
    Args:
        seq1 (str): First sequence
        seq2 (str): Second sequence
    Returns:
        int: Position of first difference, or -1 if sequences are identical
    """
    return next((i for i, (a, b) in enumerate(zip(seq1, seq2)) if a != b), -1)

def extract_kmers(sequences, k):
    non_rc_dict = get_non_rc_kmer(k)
    tbl_seq = [sliding_window(seq, k=k) for seq in sequences]
    tbl_seq = [[non_rc_dict.get(item, item) for item in sublist] for sublist in tbl_seq]
    index_map = {value: index for index, value in enumerate(sorted(set(non_rc_dict.values())))}
    tbl_seq_index = [[index_map.get(x) if x in index_map else index_map.get(reverse_complement(x)) for x in sub] for sub in tbl_seq]
    
    print(f"                - start filling the k-mer frequency matrix")
    tbl_seq_freq = np.zeros((len(tbl_seq), len(index_map)), dtype=int)
    for i, seq_idx in enumerate(tbl_seq_index):
        for idx in seq_idx:
            tbl_seq_freq[i, idx] += 1
    print("                - done filling")  

    return tbl_seq_freq, list(index_map.keys())


    








