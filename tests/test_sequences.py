import pandas as pd
import pytest

from qbic_selex import sequences as seqs

from .conftest import GENOME_SEQS

CHR1 = GENOME_SEQS["chr1"]


def variants(rows):
    return pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt"])


def test_extract_one_based_snv(genome):
    # pos 21 (1-based) is CHR1[20]
    df = seqs.extract_sequences(variants([["chr1", 21, CHR1[20], "T"]]), str(genome), context_length=5)
    row = df.iloc[0]
    assert row.qbic_status == seqs.OK
    assert row.ref_sequence == CHR1[15:26]
    assert row.alt_sequence == CHR1[15:20] + "T" + CHR1[21:26]


def test_zero_based_matches_one_based(genome):
    one = seqs.extract_sequences(variants([["chr1", 21, CHR1[20], "T"]]), str(genome), 5)
    zero = seqs.extract_sequences(variants([["chr1", 20, CHR1[20], "T"]]), str(genome), 5, zero_based=True)
    assert one.ref_sequence.tolist() == zero.ref_sequence.tolist()
    assert one.alt_sequence.tolist() == zero.alt_sequence.tolist()


def test_chrom_prefix_and_numeric_chrom(genome):
    df = seqs.extract_sequences(variants([["1", 21, CHR1[20], "T"], [1, 21, CHR1[20], "T"]]), str(genome), 5)
    assert (df.qbic_status == seqs.OK).all()


def test_lowercase_alleles(genome):
    df = seqs.extract_sequences(variants([["chr1", 21, CHR1[20].lower(), "t"]]), str(genome), 5)
    assert df.qbic_status.iloc[0] == seqs.OK
    assert df.alt_sequence.iloc[0][5] == "T"


def test_indels(genome):
    # deletion of 2 bases -> 1, insertion of 1 -> 3
    df = seqs.extract_sequences(variants([
        ["chr1", 21, CHR1[20:22], CHR1[20]],
        ["chr1", 21, CHR1[20], CHR1[20] + "GG"],
    ]), str(genome), 5)
    assert df.ref_sequence.tolist() == [CHR1[15:27], CHR1[15:26]]
    assert df.alt_sequence.tolist() == [CHR1[15:21] + CHR1[22:27], CHR1[15:21] + "GG" + CHR1[21:26]]


@pytest.mark.parametrize("row, status", [
    (["chr1", 21, "A" if CHR1[20] != "A" else "C", "T"], seqs.REF_MISMATCH),
    (["chr9", 21, "A", "T"], seqs.CHROM_NOT_FOUND),
    (["chr1", 3, CHR1[2], "T"], seqs.OUT_OF_BOUNDS),       # not enough left context
    (["chr1", 58, CHR1[57], "T"], seqs.OUT_OF_BOUNDS),     # not enough right context
    (["chr1", 21, CHR1[20], "-"], seqs.INVALID_ALLELE),
    (["chr1", None, CHR1[20], "T"], seqs.MISSING_VALUE),
])
def test_unscorable_rows_are_kept_with_status(genome, row, status):
    df = seqs.extract_sequences(variants([["chr1", 21, CHR1[20], "T"], row]), str(genome), 5)
    assert len(df) == 2
    assert df.qbic_status.tolist() == [seqs.OK, status]
    assert pd.isna(df.ref_sequence.iloc[1])


def test_integer_positions_survive_missing_values(genome):
    df = seqs.extract_sequences(variants([["chr1", 21, CHR1[20], "T"], ["chr1", None, "A", "T"]]),
                                str(genome), 5)
    assert str(df.pos.dtype) == "Int64"


def test_missing_genome_message():
    with pytest.raises(seqs.GenomeNotFoundError, match="hgdownload"):
        seqs.get_genome_path("no_such_genome_xyz")


def test_prepare_sequences():
    df = pd.DataFrame({
        "ref_sequence": ["ACGTACG", "acgtacg", "ACGNACG", "ACGXACG", None],
        "alt_sequence": ["ACGAACG", "acgaacg", "ACGAACG", "ACGAACG", "ACGT"],
    })
    ref, alt, status = seqs.prepare_sequences(df)
    assert status.tolist() == [seqs.OK, seqs.OK, seqs.CONTAINS_N, seqs.INVALID_SEQUENCE, seqs.MISSING_VALUE]
    assert ref[1] == "ACGTACG"          # uppercased
    assert ref[2] == ""                 # unscorable rows have no sequence

    ref, alt, status = seqs.prepare_sequences(df, wildcard="T")
    assert status[2] == seqs.OK
    assert ref[2] == "ACGTACG"
