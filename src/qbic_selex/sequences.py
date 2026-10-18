"""Reference/alternate context sequences: extraction from a genome and validation."""

import logging
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

log = logging.getLogger(__name__)

# Row status values. Anything other than OK gets NA predictions.
OK = "ok"
CONTAINS_N = "contains_N"
INVALID_SEQUENCE = "invalid_sequence"
MISSING_VALUE = "missing_value"
INVALID_ALLELE = "invalid_allele"
CHROM_NOT_FOUND = "chrom_not_found"
OUT_OF_BOUNDS = "out_of_bounds"
REF_MISMATCH = "ref_mismatch"
SHORTER_THAN_K = "shorter_than_k"

VARIANT_COLUMNS = ("chrom", "pos", "ref", "alt")
SEQUENCE_COLUMNS = ("ref_sequence", "alt_sequence")
STATUS_COLUMN = "qbic_status"   # per-row reason a row could not be scored ("ok" otherwise)

_ALLELE_RE = re.compile(r"^[ACGTN]+$")
_ACGT = frozenset("ACGT")


class GenomeNotFoundError(FileNotFoundError):
    pass


def get_genome_path(genome: str) -> Path:
    """Resolve ``--genome``: a FASTA path, or a name such as ``hg38``.

    Names are looked up as ``<name>.fa`` in ``$QBIC_SELEX_GENOME_DIR`` and ``./genome``.
    """
    if os.path.exists(genome):
        return Path(genome)
    fname = genome if genome.endswith((".fa", ".fasta", ".fa.gz")) else genome + ".fa"
    search_dirs = [d for d in (os.environ.get("QBIC_SELEX_GENOME_DIR"), "genome") if d]
    for d in search_dirs:
        candidate = Path(d) / fname
        if candidate.exists():
            return candidate
    raise GenomeNotFoundError(
        f"genome FASTA not found: {genome!r} (looked for {fname} in {', '.join(search_dirs)}). "
        f"Pass a path with --genome, e.g. hg38.fa from "
        f"https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz (gunzip it first).")


def _summarize(statuses, what: str) -> None:
    counts = Counter(s for s in statuses if s != OK)
    if counts:
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        log.warning("%s: %d row(s) cannot be scored and will get NA (%s)", what, sum(counts.values()), detail)


def extract_sequences(variants: pd.DataFrame, genome: str, context_length: int = 10,
                      zero_based: bool = False) -> pd.DataFrame:
    """Add ``ref_sequence``, ``alt_sequence`` and ``qbic_status`` columns for ``chrom,pos,ref,alt`` rows.

    Each sequence is ``context_length`` bp of reference on each side of the
    variant, with the ref (or alt) allele in between, uppercase. Positions are
    1-based (VCF-style) unless ``zero_based``. Rows that cannot be extracted keep
    their place, with empty sequences and a status explaining why.
    """
    missing_cols = [c for c in VARIANT_COLUMNS if c not in variants.columns]
    if missing_cols:
        raise ValueError(f"variant input needs columns {', '.join(VARIANT_COLUMNS)} "
                         f"(or ref_sequence, alt_sequence); missing: {', '.join(missing_cols)}")

    out = variants.copy()
    n = len(out)
    ref_seqs: list[str | None] = [None] * n
    alt_seqs: list[str | None] = [None] * n
    statuses = [OK] * n

    pos_numeric = pd.to_numeric(out["pos"], errors="coerce")
    if (pos_numeric.dropna() % 1 == 0).all():
        out["pos"] = pos_numeric.astype("Int64")   # keep integer formatting even with missing values
    positions = pos_numeric.to_numpy()
    chroms = out["chrom"].to_numpy()
    refs = out["ref"].to_numpy()
    alts = out["alt"].to_numpy()

    genome_path = get_genome_path(genome)
    with pysam.FastaFile(str(genome_path)) as fasta:
        contig_lengths = dict(zip(fasta.references, fasta.lengths, strict=True))
        contig_map = {}

        def contig_for(chrom: str) -> str | None:
            if chrom not in contig_map:
                alt_name = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
                contig_map[chrom] = (chrom if chrom in contig_lengths
                                     else alt_name if alt_name in contig_lengths else None)
            return contig_map[chrom]

        for i in range(n):
            chrom, pos, ref, alt = chroms[i], positions[i], refs[i], alts[i]
            if pd.isna(chrom) or pd.isna(pos) or pd.isna(ref) or pd.isna(alt):
                statuses[i] = MISSING_VALUE
                continue
            ref, alt = str(ref).upper(), str(alt).upper()
            if not _ALLELE_RE.match(ref) or not _ALLELE_RE.match(alt):
                statuses[i] = INVALID_ALLELE
                continue
            contig = contig_for(str(chrom))
            if contig is None:
                statuses[i] = CHROM_NOT_FOUND
                continue
            pos_0 = int(pos) if zero_based else int(pos) - 1
            start = pos_0 - context_length
            end = pos_0 + len(ref) + context_length
            if start < 0 or end > contig_lengths[contig]:
                statuses[i] = OUT_OF_BOUNDS
                continue
            region = fasta.fetch(contig, start, end).upper()
            if region[context_length:context_length + len(ref)] != ref:
                statuses[i] = REF_MISMATCH
                continue
            ref_seqs[i] = region
            alt_seqs[i] = region[:context_length] + alt + region[context_length + len(ref):]

    out["ref_sequence"] = ref_seqs
    out["alt_sequence"] = alt_seqs
    out[STATUS_COLUMN] = statuses
    _summarize(statuses, "Sequence extraction")
    return out


def prepare_sequences(df: pd.DataFrame, wildcard: str | None = None
                      ) -> tuple[list[str], list[str], np.ndarray]:
    """Clean ``ref_sequence``/``alt_sequence`` for scoring.

    Uppercases, replaces N with ``wildcard`` if given, and checks the alphabet.
    Returns (ref_seqs, alt_seqs, status) where rows that are not OK have empty
    sequences. An existing ``qbic_status`` column (from extraction) is respected.
    """
    n = len(df)
    statuses = (df[STATUS_COLUMN].astype(object).to_numpy().copy() if STATUS_COLUMN in df.columns
                else np.full(n, OK, dtype=object))
    ref_out, alt_out = [""] * n, [""] * n
    pairs = zip(df["ref_sequence"].to_numpy(), df["alt_sequence"].to_numpy(), strict=True)
    for i, (r, a) in enumerate(pairs):
        if statuses[i] != OK:
            continue
        if not isinstance(r, str) or not isinstance(a, str) or not r or not a:
            statuses[i] = MISSING_VALUE
            continue
        r, a = r.upper(), a.upper()
        if wildcard is not None:
            r, a = r.replace("N", wildcard), a.replace("N", wildcard)
        if not (_ACGT.issuperset(r) and _ACGT.issuperset(a)):
            statuses[i] = CONTAINS_N if _ACGT.union("N").issuperset(r + a) else INVALID_SEQUENCE
            continue
        ref_out[i], alt_out[i] = r, a
    _summarize(statuses, "Sequence check")
    return ref_out, alt_out, statuses
