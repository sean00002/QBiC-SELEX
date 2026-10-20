"""k-mer helpers shared by model loading, prediction and statistics."""

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(sequence: str) -> str:
    """Reverse complement of an uppercase ACGT sequence."""
    return sequence.translate(_COMPLEMENT)[::-1]


def sliding_window(seq: str, k: int) -> list[str]:
    """All overlapping k-mers of ``seq``, left to right."""
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def recover_non_rc_dict(partial: dict[str, float]) -> dict[str, float]:
    """Expand a model's canonical-k-mer coefficients to all 4^k k-mers.

    Model files store one coefficient per reverse-complement pair (palindromes
    once); the reverse complement of each stored k-mer gets the same coefficient.
    """
    complete = dict(partial)
    for kmer, value in partial.items():
        complete.setdefault(reverse_complement(kmer), value)
    return complete
