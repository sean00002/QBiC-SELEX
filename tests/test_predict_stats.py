import numpy as np
import scipy.stats

from qbic_selex import sequences as seqs
from qbic_selex.kmers import reverse_complement, sliding_window
from qbic_selex.models import load_model
from qbic_selex.predict import score_model
from qbic_selex.stats import build_c_matrix, compute_statistics


def reference_p_value(alt_kmers, ref_kmers, partial, cov):
    """The original per-variant computation (qbic_predict.py, compute_p_value_cpu), in float64."""
    keys = list(partial)
    idx = {k: i for i, k in enumerate(keys)}
    c = np.zeros(len(keys))
    for sign, kmers in ((1, alt_kmers), (-1, ref_kmers)):
        for kmer in kmers:
            j = idx.get(kmer, idx.get(reverse_complement(kmer)))
            if j is not None:
                c[j] += sign
    beta = np.array([float(v) for v in partial.values()])
    estimate = c @ beta
    if estimate == 0:
        return 0.0, 1.0
    z = estimate / np.sqrt(c @ cov.astype(np.float64) @ c)
    return z, 2 * scipy.stats.norm.sf(abs(z))


def random_pairs(rng, n, length):
    refs, alts = [], []
    for _ in range(n):
        ref = "".join(rng.choice(list("ACGT"), length))
        pos = length // 2
        alt = ref[:pos] + rng.choice([b for b in "ACGT" if b != ref[pos]]) + ref[pos + 1:]
        refs.append(ref)
        alts.append(alt)
    return refs, alts


def test_effect_is_alt_minus_ref_kmer_sum(toy):
    model = load_model(toy["model"])
    ref, alt = "ACGTTGCA", "ACGATGCA"
    scores = score_model(model, [ref], [alt], np.array([seqs.OK], dtype=object))
    expected = (sum(model.complete[k] for k in sliding_window(alt, 3))
                - sum(model.complete[k] for k in sliding_window(ref, 3)))
    assert scores.effect.dtype == np.float32
    assert np.isclose(scores.effect[0], expected, rtol=1e-6)


def test_effect_is_strand_symmetric(toy):
    model = load_model(toy["model"])
    ref, alt = "ACGTTGCAT", "ACGTAGCAT"
    st = np.array([seqs.OK, seqs.OK], dtype=object)
    s = score_model(model, [ref, reverse_complement(ref)], [alt, reverse_complement(alt)], st)
    assert np.isclose(s.effect[0], s.effect[1], rtol=1e-6)


def test_batched_statistics_match_per_variant_reference(toy):
    model = load_model(toy["model"])
    rng = np.random.default_rng(1)
    refs, alts = random_pairs(rng, 300, 21)
    status = np.full(len(refs), seqs.OK, dtype=object)
    s = score_model(model, refs, alts, status, cov=toy["cov_matrix"])
    for i in range(len(refs)):
        z_ref, p_ref = reference_p_value(sliding_window(alts[i], 3), sliding_window(refs[i], 3),
                                         model.partial, toy["cov_matrix"])
        assert np.isclose(s.z[i], z_ref, rtol=1e-4, atol=1e-6), i
        assert np.isclose(s.p[i], p_ref, rtol=1e-3, atol=1e-8), i


def test_chunking_does_not_change_results(toy):
    model = load_model(toy["model"])
    refs, alts = random_pairs(np.random.default_rng(2), 50, 15)
    a = [sliding_window(s, 3) for s in alts]
    r = [sliding_window(s, 3) for s in refs]
    z1, p1 = compute_statistics(a, r, model.partial, toy["cov_matrix"], chunk_size=7)
    z2, p2 = compute_statistics(a, r, model.partial, toy["cov_matrix"], chunk_size=1000)
    np.testing.assert_allclose(z1, z2, rtol=1e-6)
    np.testing.assert_allclose(p1, p2, rtol=1e-6)


def test_no_change_gives_z0_p1(toy):
    model = load_model(toy["model"])
    s = score_model(model, ["ACGTACG"], ["ACGTACG"], np.array([seqs.OK], dtype=object),
                    cov=toy["cov_matrix"])
    assert s.effect[0] == 0 and s.z[0] == 0 and s.p[0] == 1


def test_count_matrix_does_not_overflow(toy):
    # 300 identical 3-mers on one side: more than int8 can hold
    model = load_model(toy["model"])
    ref, alt = "A" * 302, "C" * 302
    C = build_c_matrix([sliding_window(alt, 3)], [sliding_window(ref, 3)], model.partial, 0, 1)
    assert C.max() == 300 and C.min() == -300


def test_unscorable_rows_get_nan(toy):
    model = load_model(toy["model"])
    status = np.array([seqs.OK, seqs.CONTAINS_N, seqs.OK], dtype=object)
    s = score_model(model, ["ACGTACG", "", "AC"], ["ACGAACG", "", "AT"], status, cov=toy["cov_matrix"])
    assert np.isfinite(s.effect[0]) and np.isfinite(s.z[0])
    assert np.isnan(s.effect[1]) and np.isnan(s.z[1]) and np.isnan(s.p[1])
    assert s.status.tolist() == [seqs.OK, seqs.CONTAINS_N, seqs.SHORTER_THAN_K]
    assert np.isnan(s.effect[2])


def test_wildcard_applies_to_statistics(toy):
    """Regression: with --wildcard the N rows used to get an effect but NaN z/p."""
    import pandas as pd
    model = load_model(toy["model"])
    df = pd.DataFrame({"ref_sequence": ["ACGNACG"], "alt_sequence": ["ACGTTCG"]})
    ref, alt, status = seqs.prepare_sequences(df, wildcard="A")
    s = score_model(model, ref, alt, status, cov=toy["cov_matrix"])
    direct = score_model(model, ["ACGAACG"], ["ACGTTCG"], np.array([seqs.OK], dtype=object),
                         cov=toy["cov_matrix"])
    assert np.isfinite(s.z[0])
    assert s.effect[0] == direct.effect[0] and s.z[0] == direct.z[0] and s.p[0] == direct.p[0]
