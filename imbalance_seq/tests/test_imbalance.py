"""Unit tests for imbalance_seq.

Grouped by the section of the construction each one verifies.  Slow tests
(large sieves, scaling checks) are marked; run `pytest -m "not slow"` to skip.
"""

from fractions import Fraction
from math import gcd, log, pi

import pytest

from imbalance_seq import (
    A, autocorrelation, autocorrelation_direct, delta_at, entry_at, is_novel,
    is_prime_born, iter_entries, iter_reduced, next_delta, novel_index,
    novelty_from_ramanujan, novelty_mask, nth_novel_index,
    numerator_progressions, occurrence_indices, phi_involution,
    ramanujan_coefficients, reduced_novel, singular_series, totient_sum, tri,
)

PRIMES_UNDER_50 = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}


@pytest.fixture(scope="module")
def mask():
    """Novelty mask for the first 300k indices."""
    return novelty_mask(300_000)


# -- 1. Indexing -----------------------------------------------------------

@pytest.mark.parametrize("i,p,q,val", [
    (0, 1, 2, "1/3"), (1, 1, 3, "1/2"), (2, 2, 3, "1/5"),
    (3, 1, 4, "3/5"), (4, 2, 4, "1/3"), (5, 3, 4, "1/7"),
    (21, 1, 8, "7/9"), (23, 3, 8, "5/11"), (27, 7, 8, "1/15"),
])
def test_table_1(i, p, q, val):
    """Table 1 of the paper, including the reduced values."""
    e = entry_at(i)
    assert (e.p, e.q) == (p, q)
    assert str(e.delta) == val


def test_random_access_matches_stream():
    for e_stream, i in zip(iter_entries(), range(200_000)):
        e_rand = entry_at(i)
        assert (e_stream.p, e_stream.q) == (e_rand.p, e_rand.q), f"at i={i}"


def test_index_zero_is_first_pair():
    assert (entry_at(0).p, entry_at(0).q) == (1, 2)


def test_negative_index_rejected():
    with pytest.raises(ValueError):
        entry_at(-1)


def test_frontier_block_boundaries():
    """Frontier q occupies T_{q-2} <= i < T_{q-1}."""
    for q in range(2, 200):
        assert entry_at(tri(q - 2)) .q == q
        assert entry_at(tri(q - 1) - 1).q == q


# -- 2. The involution -----------------------------------------------------

@pytest.mark.parametrize("n,expected", [(1, 1), (2, 1), (3, 3), (4, 2), (5, 5), (6, 3)])
def test_A_is_A026741(n, expected):
    assert A(n) == expected


def test_phi_is_an_involution():
    for y in range(2, 300):
        for x in range(1, y):
            if gcd(x, y) == 1:
                assert phi_involution(*phi_involution(x, y)) == (x, y)


def test_reduced_form_matches_fraction():
    for i in range(50_000):
        assert Fraction(*entry_at(i).reduced) == delta_at(i)


def test_reduced_novel_is_gcd_free_and_correct():
    for i in range(50_000):
        if is_novel(i):
            assert Fraction(*reduced_novel(i)) == delta_at(i)


def test_reduced_novel_rejects_non_novel_when_verifying():
    assert not is_novel(12)
    with pytest.raises(ValueError):
        reduced_novel(12, verify=True)


def test_reduced_novel_is_unchecked_by_default():
    """Novelty is a precondition; unverified, i = 12 returns (3, 9) not (1, 3)."""
    assert reduced_novel(12) == (3, 9)
    assert entry_at(12).reduced == (1, 3)


def test_iter_reduced_avoids_euclid_but_agrees():
    for (e, ab), _ in zip(iter_reduced(), range(200_000)):
        assert Fraction(*ab) == Fraction(e.q - e.p, e.q + e.p), f"at i={e.i}"


def test_iter_reduced_reproduces_ssrn_appendix_a():
    got = [f"{a}/{b}" for (_, (a, b)), _ in zip(iter_reduced(), range(14))]
    assert got == ["1/3", "1/2", "1/5", "3/5", "1/3", "1/7", "2/3",
                   "3/7", "1/4", "1/9", "5/7", "1/2", "1/3", "1/5"]


# -- 3. Novel indices and occurrence trails --------------------------------

def test_novel_index_inverts_reduction():
    for i in range(30_000):
        e = entry_at(i)
        if e.novel:
            assert novel_index(*e.reduced) == i


@pytest.mark.parametrize("a,b,trail", [
    (1, 3, [0, 4, 12, 24, 40]),
    (1, 2, [1, 11, 30, 58, 95]),
    (3, 5, [3, 22, 57, 108, 175]),
])
def test_occurrence_trails(a, b, trail):
    assert occurrence_indices(a, b, 5) == trail


@pytest.mark.parametrize("a,b,sq", [(1, 3, 4), (1, 2, 9), (3, 5, 16)])
def test_second_difference_is_a_square(a, b, sq):
    t = occurrence_indices(a, b, 3)
    assert t[2] - 2 * t[1] + t[0] == sq


def test_novel_index_rejects_unreduced_input():
    with pytest.raises(ValueError):
        novel_index(2, 4)


# -- 4. The sieve ----------------------------------------------------------

def test_sieve_mask_matches_gcd(mask):
    for i in range(0, len(mask), 7):
        assert bool(mask[i]) == is_novel(i), f"at i={i}"


def test_novelty_density_is_six_over_pi_squared(mask):
    assert abs(sum(mask) / len(mask) - 6 / pi ** 2) < 2e-3


def test_empty_mask():
    assert novelty_mask(0) == bytearray()


# -- 5. Ordinal access -----------------------------------------------------

def test_totient_sum_known_values():
    assert totient_sum(100) == 3044
    assert totient_sum(1000) == 304_192


@pytest.mark.parametrize("n", [1, 2, 3, 10, 100, 1000, 12345, 50_000])
def test_nth_novel_index_matches_brute_force(n, mask):
    brute = [i for i in range(200_000) if mask[i]]
    assert nth_novel_index(n) == brute[n - 1]


def test_nth_novel_index_rejects_zero():
    with pytest.raises(ValueError):
        nth_novel_index(0)


@pytest.mark.slow
@pytest.mark.parametrize("n,tol", [(10 ** 6, 2e-3), (10 ** 9, 1e-4), (10 ** 12, 1e-5)])
def test_nth_novel_index_scales(n, tol):
    """Index of the n-th novel entry is n * pi^2 / 6 + O(sqrt(n) log n).

    The tightening tolerances track the O(Q log Q) error in the totient sum.
    """
    v = nth_novel_index(n)
    assert abs(v / (n * pi ** 2 / 6) - 1) < tol


# -- 6. Fixed numerator ----------------------------------------------------

def test_unit_fractions_give_two_progressions():
    prog = numerator_progressions(1, 60)
    assert len(prog) == 2
    for cls in prog.values():
        for b, i in cls:
            expected = (b * b - 9) // 8 if b % 2 else b * (b + 1) // 2 - 2
            assert i == expected, f"d={b}"


def test_numerator_three_gives_four_progressions():
    assert len(numerator_progressions(3, 400)) == 4


# -- 7. The prime-born partition -------------------------------------------

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, True), (1, 3, True), (1, 5, True), (2, 3, True),
    (1, 4, True), (3, 4, True), (3, 5, False), (4, 5, False),
    (1, 7, False), (2, 7, False), (5, 7, False),
])
def test_prime_born_criterion(a, b, expected):
    """a/b is prime-born iff A(a+b) is prime."""
    assert is_prime_born(a, b) is expected


def test_prime_born_density(mask):
    x = 100
    n = tri(x - 1)
    tot = sum(1 for i in range(n) if mask[i])
    pb = sum(1 for i in range(n)
             if mask[i] and is_prime_born(*entry_at(i).reduced))
    assert abs(pb / tot - pi ** 2 / (6 * log(x))) < 0.05


# -- 8. Ramanujan expansion ------------------------------------------------

def test_expansion_reproduces_the_mask():
    for q in range(2, 60):
        for p in range(q):
            expected = 1 if gcd(p, q) == 1 else 0
            assert novelty_from_ramanujan(q, p) == expected, f"q={q}, p={p}"


def test_support_is_the_squarefree_divisors():
    assert [d for d, _ in ramanujan_coefficients(12)] == [1, 2, 3, 6]


def test_prime_powers_are_spectrally_identical_to_their_radical():
    """Only rad(q) is visible: q = 8 and q = 2 share a spectrum."""
    for q, r in [(8, 2), (9, 3), (27, 3), (25, 5)]:
        assert ([d for d, _ in ramanujan_coefficients(q)] ==
                [d for d, _ in ramanujan_coefficients(r)])


def test_primality_is_a_spectral_condition():
    for q in range(2, 50):
        support = [d for d, _ in ramanujan_coefficients(q)]
        assert (support == [1, q]) == (q in PRIMES_UNDER_50), f"q={q}"


def test_leading_coefficient_is_the_mean():
    for q in range(2, 40):
        d, c = ramanujan_coefficients(q)[0]
        assert d == 1
        assert c == Fraction(sum(1 for p in range(q) if gcd(p, q) == 1), q)


# -- 9. Autocorrelation and singular series --------------------------------

def test_autocorrelation_closed_form():
    for q in range(2, 80):
        for h in range(q):
            assert autocorrelation(q, h) == autocorrelation_direct(q, h), (q, h)


def test_zero_lag_is_the_totient_ratio():
    for q in range(2, 50):
        phi_q = sum(1 for p in range(q) if gcd(p, q) == 1)
        assert autocorrelation(q, 0) == Fraction(phi_q, q)


def test_even_frontier_odd_lag_vanishes():
    for q in range(2, 40, 2):
        for h in range(1, q, 2):
            assert autocorrelation(q, h) == 0


def test_singular_series_at_zero_lag():
    assert abs(singular_series(0) - 6 / pi ** 2) < 1e-6


def test_no_parity_obstruction():
    """Unlike Hardy-Littlewood, the r=2 factor does not vanish: even lags win."""
    assert singular_series(2) / singular_series(1) == pytest.approx(1.5, abs=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("h", [0, 1, 2, 3, 6, 30])
def test_singular_series_is_the_mean_value(h):
    """sum_{q<=Q} q R_q(h) ~ S(h) Q^2 / 2."""
    Q = 20_000
    empirical = sum(q * autocorrelation(q, h) for q in range(2, Q + 1))
    empirical /= Q * Q / 2
    assert float(empirical) == pytest.approx(singular_series(h), rel=2e-2)


# -- 10. The second-order recurrence ---------------------------------------

def test_second_order_recurrence():
    ds = [delta_at(i) for i in range(5000)]
    for i in range(1, len(ds) - 1):
        assert next_delta(ds[i - 1], ds[i]) == ds[i + 1], f"at i={i}"


def test_sequence_is_not_first_order():
    """1/3 occurs at i = 0 and i = 4 with different successors."""
    assert delta_at(0) == delta_at(4) == Fraction(1, 3)
    assert delta_at(1) != delta_at(5)
