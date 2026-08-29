# imbalance-seq

The imbalance sequence δ(p, q) = (q − p)/(q + p) for 1 ≤ p < q, enumerated
triangularly by q then p.

```python
from imbalance_seq import entry_at, novel_index, nth_novel_index

entry_at(23).reduced      # (5, 11)   -- the 23rd entry, in lowest terms
novel_index(3, 5)         # 3         -- where 3/5 first appears
nth_novel_index(10**12)   # 1644932022860
```

## Install

```
pip install -e .[test]
pytest            # 82 tests
pytest -m "not slow"
```

No dependencies beyond the standard library.

## Cost of each access pattern

With b = log i, and M(b) the cost of multiplication:

| operation | function | cost | gcd |
|---|---|---|---|
| i → (p, q), unreduced δ | `entry_at` | O(M(b)) | no |
| streaming | `iter_entries` | O(1) amortised | no |
| lowest terms, novel i | `reduced_novel` | O(b) | no |
| lowest terms, streaming | `iter_reduced` | O(ω(q)) divisions | no Euclid |
| lowest terms, arbitrary i | `Entry.reduced` | O(M(b) log b) | yes |
| reduced a/b → its novel index | `novel_index` | O(M(b)) | no |
| novelty mask to N | `novelty_mask` | O(N), O(√N) space | no |
| n-th novel index | `nth_novel_index` | O(n^{1/3+ε}) | no |

Two asymmetries are the point of the table. Addressing a novel entry by *name*
(its rational value) is cheap in both directions; addressing it by *position*
costs a totient summatory computation. And reduction is free exactly at novel
indices, where h = gcd(p, q) = 1 — which is the operational content of the
novelty criterion.

`reduced_novel` takes novelty as a precondition and does not check it: verifying
costs the gcd the routine exists to avoid. At i = 12 it returns (3, 9) rather
than (1, 3). Pass `verify=True` to pay for the check.

## What the library computes

**Indexing.** `entry_at` seeds m from `isqrt(2i)` and certifies against
T_m ≤ i < T_{m+1}. Correctness comes from the triangular comparison, not from
the square root, so a floating-point seed could not produce a wrong answer.

**The involution.** `phi_involution(x, y) = (A(y−x), A(y+x))` with A the parity
adjustment (OEIS A026741) is an involution on primitive pairs. It carries
primitive pairs to reduced fractions and back, and is the reason the whole
construction avoids Euclid.

**Occurrence trails.** Occurrences of a fixed reduced a/b sit at indices
quadratic in k with constant second difference Q², Q = A(a+b).

**Ordinal access.** Novel entries in frontiers up to Q number Φ(Q) − 1. The
seed comes from Φ(Q) ~ 3Q²/π², the certificate from exact evaluation by the
hyperbola recurrence, and the position within the frontier from an
inclusion–exclusion binary search for the r-th totative.

**Spectral structure.** The transition operator of the chain is the unilateral
shift — spectrum the closed unit disc, no invariant measure — so nothing lives
there. The content is in the emissions:

- `ramanujan_coefficients(q)` gives ν(p) = (φ(q)/q) Σ_{d|q} (μ(d)/φ(d)) c_d(p).
  Only squarefree d survive, so the frequency support is the divisor lattice of
  rad(q). Prime powers are spectrally identical to their radical, and q is prime
  exactly when the support is {1, q}.
- `autocorrelation(q, h)` = ∏_{r|gcd(h,q)} (1 − 1/r) · ∏_{r|q, r∤h} (1 − 2/r).
- `singular_series(h)` = ∏_{r|h} (1 − r⁻²) · ∏_{r∤h} (1 − 2r⁻²), the mean value
  of R_q(h). It is Hardy–Littlewood-shaped with r² in place of r, so the r = 2
  factor does not vanish: there is no parity obstruction, and even lags exceed
  odd ones by 3/2. S(0) = 6/π².

**Second-order recurrence.** δ_{i+1} = G(δ_{i−1}, δ_i) via `next_delta`. Two
consecutive terms determine (p, q): inside a frontier δ decreases and the ratio
coordinate advances by 1/q, at a boundary δ jumps and p resets to 1. Order two
is minimal — 1/3 occurs at i = 0 and i = 4 with different successors.

## Verified against the papers

The suite checks Table 1 of *The Imbalance Sequence* and Appendix A of the SSRN
note entry by entry, the three occurrence trails of Example 7.3 with second
differences 4, 9, 16, and Theorem 5.1's two quadratic progressions
(d²−9)/8 and d(d+1)/2 − 2 as the a = 1 case of the general formula.

The singular series is checked empirically: Σ_{q≤Q} q·R_q(h) against S(h)·Q²/2
at Q = 20000 agrees to four decimals for h ∈ {0, 1, 2, 3, 6, 30}.
