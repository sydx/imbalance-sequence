"""
imbalance.py -- the imbalance sequence delta(p,q) = (q-p)/(q+p), 1 <= p < q.

Covers, for the triangular enumeration ordered by q then p:

  * random access to the i-th entry            O(M(b))      no gcd
  * the involution Phi on primitive pairs      O(b)         no gcd
  * reduced value <-> novel index              O(M(b))      no gcd
  * novelty mask up to N                       O(N)         no gcd
  * ordinal access to the n-th novel index     O(n^{1/3+e})
  * Ramanujan expansion of the novelty mask on a frontier
  * exact autocorrelation R_q(h) and its singular series S(h)
  * the second-order recurrence on the scalar sequence

Conventions: indices i are 0-based; frontier q occupies T_{q-2} <= i < T_{q-1}
with T_m = m(m+1)/2.  Pairs are (p, q) with 1 <= p < q.  Reduced fractions are
written a/b with 0 < a < b and gcd(a,b) = 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt, pi
from typing import Dict, Iterator, List, Tuple

__all__ = [
    "Entry", "tri", "entry_at", "iter_entries", "delta_at",
    "A", "phi_involution", "primitive_pair", "reduced_at", "reduced_novel", "iter_reduced",
    "novel_index", "occurrence_index", "occurrence_indices",
    "is_novel", "novelty_mask", "totient_sum", "nth_novel_index",
    "iter_novel", "numerator_progressions", "is_prime_born",
    "ramanujan_sum", "ramanujan_coefficients", "novelty_from_ramanujan",
    "autocorrelation", "autocorrelation_direct", "singular_series",
    "next_delta", "frontier_of_index",
]


# --------------------------------------------------------------------------
# 1.  Indexing:  i  <->  (p, q)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Entry:
    """One term of the sequence: the index, its pair, and its imbalance."""
    i: int
    p: int
    q: int
    num: int          # unreduced numerator  q - p
    den: int          # unreduced denominator q + p

    @property
    def delta(self) -> Fraction:
        return Fraction(self.num, self.den)

    @property
    def reduced(self) -> Tuple[int, int]:
        """Lowest terms.  Uses one gcd, which is unavoidable at a general index:
        the scale h = gcd(p, q) must be known, not merely tested against 1.
        At a novel index use `reduced_novel`, which is gcd-free."""
        h = gcd(self.p, self.q)
        return (A((self.q - self.p) // h), A((self.q + self.p) // h))

    @property
    def novel(self) -> bool:
        return gcd(self.p, self.q) == 1


def tri(m: int) -> int:
    """The m-th triangular number."""
    return m * (m + 1) // 2


def entry_at(i: int) -> Entry:
    """The i-th entry, by seeded square root plus an exact triangular certificate.

    m = m(i) is characterised by T_m <= i < T_{m+1}.  Since m(m+1) <= 2i gives
    m <= isqrt(2i), and 2i < (m+1)(m+2) < (m+2)^2 gives isqrt(2i) <= m+1, the
    seed isqrt(2i) is within one of m and a single comparison decides.
    """
    if i < 0:
        raise ValueError("index must be non-negative")
    s = isqrt(2 * i)
    m = s if tri(s) <= i else s - 1
    j = i - tri(m)
    p, q = j + 1, m + 2
    return Entry(i, p, q, m + 1 - j, m + 3 + j)


def frontier_of_index(i: int) -> int:
    """The frontier label q containing index i."""
    return entry_at(i).q


def delta_at(i: int) -> Fraction:
    e = entry_at(i)
    return Fraction(e.num, e.den)


def iter_entries(start: int = 0) -> Iterator[Entry]:
    """Stream entries from `start` at O(1) amortised cost (the chain F)."""
    e = entry_at(start)
    i, p, q = e.i, e.p, e.q
    while True:
        yield Entry(i, p, q, q - p, q + p)
        i += 1
        if p < q - 1:
            p += 1
        else:
            p, q = 1, q + 1


# --------------------------------------------------------------------------
# 2.  The involution on primitive pairs
# --------------------------------------------------------------------------

def A(n: int) -> int:
    """Parity adjustment: n if n is odd, n/2 if n is even (OEIS A026741)."""
    return n if n & 1 else n >> 1


def phi_involution(x: int, y: int) -> Tuple[int, int]:
    """Phi(x, y) = (A(y - x), A(y + x)), an involution on primitive pairs x < y."""
    return (A(y - x), A(y + x))


def primitive_pair(a: int, b: int) -> Tuple[int, int]:
    """The primitive pair (P, Q) generating the reduced imbalance a/b."""
    if not (0 < a < b) or gcd(a, b) != 1:
        raise ValueError("need a reduced fraction with 0 < a < b")
    return phi_involution(a, b)


def reduced_at(i: int) -> Tuple[int, int]:
    """Lowest terms at any index (one gcd)."""
    return entry_at(i).reduced


def reduced_novel(i: int, verify: bool = False) -> Tuple[int, int]:
    """Lowest terms at a novel index, gcd-free.

    Novelty means h = gcd(p, q) = 1, so the involution alone reduces:
    (a, b) = Phi(p, q) = (A(q - p), A(q + p)), at O(b) bit cost.

    Novelty is a PRECONDITION, not something this function checks: confirming
    it costs the very gcd the routine exists to avoid.  At a non-novel index
    the result is silently wrong whenever the scale h has an odd part -- e.g.
    i = 12 has (p, q) = (3, 6) and returns (3, 9) rather than (1, 3).  Pass
    verify=True to pay for a gcd and raise instead.
    """
    e = entry_at(i)
    if verify and gcd(e.p, e.q) != 1:
        raise ValueError(f"index {i} is not novel")
    return phi_involution(e.p, e.q)


def iter_reduced(start: int = 0) -> Iterator[Tuple[Entry, Tuple[int, int]]]:
    """Stream (entry, lowest-terms pair) without ever calling Euclid.

    The frontier label q is factorised once per frontier -- O(sqrt q), amortised
    over its q-1 entries -- after which the scale h = gcd(p, q) is read off from
    the exponents of q's primes in p, at O(omega(q)) divisions per entry.
    """
    q_cur, fac = None, []
    for e in iter_entries(start):
        if e.q != q_cur:
            q_cur = e.q
            fac = []
            t, d = e.q, 2
            while d * d <= t:
                if t % d == 0:
                    k = 0
                    while t % d == 0:
                        t //= d
                        k += 1
                    fac.append((d, k))
                d += 1
            if t > 1:
                fac.append((t, 1))
        h = 1
        for r, kmax in fac:
            if e.p % r == 0:
                t, k = e.p, 0
                while k < kmax and t % r == 0:
                    t //= r
                    k += 1
                h *= r ** k
        yield e, (A((e.q - e.p) // h), A((e.q + e.p) // h))


# --------------------------------------------------------------------------
# 3.  Novel indices and occurrence trails
# --------------------------------------------------------------------------

def occurrence_index(P: int, Q: int, k: int = 1) -> int:
    """Index of the k-th occurrence of the imbalance generated by (P, Q)."""
    return (k * k * Q * Q + k * (2 * P - 3 * Q)) // 2


def novel_index(a: int, b: int) -> int:
    """The unique index at which the reduced imbalance a/b first occurs."""
    P, Q = primitive_pair(a, b)
    return occurrence_index(P, Q, 1)


def occurrence_indices(a: int, b: int, count: int) -> List[int]:
    P, Q = primitive_pair(a, b)
    return [occurrence_index(P, Q, k) for k in range(1, count + 1)]


def is_novel(i: int) -> bool:
    e = entry_at(i)
    return gcd(e.p, e.q) == 1


def novelty_mask(n: int) -> bytearray:
    """The novelty mask for indices 0 <= i < n, by sieve -- O(n), no gcd.

    Sieve smallest prime factors up to the largest frontier touched, then in
    each frontier mark the multiples of its prime divisors as non-novel.
    """
    if n <= 0:
        return bytearray()
    Q = entry_at(n - 1).q
    spf = list(range(Q + 1))
    for r in range(2, isqrt(Q) + 1):
        if spf[r] == r:
            for t in range(r * r, Q + 1, r):
                if spf[t] == t:
                    spf[t] = r
    mask = bytearray(b"\x01") * n
    for q in range(2, Q + 1):
        base = tri(q - 2)
        if base >= n:
            break
        t, primes = q, []
        while t > 1:
            r = spf[t]
            primes.append(r)
            while t % r == 0:
                t //= r
        for r in primes:
            for p in range(r, q, r):
                idx = base + p - 1
                if idx >= n:
                    break
                mask[idx] = 0
    return mask


def iter_novel(start_index: int = 0) -> Iterator[Entry]:
    for e in iter_entries(start_index):
        if gcd(e.p, e.q) == 1:
            yield e


# --------------------------------------------------------------------------
# 4.  Ordinal access:  n  ->  the n-th novel index
# --------------------------------------------------------------------------

def totient_sum(x: int, cache: Dict[int, int] | None = None) -> int:
    """Phi(x) = sum_{q<=x} phi(q), by the hyperbola recurrence.  O(x^{2/3})."""
    if x <= 0:
        return 0
    if cache is None:
        cache = {}
    if x in cache:
        return cache[x]
    total = tri(x)
    k = 2
    while k <= x:
        v = x // k
        k2 = x // v
        total -= (k2 - k + 1) * totient_sum(v, cache)
        k = k2 + 1
    cache[x] = total
    return total


def _rth_totative(q: int, r: int) -> int:
    """The r-th positive integer < q coprime to q (1-based), by inclusion-exclusion."""
    t, primes = q, []
    d = 2
    while d * d <= t:
        if t % d == 0:
            primes.append(d)
            while t % d == 0:
                t //= d
        d += 1
    if t > 1:
        primes.append(t)

    divisors = [(1, 1)]                      # (squarefree divisor, mobius sign)
    for pr in primes:
        divisors += [(dv * pr, -sg) for dv, sg in divisors]

    def count(x: int) -> int:
        return sum(sg * (x // dv) for dv, sg in divisors)

    lo, hi = 1, q - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) >= r:
            hi = mid
        else:
            lo = mid + 1
    return lo


def nth_novel_index(n: int) -> int:
    """The n-th novel index (n >= 1), in O(n^{1/3+eps}) time and O(n^{1/4}) space.

    Novel entries in frontiers up to Q number Phi(Q) - 1.  Seed Q from
    Phi(Q) ~ 3Q^2/pi^2, certify by exact evaluation, then locate the entry
    inside its frontier as a totative.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    cache: Dict[int, int] = {}
    q = max(2, int(pi * (n / 3.0) ** 0.5))
    while totient_sum(q, cache) - 1 < n:
        q += 1
    while q > 2 and totient_sum(q - 1, cache) - 1 >= n:
        q -= 1
    r = n - (totient_sum(q - 1, cache) - 1)
    p = _rth_totative(q, r)
    return tri(q - 2) + p - 1


# --------------------------------------------------------------------------
# 5.  Fixed numerator: interleaved quadratic progressions
# --------------------------------------------------------------------------

def numerator_progressions(a: int, b_max: int) -> Dict[int, List[Tuple[int, int]]]:
    """Novel indices of a/b for b <= b_max, grouped by residue of b mod 2*rad(a).

    On each residue class the index is a quadratic in b; the number of classes
    is 2*phi(rad(a)), which is 2 for a = 1 (Theorem 5.1 of the SSRN note).
    """
    rad, t, d = 1, a, 2
    while d * d <= t:
        if t % d == 0:
            rad *= d
            while t % d == 0:
                t //= d
        d += 1
    if t > 1:
        rad *= t
    modulus = 2 * rad
    out: Dict[int, List[Tuple[int, int]]] = {}
    for b in range(a + 1, b_max + 1):
        if gcd(a, b) != 1:
            continue
        out.setdefault(b % modulus, []).append((b, novel_index(a, b)))
    return out


def is_prime_born(a: int, b: int) -> bool:
    """True when the reduced imbalance a/b first appears in a prime frontier."""
    Q = A(a + b)
    if Q < 2:
        return False
    if Q < 4:
        return True
    if Q % 2 == 0:
        return False
    d = 3
    while d * d <= Q:
        if Q % d == 0:
            return False
        d += 2
    return True


# --------------------------------------------------------------------------
# 6.  Spectral analysis of the emissions
# --------------------------------------------------------------------------

def _factorise(n: int) -> List[int]:
    primes, t, d = [], n, 2
    while d * d <= t:
        if t % d == 0:
            primes.append(d)
            while t % d == 0:
                t //= d
        d += 1
    if t > 1:
        primes.append(t)
    return primes


def _euler_phi(n: int) -> int:
    result = n
    for r in _factorise(n):
        result -= result // r
    return result


def _mobius(n: int) -> int:
    if n == 1:
        return 1
    t, sign = n, 1
    for r in _factorise(n):
        if t % (r * r) == 0:
            return 0
        sign = -sign
    return sign


def ramanujan_sum(d: int, k: int) -> int:
    """c_d(k) = mu(d/g) phi(d) / phi(d/g),  g = gcd(k, d)."""
    g = gcd(k, d)
    return _mobius(d // g) * _euler_phi(d) // _euler_phi(d // g)


def ramanujan_coefficients(q: int) -> List[Tuple[int, Fraction]]:
    """Amplitudes of the novelty mask on frontier q.

    nu(p) = (phi(q)/q) * sum_{d | q} (mu(d)/phi(d)) c_d(p).
    Only squarefree d survive, so the frequency support is the divisor lattice
    of rad(q) -- the spectrum sees the radical and nothing more.
    """
    lead = Fraction(_euler_phi(q), q)
    out = []
    for d in range(1, q + 1):
        if q % d:
            continue
        mu = _mobius(d)
        if mu:
            out.append((d, lead * Fraction(mu, _euler_phi(d))))
    return out


def novelty_from_ramanujan(q: int, p: int) -> Fraction:
    """Reconstruct nu(p) on frontier q from its Ramanujan expansion."""
    return sum((c * ramanujan_sum(d, p) for d, c in ramanujan_coefficients(q)),
               Fraction(0))


def autocorrelation(q: int, h: int) -> Fraction:
    """R_q(h) = prod_{r | gcd(h,q)} (1 - 1/r) * prod_{r | q, r not| h} (1 - 2/r)."""
    out = Fraction(1)
    for r in _factorise(q):
        out *= Fraction(r - 1, r) if h % r == 0 else Fraction(r - 2, r)
    return out


def autocorrelation_direct(q: int, h: int) -> Fraction:
    c = sum(1 for p in range(q) if gcd(p, q) == 1 and gcd((p + h) % q, q) == 1)
    return Fraction(c, q)


def singular_series(h: int, n_primes: int = 200000) -> float:
    """S(h) = prod_{r|h} (1 - r^-2) * prod_{r not| h} (1 - 2 r^-2).

    The mean value of R_q(h) over q.  S(0) = 6/pi^2, since every prime divides 0.
    """
    limit = n_primes
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for r in range(2, isqrt(limit) + 1):
        if sieve[r]:
            sieve[r * r::r] = bytearray(len(sieve[r * r::r]))
    out = 1.0
    for r in range(2, limit + 1):
        if sieve[r]:
            out *= (1 - 1.0 / (r * r)) if (h % r == 0) else (1 - 2.0 / (r * r))
    return out


# --------------------------------------------------------------------------
# 7.  The second-order recurrence on the scalar sequence
# --------------------------------------------------------------------------

def _pair_from_two(prev: Fraction, cur: Fraction) -> Tuple[int, int]:
    """Recover (p_i, q_i) from delta_{i-1} and delta_i alone.

    Inside a frontier delta decreases and the ratio coordinate advances by 1/q;
    at a boundary delta jumps up and p resets to 1.
    """
    if cur < prev:
        q_frac = (1 + cur) * (1 + prev) / (2 * (prev - cur))
        q = int(q_frac)
        if q != q_frac:
            raise ValueError("inconsistent pair")
        p_frac = q * (1 - cur) / (1 + cur)
        return (int(p_frac), q)
    q_frac = (1 + cur) / (1 - cur)
    return (1, int(q_frac))


def next_delta(prev: Fraction, cur: Fraction) -> Fraction:
    """delta_{i+1} = G(delta_{i-1}, delta_i).  Minimal order is two."""
    p, q = _pair_from_two(prev, cur)
    p, q = (p + 1, q) if p < q - 1 else (1, q + 1)
    return Fraction(q - p, q + p)
