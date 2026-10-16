"""Task 585 - Implement modular FFT for polynomial multiplication (NTT) small prime. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)

Number-theoretic transform over the prime field GF(p) with p = 998244353 = 119 * 2^23 + 1 (an
NTT-friendly prime: 2^23 divides p - 1) and the primitive root g = 3.  `ntt(a, invert)` is the
iterative radix-2 Cooley-Tukey butterfly with bit-reversal; each level uses the primitive
2^k-th root of unity w = g^((p-1)/n), the inverse transform uses w^{-1} and rescales by
n^{-1} mod p.  Polynomial multiplication zero-pads both operands to the next power of two
>= len(a) + len(b) - 1 and performs two forward transforms, a pointwise product and one inverse
transform.  All arithmetic is exact over GF(p): unlike a floating-point FFT there is no rounding
error that grows with the coefficient magnitudes, so large-coefficient products come out exact.

Documented checks: the transform round trip for every size 1, 2, 4, ..., 1024; the twiddle
factors are verified to have exact order n (w^n = 1, w^(n/2) = -1) and g is verified to be a
primitive root by testing g^((p-1)/q) != 1 for every prime factor q of p - 1; the product is
compared against schoolbook multiplication for 120 random operand pairs and against two
independent exact methods -- binomial coefficients from `math.comb` for (1 + x)^k (k up to 200)
and a Kronecker-substitution product that packs the coefficients into one big integer with
2^64-wide slots, which is validated against schoolbook first.  Boundary cases: empty operands,
length-one operands, a length-5 by length-7 product, a delta polynomial, and associativity /
distributivity over GF(p).  The file also states the limitation of the modular transform plainly
rather than hiding it: the output is the product *modulo p*, so it differs from the exact integer
product as soon as a true coefficient reaches p -- both facts are asserted (the wrapped result
equals the exact product reduced mod p, and a small-coefficient instance reproduces the integer
product unchanged).  A second modulus, 167772161 = 5 * 2^25 + 1 with the same root 3, shows
that nothing is hard-coded to the first prime.

Randomness for the operand coefficients comes from the repo's 16-bit LCG only because the draws
are few and each is immediately reduced mod p; no check depends on long-range independence of
that generator.
"""
P = 998244353              # 119 * 2^23 + 1
ROOT = 3
P2 = 167772161             # 5 * 2^25 + 1
ROOT2 = 3


def sieve_is_prime(n):
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def power_mod(base, exponent, modulus):
    result = 1
    base %= modulus
    while exponent:
        if exponent & 1:
            result = result * base % modulus
        base = base * base % modulus
        exponent >>= 1
    return result


def inverse_mod(value, modulus):
    return power_mod(value, modulus - 2, modulus)


def prime_factors(n):
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def ntt(a, invert, modulus=P, root=ROOT):
    """In-place iterative Cooley-Tukey NTT; len(a) must be a power of two."""
    n = len(a)
    if n == 0 or n & (n - 1):
        raise ValueError("length must be a positive power of two")
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        step = power_mod(root, (modulus - 1) // length, modulus)
        if invert:
            step = inverse_mod(step, modulus)
        half = length >> 1
        for start in range(0, n, length):
            w = 1
            for k in range(start, start + half):
                u = a[k]
                v = a[k + half] * w % modulus
                a[k] = (u + v) % modulus
                a[k + half] = (u - v) % modulus
                w = w * step % modulus
        length <<= 1
    if invert:
        n_inv = inverse_mod(n, modulus)
        for i in range(n):
            a[i] = a[i] * n_inv % modulus
    return a


def convolve_mod(a, b, modulus=P, root=ROOT):
    """Polynomial product over GF(modulus) via two forward NTTs and one inverse NTT."""
    if not a or not b:
        return []
    size = 1
    while size < len(a) + len(b) - 1:
        size <<= 1
    fa = [value % modulus for value in a] + [0] * (size - len(a))
    fb = [value % modulus for value in b] + [0] * (size - len(b))
    ntt(fa, False, modulus, root)
    ntt(fb, False, modulus, root)
    fc = [fa[i] * fb[i] % modulus for i in range(size)]
    ntt(fc, True, modulus, root)
    return fc[:len(a) + len(b) - 1]


def convolve_naive(a, b):
    """Schoolbook product over the integers (independent reference)."""
    if not a or not b:
        return []
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def convolve_kronecker(a, b, bits=64):
    """Independent exact method: pack coefficients into a big int with 2^bits-wide slots.

    Only valid for small non-negative coefficients (no slot may carry); every use here is
    validated against the schoolbook product."""
    if not a or not b:
        return []
    packed_a = 0
    for value in reversed(a):
        packed_a = (packed_a << bits) | value
    packed_b = 0
    for value in reversed(b):
        packed_b = (packed_b << bits) | value
    product = packed_a * packed_b
    slot_mask = (1 << bits) - 1
    return [(product >> (bits * i)) & slot_mask for i in range(len(a) + len(b) - 1)]


def lcg(state, modulus=65537):
    return (state * 75 + 74) % modulus


def random_poly(state, length, bound):
    values = []
    for _ in range(length):
        state = lcg(state)
        values.append(state % bound)
    return state, values


def main():
    # --- the moduli really are NTT-friendly, with a primitive root --------------------
    assert sieve_is_prime(P) and sieve_is_prime(P2)
    for modulus, root, max_power in ((P, ROOT, 23), (P2, ROOT2, 25)):
        assert (modulus - 1) % (1 << max_power) == 0
        assert (modulus - 1) % (1 << (max_power + 1)) != 0
        assert power_mod(root, (modulus - 1) // 2, modulus) == modulus - 1, "non-residue"
        assert power_mod(root, modulus - 1, modulus) == 1
        for q in prime_factors(modulus - 1):
            assert power_mod(root, (modulus - 1) // q, modulus) != 1, (modulus, q)

    # --- round trip, with the exact root order ---------------------------------------
    for power in range(11):
        n = 1 << power
        state, original = random_poly(1000 + power, n, P) if n > 1 else (0, [5])
        transformed = ntt(list(original), False)
        back = ntt(list(transformed), True)
        assert back == original, power
        if n >= 2 and (P - 1) % n == 0:
            w = power_mod(ROOT, (P - 1) // n, P)
            assert power_mod(w, n, P) == 1 and power_mod(w, n // 2, P) == P - 1, "order"
    delta = [0] * 8
    delta[0] = 1
    assert ntt(list(delta), False) == [1] * 8, "delta transforms to all ones"
    spectrum = ntt([7] * 8, False)
    assert spectrum[0] == 7 * 8 % P and all(v == 0 for v in spectrum[1:])
    try:
        ntt([1, 2, 3], False)
        raise AssertionError("non power-of-two length must be rejected")
    except ValueError:
        pass

    # --- convolution against schoolbook and Kronecker ---------------------------------
    state = 585585
    for trial in range(120):
        state, a = random_poly(state, 1 + trial % 17, P)
        state, b = random_poly(state, 1 + (trial * 7) % 19, P)
        expected = convolve_naive(a, b)
        assert convolve_mod(a, b) == [value % P for value in expected], trial
        small_a = [value % 100 for value in a]
        small_b = [value % 100 for value in b]
        small_naive = convolve_naive(small_a, small_b)
        assert convolve_mod(small_a, small_b) == small_naive, "small coefficients: exact"
        assert convolve_kronecker(small_a, small_b) == small_naive, "Kronecker packing agrees"
    assert convolve_mod([], [1, 2]) == [] and convolve_mod([1, 2], []) == []
    assert convolve_mod([3], [4]) == [12]
    assert convolve_mod([1], [1, 2, 3]) == [1, 2, 3]
    assert convolve_naive([1] * 5, [1] * 7) == [1, 2, 3, 4, 5, 5, 5, 4, 3, 2, 1]
    assert convolve_mod([1] * 5, [1] * 7) == [1, 2, 3, 4, 5, 5, 5, 4, 3, 2, 1]
    assert convolve_mod([1] * 5, [1] * 7, P2, ROOT2) == [1, 2, 3, 4, 5, 5, 5, 4, 3, 2, 1], "prime 2"

    # --- independent closed form: (1 + x)^k has binomial coefficients -----------------
    from math import comb

    def binom_poly(k):
        out = [1]
        for _ in range(k):
            out = convolve_mod(out, [1, 1])
        return out

    for k in (0, 1, 2, 5, 17, 64, 200):
        assert binom_poly(k) == [comb(k, i) % P for i in range(k + 1)], k
    assert binom_poly(64) == convolve_mod([1, 1], binom_poly(63)), "no restarts between calls"

    # --- associativity and distributivity over GF(p) ---------------------------------
    state, a = random_poly(state, 9, P)
    state, b = random_poly(state, 11, P)
    state, c = random_poly(state, 5, P)
    assert convolve_mod(convolve_mod(a, b), c) == convolve_mod(a, convolve_mod(b, c)), "assoc"
    c_padded = c + [0] * (len(b) - len(c))
    distributed = convolve_mod(a, [(x + y) % P for x, y in zip(b, c_padded)])
    ab = convolve_mod(a, b)
    ac = convolve_mod(a, c)
    ac_padded = ac + [0] * (len(ab) - len(ac))
    assert distributed == [(x + y) % P for x, y in zip(ab, ac_padded)], "distributivity"

    # --- documented limitation: the result is the product modulo p --------------------
    wrapped_a = [P - 1, P - 1]
    wrapped_b = [P - 1, P - 1]
    exact = convolve_naive(wrapped_a, wrapped_b)
    modular = convolve_mod(wrapped_a, wrapped_b)
    assert modular == [value % P for value in exact]
    assert modular != exact, "a true coefficient reached p, so the integer product is lost"
    assert wrapped_a[0] * wrapped_b[0] >= P

    # --- determinism ------------------------------------------------------------------
    state_a, poly_a = random_poly(4242, 16, P)
    state_b, poly_b = random_poly(4242, 16, P)
    assert poly_a == poly_b and state_a == state_b
    assert convolve_mod(poly_a, poly_a) == convolve_mod(poly_b, poly_b)


if __name__ == "__main__":
    main()
    print("PASS")
