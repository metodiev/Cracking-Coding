"""Task 539 - Implement bitset convolution using FWHT (XOR convolution) small N. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)

XOR convolution: given arrays a and b of length N = 2^k, compute c[m] = sum of a[i]*b[j] over all
pairs with i XOR j = m. The fast route is the Walsh-Hadamard transform, which turns XOR
convolution into pointwise multiplication: apply the in-place butterfly FWHT (block size doubling
1, 2, 4, ...; each butterfly replaces (x, y) by (x + y, x - y)) to both inputs, multiply
component-wise, and apply the inverse transform, which is the same butterfly followed by an exact
division by N - exact because both the forward transform and the product are integral, which the
implementation asserts. Complexity O(N log N) instead of the O(N^2) of the direct double loop.

The "bitset" reading of the task is exercised with 0/1 indicator arrays, where c[m] counts the
pairs of set bits whose XOR is m; the file checks that support against an explicitly built set of
XORs of element pairs, so the combinatorial meaning is verified independently of the algebra.

Checks: the involution identity fwht(fwht(a)) = N * a, and forward-then-inverse = identity for
N = 1, 2, 4, 8, 16, 32, 64; the row-sum identities F[0] = sum(a) and sum(F) = N * a[0]; the
transform of a delta is a signed spike pattern (a delta at 0 maps to all ones); XOR convolution
against the O(N^2) double loop and, for indicator inputs, against a set-based enumeration of the
pair XORs, on LCG-generated arrays over many sizes; the hand-checked cases {0,1} *xor* {0,1} = [2,
2, 0, 0] and {0,1,2,3} *xor* {0,4} = all ones; delta behaviour c[i] = a[i XOR k] for a spike at k;
the identity c[0] = popcount(a AND b) for indicator arrays, which also equals |A| when a == b;
mass conservation sum(c) = sum(a) * sum(b) and symmetry under swapping a and b; and the
assertion that a non-power-of-two length is rejected.
"""
def fwht(values, inverse=False):
    """In-place Walsh-Hadamard transform of an integer array of power-of-two length."""
    a = list(values)
    n = len(a)
    assert n >= 1 and (n & (n - 1)) == 0, "FWHT needs a power-of-two length"
    block = 1
    while block < n:
        step = block * 2
        for start in range(0, n, step):
            for j in range(start, start + block):
                x, y = a[j], a[j + block]
                a[j] = x + y
                a[j + block] = x - y
        block = step
    if inverse:
        for i in range(n):
            assert a[i] % n == 0, "the inverse needs integral transform values"
            a[i] //= n
    return a


def xor_convolve(a, b):
    """O(N log N) XOR convolution via the Walsh-Hadamard transform."""
    assert len(a) == len(b) and (len(a) & (len(a) - 1)) == 0
    fa = fwht(a)
    fb = fwht(b)
    return fwht([x * y for (x, y) in zip(fa, fb)], inverse=True)


def xor_convolve_direct(a, b):
    """O(N^2) reference: accumulate every pair into its XOR bucket."""
    n = len(a)
    out = [0] * n
    for i in range(n):
        ai = a[i]
        if ai == 0:
            continue
        for j in range(n):
            if b[j]:
                out[i ^ j] += ai * b[j]
    return out


def xor_convolve_sets(set_a, set_b, n):
    """Third method for indicator inputs: walk the explicit element pairs."""
    out = [0] * n
    for i in sorted(set_a):
        for j in sorted(set_b):
            out[i ^ j] += 1
    return out


def indicator(values, n):
    """0/1 array of length n from an iterable of indices."""
    out = [0] * n
    for v in values:
        assert 0 <= v < n
        out[v] = 1
    return out


def popcount(value):
    return bin(value).count("1")


def bits_of(arr):
    """The array's support as a Python integer, used as a literal bitset."""
    acc = 0
    for index, value in enumerate(arr):
        if value:
            acc |= 1 << index
    return acc


def lcg_array(n, seed, modulus=6):
    out = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        out.append(seed % modulus)
    return out, seed


def main():
    # --- Transform identities -----------------------------------------------------------
    for n in (1, 2, 4, 8, 16, 32, 64):
        seed = 1000 + n
        arr, seed = lcg_array(n, seed, modulus=7)
        forward = fwht(arr)
        assert fwht(forward) == [n * x for x in arr], "the transform is involutive up to N"
        assert fwht(forward, inverse=True) == arr, "forward then inverse is the identity"
        assert forward[0] == sum(arr), "F[0] is the total mass"
        assert sum(forward) == n * arr[0], "sum of the transform is N * a[0]"
    # A delta at 0 maps to the all-ones vector; a delta at k maps to the signed spike.
    assert fwht(indicator([0], 8)) == [1, 1, 1, 1, 1, 1, 1, 1]
    spikes = fwht(indicator([2], 8))
    assert all(value in (1, -1) for value in spikes), "a single spike transforms to +/-1"
    assert sum(1 for v in spikes if v == -1) == 4
    assert fwht([0, 0, 0, 0]) == [0, 0, 0, 0]
    rejected = False
    try:
        fwht([1, 2, 3])
    except AssertionError:
        rejected = True
    assert rejected, "a length of 3 is not a power of two and must be rejected"

    # --- Hand-checked convolutions ------------------------------------------------------
    small = indicator([0, 1], 4)
    assert xor_convolve(small, small) == [2, 2, 0, 0], "0^0 = 1^1 = 0, 0^1 = 1^0 = 1"
    left = indicator([0, 1, 2, 3], 8)
    right = indicator([0, 4], 8)
    assert xor_convolve(left, right) == [1] * 8, "each of the eight XORs is hit exactly once"

    # --- Delta behaviour: convolution with a spike is an XOR permutation ----------------
    n = 16
    arr, _ = lcg_array(n, 777)
    for k in (0, 1, 5, 15):
        delta = indicator([k], n)
        out = xor_convolve(arr, delta)
        assert out == [arr[i ^ k] for i in range(n)], "the spike permutes the buckets"
        assert xor_convolve(delta, arr) == out, "convolution is commutative"

    # --- Randomised agreement with the direct O(N^2) method ----------------------------
    seed = 424242
    comparisons = 0
    for n in (1, 2, 4, 8, 16, 32, 64):
        for _ in range(4):
            a, seed = lcg_array(n, seed, modulus=5)
            b, seed = lcg_array(n, seed, modulus=4)
            fast = xor_convolve(a, b)
            slow = xor_convolve_direct(a, b)
            assert fast == slow, "FWHT and the double loop must agree exactly"
            assert sum(fast) == sum(a) * sum(b), "mass is conserved"
            assert xor_convolve(b, a) == fast, "commutative"
            comparisons += 1
            if n == 1:
                assert fast == [a[0] * b[0]]
    assert comparisons >= 20

    # --- Indicator inputs: the bitset semantics ----------------------------------------
    seed = 31337
    for n in (8, 16, 32, 64, 128):
        values, seed = lcg_array(n, seed, modulus=n)
        set_a = sorted(set(values))
        seed = (seed * 75 + 74) % 65537
        set_b = sorted({(seed + 3 * k) % n for k in range(n // 3)})
        seed = (seed * 75 + 74) % 65537
        ia = indicator(set_a, n)
        ib = indicator(set_b, n)
        fast = xor_convolve(ia, ib)
        reversed_result = xor_convolve(ib, ia)
        by_sets = xor_convolve_sets(set_a, set_b, n)
        assert fast == by_sets, "FWHT counts exactly the element pairs"
        assert fast == xor_convolve_direct(ia, ib)
        assert fast == reversed_result, "commutative"
        assert all(value >= 0 for value in fast)
        assert sum(fast) == len(set_a) * len(set_b)
        support = bits_of(fast)
        expected_support = 0
        for i in set_a:
            for j in set_b:
                expected_support |= 1 << (i ^ j)
        assert support == expected_support, "the support is exactly the XOR set"
        # c[0] counts the equal pairs, i.e. the intersection of the two bit sets.
        assert fast[0] == popcount(bits_of(ia) & bits_of(ib))
        assert xor_convolve(ia, ia)[0] == len(set_a), "a set with itself: |A| pairs at XOR 0"

    # --- A two-element XOR table by hand ------------------------------------------------
    table = xor_convolve(indicator([0, 3], 4), indicator([0, 1], 4))
    assert table == [1, 1, 1, 1], "0^0, 0^1, 3^1, 3^0 = 0, 1, 2, 3 - each hit once"


if __name__ == "__main__":
    main()
    print("PASS")
