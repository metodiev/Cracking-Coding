"""Task 302 - Convolution via number-theoretic transform (outline). (Tier: Intermediate, CTCI Ch 1)

Same butterfly as the FFT but over GF(998244353) = 119 * 2^23 + 1, whose multiplicative group
contains the needed power-of-two roots (primitive root 3). No floating point, exact results.
Outline: pad to a power of two <= 2^23, forward NTT both, pointwise multiply, inverse NTT.
"""

MOD = 998244353
PRIMITIVE_ROOT = 3


def power(base, exponent, modulus):
    result = 1
    base %= modulus
    while exponent > 0:
        if exponent & 1:
            result = result * base % modulus
        base = base * base % modulus
        exponent >>= 1
    return result


def ntt(values, invert):
    """In-place iterative NTT; `values` holds coefficients modulo MOD."""
    n = len(values)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            values[i], values[j] = values[j], values[i]

    length = 2
    while length <= n:
        root = power(PRIMITIVE_ROOT, (MOD - 1) // length, MOD)
        if invert:
            root = power(root, MOD - 2, MOD)
        half = length >> 1
        for start in range(0, n, length):
            w = 1
            for k in range(half):
                u = values[start + k]
                v = values[start + k + half] * w % MOD
                values[start + k] = (u + v) % MOD
                values[start + k + half] = (u - v) % MOD
                w = w * root % MOD
        length <<= 1

    if invert:
        inverse_n = power(n, MOD - 2, MOD)
        for i in range(n):
            values[i] = values[i] * inverse_n % MOD


def convolution_ntt(a, b):
    if not a or not b:
        return []
    size = 1
    while size < len(a) + len(b) - 1:
        size <<= 1
    assert size <= (1 << 23), "transform length exceeds the supported root order"
    left = [x % MOD for x in a] + [0] * (size - len(a))
    right = [x % MOD for x in b] + [0] * (size - len(b))
    ntt(left, False)
    ntt(right, False)
    for i in range(size):
        left[i] = left[i] * right[i] % MOD
    ntt(left, True)
    return left[: len(a) + len(b) - 1]


def convolution_naive_mod(a, b):
    if not a or not b:
        return []
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % MOD
    return out


def lcg_vectors(count, seed=7788):
    vectors = []
    for _ in range(count):
        size = 1 + seed % 7
        vector = []
        for _ in range(size):
            seed = (seed * 75 + 74) % 65537
            vector.append(seed % 1000)
        vectors.append(vector)
    return vectors


def main():
    assert power(PRIMITIVE_ROOT, MOD - 1, MOD) == 1                # Fermat: root of unity
    assert (MOD - 1) % (1 << 23) == 0 and ((MOD - 1) // (1 << 23)) == 119

    assert convolution_ntt([1, 2, 3], [0, 1, 1]) == [0, 1, 3, 5, 3]
    assert convolution_ntt([1, 1], [1, 1]) == [1, 2, 1]
    assert convolution_ntt([7], [9]) == [63]
    assert convolution_ntt([], [1]) == []
    assert convolution_ntt([MOD - 1], [2]) == [MOD - 2]            # (-1) * 2 = -2 mod MOD

    # NTT round trip on a single vector
    vector = [3, 1, 4, 1, 5, 9, 2, 6]
    copy = vector[:]
    ntt(copy, False)
    assert copy != vector
    ntt(copy, True)
    assert copy == vector

    vectors = lcg_vectors(40)
    for i in range(0, len(vectors) - 1, 2):
        a, b = vectors[i], vectors[i + 1]
        assert convolution_ntt(a, b) == convolution_naive_mod(a, b), (a, b)
        assert convolution_ntt(b, a) == convolution_naive_mod(a, b), (a, b)


if __name__ == "__main__":
    main()
    print("PASS")
