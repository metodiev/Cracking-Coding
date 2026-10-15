"""Task 300 - Implement convolution (naive) for polynomials. (Tier: Intermediate, CTCI Beyond — Numerical & Scientific Computing)

c[k] = sum_i a[i] * b[k - i] computed with the straightforward O(n*m) double loop. The
product of a degree-(n-1) and a degree-(m-1) polynomial has length n + m - 1; trailing
zeros are kept, so [0] * [1, 2] == [0, 0]. The polynomial identities -- commutativity,
associativity, the Kronecker delta [1] as multiplicative identity, a shifted delta as a
shift operator, and evaluation at a point agreeing with the product of the evaluations --
pin the implementation down on deterministic LCG inputs.
"""


def convolve(a, b):
    """Naive O(n*m) convolution: result[k] = sum of a[i] * b[k - i]."""
    if not a or not b:
        return []
    result = [0] * (len(a) + len(b) - 1)
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            result[i + j] += left * right
    return result


def evaluate(polynomial, x):
    """Horner evaluation used as an independent oracle for the LCG cross-checks."""
    value = 0
    for coefficient in reversed(polynomial):
        value = value * x + coefficient
    return value


def lcg_vector(size, seed):
    values = []
    for _ in range(size):
        seed = (seed * 75 + 74) % 65537
        values.append(seed % 10)
    return values, seed


def main():
    assert convolve([1, 2, 3], [0, 1, 1]) == [0, 1, 3, 5, 3]
    assert convolve([5], [7]) == [35]
    assert convolve([5], [0]) == [0]
    assert convolve([], [1, 2]) == []
    assert convolve([1, 2], []) == []

    a = [4, 1, 2, 3]
    b = [2, 0, 5]
    c = [1, 1, 0, 2]
    assert convolve(a, b) == convolve(b, a)  # commutativity on fixed inputs
    assert convolve(convolve(a, b), c) == convolve(a, convolve(b, c))  # associativity

    # Kronecker delta: [1] is the identity, a delta at offset 2 shifts the other factor
    assert convolve(a, [1]) == a and convolve([1], a) == a
    assert convolve([0, 0, 1], [1, 2, 3]) == [0, 0, 1, 2, 3]
    assert convolve([1, 0, 0], [3, 4]) == [3, 4, 0, 0]

    seed = 70707
    for _ in range(120):
        seed = (seed * 75 + 74) % 65537
        n = 1 + seed % 6
        seed = (seed * 75 + 74) % 65537
        m = 1 + seed % 6
        left, seed = lcg_vector(n, seed)
        right, seed = lcg_vector(m, seed)
        product = convolve(left, right)
        assert len(product) == n + m - 1
        assert product == convolve(right, left)
        for x in (2, 3, 5):
            assert evaluate(product, x) == evaluate(left, x) * evaluate(right, x), (left, right, x)


if __name__ == "__main__":
    main()
    print("PASS")
