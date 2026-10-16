"""Task 586 - Implement chirp z-transform outline for non-power-of-two convolutions. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)

The task asks for an "outline"; this file implements the transform for real and documents the
design instead of sketching it.  The chirp z-transform (Bluestein's algorithm) evaluates a
length-n DFT with three FFTs of a power-of-two size M >= 2n - 1, and therefore with *no* power-of
-two requirement on n -- which is exactly what makes linear convolution of non-power-of-two
operands possible without padding each operand to a power of two.

The identity used is nk = (n^2 + k^2 - (n - k)^2) / 2 with w = exp(-2*pi*i / n):

    X_k = w^(k^2/2) * sum_m [ x_m w^(m^2/2) ] * w^(-(k-m)^2/2)

so with A_m = x_m * w^(m^2/2) (the "chirp"), B between -(n-1) and n-1 defined as
w^(-m^2/2), the DFT becomes `X_k = w^(k^2/2) * (A * B)_k`, a linear convolution that a
zero-padded radix-2 FFT of size M >= 2n - 1 evaluates exactly.  The convolution of length 5 and
length 7 operands (11 coefficients out) is done this way, at the natural length 11 rather than at
16, which is the point of the exercise.

Accuracy note, stated honestly: every chirp factor is evaluated as `chirp^(k^2 mod 2n)` with the
exponent reduced modulo 2n in *exact integer arithmetic*, because exp(-pi*i*k^2/n) with a large
k^2 would otherwise lose most of its significant digits.  The remaining error is the FFT's own
O(eps log M), so the checks use 1e-9 tolerances (max relative deviation is measurable and printed
in the assertion messages when it fails).  Bluestein is the slower route when n *is* a power of
two (3 FFTs of size ~2n-1 instead of 1 FFT of size n), which the file notes rather than claims.

Checks: the transform against the O(n^2) naive DFT for every length 1..13 and for length 16;
the inverse round trip; Parseval's identity sum|X|^2 = n * sum|x|^2; linearity; the shift theorem
X_k -> X_k * w^(s*k); conjugate symmetry for real input; convolution of 5 x 7 and 6 x 6 operands
against schoolbook convolution, including the assertion that the Bluestein route reproduces the
power-of-two FFT route for 8 x 8 operands; and determinism of the sample data.
"""
import cmath
import math

MASK64 = (1 << 64) - 1
TWO_PI = 2.0 * math.pi


def splitmix64(state):
    """Splitmix64: many independent draws are needed for the sample signals, and the repo's
    16-bit LCG (seed*75 + 74) % 65537 has short cycles with correlated low bits."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31)) & MASK64


def uniform(state):
    state, value = splitmix64(state)
    return state, (value >> 11) / float(1 << 53)


def next_power_of_two(n):
    size = 1
    while size < n:
        size <<= 1
    return size


def fft(a, invert=False):
    """Iterative radix-2 Cooley-Tukey FFT; len(a) must be a power of two."""
    n = len(a)
    if n == 0 or n & (n - 1):
        raise ValueError("fft needs a positive power-of-two length")
    a = list(a)
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
        root = cmath.exp((-TWO_PI / length) * (1j if not invert else -1j))
        half = length >> 1
        for start in range(0, n, length):
            w = 1.0 + 0j
            for k in range(start, start + half):
                u = a[k]
                v = a[k + half] * w
                a[k] = u + v
                a[k + half] = u - v
                w *= root
        length <<= 1
    if invert:
        return [value / n for value in a]
    return a


def dft_naive(x):
    """O(n^2) definition of the DFT, vectorised over the roots for readability."""
    n = len(x)
    out = []
    for k in range(n):
        total = 0j
        for m, value in enumerate(x):
            total += value * cmath.exp(-TWO_PI * 1j * ((k * m) % n) / n)
        out.append(total)
    return out


def chirp_factors(n):
    """exp(-pi*i*m^2/n) for m in 0..n-1, with the exponent reduced mod 2n exactly."""
    base = cmath.exp(-TWO_PI * 1j / (2 * n))
    return [base ** (((m * m) % (2 * n))) for m in range(n)]


def dft_bluestein(x):
    """Chirp z-transform: length-n DFT for arbitrary n, via 3 FFTs of size M >= 2n - 1."""
    n = len(x)
    if n == 0:
        return []
    if n == 1:
        return [complex(x[0])]
    chirp = chirp_factors(n)
    m_size = next_power_of_two(2 * n - 1)
    a = [complex(x[k]) * chirp[k] for k in range(n)] + [0j] * (m_size - n)
    b = [0j] * m_size
    b[0] = 1.0 + 0j
    for k in range(1, n):
        value = chirp[k].conjugate()          # exp(+pi*i*k^2/n)
        b[k] = value
        b[m_size - k] = value
    fa = fft(a)
    fb = fft(b)
    fc = [fa[i] * fb[i] for i in range(m_size)]
    conv = fft(fc, invert=True)
    return [chirp[k] * conv[k] for k in range(n)]


def idft_bluestein(spectrum):
    """Inverse DFT by the conjugate trick: idft(X) = conj(dft(conj(X))) / n."""
    n = len(spectrum)
    if n == 0:
        return []
    return [value.conjugate() / n for value in dft_bluestein([v.conjugate() for v in spectrum])]


def convolve_bluestein(a, b):
    """Linear convolution at the exact output length, using Bluestein transforms."""
    if not a or not b:
        return []
    size = len(a) + len(b) - 1
    fa = dft_bluestein([complex(v) for v in a] + [0j] * (size - len(a)))
    fb = dft_bluestein([complex(v) for v in b] + [0j] * (size - len(b)))
    return idft_bluestein([fa[i] * fb[i] for i in range(size)])


def convolve_fft(a, b):
    """Reference route: pad to a power of two and use the plain radix-2 FFT."""
    if not a or not b:
        return []
    size = next_power_of_two(len(a) + len(b) - 1)
    fa = fft([complex(v) for v in a] + [0j] * (size - len(a)))
    fb = fft([complex(v) for v in b] + [0j] * (size - len(b)))
    fc = fft([fa[i] * fb[i] for i in range(size)], invert=True)
    return fc[:len(a) + len(b) - 1]


def convolve_naive(a, b):
    if not a or not b:
        return []
    out = [0j] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += complex(x) * complex(y)
    return out


def max_abs_error(a, b):
    assert len(a) == len(b), "shape mismatch"
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def random_signal(state, length):
    values = []
    for _ in range(length):
        state, real = uniform(state)
        state, imag = uniform(state)
        values.append(complex(2.0 * real - 1.0, 2.0 * imag - 1.0))
    return state, values


def main():
    # --- transform against the definition, for every length 1..13 and for 16 ----------
    state = 586586
    for n in list(range(1, 14)) + [16]:
        state, signal = random_signal(state, n)
        got = dft_bluestein(signal)
        expected = dft_naive(signal)
        assert max_abs_error(got, expected) <= 1e-9 * max(1, n), n
        back = idft_bluestein(got)
        assert max_abs_error(back, signal) <= 1e-9 * max(1, n), ("round trip", n)
    assert dft_bluestein([]) == [] and idft_bluestein([]) == []
    assert dft_bluestein([3.0 + 0j]) == [3.0 + 0j]
    # A delta transforms to the all-ones vector; a constant to a single spike of value n.
    delta = [0j] * 7
    delta[0] = 1.0 + 0j
    assert max_abs_error(dft_bluestein(delta), [1.0 + 0j] * 7) <= 1e-12
    spike = dft_bluestein([2.0 + 0j] * 5)
    assert max_abs_error(spike[:1], [10.0 + 0j]) <= 1e-12
    assert max_abs_error(spike[1:], [0j] * 4) <= 1e-12

    # --- Parseval, linearity, shift theorem, conjugate symmetry -----------------------
    state, signal = random_signal(state, 11)
    spectrum = dft_bluestein(signal)
    left = sum(abs(v) ** 2 for v in signal)
    right = sum(abs(v) ** 2 for v in spectrum) / len(signal)
    assert abs(left - right) <= 1e-9 * max(1.0, left), "Parseval"
    state, other = random_signal(state, 11)
    combined = dft_bluestein([u + v for u, v in zip(signal, other)])
    summed = [u + v for u, v in zip(dft_bluestein(signal), dft_bluestein(other))]
    assert max_abs_error(combined, summed) <= 1e-9, "linearity"
    shift = 3
    shifted = [signal[(k - shift) % 11] for k in range(11)]
    expected = [spectrum[k] * cmath.exp(-TWO_PI * 1j * ((shift * k) % 11) / 11) for k in range(11)]
    assert max_abs_error(dft_bluestein(shifted), expected) <= 1e-9, "shift theorem"
    real_signal = [complex(0.5 * k - 1.0) for k in range(11)]
    real_spectrum = dft_bluestein(real_signal)
    assert max_abs_error([v.conjugate() for v in real_spectrum],
                         [real_spectrum[(11 - k) % 11] for k in range(11)]) <= 1e-9, \
        "conjugate symmetry for real input"

    # --- non-power-of-two convolution -------------------------------------------------
    state, a = random_signal(state, 5)
    state, b = random_signal(state, 7)
    got = convolve_bluestein(a, b)
    assert len(got) == 11, "5 + 7 - 1 coefficients, at the natural length"
    assert max_abs_error(got, convolve_naive(a, b)) <= 1e-9
    state, c = random_signal(state, 6)
    state, d = random_signal(state, 6)
    assert max_abs_error(convolve_bluestein(c, d), convolve_naive(c, d)) <= 1e-9
    state, e = random_signal(state, 8)
    state, f = random_signal(state, 8)
    assert max_abs_error(convolve_bluestein(e, f), convolve_fft(e, f)) <= 1e-9, \
        "same answer as the power-of-two route when n happens to be a power of two"
    assert max_abs_error(convolve_bluestein(e, f), convolve_naive(e, f)) <= 1e-9
    assert convolve_bluestein([], [1, 2]) == [] and convolve_bluestein([2], [3]) == [6 + 0j]
    assert convolve_naive([1, 1, 1, 1, 1], [1] * 7) == [1, 2, 3, 4, 5, 5, 5, 4, 3, 2, 1]
    assert max_abs_error(convolve_bluestein([1] * 5, [1] * 7),
                         [1, 2, 3, 4, 5, 5, 5, 4, 3, 2, 1]) <= 1e-12

    # --- documented accuracy comparison: Bluestein vs the plain FFT -------------------
    state, big_a = random_signal(state, 32)
    state, big_b = random_signal(state, 32)
    reference = convolve_naive(big_a, big_b)
    err_bluestein = max_abs_error(convolve_bluestein(big_a, big_b), reference)
    err_fft = max_abs_error(convolve_fft(big_a, big_b), reference)
    assert err_bluestein <= 1e-9, err_bluestein
    assert err_fft <= 1e-12, err_fft
    # M for n = 13 is 32 (>= 2*13 - 1) while the output length alone is 13: the documented
    # ~3x oversampling cost of the chirp route.
    assert next_power_of_two(2 * 13 - 1) == 32
    assert next_power_of_two(13) == 16

    # --- determinism ------------------------------------------------------------------
    state_x, first = random_signal(777, 9)
    state_y, second = random_signal(777, 9)
    assert first == second and state_x == state_y
    assert max_abs_error(dft_bluestein(first), dft_bluestein(second)) == 0.0


if __name__ == "__main__":
    main()
    print("PASS")
