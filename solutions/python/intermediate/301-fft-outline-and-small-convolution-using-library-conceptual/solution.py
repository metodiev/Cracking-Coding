"""Task 301 - FFT outline and small convolution (conceptual). (Tier: Intermediate, CTCI Ch 1)

Outline: pad to a power of two, forward-transform both vectors, multiply pointwise, then
inverse-transform. Production code would call a library; the self-contained iterative
Cooley-Tukey implementation below keeps the exercise runnable in every language.
Validated against the naive O(n*m) convolution.
"""

import math


def fft(real, imag, invert):
    """In-place iterative Cooley-Tukey transform on parallel real/imag arrays."""
    n = len(real)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            real[i], real[j] = real[j], real[i]
            imag[i], imag[j] = imag[j], imag[i]

    length = 2
    while length <= n:
        angle = 2 * math.pi / length * (-1 if invert else 1)
        wlen_real, wlen_imag = math.cos(angle), math.sin(angle)
        half = length >> 1
        for start in range(0, n, length):
            w_real, w_imag = 1.0, 0.0
            for k in range(half):
                u_real, u_imag = real[start + k], imag[start + k]
                v_real = real[start + k + half] * w_real - imag[start + k + half] * w_imag
                v_imag = real[start + k + half] * w_imag + imag[start + k + half] * w_real
                real[start + k], imag[start + k] = u_real + v_real, u_imag + v_imag
                real[start + k + half], imag[start + k + half] = u_real - v_real, u_imag - v_imag
                w_real, w_imag = w_real * wlen_real - w_imag * wlen_imag, w_real * wlen_imag + w_imag * wlen_real
        length <<= 1

    if invert:
        for i in range(n):
            real[i] /= n
            imag[i] /= n


def convolution_fft(a, b):
    if not a or not b:
        return []
    size = 1
    while size < len(a) + len(b) - 1:
        size <<= 1
    real_a = [float(x) for x in a] + [0.0] * (size - len(a))
    imag_a = [0.0] * size
    real_b = [float(x) for x in b] + [0.0] * (size - len(b))
    imag_b = [0.0] * size

    fft(real_a, imag_a, False)
    fft(real_b, imag_b, False)
    for i in range(size):
        real_a[i], imag_a[i] = (
            real_a[i] * real_b[i] - imag_a[i] * imag_b[i],
            real_a[i] * imag_b[i] + imag_a[i] * real_b[i],
        )
    fft(real_a, imag_a, True)
    return [int(round(real_a[i])) for i in range(len(a) + len(b) - 1)]


def convolution_naive(a, b):
    if not a or not b:
        return []
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def lcg_vectors(count, seed=1123):
    vectors = []
    for _ in range(count):
        size = 1 + seed % 6
        vector = []
        for _ in range(size):
            seed = (seed * 75 + 74) % 65537
            vector.append(seed % 11 - 5)           # includes negatives
        vectors.append(vector)
    return vectors


def main():
    assert convolution_fft([1, 2, 3], [0, 1, 1]) == [0, 1, 3, 5, 3]
    assert convolution_fft([1, 1], [1, 1]) == [1, 2, 1]
    assert convolution_fft([7], [9]) == [63]
    assert convolution_fft([], [1, 2]) == []
    assert convolution_fft([4, 5], [6]) == [24, 30]

    # a delta vector must reproduce the other operand
    assert convolution_fft([1], [3, 4, 5]) == [3, 4, 5]

    vectors = lcg_vectors(40)
    for i in range(0, len(vectors) - 1, 2):
        a, b = vectors[i], vectors[i + 1]
        assert convolution_fft(a, b) == convolution_naive(a, b), (a, b)
        assert convolution_fft(b, a) == convolution_naive(a, b), (a, b)


if __name__ == "__main__":
    main()
    print("PASS")
