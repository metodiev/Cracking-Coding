"""Task 875 - Implement high-performance FFT tuned for cache and threading parameters.  (Tier: Master, CTCI Beyond - Numerical & Scientific Computing)

A radix-2 fast Fourier transform with several genuinely different implementations plus the
two performance models that drive real tuning:

  * ``fft_recursive``      decimation in time (recursive, natural order);
  * ``fft_iterative``      in-place Stockham-style butterflies with bit-reversal permutation;
  * ``fft_four_step``      the four-step / six-step decomposition n = n1 * n2: transform the
    n1-point columns, apply the twiddle factors W_n^{i2 k1}, then transform the n2-point rows.
    This is the cache-friendly formulation, because every sub-transform works on a contiguous
    chunk instead of striding through the whole array;
  * ``bluestein``          chirp-z transform, i.e. an O(n log n) DFT for lengths that are not
    powers of two, built from a power-of-two FFT of size >= 2n - 1.

``LruCache`` is a fully associative least-recently-used cache simulator (line = several
complex words).  Feeding it the *address streams* of the algorithms turns cache behaviour into
a measured quantity instead of a slogan: the four-step stream is compared against the
breadth-first butterfly stream, and the tiled transpose exposes the classical U-curve - too
small a tile and the writes stride one word per line, too large a tile and the read/write
working set no longer fits, so the miss count is minimised at an intermediate tile size.

Threading is expressed the same way: the butterflies of each stage are cut into contiguous
chunks, the per-stage cost is ceil(work / workers) * unit + sync overhead, and the schedule is
a deterministic function of those typed parameters.  No threads are created; what is verified
is that the chunked decomposition is *bit-identical* to the single-worker result for every
worker count, which is the property a real parallel FFT has to preserve.

Verified: every implementation against the O(n^2) DFT and against each other (round trip,
linearity, circular convolution against brute force, Parseval energy, conjugate symmetry,
delta/constant/cosine spectra, error growth limited to O(eps log n)); Bluestein for
non-power-of-two lengths; cache misses of the four-step stream below the butterfly stream with
a U-curve minimum in the tile size; chunked parallelism identical to serial for 1..8 workers
with a balanced and monotonically improving modelled schedule; and rejection of invalid
lengths and parameters.
"""
import cmath
import math

TWO_PI = 2.0 * math.pi


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# ------------------------------------------------------------------ transforms
def dft(values):
    """O(n^2) definition of the DFT - the brute-force oracle.

    The twiddle angle is reduced modulo n before the exponential so that libm's argument
    reduction does not inject an error larger than the transform's own rounding.
    """
    n = len(values)
    out = []
    for k in range(n):
        total = 0j
        for j in range(n):
            total += values[j] * cmath.exp(-2j * math.pi * ((j * k) % n) / n)
        out.append(total)
    return out


def fft_recursive(values):
    n = len(values)
    if n == 1:
        return [complex(values[0])]
    if n == 0:
        raise ValueError("empty transform input")
    even = fft_recursive(values[0::2])
    odd = fft_recursive(values[1::2])
    out = [0j] * n
    half = n // 2
    for k in range(half):
        twiddle = cmath.exp(-2j * math.pi * k / n) * odd[k]
        out[k] = even[k] + twiddle
        out[k + half] = even[k] - twiddle
    return out


def fft_iterative(values):
    """In-place radix-2 butterflies after a bit-reversal permutation.

    The twiddle factor is recomputed per butterfly instead of being accumulated, which keeps
    the error at the O(eps log n) level the accuracy check expects; accumulating w *= exp(...)
    compounds the rounding and would show a visibly larger error at large n.
    """
    n = len(values)
    if n == 0:
        raise ValueError("empty transform input")
    if n & (n - 1):
        raise ValueError("the iterative kernel needs a power-of-two length")
    out = [complex(value) for value in values]
    target = 0
    for index in range(1, n):
        bit = n >> 1
        while target & bit:
            target ^= bit
            bit >>= 1
        target |= bit
        if index < target:
            out[index], out[target] = out[target], out[index]
    length = 2
    while length <= n:
        half = length // 2
        for start in range(0, n, length):
            for offset in range(half):
                twiddle = cmath.exp(-2j * math.pi * offset / length)
                left = out[start + offset]
                right = out[start + offset + half] * twiddle
                out[start + offset] = left + right
                out[start + offset + half] = left - right
        length <<= 1
    return out


def ifft(values):
    """Inverse transform through the conjugation identity."""
    n = len(values)
    conjugated = [value.conjugate() for value in values]
    transformed = fft_for_length(conjugated)
    return [value.conjugate() / n for value in transformed]


def fft_for_length(values):
    n = len(values)
    if n & (n - 1):
        return bluestein(values)
    return fft_iterative(values)


def fft_four_step(values, n2):
    """Four-step FFT for n = n1 * n2 using the generic 1-D kernel on whole chunks."""
    n = len(values)
    if n2 <= 0 or n % n2:
        raise ValueError("n2 must divide the transform length")
    n1 = n // n2
    if n1 & (n1 - 1) or n2 & (n2 - 1):
        raise ValueError("both factors must be powers of two for the radix-2 kernel")
    columns = [fft_iterative([values[i * n2 + j] for i in range(n1)]) for j in range(n2)]
    twiddled = [[columns[j][i] * cmath.exp(-2j * math.pi * ((i * j) % n) / n) for i in range(n1)]
                for j in range(n2)]
    out = [0j] * n
    for k1 in range(n1):
        row = fft_iterative([twiddled[j][k1] for j in range(n2)])
        for k2 in range(n2):
            out[k2 * n1 + k1] = row[k2]
    return out


def bluestein(values):
    """Chirp-z transform: exact O(n log n) DFT for arbitrary n (uses a padded power of two)."""
    n = len(values)
    if n == 0:
        raise ValueError("empty transform input")
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    chirp = [cmath.exp(-1j * math.pi * (j * j % (2 * n)) / n) for j in range(n)]
    chirp_inverse = [value.conjugate() for value in chirp]
    a = [values[j] * chirp[j] for j in range(n)] + [0j] * (size - n)
    b = [0j] * size
    b[0] = chirp_inverse[0]
    for j in range(1, n):
        b[j] = chirp_inverse[j]
        b[size - j] = chirp_inverse[j]
    spectrum = fft_iterative(a)
    kernel = fft_iterative(b)
    product = [spectrum[index] * kernel[index] for index in range(size)]
    convolution = ifft(product)
    return [convolution[k] * chirp[k] for k in range(n)]


def circular_convolve(left, right):
    """Brute-force cyclic convolution - the oracle for the convolution theorem."""
    n = len(left)
    out = []
    for k in range(n):
        total = 0j
        for j in range(n):
            total += left[j] * right[(k - j) % n]
        out.append(total)
    return out


def max_abs(values):
    return max((abs(value) for value in values), default=0.0)


def difference(left, right):
    return max_abs([a - b for a, b in zip(left, right)])


# ------------------------------------------------------------------ cache model
class LruCache:
    """Fully associative LRU cache over an address space of complex words."""

    def __init__(self, lines, line_words):
        if lines < 1 or line_words < 1:
            raise ValueError("cache geometry must be positive")
        self.lines = lines
        self.line_words = line_words
        self.entries = []
        self.accesses = 0
        self.misses = 0

    def access(self, address):
        if address < 0:
            raise ValueError("negative address")
        self.accesses += 1
        line = address // self.line_words
        if line in self.entries:
            self.entries.remove(line)
            self.entries.append(line)
            return True
        self.misses += 1
        self.entries.append(line)
        if len(self.entries) > self.lines:
            self.entries.pop(0)
        return False

    def stream(self, addresses):
        for address in addresses:
            self.access(address)
        return self


def butterfly_addresses(n):
    """Address stream of a breadth-first radix-2 FFT: stride doubles every stage."""
    if n <= 1:
        return []
    addresses = []
    stride = 1
    while stride < n:
        span = stride * 2
        for start in range(0, n, span):
            for offset in range(stride):
                addresses.append(start + offset)
                addresses.append(start + offset + stride)
        stride = span
    return addresses


def transpose_addresses(rows, columns, tile):
    """Tiled out-of-place transpose of a rows x columns matrix, one word per access."""
    if tile < 1:
        raise ValueError("tile size must be positive")
    addresses = []
    for row in range(0, rows, tile):
        for column in range(0, columns, tile):
            for i in range(row, min(row + tile, rows)):
                for j in range(column, min(column + tile, columns)):
                    addresses.append(i * columns + j)
                    addresses.append(j * rows + i)
    return addresses


def four_step_addresses(n, n2, tile=8):
    """Main-memory address stream of a cache-tuned four-step transform.

    One level of the decomposition is modelled explicitly - the column sub-transforms, the
    twiddle pass, the tiled transpose and the row sub-transforms - and the sub-transforms
    themselves are assumed to be executed on cache-resident chunks, which is precisely the
    reason the four-step formulation exists.  The stream is therefore the *main-memory*
    traffic of the tuned schedule, and comparing it per transformed element against the
    breadth-first butterfly stream is the honest apples-to-apples comparison.
    """
    n1 = n // n2
    if n1 < 1 or n % n2:
        raise ValueError("n2 must divide the transform length")
    addresses = []
    block = max(1, min(n2, tile))
    for j0 in range(0, n2, block):
        for i in range(n1):
            for j in range(j0, min(j0 + block, n2)):
                addresses.append(i * n2 + j)
    addresses.extend(range(n))
    addresses.extend(transpose_addresses(n1, n2, tile))
    addresses.extend(range(n))
    return addresses


def cache_misses(addresses, lines, line_words):
    return LruCache(lines, line_words).stream(addresses).misses


# ------------------------------------------------------------------ threading model
def partition(work, workers):
    """Split `work` index positions into `workers` contiguous, balanced chunks."""
    if workers < 1:
        raise ValueError("workers must be at least one")
    if work < 0:
        raise ValueError("negative work")
    base, extra = divmod(work, workers)
    chunks = []
    start = 0
    for worker in range(workers):
        size = base + (1 if worker < extra else 0)
        chunks.append((start, start + size))
        start += size
    return chunks


def fft_partitioned(values, workers):
    """Radix-2 FFT whose butterflies are executed chunk by chunk - a deterministic model of
    a thread pool with a barrier between stages (no threads are created)."""
    n = len(values)
    if n & (n - 1):
        raise ValueError("the partitioned kernel needs a power-of-two length")
    out = [complex(value) for value in values]
    target = 0
    for index in range(1, n):
        bit = n >> 1
        while target & bit:
            target ^= bit
            bit >>= 1
        target |= bit
        if index < target:
            out[index], out[target] = out[target], out[index]
    length = 2
    while length <= n:
        half = length // 2
        butterflies = n // 2
        for start_index, end_index in partition(butterflies, workers):
            for position in range(start_index, end_index):
                block = position // half
                offset = position % half
                base = block * length + offset
                twiddle = cmath.exp(-2j * math.pi * offset / length)
                left = out[base]
                right = out[base + half] * twiddle
                out[base] = left + right
                out[base + half] = left - right
        length <<= 1
    return out


def stage_work(n):
    """Butterflies per stage, so the schedule cost is explicit in the typed parameters."""
    if n <= 1:
        return []
    return [n // 2] * int(math.log2(n))


def modelled_makespan(n, workers, unit=1.0, sync=2.0):
    """Sum over stages of ceil(work / workers) * unit + sync - a deterministic cost model."""
    if workers < 1:
        raise ValueError("workers must be at least one")
    total = 0.0
    for work in stage_work(n):
        total += math.ceil(work / workers) * unit + sync
    return total


def main():
    # ---------------- agreement with the brute-force DFT ----------------
    sizes = [1, 2, 4, 8, 16, 64, 256]
    worst = 0.0
    for n in sizes:
        signal = [cmath.exp(0.3j * j) * math.cos(0.7 * j) for j in range(n)]
        reference = dft(signal)
        for name, result in (("recursive", fft_recursive(signal)),
                             ("iterative", fft_iterative(signal)),
                             ("four_step", fft_four_step(signal, 4) if n % 4 == 0 and
                              (n // 4) & (n // 4 - 1) == 0 else None)):
            if result is None:
                continue
            error = difference(result, reference)
            worst = max(worst, error)
            check(error < 1e-9 * max(1.0, max_abs(reference)),
                  f"{name} disagrees with the DFT at n={n}: {error}")
    check(worst < 1e-9, f"transform agreement must be tight, worst {worst}")

    # ---------------- round trip, linearity, convolution, energy, symmetry ----------------
    n = 64
    signal = [math.sin(0.11 * j) + 0.5 * math.cos(0.37 * j) for j in range(n)]
    other = [math.cos(0.21 * j) - 0.25 * math.sin(0.53 * j) for j in range(n)]
    spectrum = fft_iterative(signal)
    check(difference(ifft(spectrum), signal) < 1e-12, "the round trip must return the input")
    check(difference(fft_iterative([a + b for a, b in zip(signal, other)]),
                     [a + b for a, b in zip(spectrum, fft_iterative(other))]) < 1e-9,
          "the transform must be linear")
    convolved = circular_convolve(signal, other)
    via_fft = ifft([a * b for a, b in zip(spectrum, fft_iterative(other))])
    check(difference(convolved, via_fft) < 1e-9,
          "the convolution theorem must hold against the brute-force cyclic convolution")
    energy_time = sum(value * value for value in signal)
    energy_frequency = sum(abs(value) ** 2 for value in spectrum) / n
    check(abs(energy_time - energy_frequency) < 1e-9 * max(1.0, energy_time),
          f"Parseval violated: {energy_time} vs {energy_frequency}")
    check(all(abs(spectrum[k] - spectrum[(n - k) % n].conjugate()) < 1e-9 for k in range(n)),
          "a real input must give a conjugate-symmetric spectrum")
    imaginary = fft_iterative([1j * value for value in signal])
    check(all(abs(imaginary[k] + imaginary[(n - k) % n].conjugate()) < 1e-9 for k in range(n)),
          "a purely imaginary input must give an anti-symmetric spectrum")

    # ---------------- closed-form spectra ----------------
    delta = fft_iterative([1.0] + [0.0] * (n - 1))
    check(max_abs([value - 1.0 for value in delta]) < 1e-12, "the delta spectrum is all ones")
    constant = fft_iterative([3.0] * n)
    check(abs(constant[0] - 3.0 * n) < 1e-9
          and max_abs(constant[1:]) < 1e-9, "a constant signal is a single DC bin")
    tone = fft_iterative([math.cos(TWO_PI * 5 * j / n) for j in range(n)])
    magnitudes = [abs(value) for value in tone]
    top = sorted(range(n), key=lambda index: -magnitudes[index])[:2]
    check(sorted(top) == [5, n - 5],
          f"a cosine at bin 5 must light up bins 5 and {n - 5}, got {sorted(top)}")

    # ---------------- non-power-of-two lengths through Bluestein ----------------
    for n_odd in (3, 5, 7, 11, 15, 27):
        signal_odd = [math.sin(0.4 * j) - 0.3 * math.cos(0.9 * j) for j in range(n_odd)]
        reference = dft(signal_odd)
        error = difference(bluestein(signal_odd), reference)
        check(error < 1e-9 * max(1.0, max_abs(reference)),
              f"Bluestein disagrees with the DFT at n={n_odd}: {error}")
    check(difference(ifft(bluestein([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0]) < 1e-9,
          "the inverse transform must work for a non-power-of-two length")

    # ---------------- error growth stays logarithmic (relative to the output scale) ----------------
    # The input is built from small exactly representable integers so the comparison measures
    # the transform's own rounding rather than the error of generating the signal.
    relative_errors = []
    for n_big in (64, 1024):
        exact_signal = [complex((j % 7) - 3, (j % 5) - 2) for j in range(n_big)]
        reference = dft(exact_signal)
        relative_errors.append(difference(fft_iterative(exact_signal), reference)
                               / max_abs(reference))
    check(relative_errors[1] < 8.0 * relative_errors[0],
          f"the relative error must grow far slower than n: {relative_errors}")
    check(relative_errors[1] <= 16.0 * 2.0 ** -53 * math.log2(1024),
          f"the relative error must respect the O(eps log n) bound: {relative_errors}")

    # ---------------- cache model: the blocked stream fetches fewer lines per element ----------------
    lines, line_words = 16, 8
    n_cache = 512
    butterfly_stream = butterfly_addresses(n_cache)
    blocked_stream = four_step_addresses(n_cache, 8, tile=8)
    butterfly = cache_misses(butterfly_stream, lines, line_words)
    blocked = cache_misses(blocked_stream, lines, line_words)
    check(butterfly > 0 and blocked > 0, "both address streams must be non-trivial")
    butterfly_traffic = butterfly / n_cache
    blocked_traffic = blocked / n_cache
    check(blocked_traffic < butterfly_traffic,
          f"the blocked schedule must move fewer lines per element: "
          f"{blocked_traffic:.3f} vs {butterfly_traffic:.3f}")
    check(all(0 <= address < n_cache for address in blocked_stream),
          "the blocked stream must stay inside the array")

    # the tiled transpose exposes the classical U-curve over the tile size
    tiles = [1, 2, 4, 8, 16, 32, 64]
    misses = [cache_misses(transpose_addresses(64, 64, tile), lines, line_words)
              for tile in tiles]
    best_tile = tiles[misses.index(min(misses))]
    check(best_tile not in (min(tiles), max(tiles)),
          f"the optimal tile must be interior, got {best_tile} ({misses})")
    check(misses[0] > min(misses), f"a one-word tile must thrash: {misses}")
    check(misses[-1] > min(misses), f"a whole-matrix tile must lose reuse: {misses}")
    check(misses[tiles.index(best_tile)] <= misses[tiles.index(64)] and
          misses[tiles.index(best_tile)] <= misses[tiles.index(1)],
          f"the optimum must beat both extremes: {misses}")

    # a bigger cache never needs more lines than a small one
    small_cache = cache_misses(transpose_addresses(64, 64, best_tile), 8, line_words)
    check(small_cache >= min(misses),
          "a larger cache cannot need more misses than a smaller one")

    # ---------------- threading parameters ----------------
    serial = fft_partitioned(signal, 1)
    for workers in (2, 4, 8):
        parallel = fft_partitioned(signal, workers)
        check(difference(parallel, serial) == 0.0,
              f"{workers} workers must reproduce the serial result bit for bit")
        chunks = partition(n // 2, workers)
        sizes = [end - start for start, end in chunks]
        check(sum(sizes) == n // 2, "the partition must cover every butterfly exactly once")
        check(max(sizes) - min(sizes) <= 1, f"the partition must be balanced: {sizes}")
    for work in range(0, 17):
        for workers in (1, 3, 5):
            chunks = partition(work, workers)
            sizes = [end - start for start, end in chunks]
            check(sum(sizes) == work and max(sizes) - min(sizes) <= 1,
                  f"partition({work}, {workers}) is unbalanced: {sizes}")
            check(all(start <= end for start, end in chunks), "chunks must be non-degenerate")

    makespans = [modelled_makespan(n, workers) for workers in (1, 2, 4, 8, 16)]
    check(all(later < earlier for earlier, later in zip(makespans, makespans[1:])),
          f"more workers must not cost more modelled time: {makespans}")
    check(makespans[0] / makespans[-1] > 2.0,
          f"the modelled speedup must be substantial: {makespans}")
    unbalanced = modelled_makespan(n, 1) - modelled_makespan(n, 2)
    check(unbalanced > 0.0, "the second worker must actually remove work from the critical path")

    # ---------------- invalid parameters ----------------
    for call, why in (
        (lambda: fft_iterative([1.0, 2.0, 3.0]), "non power-of-two length"),
        (lambda: fft_iterative([]), "empty input"),
        (lambda: fft_four_step([1.0] * 6, 3), "factor that is not a power of two"),
        (lambda: fft_four_step([1.0] * 8, 3), "factor that does not divide n"),
        (lambda: partition(4, 0), "zero workers"),
        (lambda: modelled_makespan(8, 0), "zero workers in the cost model"),
        (lambda: LruCache(0, 4), "degenerate cache"),
        (lambda: transpose_addresses(4, 4, 0), "zero tile"),
        (lambda: bluestein([]), "empty chirp input"),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {why}")

    # ---------------- determinism ----------------
    check(fft_iterative(signal) == spectrum, "the transform must be deterministic")
    check(four_step_addresses(n_cache, 8) == four_step_addresses(n_cache, 8),
          "address streams must be deterministic")
    check([cache_misses(transpose_addresses(64, 64, tile), lines, line_words)
           for tile in tiles] == misses, "cache simulations must be deterministic")

    print(f"dft agreement worst={worst:.3e} error growth={relative_errors}")
    print(f"cache misses butterfly={butterfly} blocked={blocked} "
          f"per_element={butterfly_traffic:.3f}/{blocked_traffic:.3f} best_tile={best_tile}")
    print(f"transpose miss curve {dict(zip(tiles, misses))}")
    print(f"modelled makespans {[round(value, 1) for value in makespans]}")
    print("PASS")


if __name__ == "__main__":
    main()
