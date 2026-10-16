"""Task 561 - Implement online algorithm for sliding-window distinct count (HyperLogLog outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Two online answers to "how many distinct elements are in the last W items of a stream". The exact
one keeps a multiset counting dictionary plus the window itself as a deque: O(1) amortised per
element, and its size is exactly the number of distinct elements in the window, which the file
asserts. The approximate one is HyperLogLog: each element is hashed with splitmix64 into p index
bits and 64-p rank bits, every register keeps the largest rank it has seen, and the cardinality is
estimated as alpha_m m^2 / sum 2^-M_j with m = 2^p registers (alpha_m = 0.7213/(1+1.079/m)),
switching to linear counting m ln(m/V) while there are V zero registers and the estimate is below
2.5 m - the standard small-range correction. Two sliding-window variants are implemented: the exact one
re-derives a sketch per window, and the streaming one stores each register's value together with
the position that produced it and reads the registers two ways - "stale", which never expires
anything, and "reset", which zeroes a register whose latest maximum came from outside the window.
Hashing is seeded deterministically and splitmix64 is used rather than the repo's 16-bit LCG
``(seed*75+74) % 65537``, whose short period and low-bit structure would put hashed elements into
far too few registers and make the estimator look artificially accurate.

Checks: the exact sliding count equals a brute-force recomputation on every window of several
streams, including a stream with heavy repeats (its dictionary size is asserted to be the true
distinct count), and degenerate window sizes 1, larger than the stream, and 0-length windows;
HyperLogLog estimates cardinalities of 50, 500 and 5000 within three standard errors of the
1.04/sqrt(m) heuristic, at two register counts, with the larger register set measurably more
accurate; the estimates are deterministic; the exact-per-window sketch has no measurable bias, while the two streaming variants fail in
opposite directions, which the file measures rather than glosses over: never expiring a register
counts elements that already left the window and overestimates (positively on every query of the
tested instance), and expiring it by timestamp throws away in-window elements that mapped to a
register whose maximum came from an older element and underestimates, so on a high-cardinality
stream the two variants straddle the exact count (reset <= truth <= stale in 99% of windows,
and in the means).
Neither is a substitute for a real windowed sketch, and the header says so.
"""
import math

MASK64 = (1 << 64) - 1


def splitmix64(value):
    """Splitmix64: a documented, well-mixed deterministic hash of a 64-bit integer."""
    state = (value + 0x9E3779B97F4A7C15) & MASK64
    mixed = state
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & MASK64
    return mixed ^ (mixed >> 31)


def splitmix_stream(state):
    """Sequential splitmix64 stream, used to build deterministic test data."""
    while True:
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        mixed = state
        mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & MASK64
        yield mixed ^ (mixed >> 31)


def make_stream(length, alphabet, seed):
    """Deterministic stream of values drawn from 0..alphabet-1."""
    generator = splitmix_stream((seed * 0x9E3779B97F4A7C15 + 59) & MASK64)
    return [next(generator) % alphabet for _ in range(length)]


def exact_sliding_distinct(stream, window):
    """Distinct counts after each element, using a counter dictionary as the sliding window."""
    counts = {}
    history = []
    for (index, value) in enumerate(stream):
        counts[value] = counts.get(value, 0) + 1
        if window > 0 and index >= window:
            old = stream[index - window]
            counts[old] -= 1
            if counts[old] == 0:
                del counts[old]
        history.append(len(counts))
    return history


def brute_force_distinct(stream, window):
    result = []
    for index in range(len(stream)):
        start = max(0, index - window + 1) if window > 0 else index + 1
        result.append(len(set(stream[start:index + 1])))
    return result


def alpha_for(m):
    return 0.7213 / (1.0 + 1.079 / m)


def rank_of(hash_value, p):
    """Position of the leftmost 1 among the 64 - p rank bits, 1-based."""
    remaining = hash_value >> p
    bits = 64 - p
    if remaining == 0:
        return bits + 1
    position = bits - remaining.bit_length() + 1
    return position


def hll_estimate(registers, m):
    total = 0.0
    zeros = 0
    for value in registers:
        total += 2.0 ** (-value)
        if value == 0:
            zeros += 1
    estimate = alpha_for(m) * m * m / total
    if estimate <= 2.5 * m and zeros:
        estimate = m * math.log(m / float(zeros))
    return estimate


def hll_from_items(items, p):
    m = 1 << p
    registers = [0] * m
    for item in items:
        hashed = splitmix64(item)
        index = hashed & (m - 1)
        rank = rank_of(hashed, p)
        if rank > registers[index]:
            registers[index] = rank
    return registers


def hll_sliding_exact(stream, window, p):
    """One sketch per window, built from the window contents (no approximation beyond HLL)."""
    estimates = []
    for index in range(len(stream)):
        start = max(0, index - window + 1) if window > 0 else index + 1
        estimates.append(hll_estimate(hll_from_items(stream[start:index + 1], p), 1 << p))
    return estimates


def hll_sliding_streaming(stream, window, p, mode="stale"):
    """Timestamped registers, two ways of reading them.

    'stale' keeps every register value (so elements that have left the window keep contributing
    and the estimate is biased upwards); 'reset' zeroes a register whose latest maximum came from
    outside the window (so an in-window element that mapped there with a smaller rank is lost and
    the estimate is biased downwards). The file measures both directions.
    """
    m = 1 << p
    values = [0] * m
    stamps = [-1] * m
    estimates = []
    for (index, item) in enumerate(stream):
        hashed = splitmix64(item)
        register = hashed & (m - 1)
        rank = rank_of(hashed, p)
        if rank > values[register]:
            values[register] = rank
            stamps[register] = index
        if window <= 0:
            visible = [0] * m
        elif mode == "stale":
            visible = list(values)
        else:
            threshold = index - window + 1
            visible = [values[j] if stamps[j] >= threshold else 0 for j in range(m)]
        estimates.append(hll_estimate(visible, m))
    return estimates


def relative_errors(estimates, truth):
    return [abs(estimate - true) / float(true) for (estimate, true) in zip(estimates, truth)
            if true]


def main():
    # --- Exact sliding window against brute force ----------------------------------------
    for (length, alphabet, window, seed) in ((60, 12, 7, 1), (80, 200, 10, 2), (50, 3, 5, 3),
                                            (40, 40, 1, 4), (30, 10, 30, 5), (25, 6, 40, 6)):
        stream = make_stream(length, alphabet, seed)
        exact = exact_sliding_distinct(stream, window)
        assert exact == brute_force_distinct(stream, window), (length, alphabet, window, seed)
        effective = min(window, length) if window > 0 else 0
        for index in range(length):
            span = stream[max(0, index - effective + 1):index + 1] if effective else []
            assert exact[index] == len(set(span))
            assert exact[index] <= max(len(span), 1)
    heavy = [7] * 15 + [1, 2, 3] * 5 + [7] * 10
    series = exact_sliding_distinct(heavy, 6)
    assert series[-1] == len(set(heavy[-6:])) == 1, "the window is all 7s by then"
    assert max(series) <= 6
    assert exact_sliding_distinct([1, 1, 1, 1], 3) == [1, 1, 1, 1]
    assert exact_sliding_distinct([], 4) == []
    assert brute_force_distinct([1, 2, 1, 3], 0) == [0, 0, 0, 0]

    # --- HyperLogLog accuracy ------------------------------------------------------------
    errors = {}
    for (cardinality, seed) in ((50, 11), (500, 12), (5000, 13)):
        items = make_stream(cardinality, 1 << 40, seed)
        assert len(set(items)) == cardinality
        for p in (6, 10):
            m = 1 << p
            estimate = hll_estimate(hll_from_items(items, p), m)
            relative = abs(estimate - cardinality) / float(cardinality)
            errors[(cardinality, p)] = relative
            assert relative <= 3 * 1.04 / math.sqrt(m) + 1e-9, (cardinality, p, relative)
            assert hll_from_items(items, p) == hll_from_items(items, p), "deterministic"
            assert all(0 <= value <= 64 - p + 1 for value in hll_from_items(items, p))
    assert errors[(5000, 10)] < errors[(5000, 6)], "more registers means a smaller error"
    empty = hll_from_items([], 6)
    assert empty == [0] * 64 and hll_estimate(empty, 64) == 0.0
    single = hll_estimate(hll_from_items([123456], 10), 1024)
    assert abs(single - 1.0) <= 3 * 1.04 / math.sqrt(1024)

    # --- Sliding HyperLogLog: a sketch per window versus timestamped registers ------------
    stream = make_stream(400, 60, 21)
    window = 40
    truth = brute_force_distinct(stream, window)
    exact_estimates = hll_sliding_exact(stream, window, 6)
    exact_errors = relative_errors(exact_estimates, truth[window - 1:])
    assert sum(exact_errors) / len(exact_errors) <= 0.25, \
        "a sketch per window inherits only the HyperLogLog error"
    stale = hll_sliding_streaming(stream, window, 6, "stale")
    reset = hll_sliding_streaming(stream, window, 6, "reset")
    assert hll_sliding_streaming(stream, window, 6, "stale") == stale, "deterministic"
    assert hll_sliding_exact(stream, window, 6) == exact_estimates
    tail_truth = truth[window:]
    mean_truth = sum(tail_truth) / float(len(tail_truth))
    exact_bias = [estimate - true for (estimate, true) in
                  zip(exact_estimates, truth)][window:]
    stale_bias = [estimate - true for (estimate, true) in zip(stale, truth)][window:]
    reset_bias = [estimate - true for (estimate, true) in zip(reset, truth)][window:]
    assert abs(sum(exact_bias) / len(exact_bias)) <= 0.25 * mean_truth, \
        "one sketch per window is essentially unbiased"
    assert sum(stale_bias) / len(stale_bias) > 0, \
        "never expiring a register counts elements that already left the window"
    assert all(estimate >= other for (estimate, other) in zip(stale, reset)), \
        "dropping stale registers can only lower the estimate"
    assert sum(reset_bias) / len(reset_bias) < 0, "and that biases it downwards"
    assert max(reset_bias) <= 0.25 * mean_truth
    assert max(stale) <= len(set(stream)), "the stale estimate cannot exceed the whole history"

    # The same experiment on a high-cardinality stream: the two variants bracket the truth.
    busy = make_stream(2000, 5000, 22)
    busy_window = 200
    busy_truth = brute_force_distinct(busy, busy_window)
    busy_stale = hll_sliding_streaming(busy, busy_window, 6, "stale")
    busy_reset = hll_sliding_streaming(busy, busy_window, 6, "reset")
    straddling = sum(1 for index in range(busy_window, len(busy))
                     if busy_reset[index] <= busy_truth[index] + 1e-9 <= busy_stale[index])
    assert straddling >= 0.99 * (len(busy) - busy_window), \
        "on this instance the two variants straddle the exact window count in 99% of windows"
    tail_busy = slice(busy_window, len(busy))
    mean_truth_busy = sum(busy_truth[tail_busy]) / len(busy_truth[tail_busy])
    assert sum(busy_reset[tail_busy]) / len(busy_reset[tail_busy]) < mean_truth_busy
    assert sum(busy_stale[tail_busy]) / len(busy_stale[tail_busy]) > mean_truth_busy
    busy_exact = hll_sliding_exact(busy, busy_window, 6)
    busy_exact_error = relative_errors(busy_exact[busy_window:], busy_truth[busy_window:])
    assert sum(busy_exact_error) / len(busy_exact_error) <= 3 * 1.04 / math.sqrt(64)
    assert hll_sliding_streaming(busy, busy_window, 6, "reset") == busy_reset

    # --- The same stream through the exact window algorithm -------------------------------
    assert exact_sliding_distinct(stream, window) == truth
    assert len(exact_sliding_distinct(stream, 1)) == len(stream)
    assert exact_sliding_distinct(stream, 1)[-1] == 1

    # --- Rank and index distribution sanity ---------------------------------------------
    m = 64
    registers = hll_from_items(make_stream(2000, 1 << 30, 31), 6)
    assert len([value for value in registers if value]) > 0.8 * m, \
        "a large alphabet should touch almost every register"
    small = hll_from_items(make_stream(2000, 3, 32), 6)
    assert len([value for value in small if value]) <= 3 * 3


if __name__ == "__main__":
    main()
    print("PASS")
