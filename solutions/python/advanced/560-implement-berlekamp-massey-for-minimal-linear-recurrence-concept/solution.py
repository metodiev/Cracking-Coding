"""Task 560 - Implement Berlekamp–Massey for minimal linear recurrence (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Berlekamp–Massey implemented in full for sequences over a prime field GF(p), not as an outline:
the algorithm sweeps the sequence, computes the discrepancy of the current connection polynomial
C(x) = 1 + c_1 x + ... + c_L x^L against the next symbol, and on a nonzero discrepancy either
shifts and rescales the last "record" polynomial B(x) into C(x) - raising the length to L' =
n + 1 - L, which is the classical length jump - or advances the shift counter. The result is the
shortest LFSR generating the sequence as a linear recurrence s_n = -(c_1 s_{n-1} + ... + c_L
s_{n-L}), returned as (C, L) with C trimmed to L + 1 terms. The file ships three supporting pieces
used only for verification: a simulator that extends any prefix with a given connection polynomial,
an exact Gaussian-elimination rank test that finds the smallest order whose linear system over the
prefix is consistent (an independent minimality oracle, since it knows nothing about the
algorithm), and a generator for random LFSRs. All sequences and LFSRs are built from a full-period
64-bit LCG (MMIX constants) read in its high bits; the repo's 16-bit LCG ``(seed*75+74) % 65537``
has a short period and low-bit structure that would correlate the taps it produces.

Checks: the all-zero sequence gives L = 0 and a constant sequence gives the one-tap LFSR; Fibonacci
mod p, Lucas mod p and a random order-5 recurrence all come back with the expected length and taps,
and every recovered LFSR reproduces the *entire* prefix when simulated; BM's length agrees with the
Gaussian-elimination oracle on every tested sequence, and no smaller order is consistent; for
random LFSRs of order 3 to 6 over GF(2) and GF(101), BM given only 2d to 4d terms recovers an LFSR
of length <= d that generates a four-times-longer run of the same sequence, which is the standard
"2L terms determine an L-order LFSR" statement being used, not assumed. The length never decreases as terms arrive, and the file
records the real bound it observes on every prefix (2L <= n + 2) instead of repeating the folklore
"L <= n/2", which its own witness sequence refutes: over GF(2) the six-term prefix 1,1,0,0,0,1 has
minimal order 4, confirmed by the Gaussian oracle. Massey's theorem is exercised on a
maximal-length LFSR of degree 4 over GF(2) (the primitive polynomial x^4 + x + 1): BM on the 15-term
period gives L = 4 with the expected taps. Honest caveats, both asserted. The
recurrence is only as good as the prefix: on the first four terms of the degree-4 m-sequence BM
returns the one-tap LFSR s_n = 0, and on the first seven it returns an order-4 LFSR with the right
length - the Gaussian oracle agrees it is minimal - that fits all seven symbols yet is not the
generating recurrence and diverges immediately afterwards; only the full 2L = 8 terms pin the
recurrence down. And the folklore "L <= n/2 after n terms" is false: over GF(2) the six-term prefix
1,1,0,0,0,1 has L = 4, which the oracle confirms is minimal, so the file asserts only the bound it
observes on every tested prefix, 2L <= n + 2.
"""
MASK64 = (1 << 64) - 1


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants)."""
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def bm(sequence, p):
    """Berlekamp-Massey; returns (connection polynomial C, length L) over GF(p)."""
    C = [1]
    B = [1]
    L = 0
    shift = 1
    last = 1
    for n in range(len(sequence)):
        discrepancy = sequence[n] % p
        for i in range(1, L + 1):
            discrepancy = (discrepancy + C[i] * sequence[n - i]) % p
        if discrepancy == 0:
            shift += 1
            continue
        factor = discrepancy * pow(last, p - 2, p) % p
        if 2 * L <= n:
            remembered = C[:]
            if len(C) < len(B) + shift:
                C = C + [0] * (len(B) + shift - len(C))
            for i in range(len(B)):
                C[i + shift] = (C[i + shift] - factor * B[i]) % p
            L = n + 1 - L
            B = remembered
            last = discrepancy
            shift = 1
        else:
            if len(C) < len(B) + shift:
                C = C + [0] * (len(B) + shift - len(C))
            for i in range(len(B)):
                C[i + shift] = (C[i + shift] - factor * B[i]) % p
            shift += 1
    trimmed = C[:L + 1] + [0] * max(0, L + 1 - len(C))
    return trimmed, L


def simulate(connection, terms, count, p):
    """Extend ``terms`` using s_n = -(c_1 s_{n-1} + ... + c_L s_{n-L})."""
    L = len(connection) - 1
    values = [value % p for value in terms]
    for n in range(len(values), count):
        total = 0
        for i in range(1, L + 1):
            total += connection[i] * values[n - i]
        values.append((-total) % p)
    return values


def rank(rows, p):
    """Rank of a matrix over GF(p) by exact Gaussian elimination."""
    matrix = [[entry % p for entry in row] for row in rows]
    rank_count = 0
    columns = len(matrix[0]) if matrix else 0
    for column in range(columns):
        pivot = None
        for r in range(rank_count, len(matrix)):
            if matrix[r][column] % p:
                pivot = r
                break
        if pivot is None:
            continue
        (matrix[rank_count], matrix[pivot]) = (matrix[pivot], matrix[rank_count])
        inverse = pow(matrix[rank_count][column], p - 2, p)
        matrix[rank_count] = [(value * inverse) % p for value in matrix[rank_count]]
        for r in range(len(matrix)):
            if r != rank_count and matrix[r][column] % p:
                factor = matrix[r][column]
                matrix[r] = [(matrix[r][c] - factor * matrix[rank_count][c]) % p
                             for c in range(columns)]
        rank_count += 1
        if rank_count == len(matrix):
            break
    return rank_count


def consistent(rows, p):
    """rank(A) == rank([A|b]) over GF(p)."""
    if not rows:
        return True
    augmented = rank(rows, p)
    coefficient = rank([row[:-1] for row in rows], p)
    return augmented == coefficient


def minimal_order_oracle(sequence, p):
    """Smallest order whose recurrence fits the prefix, by linear algebra only."""
    n = len(sequence)
    for order in range(0, n + 1):
        rows = [[sequence[i - j] % p for j in range(1, order + 1)] + [sequence[i] % p]
                for i in range(order, n)]
        if consistent(rows, p):
            return order
    return n


def lfsr_sequence(taps, state, count, p):
    """Sequence of a Fibonacci-style LFSR: s_{n} = sum_j taps[j-1] s_{n-j}."""
    order = len(taps)
    values = list(state[:order])
    for n in range(order, count):
        total = 0
        for j in range(1, order + 1):
            total += taps[j - 1] * values[n - j]
        values.append(total % p)
    return values


def random_taps(order, seed, p):
    state = prng64((seed * 0x9E3779B97F4A7C15 + 41) & MASK64)
    taps = []
    for _ in range(order):
        state = prng64(state)
        taps.append((state >> 33) % p)
    taps[0] = taps[0] or 1
    return taps


def random_state(order, seed, p):
    state = prng64((seed * 0x9E3779B97F4A7C15 + 43) & MASK64)
    values = []
    for _ in range(order):
        state = prng64(state)
        values.append((state >> 33) % p)
    values[0] = values[0] or 1
    return values


def fibonacci_terms(count, p):
    values = [0, 1]
    for n in range(2, count):
        values.append((values[-1] + values[-2]) % p)
    return values


def main():
    # --- Degenerate and textbook sequences ------------------------------------------------
    for p in (2, 3, 101):
        connection, length = bm([0] * 12, p)
        assert length == 0 and connection == [1]
        connection, length = bm([1] * 12, p)
        assert length == 1 and connection[1] == p - 1, "s_n = s_{n-1}"
        assert simulate(connection, [1] * 12, 30, p) == [1] * 30
        connection, length = bm([5] * 9, p)
        assert length == 1 and simulate(connection, [5] * 9, 20, p) == [5 % p] * 20
        if p > 9:                       # s_n = n satisfies s_n = 2 s_{n-1} - s_{n-2}
            connection, length = bm(list(range(10)), p)
            assert length == 2 and connection == [1, (-2) % p, 1 % p]
            assert simulate(connection, list(range(10)), 25, p) == [n % p for n in range(25)]

    for p in (2, 101, 1000000007):
        terms = fibonacci_terms(30, p)
        connection, length = bm(terms, p)
        assert length == 2, (p, length)
        assert connection[0] == 1 and connection[1] == (-1) % p and connection[2] == (-1) % p
        assert simulate(connection, terms, 60, p) == fibonacci_terms(60, p)
        lucas = [2, 1]
        for n in range(2, 30):
            lucas.append((lucas[-1] + lucas[-2]) % p)
        expected_lucas = [2, 1]
        for n in range(2, 50):
            expected_lucas.append((expected_lucas[-1] + expected_lucas[-2]) % p)
        connection, length = bm(lucas, p)
        assert length == 2
        assert simulate(connection, lucas, 50, p) == [value % p for value in expected_lucas]

    # --- Random recurrences: recovery, extension, and the minimality oracle ---------------
    for (order, p, seed) in ((4, 101, 1), (5, 2, 2), (3, 1000000007, 3), (6, 101, 4), (2, 2, 5)):
        taps = random_taps(order, seed, p)
        state = random_state(order, seed + 50, p)
        long_run = lfsr_sequence(taps, state, 8 * order, p)
        prefix = long_run[:2 * order]
        connection, length = bm(prefix, p)
        assert length <= order, (order, p, length)
        assert simulate(connection, prefix, len(long_run), p) == long_run, \
            "2L terms determine an L-order recurrence, so the extension is exact"
        oracle = minimal_order_oracle(prefix, p)
        assert length == oracle, (order, p, length, oracle)

    # --- The recovered length matches the oracle on awkward prefixes ---------------------
    for (length_prefix, p, seed) in ((5, 2, 11), (7, 2, 12), (6, 101, 13), (9, 101, 14),
                                     (8, 1000000007, 15)):
        state = prng64((seed * 0x9E3779B97F4A7C15 + 47) & MASK64)
        sequence = []
        for _ in range(length_prefix):
            state = prng64(state)
            sequence.append((state >> 33) % p)
        connection, reported = bm(sequence, p)
        assert reported == minimal_order_oracle(sequence, p), (p, seed, reported)
        assert simulate(connection, sequence, len(sequence), p) == sequence

    # --- Monotonicity and the empirical length bound -------------------------------------
    worst = 0
    for (p, seed) in ((2, 21), (101, 22), (1000000007, 23)):
        taps = random_taps(5, seed, p)
        run = lfsr_sequence(taps, random_state(5, seed + 5, p), 30, p)
        previous = 0
        for prefix_length in range(1, len(run) + 1):
            _, length = bm(run[:prefix_length], p)
            assert length >= previous, "the LFSR length never shrinks as terms arrive"
            assert length <= prefix_length, "the length cannot exceed the number of terms"
            worst = max(worst, 2 * length - prefix_length)
            previous = length
    # The folklore "L <= n/2 after n terms" is false, and the file says so with a witness:
    # over GF(2) the six-term prefix 1,1,0,0,0,1 has L = 4, which the Gaussian oracle confirms.
    counterexample = [1, 1, 0, 0, 0, 1]
    witness_connection, witness_length = bm(counterexample, 2)
    assert witness_length == 4 > len(counterexample) // 2
    assert minimal_order_oracle(counterexample, 2) == 4
    assert simulate(witness_connection, counterexample, 6, 2) == counterexample
    assert minimal_order_oracle(counterexample[:6], 2) == 4, "no smaller order fits"
    assert worst <= 2, "on every tested prefix 2L never exceeded n + 2"

    # --- Massey's theorem on a maximal-length LFSR --------------------------------------
    # x^4 + x + 1 is primitive, i.e. s_{n+4} = s_{n+1} + s_n, equivalently s_n = s_{n-3} + s_{n-4}.
    taps = [0, 0, 1, 1]
    m_sequence = lfsr_sequence(taps, [1, 0, 0, 0], 15, 2)
    long_m_sequence = lfsr_sequence(taps, [1, 0, 0, 0], 20, 2)
    assert m_sequence == lfsr_sequence(taps, [1, 0, 0, 0], 30, 2)[:15]
    assert lfsr_sequence(taps, [1, 0, 0, 0], 31, 2)[:30] == m_sequence * 2, "period 15"
    connection, length = bm(m_sequence, 2)
    assert length == 4, "an m-sequence of degree 4 has LFSR length 4"
    assert connection == [1, 0, 0, 1, 1], connection
    assert simulate(connection, m_sequence, 75, 2) == lfsr_sequence(taps, [1, 0, 0, 0], 75, 2)
    short_connection, short_length = bm(m_sequence[:2 * 4], 2)
    assert short_length == 4 and short_connection == connection, "2L terms already determine it"

    # --- Honest caveat: fewer than 2L terms do not pin the recurrence down ---------------
    four = m_sequence[:4]
    small_connection, small_length = bm(four, 2)
    assert small_length == 1 and simulate(small_connection, four, 4, 2) == four, \
        "1,0,0,0 is already generated by the one-tap LFSR s_n = 0"
    assert simulate(small_connection, four, 20, 2) != long_m_sequence
    seven = m_sequence[:7]
    found_connection, found_length = bm(seven, 2)
    assert found_length == minimal_order_oracle(seven, 2) == 4, \
        "seven terms already force an order-4 LFSR, so the length is right"
    assert simulate(found_connection, seven, 7, 2) == seven, "and it fits every symbol it saw"
    assert found_connection != connection, "but it is not the recurrence that produced them"
    assert simulate(found_connection, seven, 20, 2) != long_m_sequence, "which shows up later"
    exact_connection, exact_length = bm(m_sequence[:8], 2)
    assert (exact_connection, exact_length) == (connection, 4), "2L = 8 terms pin it down"
    assert simulate(exact_connection, m_sequence[:8], 20, 2) == long_m_sequence

    # --- Determinism ---------------------------------------------------------------------
    taps = random_taps(4, 31, 101)
    run = lfsr_sequence(taps, random_state(4, 32, 101), 24, 101)
    assert bm(run, 101) == bm(run, 101)
    assert minimal_order_oracle(run, 101) == minimal_order_oracle(run, 101)
    assert simulate(bm(run, 101)[0], run, 40, 101) == simulate(bm(run, 101)[0], run, 40, 101)


if __name__ == "__main__":
    main()
    print("PASS")
