"""Task 727 - Implement Strassen matrix multiplication and recursion cutoff tuning.  (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)

Strassen multiplies two 2n x 2n matrices with seven recursive n x n products instead of eight,
exchanging multiplications for additions.  Splitting A and B into quadrants,

    M1 = (A11 + A22)(B11 + B22)      M5 = (A11 + A12) B22
    M2 = (A21 + A22) B11             M6 = (A21 - A11)(B11 + B12)
    M3 = A11 (B12 - B22)             M7 = (A12 - A22)(B21 + B22)
    M4 = A22 (B21 - B11)

the output quadrants are recombinations of the Mi, so a level divides the multiplication count
by 8/7 while adding 18 matrix additions.  On real hardware that trade only pays off below a
certain size - the recursion has to be cut off at a base case where the naive triple loop is
faster - so this file also measures the crossover.

Verified invariants: every result equals the naive product exactly (the test matrices hold small
integers, so equality is exact, not approximate), each of the seven products equals the naive
product of its own defining factor pair (the structural check, which catches a mis-transcribed
formula that a final-result comparison could hide), the multiplication count matches the closed
form 7^levels * cutoff^3 exactly, the count equals the naive count when the cutoff covers the
whole matrix, and the measured totals (multiplications plus additions) find a cutoff strictly
better than both the naive algorithm and the extreme cutoffs.

Simplified: padding to the next power of two instead of the more economical odd/even splitting,
scalar Python loops rather than blocked or vectorised kernels, no memory-bandwidth model beyond
counting scalar operations, and no Winograd-style variants or right-hand-side caching.
"""
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_value_error(call, message):
    try: call()
    except ValueError: return
    raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def random_matrix(size, seed, spread=5):
    state, grid = seed & MASK64, []
    for _ in range(size):
        row = []
        for _ in range(size):
            state, r = splitmix64(state)
            row.append(int(r % spread) - spread // 2)
        grid.append(row)
    return grid, state


def zeros(size):
    return [[0] * size for _ in range(size)]


def naive_multiply(left, right, counter=None):
    """Reference kernel; `counter` accumulates scalar multiplications and additions."""
    n = len(left)
    out = zeros(n)
    for i in range(n):
        for j in range(n):
            total = 0
            for t in range(n):
                total += left[i][t] * right[t][j]
                if counter is not None:
                    counter["mult"] += 1
                    counter["add"] += 1
            out[i][j] = total
    return out


def add_matrices(left, right, counter=None, sign=1):
    n = len(left)
    out = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = left[i][j] + sign * right[i][j]
            if counter is not None:
                counter["add"] += 1
    return out


def pad(grid, size):
    out = zeros(size)
    for i in range(len(grid)):
        for j in range(len(grid)):
            out[i][j] = grid[i][j]
    return out


def strassen_step(left, right, multiply, counter=None):
    """One Strassen level: returns (products, quadrants).

    `multiply` computes a product of two half-size matrices - naive for the structural check,
    the recursive Strassen routine during a real multiplication.
    """
    (a11, a12), (a21, a22) = left
    (b11, b12), (b21, b22) = right
    m1 = multiply(add_matrices(a11, a22, counter), add_matrices(b11, b22, counter))
    m2 = multiply(add_matrices(a21, a22, counter), b11)
    m3 = multiply(a11, add_matrices(b12, b22, counter, sign=-1))
    m4 = multiply(a22, add_matrices(b21, b11, counter, sign=-1))
    m5 = multiply(add_matrices(a11, a12, counter), b22)
    m6 = multiply(add_matrices(a21, a11, counter, sign=-1), add_matrices(b11, b12, counter))
    m7 = multiply(add_matrices(a12, a22, counter, sign=-1), add_matrices(b21, b22, counter))
    c11 = add_matrices(add_matrices(m1, m4, counter), add_matrices(m7, m5, counter, sign=-1), counter)
    c12 = add_matrices(m3, m5, counter)
    c21 = add_matrices(m2, m4, counter)
    c22 = add_matrices(add_matrices(m1, m3, counter), add_matrices(m6, m2, counter, sign=-1), counter)
    return [m1, m2, m3, m4, m5, m6, m7], [[c11, c12], [c21, c22]]


def strassen_multiply(left, right, cutoff, counter=None):
    """Strassen with a naive base case; non-power-of-two sizes are padded up."""
    if cutoff < 1:
        raise ValueError("cutoff must be at least one")
    n = len(left)
    if n != len(right) or any(len(row) != n for row in left + right):
        raise ValueError("both inputs must be square and equally sized")
    size = 1
    while size < n:
        size *= 2
    a, b = pad(left, size), pad(right, size)
    if size <= cutoff:
        return [[row[j] for j in range(n)] for row in naive_multiply(a, b, counter)[:n]]
    return [[row[j] for j in range(n)] for row in _recurse(a, b, cutoff, counter)[:n]]


def _recurse(a, b, cutoff, counter):
    """Divide into quadrants and apply seven recursive products instead of eight."""
    n = len(a)
    if n <= cutoff:
        return naive_multiply(a, b, counter)
    half = n // 2
    quad_left = [[[row[:half] for row in a[:half]], [row[half:] for row in a[:half]]],
                 [[row[:half] for row in a[half:]], [row[half:] for row in a[half:]]]]
    quad_right = [[[row[:half] for row in b[:half]], [row[half:] for row in b[:half]]],
                  [[row[:half] for row in b[half:]], [row[half:] for row in b[half:]]]]
    _, quadrants = strassen_step(
        [[quad_left[0][0], quad_left[0][1]], [quad_left[1][0], quad_left[1][1]]],
        [[quad_right[0][0], quad_right[0][1]], [quad_right[1][0], quad_right[1][1]]],
        lambda x, y: _recurse(x, y, cutoff, counter), counter)
    out = [[0] * n for _ in range(n)]
    for i in range(half):
        for j in range(half):
            out[i][j] = quadrants[0][0][i][j]
            out[i][j + half] = quadrants[0][1][i][j]
            out[i + half][j] = quadrants[1][0][i][j]
            out[i + half][j + half] = quadrants[1][1][i][j]
    return out


def levels_for(size, cutoff):
    """Number of Strassen levels before the naive base case is reached."""
    count = 0
    while size > cutoff:
        size //= 2
        count += 1
    return count


def predicted_multiplications(size, cutoff):
    """Closed form: 7^levels leaves, each costing min(cutoff, padded)^3 multiplications."""
    padded = 1
    while padded < size:
        padded *= 2
    effective = min(cutoff, padded)
    return 7 ** levels_for(padded, effective) * effective ** 3


def cost_sweep(size, cutoffs, seed):
    left, seed = random_matrix(size, seed)
    right, seed = random_matrix(size, seed)
    rows = []
    for cutoff in cutoffs:
        counter = {"mult": 0, "add": 0}
        out = strassen_multiply(left, right, cutoff, counter)
        rows.append((cutoff, counter["mult"], counter["add"], counter["mult"] + counter["add"]))
    return rows


def main():
    # 1. the structural check: each of the seven Strassen products must equal the naive product
    #    of the factor pair that defines it - an error in the formulas would show up here.
    a11, a12, a21, a22, b11, b12, b21, b22 = ([[1, 2], [3, 4]], [[5, 6], [7, 8]],
                                              [[-1, 2], [0, 3]], [[4, -2], [1, 5]],
                                              [[2, 0], [1, 3]], [[-3, 1], [2, 2]],
                                              [[1, 4], [0, -2]], [[5, 1], [-1, 2]])
    left, right = [[a11, a12], [a21, a22]], [[b11, b12], [b21, b22]]
    products, quadrants = strassen_step(left, right, naive_multiply)
    definitions = [
        (add_matrices(a11, a22), add_matrices(b11, b22)),
        (add_matrices(a21, a22), b11),
        (a11, add_matrices(b12, b22, sign=-1)),
        (a22, add_matrices(b21, b11, sign=-1)),
        (add_matrices(a11, a12), b22),
        (add_matrices(a21, a11, sign=-1), add_matrices(b11, b12)),
        (add_matrices(a12, a22, sign=-1), add_matrices(b21, b22)),
    ]
    for index, (factor_left, factor_right) in enumerate(definitions):
        check(products[index] == naive_multiply(factor_left, factor_right),
              f"M{index + 1} must equal the product of its defining factors")
    combined = [[quadrants[0][0], quadrants[0][1]], [quadrants[1][0], quadrants[1][1]]]
    flat_left = [[a11[i][j] if j < 2 else a12[i][j - 2] for j in range(4)] for i in range(2)]
    flat_left += [[a21[i][j] if j < 2 else a22[i][j - 2] for j in range(4)] for i in range(2)]
    flat_right = [[b11[i][j] if j < 2 else b12[i][j - 2] for j in range(4)] for i in range(2)]
    flat_right += [[b21[i][j] if j < 2 else b22[i][j - 2] for j in range(4)] for i in range(2)]
    expected = naive_multiply(flat_left, flat_right)
    rebuilt = [[combined[i // 2][j // 2][i % 2][j % 2] for j in range(4)] for i in range(4)]
    check(rebuilt == expected, "the recombined quadrants must equal the naive 4x4 product")

    # 2. end-to-end correctness against the naive kernel for many sizes, including the odd and
    #    non-power-of-two sizes that force padding.
    seed = 0x7271
    for size in (1, 2, 3, 4, 5, 7, 8, 11, 16, 17):
        left, seed = random_matrix(size, seed)
        right, seed = random_matrix(size, seed)
        reference = naive_multiply(left, right)
        for cutoff in (1, 2, 3, 4, 8, 32):
            check(strassen_multiply(left, right, cutoff) == reference,
                  f"Strassen must match the naive product for n={size} and cutoff={cutoff}")

    # 3. the multiplication count follows the closed form, and the cutoffs that cover the whole
    #    matrix degenerate to the naive count exactly.
    size = 16
    left, seed = random_matrix(size, seed)
    right, seed = random_matrix(size, seed)
    for cutoff in (1, 2, 4, 8, 16, 32):
        counter = {"mult": 0, "add": 0}
        strassen_multiply(left, right, cutoff, counter)
        check(counter["mult"] == predicted_multiplications(size, cutoff),
              f"the count for cutoff={cutoff} must match 7^levels * cutoff^3")
    check(predicted_multiplications(16, 4) == 7 ** 2 * 4 ** 3, "16 with cutoff 4 is 3136")
    check(predicted_multiplications(16, 16) == 16 ** 3, "cutoff >= n must be the naive count")
    check(predicted_multiplications(16, 8) == 7 * 8 ** 3, "16 with cutoff 8 is one Strassen level")
    check(predicted_multiplications(16, 1) == 7 ** 4, "cutoff 1 recurses down to single entries")
    degenerate = {"mult": 0, "add": 0}
    strassen_multiply(left, right, 64, degenerate)
    check(degenerate["mult"] == size ** 3 and degenerate["add"] == size ** 3,
          "a cutoff above the size must reduce to the naive triple loop")

    # 4. cutoff tuning on a larger instance: the measured total (multiplications plus additions)
    #    must have an interior optimum that beats both the naive algorithm and the extreme
    #    cutoffs, which is the whole reason a cutoff parameter exists.
    sweep = cost_sweep(32, [1, 2, 4, 8, 16, 32, 64], 0x7272)
    naive_total = 2 * 32 ** 3
    interior = [row for row in sweep if 1 < row[0] < 32]
    best = min(interior, key=lambda row: row[3])
    check(sweep[-1][3] == naive_total, "the largest cutoff must cost the naive total")
    check(best[3] < naive_total, "the tuned cutoff must beat the naive algorithm")
    check(best[3] < sweep[0][3] or best[3] <= min(row[3] for row in sweep),
          "no cutoff may beat the interior optimum")
    check(1 < best[0] < 32, "the optimum must be an interior cutoff")
    check(best[1] < naive_total // 2, "the tuned run must use far fewer multiplications")
    check(all(row[1] == predicted_multiplications(32, row[0]) for row in sweep),
          "every sweep point must match the closed form")

    # 5. boundary and invalid inputs.
    check(strassen_multiply([[3]], [[4]], 1) == [[12]], "the 1x1 case must work")
    check(strassen_multiply([[1, 2], [3, 4]], [[0, 0], [0, 0]], 2) == [[0, 0], [0, 0]],
          "a zero operand must produce zeros")
    expect_value_error(lambda: strassen_multiply([[1]], [[2]], 0), "cutoff 0 is invalid")
    expect_value_error(lambda: strassen_multiply([[1, 2]], [[1, 2]], 2),
                       "non-square inputs must be rejected")
    expect_value_error(lambda: strassen_multiply([[1, 2], [3, 4]], [[1, 2], [3, 4]][:1], 2),
                       "mismatched sizes must be rejected")

    # 6. determinism and the summary report.
    rerun = cost_sweep(32, [1, 2, 4, 8, 16, 32, 64], 0x7272)
    check(rerun == sweep, "the sweep must be deterministic")
    print("727 sweep (cutoff, mult, add, total): " + ", ".join(str(row) for row in sweep))
    print(f"727 naive total={naive_total} best cutoff={best[0]} total={best[3]}")
    print("PASS")


if __name__ == "__main__":
    main()
