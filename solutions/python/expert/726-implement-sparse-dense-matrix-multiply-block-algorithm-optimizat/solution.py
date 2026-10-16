"""Task 726 - Implement sparse-dense matrix multiply block algorithm optimizations.  (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)

Sparse-dense product C = A @ B with A an m x k sparse matrix in CSR and B a dense k x n matrix
is the inner loop of every sparse layer in a neural network.  The naive form walks the nonzeros
of one CSR row and adds value * B[row, :] into the output row, so each nonzero touches n
elements of B: nnz * n bytes of B traffic, mostly re-read over and over.

Repacking the loop nests over blocks fixes the locality: a block of rows of A that share a
common column block reuses exactly one k_block x n_block tile of B, which is read once into a
local accumulator.  The blocked variant here does exactly that and counts the B elements it
touches, so the improvement is measured rather than asserted.

Verified invariants: the blocked result equals the naive result equals a dense reference (the
test matrices hold small integers, so the comparison is exact), the blocked traversal reads
strictly fewer B elements than the naive traversal while performing the same number of scalar
multiply-adds (a real change of data reuse, not of arithmetic), the CSR invariant
(row_ptr monotone, indices inside their row and sorted, values aligned with indices) is checked
by an independent validator that also rejects malformed matrices, and the boundary cases (no
nonzeros, a single row, a single column, a fully dense CSR) all behave.

Simplified versus a tuned kernel (cuSPARSE spmm, MKL sparse BLAS): scalar loops with explicit
index arithmetic instead of register blocking, SIMD or a compressed sparse block format, no
multi-threaded partitioning, no sparse accumulator (here the accumulator is a dense row block),
and a cache-access model that counts element touches rather than modelling the hierarchy.
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


class CSR:
    """Compressed sparse row matrix with an explicit structural validator."""

    def __init__(self, rows, cols, row_ptr, col_idx, values):
        self.rows = rows
        self.cols = cols
        self.row_ptr = list(row_ptr)
        self.col_idx = list(col_idx)
        self.values = list(values)

    def nnz(self):
        return len(self.values)

    def validate(self):
        if len(self.row_ptr) != self.rows + 1:
            raise ValueError("row_ptr must have rows + 1 entries")
        if self.row_ptr[0] != 0 or self.row_ptr[-1] != self.nnz():
            raise ValueError("row_ptr must start at zero and end at nnz")
        if any(self.row_ptr[i] > self.row_ptr[i + 1] for i in range(self.rows)):
            raise ValueError("row_ptr must be non-decreasing")
        if len(self.col_idx) != self.nnz():
            raise ValueError("col_idx must align with values")
        for row in range(self.rows):
            start, end = self.row_ptr[row], self.row_ptr[row + 1]
            for position in range(start, end):
                if not 0 <= self.col_idx[position] < self.cols:
                    raise ValueError("column index out of range")
                if position > start and self.col_idx[position - 1] >= self.col_idx[position]:
                    raise ValueError("indices inside a row must be strictly increasing")
        return True

    def dense(self):
        grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for row in range(self.rows):
            for position in range(self.row_ptr[row], self.row_ptr[row + 1]):
                grid[row][self.col_idx[position]] = self.values[position]
        return grid

    def row(self, index):
        return [(self.col_idx[p], self.values[p])
                for p in range(self.row_ptr[index], self.row_ptr[index + 1])]


def from_dense(grid):
    """Build a CSR from a dense grid; zeros are dropped, so a dense grid yields a full CSR."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    row_ptr, col_idx, values = [0], [], []
    for row in grid:
        for column, value in enumerate(row):
            if value != 0:
                col_idx.append(column)
                values.append(value)
        row_ptr.append(len(values))
    return CSR(rows, cols, row_ptr, col_idx, values)


def dense_reference(left, right):
    """Plain triple loop over dense grids - the ground truth for the sparse kernels."""
    rows, inner, cols = len(left), len(right), len(right[0])
    out = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(inner):
            if left[i][j] == 0:
                continue
            for c in range(cols):
                out[i][c] += left[i][j] * right[j][c]
    return out


def spmm_naive(matrix, right):
    """Row-wise kernel: every nonzero reads one whole row of B."""
    matrix.validate()
    cols = len(right[0]) if right else 0
    out = [[0 for _ in range(cols)] for _ in range(matrix.rows)]
    counter = {"b_reads": 0, "fma": 0}
    for row in range(matrix.rows):
        for column, value in matrix.row(row):
            counter["b_reads"] += cols
            for c in range(cols):
                counter["fma"] += 1
                out[row][c] += value * right[column][c]
    return out, counter


def spmm_blocked(matrix, right, row_block=4, col_block=8, inner_block=4):
    """Blocked kernel: row block x inner block x column block, so one B tile serves the block.

    The tile B[inner block, column block] is loaded once and reused by every row of the row
    block that has a nonzero in that inner block, which is where the traffic saving comes from.
    """
    matrix.validate()
    for name, value in (("row_block", row_block), ("col_block", col_block),
                        ("inner_block", inner_block)):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    cols = len(right[0]) if right else 0
    out = [[0 for _ in range(cols)] for _ in range(matrix.rows)]
    counter = {"b_reads": 0, "fma": 0}
    for row_start in range(0, matrix.rows, row_block):
        row_end = min(row_start + row_block, matrix.rows)
        rows = list(range(row_start, row_end))
        for inner_start in range(0, matrix.cols, inner_block):
            inner_end = min(inner_start + inner_block, matrix.cols)
            block_columns = {}
            for row in rows:
                for column, value in matrix.row(row):
                    if inner_start <= column < inner_end:
                        block_columns.setdefault(column, []).append((row, value))
            if not block_columns:
                continue
            for col_start in range(0, cols, col_block):
                col_end = min(col_start + col_block, cols)
                width = col_end - col_start
                for column, entries in sorted(block_columns.items()):
                    counter["b_reads"] += width          # one tile segment, reused below
                    segment = right[column][col_start:col_end]
                    for row, value in entries:
                        line = out[row]
                        for offset in range(width):
                            counter["fma"] += 1
                            line[col_start + offset] += value * segment[offset]
    return out, counter


def random_sparse(rows, cols, density, seed, spread=5):
    """Deterministic sparse grid, then CSR-ified."""
    state, grid = seed & MASK64, []
    for _ in range(rows):
        line = []
        for _ in range(cols):
            state, r = splitmix64(state)
            line.append((r % spread) + 1 if (r >> 8) % 100 < density else 0)
        grid.append(line)
    return from_dense(grid), grid, state


def random_dense(rows, cols, seed, spread=7):
    state, grid = seed & MASK64, []
    for _ in range(rows):
        line = []
        for _ in range(cols):
            state, r = splitmix64(state)
            line.append((r % spread) - spread // 2)
        grid.append(line)
    return grid, state


def main():
    # 1. CSR structure: the validator accepts a well-formed matrix and the dense round-trip is
    #    exact, including for a matrix whose rows are empty.
    matrix, grid, seed = random_sparse(12, 9, 30, 0x7261)
    check(matrix.validate(), "a generated matrix must be structurally valid")
    check(matrix.dense() == grid, "the dense round-trip must be identical")
    check(matrix.nnz() == sum(1 for row in grid for value in row if value != 0),
          "nnz must equal the number of stored values")
    holed = CSR(3, 3, [0, 2, 2, 3], [0, 2, 1], [1, 2, 3])
    check(holed.validate() and holed.dense() == [[1, 0, 2], [0, 0, 0], [0, 3, 0]],
          "an empty middle row must be handled")

    # 2. correctness: naive and blocked kernels must both equal the dense reference exactly, on
    #    random matrices and on a fully dense CSR.
    for rows, cols, inner, density in ((6, 5, 4, 40), (10, 8, 6, 20), (4, 4, 12, 100)):
        probe, _, seed = random_sparse(rows, inner, density, seed)
        right, seed = random_dense(inner, cols, seed)
        naive, _ = spmm_naive(probe, right)
        blocked, _ = spmm_blocked(probe, right, 3, 4, 2)
        check(naive == dense_reference(probe.dense(), right),
              "the naive kernel must equal the dense reference")
        check(blocked == naive, "the blocked kernel must equal the naive kernel exactly")
        wide, _ = spmm_blocked(probe, right, 7, 5, 3)
        check(wide == naive, "the result must not depend on the block sizes")

    # 3. the measured win: the blocked traversal touches far fewer B elements while performing
    #    exactly the same arithmetic.
    big, _, seed = random_sparse(64, 48, 25, seed)
    operand, seed = random_dense(48, 40, seed)
    naive, naive_counter = spmm_naive(big, operand)
    blocked, blocked_counter = spmm_blocked(big, operand, 32, 40, 8)
    check(naive == blocked, "the large comparison must still agree")
    check(blocked_counter["fma"] == naive_counter["fma"],
          "blocking must not change the number of multiply-adds")
    check(blocked_counter["fma"] == big.nnz() * 40, "the FMA count must be nnz * n")
    check(blocked_counter["b_reads"] < naive_counter["b_reads"],
          "the blocked traversal must read fewer B elements")
    check(blocked_counter["b_reads"] * 4 < naive_counter["b_reads"],
          "the reuse win must be substantial, not marginal")
    check(naive_counter["b_reads"] == big.nnz() * 40, "the naive model is nnz * n touches")

    # 4. block size effect: the traffic falls monotonically as the row block widens (more rows
    #    share one tile), while the inner-block split only reorders the loop and cannot change
    #    the number of tile loads - both are checked.
    sweep = [spmm_blocked(big, operand, row_block, 40, 8)[1]["b_reads"]
             for row_block in (4, 8, 16, 32, 64)]
    check(all(sweep[i] >= sweep[i + 1] for i in range(len(sweep) - 1)),
          "a wider row block must not increase the B traffic")
    check(sweep[-1] * 4 < sweep[0], "wide row blocks must cut the traffic by more than 4x")
    invariant = {spmm_blocked(big, operand, 8, 16, inner_block)[1]["b_reads"]
                 for inner_block in (1, 2, 4, 8, 48)}
    check(len(invariant) == 1, "the inner-block split must not change the traffic")
    check(min(sweep) > 0, "a nonempty product must read something")

    # 5. boundary cases: no nonzeros, one row, one column, an all-zero dense operand and a
    #    single inner dimension.
    empty = from_dense([[0, 0, 0], [0, 0, 0]])
    right, seed = random_dense(3, 4, seed)
    result, counter = spmm_blocked(empty, right, 2, 2, 2)
    check(result == [[0] * 4, [0] * 4], "a matrix without nonzeros must produce zeros")
    check(counter["fma"] == 0 and counter["b_reads"] == 0,
          "an empty matrix must touch nothing")
    single_row = CSR(1, 3, [0, 2], [0, 2], [3, 4])
    right, seed = random_dense(3, 2, seed)
    naive, counter = spmm_naive(single_row, right)
    check(counter["fma"] == 2 * 2, "a single row must cost two FMAs per output column")
    check(naive == dense_reference(single_row.dense(), right), "a single row must be correct")
    single_col = CSR(2, 3, [0, 1, 2], [1, 1], [5, 6])
    right, seed = random_dense(3, 1, seed)
    check(spmm_blocked(single_col, right)[0] == dense_reference(single_col.dense(), right),
          "a single output column must be correct")
    zero_operand = [[0] * 3 for _ in range(48)]
    check(spmm_blocked(big, zero_operand)[0] == [[0] * 3 for _ in range(big.rows)],
          "a zero operand must produce a zero result")

    # 6. rejected inputs: malformed CSR structures and invalid block sizes.
    expect_value_error(lambda: CSR(2, 2, [0, 1], [0], [1]).validate(), "a short row_ptr")
    expect_value_error(lambda: CSR(2, 2, [0, 2, 1], [0, 1], [1, 1]).validate(),
                       "a decreasing row_ptr")
    expect_value_error(lambda: CSR(1, 2, [0, 1], [3], [1]).validate(), "an out-of-range column")
    expect_value_error(lambda: CSR(1, 3, [0, 2], [2, 1], [1, 1]).validate(),
                       "unsorted indices inside a row")
    expect_value_error(lambda: CSR(1, 1, [0, 1], [0, 0], [1, 2]).validate(), "misaligned values")
    expect_value_error(lambda: spmm_blocked(empty, right, 0, 2, 2), "a zero row block")
    expect_value_error(lambda: spmm_blocked(empty, right, 2, -1, 2), "a negative column block")

    # 7. determinism and the summary report.
    first, _ = spmm_blocked(big, operand, 4, 8, 4)
    second, _ = spmm_blocked(big, operand, 4, 8, 4)
    check(first == second, "the blocked kernel must be deterministic")
    print(f"726 nnz={big.nnz()} shape={big.rows}x{big.cols}x{len(operand[0])} "
          f"fma={blocked_counter['fma']}")
    print(f"726 B touches naive={naive_counter['b_reads']} blocked={blocked_counter['b_reads']} "
          f"ratio={blocked_counter['b_reads'] / naive_counter['b_reads']:.3f}")
    print(f"726 row-block sweep={sweep} inner-split invariant={sorted(invariant)}")
    print("PASS")


if __name__ == "__main__":
    main()
