"""Task 728 - Implement cache-oblivious matrix multiplication idea demo.  (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)

A doubly nested-loop matrix multiply is limited by memory traffic: whichever operand layout the
inner loop walks, one of A, B or C is re-read for every element.  Two fixes exist.  A
*cache-aware* kernel tiles the loops with a block size tuned to the machine, so a tile of each
operand stays resident until it is exhausted.  A *cache-oblivious* kernel instead recurses by
halving the matrix until the subproblems fit in cache at some level on their own - the same code
works for every cache size, which is the property this demo measures.

The demo pairs the three kernels (naive ijk, blocked with a block size, recursive halving) with
a small two-level cache simulator: `line_size` consecutive elements share a cache line, the
cache holds a fixed number of lines, eviction is LRU, and the simulator counts misses.  All
three kernels execute the same n^3 multiply-adds, so every difference in the miss count is a
locality effect and nothing else - that equality is checked, not assumed.

Verified invariants: the three kernels return identical matrices (integer-valued inputs make the
comparison exact), they perform exactly the same number of multiply-adds, the recursive kernel
suffers far fewer misses than the naive one at every cache size, its miss count stays within a
small factor of the *best* tuned block size for each cache size (cache obliviousness), while a
single fixed block size cannot do that across cache sizes, and the boundary cases (n = 1, a base
case larger than the matrix, a zero operand) behave.

Simplified: the memory model has two levels with a single line size, no associativity or
prefetching, no write-back policy (every store is charged a line touch), and the matrices are
small enough to keep this fast - tens of elements per side rather than hundreds.  The kernels
are scalar Python loops, so this measures the *model*, not real throughput.
"""
import math
from collections import OrderedDict

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


class Cache:
    """Two-level cache model: `lines` LRU entries of `line_size` consecutive elements."""

    def __init__(self, lines, line_size):
        if lines < 1 or line_size < 1:
            raise ValueError("cache geometry must be positive")
        self.lines = lines
        self.line_size = line_size
        self.resident = OrderedDict()
        self.hits = 0
        self.misses = 0

    def touch(self, index):
        line = index // self.line_size
        if line in self.resident:
            self.resident.move_to_end(line)
            self.hits += 1
            return False
        self.misses += 1
        self.resident[line] = None
        if len(self.resident) > self.lines:
            self.resident.popitem(last=False)
        return True


class Machine:
    """Flat row-major memory plus the cache; every read and write goes through it."""

    def __init__(self, cache):
        self.cache = cache
        self.data = {}
        self.fma = 0

    def read(self, name, index):
        self.cache.touch(self._address(name, index))
        return self.data[(name, index)]

    def write(self, name, index, value):
        self.cache.touch(self._address(name, index))
        self.data[(name, index)] = value

    def _address(self, name, index):
        base = {"A": 0, "B": 1_000_000, "C": 2_000_000}[name]
        return base + index

    def load(self, name, grid):
        for i, row in enumerate(grid):
            for j, value in enumerate(row):
                self.data[(name, i * len(grid) + j)] = value

    def dump(self, name, n):
        return [[self.data[(name, i * n + j)] for j in range(n)] for i in range(n)]


def naive_ijk(machine, n):
    """Classic triple loop; the inner loop walks a row of B, but C is re-read per k."""
    for i in range(n):
        for k in range(n):
            a = machine.read("A", i * n + k)
            for j in range(n):
                machine.fma += 1
                machine.write("C", i * n + j,
                              machine.read("C", i * n + j) + a * machine.read("B", k * n + j))


def blocked_ikj(machine, n, block):
    """Cache-aware: tiles of A, B and C are kept resident until the tile is exhausted."""
    if block < 1:
        raise ValueError("block size must be positive")
    for i0 in range(0, n, block):
        for k0 in range(0, n, block):
            for j0 in range(0, n, block):
                for i in range(i0, min(i0 + block, n)):
                    for k in range(k0, min(k0 + block, n)):
                        a = machine.read("A", i * n + k)
                        for j in range(j0, min(j0 + block, n)):
                            machine.fma += 1
                            machine.write("C", i * n + j,
                                          machine.read("C", i * n + j)
                                          + a * machine.read("B", k * n + j))


def recursive_multiply(machine, n, base, offsets=None):
    """Cache-oblivious: halve all three dimensions until a subproblem fits, then recurse.

    Offsets are (row, column, stride) triples so a submatrix is addressed inside the parent.
    """
    if base < 1:
        raise ValueError("base case must be positive")
    a = offsets["A"] if offsets else (0, 0, n)
    b = offsets["B"] if offsets else (0, 0, n)
    c = offsets["C"] if offsets else (0, 0, n)
    _recurse(machine, n, base, a, b, c)


def _recurse(machine, n, base, a, b, c):
    if n <= base:
        for i in range(n):
            for k in range(n):
                value = machine.read("A", (a[0] + i) * a[2] + a[1] + k)
                for j in range(n):
                    machine.fma += 1
                    machine.write("C", (c[0] + i) * c[2] + c[1] + j,
                                  machine.read("C", (c[0] + i) * c[2] + c[1] + j)
                                  + value * machine.read("B", (b[0] + k) * b[2] + b[1] + j))
        return
    half = n // 2
    for row in (0, 1):
        for column in (0, 1):
            for inner in (0, 1):
                _recurse(machine, half, base,
                         (a[0] + row * half, a[1] + inner * half, a[2]),
                         (b[0] + inner * half, b[1] + column * half, b[2]),
                         (c[0] + row * half, c[1] + column * half, c[2]))


def run_kernel(kind, grids, n, lines, line_size, parameter=1):
    """Execute one kernel under a fresh cache and report misses plus multiply-adds."""
    machine = Machine(Cache(lines, line_size))
    for name in ("A", "B", "C"):
        machine.load(name, grids[name])
    if kind == "naive":
        naive_ijk(machine, n)
    elif kind == "blocked":
        blocked_ikj(machine, n, parameter)
    elif kind == "recursive":
        recursive_multiply(machine, n, parameter)
    else:
        raise ValueError("unknown kernel")
    return machine.dump("C", n), machine.cache.misses, machine.fma


def reference_product(left, right):
    n = len(left)
    out = [[0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            a = left[i][k]
            for j in range(n):
                out[i][j] += a * right[k][j]
    return out


def make_grids(n, seed, spread=3):
    state, grids = seed & MASK64, {}
    for name in ("A", "B"):
        grid = []
        for _ in range(n):
            row = []
            for _ in range(n):
                state, r = splitmix64(state)
                row.append(int(r % spread) - spread // 2)
            grid.append(row)
        grids[name] = grid
    grids["C"] = [[0] * n for _ in range(n)]
    return grids, state


def main():
    n = 32
    grids, seed = make_grids(n, 0x7281)
    truth = reference_product(grids["A"], grids["B"])
    line_size = 4

    # 1. all three kernels must compute the same integer matrix, and must perform exactly the
    #    same number of multiply-adds - any miss difference is therefore a locality effect.
    naive, naive_misses, naive_fma = run_kernel("naive", grids, n, 128, line_size)
    check(naive == truth, "the naive kernel must equal the reference product")
    check(naive_fma == n ** 3, "the naive kernel must perform n^3 multiply-adds")
    for kind, parameter in (("blocked", 4), ("blocked", 8), ("recursive", 4), ("recursive", 8)):
        result, misses, fma = run_kernel(kind, grids, n, 128, line_size, parameter)
        check(result == truth, f"{kind}({parameter}) must equal the reference product")
        check(fma == naive_fma, f"{kind}({parameter}) must perform the same arithmetic")

    # 2. locality: the recursive kernel must beat the naive one handily, and there must be a
    #    block size that does even better when it is tuned for this particular cache.
    capacities = (32, 64, 128, 256)
    naive_curve, recursive_curve, best_blocked = [], [], []
    for lines in capacities:
        _, misses, _ = run_kernel("naive", grids, n, lines, line_size)
        naive_curve.append(misses)
        _, misses, _ = run_kernel("recursive", grids, n, lines, line_size, 8)
        recursive_curve.append(misses)
        tuned = [(run_kernel("blocked", grids, n, lines, line_size, block)[1], block)
                 for block in (2, 4, 8, 16, 32)]
        best_blocked.append(min(tuned))
    check(all(b < a for a, b in zip(naive_curve, recursive_curve)),
          "the recursive kernel must beat the naive one at every cache size")
    check(all(b * 2 < a for a, b in zip(naive_curve, recursive_curve)),
          "the recursive win must be more than a factor of two")
    check(all(recursive <= 2 * best for recursive, (best, _) in zip(recursive_curve, best_blocked)),
          "the recursion must stay within 2x of the best tuned block size")

    # 3. cache obliviousness: the recursion needs no retuning, whereas one fixed block size
    #    cannot be optimal for every cache size.
    fixed = [run_kernel("blocked", grids, n, lines, line_size, 8)[1] for lines in capacities]
    recursive_ratios = [recursive_curve[i] / best_blocked[i][0] for i in range(len(capacities))]
    check(len({block for _, block in best_blocked}) > 1,
          "the best block size must differ between cache sizes")
    check(sum(fixed) > sum(best for best, _ in best_blocked),
          "no single fixed block size is optimal for every cache size")
    check(0.5 <= min(recursive_ratios) and max(recursive_ratios) <= 2.0,
          "the recursion must track the tuned optimum at every cache size")
    check(min(recursive_ratios) < 1.0,
          "on at least one cache size the recursion must even beat the tuned block sweep")

    # 4. block size sweep at a fixed cache: too small leaves the cache underused, too large
    #    evicts before the tile is exhausted, so an interior optimum exists.
    sweep = [(block, run_kernel("blocked", grids, n, 64, line_size, block)[1])
             for block in (2, 4, 8, 16, 32)]
    best = min(sweep, key=lambda entry: entry[1])
    check(1 < best[0] < 32, "the block-size optimum must be interior")
    check(best[1] < sweep[0][1], "a tuned block must beat the smallest block")
    check(all(entry[1] > 0 for entry in sweep), "every kernel must touch memory")
    check(naive_curve[0] > min(entry[1] for entry in sweep),
          "a tuned kernel must beat the naive one at the smallest cache")

    # 5. boundary cases: a 1x1 product, a base case larger than the matrix (which must behave
    #    exactly like the naive kernel), and a zero operand.
    one, _ = make_grids(1, 0x7282)
    check(run_kernel("recursive", one, 1, 64, line_size, 8)[0] == [[one["A"][0][0] * one["B"][0][0]]],
          "the 1x1 case must work")
    small, _ = make_grids(8, 0x7283)
    small_truth = reference_product(small["A"], small["B"])
    base_large, base_misses, _ = run_kernel("recursive", small, 8, 64, line_size, 64)
    naive_small, naive_misses, _ = run_kernel("naive", small, 8, 64, line_size)
    check(base_large == small_truth and naive_small == small_truth,
          "the small products must be correct")
    check(base_misses == naive_misses,
          "a base case covering the whole matrix must degenerate to the naive traversal")
    zero = {"A": [[0] * 8 for _ in range(8)], "B": small["B"], "C": [[0] * 8 for _ in range(8)]}
    check(run_kernel("recursive", zero, 8, 64, line_size, 4)[0] == [[0] * 8 for _ in range(8)],
          "a zero operand must produce zeros")

    # 6. rejected inputs and determinism.
    expect_value_error(lambda: Cache(0, 4), "a cache without lines is invalid")
    expect_value_error(lambda: Cache(4, 0), "a line size of zero is invalid")
    expect_value_error(lambda: run_kernel("blocked", grids, n, 64, line_size, 0),
                       "a zero block is invalid")
    expect_value_error(lambda: run_kernel("recursive", grids, n, 64, line_size, 0),
                       "a zero base case is invalid")
    expect_value_error(lambda: run_kernel("magic", grids, n, 64, line_size, 1),
                       "an unknown kernel is invalid")
    repeat = run_kernel("recursive", grids, n, 64, line_size, 8)
    again = run_kernel("recursive", grids, n, 64, line_size, 8)
    check(repeat == again, "the kernel must be deterministic")

    print(f"728 n={n} line={line_size} capacities={capacities}")
    print(f"728 misses naive={naive_curve} recursive={recursive_curve} "
          f"best-blocked={[entry[0] for entry in best_blocked]}")
    print("PASS")


if __name__ == "__main__":
    main()
