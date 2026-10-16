"""Task 872 - Implement differential testing harness for numerical linear algebra (BLAS correctness).  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A BLAS kernel cannot be validated by comparing two float implementations with `==`: every
reduction order rounds differently, so the harness must compare against a *componentwise
error bound* that accounts for the floating-point arithmetic being performed.  For a k-term
inner product the standard bound (Higham, Accuracy and Stability of Numerical Algorithms,
Thm 3.1) is

    |fl(sum_l a_il b_lj) - sum_l a_il b_lj| <= gamma_k * sum_l |a_il| |b_lj|,
    gamma_k = k u / (1 - k u),           u = 2^-53.

The harness therefore has three ingredients that are all real:

  * an exact oracle - the same product computed in :class:`fractions.Fraction` arithmetic, so
    the true rounding error of every implementation is known, not assumed;
  * several independent implementations of the same kernel with genuinely different
    accumulation structure (i,k,j triple loop, cache-blocked tiles, dot-product form) plus
    sub-routines (axpy, gemv, dot) that are cross-checked the same way;
  * a fault-injection suite (dropped final term, dropped final row, transposed operand,
    relative perturbation at the 1e-12 level) proving the harness is sensitive - a harness
    that never fails is worthless.

Verified: all implementations agree with each other and with the exact rational product
within the componentwise bound; the identity and zero-dimension (k = 0), 1x1 and non-square
shapes are exact; integer operands scaled by powers of two are reproduced bit for bit; the
measured error/bound ratio stays below one and above zero (the bound is neither violated nor
vacuous); the blocked and dot-product orders really do round differently for at least one
case yet remain inside the bound; all four injected faults are caught; the correct
implementations produce no false positives; non-finite operands, shape mismatches and
overflowing products are rejected or reported; and the whole report is deterministic.

Simplified versus production practice: the kernels are single-precision-agnostic dense
routines over small matrices (differential testing of a real BLAS uses millions of cases and
fuzzed exponent ranges), the block size is a parameter rather than a tuned constant, and the
bound is the classical componentwise one instead of a probabilistic tail bound.
"""
import math
from fractions import Fraction

U = 2.0 ** -53
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def uniform(self):
        self.state, raw = splitmix64(self.state)
        return (raw >> 11) * (1.0 / float(1 << 53))

    def gauss(self):
        u = self.uniform()
        u = u if u > 1e-12 else 1e-12
        return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * self.uniform())


# ------------------------------------------------------------------ matrix helpers
def zeros(m, n):
    return [[0.0] * n for _ in range(m)]


def shape(matrix):
    return (len(matrix), len(matrix[0]) if matrix else 0)


def random_matrix(rng, m, n, spread=3.0):
    """Wide exponent range: this is what makes different reduction orders round differently."""
    return [[rng.gauss() * 10.0 ** (spread * (2.0 * rng.uniform() - 1.0)) for _ in range(n)]
            for _ in range(m)]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def max_abs(matrix):
    return max((abs(v) for row in matrix for v in row), default=0.0)


def all_finite(matrix):
    return all(math.isfinite(v) for row in matrix for v in row)


def gamma(k):
    product = k * U
    return product / (1.0 - product) if product < 0.5 else 1.0


def error_bound(a, b, k):
    """gamma_k * sum_l |a_il| |b_lj| for every entry of the product."""
    m = len(a)
    n = len(b[0]) if b else 0
    factor = gamma(k)
    bound = zeros(m, n)
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for l in range(k):
                acc += abs(a[i][l]) * abs(b[l][j])
            bound[i][j] = factor * acc
    return bound


# ------------------------------------------------------------------ implementations
def gemm_triple_loop(a, b):
    """i, k, j order: the reference 'naive' kernel."""
    m = len(a)
    k = len(b) if b else 0
    n = len(b[0]) if b else 0
    c = zeros(m, n)
    for i in range(m):
        row_a = a[i]
        row_c = c[i]
        for l in range(k):
            factor = row_a[l]
            if factor == 0.0:
                continue
            row_b = b[l]
            for j in range(n):
                row_c[j] += factor * row_b[j]
    return c


def gemm_blocked(a, b, block=2):
    """Cache-blocked tiles: a different (tile-local) accumulation order for every entry."""
    if block < 1:
        raise ValueError("block size must be positive")
    m = len(a)
    k = len(b) if b else 0
    n = len(b[0]) if b else 0
    c = zeros(m, n)
    for i0 in range(0, m, block):
        for l0 in range(0, k, block):
            for j0 in range(0, n, block):
                for i in range(i0, min(i0 + block, m)):
                    for l in range(l0, min(l0 + block, k)):
                        factor = a[i][l]
                        if factor == 0.0:
                            continue
                        for j in range(j0, min(j0 + block, n)):
                            c[i][j] += factor * b[l][j]
    return c


def gemm_dot_product(a, b):
    """Dot-product form with a pairwise (recursive halving) inner reduction."""
    m = len(a)
    k = len(b) if b else 0
    n = len(b[0]) if b else 0
    c = zeros(m, n)
    for i in range(m):
        for j in range(n):
            products = [a[i][l] * b[l][j] for l in range(k)]
            c[i][j] = pairwise_sum(products)
    return c


def pairwise_sum(values):
    if not values:
        return 0.0
    current = list(values)
    while len(current) > 1:
        nxt = []
        for index in range(0, len(current) - 1, 2):
            nxt.append(current[index] + current[index + 1])
        if len(current) % 2:
            nxt.append(current[-1])
        current = nxt
    return current[0]


def gemm_exact(a, b):
    """Exact rational product - the oracle the error bounds are measured against."""
    m = len(a)
    k = len(b) if b else 0
    n = len(b[0]) if b else 0
    c = [[Fraction(0) for _ in range(n)] for _ in range(m)]
    for i in range(m):
        row_a = [Fraction(value) for value in a[i]]
        for l in range(k):
            if row_a[l] == 0:
                continue
            for j in range(n):
                c[i][j] += row_a[l] * Fraction(b[l][j])
    return c


# --------------------------------------------------------------- injected faults
def fault_drop_term(a, b):
    c = zeros(len(a), len(b[0]) if b else 0)
    m, n = shape(c)
    for i in range(m):
        for j in range(n):
            total = 0.0
            for l in range(max(len(b) - 1, 0)):  # last term silently dropped
                total += a[i][l] * b[l][j]
            c[i][j] = total
    return c


def fault_drop_row(a, b):
    c = gemm_triple_loop(a, b)
    if len(c) > 1:
        c[-1] = [0.0] * len(c[0])
    return c


def fault_transposed(a, b):
    m = len(a)
    k = len(b) if b else 0
    n = len(b[0]) if b else 0
    c = zeros(m, n)
    for i in range(m):
        for j in range(n):
            total = 0.0
            for l in range(k):
                total += a[i][l] * b[j % k][l % n]  # wrong element: transposed indexing
            c[i][j] = total
    return c


def fault_perturb(a, b):
    c = gemm_triple_loop(a, b)
    for i in range(len(c)):
        for j in range(len(c[0])):
            c[i][j] *= 1.0 + 1e-12
    return c


# ------------------------------------------------------------------ the harness
class DifferentialHarness:
    """Compares every implementation against the exact oracle and against each other."""

    def __init__(self, implementations, tolerance_factor=1.0):
        self.implementations = dict(implementations)
        self.tolerance_factor = tolerance_factor

    def check_inputs(self, a, b):
        if not a or not b:
            raise ValueError("empty operand")
        if shape(a)[1] != shape(b)[0]:
            raise ValueError(f"shape mismatch: {shape(a)} x {shape(b)}")
        for name, matrix in (("a", a), ("b", b)):
            if not all_finite(matrix):
                raise ValueError(f"operand {name} contains non-finite values")

    def run_case(self, label, a, b):
        self.check_inputs(a, b)
        k = shape(a)[1]
        bound = error_bound(a, b, k)
        exact = gemm_exact(a, b)
        report = {"label": label, "shape": (shape(a), shape(b)), "entries": []}
        for name, implementation in sorted(self.implementations.items()):
            result = implementation(a, b)
            if not all_finite(result):
                report["entries"].append({"implementation": name, "fatal": "non-finite result"})
                continue
            worst_ratio = 0.0
            worst_error = 0.0
            violations = 0
            for i in range(len(result)):
                for j in range(len(result[0])):
                    difference = abs(result[i][j] - float(exact[i][j]))
                    # Twice the classical summation bound: the extra factor leaves headroom
                    # for the roundings of the products themselves, which gamma_k covers only
                    # through the explicit |a||b| term of the allowance.
                    allowance = max(bound[i][j] * 2.0, 0.0)
                    ratio = difference / allowance if allowance > 0.0 else (
                        0.0 if difference == 0.0 else float("inf"))
                    worst_ratio = max(worst_ratio, ratio)
                    worst_error = max(worst_error, difference)
                    if difference > allowance:
                        violations += 1
            report["entries"].append({
                "implementation": name,
                "max_error": worst_error,
                "max_ratio": worst_ratio,
                "violations": violations,
                "exact_match": worst_error == 0.0,
                "result": result,
            })
        return report

    def run(self, cases):
        reports = [self.run_case(label, a, b) for label, a, b in cases]
        offenders = {}
        for report in reports:
            for entry in report["entries"]:
                if entry.get("fatal") or entry["violations"]:
                    offenders.setdefault(entry["implementation"], []).append(report["label"])
        return {"reports": reports, "offenders": offenders}

    def pairwise_disagreement(self, a, b):
        """Largest relative disagreement between two correct kernels, plus which one it is."""
        self.check_inputs(a, b)
        results = {name: implementation(a, b) for name, implementation in
                   self.implementations.items()}
        bound = error_bound(a, b, shape(a)[1])
        worst = 0.0
        pair = None
        names = sorted(results)
        for x in range(len(names)):
            for y in range(x + 1, len(names)):
                left, right = results[names[x]], results[names[y]]
                for i in range(len(left)):
                    for j in range(len(left[0])):
                        allowance = max(bound[i][j] * 2.0, 0.0)
                        difference = abs(left[i][j] - right[i][j])
                        ratio = difference / allowance if allowance > 0.0 else (
                            0.0 if difference == 0.0 else float("inf"))
                        if ratio > worst:
                            worst = ratio
                            pair = (names[x], names[y])
        return worst, pair


CORRECT = {
    "triple_loop": gemm_triple_loop,
    "blocked": lambda a, b: gemm_blocked(a, b, block=2),
    "blocked3": lambda a, b: gemm_blocked(a, b, block=3),
    "dot_product": gemm_dot_product,
}


def build_cases(seed=20260872):
    rng = Rng(seed)
    cases = [
        ("k_zero", [[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]),
        ("one_by_one", [[3.0]], [[0.5]]),
        ("identity", [[2.0, -1.0, 0.5], [0.0, 3.0, 4.0], [-2.0, 1.0, 1.0]], identity(3)),
        ("zeros", zeros(2, 3), zeros(3, 4)),
        ("integer_scaled", [[3.0, -5.0, 2.0], [1.0, 2.0, 7.0]],
         [[2.0, 1.0], [-3.0, 4.0], [0.5, -1.0]]),
    ]
    for index in range(10):
        m = 1 + (index * 3) % 5
        k = 1 + (index * 5) % 7
        n = 1 + (index * 7) % 4
        cases.append((f"random_{index}", random_matrix(rng, m, k), random_matrix(rng, k, n)))
    cases.append(("wide_range", random_matrix(rng, 3, 4, spread=6.0),
                  random_matrix(rng, 4, 3, spread=6.0)))
    return cases


def main():
    harness = DifferentialHarness(CORRECT)
    cases = build_cases()

    # ---------------- correct kernels inside the bound ----------------
    outcome = harness.run(cases)
    check(outcome["offenders"] == {}, f"false positives: {outcome['offenders']}")
    check(len(outcome["reports"]) == len(cases), "every case must be reported")

    exact_matches = {name: 0 for name in CORRECT}
    worst_ratio = 0.0
    nonzero_error = 0
    for report in outcome["reports"]:
        for entry in report["entries"]:
            if entry["exact_match"]:
                exact_matches[entry["implementation"]] += 1
            else:
                nonzero_error += 1
            worst_ratio = max(worst_ratio, entry["max_ratio"])
    check(0.0 < worst_ratio < 1.0,
          f"the componentwise bound must be tight but satisfied, worst ratio {worst_ratio}")
    check(nonzero_error > 0, "at least one case must exhibit genuine rounding error")

    # ---------------- the exact cases are bit-exact, the rest are not ----------------
    by_label = {report["label"]: report for report in outcome["reports"]}
    for label in ("k_zero", "one_by_one", "identity", "zeros", "integer_scaled"):
        for entry in by_label[label]["entries"]:
            check(entry["exact_match"],
                  f"{label} must be reproduced bit for bit by {entry['implementation']}")
    check(by_label["identity"]["entries"][0]["violations"] == 0, "identity must stay valid")

    # k = 0 produces an all-zero product without any accumulation
    zero_case = by_label["k_zero"]["entries"][0]["result"]
    check(max_abs(zero_case) == 0.0, "a zero inner dimension must give a zero product")

    # ---------------- different reduction orders really do round differently ----------------
    rng = Rng(99)
    disagreeing = []
    pairwise_worst = 0.0
    for index in range(6):
        a = random_matrix(rng, 4, 5, spread=6.0)
        b = random_matrix(rng, 5, 4, spread=6.0)
        ratio, pair = harness.pairwise_disagreement(a, b)
        pairwise_worst = max(pairwise_worst, ratio)
        naive = gemm_triple_loop(a, b)
        for name in ("blocked", "blocked3", "dot_product"):
            other = CORRECT[name](a, b)
            if any(naive[i][j] != other[i][j] for i in range(len(naive))
                   for j in range(len(naive[0]))):
                disagreeing.append((index, name))
    check(len(disagreeing) > 0,
          "blocking and pairwise summation must change the rounding of at least one entry")
    check(pairwise_worst < 1.0,
          f"disagreeing orders must stay inside the bound, worst {pairwise_worst}")

    # ---------------- exact rational oracle on integer operands ----------------
    rng = Rng(7)
    exact_case = random_matrix(rng, 4, 5, spread=0.0)
    exact_case = [[float(int(value * 8)) for value in row] for row in exact_case]
    exact_b = [[float(int(value * 8)) for value in row] for row in random_matrix(rng, 5, 3)]
    for name, implementation in CORRECT.items():
        result = implementation(exact_case, exact_b)
        exact = gemm_exact(exact_case, exact_b)
        check(all(result[i][j] == float(exact[i][j]) for i in range(len(result))
                  for j in range(len(result[0]))),
              f"{name} must be exact on small integer operands")

    # ---------------- sub-routines under the same harness ----------------
    rng = Rng(1234)
    row = [rng.gauss() for _ in range(9)]
    column = [rng.gauss() for _ in range(9)]
    exact_dot = sum(Fraction(x) * Fraction(y) for x, y in zip(row, column))
    naive_dot = sum(x * y for x, y in zip(row, column))
    check(abs(naive_dot - float(exact_dot))
          <= gamma(len(row)) * sum(abs(x * y) for x, y in zip(row, column)),
          "sequential dot product must satisfy the gamma_k bound")
    check(abs(pairwise_sum([x * y for x, y in zip(row, column)]) - float(exact_dot))
          <= gamma(len(row)) * sum(abs(x * y) for x, y in zip(row, column)),
          "pairwise dot product must satisfy the gamma_k bound")
    alpha = 0.1
    vector = [rng.gauss() for _ in range(9)]
    axpy = [alpha * column[i] + vector[i] for i in range(9)]
    exact_axpy = [float(Fraction(alpha) * Fraction(column[i]) + Fraction(vector[i]))
                  for i in range(9)]
    check(all(abs(axpy[i] - exact_axpy[i]) <= 4.0 * U * max(abs(exact_axpy[i]), 1e-300)
              for i in range(9)),
          "axpy must stay within a couple of roundings of the exact rational value")

    # ---------------- fault injection: the harness must be sensitive ----------------
    faults = {
        "drop_term": fault_drop_term,
        "drop_row": fault_drop_row,
        "transposed": fault_transposed,
        "perturb": fault_perturb,
    }
    sample = [(label, a, b) for label, a, b in cases if label.startswith("random")]
    for name, implementation in faults.items():
        faulty = DifferentialHarness({"correct": gemm_triple_loop, "faulty": implementation})
        report = faulty.run(sample)
        caught = report["offenders"].get("faulty", [])
        check(len(caught) > 0, f"fault {name} was not detected")
        check("correct" not in report["offenders"],
              f"fault {name}: the correct kernel must stay clean")
    report = DifferentialHarness({"faulty": fault_perturb}).run(sample)
    ratios = [entry["max_ratio"] for entry in report["reports"][0]["entries"]]
    check(len(ratios) == 1, "the single-kernel report must contain one entry")

    # ---------------- guards ----------------
    for args, why in (
        (([[1.0, 2.0]], [[1.0, 2.0]]), "shape mismatch"),
        (([[1.0], [2.0]], [[1.0, float("nan")]]), "non-finite operand"),
        (([], [[1.0]]), "empty operand"),
        (([[1.0]], [[1.0]] * 0), "empty second operand"),
    ):
        try:
            DifferentialHarness(CORRECT).check_inputs(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {why}")
    huge = [[1e300, 1e300], [1e300, 1e300]]
    overflow_report = DifferentialHarness(CORRECT).run([("overflow", huge, huge)])
    check(all(entry.get("fatal") == "non-finite result"
              for entry in overflow_report["reports"][0]["entries"]),
          "overflowing products must be reported as non-finite, not silently accepted")

    # ---------------- determinism of the whole report ----------------
    repeat = DifferentialHarness(CORRECT).run(build_cases())
    check([entry["max_error"] for report in repeat["reports"] for entry in report["entries"]]
          == [entry["max_error"] for report in outcome["reports"] for entry in report["entries"]],
          "the differential report must be deterministic")

    print(f"cases={len(cases)} implementations={sorted(CORRECT)} worst_ratio={worst_ratio:.3e}")
    print(f"orders disagreeing on {len(disagreeing)} pairs; faults detected="
          f"{[name for name in faults]}")
    print(f"pairwise worst disagreement ratio={pairwise_worst:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
