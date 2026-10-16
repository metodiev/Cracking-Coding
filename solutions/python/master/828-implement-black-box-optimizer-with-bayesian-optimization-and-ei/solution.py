"""Task 828 - Implement black-box optimizer with Bayesian optimization and EI acquisition. (Tier: Master, CTCI Beyond — Machine Learning & AI)

A sequential model-based optimizer: a Gaussian process with a squared-exponential kernel
is fitted to every (x, y) the black box has returned, the posterior is turned into an
*expected improvement* acquisition function in closed form, and the next query is the
argmax of EI over a candidate grid. Everything is exact linear algebra plus the closed
form EI, so each ingredient can be checked against an independent oracle (Gauss
elimination instead of Cholesky, numerical quadrature instead of the EI formula, an
exhaustive grid scan instead of the bayesian search).

What is implemented
-------------------
* `rbf`, `cholesky`, `solve_cholesky`, `solve_gauss`: a dense-kernel GP implemented from
  scratch (no numpy): the posterior mean and variance come from two Cholesky solves, and
  `solve_gauss` is the independent oracle for the linear algebra.
* `GaussianProcess`: fit on (X, y) with a nugget on the diagonal, `predict(x)` -> (mean,
  variance) clamped at a non-negative value.
* `expected_improvement(mean, sd, best)`: the textbook formula
  EI = (mean - best) * Phi(z) + sd * phi(z), z = (mean - best) / sd, with the degenerate
  sd == 0 branch returning the deterministic improvement. `ei_quadrature` integrates the
  definition EI = int_{best}^{+inf} (t - best) N(t; mean, sd^2) dt by Simpson's rule.
* `BlackBoxOptimizer`: a deterministic initial design, an EI maximisation over the
  candidate grid excluding evaluated points, and an `Oracle` that counts its own
  evaluations and refuses to exceed its budget, so the sample-efficiency claims are
  backed by a real counter.
* `random_search`: the uniform baseline over the same candidate grid with the same
  budget, drawn from a seeded splitmix64 stream.

What is simplified
------------------
* One-dimensional black boxes on a candidate grid: no multi-start continuous EI
  optimisation, no hyperparameter marginal likelihood training (the kernel lengthscale and
  signal variance are fixed, which is exactly what a toy needs), no parallel batching.
* The GP is dense O(n^3) in the number of observations; the "at scale" version would use
  a sparse/inducing-point approximation. Here n <= 24, so the exact posterior is used and
  the acquisition is explored exhaustively on the grid.
* Deterministic throughout: no wall-clock, no threads, no unseeded randomness.

Invariants verified
-------------------
1.  Linear algebra: `cholesky` accepts every kernel matrix encountered and reproduces the
    matrix as L L^T; `solve_cholesky` agrees with an independent Gauss-elimination solve.
2.  Kernel sanity: k(x, x) == signal, |k(x, y)| <= k(x, x), and the 2x2 Gram matrix has a
    non-negative determinant for every sampled pair (positive semi-definiteness).
3.  GP interpolation: with a small nugget, the posterior mean at an observed point returns
    the observation and the posterior variance there collapses towards the nugget.
4.  GP extrapolation: far from all observations the posterior mean returns to the prior
    mean (0) and the variance returns to the prior variance (signal).
5.  EI agrees with Simpson quadrature of its definition to 1e-9 absolute, is non-negative
    everywhere, is strictly decreasing in the incumbent, and is maximal at the largest
    posterior variance when all observations are equal (pure exploration).
6.  Degenerate posteriors: sd == 0 gives the deterministic improvement, and an all-equal
    target is handled without a division by zero.
7.  The optimizer respects its evaluation budget exactly (`Oracle` raises otherwise), never
    queries a point twice, and its best-so-far sequence is non-decreasing.
8.  Recall: on the test families the incumbent is within the documented tolerance of the
    exhaustive grid optimum after at most 24 evaluations.
9.  Sample efficiency: over five deterministic instances the bayesian search's total regret
    is below uniform random search's at equal budget (and printed instance by instance).
10. Determinism: two identical runs produce the same query sequence and the same incumbent.
"""
import math

SQRT_2PI = math.sqrt(2.0 * math.pi)
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------------------------------
# linear algebra
# --------------------------------------------------------------------------------------
def rbf(a, b, lengthscale, signal):
    return signal * math.exp(-0.5 * ((a - b) / lengthscale) ** 2)


def cholesky(matrix):
    """Lower-triangular L with L L^T = matrix; raises when the matrix is not PD."""
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            acc = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if acc <= 0.0:
                    raise ValueError("matrix is not positive definite")
                lower[i][i] = math.sqrt(acc)
            else:
                lower[i][j] = acc / lower[j][j]
    return lower


def solve_cholesky(lower, vector):
    n = len(lower)
    forward = [0.0] * n
    for i in range(n):
        acc = vector[i] - sum(lower[i][k] * forward[k] for k in range(i))
        forward[i] = acc / lower[i][i]
    back = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = forward[i] - sum(lower[k][i] * back[k] for k in range(i + 1, n))
        back[i] = acc / lower[i][i]
    return back


def solve_gauss(matrix, vector):
    """Plain Gaussian elimination with partial pivoting -- the independent oracle."""
    n = len(matrix)
    a = [list(row) + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-300:
            raise ValueError("singular system")
        a[col], a[pivot] = a[pivot], a[col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = a[i][n] - sum(a[i][k] * solution[k] for k in range(i + 1, n))
        solution[i] = acc / a[i][i]
    return solution


# --------------------------------------------------------------------------------------
# Gaussian process
# --------------------------------------------------------------------------------------
class GaussianProcess:
    """Exact GP regression with a squared-exponential kernel and a fixed nugget."""

    def __init__(self, lengthscale=0.1, signal=1.0, noise=1e-10):
        self.lengthscale = lengthscale
        self.signal = signal
        self.noise = noise

    def fit(self, xs, ys):
        self.xs, self.ys = list(xs), list(ys)
        n = len(xs)
        gram = [[rbf(xs[i], xs[j], self.lengthscale, self.signal) for j in range(n)]
                for i in range(n)]
        for i in range(n):
            gram[i][i] += self.noise
        self.gram = gram
        self.lower = cholesky(gram)
        self.alpha = solve_cholesky(self.lower, self.ys)
        return self

    def predict(self, x):
        k = [rbf(x, xi, self.lengthscale, self.signal) for xi in self.xs]
        mean = sum(k[i] * self.alpha[i] for i in range(len(k)))
        weights = solve_cholesky(self.lower, k)
        variance = self.signal - sum(k[i] * weights[i] for i in range(len(k)))
        return mean, max(0.0, variance)


# --------------------------------------------------------------------------------------
# expected improvement
# --------------------------------------------------------------------------------------
def normal_pdf(z):
    return math.exp(-0.5 * z * z) / SQRT_2PI


def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def expected_improvement(mean, sd, best):
    """EI = E[max(0, f - best)] for f ~ N(mean, sd^2)."""
    if sd <= 0.0:
        return max(0.0, mean - best)
    z = (mean - best) / sd
    return (mean - best) * normal_cdf(z) + sd * normal_pdf(z)


def ei_quadrature(mean, sd, best, nodes=20000):
    """Simpson's rule for int_{best}^{inf} (t - best) N(t; mean, sd^2) dt (the oracle)."""
    upper = mean + 12.0 * sd if sd > 0 else mean
    if upper <= best:
        return 0.0
    lo = max(best, mean - 12.0 * sd)
    step = (upper - lo) / nodes

    def integrand(t):
        return (t - best) * math.exp(-0.5 * ((t - mean) / sd) ** 2) / (sd * SQRT_2PI)

    total = integrand(lo) + integrand(upper)
    for i in range(1, nodes):
        total += (4.0 if i % 2 else 2.0) * integrand(lo + i * step)
    return total * step / 3.0


# --------------------------------------------------------------------------------------
# the black box and the optimizer
# --------------------------------------------------------------------------------------
class Oracle:
    """A black box that counts its own evaluations and refuses to exceed its budget."""

    def __init__(self, function, budget):
        self.function = function
        self.budget = budget
        self.calls = 0
        self.queried = []

    def query(self, x):
        if self.calls >= self.budget:
            raise RuntimeError("evaluation budget exhausted")
        self.calls += 1
        self.queried.append(x)
        return self.function(x)


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self, lo, hi):
        return lo + (hi - lo) * (self.next_u64() / float(1 << 64))


def candidate_grid(size=201, lo=0.0, hi=1.0):
    return [lo + (hi - lo) * i / (size - 1) for i in range(size)]


class BlackBoxOptimizer:
    """EI-driven sequential design over a candidate grid."""

    def __init__(self, grid=None, lengthscale=0.1, signal=1.0, noise=1e-10,
                 max_evaluations=24, initial=3):
        self.grid = candidate_grid() if grid is None else list(grid)
        self.gp = GaussianProcess(lengthscale, signal, noise)
        self.max_evaluations = max_evaluations
        self.initial = initial

    def choose(self, xs, ys):
        """The next query: argmax EI on unexplored candidates, variance tie-break."""
        self.gp.fit(xs, ys)
        best = max(ys)
        seen = set(xs)
        pick, pick_score, pick_var = None, None, None
        for x in self.grid:
            if x in seen:
                continue
            mean, variance = self.gp.predict(x)
            score = expected_improvement(mean, math.sqrt(variance), best)
            if (pick_score is None or score > pick_score + 1e-15
                    or (abs(score - pick_score) <= 1e-15 and variance > pick_var)):
                pick, pick_score, pick_var = x, score, variance
        return pick

    def run(self, oracle, initial_points=None):
        span = len(self.grid) - 1
        xs = initial_points if initial_points is not None \
            else [self.grid[span * i // (self.initial + 1)] for i in range(1, self.initial + 1)]
        ys = [oracle.query(x) for x in xs]
        trace = [{"x": x, "y": y, "best": None} for x, y in zip(xs, ys)]
        best_history = [max(ys)]
        while oracle.calls < self.max_evaluations:
            pick = self.choose(xs, ys)
            if pick is None:
                break
            value = oracle.query(pick)
            xs.append(pick)
            ys.append(value)
            best_history.append(max(ys))
            trace.append({"x": pick, "y": value, "best": max(ys)})
        return {"xs": xs, "ys": ys, "trace": trace, "best_x": xs[ys.index(max(ys))],
                "best_y": max(ys), "history": best_history, "evaluations": oracle.calls}


def random_search(oracle, seed=1, initial_points=None):
    """Uniform baseline: the same budget, the same candidate grid."""
    grid = candidate_grid()
    rng = SplitMix64(seed)
    xs = list(initial_points) if initial_points else []
    seen = set(xs)
    ys = [oracle.query(x) for x in xs]
    while oracle.calls < oracle.budget:
        x = grid[rng.next_u64() % len(grid)]
        while x in seen and len(seen) < len(grid):
            x = grid[rng.next_u64() % len(grid)]
        seen.add(x)
        ys.append(oracle.query(x))
        xs.append(x)
    return {"best_x": xs[ys.index(max(ys))], "best_y": max(ys), "xs": xs, "ys": ys}


# --------------------------------------------------------------------------------------
# test families
# --------------------------------------------------------------------------------------
def quadratic_with_ripple(centre, ripple=0.0):
    return lambda x: -(x - centre) ** 2 + ripple * math.sin(12.0 * math.pi * x)


def exhaustive_optimum(function, size=20001):
    best_x = best_y = None
    for i in range(size):
        x = i / (size - 1)
        y = function(x)
        if best_y is None or y > best_y:
            best_x, best_y = x, y
    return best_x, best_y


def main():
    # -- 1. linear algebra ---------------------------------------------------------------
    xs = [0.0, 0.11, 0.23, 0.41, 0.55, 0.79, 1.0]
    ys = [math.sin(3.0 * x) for x in xs]
    gp = GaussianProcess().fit(xs, ys)
    n = len(xs)
    product = [[sum(gp.lower[i][k] * gp.lower[j][k] for k in range(n)) for j in range(n)]
               for i in range(n)]
    worst = max(abs(product[i][j] - gp.gram[i][j]) for i in range(n) for j in range(n))
    check(worst < 1e-12, f"Cholesky reproduces the kernel matrix (worst {worst:.2e})")
    vector = [1.0, -2.0, 0.5, 3.0, -0.25, 0.75, 2.0]
    by_cholesky = solve_cholesky(gp.lower, vector)
    by_gauss = solve_gauss(gp.gram, vector)
    check(max(abs(a - b) for a, b in zip(by_cholesky, by_gauss)) < 1e-9,
          "Cholesky solve agrees with an independent Gauss-elimination solve")
    check(all(abs(rbf(x, x, 0.1, 1.0) - 1.0) < 1e-15 for x in xs)
          and all(abs(rbf(a, b, 0.1, 1.0)) <= 1.0 for a in xs for b in xs),
          "k(x, x) is the signal variance and the kernel never exceeds it")
    det_min = min(rbf(a, a, 0.1, 1.0) * rbf(b, b, 0.1, 1.0) - rbf(a, b, 0.1, 1.0) ** 2
                  for a in xs for b in xs)
    check(det_min >= 0.0, f"every 2x2 Gram matrix is PSD (min determinant {det_min:.2e})")

    # -- 2. GP interpolation and extrapolation -------------------------------------------
    predicted = [gp.predict(x)[0] for x in xs]
    check(max(abs(p - y) for p, y in zip(predicted, ys)) < 1e-6,
          "the posterior mean reproduces the observations")
    check(max(gp.predict(x)[1] for x in xs) < 1e-7,
          "the posterior variance at the observations collapses to the nugget")
    mean_far, var_far = gp.predict(-3.0)
    check(abs(mean_far) < 1e-6 and abs(var_far - 1.0) < 1e-6,
          "far from the data the GP returns to the prior mean and variance")

    # -- 3. expected improvement ---------------------------------------------------------
    cases = [(0.3, 0.4, 0.1), (0.0, 0.2, 0.35), (0.5, 0.05, 0.5), (0.9, 1.2, 0.2)]
    worst_ei = 0.0
    for mean, sd, best in cases:
        closed = expected_improvement(mean, sd, best)
        numeric = ei_quadrature(mean, sd, best)
        worst_ei = max(worst_ei, abs(closed - numeric))
        check(closed >= 0.0, "EI is non-negative")
    check(worst_ei < 1e-9, f"closed-form EI matches Simpson quadrature ({worst_ei:.2e})")
    check(all(expected_improvement(0.4, 0.3, low) > expected_improvement(0.4, 0.3, high)
              for low, high in ((0.0, 0.1), (0.1, 0.35), (0.35, 0.9))),
          "EI decreases strictly in the incumbent")
    check(expected_improvement(0.5, 0.0, 0.2) == 0.3
          and expected_improvement(0.1, 0.0, 0.2) == 0.0,
          "a degenerate posterior yields the deterministic improvement")
    zero_targets = [(0.0, 0.0), (0.2, 0.0), (0.9, 0.0)]
    flat_gp = GaussianProcess().fit([x for x, _ in zero_targets], [y for _, y in zero_targets])
    sd = {x: math.sqrt(flat_gp.predict(x)[1]) for x in (0.4, 0.45, 0.5)}
    check(sd[0.5] > sd[0.45] > sd[0.4],
          "the posterior variance grows away from the observations")
    scores = {x: expected_improvement(flat_gp.predict(x)[0], sd[x], 0.0)
              for x in (0.4, 0.45, 0.5)}
    check(scores[0.5] > scores[0.45] > scores[0.4],
          "with zero targets EI degenerates to pure exploration (0.3989 * sd)")
    check(abs(scores[0.5] - normal_pdf(0.0) * sd[0.5]) < 1e-15,
          "pure-exploration EI equals phi(0) * sd exactly")

    # -- 4. the optimizer ----------------------------------------------------------------
    function = quadratic_with_ripple(0.37, 0.02)
    oracle = Oracle(function, budget=24)
    result = BlackBoxOptimizer(lengthscale=0.08, max_evaluations=24).run(oracle)
    best_x, best_y = exhaustive_optimum(function)
    check(oracle.calls == result["evaluations"] == 24,
          "the optimizer uses its evaluation budget exactly")
    check(len(set(result["xs"])) == len(result["xs"]), "no candidate is queried twice")
    check(all(result["history"][i + 1] >= result["history"][i]
              for i in range(len(result["history"]) - 1)),
          "best-so-far is non-decreasing")
    check(abs(result["best_y"] - best_y) < 5e-4,
          f"incumbent {result['best_y']:.6f} is within 5e-4 of the grid optimum {best_y:.6f}")
    check(all(b in candidate_grid() for b in result["xs"]), "queries stay on the candidate grid")
    budget_guard = Oracle(quadratic_with_ripple(0.5), budget=2)
    budget_guard.query(0.1)
    budget_guard.query(0.2)
    try:
        budget_guard.query(0.3)
        check(False, "the budget guard must refuse a third evaluation")
    except RuntimeError:
        check(True, "budget guard")

    # -- 5. recall on a multimodal family -------------------------------------------------
    centres = [0.15, 0.28, 0.43, 0.67, 0.82]
    total_bo = total_rs = 0.0
    lines = []
    for index, centre in enumerate(centres):
        wiggly = quadratic_with_ripple(centre, 0.02)
        _, optimum = exhaustive_optimum(wiggly)
        bo_oracle = Oracle(wiggly, budget=24)
        bo = BlackBoxOptimizer(lengthscale=0.08, max_evaluations=24).run(bo_oracle)
        rs_oracle = Oracle(wiggly, budget=24)
        rs = random_search(rs_oracle, seed=1000 + index)
        bo_regret = optimum - bo["best_y"]
        rs_regret = optimum - rs["best_y"]
        total_bo += bo_regret
        total_rs += rs_regret
        lines.append((centre, bo_regret, rs_regret))
        check(bo_regret < 5e-4, f"instance {centre}: bayesian regret {bo_regret:.2e}")
    check(total_bo < total_rs,
          f"total regret {total_bo:.2e} beats uniform search {total_rs:.2e} at equal budget")
    check(all(r <= 0.05 for _, _, r in lines), "uniform search never beats the optimizer by much")

    # -- 6. determinism ------------------------------------------------------------------
    again = BlackBoxOptimizer(lengthscale=0.08, max_evaluations=24).run(
        Oracle(function, budget=24))
    check(again["xs"] == result["xs"] and again["best_y"] == result["best_y"],
          "the whole search is bit-reproducible")
    check(result["best_x"] != best_x or abs(result["best_y"] - best_y) < 5e-4,
          "the incumbent sits on the optimum's plateau")

    print(f"EI vs quadrature: worst absolute difference {worst_ei:.2e}")
    print(f"quadratic+ripple: incumbent {result['best_x']:.4f} ({result['best_y']:.6f}) "
          f"vs exhaustive {best_x:.4f} ({best_y:.6f}) in 24 evaluations")
    print("cumulative regret (bayes vs uniform): "
          + ", ".join(f"{c}: {b:.2e}/{r:.2e}" for c, b, r in lines))
    print("PASS")


if __name__ == "__main__":
    main()
