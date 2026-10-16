"""Task 869 - Implement causality identification in time series with Granger causality and robustness.  (Tier: Master, CTCI Beyond - Machine Learning & AI)

For a bivariate stationary process "x Granger-causes y" means that the past of x carries
information about y_t beyond the past of y.  Operationally the lag-p target equation is
regressed in two nested ways

    restricted    y_t = a + sum_k b_k y_{t-k} + e_t
    unrestricted  y_t = a + sum_k b_k y_{t-k} + sum_k c_k x_{t-k} + e_t

and the F statistic comparing the residual sums of squares

    F = ((RSS_r - RSS_u) / p) / (RSS_u / (n_obs - 2p - 1))

is referred to the F(p, n_obs - 2p - 1) distribution.  Both regressions are solved with the
normal equations plus Gaussian elimination with partial pivoting; ``lstsq_mgs`` (modified
Gram-Schmidt QR) is an independent second solver used to cross-check the coefficients, and
``lstsq_exact`` repeats the arithmetic in :class:`fractions.Fraction` so the statistic can be
compared against an exact rational value.

Robustness: the OLS fit has a breakdown point of zero, so a handful of outliers can destroy
the statistic.  ``robust_fit`` runs iteratively reweighted least squares with Huber weights
w_i = min(1, c s / |r_i|) where s = 1.4826 MAD(residuals), c = 1.345 (86% Gaussian
efficiency), and the F statistic is then formed from the weighted residual sums of squares
with the weights taken from the unrestricted fit (the standard weighted-LS F test).  The
nominal p-value is an approximation - the weighted statistics are not exactly
F distributed - which is stated here and is exactly why the robustness checks below assert
*estimator accuracy* and *detection* rather than nominal coverage.

Also implemented: lag-order selection by BIC, an augmented Dickey-Fuller style unit-root
guard (t statistic of the lagged-level coefficient compared with the 5% critical value
-2.86) that decides whether the series must be differenced first, and input validation.

Verified against oracles: the F upper tail is checked against the exact closed form
1 - (1 + f/2)^-2 for F(2, 4) and the Cauchy identity P(F_{1,1} > 1) = 1/2; the whole
statistic is compared with exact rational arithmetic on a small integer series; OLS
coefficients are recovered to tolerance on a series generated with known coefficients; the
test detects the true causal direction and rejects the reverse one; the size of the test is
checked to be near nominal under H0 over 30 independent pairs; power grows monotonically
with the coupling strength; on outlier-contaminated data the Huber estimate is strictly
closer to the truth than OLS while on clean data the two agree; an integrated pair is flagged
non-stationary and the causal link is recovered after differencing; invalid inputs raise.
"""
import math
from fractions import Fraction

CRITICAL_VALUE_5PCT = -2.86  # published Dickey-Fuller critical value for the smallest sample sizes
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
    """splitmix64 stream; the only source of randomness in this file."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def uniform(self):
        self.state, raw = splitmix64(self.state)
        return (raw >> 11) * (1.0 / float(1 << 53))

    def gauss(self):
        u = self.uniform()
        u = u if u > 1e-12 else 1e-12
        return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * self.uniform())


# ------------------------------------------------------------------ linear algebra
def solve_gaussian(matrix, rhs):
    """Dense Gaussian elimination with partial pivoting; None when the system is singular."""
    n = len(rhs)
    a = [[float(matrix[i][j]) for j in range(n)] + [float(rhs[i])] for i in range(n)]
    for col in range(n):
        pivot = col
        for row in range(col + 1, n):
            if abs(a[row][col]) > abs(a[pivot][col]):
                pivot = row
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        inverse = 1.0 / a[col][col]
        for row in range(col + 1, n):
            factor = a[row][col] * inverse
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
    out = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = a[row][n]
        for k in range(row + 1, n):
            total -= a[row][k] * out[k]
        out[row] = total / a[row][row]
    return out


def gram(design):
    width = len(design[0])
    matrix = [[0.0] * width for _ in range(width)]
    for row in design:
        for i in range(width):
            left = row[i]
            if left == 0.0:
                continue
            for j in range(width):
                matrix[i][j] += left * row[j]
    return matrix


def cross(design, target):
    width = len(design[0])
    out = [0.0] * width
    for row, value in zip(design, target):
        for i in range(width):
            out[i] += row[i] * value
    return out


def lstsq_normal(design, target):
    """Least squares through X'X beta = X'y; None when the design is rank deficient."""
    return solve_gaussian(gram(design), cross(design, target))


def lstsq_mgs(design, target):
    """Least squares by modified Gram-Schmidt QR - an independent solver for cross-checks."""
    width = len(design[0])
    q_columns = []
    r = [[0.0] * width for _ in range(width)]
    for j in range(width):
        v = [row[j] for row in design]
        for k, q in enumerate(q_columns):
            projection = sum(q[i] * v[i] for i in range(len(v)))
            r[k][j] = projection
            for i in range(len(v)):
                v[i] -= projection * q[i]
        norm = math.sqrt(sum(value * value for value in v))
        r[j][j] = norm
        if norm < 1e-12:
            q_columns.append([0.0] * len(v))
            continue
        q_columns.append([value / norm for value in v])
    projections = [sum(q_columns[j][i] * target[i] for i in range(len(target)))
                   for j in range(width)]
    beta = [0.0] * width
    for j in range(width - 1, -1, -1):
        if abs(r[j][j]) < 1e-12:
            continue
        total = projections[j]
        for k in range(j + 1, width):
            total -= r[j][k] * beta[k]
        beta[j] = total / r[j][j]
    return beta


def lstsq_exact(design, target):
    """Exact (rational) least squares through the normal equations."""
    width = len(design[0])
    matrix = [[Fraction(0) for _ in range(width)] for _ in range(width)]
    rhs = [Fraction(0) for _ in range(width)]
    for row, value in zip(design, target):
        value = Fraction(value)
        for i in range(width):
            value_i = Fraction(row[i])
            rhs[i] += value_i * value
            for j in range(width):
                matrix[i][j] += value_i * Fraction(row[j])
    for col in range(width):
        pivot = None
        for row in range(col, width):
            if matrix[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            return None
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        for row in range(col + 1, width):
            if matrix[row][col] == 0:
                continue
            factor = matrix[row][col] / matrix[col][col]
            for k in range(col, width):
                matrix[row][k] -= factor * matrix[col][k]
            rhs[row] -= factor * rhs[col]
    beta = [Fraction(0) for _ in range(width)]
    for row in range(width - 1, -1, -1):
        total = rhs[row]
        for k in range(row + 1, width):
            total -= matrix[row][k] * beta[k]
        beta[row] = total / matrix[row][row]
    return beta


def dot(row, beta):
    total = 0.0
    for value, coefficient in zip(row, beta):
        total += value * coefficient
    return total


def residual_sum_of_squares(design, target, beta, weights=None):
    total = 0.0
    for row, value, weight in zip(design, target, weights if weights else [1.0] * len(target)):
        residual = value - dot(row, beta)
        total += weight * residual * residual
    return total


# ------------------------------------------------------------ F distribution (tail)
def betacf(a, b, x):
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betai(a, b, x):
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1.0 - x) / b


def f_upper_tail(f, d1, d2):
    """P(F > f) for an F(d1, d2) variate; exact up to the continued-fraction tolerance."""
    if f <= 0.0:
        return 1.0
    return betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


# ----------------------------------------------------------------- data generation
def make_var(seed, n, x_ar=(0.5, 0.1), y_ar=(0.4, 0.1), y_from_x=(0.0, 0.7), noise=1.0,
             burn=200):
    """Stationary VAR(2): x_t = a1 x_{t-1} + a2 x_{t-2} + e1,
    y_t = b1 y_{t-1} + b2 y_{t-2} + c1 x_{t-1} + c2 x_{t-2} + e2."""
    rng = Rng(seed)
    xs = [0.0, 0.0]
    ys = [0.0, 0.0]
    for _ in range(burn + n):
        e1 = noise * rng.gauss()
        e2 = noise * rng.gauss()
        x_t = x_ar[0] * xs[-1] + x_ar[1] * xs[-2] + e1
        y_t = (y_ar[0] * ys[-1] + y_ar[1] * ys[-2]
               + y_from_x[0] * xs[-1] + y_from_x[1] * xs[-2] + e2)
        xs.append(x_t)
        ys.append(y_t)
    return xs[burn:burn + n], ys[burn:burn + n]


# --------------------------------------------------------------- Granger machinery
def huber_weights(residuals, c=1.345):
    ordered = sorted(residuals)
    median = ordered[len(ordered) // 2]
    deviations = sorted(abs(value - median) for value in residuals)
    scale = 1.4826 * deviations[len(deviations) // 2]
    if scale <= 1e-12:
        return [1.0] * len(residuals)
    return [1.0 if abs(r) <= c * scale else c * scale / abs(r) for r in residuals]


def weighted_fit(design, target, weights):
    scaled = [[math.sqrt(w) * value for value in row] for row, w in zip(design, weights)]
    scaled_target = [math.sqrt(w) * value for value, w in zip(target, weights)]
    return lstsq_normal(scaled, scaled_target)


def robust_fit(design, target, iterations=25):
    """IRLS with Huber weights; returns (coefficients, final weights)."""
    beta = lstsq_normal(design, target)
    if beta is None:
        return None, [1.0] * len(target)
    weights = [1.0] * len(target)
    for _ in range(iterations):
        residuals = [value - dot(row, beta) for row, value in zip(design, target)]
        weights = huber_weights(residuals)
        updated = weighted_fit(design, target, weights)
        if updated is None:
            break
        beta = updated
    return beta, weights


def design_blocks(xs, ys, p):
    """Returns (restricted, unrestricted, target); lag blocks run from t-1 to t-p."""
    n = len(ys)
    observations = n - p
    if observations <= 2 * p + 2:
        raise ValueError("not enough observations for this lag order")
    target = [ys[t] for t in range(p, n)]
    restricted = []
    unrestricted = []
    for t in range(p, n):
        y_lags = [ys[t - k] for k in range(1, p + 1)]
        x_lags = [xs[t - k] for k in range(1, p + 1)]
        restricted.append([1.0] + y_lags)
        unrestricted.append([1.0] + y_lags + x_lags)
    return restricted, unrestricted, target


def granger_test(xs, ys, p, robust=False):
    """Test whether x Granger-causes y at lag order p."""
    if p <= 0:
        raise ValueError("lag order must be positive")
    if len(xs) != len(ys):
        raise ValueError("the two series must have the same length")
    restricted, unrestricted, target = design_blocks(xs, ys, p)
    observations = len(target)
    df_unrestricted = observations - (2 * p + 1)

    if robust:
        beta_unrestricted, weights = robust_fit(unrestricted, target)
        beta_restricted = weighted_fit(restricted, target, weights)
    else:
        weights = None
        beta_unrestricted = lstsq_normal(unrestricted, target)
        beta_restricted = lstsq_normal(restricted, target)

    if beta_unrestricted is None or beta_restricted is None:
        return {"f": 0.0, "p_value": 1.0, "effect": 0.0, "beta": None, "degenerate": True}

    rss_unrestricted = residual_sum_of_squares(unrestricted, target, beta_unrestricted, weights)
    rss_restricted = residual_sum_of_squares(restricted, target, beta_restricted, weights)
    if rss_unrestricted <= 1e-300:
        f_statistic = 0.0 if rss_restricted <= rss_unrestricted else float("inf")
        p_value = 1.0 if f_statistic == 0.0 else 0.0
    else:
        numerator = max(rss_restricted - rss_unrestricted, 0.0) / p
        f_statistic = numerator / (rss_unrestricted / df_unrestricted)
        p_value = f_upper_tail(f_statistic, p, df_unrestricted)
    return {
        "f": f_statistic,
        "p_value": p_value,
        "effect": sum(beta_unrestricted[1 + p:1 + 2 * p]),
        "beta": beta_unrestricted,
        "rss_restricted": rss_restricted,
        "rss_unrestricted": rss_unrestricted,
        "df_unrestricted": df_unrestricted,
        "degenerate": False,
    }


def select_lag_bic(xs, ys, p_max=5):
    """BIC over the target equation for lag orders 1..p_max; returns (best p, scores)."""
    if p_max < 1:
        raise ValueError("p_max must be at least 1")
    scores = []
    for p in range(1, p_max + 1):
        _restricted, unrestricted, target = design_blocks(xs, ys, p)
        beta = lstsq_normal(unrestricted, target)
        if beta is None:
            scores.append("inf")
            continue
        rss = residual_sum_of_squares(unrestricted, target, beta)
        observations = len(target)
        if rss <= 1e-300:
            scores.append(-math.inf)
            continue
        parameters = 2 * p + 1
        scores.append(observations * math.log(rss / observations) + parameters * math.log(observations))
    best = min(range(len(scores)), key=lambda index: scores[index])
    return best + 1, scores


def unit_root_t(values, lags=1):
    """t statistic of the lagged level in an ADF regression (compare with -2.86)."""
    n = len(values)
    design = []
    target = []
    for t in range(lags + 1, n):
        row = [1.0, values[t - 1]]
        for k in range(1, lags + 1):
            row.append(values[t - k] - values[t - k - 1])
        design.append(row)
        target.append(values[t] - values[t - 1])
    degree_of_freedom = len(target) - len(design[0])
    if degree_of_freedom <= 0:
        raise ValueError("not enough observations for a unit-root regression")
    beta = lstsq_normal(design, target)
    if beta is None:
        return 0.0
    rss = residual_sum_of_squares(design, target, beta)
    sigma_squared = rss / degree_of_freedom
    inverse_column = solve_gaussian(gram(design), [1.0 if i == 1 else 0.0 for i in range(2 + lags)])
    if inverse_column is None or inverse_column[1] <= 0.0:
        return 0.0
    return beta[1] / math.sqrt(sigma_squared * inverse_column[1])


def difference(values):
    return [values[i] - values[i - 1] for i in range(1, len(values))]


def main():
    # ---------------- F tail against closed forms ----------------
    check(abs(f_upper_tail(1.0, 1, 1) - 0.5) < 1e-12, "Cauchy identity P(F_{1,1} > 1) = 1/2")
    exact = (1.0 + 1.0 / 2.0) ** -2.0  # P(F_{2,4} > f) = (1 + d1 f / d2)^(-d2/2)
    check(abs(f_upper_tail(1.0, 2, 4) - exact) < 1e-12,
          f"F(2,4) tail mismatch: {f_upper_tail(1.0, 2, 4)} vs {exact}")
    check(f_upper_tail(0.0, 3, 7) == 1.0 and f_upper_tail(1e9, 3, 7) < 1e-9,
          "F tail must be 1 at 0 and vanish for large f")
    check(f_upper_tail(2.5, 4, 9) < f_upper_tail(1.5, 4, 9), "F tail must decrease in f")

    # ---------------- two independent least-squares solvers agree ----------------
    rng = Rng(4242)
    design = [[1.0, rng.gauss(), rng.gauss()] for _ in range(8)]
    target = [rng.gauss() for _ in range(8)]
    beta_normal = lstsq_normal(design, target)
    beta_qr = lstsq_mgs(design, target)
    check(all(abs(a - b) < 1e-9 for a, b in zip(beta_normal, beta_qr)),
          f"normal equations and QR disagree: {beta_normal} vs {beta_qr}")

    # ---------------- coefficient recovery and detection ----------------
    xs, ys = make_var(seed=80869, n=400, y_from_x=(0.0, 0.7))
    forward = granger_test(xs, ys, p=2)
    backward = granger_test(ys, xs, p=2)
    check(abs(forward["effect"] - 0.7) < 0.15,
          f"total causal effect should be ~0.7, got {forward['effect']:.4f}")
    check(forward["p_value"] < 0.01,
          f"x -> y must be detected, p={forward['p_value']:.3e}")
    check(forward["f"] > 0.0, "the unrestricted fit must improve on the restricted one")
    check(abs(forward["effect"]) > abs(backward["effect"]),
          "the detected direction must have the larger effect estimate")
    check(backward["p_value"] > 0.01,
          f"y -> x must not be detected, p={backward['p_value']:.3e}")

    lag, scores = select_lag_bic(xs, ys, p_max=5)
    check(lag == 2, f"BIC must choose the true lag order 2, got {lag} ({scores})")

    # a delayed effect on a white-noise cause: the minimal sufficient lag order is identified
    # and using too few lags genuinely misses the link
    wx, wy = make_var(seed=4242, n=400, x_ar=(0.0, 0.0), y_from_x=(0.0, 0.7))
    white_lag, _white_scores = select_lag_bic(wx, wy, p_max=5)
    check(white_lag == 2, f"BIC must identify the delayed lag order 2, got {white_lag}")
    check(granger_test(wx, wy, p=1)["p_value"] > 0.05,
          "lag 1 alone must miss a lag-2-only effect")
    check(granger_test(wx, wy, p=2)["p_value"] < 1e-6,
          "lag 2 must detect the delayed effect")
    check(granger_test(wy, wx, p=2)["p_value"] > 0.05, "reverse test must stay insignificant")

    # ---------------- exact rational cross-check of the whole statistic ----------------
    tiny_x = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    tiny_y = [2, 7, 1, 8, 2, 8, 1, 8, 2, 8]
    restricted, unrestricted, exact_target = design_blocks(tiny_x, tiny_y, 2)
    exact_beta_u = lstsq_exact(unrestricted, exact_target)
    exact_beta_r = lstsq_exact(restricted, exact_target)
    exact_rss_u = sum((Fraction(value) - sum(Fraction(coefficient) * Fraction(row[i])
                                             for i, coefficient in enumerate(exact_beta_u))) ** 2
                      for row, value in zip(unrestricted, exact_target))
    exact_rss_r = sum((Fraction(value) - sum(Fraction(coefficient) * Fraction(row[i])
                                             for i, coefficient in enumerate(exact_beta_r))) ** 2
                      for row, value in zip(restricted, exact_target))
    df_exact = len(exact_target) - 5
    exact_f = float(((exact_rss_r - exact_rss_u) / 2) / (exact_rss_u / df_exact))
    float_test = granger_test(tiny_x, tiny_y, p=2)
    check(abs(float_test["f"] - exact_f) <= 1e-9 * max(1.0, abs(exact_f)),
          f"exact rational F {exact_f} differs from float F {float_test['f']}")

    # ---------------- size of the test under H0 ----------------
    rejections = 0
    trials = 30
    for seed in range(trials):
        hx, hy = make_var(seed=1000 + seed, n=200, y_from_x=(0.0, 0.0))
        if granger_test(hx, hy, p=2)["p_value"] < 0.05:
            rejections += 1
    check(rejections <= 6, f"false-positive rate too high: {rejections}/{trials} at 5%")

    # ---------------- power grows with the coupling ----------------
    stats = []
    for coupling in (0.15, 0.35, 0.55, 0.75):
        px, py = make_var(seed=555, n=300, y_from_x=(0.0, coupling))
        stats.append(granger_test(px, py, p=2))
    for previous, current in zip(stats, stats[1:]):
        check(current["f"] > previous["f"], "power must increase with the coupling strength")
    check(stats[-1]["p_value"] < stats[0]["p_value"], "p-values must fall as coupling grows")

    # ---------------- robustness against outliers ----------------
    clean_x, clean_y = make_var(seed=31337, n=300, y_from_x=(0.0, 0.6))
    dirty_y = list(clean_y)
    for index in range(0, len(dirty_y), 17):
        dirty_y[index] += 40.0 * (1.0 if (index // 17) % 2 == 0 else -1.0)
    clean_ols = granger_test(clean_x, clean_y, p=2)
    dirty_ols = granger_test(clean_x, dirty_y, p=2)
    dirty_robust = granger_test(clean_x, dirty_y, p=2, robust=True)
    truth = 0.6
    check(clean_ols["effect"] > 0.5, "clean-data estimate must be near the true effect")
    check(abs(dirty_robust["effect"] - truth) < abs(dirty_ols["effect"] - truth),
          f"robust estimate {dirty_robust['effect']:.4f} not closer to {truth} than OLS "
          f"{dirty_ols['effect']:.4f}")
    check(abs(dirty_robust["effect"] - clean_ols["effect"])
          < abs(dirty_ols["effect"] - clean_ols["effect"]),
          "robust estimate must stay closer to the uncontaminated estimate")
    check(abs(dirty_robust["effect"] - truth) < 0.15,
          f"robust estimate drifted too far: {dirty_robust['effect']:.4f}")
    check(dirty_robust["p_value"] < 0.05,
          f"robust test must still detect the true link, p={dirty_robust['p_value']:.3e}")
    check(abs(clean_ols["f"] - granger_test(clean_x, clean_y, p=2, robust=True)["f"])
          < 0.5 * clean_ols["f"],
          "on clean data the robust and OLS statistics must not be far apart")

    # ---------------- stationarity guard and differencing ----------------
    d_x, d_y = make_var(seed=97, n=300, y_from_x=(0.0, 0.9))
    level_x = []
    level_y = []
    running_x = 0.0
    running_y = 0.0
    for step in range(len(d_x)):
        running_x += d_x[step]
        running_y += d_y[step]
        level_x.append(running_x)
        level_y.append(running_y)
    check(unit_root_t(level_x) > CRITICAL_VALUE_5PCT,
          f"integrated series must look non-stationary, t={unit_root_t(level_x):.3f}")
    check(unit_root_t(d_x) < CRITICAL_VALUE_5PCT,
          f"differenced series must look stationary, t={unit_root_t(d_x):.3f}")
    check(granger_test(level_x, level_y, p=2)["f"] > 0.0, "levels regression must still fit")
    detrended = granger_test(difference(level_x), difference(level_y), p=2)
    check(detrended["p_value"] < 0.01,
          f"causality must be recovered after differencing, p={detrended['p_value']:.3e}")
    check(abs(detrended["effect"] - 0.9) < 0.15,
          f"differenced effect estimate {detrended['effect']:.4f} is off")

    # ---------------- degenerate input and validation ----------------
    degenerate = granger_test([1.0] * 40, [5.0] * 40, p=2)
    check(degenerate["p_value"] == 1.0 and degenerate["degenerate"],
          "constant series must be reported as degenerate, not crash")
    for args, why in (((xs, ys, 0), "p = 0"), ((xs, ys, -1), "negative p"),
                      ((xs, ys[:-1], 2), "length mismatch"),
                      ((xs[:6], ys[:6], 2), "too few observations")):
        try:
            granger_test(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {why}")

    # ---------------- determinism ----------------
    check(granger_test(xs, ys, p=2) == forward, "repeated runs must agree bit for bit")
    check(granger_test(clean_x, dirty_y, p=2, robust=True) == dirty_robust,
          "robust runs must be deterministic")

    print(f"x->y F={forward['f']:.3f} p={forward['p_value']:.3e} effect={forward['effect']:.4f}")
    print(f"y->x F={backward['f']:.3f} p={backward['p_value']:.3e} (BIC lag={lag})")
    print(f"H0 rejections {rejections}/{trials}  power F={[round(s['f'], 2) for s in stats]}")
    print(f"outliers: ols={dirty_ols['effect']:.4f} huber={dirty_robust['effect']:.4f} "
          f"clean={clean_ols['effect']:.4f} truth={truth}")
    print("PASS")


if __name__ == "__main__":
    main()
