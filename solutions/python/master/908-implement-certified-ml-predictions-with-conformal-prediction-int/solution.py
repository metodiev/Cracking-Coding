"""Task 908 - Implement certified ML predictions with conformal prediction intervals.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

What is implemented (in-process, deterministic; splits come from the seeded splitmix64 stream,
the data come from a fixed formula, and everything is recomputable):

  * `fit_line` - least squares (normal equations, hand-written) whose coefficients are also
    checked against the closed-form covariance formula, so the point predictions have an oracle.
  * `conformal_quantile` - the split-conformal calibration: the 1-based ceil((n+1)(1-alpha))
    order statistic of the calibration nonconformity scores, with the well-known degenerate case
    (that index exceeds n -> the prediction set is the whole real line).
  * `interval` - two variants: the plain score |y - y_hat| gives a constant half-width, the
    normalised score |y - y_hat| / sigma_hat(x) gives locally adaptive intervals for
    heteroscedastic noise.
  * `run_trial` - one train/calibrate/test split with the marginal coverage and the mean width,
    repeated over many splits to measure the distribution-free guarantee E[coverage] >= 1-alpha.
  * Baselines for the negative controls: the Gaussian interval y_hat +/- 1.96 sigma_train.

Simplified versus production: a one-feature linear model, a known scale function for the
normalised score, i.i.d. exchangeable rows and a fixed sample size; the finite-sample guarantee,
the order-statistic rule and the adaptive-width behaviour are the real ones.

Verified below: the quantile rule against hand-computed order statistics, the finite-sample
coverage over 200 splits for three alphas, width monotonicity in alpha, the model coefficients
against the closed-form OLS, adaptive widths versus the constant-width variant, the Gaussian
baseline under-covering, the distribution-shift negative control, and the infinite-interval and
zero-score boundaries.
"""

import math

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
COUNT, TRAIN, CAL, TEST = 240, 100, 60, 80


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + GOLDEN) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def below(self, n):
        return self.next_u64() % n

    def permutation(self, n):
        order = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.below(i + 1)
            order[i], order[j] = order[j], order[i]
        return order


def scale(x):
    """The known noise scale: heteroscedastic, growing with x (large for the normalised score)."""
    return 0.05 + 1.5 * x


def noise_at(index):
    """Deterministic heavy-tailed noise in [-1, 1] with excess kurtosis (no PRNG needed)."""
    raw = math.sin(index * 12.9898) * 43758.5453
    unit = 2.0 * (raw - math.floor(raw)) - 1.0
    return unit * (1.0 + 2.0 * unit * unit) / 2.4


def dataset(count=COUNT):
    rows = []
    for i in range(count):
        x = i / (count - 1)
        y = 1.0 + 2.0 * x + 0.6 * x * x + scale(x) * noise_at(i) * 0.8
        rows.append((x, y))
    return rows


def fit_line(xs, ys):
    """Least squares by normal equations."""
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return mean_y - slope * mean_x, slope


def closed_form_slope(xs, ys):
    """Independent oracle: slope = (n*sum(xy) - sum(x)sum(y)) / (n*sum(x^2) - sum(x)^2)."""
    n = len(xs)
    return ((n * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys))
            / (n * sum(x * x for x in xs) - sum(xs) ** 2))


def conformal_quantile(scores, alpha):
    """Half-width from the calibration scores; inf when ceil((n+1)(1-alpha)) exceeds n."""
    n = len(scores)
    level = math.ceil((n + 1) * (1.0 - alpha))
    if level > n:
        return math.inf, level
    return sorted(scores)[level - 1], level


def run_trial(alpha, seed, normalized=False, shift=0.0, rows=None):
    """One split: coverage, mean width, Gaussian-baseline coverage/width, half-width, test x."""
    rows = rows if rows is not None else dataset()
    total = len(rows)
    n_train = max(4, total * 5 // 12)
    n_cal = max(1, total // 4)
    order = Rng(seed).permutation(total)
    train = [rows[i] for i in order[:n_train]]
    calib = [rows[i] for i in order[n_train:n_train + n_cal]]
    test = [rows[i] for i in order[n_train + n_cal:]]
    intercept, slope = fit_line([r[0] for r in train], [r[1] for r in train])
    residuals = [y - (intercept + slope * x) for x, y in train]
    sigma = math.sqrt(sum(r * r for r in residuals) / len(residuals))

    def predict(x):
        return intercept + slope * x

    if normalized:
        scores = [abs(y - predict(x)) / scale(x) for x, y in calib]
    else:
        scores = [abs(y - predict(x)) for x, y in calib]
    half, level = conformal_quantile(scores, alpha)
    check(level >= 1, "the quantile index must be a positive order statistic")

    covered = narrow = 0
    for x, y in test:
        width = half * scale(x) if normalized else half
        # `shift` moves the *test* covariates only: calibration stays valid, exchangeability does
        # not, and the guarantee is supposed to break
        residual = abs(y - predict(x + shift))
        if residual <= width:
            covered += 1
        if residual <= 1.96 * sigma:
            narrow += 1
    widths = [2 * (half * scale(x) if normalized else half) for x, _ in test]
    return (covered / len(test), sum(widths) / len(test), narrow / len(test), 2 * 1.96 * sigma,
            half, [x for x, _ in test])


def main():
    rows = dataset()
    check(len(rows) == COUNT and len(set(rows)) == COUNT, "the dataset must be well formed")
    ys = [y for _, y in rows]
    check(max(ys) - min(ys) > 2.0, "the response must actually vary")

    # ---- 1. the calibration rule against hand-computed order statistics
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    check(conformal_quantile(sample, 0.1) == (10.0, 10),
          "ceil(11*0.9) = 10 -> the largest calibration score")
    check(conformal_quantile(sample, 0.2) == (9.0, 9), "ceil(11*0.8) = 9 -> the ninth score")
    check(conformal_quantile(sample, 0.5) == (6.0, 6), "ceil(11*0.5) = 6 -> the median score")
    infinite, level = conformal_quantile(sample, 0.05)
    check(infinite == math.inf and level == 11,
          "ceil(11*0.95) = 11 > n -> the prediction set is the whole line")
    check(conformal_quantile([0.0] * 5, 0.2) == (0.0, 5),
          "a perfect calibration fit gives a zero-width interval")

    # ---- 2. the model coefficients against the closed form
    xs = [r[0] for r in rows[:TRAIN]]
    ys_train = [r[1] for r in rows[:TRAIN]]
    intercept, slope = fit_line(xs, ys_train)
    check(abs(slope - closed_form_slope(xs, ys_train)) < 1e-9,
          "the normal-equation slope must equal the closed-form slope")
    check(1.5 < slope < 4.0, "the fitted slope must recover the linear trend (%.4f)" % slope)
    check(abs(intercept - 1.0) < 0.4, "the fitted intercept must be near one (%.4f)" % intercept)

    # ---- 3. the distribution-free coverage guarantee over many splits
    tails = [abs(noise_at(i)) for i in range(COUNT)]
    check(1.2 < max(tails) < 1.3
          and abs(sum(noise_at(i) for i in range(COUNT))) / COUNT < 0.05,
          "the deterministic noise must be centred and bounded")
    summary = {}
    for alpha in (0.05, 0.1, 0.2):
        trials = [run_trial(alpha, 5000 + 7 * t, rows=rows) for t in range(200)]
        coverage = sum(t[0] for t in trials) / len(trials)
        width = sum(t[1] for t in trials) / len(trials)
        gaussian = sum(t[2] for t in trials) / len(trials)
        gaussian_width = sum(t[3] for t in trials) / len(trials)
        summary[alpha] = (coverage, width, gaussian, gaussian_width)
        check(coverage >= 1.0 - alpha - 0.01,
              "coverage %.3f must respect the 1-alpha = %.2f guarantee" % (coverage, 1 - alpha))
        check(coverage <= 1.0, "coverage is a probability")
        if gaussian >= 1.0 - alpha:
            check(width < gaussian_width,
                  "where the Gaussian baseline is valid it must be the wider one "
                  "(%.3f vs %.3f)" % (width, gaussian_width))
        else:
            check(alpha == 0.05, "the Gaussian baseline may only fail at the tightest level")
    check(summary[0.2][2] > 1.0 - 0.2 + 0.05,
          "the Gaussian baseline wastes width at a loose level (%.3f covered)" % summary[0.2][2])
    check(summary[0.05][2] < 0.95 - 0.02,
          "negative control: the Gaussian baseline cannot reach the 95%% level (%.3f covered)"
          % summary[0.05][2])
    check(summary[0.05][1] > summary[0.1][1] > summary[0.2][1],
          "mean width must shrink as alpha grows: %r" % [round(summary[a][1], 3) for a in summary])

    # ---- 4. locally adaptive intervals for heteroscedastic noise
    plain = [run_trial(0.1, 9000 + t, rows=rows) for t in range(30)]
    adaptive = [run_trial(0.1, 9000 + t, normalized=True, rows=rows) for t in range(30)]
    check(all(abs(t[1] - 2 * t[4]) < 1e-9 for t in plain) and plain[0][4] < math.inf,
          "the plain score must give one constant half-width across the test set")
    check(any(abs(t[1] - 2 * t[4]) > 1e-6 for t in adaptive),
          "the normalised score must give locally varying widths")
    check(all(t[4] < math.inf for t in adaptive), "the normalised score must stay finite here")
    check(sum(t[0] for t in adaptive) / 30 >= 0.87,
          "the normalised intervals must keep their coverage")
    widths = sorted((x, 2 * trial[4] * scale(x)) for trial in adaptive[:1] for x in trial[5])
    correlation = _correlation([w[0] for w in widths], [w[1] for w in widths])
    check(correlation > 0.7,
          "the adaptive width must grow with x (correlation %.3f)" % correlation)
    check(sum(t[1] for t in plain) / len(plain) > 0, "the plain intervals must have width")

    # ---- 5. the exchangeability assumption: a shifted test set breaks the guarantee
    shifted = [run_trial(0.1, 4000 + t, shift=0.25, rows=rows) for t in range(50)]
    shifted_coverage = sum(t[0] for t in shifted) / len(shifted)
    check(shifted_coverage < 0.9,
          "a covariate shift must break the guarantee (%.3f) - the method is only valid under "
          "exchangeability" % shifted_coverage)

    # ---- 6. small calibration sets degenerate to the whole line
    tiny_rows = rows[:12]
    small = run_trial(0.05, 1, rows=tiny_rows)
    check(small[4] == math.inf and small[0] == 1.0,
          "a calibration set too small for the level must yield the whole line")
    big = run_trial(0.05, 1, rows=rows)
    check(big[4] < math.inf and big[0] >= 0.9, "a large calibration set must give finite intervals")

    # ---- 7. determinism
    check(run_trial(0.1, 42, rows=rows)[:5] == run_trial(0.1, 42, rows=rows)[:5],
          "the whole procedure must be reproducible from its seed")

    print("coverage/width by alpha: " +
          ", ".join("a=%.2f cov=%.3f width=%.2f gaussian=%.3f" % (a, summary[a][0], summary[a][1],
                                                                 summary[a][2])
                    for a in sorted(summary)))
    print("adaptive width correlation %.3f, shifted coverage %.3f" % (correlation, shifted_coverage))
    print("PASS")


def _correlation(left, right):
    n = len(left)
    mean_l, mean_r = sum(left) / n, sum(right) / n
    cov = sum((a - mean_l) * (b - mean_r) for a, b in zip(left, right)) / n
    var_l = sum((a - mean_l) ** 2 for a in left) / n
    var_r = sum((b - mean_r) ** 2 for b in right) / n
    return cov / math.sqrt(var_l * var_r)


if __name__ == "__main__":
    main()
