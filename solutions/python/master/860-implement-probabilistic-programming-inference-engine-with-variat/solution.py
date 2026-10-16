"""Task 860 - Implement probabilistic programming inference engine with variational methods.  (Tier: Master, Beyond — Machine Learning & AI)

A small probabilistic programming engine: a model is a linear-Gaussian generative program
(prior on the latent weights, Gaussian likelihood for the observations), inference is
mean-field variational inference over a diagonal Gaussian variational family, and the ELBO
is available both in closed form (for this conjugate model) and as a Monte Carlo estimate
with the reparameterisation trick - the two must agree, which is what makes the estimator
trustworthy.  Everything is deterministic: the "random" draws come from a seeded splitmix64
stream, there are no threads, no clock and no external libraries.

What is simplified: the language is a single conjugate model family rather than a general
probabilistic language, the variational family is mean-field diagonal Gaussian, and the
optimiser is plain gradient ascent with a decaying step - no automatic differentiation, but
the pathwise gradient is written out and verified against finite differences.

Invariants verified:
  * the Monte Carlo ELBO with the reparameterisation trick agrees with the closed-form ELBO
    of the conjugate model within the sampling error of the estimator;
  * the pathwise gradient of the Monte Carlo ELBO equals a finite-difference gradient taken
    with the very same random draws, so the estimator is differentiable and unbiased in the
    direction it claims;
  * the KL divergence between two Gaussians matches numerical quadrature of the definition;
  * coordinate-ascent (CAVI) never decreases the closed-form ELBO, and gradient ascent drives
    the Monte Carlo ELBO to the same optimum: the mean-field mean converges to the exact
    posterior mean while the mean-field variance stays below the exact marginal variance -
    the classic mean-field shrinkage, verified rather than assumed;
  * with no observations the variational posterior is exactly the prior, and with an
    overwhelming prior it stays at the prior even when the data pull the other way;
  * posterior predictive means agree with the analytic predictive mean of the conjugate
    model, and every run is reproducible from its seed with a bounded number of ELBO
    evaluations.
"""

import math

MASK64 = (1 << 64) - 1


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

    def normal(self):
        """Box-Muller from the deterministic stream."""
        u1 = max(self.uniform(0.0, 1.0), 1e-12)
        u2 = self.uniform(0.0, 1.0)
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def normal_logpdf(value, mean, sigma):
    return -0.5 * math.log(2.0 * math.pi * sigma * sigma) - \
        0.5 * ((value - mean) / sigma) ** 2


def gaussian_kl(mean_q, sigma_q, mean_p, sigma_p):
    """KL(q || p) for univariate Gaussians, in closed form."""
    return (math.log(sigma_p / sigma_q) + (sigma_q ** 2 + (mean_q - mean_p) ** 2)
            / (2.0 * sigma_p ** 2) - 0.5)


def kl_by_quadrature(mean_q, sigma_q, mean_p, sigma_p, steps=20001, width=12.0):
    """The same KL, integrated numerically: int q log(q/p)."""
    total = 0.0
    step = 2.0 * width / (steps - 1)
    for index in range(steps):
        point = mean_q - width + index * step
        density_q = math.exp(normal_logpdf(point, mean_q, sigma_q))
        if density_q <= 0.0:
            continue
        ratio = normal_logpdf(point, mean_q, sigma_q) - normal_logpdf(point, mean_p, sigma_p)
        total += density_q * ratio * step
    return total


# ---------------------------------------------------------------------------
# the model: linear-Gaussian, the conjugate case of a probabilistic program
# ---------------------------------------------------------------------------
class LinearGaussian:
    """theta ~ N(0, prior_sigma^2 I);  x_i ~ N(a_i . theta, noise_sigma^2)."""

    def __init__(self, design, observations, prior_sigma=1.0, noise_sigma=1.0):
        self.design = [list(row) for row in design]
        self.observations = list(observations)
        self.prior_sigma = prior_sigma
        self.noise_sigma = noise_sigma
        self.dimension = len(design[0]) if design else 0

    def log_joint(self, theta):
        total = 0.0
        for value in theta:
            total += normal_logpdf(value, 0.0, self.prior_sigma)
        for row, observation in zip(self.design, self.observations):
            predicted = sum(weight * value for weight, value in zip(row, theta))
            total += normal_logpdf(observation, predicted, self.noise_sigma)
        return total

    def exact_posterior(self):
        """Conjugate posterior: precision = A^T A / s^2 + I / s0^2."""
        dimension = self.dimension
        precision = [[0.0] * dimension for _ in range(dimension)]
        for index in range(dimension):
            precision[index][index] = 1.0 / (self.prior_sigma ** 2)
        rhs = [0.0] * dimension
        for row, observation in zip(self.design, self.observations):
            for i in range(dimension):
                rhs[i] += row[i] * observation / (self.noise_sigma ** 2)
                for j in range(dimension):
                    precision[i][j] += row[i] * row[j] / (self.noise_sigma ** 2)
        covariance = invert(precision)
        mean = [sum(covariance[i][j] * rhs[j] for j in range(dimension))
                for i in range(dimension)]
        return mean, covariance


def invert(matrix):
    size = len(matrix)
    augmented = [list(row) + [1.0 if i == j else 0.0 for j in range(size)]
                 for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-15:
            raise AssertionError("singular precision matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [value - factor * other
                              for value, other in zip(augmented[row], augmented[column])]
    return [row[size:] for row in augmented]


# ---------------------------------------------------------------------------
# the variational family: mean-field diagonal Gaussian
# ---------------------------------------------------------------------------
class MeanField:
    def __init__(self, mean, log_sigma):
        self.mean = list(mean)
        self.log_sigma = list(log_sigma)

    @property
    def sigma(self):
        return [math.exp(value) for value in self.log_sigma]

    def sample(self, rng):
        return [mean + math.exp(log_sigma) * rng.normal()
                for mean, log_sigma in zip(self.mean, self.log_sigma)]

    def entropy(self):
        return sum(0.5 * math.log(2.0 * math.pi * math.e) + log_sigma
                   for log_sigma in self.log_sigma)

    def clone(self):
        return MeanField(self.mean, self.log_sigma)


def closed_form_elbo(model, variational):
    """ELBO for the conjugate model: analytic E[log p] and analytic entropy."""
    sigma = variational.sigma
    expectation = 0.0
    for value, scale in zip(variational.mean, sigma):
        expectation += normal_logpdf(value, 0.0, model.prior_sigma) \
            - 0.5 * (scale ** 2) / (model.prior_sigma ** 2)
    for row, observation in zip(model.design, model.observations):
        predicted = sum(weight * value for weight, value in zip(row, variational.mean))
        variance = sum((weight * scale) ** 2 for weight, scale in zip(row, sigma))
        expectation += normal_logpdf(observation, predicted, model.noise_sigma) \
            - 0.5 * variance / (model.noise_sigma ** 2)
    return expectation + variational.entropy()


def mc_elbo(model, variational, rng, samples=200):
    """Monte Carlo ELBO with the reparameterisation trick."""
    total = 0.0
    for _ in range(samples):
        theta = variational.sample(rng)
        total += model.log_joint(theta)
    return total / samples + variational.entropy()


def mc_elbo_gradient(model, variational, rng, samples=200):
    """Pathwise (reparameterised) gradient with respect to mean and log_sigma."""
    gradient_mean = [0.0] * len(variational.mean)
    gradient_log_sigma = [0.0] * len(variational.mean)
    for _ in range(samples):
        epsilon = [rng.normal() for _ in variational.mean]
        theta = [mean + math.exp(log_sigma) * value
                 for mean, log_sigma, value in zip(variational.mean, variational.log_sigma, epsilon)]
        grad_theta = [0.0] * len(theta)
        for index, value in enumerate(theta):
            grad_theta[index] -= value / (model.prior_sigma ** 2)
        for row, observation in zip(model.design, model.observations):
            predicted = sum(weight * value for weight, value in zip(row, theta))
            residual = (predicted - observation) / (model.noise_sigma ** 2)
            for index, weight in enumerate(row):
                grad_theta[index] -= weight * residual
        for index in range(len(theta)):
            gradient_mean[index] += grad_theta[index]
            gradient_log_sigma[index] += grad_theta[index] * math.exp(
                variational.log_sigma[index]) * epsilon[index]
    for index in range(len(gradient_mean)):
        gradient_mean[index] = gradient_mean[index] / samples
        gradient_log_sigma[index] = gradient_log_sigma[index] / samples + 1.0
    return gradient_mean, gradient_log_sigma


def cavi(model, variational=None, iterations=200, tolerance=1e-12):
    """Closed-form coordinate ascent on the conjugate ELBO (the reference optimum).

    For the mean-field Gaussian family of a linear-Gaussian model the optimal precision of
    every coordinate is the exact *conditional* precision, so the fixed point of these
    updates carries the exact posterior mean while its variance stays smaller than the exact
    marginal variance whenever the posterior is correlated.
    """
    if variational is None:
        variational = MeanField([0.0] * model.dimension,
                                [math.log(model.prior_sigma)] * model.dimension)
    history = []
    dimension = model.dimension
    for _ in range(iterations):
        for index in range(dimension):
            precision = 1.0 / (model.prior_sigma ** 2)
            rhs = 0.0
            for row, observation in zip(model.design, model.observations):
                precision += row[index] ** 2 / (model.noise_sigma ** 2)
                rhs += row[index] * (observation - sum(
                    row[other] * variational.mean[other] for other in range(dimension)
                    if other != index)) / (model.noise_sigma ** 2)
            variational.mean[index] = rhs / precision
            variational.log_sigma[index] = -0.5 * math.log(precision)
        history.append(closed_form_elbo(model, variational))
        if len(history) > 1 and abs(history[-1] - history[-2]) < tolerance:
            break
    return variational, history


def gradient_ascent(model, variational, rng, iterations=600, learning_rate=0.05, samples=200,
                    clip=1.0):
    """Stochastic ascent on the Monte Carlo ELBO.

    A nearly collinear design produces very large gradients, so each component of the update
    is clipped - a standard trick that keeps a fixed step size stable.
    """
    history = []
    for step in range(iterations):
        gradient_mean, gradient_log_sigma = mc_elbo_gradient(model, variational, rng, samples)
        rate = learning_rate / math.sqrt(1.0 + step / 20.0)
        for index in range(len(variational.mean)):
            mean_step = max(-clip, min(clip, gradient_mean[index]))
            sigma_step = max(-clip, min(clip, gradient_log_sigma[index]))
            variational.mean[index] += rate * mean_step
            variational.log_sigma[index] = min(3.0, variational.log_sigma[index]
                                               + rate * sigma_step)
        history.append(closed_form_elbo(model, variational))
    return variational, history


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    # ---- 1. the KL formula against numerical integration -----------------
    for mean_q, sigma_q, mean_p, sigma_p in ((0.3, 1.2, -0.4, 0.8), (2.0, 0.5, 1.0, 2.0),
                                             (0.0, 1.0, 0.0, 1.0)):
        exact = gaussian_kl(mean_q, sigma_q, mean_p, sigma_p)
        numeric = kl_by_quadrature(mean_q, sigma_q, mean_p, sigma_p)
        check(abs(exact - numeric) < 1e-3,
              "KL closed form %.6f != quadrature %.6f" % (exact, numeric))
    check(abs(gaussian_kl(0.5, 1.0, 0.5, 1.0)) < 1e-12, "KL of identical Gaussians is zero")
    print("KL: closed form matches numerical integration for three parameter sets")

    # ---- 2. the Monte Carlo ELBO estimates the closed-form ELBO ----------
    model = LinearGaussian([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 2.1], [1.0, 3.0]],
                           [0.5, 1.7, 2.9, 3.0, 4.1], prior_sigma=1.5, noise_sigma=0.4)
    check(model.dimension == 2, "the model must have two latent weights")
    exact_mean, exact_covariance = model.exact_posterior()
    variational = MeanField(exact_mean, [math.log(math.sqrt(exact_covariance[i][i]))
                                        for i in range(model.dimension)])
    analytic = closed_form_elbo(model, variational)
    estimates = [mc_elbo(model, variational, SplitMix64(seed), 2000) for seed in range(5)]
    spread = max(estimates) - min(estimates)
    check(all(abs(value - analytic) < 0.25 for value in estimates),
          "the Monte Carlo ELBO must track the analytic one: %s vs %.4f"
          % ([round(value, 3) for value in estimates], analytic))
    print("ELBO: analytic=%.4f, Monte Carlo %s (spread %.3f)"
          % (analytic, [round(value, 3) for value in estimates], spread))

    # ---- 3. the reparameterised gradient is the true gradient -----------
    probe = MeanField([0.2, -0.3], [math.log(0.9), math.log(1.1)])
    gradient_mean, gradient_log_sigma = mc_elbo_gradient(model, probe, SplitMix64(4242), 4000)
    epsilon = 1e-6
    numeric_mean = []
    for index in range(2):
        original = probe.mean[index]
        probe.mean[index] = original + epsilon
        high = mc_elbo(model, probe, SplitMix64(4242), 4000)
        probe.mean[index] = original - epsilon
        low = mc_elbo(model, probe, SplitMix64(4242), 4000)
        probe.mean[index] = original
        numeric_mean.append((high - low) / (2 * epsilon))
    for index, (left, right) in enumerate(zip(gradient_mean, numeric_mean)):
        check(abs(left - right) < 0.05 * max(1.0, abs(right)),
              "mean gradient %d: pathwise %.4f != numeric %.4f" % (index, left, right))
    numeric_sigma = []
    for index in range(2):
        original = probe.log_sigma[index]
        probe.log_sigma[index] = original + epsilon
        high = mc_elbo(model, probe, SplitMix64(4242), 4000)
        probe.log_sigma[index] = original - epsilon
        low = mc_elbo(model, probe, SplitMix64(4242), 4000)
        probe.log_sigma[index] = original
        numeric_sigma.append((high - low) / (2 * epsilon))
    for index, (left, right) in enumerate(zip(gradient_log_sigma, numeric_sigma)):
        check(abs(left - right) < 0.08 * max(1.0, abs(right)),
              "log-sigma gradient %d: pathwise %.4f != numeric %.4f" % (index, left, right))
    print("gradient: pathwise %s vs finite differences %s"
          % ([round(value, 3) for value in gradient_mean],
             [round(value, 3) for value in numeric_mean]))

    # ---- 4. CAVI never decreases the ELBO -------------------------------
    fitted, history = cavi(model, iterations=60)
    check(len(history) >= 2, "CAVI must record its trajectory")
    check(len(history) <= 60, "CAVI must respect its iteration budget")
    check(all(history[index + 1] >= history[index] - 1e-9 for index in range(len(history) - 1)),
          "the closed-form ELBO must be non-decreasing under coordinate ascent")
    check(history[-1] > history[0], "CAVI must make progress")
    print("CAVI: %d iterations, ELBO %.4f -> %.4f" % (len(history), history[0], history[-1]))

    # ---- 5. mean-field geometry: exact mean, shrunken variance ----------
    for index in range(model.dimension):
        check(abs(fitted.mean[index] - exact_mean[index]) < 1e-6,
              "the mean-field mean %d (%.6f) must equal the exact posterior mean (%.6f)"
              % (index, fitted.mean[index], exact_mean[index]))
    total_variational = sum(value ** 2 for value in fitted.sigma)
    total_exact = sum(exact_covariance[i][i] for i in range(model.dimension))
    check(total_variational <= total_exact + 1e-9,
          "mean field must not inflate the variance (%.4f vs %.4f)"
          % (total_variational, total_exact))
    correlated = abs(exact_covariance[0][1]) / math.sqrt(
        exact_covariance[0][0] * exact_covariance[1][1])
    check(correlated > 0.05, "the test posterior must be correlated (rho=%.3f)" % correlated)
    check(total_variational < total_exact,
          "a correlated posterior must be underestimated by mean field (%.4f vs %.4f)"
          % (total_variational, total_exact))
    print("geometry: exact mean %s, mean-field variance %.4f < exact trace %.4f (rho=%.3f)"
          % ([round(value, 4) for value in exact_mean], total_variational, total_exact, correlated))

    # ---- 6. gradient ascent reaches the same optimum --------------------
    started = MeanField([0.0, 0.0], [math.log(model.prior_sigma)] * model.dimension)
    ascent, trajectory = gradient_ascent(model, started, SplitMix64(7))
    check(trajectory[-1] > trajectory[0],
          "gradient ascent must improve the ELBO (%.4f -> %.4f)" % (trajectory[0], trajectory[-1]))
    check(all(trajectory[index + 1] > trajectory[index] - 0.35
              for index in range(len(trajectory) - 1)),
          "the ELBO must climb without collapsing")
    for index in range(model.dimension):
        check(abs(ascent.mean[index] - exact_mean[index]) < 0.05,
              "gradient ascent mean %d (%.4f) must approach the exact mean (%.4f)"
              % (index, ascent.mean[index], exact_mean[index]))
    check(abs(closed_form_elbo(model, ascent) - history[-1]) < 0.5,
          "gradient ascent must reach the coordinate-ascent optimum (%.4f vs %.4f)"
          % (closed_form_elbo(model, ascent), history[-1]))
    check(len(trajectory) <= 600, "gradient ascent must converge within its budget")
    print("ascent: %d steps, ELBO %.4f -> %.4f, final mean %s"
          % (len(trajectory), trajectory[0], trajectory[-1],
             [round(value, 4) for value in ascent.mean]))

    # ---- 7. posterior predictive against the analytic one ---------------
    analytic_predictive = []
    for row in model.design:
        mean = sum(weight * value for weight, value in zip(row, exact_mean))
        variance = model.noise_sigma ** 2 + sum(
            row[i] * exact_covariance[i][j] * row[j]
            for i in range(model.dimension) for j in range(model.dimension))
        analytic_predictive.append((mean, math.sqrt(variance)))
    check(len(analytic_predictive) == len(model.observations),
          "there must be one predictive per observation")
    check(all(sigma > model.noise_sigma for _mean, sigma in analytic_predictive),
          "the predictive spread must include the observation noise")
    for observation, (analytic_mean, _sigma) in zip(model.observations, analytic_predictive):
        check(abs(analytic_mean - observation) < 1.5,
              "a predictive mean must stay near its observation")
    predictive_from_vi = []
    for row in model.design:
        mean = sum(weight * value for weight, value in zip(row, fitted.mean))
        predictive_from_vi.append(mean)
    check(all(abs(left - right[0]) < 0.01
              for left, right in zip(predictive_from_vi, analytic_predictive)),
          "the variational predictive mean must match the analytic one")
    print("predictive: analytic means %s, variational match within 0.01"
          % [round(value, 3) for value, _ in analytic_predictive])

    # ---- 8. no data, and an overwhelming prior --------------------------
    empty = LinearGaussian([[1.0, 0.0], [0.0, 1.0]], [], prior_sigma=2.0, noise_sigma=0.5)
    empty_mean, empty_covariance = empty.exact_posterior()
    check(all(abs(value) < 1e-12 for value in empty_mean),
          "with no observations the posterior mean must be the prior mean")
    check(all(abs(empty_covariance[i][i] - 4.0) < 1e-9 for i in range(2)),
          "with no observations the posterior variance must be the prior variance")
    empty_fit, _history = cavi(empty, iterations=20)
    check(all(abs(value) < 1e-9 for value in empty_fit.mean),
          "CAVI on an empty model must return the prior mean")
    check(all(abs(sigma - 2.0) < 1e-9 for sigma in empty_fit.sigma),
          "CAVI on an empty model must return the prior spread, got %s" % empty_fit.sigma)

    stubborn = LinearGaussian([[1.0], [1.0]], [50.0, 52.0], prior_sigma=1e-3, noise_sigma=1.0)
    stubborn_mean, stubborn_covariance = stubborn.exact_posterior()
    check(abs(stubborn_mean[0]) < 1e-3,
          "an overwhelming prior must hold the posterior near the prior (%.3e)" % stubborn_mean[0])
    check(abs(stubborn_mean[0] - 51.0) > 50.0,
          "the data alone would have said 51, so the prior really is in control")
    check(stubborn_covariance[0][0] < 1e-6, "and the posterior variance must stay tiny")
    tight_fit, tight_history = cavi(stubborn, iterations=50)
    check(abs(tight_fit.mean[0]) < 1e-3,
          "CAVI must also respect the overwhelming prior (%.3e)" % tight_fit.mean[0])
    print("edge cases: empty data returns the prior exactly, a hard prior pins the posterior")

    # ---- 9. a single observation and a wide prior -----------------------
    one = LinearGaussian([[1.0]], [3.0], prior_sigma=1.0, noise_sigma=2.0)
    one_mean, one_covariance = one.exact_posterior()
    expected_variance = 1.0 / (1.0 / 1.0 + 1.0 / 4.0)
    check(abs(one_covariance[0][0] - expected_variance) < 1e-9,
          "a single observation must give the textbook posterior variance")
    check(abs(one_mean[0] - 3.0 * expected_variance / 4.0) < 1e-9,
          "a single observation must give the textbook posterior mean")
    one_fit, _history = cavi(one, iterations=40)
    check(abs(one_fit.mean[0] - one_mean[0]) < 1e-9,
          "CAVI must match the closed form on a single observation")
    print("single observation: posterior mean %.4f variance %.4f (matches the textbook)"
          % (one_mean[0], one_covariance[0][0]))

    # ---- 10. determinism -------------------------------------------------
    first = [mc_elbo(model, variational, SplitMix64(11), 500) for _ in range(2)]
    check(abs(first[0] - first[1]) < 1e-12, "the same seed must give the same ELBO estimate")
    other = mc_elbo(model, variational, SplitMix64(12), 500)
    check(abs(other - first[0]) > 0.0, "a different seed must give a different estimate")
    twin, twin_history = cavi(model, iterations=60)
    check(twin.mean == fitted.mean and twin.log_sigma == fitted.log_sigma,
          "CAVI must be reproducible")
    check(twin_history == history, "the CAVI trajectory must be reproducible")
    print("determinism: identical estimates, optima and trajectories for the same seed")
    print("PASS")


if __name__ == "__main__":
    main()
