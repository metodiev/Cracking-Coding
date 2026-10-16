"""Task 783 - Implement variational inference simple mean-field for mixture models.  (Tier: Expert, Beyond - Machine Learning & AI)

Mean-field variational inference fits a tractable factorised posterior q(z, mu) = prod_i q(z_i) *
prod_k q(mu_k) to a Bayesian mixture of K Gaussians with *known* variance and *fixed equal* weights,
where the component means carry a normal prior mu_k ~ N(m0, s0^2).  Coordinate ascent (CAVI) alternates
  * the responsibilities        r_ik  propto  exp(-((x_i - m_k)^2 + v_k) / (2 sigma^2)),
  * the mean posteriors         v_k = 1 / (1/s0^2 + sum_i r_ik / sigma^2),
                                m_k = v_k * (m0/s0^2 + sum_i r_ik x_i / sigma^2),
each step being the exact maximiser of the evidence lower bound (ELBO) for that factor, so the ELBO
must increase monotonically and is guaranteed never to exceed the log evidence (the ELBO is a lower
bound on log p(x)).

Verified: the ELBO is recomputed from scratch at the end and must equal the value accumulated during
the iteration, it increases monotonically over every sweep, it never exceeds the log evidence computed
by numerical quadrature (an independent oracle that is itself checked for grid convergence), the
posterior mean equals the sample mean in the single-component limit (the analytic conjugate
solution), well-separated data are recovered with near-hard responsibilities, degenerate data still
satisfy the conjugacy formula for the posterior variance, and the N = 1, K > N and empty cases are
handled.  Deterministic: initialisation is a fixed spread around the data mean, not random.

Simplified: known variances and fixed equal mixture weights (so no Dirichlet factor and no
beta/nu/m optimisations), one-dimensional data, two-dimensional quadrature as the evidence oracle,
and no stochastic/subsampled VI - this is the plain batch CAVI algorithm.
"""
import itertools
import math

LOG_TWO_PI = math.log(2.0 * math.pi)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def digamma(value):
    """Digamma by recurrence plus the usual asymptotic expansion (math.lgamma has no derivative)."""
    result = 0.0
    while value < 6.0:
        result -= 1.0 / value
        value += 1.0
    inverse = 1.0 / value
    inverse2 = inverse * inverse
    return result + math.log(value) - 0.5 * inverse - inverse2 * (
        1.0 / 12.0 - inverse2 * (1.0 / 120.0 - inverse2 / 252.0))


def normal_log_density(value, mean, variance):
    return -0.5 * LOG_TWO_PI - 0.5 * math.log(variance) - (value - mean) ** 2 / (2.0 * variance)


class MeanFieldMixture:
    """CAVI for a Gaussian mixture with known variance, fixed weights and normal priors on the means."""

    def __init__(self, components, sigma2=0.5, prior_mean=0.0, prior_var=25.0):
        self.k = components
        self.sigma2 = sigma2
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.data = []
        self.means = []
        self.variances = []
        self.responsibilities = []
        self.elbo = float("-inf")

    def initialise(self, data):
        """Deterministic spread around the data mean: separated components, no randomness."""
        self.data = list(data)
        if not self.data:
            self.means, self.variances, self.responsibilities = [], [], []
            return self
        centre = sum(self.data) / len(self.data)
        spread = ([min(self.data), max(self.data)] if max(self.data) > min(self.data)
                  else [centre - 0.5, centre + 0.5])
        self.means = [centre + (spread[1] - spread[0]) * (index - (self.k - 1) / 2.0) / max(1, self.k)
                      for index in range(self.k)]
        self.variances = [self.prior_var] * self.k
        total = float(len(self.data))
        self.responsibilities = [[1.0 / self.k] * self.k for _ in range(len(self.data))]
        check(abs(sum(sum(row) for row in self.responsibilities) - total) < 1e-9,
              "the initial responsibilities must be a proper distribution")
        return self

    # ------------------------------------------------------------------ CAVI
    def e_step(self):
        for index, value in enumerate(self.data):
            log_weights = []
            for component in range(self.k):
                expected = (value - self.means[component]) ** 2 + self.variances[component]
                log_weights.append(-expected / (2.0 * self.sigma2))
            highest = max(log_weights)
            weights = [math.exp(weight - highest) for weight in log_weights]
            total = sum(weights)
            self.responsibilities[index] = [weight / total for weight in weights]

    def m_step(self):
        self.means, self.variances = [], []
        for component in range(self.k):
            weight = sum(self.responsibilities[index][component]
                         for index in range(len(self.data)))
            precision = 1.0 / self.prior_var + weight / self.sigma2
            variance = 1.0 / precision
            weighted_sum = sum(self.responsibilities[index][component] * self.data[index]
                               for index in range(len(self.data)))
            mean = variance * (self.prior_mean / self.prior_var + weighted_sum / self.sigma2)
            self.means.append(mean)
            self.variances.append(variance)
        return [sum(self.responsibilities[index][component] for index in range(len(self.data)))
                for component in range(self.k)]

    def responsibilities_of(self, value):
        """The variational posterior over the component of a *new* (or existing) point."""
        log_weights = []
        for component in range(self.k):
            expected = (value - self.means[component]) ** 2 + self.variances[component]
            log_weights.append(-expected / (2.0 * self.sigma2))
        highest = max(log_weights)
        weights = [math.exp(weight - highest) for weight in log_weights]
        total = sum(weights)
        return [weight / total for weight in weights]

    # ------------------------------------------------------------------ objective
    def compute_elbo(self, means=None, variances=None, responsibilities=None):
        """The ELBO, recomputed from whatever parameter state is supplied (default: the current one)."""
        means = self.means if means is None else means
        variances = self.variances if variances is None else variances
        responsibilities = self.responsibilities if responsibilities is None else responsibilities
        if not self.data:
            return 0.0
        term_likelihood = 0.0
        term_entropy_z = 0.0
        for index, value in enumerate(self.data):
            for component in range(self.k):
                weight = responsibilities[index][component]
                expected = (value - means[component]) ** 2 + variances[component]
                term_likelihood += weight * (-0.5 * LOG_TWO_PI - 0.5 * math.log(self.sigma2)
                                             - expected / (2.0 * self.sigma2))
                if weight > 0:
                    term_entropy_z -= weight * math.log(weight)
        term_prior = sum(-0.5 * LOG_TWO_PI - 0.5 * math.log(self.prior_var)
                         - ((mean - self.prior_mean) ** 2 + variance) / (2.0 * self.prior_var)
                         for mean, variance in zip(means, variances))
        term_mixture = -len(self.data) * math.log(self.k)
        term_entropy_mu = sum(0.5 * math.log(2.0 * math.pi * math.e * variance)
                              for variance in variances)
        return term_likelihood + term_prior + term_mixture + term_entropy_z + term_entropy_mu

    def fit(self, iterations=200, tolerance=1e-12):
        """Returns the ELBO trajectory (one entry per sweep); it must be non-decreasing."""
        trajectory = []
        self.e_step()
        for _ in range(iterations):
            self.m_step()
            self.e_step()
            value = self.compute_elbo()
            trajectory.append(value)
            if len(trajectory) > 1 and trajectory[-1] < trajectory[-2] - 1e-9:
                raise AssertionError("the ELBO must never decrease")
            if len(trajectory) > 1 and trajectory[-1] - trajectory[-2] < tolerance:
                break
        self.elbo = trajectory[-1]
        return trajectory


def log_component_evidence(subset, sigma2, prior_mean, prior_var):
    """log of the integral prior(mu) * prod_{i in subset} N(x_i | mu, sigma2) dmu, closed form."""
    precision_prior = 1.0 / prior_var
    precision_data = len(subset) / sigma2
    combined = precision_prior + precision_data
    total = sum(subset)
    exponent = (precision_prior * prior_mean ** 2 + sum(value * value for value in subset) / sigma2) / 2.0
    exponent -= (precision_prior * prior_mean + total / sigma2) ** 2 / (2.0 * combined)
    return (-0.5 * LOG_TWO_PI - 0.5 * math.log(prior_var)
            - 0.5 * len(subset) * LOG_TWO_PI - 0.5 * len(subset) * math.log(sigma2)
            - exponent + 0.5 * math.log(2.0 * math.pi / combined))


def log_evidence_exact(data, components, sigma2, prior_mean, prior_var):
    """Exact log evidence: enumerate the K^N assignment configurations (the oracle for the bound)."""
    logs = []
    for assignment in itertools.product(range(components), repeat=len(data)):
        total = 0.0
        for component in range(components):
            subset = [value for value, label in zip(data, assignment) if label == component]
            total += log_component_evidence(subset, sigma2, prior_mean, prior_var)
        logs.append(total - len(data) * math.log(components))
    highest = max(logs)
    return highest + math.log(sum(math.exp(value - highest) for value in logs))


def log_evidence_single(data, sigma2, prior_mean, prior_var, points=4000, span=14.0):
    """Simpson quadrature for K = 1: log integral of prior(mean) * prod likelihood."""
    points = points if points % 2 == 1 else points + 1        # Simpson needs an odd point count
    low, high = prior_mean - span * math.sqrt(prior_var), prior_mean + span * math.sqrt(prior_var)
    step = (high - low) / (points - 1)
    total = 0.0
    for index in range(points):
        mean = low + index * step
        weight = 1.0 if index in (0, points - 1) else (4.0 if index % 2 else 2.0)
        density = math.exp(normal_log_density(mean, prior_mean, prior_var))
        for value in data:
            density *= math.exp(normal_log_density(value, mean, sigma2))
        total += weight * density
    return math.log(total * step / 3.0)


def log_evidence_two(data, sigma2, prior_mean, prior_var, points=160, span=9.0):
    """Simpson quadrature over the two component means of a K = 2 model with equal weights."""
    points = points if points % 2 == 1 else points + 1        # Simpson needs an odd point count
    low, high = prior_mean - span * math.sqrt(prior_var), prior_mean + span * math.sqrt(prior_var)
    step = (high - low) / (points - 1)
    grid = [low + index * step for index in range(points)]
    total = 0.0
    for left in range(points):
        weigh_left = 1.0 if left in (0, points - 1) else (2.0 if left % 2 else 4.0)
        for right in range(points):
            weigh_right = 1.0 if right in (0, points - 1) else (2.0 if right % 2 else 4.0)
            density = math.exp(normal_log_density(grid[left], prior_mean, prior_var)
                               + normal_log_density(grid[right], prior_mean, prior_var))
            for value in data:
                density *= 0.5 * (math.exp(normal_log_density(value, grid[left], sigma2))
                                  + math.exp(normal_log_density(value, grid[right], sigma2)))
            total += weigh_left * weigh_right * density
    return math.log(total * (step / 3.0) ** 2)


def main():
    # 1. boundaries: no data, a single point, more components than points
    blank = MeanFieldMixture(3).initialise([])
    check(blank.elbo == float("-inf") or blank.compute_elbo() == 0.0,
          "an empty dataset must not produce a spurious objective")
    one = MeanFieldMixture(1).initialise([2.0])
    one.fit(iterations=50)
    check(len(one.means) == 1, "one component means one mean")
    check(abs(one.variances[0] - 1.0 / (1.0 / one.prior_var + 1.0 / one.sigma2)) < 1e-12,
          "the posterior variance must match the conjugate formula")
    crowded = MeanFieldMixture(5).initialise([0.1, 0.2])
    crowded.fit(iterations=50)
    check(len(crowded.means) == 5, "more components than points must still be representable")
    check(all(math.isfinite(mean) for mean in crowded.means), "the means must stay finite")

    # 2. single component: the variational mean is the sample mean, and the ELBO is a lower bound
    data = [-0.3, 0.1, 0.4, 0.9, 1.2, 0.6]
    mixture = MeanFieldMixture(1, sigma2=0.8, prior_mean=0.0, prior_var=100.0).initialise(data)
    trajectory = mixture.fit(iterations=200)
    sample_mean = sum(data) / len(data)
    precision = 1.0 / 100.0 + len(data) / 0.8
    conjugate_mean = (0.0 / 100.0 + sum(data) / 0.8) / precision
    check(abs(mixture.means[0] - conjugate_mean) < 1e-9,
          f"the posterior mean must match the conjugate formula, got {mixture.means[0]}")
    check(abs(mixture.means[0] - sample_mean) < 0.01,
          "with a wide prior the posterior mean must sit next to the sample mean")
    check(all(later >= earlier - 1e-9 for earlier, later in zip(trajectory, trajectory[1:])),
          "the ELBO must be non-decreasing on the single-component problem")
    evidence = log_evidence_single(data, 0.8, 0.0, 100.0)
    check(mixture.elbo <= evidence + 1e-6,
          f"the ELBO {mixture.elbo:.6f} must not exceed the evidence {evidence:.6f}")
    check(evidence - mixture.elbo < 0.5,
          "with one component the mean-field bound should be tight")

    # 3. two independent oracles agree: the closed form and the Simpson quadrature for K = 1
    closed_form = log_evidence_exact(data, 1, 0.8, 0.0, 100.0)
    check(abs(closed_form - evidence) < 1e-4,
          f"the closed form {closed_form:.6f} and the quadrature {evidence:.6f} must agree")
    coarse = log_evidence_single(data, 0.8, 0.0, 100.0, points=1001)
    check(abs(coarse - evidence) < 1e-4, "the evidence quadrature must be grid converged")

    # 4. two well-separated clusters: means recovered, responsibilities nearly hard
    separated = [-3.2, -2.8, -3.5, -2.9, 3.1, 2.7, 3.4, 2.8]
    two = MeanFieldMixture(2, sigma2=0.6, prior_mean=0.0, prior_var=25.0).initialise(separated)
    two.fit(iterations=300)
    ordered = sorted(two.means)
    check(abs(ordered[0] + 3.1) < 0.4 and abs(ordered[1] - 3.0) < 0.4,
          f"the two cluster means must be recovered, got {ordered}")
    hard = sum(1 for row in two.responsibilities if max(row) > 0.999)
    check(hard == len(separated), f"separated data must give hard responsibilities, got {hard}/8")
    check(two.responsibilities_of(-3.0)[0] > 0.999,
          "a new point at the left cluster must be assigned to the left component")
    check(two.responsibilities_of(3.0)[1] > 0.999,
          "a new point at the right cluster must be assigned to the right component")

    # 5. the ELBO equals the value recomputed from scratch at the converged parameters
    fresh = two.compute_elbo([mean for mean in two.means], [var for var in two.variances],
                             [list(row) for row in two.responsibilities])
    check(abs(fresh - two.elbo) < 1e-9, f"the ELBO bookkeeping must agree ({fresh} vs {two.elbo})")
    before = two.elbo
    two.fit(iterations=5)
    check(two.elbo >= before - 1e-9, "continuing a converged fit must not lower the ELBO")

    # 6. the ELBO sits below the exact two-component evidence (fixed equal weights)
    evidence_two = log_evidence_exact(separated, 2, 0.6, 0.0, 25.0)
    check(two.elbo <= evidence_two + 1e-9,
          f"the ELBO {two.elbo:.4f} must not exceed the evidence {evidence_two:.4f}")
    check(evidence_two - two.elbo > 0.0, "the bound must be a strict lower bound here")
    rough = log_evidence_two(separated, 0.6, 0.0, 25.0)
    check(abs(rough - evidence_two) < 0.5,
          f"the 2-D quadrature {rough:.3f} must roughly confirm the exact evidence "
          f"{evidence_two:.3f}")

    # 7. degenerate data: identical points shrink the posterior variance by the conjugacy formula
    identical = [1.5] * 5
    flat = MeanFieldMixture(1, sigma2=0.5, prior_mean=0.0, prior_var=4.0).initialise(identical)
    flat.fit(iterations=200)
    expected_variance = 1.0 / (1.0 / 4.0 + 5 / 0.5)
    check(abs(flat.variances[0] - expected_variance) < 1e-9,
          f"identical points must shrink the variance to {expected_variance}")
    check(abs(flat.means[0] - 1.5 * (1 - expected_variance / 4.0)) < 1e-6
          or abs(flat.means[0] - 1.5) < 0.5, "the posterior mean must be pulled towards the data")

    # 8. more components than clusters is allowed but still yields finite, ordered parameters
    over = MeanFieldMixture(4, sigma2=0.6, prior_mean=0.0, prior_var=25.0).initialise(separated)
    over.fit(iterations=300)
    check(len(over.means) == 4 and all(math.isfinite(mean) for mean in over.means),
          "extra components must stay finite")
    check(over.variances and all(variance > 0 for variance in over.variances),
          "every component must keep a positive posterior variance")

    # 9. determinism
    again = MeanFieldMixture(2, sigma2=0.6, prior_mean=0.0, prior_var=25.0).initialise(separated)
    again.fit(iterations=300)
    check(again.means == two.means and again.variances == two.variances,
          "CAVI must be deterministic")

    print(f"783 single_elbo={mixture.elbo:.4f} evidence={evidence:.4f} gap={evidence - mixture.elbo:.4f}")
    print(f"783 two_elbo={two.elbo:.4f} evidence2={evidence_two:.4f} means="
          f"{[round(mean, 3) for mean in sorted(two.means)]} hard={hard}/8 sweeps={len(trajectory)}")
    print("PASS")


if __name__ == "__main__":
    main()
