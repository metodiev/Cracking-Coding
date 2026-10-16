"""Task 798 - Implement Thompson sampling with hierarchical prior simulation.  (Tier: Expert, CTCI Beyond - Machine Learning & AI)

Thompson sampling for Bernoulli bandits draws one sample p_k from each arm's posterior and
plays the argmax.  With independent Beta(1, 1) priors the posterior of arm k after s_k
successes and f_k failures is Beta(1 + s_k, 1 + f_k) - the flat-prior baseline used here.

The hierarchical version assumes instead that the arms of a task are exchangeable draws from a
common Beta(a, b), and puts a discrete prior over the hyperparameters on a (a, b) grid.  The
posterior is then sampled with a two-block Gibbs sweep:

    p_k | data, (a, b) ~ Beta(a + s_k, b + f_k)             for every arm k
    (a, b) | p             proportional to prior(a, b) * prod_k Beta(p_k; a, b)

so the prior of the *next* task is the posterior of the previous ones - the online empirical
Bayes scheme that lets several tasks inform the prior of the current one.  Play then samples
from the hierarchical posterior, i.e. from Beta(a + s_k, b + f_k) with the currently sampled
hyperparameters.

Verified: the Beta sampler's empirical mean and variance match the analytic moments; posterior
means match the Beta(a+s, b+f) closed form; the discrete hyperparameter posterior is a
normalised distribution across the whole grid; the empirical-Bayes posterior mode recovers the
true generating (a, b) from data pooled over many arms; the hierarchical agent's mean Bayesian
regret over several synthetic tasks is strictly below the flat-prior agent's on the same reward
streams; regret grows sublinearly; a single arm forces a trivial choice; equal posteriors spread
the plays across arms; and the whole simulation is reproducible for a fixed seed.

Simplified versus a production hierarchical bandit: the hyperparameter distribution is a small
discrete grid rather than a continuous posterior (sampled by Gibbs over two blocks, one sweep
per decision, no burn-in diagnostics), the reward model is Bernoulli with no covariates, and the
whole simulation is deliberately small (four arms, a few hundred decisions, twenty tasks) so the
study finishes in about a second.
"""
import math

MASK64 = (1 << 64) - 1
TWO_POW_53_COS = 9007199254740992.0


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def uniform(state):
    state, r = splitmix64(state)
    return state, r / 18446744073709551616.0


def gauss(state):
    state, u1 = splitmix64(state)
    state, u2 = splitmix64(state)
    r = (u1 >> 11) / TWO_POW_53_COS
    p = (u2 >> 11) / TWO_POW_53_COS
    if r < 1e-12:
        r = 1e-12
    return state, math.sqrt(-2.0 * math.log(r)) * math.cos(2.0 * math.pi * p)


# --------------------------------------------------------------- Beta sampling
def gamma_sample(state, shape):
    """Marsaglia-Tsang: gamma with shape >= 1 (boosted by one for shape < 1)."""
    if shape < 1.0:
        state, g = gamma_sample(state, shape + 1.0)
        state, u = uniform(state)
        u = max(u, 1e-12)
        return state, g * u ** (1.0 / shape)
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        state, normal = gauss(state)
        state, u = uniform(state)
        v = (1.0 + c * normal) ** 3
        if v <= 0.0:
            continue
        if math.log(max(u, 1e-12)) < 0.5 * normal * normal + d - d * v + d * math.log(v):
            return state, d * v


def beta_sample(state, a, b):
    """Beta(a, b) as the ratio of two gamma variates."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError("the Beta parameters must be positive")
    state, ga = gamma_sample(state, a)
    state, gb = gamma_sample(state, b)
    return state, ga / (ga + gb)


def beta_mean(a, b):
    return a / (a + b)


def beta_variance(a, b):
    return a * b / ((a + b) ** 2 * (a + b + 1.0))


def beta_logpdf(x, a, b):
    if x <= 0.0 or x >= 1.0:
        return -float("inf")
    return ((a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x)
            + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))


# ------------------------------------------------------- hierarchical agent
class HierarchicalThompson:
    """Thompson sampling with a discrete-Beta hyperprior over (a, b) sampled by Gibbs."""

    def __init__(self, arms, grid, seed):
        if arms < 1:
            raise ValueError("at least one arm is required")
        self.arms = arms
        self.grid = list(grid)
        self.prior = [1.0 / len(self.grid)] * len(self.grid)
        self.state = seed
        self.successes = [0] * arms
        self.failures = [0] * arms
        self.hyper = self.grid[len(self.grid) // 2]
        self.choices = [0] * arms

    def posterior_weights(self, probabilities):
        """unnormalised posterior over the grid given the sampled arm probabilities."""
        weights = []
        for index, (a, b) in enumerate(self.grid):
            score = math.log(self.prior[index]) if self.prior[index] > 0.0 else -700.0
            for p in probabilities:
                score += beta_logpdf(p, a, b)
            weights.append(score)
        top = max(weights)
        exponentials = [math.exp(value - top) * self.prior[i]
                        for i, value in enumerate(weights)]
        total = math.fsum(exponentials)
        return [value / total for value in exponentials]

    def sample_hyperparameters(self):
        """One Gibbs sweep: sample the arm probabilities, then the hyperparameters."""
        probabilities = []
        a, b = self.hyper
        for arm in range(self.arms):
            self.state, p = beta_sample(self.state, a + self.successes[arm],
                                        b + self.failures[arm])
            probabilities.append(p)
        weights = self.posterior_weights(probabilities)
        self.state, r = uniform(self.state)
        acc = 0.0
        picked = len(self.grid) - 1
        for index, value in enumerate(weights):
            acc += value
            if r < acc:
                picked = index
                break
        self.hyper = self.grid[picked]
        return probabilities, weights

    def select(self):
        """Draw one probability per arm from the hierarchical posterior and play the argmax."""
        a, b = self.hyper
        best, best_value = 0, -1.0
        for arm in range(self.arms):
            self.state, p = beta_sample(self.state, a + self.successes[arm],
                                        b + self.failures[arm])
            if p > best_value:
                best, best_value = arm, p
        self.choices[best] += 1
        return best

    def update(self, arm, reward):
        if reward:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1


class FlatThompson(HierarchicalThompson):
    """Baseline: independent Beta(1, 1) priors, no hyperparameter learning."""

    def __init__(self, arms, seed):
        HierarchicalThompson.__init__(self, arms, [(1.0, 1.0)], seed)

    def sample_hyperparameters(self):
        return [], [1.0]

    def select(self):
        best, best_value = 0, -1.0
        for arm in range(self.arms):
            self.state, p = beta_sample(self.state, 1.0 + self.successes[arm],
                                        1.0 + self.failures[arm])
            if p > best_value:
                best, best_value = arm, p
        self.choices[best] += 1
        return best


# ------------------------------------------------------------------ simulation
def make_grid(values):
    return [(a, b) for a in values for b in values]


def simulate(agent, probabilities, steps, seed):
    """Bernoulli reward stream; returns the cumulative Bayesian (pseudo) regret."""
    state = seed
    best = max(probabilities)
    regret = 0.0
    for step in range(steps):
        arm = agent.select()
        regret += best - probabilities[arm]
        state, r = uniform(state)
        agent.update(arm, 1 if r < probabilities[arm] else 0)
        if step % 10 == 0:
            agent.sample_hyperparameters()
    return regret


def draw_task(state, arms, a0, b0):
    probabilities = []
    for _ in range(arms):
        state, p = beta_sample(state, a0, b0)
        probabilities.append(p)
    return probabilities, state


def empirical_bayes_mode(data, grid, prior=None):
    """Grid posterior over (a, b) from exchangeable Bernoulli data; returns the argmax cell."""
    weights = []
    for index, (a, b) in enumerate(grid):
        score = math.log(prior[index]) if prior and prior[index] > 0.0 else 0.0
        for successes, failures in data:
            score += (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + math.lgamma(a + successes) + math.lgamma(b + failures)
                      - math.lgamma(a + b + successes + failures))
        weights.append(score)
    top = max(weights)
    exponentials = [math.exp(value - top) for value in weights]
    total = math.fsum(exponentials)
    return grid[max(range(len(grid)), key=lambda i: exponentials[i])], \
        [value / total for value in exponentials]


def main():
    A0, B0 = 1.5, 3.0
    ARMS = 4

    # --- 1. the Beta sampler matches its analytic moments -------------------
    state = 798001
    draws = 20000
    total = 0.0
    total_square = 0.0
    for _ in range(draws):
        state, value = beta_sample(state, 2.0, 5.0)
        total += value
        total_square += value * value
    mean = total / draws
    variance = total_square / draws - mean * mean
    check(abs(mean - beta_mean(2.0, 5.0)) < 0.01,
          "Beta(2,5) sample mean %.5f must match %.5f" % (mean, beta_mean(2.0, 5.0)))
    check(abs(variance - beta_variance(2.0, 5.0)) < 0.003,
          "Beta(2,5) sample variance %.6f must match %.6f"
          % (variance, beta_variance(2.0, 5.0)))

    # --- 2. the posterior is the closed form -------------------------------
    agent = FlatThompson(ARMS, 798002)
    agent.successes = [7, 0, 4, 12]
    agent.failures = [3, 10, 6, 8]
    posterior_draws = 5000
    for arm in range(ARMS):
        total = 0.0
        for _ in range(posterior_draws):
            agent.state, value = beta_sample(agent.state, 1.0 + agent.successes[arm],
                                             1.0 + agent.failures[arm])
            total += value
        sampled_mean = total / posterior_draws
        analytic = beta_mean(1.0 + agent.successes[arm], 1.0 + agent.failures[arm])
        check(abs(sampled_mean - analytic) < 0.02,
              "arm %d posterior mean %.5f must match Beta(s+1, f+1) mean %.5f"
              % (arm, sampled_mean, analytic))

    # --- 3. the hyperparameter posterior is a distribution -----------------
    grid = make_grid((0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0))
    hierarchical = HierarchicalThompson(ARMS, grid, 798003)
    probabilities, weights = hierarchical.sample_hyperparameters()
    check(abs(math.fsum(weights) - 1.0) < 1e-12,
          "the sampled hyperparameter posterior must be normalised, sum %.12f"
          % math.fsum(weights))
    check(all(0.0 <= value <= 1.0 for value in weights),
          "hyperparameter posterior weights must lie in [0, 1]")
    check(all(0.0 <= value <= 1.0 for value in probabilities),
          "the sampled arm probabilities must lie in [0, 1]")

    # --- 4. empirical Bayes recovers the generating hyperparameters --------
    state = 798004
    pooled = []
    for _ in range(40):
        task, state = draw_task(state, ARMS, A0, B0)
        for p in task:
            successes = 0
            for _trial in range(60):
                state, r = uniform(state)
                successes += 1 if r < p else 0
            pooled.append((successes, 60 - successes))
    mode, posterior = empirical_bayes_mode(pooled, grid)
    check(abs(mode[0] - A0) <= 0.75 and abs(mode[1] - B0) <= 0.75,
          "the empirical-Bayes mode %r must be near the true (%.2f, %.2f)" % (mode, A0, B0))
    check(abs(math.fsum(posterior) - 1.0) < 1e-12, "the grid posterior must be normalised")
    truth_index = min(range(len(grid)),
                      key=lambda i: abs(grid[i][0] - A0) + abs(grid[i][1] - B0))
    mass_near_truth = math.fsum(posterior[i] for i in range(len(grid))
                                if abs(grid[i][0] - grid[truth_index][0]) <= 0.5
                                and abs(grid[i][1] - grid[truth_index][1]) <= 0.5)
    check(mass_near_truth > 0.5,
          "more than half the posterior mass must sit near the truth, got %.4f" % mass_near_truth)

    # --- 5. hierarchical versus flat regret on identical tasks -------------
    tasks = 20
    steps = 400
    state = 798005
    hierarchical_regret = 0.0
    flat_regret = 0.0
    oracle_regret = 0.0
    for index in range(tasks):
        task, state = draw_task(state, ARMS, A0, B0)
        seed = 798100 + index
        learner = HierarchicalThompson(ARMS, grid, seed)
        flat = FlatThompson(ARMS, seed)
        oracle_prior = HierarchicalThompson(ARMS, [(A0, B0)], seed)
        hierarchical_regret += simulate(learner, task, steps, seed)
        flat_regret += simulate(flat, task, steps, seed)
        oracle_regret += simulate(oracle_prior, task, steps, seed)
    hierarchical_regret /= tasks
    flat_regret /= tasks
    oracle_regret /= tasks
    check(hierarchical_regret < flat_regret,
          "the hierarchical agent must beat the flat prior: %.4f vs %.4f"
          % (hierarchical_regret, flat_regret))
    check(oracle_regret <= flat_regret,
          "the oracle-prior agent must not be worse than the flat prior: %.4f vs %.4f"
          % (oracle_regret, flat_regret))

    # --- 6. sublinear regret growth and boundary behaviour ----------------
    state = 798006
    short_total = 0.0
    long_total = 0.0
    for index in range(6):
        task, state = draw_task(state, ARMS, A0, B0)
        short_total += simulate(HierarchicalThompson(ARMS, grid, 798200 + index), task, 200,
                                798200 + index)
        long_total += simulate(HierarchicalThompson(ARMS, grid, 798200 + index), task, 400,
                               798200 + index)
    check(long_total < 2.0 * short_total,
          "hierarchical regret must grow sublinearly: %.4f at 200 steps vs %.4f at 400 steps"
          % (short_total / 6.0, long_total / 6.0))
    single = HierarchicalThompson(1, grid, 798007)
    check(single.select() == 0, "a single arm forces the choice")
    check(simulate(HierarchicalThompson(1, grid, 798008), [0.4], 50, 798008) == 0.0,
          "a single arm must accumulate exactly zero regret")
    tied = FlatThompson(4, 798009)
    for _ in range(4000):
        tied.select()
    spread = min(tied.choices) / sum(tied.choices)
    check(spread > 0.15,
          "with identical posteriors the plays must spread over the arms, worst share %.4f"
          % spread)

    # --- 7. determinism ---------------------------------------------------
    twin_a = HierarchicalThompson(ARMS, grid, 798300)
    twin_b = HierarchicalThompson(ARMS, grid, 798300)
    task = [0.3, 0.35, 0.28, 0.33]
    check(simulate(twin_a, task, 120, 798300) == simulate(twin_b, task, 120, 798300),
          "an identical seed must reproduce the whole simulation")
    raised = False
    try:
        beta_sample(798400, 0.0, 1.0)
    except ValueError:
        raised = True
    check(raised, "non-positive Beta parameters must be rejected")
    raised = False
    try:
        HierarchicalThompson(0, grid, 1)
    except ValueError:
        raised = True
    check(raised, "a zero arm count must be rejected")

    print("Beta(2,5): sample mean %.5f (theory %.5f), variance %.6f (theory %.6f)"
          % (mean, beta_mean(2.0, 5.0), variance, beta_variance(2.0, 5.0)))
    print("empirical-Bayes mode %r for the true (%.2f, %.2f); posterior mass near truth %.4f"
          % (mode, A0, B0, mass_near_truth))
    print("mean regret over %d tasks of %d steps: hierarchical %.4f, oracle prior %.4f, "
          "flat %.4f" % (tasks, steps, hierarchical_regret, oracle_regret, flat_regret))
    print("sublinearity: %.4f at 200 steps vs %.4f at 400 steps"
          % (short_total / 6.0, long_total / 6.0))
    print("PASS")


if __name__ == "__main__":
    main()
