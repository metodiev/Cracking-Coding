"""Task 495 - Implement UCB (upper confidence bound) algorithm simulation. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

UCB1 on a planted three-armed bandit: each arm is pulled once to initialise its averages
(arms in index order), then every trial pulls argmax(mu_hat[i] + sqrt(2 ln t / n_i)), ties to
the lowest index, t counting trials so far. Rewards are Bernoulli draws from the fixed LCG
(seed = (seed*75 + 74) % 65537, reward when (seed + 0.5)/65537 < p_arm), so the whole run is
deterministic. The pseudo-regret uses the known true means, R(T) = sum over trials of
(max(mu) - mu[a_t]) with mu = [0.2, 0.5, 0.8] as planted. Documented expectations over 2000
fixed trials: the best arm is pulled at least 80% of the time, pulls are strictly ordered
best > middle > worst, total regret stays under 100, and the mean per-trial regret of the
second half is strictly below that of the first half (the regret rate decreases as UCB1
learns). A uniform-random baseline using the same reward stream is used to show UCB1's
advantage.
"""
import math

ARMS = (0.2, 0.5, 0.8)
TRIALS = 2000
SEED = 99


class UCB1:
    def __init__(self, arms, seed):
        self.arms = tuple(arms)
        self.seed = seed
        self.counts = [0] * len(self.arms)
        self.sums = [0.0] * len(self.arms)
        self.pulls = [0] * len(self.arms)

    def reward(self, p):
        self.seed = (self.seed * 75 + 74) % 65537
        return 1 if (self.seed + 0.5) / 65537.0 < p else 0

    def index(self, arm, t):
        if self.counts[arm] == 0:
            return float("inf")                # untried arms are taken first
        mean = self.sums[arm] / self.counts[arm]
        return mean + math.sqrt(2.0 * math.log(t) / self.counts[arm])

    def run(self, trials):
        best = max(self.arms)
        regret = 0.0
        regret_trace = []
        for t in range(1, trials + 1):
            choice = 0
            for i in range(1, len(self.arms)):
                if self.index(i, t) > self.index(choice, t):
                    choice = i
            r = self.reward(self.arms[choice])
            self.counts[choice] += 1
            self.sums[choice] += r
            self.pulls[choice] += 1
            regret += best - self.arms[choice]
            regret_trace.append(regret)
        return regret, regret_trace


class RandomBaseline:
    """Uniform-random arm choice over the same LCG reward stream (control experiment)."""

    def __init__(self, arms, seed):
        self.arms = tuple(arms)
        self.seed = seed

    def run(self, trials):
        best = max(self.arms)
        regret = 0.0
        for _ in range(trials):
            self.seed = (self.seed * 75 + 74) % 65537
            choice = self.seed % len(self.arms)
            self.seed = (self.seed * 75 + 74) % 65537
            reward = 1 if (self.seed + 0.5) / 65537.0 < self.arms[choice] else 0
            del reward
            regret += best - self.arms[choice]
        return regret


def main():
    ucb = UCB1(ARMS, SEED)
    regret, trace = ucb.run(TRIALS)

    assert sum(ucb.pulls) == TRIALS
    assert ucb.pulls == ucb.counts

    best_frequency = ucb.pulls[2] / TRIALS
    assert best_frequency >= 0.80, "documented: best arm picked >= 80% of the time"
    assert ucb.pulls[2] > ucb.pulls[1] > ucb.pulls[0], "strict pull ordering"
    assert regret <= 100.0, "documented: total pseudo-regret bound"

    # Regret trace: non-decreasing by definition, and the per-trial regret rate of the second
    # half must be strictly lower than that of the first half.
    for a, b in zip(trace, trace[1:]):
        assert b >= a, "regret is a cumulative sum of non-negative gaps"
    half = TRIALS // 2
    first_rate = trace[half - 1] / half
    second_rate = (trace[-1] - trace[half - 1]) / (TRIALS - half)
    assert second_rate < first_rate, "documented: regret rate decreases"

    # Control: the uniform baseline pays far more regret on the same stream.
    baseline = RandomBaseline(ARMS, SEED)
    baseline_regret = baseline.run(TRIALS)
    assert baseline_regret > 2.0 * regret, "UCB1 must beat uniform sampling"

    # Determinism: same seed reproduces the run exactly.
    repeat = UCB1(ARMS, SEED)
    repeat_regret, _ = repeat.run(TRIALS)
    assert repeat.pulls == ucb.pulls and repeat_regret == regret

    # Warm-up structure: until every arm has been tried, untried arms win with index +inf,
    # so exactly the first three pulls visit arms 0, 1, 2 once each, in index order.
    warm = UCB1(ARMS, SEED)
    warm.run(3)
    assert warm.counts == [1, 1, 1], "warm-up visits each arm exactly once"
    warm.run(7)
    assert min(warm.counts) >= 1, "no arm is skipped during warm-up"
    assert sum(warm.counts) == 10

    # Two-arm large-gap case: concentration is even stronger (documented 90% threshold).
    duet = UCB1((0.3, 0.7), SEED + 1)
    duet.run(1000)
    assert duet.pulls[1] >= 0.9 * 1000
    assert duet.pulls[1] > duet.pulls[0]


if __name__ == "__main__":
    main()
    print("PASS")
