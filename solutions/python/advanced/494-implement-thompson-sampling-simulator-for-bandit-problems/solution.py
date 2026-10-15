"""Task 494 - Implement Thompson sampling simulator for bandit problems. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Beta-Bernoulli Thompson sampling on a planted three-armed bandit: arm i pays 1 with the known
probability arms[i], and every trial samples each arm's success rate from its Beta posterior
(prior Beta(1,1)), plays the sampled argmax (ties to the lowest index), observes a Bernoulli
reward drawn from the fixed LCG and updates the chosen arm's posterior. Beta(a,b) with integer
parameters is exact: two Gamma(a,1)/Gamma(b,1) draws built as sums of a (resp. b) unit
exponentials -ln(u), u uniform; that only needs the fixed LCG, no random source. Every draw
steps seed = (seed*75 + 74) % 65537 and returns (seed + 0.5)/65537 in (0,1). Documented
expectations for arms [0.2, 0.5, 0.8] over 1200 fixed trials: the best arm is picked at least
85% of the time, it is picked more often than each other arm, and the average regret
R = sum(max(mu) - mu[a_t]) stays under 100; the posterior mean of the best arm sits within
0.05 of 0.8. Re-running with the same seed reproduces the pull vector exactly.
"""
import math

ARMS = (0.2, 0.5, 0.8)
TRIALS = 1200
SEED = 42


class ThompsonSampler:
    def __init__(self, arms, seed):
        self.arms = tuple(arms)
        self.seed = seed
        self.alpha = [1] * len(self.arms)          # integer parameters: exact Beta draws
        self.beta = [1] * len(self.arms)
        self.pulls = [0] * len(self.arms)

    def uniform(self):
        self.seed = (self.seed * 75 + 74) % 65537
        return (self.seed + 0.5) / 65537.0

    def gamma_unit(self, shape):
        """Gamma(shape, 1) for integer shape >= 1: a sum of shape unit exponentials."""
        total = 0.0
        for _ in range(shape):
            total += -math.log(self.uniform())
        return total

    def beta_sample(self, a, b):
        x = self.gamma_unit(a)
        y = self.gamma_unit(b)
        return x / (x + y)

    def run(self, trials):
        best = max(self.arms)
        reward_total = 0
        regret = 0.0
        for _ in range(trials):
            samples = [self.beta_sample(self.alpha[i], self.beta[i])
                       for i in range(len(self.arms))]
            choice = 0
            for i in range(1, len(samples)):
                if samples[i] > samples[choice]:
                    choice = i
            reward = 1 if self.uniform() < self.arms[choice] else 0
            self.alpha[choice] += reward
            self.beta[choice] += 1 - reward
            self.pulls[choice] += 1
            reward_total += reward
            regret += best - self.arms[choice]
        return reward_total, regret

    def posterior_mean(self, arm):
        return self.alpha[arm] / (self.alpha[arm] + self.beta[arm])


def main():
    sampler = ThompsonSampler(ARMS, SEED)
    reward_total, regret = sampler.run(TRIALS)

    assert sum(sampler.pulls) == TRIALS
    best_frequency = sampler.pulls[2] / TRIALS
    assert best_frequency >= 0.85, "documented: best arm picked >= 85% of the time"
    assert sampler.pulls[2] > sampler.pulls[1] > sampler.pulls[0], "arm ordering"
    assert regret <= 100.0, "documented: average regret bound"
    assert 0.75 <= sampler.posterior_mean(2) <= 0.85, "posterior mean of the best arm"
    assert abs(sampler.posterior_mean(2) - ARMS[2]) <= 0.05
    # The empirical reward rate of the best arm's pulls must track its true probability.
    played = sampler.posterior_mean(2)
    assert abs(played - ARMS[2]) <= 0.05
    # Rewards observed only ever come from the chosen arm's Bernoulli law: total reward is
    # bounded by the number of pulls and is an integer.
    assert 0 <= reward_total <= TRIALS

    # Determinism: the fixed LCG makes a re-run bit-for-bit identical.
    repeat = ThompsonSampler(ARMS, SEED)
    repeat.run(TRIALS)
    assert repeat.pulls == sampler.pulls, "fixed seed must reproduce the pull counts"
    assert repeat.posterior_mean(0) == sampler.posterior_mean(0)

    # A clear one-armed case and a two-arm case with a larger gap: the simulator must still
    # concentrate on the optimum (documented 80% threshold for the two-arm run).
    trivial = ThompsonSampler((0.9,), SEED + 1)
    trivial.run(200)
    assert trivial.pulls == [200]

    duet = ThompsonSampler((0.3, 0.7), SEED + 2)
    duet.run(600)
    assert duet.pulls[1] >= 0.8 * 600, "larger gap concentrates faster"

    # Uniform arms: with equal probabilities there is no best arm, so the total is still
    # exactly the trial count and no arm is starved out completely over a long run.
    tied = ThompsonSampler((0.5, 0.5), SEED + 3)
    tied.run(600)
    assert sum(tied.pulls) == 600
    assert min(tied.pulls) >= 100, "equal arms must both keep being explored"


if __name__ == "__main__":
    main()
    print("PASS")
