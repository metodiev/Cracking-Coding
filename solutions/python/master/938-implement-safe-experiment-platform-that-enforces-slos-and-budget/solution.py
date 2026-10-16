"""Task 938 - Implement safe experiment platform that enforces SLOs and budget on experiments.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A deterministic, in-process experiment platform.  Experiments declare a traffic allocation, an
expected effect on the primary metric (conversion) and a guardrail metric (harmful events, e.g.
error or latency breach) with an SLO limit.  Every window the platform samples both arms, bills the
observed excess guardrail events to a shared harmed-request budget, pauses any experiment whose
guardrail breach is statistically significant, aborts everything once the budget is exhausted, and
otherwise runs the primary test at an alpha level given by a *harmonic alpha-spending* schedule
(level_k = alpha / (k * H_K), whose levels sum to exactly alpha) so that repeated peeking does not
inflate the false-positive rate.

Simplified versus Optimizely/Statsig: a single primary metric compared with a pooled two-proportion
z-test, no CUPED or variance reduction, no per-tenant budgets, no ramp schedules and no
multi-armed allocation.  Verified: a harmful experiment is paused on the guardrail within a couple of
windows while a beneficial one reaches significance and ships; the billed budget equals the summed
harmful excess; the budget boundary and the zero-allocation boundary; the guardrail statistic matches
a hand-computed pooled z; the spending schedule sums to at most alpha; an A/A simulation (the
independent oracle) shows that fixed-alpha peeking inflates the false-positive rate while the
spending schedule keeps it near nominal; and determinism.
"""
import math


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64 - fixed seed, so every simulated stream is reproducible."""

    def __init__(self, seed):
        self.state = seed & ((1 << 64) - 1)

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return z ^ (z >> 31)

    def below(self, bound):
        return self.next() % bound


def two_prop_z(n_control, s_control, n_treatment, s_treatment):
    """Pooled two-proportion z statistic: positive when the treatment rate is higher."""
    pooled = (s_control + s_treatment) / (n_control + n_treatment)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment))
    return (s_treatment / n_treatment - s_control / n_control) / se if se else 0.0


def tail(z):
    """One-sided standard-normal tail probability."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def alpha_level(peek, peeks, alpha):
    """Harmonic alpha spending: levels sum to alpha over the pre-registered peeks."""
    return alpha / (peek * sum(1.0 / step for step in range(1, peeks + 1)))


class Experiment:
    def __init__(self, name, allocation, effect=0.0, harm=0.0, base_rate=0.10, base_harm=0.02,
                 per_arm=400):
        self.name, self.allocation = name, allocation
        self.effect, self.harm = effect, harm
        self.base_rate, self.base_harm, self.per_arm = base_rate, base_harm, per_arm
        self.status, self.decision = "running", None
        self.peeks, self.harmed = 0, 0

    def window(self, rng, scale):
        """Sample both arms; returns (n, primary successes, harmful events) per arm."""
        size = int(self.per_arm * self.allocation * scale)
        if size <= 0:
            return None
        arms = []
        for rate, harm_rate in ((self.base_rate, self.base_harm),
                                (self.base_rate + self.effect, self.base_harm + self.harm)):
            successes = sum(1 for _ in range(size) if rng.below(1000) < rate * 1000)
            harms = sum(1 for _ in range(size) if rng.below(1000) < harm_rate * 1000)
            arms.append((size, successes, harms))
        self.peeks += 1
        return {"control": arms[0], "treatment": arms[1]}


class Platform:
    """Guardrail enforcement + shared budget + sequential primary test."""

    GUARDRAIL_Z = 2.0

    def __init__(self, alpha=0.05, peeks=6, budget=300):
        self.alpha, self.peeks, self.budget = alpha, peeks, budget
        self.consumed = 0
        self.audit = []

    def has_budget(self):
        return self.consumed < self.budget

    def guardrail_breached(self, sample):
        n_control, _, harm_control = sample["control"]
        n_treatment, _, harm_treatment = sample["treatment"]
        return two_prop_z(n_control, harm_control, n_treatment, harm_treatment) > self.GUARDRAIL_Z

    def primary_result(self, sample, peek):
        n_control, ok_control, _ = sample["control"]
        n_treatment, ok_treatment, _ = sample["treatment"]
        z = two_prop_z(n_control, ok_control, n_treatment, ok_treatment)
        if tail(abs(z)) <= alpha_level(peek, self.peeks, self.alpha):
            return "ship" if z > 0 else "kill"
        return "continue"

    def run_window(self, rng, experiments, window):
        for experiment in experiments:
            if experiment.status != "running":
                continue
            if not self.has_budget():
                experiment.status, experiment.decision = "aborted", "budget"
                self.audit.append((window, experiment.name, "aborted", self.consumed))
                continue
            sample = experiment.window(rng, window)
            if sample is None:
                continue
            excess = max(0, sample["treatment"][2] - sample["control"][2])
            experiment.harmed += excess
            self.consumed += excess
            if self.guardrail_breached(sample):
                experiment.status, experiment.decision = "paused", "guardrail"
                self.audit.append((window, experiment.name, "paused", self.consumed))
            else:
                verdict = self.primary_result(sample, window)
                if verdict != "continue":
                    experiment.status, experiment.decision = verdict, "primary"
                    self.audit.append((window, experiment.name, verdict, self.consumed))
        return self.audit

    def run(self, experiments, windows=6, seed=2024):
        rng = Rng(seed)
        for window in range(1, windows + 1):
            self.run_window(rng, experiments, window)
            if not self.has_budget():                       # platform-wide kill switch
                for experiment in experiments:
                    if experiment.status == "running":
                        experiment.status, experiment.decision = "aborted", "budget"
                        self.audit.append((window, experiment.name, "aborted", self.consumed))
        return experiments


def false_positive_rate(trials, peeks, alpha, spending, seed=12345, per_arm=100):
    """A/A streams (no true effect): how often does the policy stop with a false positive?"""
    hits = 0
    for trial in range(trials):
        rng = Rng(seed + 7919 * trial)
        for peek in range(1, peeks + 1):
            size = per_arm * peek
            control = sum(1 for _ in range(size) if rng.below(1000) < 100)
            treatment = sum(1 for _ in range(size) if rng.below(1000) < 100)
            z = two_prop_z(size, control, size, treatment)
            level = alpha_level(peek, peeks, alpha) if spending else alpha
            if tail(abs(z)) <= level:
                hits += 1
                break
    return hits / trials


def main():
    # 1) a harmful experiment is paused on the guardrail, a beneficial one ships.
    platform = Platform(peeks=6)
    harmful = Experiment("harmful", 0.5, effect=0.0, harm=0.06)
    beneficial = Experiment("beneficial", 0.5, effect=0.05, harm=0.0)
    idle = Experiment("idle", 0.0, effect=0.02)
    platform.run([harmful, beneficial, idle], windows=6)
    check(harmful.status == "paused" and harmful.decision == "guardrail"
          and harmful.peeks <= 2,
          f"the harmful experiment must be paused quickly: {harmful.status} {harmful.decision} "
          f"after {harmful.peeks} windows")
    check(beneficial.status == "ship" and beneficial.decision == "primary",
          f"the beneficial experiment must reach significance: {beneficial.status} "
          f"after {beneficial.peeks} windows")
    # 2) SLO enforcement: the billed budget equals the summed guardrail excess and stays inside it.
    excess = sum(experiment.harmed for experiment in (harmful, beneficial, idle))
    check(platform.consumed == excess and platform.consumed <= platform.budget,
          f"billing mismatch: consumed={platform.consumed} excess={excess}")

    # 3) boundary: the budget predicate flips exactly at the budget, and an exhausted platform
    #    serves no further traffic at all.
    probe = Platform(budget=10)
    probe.consumed = 9
    check(probe.has_budget(), "one request below the budget must still be admissible")
    probe.consumed = 10
    check(not probe.has_budget(), "the budget itself must close the platform")
    starved = Platform(budget=0)
    frosty = Experiment("frosty", 1.0, effect=0.02)
    starved.run([frosty], windows=4)
    check(frosty.peeks == 0 and frosty.status == "aborted"
          and [entry[2] for entry in starved.audit] == ["aborted"],
          f"an exhausted platform must serve no traffic: {starved.audit}")

    # 4) boundary: a zero-allocation experiment is never sampled or charged.
    check(idle.peeks == 0 and idle.harmed == 0 and idle.status == "running",
          f"a zero-allocation experiment must not receive traffic: {idle.status} {idle.peeks}")

    # 5) the guardrail statistic matches a hand-computed pooled two-proportion z, at the threshold.
    reader = Platform()
    quiet = {"control": (500, 100, 10), "treatment": (500, 110, 20)}
    loud = {"control": (500, 100, 10), "treatment": (500, 110, 22)}
    pooled = (10 + 20) / 1000
    hand = (20 / 500 - 10 / 500) / math.sqrt(pooled * (1 - pooled) * (2 / 500))
    check(abs(hand - 1.8542) < 0.001 and not reader.guardrail_breached(quiet),
          f"z={hand:.4f} is below the guardrail threshold and must not pause")
    check(reader.guardrail_breached(loud) and reader.primary_result(quiet, 1) == "continue",
          "a 2.16 z on harmful events must pause while the primary test stays open")

    # 6) the spending schedule is a valid alpha-spending function: the levels sum to alpha.
    alpha, peeks = 0.05, 6
    total = sum(alpha_level(peek, peeks, alpha) for peek in range(1, peeks + 1))
    check(abs(total - alpha) < 1e-12 and alpha_level(1, peeks, alpha) > alpha_level(peeks, peeks,
                                                                                    alpha),
          f"the spending levels must sum to alpha={alpha}: {total}")

    # 7) independent oracle: A/A streams under the two sequential policies.
    nominal = false_positive_rate(300, 5, 0.05, spending=False)
    sequential = false_positive_rate(300, 5, 0.05, spending=True)
    check(nominal >= 0.25 and sequential <= 0.10 and sequential < nominal / 2,
          f"fixed-alpha peeking must inflate the false-positive rate ({nominal:.3f}) and "
          f"spending must control it ({sequential:.3f})")

    # 8) determinism.
    twin = Platform(peeks=6)
    twin.run([Experiment("harmful", 0.5, effect=0.0, harm=0.06),
              Experiment("beneficial", 0.5, effect=0.05, harm=0.0),
              Experiment("idle", 0.0, effect=0.02)], windows=6)
    check(twin.audit == platform.audit and twin.consumed == platform.consumed,
          "two platform sweeps over the same streams must be identical")

    print(f"harmful: {harmful.status} after {harmful.peeks} windows ({harmful.harmed} billed)")
    print(f"beneficial: {beneficial.status} after {beneficial.peeks} windows; "
          f"budget consumed={platform.consumed}/{platform.budget}")
    print(f"A/A false positives: fixed-alpha={nominal:.3f} harmonic-spending={sequential:.3f} "
          f"(nominal 0.050)")
    print("PASS")


if __name__ == "__main__":
    main()
