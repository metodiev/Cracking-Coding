"""Task 937 - Implement deployment pipeline with canary analysis using online metrics and rollback.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A deterministic canary rollout driven by online metrics.  A stage raises the canary traffic weight
(5%, 25%, 50%, 100%) and opens one observation window in which a seeded generator produces, for the
baseline and the canary, the request count, the error count and a latency sample.  The analyst then
decides strictly from those recorded counts: a guardrail rollback when the canary error rate is
above the absolute limit, a rollback when the canary's share of the shared error budget is used up,
HOLD when the window is too small to judge, a one-sided pooled two-proportion z-test on the error
rate, and a p95 latency ratio test.  PROMOTE advances the weight, ROLLBACK restores the previous
stable version and stops the rollout, HOLD waits for more traffic.

Simplified versus a real pipeline (Argo/Spinnaker + Kayenta): one metric window per stage instead
of continuous analysis with automated re-analysis, latency is compared at p95 by ratio rather than
by a distribution test, and there is no traffic mirroring.  Verified: a healthy candidate promotes
through every stage; an error-rate regression rolls back through the statistical path; a
latency-only regression rolls back on latency while the error test stays clean; the error budget
forces a rollback when the per-stage tests do not; the sample-size boundary and the zero-traffic
boundary behave; the z statistic matches an independently hand-computed value and the PROMOTE ->
ROLLBACK transition is monotone in the canary error count; rollback restores the stable version; and
the rollout is deterministic.
"""
import math


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def quantile(values, q):
    """Nearest-rank quantile, so no interpolation decides the verdict."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))]


def zscore(n_base, e_base, n_can, e_can):
    """One-sided pooled two-proportion z statistic on the error rate."""
    pooled = (e_base + e_can) / (n_base + n_can)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_base + 1 / n_can))
    return (e_can / n_can - e_base / n_base) / se if se else 0.0


class Rng:
    """splitmix64 - fixed seed, so every rollout is reproducible."""

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


class Service:
    """Deterministic traffic generator; error rates are per mille so counts stay integral."""

    def __init__(self, base_error, canary_error, base_latency, canary_latency, requests=1000,
                 spread=8):
        self.base_error, self.canary_error = base_error, canary_error
        self.base_latency, self.canary_latency = base_latency, canary_latency
        self.requests, self.spread = requests, spread

    def window(self, rng, error_rate, latency):
        per_mille = int(round(error_rate * 1000))
        errors = sum(1 for _ in range(self.requests) if rng.below(1000) < per_mille)
        latencies = [latency + rng.below(self.spread) for _ in range(self.requests)]
        return (self.requests, errors, latencies)

    def __call__(self, stage, weight):
        rng = Rng(97_531 * (stage + 1) + int(weight * 100))
        return {"baseline": self.window(rng, self.base_error, self.base_latency),
                "canary": self.window(rng, self.canary_error, self.canary_latency)}


class Canary:
    """Stage-wise rollout state machine over an online-metric analyst."""

    STAGES = (0.05, 0.25, 0.5, 1.0)
    MIN_REQUESTS = 200
    MAX_ERROR_RATE = 0.05
    Z_CRITICAL = 2.326          # one-sided 1% level
    LATENCY_TOLERANCE = 1.5
    DEFAULT_ERROR_BUDGET = 150

    def __init__(self, service, stable="v1", candidate="v2", error_budget=DEFAULT_ERROR_BUDGET):
        self.service = service
        self.stable, self.candidate = stable, candidate
        self.error_budget = error_budget
        self.version = stable
        self.weight = 0.0
        self.canary_errors = 0
        self.status = "pending"
        self.trail = []           # (stage index, weight, decision, reason)

    def analyze(self, sample):
        """Decide from one recorded window; returns (decision, reason)."""
        base_n, base_errors, base_lat = sample["baseline"]
        can_n, can_errors, can_lat = sample["canary"]
        if can_n == 0 or base_n == 0:
            return "HOLD", "no_traffic"
        self.canary_errors += can_errors
        p_base, p_can = base_errors / base_n, can_errors / can_n
        if p_can > self.MAX_ERROR_RATE and p_can > p_base:
            return "ROLLBACK", "guardrail_error_rate"
        if self.canary_errors > self.error_budget:
            return "ROLLBACK", "error_budget"
        if can_n < self.MIN_REQUESTS:
            return "HOLD", "insufficient_samples"
        z = zscore(base_n, base_errors, can_n, can_errors)
        if z > self.Z_CRITICAL:
            return "ROLLBACK", "error_rate_regression"
        ratio = quantile(can_lat, 0.95) / max(quantile(base_lat, 0.95), 1e-9)
        if ratio > self.LATENCY_TOLERANCE:
            return "ROLLBACK", "latency_regression"
        return "PROMOTE", f"z={z:.2f} p95_ratio={ratio:.2f}"

    def roll(self):
        self.status = "running"
        for stage, weight in enumerate(self.STAGES):
            if self.status != "running":
                break
            decision, reason = self.analyze(self.service(stage, weight))
            self.trail.append((stage, weight, decision, reason))
            if decision == "PROMOTE":
                self.weight = weight
                self.version = self.candidate if weight == 1.0 else self.stable
            elif decision == "ROLLBACK":
                self.status, self.weight, self.version = "rolled_back", 0.0, self.stable
            else:
                self.status = "held"
        if self.status == "running":
            self.status, self.weight, self.version = "active", 1.0, self.candidate
        return self

    def reasons(self):
        return [reason for _, _, _, reason in self.trail]


def fixed_sample(errors, requests, canary_errors, canary_requests, latency=(100, 100)):
    return {"baseline": (requests, errors, [latency[0]] * requests),
            "canary": (canary_requests, canary_errors, [latency[1]] * canary_requests)}


def main():
    # 1) happy path: a healthy candidate is promoted to full traffic.
    healthy = Canary(Service(0.02, 0.02, 120, 122)).roll()
    check([decision for _, _, decision, _ in healthy.trail] == ["PROMOTE"] * 4
          and healthy.status == "active" and healthy.weight == 1.0
          and healthy.version == "v2",
          f"healthy rollout: {healthy.trail} status={healthy.status}")

    # 2) error-rate regression rolls back through the statistical test, not the guardrail.
    regressed = Canary(Service(0.005, 0.03, 120, 120)).roll()
    check(regressed.status == "rolled_back" and regressed.reasons()[0] == "error_rate_regression"
          and regressed.weight == 0.0 and regressed.version == "v1",
          f"error regression: {regressed.trail}")

    # 3) independent oracle: the z statistic matches the hand-computed value for the window.
    window = Service(0.005, 0.03, 120, 120)(0, 0.05)
    observed = zscore(*window["baseline"][:2], *window["canary"][:2])
    base_n, base_e, _ = window["baseline"]
    can_n, can_e, _ = window["canary"]
    pooled = (base_e + can_e) / (base_n + can_n)
    hand = ((can_e / can_n) - (base_e / base_n)) / math.sqrt(
        pooled * (1 - pooled) * (2 / base_n))
    check(abs(observed - hand) < 1e-12 and observed > Canary.Z_CRITICAL,
          f"z statistic {observed} must equal the hand-computed {hand}")

    # 4) guardrail boundary is strict: exactly the maximum error rate is not a guardrail breach.
    edge = Canary(Service(0.0, 0.0, 100, 100))
    decision, reason = edge.analyze(fixed_sample(0, 1000, 50, 1000))
    check((decision, reason) == ("ROLLBACK", "error_rate_regression"),
          f"a 5% canary error rate must not trip the strict guardrail: {decision} {reason}")
    over = Canary(Service(0.0, 0.0, 100, 100))
    check(over.analyze(fixed_sample(0, 1000, 51, 1000)) == ("ROLLBACK", "guardrail_error_rate"),
          "one request above the limit must trip the guardrail")

    # 5) sample-size boundary: MIN_REQUESTS is analyzable, one request less is a HOLD.
    small = Canary(Service(0.0, 0.0, 100, 100))
    check(small.analyze(fixed_sample(2, 200, 8, 200))[0] == "PROMOTE",
          "exactly the minimum sample must be analyzed")
    tiny = Canary(Service(0.0, 0.0, 100, 100))
    check(tiny.analyze(fixed_sample(2, 199, 8, 199)) == ("HOLD", "insufficient_samples"),
          "one request short of the minimum must hold")

    # 6) latency-only regression: errors are clean, p95 grows by 3x.
    slow = Canary(Service(0.02, 0.02, 100, 300)).roll()
    check(slow.status == "rolled_back" and slow.reasons()[0] == "latency_regression",
          f"latency regression: {slow.trail}")

    # 7) error budget: many small windows exhaust the shared budget without a significant test.
    budget = Canary(Service(0.02, 0.04, 120, 120, requests=400), error_budget=30).roll()
    check(budget.status == "rolled_back" and "error_budget" in budget.reasons()
          and budget.canary_errors > budget.error_budget,
          f"budget rollout: {budget.trail} errors={budget.canary_errors}")

    # 8) monotonicity property: raising the canary error count never softens the verdict.
    ranks = {"PROMOTE": 0, "HOLD": 1, "ROLLBACK": 2}
    verdicts = [Canary(Service(0.0, 0.0, 100, 100)).analyze(
        fixed_sample(20, 1000, canary_errors, 1000))[0] for canary_errors in range(0, 90, 5)]
    check(ranks[verdicts[0]] == 0 and ranks[verdicts[-1]] == 2
          and all(ranks[first] <= ranks[second]
                  for first, second in zip(verdicts, verdicts[1:])),
          f"the decision must be monotone in the canary error count: {verdicts}")

    # 9) zero-traffic and determinism boundaries.
    idle = Canary(Service(0.0, 0.0, 100, 100))
    check(idle.analyze(fixed_sample(0, 0, 0, 0)) == ("HOLD", "no_traffic"),
          "an empty window must hold instead of dividing by zero")
    first, second = Canary(Service(0.005, 0.03, 120, 120)).roll(), regressed
    check(first.trail == second.trail and first.status == second.status,
          "two rollouts over the same scenario must be identical")

    print(f"healthy={healthy.status}@{healthy.weight} trail={healthy.reasons()}")
    print(f"regressed={regressed.status} trail={regressed.trail}")
    print(f"z(window0)={observed:.4f} budget_errors={budget.canary_errors} "
          f"verdict ladder={verdicts}")
    print("PASS")


if __name__ == "__main__":
    main()
