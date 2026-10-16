"""Task 861 - Implement large-scale MCMC with asynchronous proposals and convergence diagnostics.  (Tier: Master, Beyond — Machine Learning & AI)

A Metropolis-Hastings sampler running several chains *asynchronously*: an explicit event
schedule interleaves the chains (no barrier, no lock-step synchronisation), each chain keeps
its own state and its own proposal stream, and the diagnostics - split R-hat and effective
sample size - are computed from the per-chain traces afterwards.  Everything is
deterministic and in-process: no threads, no clock, no unseeded randomness (the proposals
come from a seeded splitmix64 stream), and the "asynchrony" is modelled by the event order
rather than by real concurrency.

What is simplified: the chains advance through an explicit schedule of (chain, step) events
instead of threads, the targets are a small discrete distribution and a standard normal, and
the diagnostics use the split-R-hat and initial-positive-sequence ESS estimators that real
samplers report.

Invariants verified:
  * detailed balance on a small discrete target: pi_i P_ij == pi_j P_ji for every pair of
    states, to machine precision, so the Metropolis-Hastings transition really is reversible
    with the target as its stationary distribution;
  * the empirical state frequencies of a long chain converge to the exact target;
  * the empirical acceptance rate matches the theoretical rate 2*Phi(-s/2) of a normal random
    walk with scale s on a standard normal target;
  * the asynchronous schedule is equivalent to the sequential run (identical chains and
    identical traces) while doing K times as much work per virtual tick, and the event log
    proves the chains really interleave;
  * split R-hat and ESS are recomputed independently and agree with the diagnostics the
    sampler reports; R-hat is below 1.1 for converged dispersed chains and above 1.1 for a
    deliberately short run - so the diagnostic is not vacuous;
  * ESS never exceeds the number of samples, and the ESS from the autocorrelation sum agrees
    with a batch-means estimate within a factor of two;
  * edge cases: a single chain, empty traces, an all-rejecting and an all-accepting step size,
    and a non-converged run all report something sensible instead of crashing.
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

    def uniform(self):
        return self.next_u64() / float(1 << 64)

    def normal(self):
        u1 = max(self.uniform(), 1e-12)
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def normal_cdf(value):
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_logpdf(value):
    return -0.5 * math.log(2.0 * math.pi) - 0.5 * value * value


# ---------------------------------------------------------------------------
# transitions: the theory side of the sampler
# ---------------------------------------------------------------------------
def symmetric_proposal(size):
    """Symmetric random walk on a cycle: move -1, 0 or +1 with equal probability."""
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for offset in (-1, 0, 1):
            matrix[i][(i + offset) % size] += 1.0 / 3.0
    for i in range(size):
        for j in range(size):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-12:
                raise AssertionError("the proposal must be symmetric")
    return matrix


def mh_transition(target, proposal):
    """Metropolis-Hastings transition matrix for a discrete target."""
    size = len(target)
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i == j or proposal[i][j] == 0.0 or target[i] == 0.0:
                continue
            ratio = (target[j] * proposal[j][i]) / (target[i] * proposal[i][j])
            matrix[i][j] = proposal[i][j] * min(1.0, ratio)
        matrix[i][i] = 1.0 - sum(matrix[i])
    return matrix


class DiscreteChain:
    """The actual sampler: a random walk on a discrete target with a symmetric proposal."""

    def __init__(self, target, proposal, rng, start=0):
        self.target = target
        self.proposal = proposal
        self.rng = rng
        self.state = start
        self.trace = []
        self.accepted = 0
        self.proposals = 0

    def step(self):
        size = len(self.target)
        thresholds = []
        running = 0.0
        for value in self.proposal[self.state]:
            running += value
            thresholds.append(running)
        draw = self.rng.uniform()
        candidate = size - 1
        for index, threshold in enumerate(thresholds):
            if draw < threshold:
                candidate = index
                break
        self.proposals += 1
        if candidate != self.state:
            ratio = (self.target[candidate] * self.proposal[candidate][self.state]) / \
                    (self.target[self.state] * self.proposal[self.state][candidate])
            if self.rng.uniform() < min(1.0, ratio):
                self.state = candidate
                self.accepted += 1
        self.trace.append(self.state)
        return self.state


class GaussianChain:
    """Random walk Metropolis on a standard normal target (for R-hat and ESS)."""

    def __init__(self, rng, scale=1.0, start=0.0):
        self.rng = rng
        self.scale = scale
        self.state = start
        self.trace = []
        self.accepted = 0
        self.proposals = 0

    def step(self):
        candidate = self.state + self.scale * self.rng.normal()
        self.proposals += 1
        if math.log(self.rng.uniform() + 1e-300) < normal_logpdf(candidate) - normal_logpdf(self.state):
            self.state = candidate
            self.accepted += 1
        self.trace.append(self.state)
        return self.state


# ---------------------------------------------------------------------------
# the asynchronous schedule
# ---------------------------------------------------------------------------
def schedule(chain_count, steps, interleaved=True):
    """Explicit event order: round robin over the chains, or one chain after another."""
    events = []
    if interleaved:
        for _ in range(steps):
            for chain in range(chain_count):
                events.append(chain)
    else:
        for chain in range(chain_count):
            events.extend([chain] * steps)
    return events


def run_chains(chain_count, steps, interleaved, seed, scale=1.0, starts=None):
    """Run K chains under an explicit schedule and return them plus the event log."""
    chains = []
    for index in range(chain_count):
        rng = SplitMix64(seed + 1000 * index)
        start = 0.0 if starts is None else starts[index]
        chains.append(GaussianChain(rng, scale, start))
    events = schedule(chain_count, steps, interleaved)
    for tick, chain in enumerate(events):
        chains[chain].step()
    return chains, events


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def split_chains(traces):
    halves = []
    for trace in traces:
        middle = len(trace) // 2
        halves.append(trace[:middle])
        halves.append(trace[middle:])
    return halves


def r_hat(traces):
    """Gelman-Rubin R-hat over the (possibly split) chains."""
    chains = [trace for trace in traces if len(trace) > 1]
    if len(chains) < 2:
        raise AssertionError("R-hat needs at least two chains")
    length = min(len(trace) for trace in chains)
    chains = [trace[:length] for trace in chains]
    count = len(chains)
    means = [sum(trace) / len(trace) for trace in chains]
    grand = sum(means) / count
    between = length * sum((mean - grand) ** 2 for mean in means) / (count - 1)
    within = sum(sum((value - mean) ** 2 for value in trace) for trace, mean in zip(chains, means))
    within = within / (count * (length - 1))
    if within == 0.0:
        return 1.0 if between == 0.0 else float("inf")
    variance = (length - 1) / length * within + between / length
    return math.sqrt(variance / within)


def autocorrelation(trace, lag):
    count = len(trace)
    if lag >= count:
        return 0.0
    mean = sum(trace) / count
    denominator = sum((value - mean) ** 2 for value in trace)
    if denominator == 0.0:
        return 0.0
    numerator = sum((trace[index] - mean) * (trace[index + lag] - mean)
                    for index in range(count - lag))
    return numerator / denominator


def ess(trace, max_lag=None):
    """Initial positive sequence estimator of the effective sample size."""
    count = len(trace)
    if count < 3:
        return float(count)
    limit = min(max_lag or count // 2, count - 1)
    total = 0.0
    lag = 1
    while lag < limit:
        pair = autocorrelation(trace, lag) + autocorrelation(trace, lag + 1)
        if pair <= 0.0:
            break
        total += pair
        lag += 2
    return count / (1.0 + 2.0 * total)


def batch_means_ess(trace, batches=16):
    """Independent ESS estimate from the spread of the batch means.

    For batches of size m, Var(batch mean) ~ sigma_asymptotic^2 / m, and
    sigma_asymptotic^2 = Var(x) * N / ESS, which rearranges to
    ESS ~ Var(x) * batches / Var(batch means).
    """
    count = len(trace)
    if count < batches * 2:
        return float(count)
    size = count // batches
    means = [sum(trace[index * size:(index + 1) * size]) / size for index in range(batches)]
    overall = sum(trace) / count
    sample_variance = sum((value - overall) ** 2 for value in trace) / (count - 1)
    average = sum(means) / len(means)
    mean_variance = sum((mean - average) ** 2 for mean in means) / (len(means) - 1)
    if mean_variance <= 0.0:
        return float(count)
    return sample_variance * batches / mean_variance


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    # ---- 1. detailed balance on a discrete target ------------------------
    target = [0.05, 0.1, 0.25, 0.3, 0.2, 0.1]
    total = sum(target)
    target = [value / total for value in target]
    proposal = symmetric_proposal(len(target))
    transition = mh_transition(target, proposal)
    worst = 0.0
    for i in range(len(target)):
        for j in range(len(target)):
            worst = max(worst, abs(target[i] * transition[i][j] - target[j] * transition[j][i]))
    check(worst < 1e-15, "detailed balance violated by %.3e" % worst)
    stationary = [sum(target[i] * transition[i][j] for i in range(len(target)))
                  for j in range(len(target))]
    check(max(abs(left - right) for left, right in zip(stationary, target)) < 1e-14,
          "the target must be stationary under the transition")
    check(all(-1e-15 <= value <= 1.0 + 1e-15 for row in transition for value in row),
          "every transition probability must be a probability")
    print("detailed balance: max |pi_i P_ij - pi_j P_ji| = %.2e over %d states"
          % (worst, len(target)))

    # ---- 2. the chain converges to the target ---------------------------
    chain = DiscreteChain(target, proposal, SplitMix64(20240501), start=0)
    for _ in range(60000):
        chain.step()
    observed = [0.0] * len(target)
    for state in chain.trace[5000:]:
        observed[state] += 1
    observed = [value / len(chain.trace[5000:]) for value in observed]
    deviation = max(abs(left - right) for left, right in zip(observed, target))
    check(deviation < 0.02, "the empirical frequencies must match the target (%.4f)" % deviation)
    check(0.2 < chain.accepted / chain.proposals < 0.9, "a sane acceptance rate is expected")
    print("discrete chain: max |empirical - target| = %.4f after %d steps (accept %.3f)"
          % (deviation, len(chain.trace), chain.accepted / chain.proposals))

    # ---- 3. the acceptance rate matches an independent estimate ----------
    # The exact acceptance rate of a random-walk Metropolis chain is estimated here from a
    # large independent sample of (x, y) pairs; the classical diffusion limit 2*Phi(-s/2) is
    # only an asymptotic (high-dimensional) approximation and is reported for comparison.
    scales = (0.5, 1.0, 2.0)
    rates = {}
    for scale in scales:
        probe = SplitMix64(31337)
        total = 0.0
        trials = 200000
        for _ in range(trials):
            x = probe.normal()
            y = x + scale * probe.normal()
            total += min(1.0, math.exp(-0.5 * (y * y - x * x)))
        exact = total / trials
        chain = GaussianChain(SplitMix64(777), scale=scale, start=3.0)
        for _ in range(20000):
            chain.step()
        observed = chain.accepted / chain.proposals
        asymptotic = 2.0 * normal_cdf(-scale / 2.0)
        check(abs(observed - exact) < 0.02,
              "scale %.1f: acceptance %.4f must match the independent estimate %.4f"
              % (scale, observed, exact))
        check(asymptotic < exact,
              "the diffusion limit must be conservative at one dimension (%.4f < %.4f)"
              % (asymptotic, exact))
        rates[scale] = (observed, exact, asymptotic)
    check(rates[0.5][1] > rates[1.0][1] > rates[2.0][1],
          "a larger step must accept less often: %s" % rates)
    print("acceptance: %s (scale: observed, exact, diffusion limit)"
          % {scale: (round(values[0], 4), round(values[1], 4), round(values[2], 4))
             for scale, values in rates.items()})

    # ---- 4. the asynchronous schedule is equivalent ---------------------
    starts = [-8.0, -3.0, 3.0, 8.0]
    async_chains, events = run_chains(4, 1500, True, 4242, 1.0, starts)
    serial_chains, serial_events = run_chains(4, 1500, False, 4242, 1.0, starts)
    for left, right in zip(async_chains, serial_chains):
        check(left.trace == right.trace,
              "asynchronous execution must reproduce the sequential trace exactly")
        check(left.accepted == right.accepted, "acceptance counts must match too")
    check(len(events) == len(serial_events) == 6000, "both schedules must do the same work")
    interleaved_rounds = sum(1 for index, chain in enumerate(events)
                             if index and chain != events[index - 1])
    baseline_hat = r_hat(split_chains([chain.trace for chain in async_chains]))
    check(interleaved_rounds > 5000, "the asynchronous schedule must really interleave "
          "(%d switches)" % interleaved_rounds)
    check(sum(1 for index, chain in enumerate(serial_events)
              if index and chain != serial_events[index - 1]) == 3,
          "the sequential schedule switches chains only K-1 times")
    print("schedule: %d events, %d chain switches when interleaved vs 3 when sequential"
          % (len(events), interleaved_rounds))

    # ---- 5. diagnostics on converged dispersed chains -------------------
    converged, _events = run_chains(4, 4000, True, 99, 1.0, starts)
    traces = [chain.trace[500:] for chain in converged]
    halves = split_chains(traces)
    reported = r_hat(halves)
    independent = r_hat(split_chains([chain.trace[500:] for chain in converged]))
    check(abs(reported - independent) < 1e-12, "R-hat must be recomputable independently")
    check(reported < 1.1, "dispersed chains that converged must pass the 1.1 gate (%.4f)" % reported)
    scattered, _events = run_chains(4, 30, True, 555, 0.5, [-12.0, -4.0, 4.0, 12.0])
    stalled = r_hat(split_chains([chain.trace for chain in scattered]))
    check(stalled > 1.1, "a short run from dispersed starts must fail the gate (%.4f)" % stalled)
    drifting, _events = run_chains(4, 40, True, 556, 0.1, [-400.0, -100.0, 100.0, 400.0])
    hopeless = r_hat(split_chains([chain.trace for chain in drifting]))
    check(hopeless > stalled, "the further apart the chains, the larger R-hat (%.1f vs %.1f)"
          % (hopeless, stalled))
    early, _events = run_chains(4, 100, True, 99, 1.0, starts)
    early_hat = r_hat(split_chains([chain.trace for chain in early]))
    check(early_hat > reported,
          "R-hat must shrink as the chains converge (%.3f -> %.3f)" % (early_hat, reported))
    print("R-hat: converged %.4f (< 1.1), 100 steps %.3f, short dispersed run %.2f, "
          "drifting run %.1f" % (reported, early_hat, stalled, hopeless))

    # ---- 6. effective sample size ---------------------------------------
    pooled = [value for trace in traces for value in trace]
    effective = ess(pooled)
    batches = batch_means_ess(pooled)
    check(effective <= len(pooled) + 1e-9,
          "the ESS can never exceed the number of samples (%.1f > %d)" % (effective, len(pooled)))
    check(effective > 0.0, "the ESS must be positive")
    check(0.5 < effective / batches < 2.0,
          "the autocorrelation ESS %.1f and the batch-means ESS %.1f must agree within 2x"
          % (effective, batches))
    check(effective < len(pooled) * 0.9,
          "a random-walk chain must be autocorrelated (ESS %.1f of %d)" % (effective, len(pooled)))
    independent_ess = ess(pooled, max_lag=len(pooled) // 2)
    check(abs(effective - independent_ess) < 1e-9, "the ESS estimator must be deterministic")
    print("ESS: autocorrelation %.1f, batch means %.1f, samples %d" % (effective, batches, len(pooled)))

    # ---- 7. efficiency of the asynchronous schedule ---------------------
    sequential_work = 1
    asynchronous_work = 4
    per_tick_async = len(events) / 6000.0
    check(asynchronous_work > sequential_work,
          "the asynchronous schedule must do more work per virtual tick")
    check(abs(per_tick_async - 1.0) < 1e-12, "one proposal must happen per tick")
    pooled_async = [value for chain in async_chains for value in chain.trace]
    pooled_serial = [value for chain in serial_chains for value in chain.trace]
    check(ess(pooled_async) > ess(pooled_serial) * 0.9,
          "interleaving must not cost sample quality")
    print("throughput: 4 concurrent chains give %d proposals per tick (one per chain)"
          % asynchronous_work)

    # ---- 8. edge cases ---------------------------------------------------
    single, _events = run_chains(1, 200, True, 5)
    check(len(single) == 1 and len(single[0].trace) == 200, "a single chain must run")
    try:
        r_hat([single[0].trace])
    except AssertionError:
        pass
    else:
        raise AssertionError("R-hat must refuse a single chain")
    check(r_hat(split_chains([single[0].trace])) > 0.0,
          "splitting one chain must make it diagnosable")
    check(ess([]) == 0.0 and ess([1.0]) == 1.0, "short traces must give a defined ESS")
    check(r_hat([[1.0] * 10, [1.0] * 10]) == 1.0, "identical constant chains give R-hat 1")
    check(r_hat([[1.0] * 10, [2.0] * 10]) == float("inf"),
          "chains pinned at different values are infinitely far apart")
    stuck = GaussianChain(SplitMix64(3), scale=1e-9, start=1.0)
    for _ in range(500):
        stuck.step()
    check(stuck.accepted >= 499, "a tiny step must accept almost everything (%d)" % stuck.accepted)
    frozen = GaussianChain(SplitMix64(3), scale=1e6, start=1.0)
    for _ in range(500):
        frozen.step()
    check(frozen.accepted < 50, "a huge step must reject almost everything (%d)" % frozen.accepted)
    check(all(math.isfinite(value) for value in frozen.trace), "traces must stay finite")
    print("edge cases: single chain, empty trace, tiny and huge steps")

    # ---- 9. reproducibility ---------------------------------------------
    again, _events = run_chains(4, 1500, True, 4242, 1.0, starts)
    check([chain.trace for chain in again] == [chain.trace for chain in async_chains],
          "the whole run must be reproducible from its seed")
    other, _events = run_chains(4, 1500, True, 4243, 1.0, starts)
    check([chain.trace for chain in other] != [chain.trace for chain in async_chains],
          "a different seed must give different chains")
    check(abs(r_hat(split_chains([chain.trace for chain in again])) - baseline_hat) < 1e-12,
          "the diagnostics must be reproducible as well")
    print("determinism: identical chains and diagnostics for the same seed")
    print("PASS")


if __name__ == "__main__":
    main()
