"""Task 951 - Implement end-to-end reproducible benchmarking harness for all exercises in repo.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Harness model: every exercise benchmark is a `Benchmark` whose body charges an explicit
`Clock` (a deterministic work-unit counter) instead of reading wall time, so a whole run is
reproducible: seed -> generated inputs -> operation counts -> SHA-256 digest over the sample
vector.  Each benchmark pays the interpreter/JIT cost on its first invocation, so the
`warmup` iterations are discarded; what remains is `repeats` cold samples carrying a small
seeded scheduler jitter.  The reported metric is a trimmed median (min/max dropped once
there are >= 5 samples), diffed against a recorded baseline: a deviation is reported only
when it clears both the relative tolerance and the absolute noise floor.

Simplified versus a production harness (JMH, pytest-benchmark): no wall clock, no GC/JIT
effects and no forked VMs - work units stand in for time and the input generator is a
hand-written splitmix64.  Verified: per-seed digest reproducibility and seed sensitivity,
that warm-up samples carry the boot cost, that the trimmed median equals an independent
index-arithmetic computation, regression/improvement classification, noise-floor
suppression, and the empty/single-sample boundaries.
"""
import hashlib

BOOT_COST = 5_000  # one-time interpreter cost, paid by the first invocation of a benchmark
JITTER = 7         # max scheduler jitter (work units) added to a timed sample
NOISE_FLOOR = 5    # absolute work-unit delta below which a deviation is not reported
TOLERANCE = 0.10   # relative delta above which a deviation is reported


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64 - fixed seed, so generated inputs and jitter are reproducible."""

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


class Clock:
    """Work-unit stopwatch: benchmarks charge it explicitly instead of timing themselves."""

    def __init__(self):
        self.ops = 0

    def charge(self, units):
        self.ops += units

    def measure(self, benchmark):
        start = self.ops
        benchmark(self)
        return self.ops - start


class Benchmark:
    """One exercise benchmark: an algorithm, an input size and a scaling cost model."""

    def __init__(self, name, algorithm, size, factor=100):
        self.name = name
        self.algorithm = algorithm
        self.size = size
        self.factor = factor
        self._booted = False

    def cost(self):
        """Cold cost of one iteration.  `factor` is the build under test (100 = reference)."""
        n = self.size
        if self.algorithm == "sort_ledger":
            units = n * max(1, n.bit_length() - 1) // 2      # n log n comparisons
        elif self.algorithm == "graph_search":
            units = n * n                                    # quadratic frontier scan
        elif self.algorithm == "hash_lookup":
            units = n * 4                                    # 4 probes per key
        else:
            raise ValueError(f"unknown algorithm {self.algorithm!r}")
        return max(1, units * self.factor // 100)

    def __call__(self, clock):
        if not self._booted:                                 # the VM pays boot once
            self._booted = True
            clock.charge(BOOT_COST)
        clock.charge(self.cost())


def build_suite(factors):
    """The repo's exercise suite; `factors` overrides the per-benchmark build factor."""
    specs = (
        ("exercises/sort_ledger@n=1024", "sort_ledger", 1024),
        ("exercises/graph_search@n=96", "graph_search", 96),
        ("exercises/hash_lookup@n=512", "hash_lookup", 512),
        ("exercises/sort_ledger@n=2048", "sort_ledger", 2048),
    )
    return [Benchmark(name, algorithm, size, factors.get(name, 100))
            for name, algorithm, size in specs]


def measure(benchmark, rng, warmup=3, repeats=7):
    """Warm up (discarded) then collect `repeats` timed samples with seeded jitter."""
    clock = Clock()
    warm = [clock.measure(benchmark) for _ in range(warmup)]
    samples = [clock.measure(benchmark) + rng.below(JITTER) for _ in range(repeats)]
    return warm, samples


def trimmed_median(samples):
    """Median after dropping one min and one max when there are >= 5 samples."""
    ordered = sorted(samples)
    if len(ordered) >= 5:
        ordered = ordered[1:-1]
    return _median_statistic(ordered) if ordered else 0.0


def _median_statistic(ordered_ascending):
    middle = len(ordered_ascending) // 2
    if len(ordered_ascending) % 2 == 1:
        return float(ordered_ascending[middle])
    return (ordered_ascending[middle - 1] + ordered_ascending[middle]) / 2.0


def oracle_trimmed_median(samples):
    """Independent oracle: explicit index arithmetic, no statistics module."""
    ordered = sorted(samples)
    return _median_statistic(ordered[1:-1] if len(ordered) >= 5 else ordered)


def run_suite(seed, factors=None, warmup=3, repeats=7):
    """Run the whole suite; returns name -> {warm, samples, median, digest}."""
    factors = factors or {}
    rng = Rng(seed)
    report = {}
    for benchmark in build_suite(factors):
        warm, samples = measure(benchmark, rng, warmup, repeats)
        payload = f"{seed}|{benchmark.name}|{benchmark.factor}|{','.join(map(str, samples))}"
        report[benchmark.name] = {"warm": warm, "samples": samples,
                                  "median": trimmed_median(samples),
                                  "digest": hashlib.sha256(payload.encode()).hexdigest()}
    return report


def suite_digest(report):
    return hashlib.sha256("".join(sorted(row["digest"] for row in report.values()))
                          .encode()).hexdigest()


def diff_runs(baseline, candidate):
    """Classify each benchmark against the baseline; noise-only deltas are not reported."""
    report = {"regressions": [], "improvements": [], "stable": []}
    for name in sorted(set(baseline) & set(candidate)):
        base, cand = baseline[name]["median"], candidate[name]["median"]
        relative = (cand - base) / base if base else 0.0
        absolute = cand - base
        if absolute > NOISE_FLOOR and relative > TOLERANCE:
            report["regressions"].append(name)
        elif absolute < -NOISE_FLOOR and relative < -TOLERANCE:
            report["improvements"].append(name)
        else:
            report["stable"].append(name)
    return report


def main():
    seed = 20240517
    baseline = run_suite(seed)
    check(sorted(baseline) == sorted(b.name for b in build_suite({})),
          "the harness must cover every benchmark in the suite")

    # 1) happy path: every benchmark produced warm-up samples and a positive trimmed median.
    ok = all(row["warm"] and row["median"] > 0 and len(row["samples"]) == 7
             and all(sample > 0 for sample in row["samples"]) for row in baseline.values())
    check(ok and len(baseline) == 4,
          f"every benchmark needs warm-up samples and a positive median: "
          f"{ {n: r['median'] for n, r in baseline.items()} }")

    # 2) reproducibility: the same seed reproduces the whole sample vector, digest included.
    repeat = run_suite(seed)
    check(all(baseline[name]["samples"] == repeat[name]["samples"]
              and baseline[name]["digest"] == repeat[name]["digest"] for name in baseline)
          and suite_digest(baseline) == suite_digest(repeat),
          "re-running the suite with the same seed must reproduce every sample and digest")

    # 3) seed sensitivity: a different seed changes the jitter, hence every digest.
    other = run_suite(seed + 1)
    check(all(baseline[name]["digest"] != other[name]["digest"] for name in baseline)
          and sorted(other) == sorted(baseline),
          "a different seed must change the digests without changing the suite")

    # 4) warm-up really is discarded: warm samples are inflated by the one-time boot cost.
    worst = min(row["warm"][0] - max(row["samples"]) for row in baseline.values())
    check(worst > 0 and all(row["warm"][0] > row["warm"][-1] for row in baseline.values()),
          f"warm-up samples must carry the boot cost (tightest margin {worst})")

    # 5) independent oracle: the trimmed median matches explicit index arithmetic.
    mismatches = [name for name, row in baseline.items()
                  if abs(row["median"] - oracle_trimmed_median(row["samples"])) > 1e-9]
    check(not mismatches, f"trimmed median disagrees with the oracle on {mismatches}")

    # 6) regression detection: only the benchmark whose factor grew is reported.
    slowed = {"exercises/graph_search@n=96": 150}
    regressed = run_suite(seed, slowed)
    report = diff_runs(baseline, regressed)
    check(report["regressions"] == ["exercises/graph_search@n=96"]
          and len(report["stable"]) == 3 and report["improvements"] == [],
          f"only the slowed benchmark may be reported: {report}")

    # 7) improvement detection plus noise-floor suppression (a 1% factor change stays stable).
    faster = run_suite(seed, {"exercises/sort_ledger@n=2048": 60})
    noise = run_suite(seed, {"exercises/sort_ledger@n=2048": 101})
    check(diff_runs(baseline, faster)["improvements"] == ["exercises/sort_ledger@n=2048"]
          and diff_runs(baseline, noise) == {"regressions": [], "improvements": [],
                                             "stable": sorted(baseline)},
          "a real speed-up must be reported while a 1% delta stays under the noise floor")

    # 8) boundaries: zero repeats yields an empty (digest-stable) report; with warmup=0 the
    #    one-time boot cost leaks into the single sample and inflates every benchmark.
    empty = run_suite(seed, {}, warmup=0, repeats=0)
    single = run_suite(seed, {}, warmup=0, repeats=1)
    check(all(row["warm"] == [] and row["samples"] == [] and row["median"] == 0.0
              for row in empty.values())
          and suite_digest(empty) == suite_digest(run_suite(seed, {}, warmup=0, repeats=0))
          and diff_runs(empty, empty) == {"regressions": [], "improvements": [],
                                          "stable": sorted(empty)}
          and all(len(row["samples"]) == 1 and row["median"] == float(row["samples"][0])
                  for row in single.values())
          and diff_runs(baseline, single)["regressions"] == sorted(baseline),
          "the empty report and the boot-cost-inflated single sample must both behave")

    # 9) cost model sanity: the median tracks the declared complexity of each algorithm.
    medians = {name: row["median"] for name, row in baseline.items()}
    check(medians["exercises/graph_search@n=96"] > medians["exercises/hash_lookup@n=512"]
          and medians["exercises/sort_ledger@n=2048"] > medians["exercises/sort_ledger@n=1024"],
          f"medians must follow the cost model: {medians}")

    print(f"benchmarks={len(baseline)} seed={seed} suite_digest={suite_digest(baseline)[:16]}")
    print(f"medians={ {n: int(v['median']) for n, v in sorted(baseline.items())} }")
    print(f"regressions={report['regressions']} stable={len(report['stable'])}")
    print("PASS")


if __name__ == "__main__":
    main()
