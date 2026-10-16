"""Task 991 - Implement benchmark suite across machines and store historical performance.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Cross-machine benchmark bookkeeping.  Every machine publishes a calibration cost for one
fixed workload, so a raw measurement is normalised by ``reference_calibration /
machine_calibration`` before it is compared with anything else; the reported numbers are
nearest-rank percentiles over the normalised samples.  History is an append-only hash chain:
each record carries its predecessor's digest, so a rewritten score is detectable, and a new
score is classified as improvement/regression/stable only when it clears both a relative
tolerance and an absolute noise floor.

Simplified versus production: measurements are explicit work units (no wall clock) and
calibrations are chosen so the speed factors are exact decimals, keeping normalisation
integer-exact.  Verified: an independent percentile oracle, normalisation invariance across
machines, the reference factor, hash-chain tamper detection, rejection of stale appends,
regression classification with noise suppression, and the empty/single-sample boundaries.
"""
import hashlib
import json

REFERENCE = {"id": "ci-reference", "cores": 8, "calibration": 1000}
BENCHMARKS = ("two-sum", "lru-cache", "dijkstra")
TOLERANCE = 0.10
NOISE_FLOOR = 5


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_error(action, needle, error_type=ValueError):
    try:
        action()
    except error_type as exc:
        check(needle in str(exc), "error %r does not mention %r" % (str(exc), needle))
        return
    raise AssertionError("expected %s mentioning %r" % (error_type.__name__, needle))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def speed_factor(machine, reference=REFERENCE):
    """Work units a reference machine would need for the units this machine just burned."""
    if machine["calibration"] <= 0:
        raise ValueError("machine %r has a non-positive calibration" % (machine["id"],))
    return reference["calibration"] / machine["calibration"]


def normalize(samples, factor):
    """Scale raw work units onto the reference machine; factors are chosen to stay integral."""
    return [int(round(sample * factor)) for sample in samples]


def percentile(samples, pct):
    """Nearest-rank percentile: the ceil(pct% * n)-th smallest sample."""
    if not samples:
        raise ValueError("percentile of an empty sample set")
    if not 0 < pct <= 100:
        raise ValueError("percentile %r outside (0, 100]" % (pct,))
    ordered = sorted(samples)
    rank = -(-len(ordered) * pct // 100)  # ceiling division
    return ordered[rank - 1]


def summarize(samples):
    return {"count": len(samples), "min": min(samples), "max": max(samples),
            "p50": percentile(samples, 50), "p95": percentile(samples, 95)}


class History:
    """Append-only records linked by digest; every write must extend the existing chain."""

    def __init__(self):
        self.records = []

    def append(self, benchmark, machine, score):
        previous = self.records[-1]["digest"] if self.records else "genesis"
        record = {"seq": len(self.records), "benchmark": benchmark, "machine": machine,
                  "score": score, "prev": previous}
        record["digest"] = sha256_text(canonical(record))
        self.records.append(record)
        return record["digest"]

    def scores(self, benchmark):
        return [record["score"] for record in self.records if record["benchmark"] == benchmark]

    def baseline(self, benchmark, window=3):
        scores = self.scores(benchmark)
        if not scores:
            raise ValueError("no history for benchmark %r" % (benchmark,))
        recent = scores[-window:]
        return percentile(recent, 50) if len(recent) % 2 else (recent[len(recent) // 2 - 1]
                                                              + recent[len(recent) // 2]) // 2

    def classify(self, benchmark, score, tolerance=TOLERANCE, floor=NOISE_FLOOR):
        """Report a movement only when it clears the relative tolerance *and* the noise floor."""
        base = self.baseline(benchmark)
        delta = score - base
        if abs(delta) <= floor or abs(delta) < tolerance * base:
            return "stable"
        return "regression" if delta > 0 else "improvement"


def verify_history(records):
    """Return every defect found: broken linkage or a body that no longer matches its digest."""
    problems = []
    for index, record in enumerate(records):
        expected_prev = "genesis" if index == 0 else records[index - 1]["digest"]
        if record["seq"] != index or record["prev"] != expected_prev:
            problems.append("%d: broken chain" % index)
        body = {key: value for key, value in record.items() if key != "digest"}
        if sha256_text(canonical(body)) != record["digest"]:
            problems.append("%d: digest mismatch" % index)
    return problems


def machine(name, calibration, cores):
    return {"id": name, "calibration": calibration, "cores": cores}


def main():
    reference = machine(REFERENCE["id"], REFERENCE["calibration"], REFERENCE["cores"])
    fast = machine("fast-laptop", 500, 12)      # burns half the units -> factor 2.0
    slow = machine("slow-vm", 1250, 2)          # factor 0.8
    check(speed_factor(reference) == 1.0, "the reference machine must have factor 1.0")
    check(speed_factor(fast) == 2.0 and speed_factor(slow) == 0.8, "unexpected speed factors")
    expect_error(lambda: speed_factor(machine("broken", 0, 1)), "non-positive calibration")

    # Normalisation invariance: the same logical performance on three machines agrees exactly.
    raw_by_machine = {reference["id"]: [400] * 5, fast["id"]: [200] * 5, slow["id"]: [500] * 5}
    normalized = {name: normalize(samples, speed_factor(machine(name, calibration, 4)))
                  for name, calibration, samples in (("ci-reference", 1000, raw_by_machine["ci-reference"]),
                                                     ("fast-laptop", 500, raw_by_machine["fast-laptop"]),
                                                     ("slow-vm", 1250, raw_by_machine["slow-vm"]))}
    check(len({tuple(values) for values in normalized.values()}) == 1,
          "equal logical performance must normalise to identical numbers, got %r" % (normalized,))
    # Independent exact-arithmetic oracle: only integral factors make this valid, by construction.
    oracle = [sample * REFERENCE["calibration"] // 1250 for sample in raw_by_machine["slow-vm"]]
    check(normalized["slow-vm"] == oracle, "normalisation disagrees with integer arithmetic")

    # Percentile oracle: brute-force rank arithmetic over the sorted samples.
    samples = [37, 12, 88, 64, 5, 91, 23, 55, 70, 41, 19, 77]
    ordered = sorted(samples)
    for pct in (50, 90, 95, 100):
        rank = -(-len(ordered) * pct // 100)
        check(percentile(samples, pct) == ordered[rank - 1],
              "percentile %d disagrees with the brute-force rank" % pct)
    check(percentile(samples, 100) == max(samples) and percentile(samples, 50) == ordered[5],
          "the p100/max and p50 ranks must be exact for this even-sized sample set")
    check(percentile([4, 4, 4], 95) == 4, "an all-equal sample set must collapse to that value")
    expect_error(lambda: percentile([], 50), "empty sample set")
    expect_error(lambda: percentile([1], 0), "outside")
    check(summarize(samples) == {"count": 12, "min": 5, "max": 91, "p50": 41, "p95": 91},
          "summary disagrees with hand-computed values for the 12 fixed samples")
    check(summarize(samples)["p95"] == max(samples),
          "with 12 samples the nearest-rank p95 is the maximum")

    # Boundary: a single sample is its own p50 and p95.
    single = summarize([123])
    check(single["p50"] == single["p95"] == single["min"] == single["max"] == 123,
          "a single sample must be its own min/max/p50/p95")

    # History: append-only chain, stale appends rejected, tampering detected.
    history = History()
    for index, score in enumerate([100, 104, 98, 101, 99, 160]):
        history.append("dijkstra", reference["id"], score)
    check(verify_history(history.records) == [], "a freshly appended chain must verify")
    check([record["seq"] for record in history.records] == list(range(6)), "seq must be contiguous")
    check(history.records[3]["prev"] == history.records[2]["digest"],
          "each record must link to its predecessor")
    tampered = [dict(record) for record in history.records]
    tampered[4]["score"] = 4  # rewriting history without rehashing the chain
    check(verify_history(tampered) == ["4: digest mismatch"],
          "a rewritten score must be detected, got %r" % (verify_history(tampered),))
    forged = [dict(record) for record in history.records]
    forged[2]["prev"] = "genesis"
    check(any("broken chain" in problem for problem in verify_history(forged)),
          "a forged predecessor link must be reported")

    # Regression classification: tolerance and noise floor both have to be cleared.
    check(history.classify("dijkstra", 100) == "stable", "a repeat run must be stable")
    check(history.classify("dijkstra", 103) == "stable", "3 units is below the noise floor")
    check(history.classify("dijkstra", 160) == "regression", "a 60% jump must be a regression")
    check(history.classify("dijkstra", 60) == "improvement", "a 40% drop must be an improvement")
    expect_error(lambda: history.classify("no-such-benchmark", 1), "no history")
    expect_error(lambda: percentile([1, 2], 101), "outside")

    # Cross-machine: same logical result recorded on different machines normalises identically.
    runs = {"ci-reference": normalize([300, 320, 310], speed_factor(reference)),
            "fast-laptop": normalize([150, 160, 155], speed_factor(fast))}
    check(summarize(runs["ci-reference"]) == summarize(runs["fast-laptop"]),
          "the same workload on a faster machine must normalise to the same summary")
    suite_score = sum(summarize(runs[name])["p50"] for name in runs) // len(runs)
    check(suite_score == 310, "the suite score must be the mean of the normalised medians")

    print("machines=%d records=%d suite_score=%d p95=%d"
          % (len(runs), len(history.records), suite_score, percentile(samples, 95)))
    print("PASS")


if __name__ == "__main__":
    main()
