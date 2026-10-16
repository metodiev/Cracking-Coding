"""Task 992 - Implement reproducible random-seed protocol for benchmarking fairness.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A seed protocol that gives every benchmark (and every parallel worker of it) an independent,
reproducible stream of inputs.  Seeds are derived as SHA-256(master | namespace | benchmark |
worker) truncated to 64 bits, so the protocol is a pure function of the master seed: two runs
re-derive identical streams while no benchmark can piggyback on another's inputs.  The protocol
also refuses to issue the same seed twice for different work (a collision would silently couple
two benchmarks) and guards against degenerate seeds.

Simplified versus production: streams are produced by a hand-written splitmix64 generator
rather than a language PRNG (so it is portable across languages), and the fairness evidence is
a bounded chi-square statistic plus a mean check over a few thousand draws.  Verified:
determinism of the protocol digest, sensitivity to the master seed, uniqueness across
benchmarks/workers, no reuse, uniform bucket counts within bound, the weak-seed guard, and the
empty-stream and bad-identifier boundaries.
"""
import hashlib

MASTER_SEED = 0x5EED20261016
BENCHMARKS = ("two-sum", "lru-cache", "dijkstra", "bubble-sort", "binary-search", "n-queens")
DRIFT_TOLERANCE = 0.05          # |mean - 0.5| allowed for a uniform stream
CHI_SQUARE_BOUND = 30.0         # 7 degrees of freedom, ~1e-4 tail
MIN_POPCOUNT = 20               # a derived seed must not be sparse


class SeedCollision(RuntimeError):
    """Two different pieces of work were handed the same seed."""


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


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def splitmix64(seed, count):
    """Portable deterministic stream: fixed seed in, identical numbers out, every language."""
    state, out = seed & ((1 << 64) - 1), []
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        out.append(z ^ (z >> 31))
    return out


class SeedProtocol:
    """Pure function from (master, namespace, benchmark, worker) to a 64-bit seed."""

    def __init__(self, master, namespace="cracking-coding-v1"):
        if not isinstance(master, int) or master < 0:
            raise ValueError("master seed must be a non-negative integer")
        if not namespace:
            raise ValueError("namespace must be non-empty")
        self.master = master
        self.namespace = namespace
        self.issued = {}

    def seed_for(self, benchmark, worker=0):
        if not benchmark or not isinstance(benchmark, str):
            raise ValueError("benchmark id must be a non-empty string")
        if worker < 0:
            raise ValueError("worker index must be non-negative")
        material = "%d|%s|%s|%d" % (self.master, self.namespace, benchmark, worker)
        seed = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
        key = (benchmark, worker)
        owner = self.issued.setdefault(seed, key)
        if owner != key:
            raise SeedCollision("seed collision between %r and %r" % (owner, key))
        if seed == self.master or bin(seed).count("1") < MIN_POPCOUNT:
            raise SeedCollision("degenerate seed for %r" % (key,))
        return seed

    def stream(self, benchmark, worker=0, count=8):
        if count < 0:
            raise ValueError("stream length must not be negative")
        return splitmix64(self.seed_for(benchmark, worker), count)

    def digest(self, benchmarks, workers=1, count=8):
        """The reproducibility key: same master seed must reproduce every stream exactly."""
        parts = []
        for benchmark in sorted(benchmarks):
            for worker in range(workers):
                seed = self.seed_for(benchmark, worker)
                drawn = ",".join(str(value) for value in self.stream(benchmark, worker, count))
                parts.append("%s#%d=%d:%s" % (benchmark, worker, seed, drawn))
        return sha256_text(";".join(parts))


def bucket_counts(values, buckets=8):
    counts = [0] * buckets
    for value in values:
        counts[value % buckets] += 1
    return counts


def chi_square(counts):
    """Goodness-of-fit statistic against a uniform bucket prior."""
    total = sum(counts)
    expected = total / len(counts)
    return sum((count - expected) ** 2 / expected for count in counts)


def stream_mean(values):
    return sum(value / (1 << 64) for value in values) / len(values)


def main():
    protocol = SeedProtocol(MASTER_SEED)
    twin = SeedProtocol(MASTER_SEED)

    # Determinism: the protocol digest is a pure function of the master seed.
    digest = protocol.digest(BENCHMARKS, workers=3, count=12)
    check(digest == twin.digest(BENCHMARKS, workers=3, count=12),
          "two protocols with the same master seed must reproduce the same digest")
    check(digest == SeedProtocol(MASTER_SEED).digest(tuple(reversed(BENCHMARKS)), workers=3, count=12),
          "the digest must not depend on the order benchmarks are requested in")
    check(len(digest) == 64 and SeedProtocol(MASTER_SEED + 1).digest(BENCHMARKS, workers=3) != digest,
          "a different master seed must give a different protocol digest")
    check(SeedProtocol(MASTER_SEED, "other-namespace").digest(BENCHMARKS, workers=3) != digest,
          "a different namespace must give a different protocol digest")

    # Uniqueness: seeds never repeat across benchmarks and workers, and are not the master seed.
    seeds = {protocol.seed_for(benchmark, worker): (benchmark, worker)
             for benchmark in BENCHMARKS for worker in range(3)}
    check(len(seeds) == len(BENCHMARKS) * 3, "every (benchmark, worker) pair needs its own seed")
    check(MASTER_SEED not in seeds, "no benchmark may reuse the raw master seed")
    check(all(bin(seed).count("1") >= MIN_POPCOUNT for seed in seeds),
          "every derived seed must be dense enough to avoid degenerate streams")
    check(protocol.seed_for("two-sum", 0) == SeedProtocol(MASTER_SEED).seed_for("two-sum", 0),
          "a benchmark seed must be re-derivable from the master seed alone")

    # Independence: different work must not share a stream.
    first = protocol.stream("two-sum", count=16)
    check(first != protocol.stream("lru-cache", count=16), "benchmarks must not share an input stream")
    check(first != protocol.stream("two-sum", 1, 16), "workers must not share an input stream")
    check(len(set(protocol.stream("two-sum", worker=1, count=64))) == 64,
          "a worker stream must not repeat a value within 64 draws")
    check(protocol.stream("two-sum", count=0) == [] and protocol.stream("two-sum", 3, 0) == [],
          "a zero-length stream must be empty and still allocate a seed")

    # Fairness evidence: bucket counts stay uniform for every benchmark.
    worst_chi, worst_drift, worst_benchmark = 0.0, 0.0, ""
    for benchmark in BENCHMARKS:
        values = protocol.stream(benchmark, count=512)
        statistic = chi_square(bucket_counts(values))
        drift = abs(stream_mean(values) - 0.5)
        worst_chi, worst_drift, worst_benchmark = (
            max(worst_chi, statistic), max(worst_drift, drift), benchmark)
        check(statistic < CHI_SQUARE_BOUND,
              "chi-square %.1f for %s exceeds the bound %.1f" % (statistic, benchmark, CHI_SQUARE_BOUND))
        check(drift < DRIFT_TOLERANCE,
              "mean drift %.3f for %s exceeds %.2f" % (drift, benchmark, DRIFT_TOLERANCE))
        check(sum(bucket_counts(values)) == 512, "bucket counts must cover every draw")
    check(worst_chi >= 0.0 and worst_drift < DRIFT_TOLERANCE,
          "the worst observed statistics must still respect the bounds")

    # Reuse detection: a tampered registry would otherwise hand one seed to two benchmarks.
    replay = SeedProtocol(MASTER_SEED)
    replay.issued[protocol.seed_for("two-sum")] = ("some-other-benchmark", 0)
    expect_error(lambda: replay.seed_for("two-sum"), "collision", SeedCollision)
    expect_error(lambda: SeedProtocol(-1), "non-negative")
    expect_error(lambda: SeedProtocol(MASTER_SEED, ""), "namespace")
    expect_error(lambda: protocol.seed_for(""), "non-empty string")
    expect_error(lambda: protocol.seed_for("two-sum", -1), "worker index")
    expect_error(lambda: protocol.stream("two-sum", 0, -1), "negative")

    # Boundary: a one-draw protocol digest is still well defined and distinct per benchmark.
    tiny = {benchmark: protocol.digest([benchmark], count=1) for benchmark in BENCHMARKS[:3]}
    check(len(set(tiny.values())) == 3, "single-draw digests must differ per benchmark")
    check(protocol.digest([], count=1) == sha256_text(""), "an empty suite hashes the empty string")

    print("seeds=%d digest=%s worst_chi=%.2f (%s) worst_drift=%.4f"
          % (len(seeds), digest[:16], worst_chi, worst_benchmark, worst_drift))
    print("PASS")


if __name__ == "__main__":
    main()
