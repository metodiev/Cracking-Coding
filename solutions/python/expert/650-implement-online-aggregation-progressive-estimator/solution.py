"""Task 650 - Implement online aggregation progressive estimator.  (Tier: Expert, Ch 14 — Databases)

Online aggregation: instead of running the query to completion and then reporting a number,
the engine reports COUNT, SUM, AVG (and the exact running MIN/MAX) after *every* tuple and
attaches a confidence interval to the mean.  The interval is Hoeffding's bound for bounded
data, `half_width = span * sqrt(ln(2/alpha) / (2n))`, where `span` is metadata about the value
domain and n the number of tuples accepted so far - so it needs no assumption about the
distribution, only boundedness, and it shrinks like 1/sqrt(n).

The stream arrives in a deterministic pseudo-random order (a seeded Fisher-Yates
permutation), which is what makes the sampling interpretation of the first n tuples valid and
therefore makes the interval honest.  Because the interval narrows quickly, a consumer that
only needs 10% accuracy can stop early; `run_until` implements that stopping rule and reports
how many tuples it consumed.

Invariants verified: the running COUNT/SUM/MIN/MAX are exact at every checkpoint, the
interval always brackets its own estimate and shrinks monotonically, a completed pass is
exact, the interval contains the true mean for the seeded permutations (checked over many
seeds), early stopping needs far fewer tuples than a full scan, and everything is
deterministic.
"""

import math

MASK = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return (z ^ (z >> 31)) & MASK

    def below(self, bound):
        return self.next() % bound


def shuffled_stream(size, seed):
    """A deterministic Fisher-Yates permutation: random arrival order, fixed per seed."""
    values = list(range(size))
    rng = SplitMix64(seed)
    for index in range(size - 1, 0, -1):
        other = rng.below(index + 1)
        values[index], values[other] = values[other], values[index]
    return values


def half_width(span, count, alpha=0.05):
    if count <= 0:
        return float("inf")
    return span * math.sqrt(math.log(2.0 / alpha) / (2 * count))


class OnlineAggregation:
    """Progressive estimator: exact counters plus a confidence interval on the mean."""

    def __init__(self, span, alpha=0.05):
        self.span = span
        self.alpha = alpha
        self.count = 0
        self.total = 0.0
        self.minimum = None
        self.maximum = None

    def accept(self, value):
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None or value < self.minimum else self.minimum
        self.maximum = value if self.maximum is None or value > self.maximum else self.maximum
        return self.snapshot()

    def snapshot(self):
        mean = self.total / self.count if self.count else 0.0
        width = half_width(self.span, self.count, self.alpha)
        return {"count": self.count, "sum": self.total, "avg": mean, "half_width": width,
                "low": mean - width, "high": mean + width,
                "min_seen": self.minimum, "max_seen": self.maximum}

    def progressive(self, stream, checkpoints):
        """Yield (tuples consumed, snapshot) at each checkpoint; stops after the last one."""
        schedule = sorted({point for point in checkpoints if point > 0})
        consumed = 0
        for value in stream:
            snapshot = self.accept(value)
            consumed += 1
            if schedule and consumed == schedule[0]:
                yield consumed, snapshot
                schedule.pop(0)
                if not schedule:
                    return

    def run_until(self, stream, tolerance, max_tuples=None):
        """Stop as soon as the relative half width fits `tolerance`; (snapshot, consumed, met)."""
        consumed, snapshot, met = 0, self.snapshot(), False
        for value in stream:
            consumed += 1
            snapshot = self.accept(value)
            if snapshot["avg"] and snapshot["half_width"] <= tolerance * abs(snapshot["avg"]):
                return snapshot, consumed, True
            if max_tuples is not None and consumed >= max_tuples:
                break
        return snapshot, consumed, met


def prefix_stats(values):
    """Exact prefix aggregates: the oracle for the running counters."""
    out, total = [], 0.0
    for index, value in enumerate(values):
        total += value
        previous = out[-1] if out else None
        out.append({"count": index + 1, "sum": total,
                    "min": value if previous is None else min(previous["min"], value),
                    "max": value if previous is None else max(previous["max"], value)})
    return out


def main():
    size = 4096
    stream = shuffled_stream(size, 20240601)
    truth = sum(range(size)) / size
    span = size - 1
    check(stream == shuffled_stream(size, 20240601), "the stream is a deterministic permutation")
    check(sorted(stream) == list(range(size)), "the permutation contains every value once")
    check(truth == 2047.5, "the population mean is known exactly: %r" % (truth,))
    exact = prefix_stats(stream)
    check(exact[0]["count"] == 1, "the prefix oracle starts with one tuple")

    checkpoints = [2 ** power for power in range(4, 13)]        # 16, 32, ... 4096
    snapshots = list(OnlineAggregation(span).progressive(iter(stream), checkpoints))
    check([count for count, _ in snapshots] == checkpoints,
          "one snapshot per checkpoint, in order: %r" % ([c for c, _ in snapshots],))
    check(len(snapshots) == 9, "the geometric schedule yields nine checkpoints")

    previous_width = float("inf")
    for count, snapshot in snapshots:
        oracle = exact[count - 1]
        check(snapshot["count"] == count, "the running count is exact at %d" % count)
        check(snapshot["sum"] == oracle["sum"], "the running sum is exact at %d" % count)
        check(snapshot["min_seen"] == oracle["min"] and snapshot["max_seen"] == oracle["max"],
              "running MIN/MAX are exact at %d" % count)
        check(snapshot["avg"] == snapshot["sum"] / snapshot["count"],
              "the average is the running sum over the running count")
        check(snapshot["low"] <= snapshot["avg"] <= snapshot["high"],
              "the interval brackets its own estimate at %d" % count)
        check(snapshot["half_width"] < previous_width,
              "the interval shrinks monotonically: %.2f then %.2f"
              % (previous_width, snapshot["half_width"]))
        check(snapshot["half_width"] == half_width(span, count),
              "the width follows the Hoeffding formula at %d" % count)
        previous_width = snapshot["half_width"]

    final = snapshots[-1][1]
    check(final["count"] == size and final["avg"] == truth,
          "a completed pass is exact: %.4f" % final["avg"])
    check(final["low"] <= truth <= final["high"], "the final interval contains the true mean")

    # accuracy along the way ---------------------------------------------------------------
    for count, snapshot in snapshots:
        check(abs(snapshot["avg"] - truth) <= 3 * snapshot["half_width"],
              "the estimate at %d is within three half widths of the truth" % count)
    quarter = snapshots[2][1]
    half = snapshots[5][1]
    check(abs(quarter["avg"] - truth) / truth < 0.15,
          "25%% of the stream gives %.2f%% relative error"
          % (abs(quarter["avg"] - truth) / truth * 100))
    check(abs(half["avg"] - truth) / truth < 0.05,
          "50%% of the stream gives %.2f%% relative error"
          % (abs(half["avg"] - truth) / truth * 100))
    check(abs(quarter["avg"] - truth) > 0, "an intermediate estimate is a real estimate")

    # the interval contains the true mean for other arrival orders too ---------------------
    failures, trials = 0, 30
    for trial in range(trials):
        other_stream = shuffled_stream(size, 9000 + trial * 977)
        estimator = OnlineAggregation(span)
        picked = None
        for count, snapshot in estimator.progressive(iter(other_stream), [size // 5]):
            picked = snapshot
        if not (picked["low"] <= truth <= picked["high"]):
            failures += 1
    check(failures <= 1,
          "the interval missed the truth %d times out of %d (alpha = 0.05)" % (failures, trials))

    # early stopping -----------------------------------------------------------------------
    cheap, consumed_cheap, met_cheap = OnlineAggregation(span).run_until(iter(stream), 0.10)
    check(met_cheap and consumed_cheap < size // 3,
          "10%% accuracy needed %d of %d tuples" % (consumed_cheap, size))
    check(cheap["low"] <= truth <= cheap["high"], "the early-stopped interval contains the truth")
    tight, consumed_tight, met_tight = OnlineAggregation(span).run_until(iter(stream), 0.01)
    check(not met_tight and consumed_tight == size,
          "1%% accuracy cannot be reached within this stream (%d tuples)" % consumed_tight)
    check(consumed_cheap < consumed_tight, "a looser tolerance stops earlier")

    pulled = {"count": 0}

    def counting(source, stats):
        for value in source:
            stats["count"] += 1
            yield value

    _, consumed, met = OnlineAggregation(span).run_until(counting(iter(stream), pulled), 0.10)
    check(met and pulled["count"] == consumed < size,
          "the source was abandoned after %d of %d tuples: %d never generated"
          % (consumed, size, size - consumed))

    # the confidence level widens the interval --------------------------------------------
    strict = OnlineAggregation(span, alpha=0.01)
    strict_snapshot = None
    for _, strict_snapshot in strict.progressive(iter(stream), [size]):
        pass
    loose = snapshots[-1][1]
    check(strict_snapshot["half_width"] > loose["half_width"],
          "alpha 0.01 needs a wider interval than 0.05: %.2f > %.2f"
          % (strict_snapshot["half_width"], loose["half_width"]))
    check(strict_snapshot["count"] == size and strict_snapshot["avg"] == truth,
          "the tighter confidence level still reports the exact completed result")

    # determinism and the empty stream -----------------------------------------------------
    repeat = list(OnlineAggregation(span).progressive(iter(stream), checkpoints))
    check(repeat == snapshots, "progressive estimation is deterministic")
    empty = OnlineAggregation(span).snapshot()
    check(empty["count"] == 0 and empty["half_width"] == float("inf"),
          "before any tuple arrives the interval is vacuous, not wrong")
    check(list(OnlineAggregation(span).progressive(iter([]), checkpoints)) == [],
          "an empty stream produces no checkpoint")
    print("[650] online aggregation: exact at n=%d, 10%% accuracy after %d of %d tuples"
          % (size, consumed_cheap, size))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
    print("PASS")
