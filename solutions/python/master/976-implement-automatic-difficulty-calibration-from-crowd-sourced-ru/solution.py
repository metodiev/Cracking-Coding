"""Task 976 - Implement automatic difficulty calibration from crowd-sourced runtimes.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Crowd-sourced runtimes arrive from machines of very different speed, so a raw runtime says more about the
machine than about the problem.  The calibrator works in log space: it estimates each machine's offset from
the shared problems (the median of that machine's per-problem medians), subtracts it, trims samples that are
more than four scaled MADs from the problem's centre, and takes the trimmed median as the problem's
difficulty.  A percentile cut table then turns the calibrated difficulties into bands (easy, medium, hard,
expert).  Because the median and the MAD are robust, a handful of aborted or throttled runs cannot move a
problem's band, whereas the mean would follow them.

Simplified versus a production calibrator: one global machine offset instead of per-language or per-input
models, a fixed log-normal noise model and a fixed cut table.  Verified: the recovered machine offsets
against the planted ones, machine medians equalized by normalization, the difficulty order against the
planted ground truth, the bands against both a counting oracle and an ideal-world (noise-free) ranking,
stability under a mean-based offset estimator, outlier robustness (median vs mean drift), trimming
accounting, single-sample and constant-value boundaries, order invariance and reproducibility.
"""
PARAMETERS = {"machines": 5, "problems": 12, "samples": 4}
MACHINE_SPEED = {"m1": 0.0, "m2": -0.45, "m3": 0.5, "m4": -0.2, "m5": 0.3}   # planted log offsets
PROBLEMS = tuple(f"p{index:02d}" for index in range(1, PARAMETERS["problems"] + 1))
DIFFICULTY = {problem: round(0.2 * index, 3) for index, problem in enumerate(PROBLEMS)}  # ground truth
CUTS, BANDS = (0.30, 0.65, 0.88), ("easy", "medium", "hard", "expert")
MASK64 = (1 << 64) - 1
NOISE = 0.15   # half-range of the uniform runtime noise in log space


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64 - a fixed seed makes the whole crowd-sourced table reproducible."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def below(self, bound):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return (value ^ (value >> 31)) % bound


def median(values):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    return float(ordered[middle]) if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def mad(values):
    """Median absolute deviation - the robust spread behind the trimming rule."""
    if not values:
        return 0.0
    return median([abs(value - median(values)) for value in values])


def mean(values):
    return sum(values) / len(values) if values else 0.0


def generate(seed=2026, samples=PARAMETERS["samples"]):
    """(machine, problem) -> tuple of log runtimes: ground-truth difficulty + machine speed + noise."""
    rng = Rng(seed)
    table = {}
    for machine, speed in MACHINE_SPEED.items():
        for problem in PROBLEMS:
            table[(machine, problem)] = tuple(DIFFICULTY[problem] + speed
                                              + (rng.below(1000) / 1000.0 - 0.5) * 2 * NOISE
                                              for _ in range(samples))
    return table


def machine_offsets(table, combiner=median, machines=None):
    """Per-machine log offset, estimated from the shared problems and centred on the global median."""
    medians = {(machine, problem): median(samples) for (machine, problem), samples in table.items()}
    reference = median(list(medians.values())) if medians else 0.0
    return {machine: combiner([value for (owner, _problem), value in medians.items() if owner == machine])
            - reference for machine in (machines or MACHINE_SPEED)}


def machine_medians(table):
    return {machine: median([median(samples) for (owner, _problem), samples in table.items()
                             if owner == machine]) for machine in MACHINE_SPEED}


def robust_center(values, sigma=4.0):
    """(median, kept, dropped): samples further than `sigma` scaled MADs from the centre are trimmed."""
    center, spread = median(values), mad(values)
    if spread == 0:
        return center, list(values), []
    kept, dropped = [], []
    for value in values:
        (kept if abs(value - center) <= sigma * 1.4826 * spread else dropped).append(value)
    return center, kept, dropped


def calibrate(table, combiner=median):
    """offsets, problem -> (trimmed median difficulty, kept count, dropped count)."""
    offsets = machine_offsets(table, combiner)
    normalized = {}
    for (machine, problem), samples in table.items():
        normalized.setdefault(problem, []).extend(value - offsets[machine] for value in samples)
    report = {}
    for problem in sorted(normalized):
        center, kept, dropped = robust_center(normalized[problem])
        report[problem] = (median(kept) if kept else center, len(kept), len(dropped))
    return offsets, report


def assign_bands(values, cuts=CUTS, names=BANDS):
    """Percentile bands: a problem's share is the midpoint of the tie group it falls into."""
    total = len(values)
    bands = {}
    for key, value in values.items():
        below = sum(1 for other, other_value in values.items() if other_value < value)
        equal = sum(1 for other, other_value in values.items() if other_value == value)
        share = (below + equal / 2) / total if total else 0.0
        bands[key] = names[min(sum(1 for cut in cuts if share >= cut), len(names) - 1)]
    return bands


def bands_oracle(values, cuts=CUTS, names=BANDS):
    """Independent oracle: place each value by scanning the sorted array instead of counting keys."""
    ordered = sorted(values.values())
    total = len(ordered)
    bands = {}
    for key, value in values.items():
        below = sum(1 for other in ordered if other < value)
        at_or_below = sum(1 for other in ordered if other <= value)
        share = (below + at_or_below) / 2 / total if total else 0.0
        bands[key] = names[min(sum(1 for cut in cuts if share >= cut), len(names) - 1)]
    return bands


def inject_outliers(table, seed=5, count=6):
    """Simulate aborted/throttled runs: a few samples come back 100x slower."""
    rng = Rng(seed)
    keys = sorted(table)
    mutated = dict(table)
    for key in [keys[rng.below(len(keys))] for _ in range(count)]:
        samples = list(mutated[key])
        samples[rng.below(len(samples))] *= 100.0
        mutated[key] = tuple(samples)
    return mutated


def main():
    table = generate()
    before = machine_medians(table)
    offsets, report = calibrate(table)
    # 1) the estimator recovers the planted machine speeds, and centres them on the crowd.
    check(all(abs(offsets[machine] - MACHINE_SPEED[machine]) < 0.06 for machine in MACHINE_SPEED)
          and abs(median(list(offsets.values()))) < 0.02
          and max(before.values()) - min(before.values()) > 0.3,
          f"machine offsets are wrong: {offsets} (planted {MACHINE_SPEED})")
    # 2) normalization equalizes the machines and preserves the planted difficulty order.
    spread = max(report[problem][0] - DIFFICULTY[problem] for problem in PROBLEMS)
    check(all(abs(report[problem][0] - DIFFICULTY[problem]) < 0.06 for problem in PROBLEMS)
          and spread < 0.06 and sorted(PROBLEMS, key=lambda key: report[key][0]) == list(PROBLEMS),
          f"calibrated difficulties do not track the ground truth: {[round(report[p][0], 3) for p in PROBLEMS]}")
    # 3) bands agree with the counting oracle and with an ideal-world ranking of the planted truth.
    calibrated = {problem: report[problem][0] for problem in PROBLEMS}
    bands = assign_bands(calibrated)
    order = sorted(PROBLEMS, key=lambda problem: calibrated[problem])
    check(bands == bands_oracle(calibrated) and bands == assign_bands(DIFFICULTY)
          and list(bands.values()).count("easy") == 4 and list(bands.values()).count("expert") == 1
          and [BANDS.index(bands[problem]) for problem in order] == sorted(BANDS.index(bands[p]) for p in order),
          f"band assignment is wrong: {bands}")
    # 4) stability: a mean-based offset estimator (a different route) lands on the same bands.
    mean_offsets, mean_report = calibrate(table, combiner=mean)
    check(all(abs(mean_offsets[machine] - MACHINE_SPEED[machine]) < 0.08 for machine in MACHINE_SPEED)
          and assign_bands({p: mean_report[p][0] for p in PROBLEMS}) == bands,
          f"bands are not stable under the alternative estimator: {mean_offsets}")
    # 5) robustness: six throttled samples barely move the median (and never the band) but wreck the mean.
    noisy = inject_outliers(table)
    _noisy_offsets, noisy_report = calibrate(noisy)
    raw_problem = PROBLEMS[5]
    raw_values = [value for (machine, problem), samples in noisy.items() if problem == raw_problem
                  for value in samples]
    clean_values = [value for (machine, problem), samples in table.items() if problem == raw_problem
                    for value in samples]
    check(abs(noisy_report[raw_problem][0] - report[raw_problem][0]) < 0.05
          and assign_bands({p: noisy_report[p][0] for p in PROBLEMS}) == bands
          and mean(raw_values) - mean(clean_values) > 0.3
          and noisy_report[raw_problem][2] >= 1,
          f"the median is not robust: {noisy_report[raw_problem]} vs {report[raw_problem]}")
    # 6) trimming accounting: clean data drops nothing, the outlier run drops exactly what was injected.
    total_samples = sum(len(samples) for samples in table.values())
    check(sum(kept for _center, kept, _dropped in report.values()) == total_samples
          and sum(dropped for _center, _kept, dropped in report.values()) == 0
          and sum(dropped for _center, _kept, dropped in noisy_report.values()) == 6
          and all(noisy_report[problem][1] + noisy_report[problem][2]
                  == sum(len(samples) for (_machine, owner), samples in noisy.items() if owner == problem)
                  for problem in PROBLEMS),
          f"trimming accounting is wrong: {sum(dropped for _, _, dropped in noisy_report.values())} dropped")
    # 7) boundaries: a single sample per problem, constant runtimes, and an empty table.
    single = {(machine, problem): (DIFFICULTY[problem] + MACHINE_SPEED[machine],)
              for machine in MACHINE_SPEED for problem in PROBLEMS}
    _single_offsets, single_report = calibrate(single)
    flat = {key: (1.5, 1.5, 1.5) for key in table}
    _flat_offsets, flat_report = calibrate(flat)
    check(all(single_report[problem][1] == len(MACHINE_SPEED) and single_report[problem][2] == 0
              for problem in PROBLEMS)
          and max(abs(single_report[problem][0] - DIFFICULTY[problem]) for problem in PROBLEMS) < 0.05
          and sum(dropped for _c, _k, dropped in flat_report.values()) == 0
          and len(set(assign_bands({p: flat_report[p][0] for p in PROBLEMS}).values())) == 1
          and calibrate({})[1] == {} and machine_medians({}) == {m: 0.0 for m in MACHINE_SPEED}
          and assign_bands({}) == {},
          "the single-sample, constant and empty boundaries are wrong")
    # 8) order invariance and reproducibility: shuffling samples and re-running changes nothing.
    shuffled = {key: tuple(reversed(samples)) for key, samples in table.items()}
    _shuffled_offsets, shuffled_report = calibrate(shuffled)
    check(shuffled_report == report and assign_bands({p: shuffled_report[p][0] for p in PROBLEMS}) == bands
          and generate() == table and calibrate(generate())[1] == report
          and inject_outliers(table) == noisy,
          "calibration must be order invariant and reproducible")
    # 9) crowd accounting: every problem has machine* samples, all accounted for as kept or dropped.
    check(len(table) == len(MACHINE_SPEED) * len(PROBLEMS)
          and sorted(report) == list(PROBLEMS)
          and all(report[p][1] == len(MACHINE_SPEED) * PARAMETERS["samples"] for p in PROBLEMS)
          and len(offsets) == PARAMETERS["machines"],
          f"crowd accounting is wrong: {len(table)} rows")
    print(f"machines={len(MACHINE_SPEED)} problems={len(PROBLEMS)} samples={total_samples} "
          f"max_offset_error={max(abs(offsets[m] - MACHINE_SPEED[m]) for m in MACHINE_SPEED):.4f}")
    print(f"bands={{{', '.join(f'{p}:{bands[p]}' for p in PROBLEMS)}}} "
          f"dropped_outliers={sum(dropped for _c, _k, dropped in noisy_report.values())}")
    print("PASS")


if __name__ == "__main__":
    main()
