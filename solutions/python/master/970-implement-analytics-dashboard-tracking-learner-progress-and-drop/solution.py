"""Task 970 - Implement analytics dashboard tracking learner progress and drop-off.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

An in-memory progress dashboard over a deterministic synthetic telemetry log (24 learners, 10 days, one
attempt event per line with problem, minutes and outcome).  It computes the metrics a learner dashboard
actually shows: daily active learners, a solve funnel with stage conversion, per-day drop-off (learners
lost over the active base), a cohort retention curve, per-learner solved counts, robust median
time-to-solve, the stuck-learner list and a robust MAD z-score alert that flags step changes in the
drop-off series but stays quiet on a flat one.

Simplified versus a real warehouse: the log lives in memory and is generated from a fixed seed instead of
read from storage, days are integers rather than timestamps and there is no per-cohort regression model.
Verified against independent oracles: daily actives and drop-off by a brute-force scan, the funnel by
nested loops over learners, the median by `statistics.median`, retention by explicit cohort arithmetic,
plus the alert's degenerate-MAD boundary, empty telemetry and end-to-end determinism.
"""
import statistics
from collections import namedtuple

LEARNERS = tuple(f"L{index:02d}" for index in range(1, 25))
DAYS = tuple(range(10))
PROBLEMS = tuple(range(101, 116))
CHURNED = ("L21", "L22", "L23", "L24")  # active through day 6, then gone: the planted drop-off spike
CHURN_DAY = 6
FUNNEL_THRESHOLDS = (1, 3, 5, 10)
MASK64 = (1 << 64) - 1

Event = namedtuple("Event", "learner day problem minutes solved")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64 - fixed seed, so the generated telemetry is exactly reproducible."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def below(self, bound):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return (value ^ (value >> 31)) % bound


def generate(seed):
    """24 learners over 10 days; the churn cohort is forced active on its last day."""
    rng = Rng(seed)
    events = []
    for learner in LEARNERS:
        skill = 35 + rng.below(60)      # percent chance an attempt succeeds
        activity = 40 + rng.below(55)   # percent chance the learner shows up on a day
        last_day = CHURN_DAY if learner in CHURNED else DAYS[-1]
        for day in DAYS:
            forced = learner in CHURNED and day == CHURN_DAY
            if day > last_day or (not forced and rng.below(100) >= activity):
                continue
            for _ in range(1 + rng.below(3)):
                events.append(Event(learner, day, PROBLEMS[rng.below(len(PROBLEMS))],
                                    5 + rng.below(60), rng.below(100) < skill))
    return events


def active_sets(events):
    """day -> set of learners seen that day (a day with no traffic is simply absent)."""
    active = {}
    for event in events:
        active.setdefault(event.day, set()).add(event.learner)
    return active


def daily_active(events):
    return {day: len(learners) for day, learners in active_sets(events).items()}


def daily_active_oracle(events, learners=LEARNERS, days=DAYS):
    """Brute force: for every learner/day pair, scan the whole log."""
    return {day: sum(1 for learner in learners
                     if any(e.learner == learner and e.day == day for e in events)) for day in days}

def solved_counts(events, learners=LEARNERS):
    solved = {learner: set() for learner in learners}
    for event in events:
        if event.solved:
            solved[event.learner].add(event.problem)
    return {learner: len(problems) for learner, problems in solved.items()}


def solved_counts_oracle(events, learners=LEARNERS):
    """Independent oracle: for each learner, walk the log and keep a list of distinct solves."""
    counts = {}
    for learner in learners:
        distinct = []
        for event in events:
            if event.learner == learner and event.solved and event.problem not in distinct:
                distinct.append(event.problem)
        counts[learner] = len(distinct)
    return counts


def funnel(events, thresholds=FUNNEL_THRESHOLDS):
    """[(threshold, learners reaching it, conversion from the previous stage)]."""
    counts = solved_counts(events)
    stages, previous = [], None
    for threshold in thresholds:
        reached = {learner for learner, count in counts.items() if count >= threshold}
        rate = 1.0 if previous is None else (len(reached) / len(previous) if previous else 0.0)
        stages.append((threshold, reached, rate))
        previous = reached
    return stages


def median(values):
    return float(statistics.median(values)) if values else 0.0


def mad(values):
    """Median absolute deviation - the robust spread used by the alerting rule."""
    if not values:
        return 0.0
    return float(statistics.median([abs(value - median(values)) for value in values]))


def drop_off(events):
    """day -> (learners lost before the next observed day, active base on that day)."""
    active = active_sets(events)
    days = sorted(active)
    return {day: (len(active[day] - active.get(day + 1, set())), len(active[day])) for day in days[:-1]}


def drop_off_rates(events):
    return {day: lost / base for day, (lost, base) in drop_off(events).items()}


def last_active(events):
    seen = {}
    for event in events:
        seen[event.learner] = max(seen.get(event.learner, -1), event.day)
    return seen


def retention(events, cohort_day=0, horizon=6):
    """Share of the day-`cohort_day` cohort still active k days later (k = 0..horizon-1)."""
    active = active_sets(events)
    cohort = active.get(cohort_day, set())
    if not cohort:
        return {step: 0.0 for step in range(horizon)}
    return {step: len(cohort & active.get(cohort_day + step, set())) / len(cohort)
            for step in range(horizon)}


def stuck_learners(events, attempts=5):
    """Learners with at least `attempts` attempts and not a single solve."""
    tries, solved = {}, solved_counts(events)
    for event in events:
        tries[event.learner] = tries.get(event.learner, 0) + 1
    return sorted(learner for learner, count in tries.items() if count >= attempts and not solved[learner])


def solve_times(events):
    return [event.minutes for event in events if event.solved]


def alerts(series, threshold=3.5, window=3):
    """Robust MAD z-score against the preceding `window` values; a zero-MAD step also flags."""
    flagged = []
    for index, value in enumerate(series):
        history = series[max(0, index - window):index]
        if not history:
            continue
        spread = mad(history)
        if spread == 0:
            if value > max(history):
                flagged.append(index)
        elif 0.6745 * (value - median(history)) / spread >= threshold:
            flagged.append(index)
    return flagged


def main():
    events = generate(20261016)
    active, last = active_sets(events), last_active(events)
    # 1) daily actives agree with the brute-force per-learner/per-day scan; no day is phantom.
    dau = daily_active(events)
    check(dau == daily_active_oracle(events) and all(count > 0 for count in dau.values())
          and set(dau) == set(DAYS), f"daily actives are wrong: {dau}")
    # 2) funnel: sizes monotone, stages equal the nested-loop oracle, conversions in (0, 1], boundaries too.
    stages, counts, oracle = funnel(events), solved_counts(events), solved_counts_oracle(events)
    check(counts == oracle and all(len(stages[i + 1][1]) <= len(stages[i][1]) for i in range(len(stages) - 1))
          and all(0.0 < rate <= 1.0 for _, _, rate in stages) and stages[0][2] == 1.0
          and all(len(who) == sum(1 for n in oracle.values() if n >= threshold) for threshold, who, _ in stages)
          and len(funnel(events, (0,))[0][1]) == len(LEARNERS) and funnel(events, (999,))[0][1] == set(),
          f"funnel stages are wrong: {[(t, len(w), round(r, 3)) for t, w, r in stages]}")
    # 3) drop-off: arithmetic matches the brute-force lost sets and the planted churn cohort shows up.
    losses = drop_off(events)
    lost_on_churn = {learner for learner in active.get(CHURN_DAY, set())
                     if learner not in active.get(CHURN_DAY + 1, set())}
    check(losses[CHURN_DAY][0] == len(lost_on_churn) and set(CHURNED) <= lost_on_churn
          and all(last[learner] == CHURN_DAY for learner in CHURNED)
          and all(lost < base for lost, base in losses.values())
          and drop_off_rates(events)[CHURN_DAY] == len(lost_on_churn) / losses[CHURN_DAY][1],
          f"drop-off at day {CHURN_DAY}: {losses.get(CHURN_DAY)} vs {sorted(lost_on_churn)}")
    # 4) statistics: the median matches `statistics.median` on even and odd lengths; MAD degenerates to 0.
    times = solve_times(events)
    check(median(times) == float(statistics.median(times)) and median([7, 3, 9, 5]) == 6.0
          and median([7, 3, 9]) == 7.0 and median([]) == 0.0 and mad([2, 2, 2]) == 0.0,
          f"median time-to-solve is wrong: {median(times)}")
    # 5) alerting: a planted step is flagged, a flat series is quiet, the zero-MAD and empty cases hold.
    series = [drop_off_rates(events).get(day, 0.0) for day in sorted(DAYS)]
    check(alerts([0.2] * 8 + [0.6]) == [8] and alerts([0.2] * 9) == [] and alerts([0.1] * 4 + [0.9]) == [4]
          and alerts([]) == [] and CHURN_DAY in alerts(series),
          f"alerting is wrong: {alerts([0.1] * 4 + [0.9])}")
    # 6) retention pins the day-0 cohort exactly, starts at 1.0 and never leaves [0, 1].
    curve = retention(events)
    brute = {step: len([learner for learner in active[0] if learner in active.get(step, set())])
             for step in range(6)}
    check(curve[0] == 1.0 and all(0.0 <= share <= 1.0 for share in curve.values())
          and all(abs(curve[step] * len(active[0]) - brute[step]) < 1e-12 for step in brute)
          and retention(events, horizon=99)[98] == 0.0 and retention([], horizon=3) == {0: 0.0, 1: 0.0, 2: 0.0},
          f"the retention curve is wrong: {curve}")
    # 7) stuck learners: exactly those with at least five attempts and an empty solved set.
    stuck, attempts = stuck_learners(events), {}
    for event in events:
        attempts[event.learner] = attempts.get(event.learner, 0) + 1
    check(stuck == sorted(learner for learner, n in attempts.items() if n >= 5 and not counts[learner])
          and stuck_learners([], 5) == [], f"stuck-learner detection is wrong: {stuck}")
    # 8) empty telemetry and end-to-end determinism of every metric in the dashboard.
    repeat = generate(20261016)
    check(daily_active([]) == {} and drop_off([]) == {} and drop_off_rates([]) == {} and solve_times([]) == []
          and retention([]) == {k: 0.0 for k in range(6)} and stuck_learners([]) == []
          and all(who == set() for _, who, _ in funnel([])) and list(repeat) == list(events)
          and drop_off_rates(repeat) == drop_off_rates(events) and retention(repeat) == curve
          and daily_active_oracle(repeat) == dau,
          "empty telemetry and reproducibility are broken")
    peak = max(dau, key=lambda day: dau[day])
    print(f"learners={len(LEARNERS)} events={len(events)} peak_dau={dau[peak]}@day{peak} "
          f"funnel={[len(who) for _, who, _ in stages]} churn_lost={len(lost_on_churn)}")
    print(f"drop_off={{{', '.join(f'{d}:{r:.2f}' for d, r in sorted(drop_off_rates(events).items()))}}} "
          f"retention={{{', '.join(f'D{k}:{v:.2f}' for k, v in curve.items())}}} stuck={stuck}")
    print("PASS")


if __name__ == "__main__":
    main()
