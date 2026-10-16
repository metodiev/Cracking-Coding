"""Task 960 - Implement gamification layer with badges for completing tiers and streaks.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Gamification model: completion events `(user, task, day)` feed three rule families - tier badges
awarded when every task of a tier has been completed, streak badges awarded when the user's longest
run of consecutive distinct days reaches a threshold, and volume badges for distinct tasks solved.
Awarding is a pure function of the event log plus the badges already held, which makes it idempotent
(replaying the log never duplicates a badge) and monotone (an extra event can only add badges).

Simplified: days are integers instead of dates, the catalogue is a handful of tasks and the reward
layer is decoupled from the submission pipeline.  Verified: the acceptance path, the streak boundary
with duplicate days and gaps against a brute-force oracle over the day range, tier completeness
including the missing-task case, idempotency, monotonicity over growing prefixes, the volume
threshold boundary, and the empty/single-event cases.
"""
TIER_TASKS = {"beginner": (1, 2, 3), "intermediate": (4, 5, 6), "master": (7, 8)}
STREAK_THRESHOLD = 5
VOLUME_THRESHOLDS = (1, 10, 30)
TIER_ORDER = ("beginner", "intermediate", "advanced", "expert", "master")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def completions(events, user):
    """Distinct (task, day) pairs for one user, ordered by day then task."""
    return sorted({(task, day) for who, task, day in events if who == user})


def days_of(events, user):
    return sorted({day for _, day in completions(events, user)})


def longest_streak(days):
    """Longest run of consecutive integers in the day set."""
    best = current = 0
    previous = None
    for day in days:
        current = current + 1 if previous is not None and day == previous + 1 else 1
        best = max(best, current)
        previous = day
    return best


def oracle_streak(days):
    """Independent oracle: walk every integer in [min, max] and count the longest run."""
    if not days:
        return 0
    present, best, current = set(days), 0, 0
    for day in range(min(days), max(days) + 1):
        current = current + 1 if day in present else 0
        best = max(best, current)
    return best


def badges(events, user, previous=frozenset()):
    """The badge set after replaying the log, always including `previous` (monotone)."""
    completed = {task for task, _ in completions(events, user)}
    earned = set(previous)
    for tier, tasks in TIER_TASKS.items():
        if set(tasks) <= completed:
            earned.add(f"tier:{tier}")
    streak = longest_streak(days_of(events, user))
    if streak >= STREAK_THRESHOLD:
        earned.add(f"streak:{STREAK_THRESHOLD}")
    for threshold in VOLUME_THRESHOLDS:
        if len(completed) >= threshold:
            earned.add(f"solved:{threshold}")
    return sorted(earned)


def badge_rank(badge):
    """Stable display order: tiers by catalogue order, then streaks, then volumes."""
    family, _, value = badge.partition(":")
    if family == "tier":
        order = TIER_ORDER.index(value) if value in TIER_ORDER else len(TIER_ORDER)
        return (0, order)
    if family == "streak":
        return (1, int(value))
    return (2, int(value))


def progress(events, user):
    """The summary the UI shows: per-tier completion, streak, distinct solved count."""
    completed = {task for task, _ in completions(events, user)}
    return {"solved": len(completed), "streak": longest_streak(days_of(events, user)),
            "tiers": {tier: len(set(tasks) & completed) for tier, tasks in TIER_TASKS.items()}}


def ada_events():
    """Ada completes every catalogued task across days 1..8, with a duplicate on day 3."""
    events = [("ada", task, day) for task, day in
              [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]]
    events.append(("ada", 1, 3))       # replaying a day must not stretch the streak
    events.append(("bob", 99, 2))      # another user's event must not leak in
    return events


def main():
    events = ada_events()

    # 1) happy path: three tiers, one streak badge and the first volume badge - and nothing else.
    earned = set(badges(events, "ada"))
    check(earned == {"tier:beginner", "tier:intermediate", "tier:master", "streak:5", "solved:1"}
          and badges(events, "bob") == ["solved:1"],
          f"ada's badges wrong: {sorted(earned)}")

    # 2) streak boundary: four consecutive days earn nothing, five do, duplicates and gaps break it.
    four = [("u", 1, day) for day in (1, 2, 3, 4)]
    five = four + [("u", 1, 5), ("u", 1, 5)]
    gap = [("u", 1, day) for day in (1, 2, 3, 4, 10)]
    check(longest_streak(days_of(four, "u")) == 4 and "streak:5" not in badges(four, "u")
          and longest_streak(days_of(five, "u")) == 5 and "streak:5" in badges(five, "u")
          and longest_streak(days_of(gap, "u")) == 4,
          f"the streak boundary must be exact: {longest_streak(days_of(five, 'u'))}")

    # 3) independent oracle: a walk over the whole day range agrees with the run scan.
    samples = [(), (7,), (5, 6, 7), (1, 2, 4, 5, 6, 9), (10, 11, 12, 20, 21)]
    check(all(longest_streak(sorted(day for day in sample)) == oracle_streak(sorted(sample))
              for sample in samples)
          and oracle_streak(sorted((5, 6, 7))) == 3 and oracle_streak([]) == 0,
          "the streak scan must agree with the brute-force oracle")

    # 4) tier badges need the complete catalogue: two of three beginner tasks is not a badge.
    partial = [("u", task, 1) for task in (1, 2)]
    check("tier:beginner" not in badges(partial, "u")
          and "tier:beginner" in badges(partial + [("u", 3, 2)], "u")
          and progress(partial, "u")["tiers"]["beginner"] == 2
          and progress(partial + [("u", 3, 2)], "u")["tiers"]["beginner"] == 3,
          f"tier completeness wrong: {progress(partial + [('u', 3, 2)], 'u')}")

    # 5) idempotency: replaying the log, and replaying it into the badge set it just produced, is a
    #    fixed point.
    once = badges(events, "ada")
    check(badges(events, "ada") == once and badges(events, "ada", previous=once) == once
          and badges(events, "ada", previous=frozenset(once)) == once,
          f"awarding must be idempotent: {once}")

    # 6) monotonicity: badges only grow as the log grows (checked over every prefix).
    growing = [("m", task, day) for task, day in
               [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]]
    ladder = [set(badges(growing[:size], "m")) for size in range(len(growing) + 1)]
    check(all(ladder[index] <= ladder[index + 1] for index in range(len(ladder) - 1))
          and ladder[0] == set() and ladder[-1] >= ladder[3],
          f"badges must never be revoked: {[sorted(step) for step in ladder]}")

    # 7) volume boundary: 29 distinct tasks earn no solved:30, the 30th does.
    def volume(count):
        return [("v", task, index) for index, task in enumerate(range(1, count + 1), start=1)]
    check("solved:10" in badges(volume(10), "v") and "solved:10" not in badges(volume(9), "v")
          and "solved:30" in badges(volume(30), "v")
          and "solved:30" not in badges(volume(29), "v")
          and progress(volume(30), "v")["solved"] == 30,
          f"the volume boundary must be exact: {badges(volume(29), 'v')}")

    # 8) boundaries: an empty log earns nothing, a single event earns only the first volume badge.
    check(badges([], "nobody") == [] and badges([("solo", 1, 1)], "solo") == ["solved:1"]
          and longest_streak([]) == 0 and days_of([], "nobody") == []
          and badges([("other", 1, 1)], "solo") == [],
          f"empty/single boundaries wrong: {badges([], 'nobody')}")

    # 9) determinism and order independence: shuffling the log changes nothing, and the rendered
    #    order follows the documented badge ranking.
    shuffled = [events[index] for index in (5, 0, 7, 2, 9, 1, 3, 8, 4, 6)]
    rendered = sorted(once, key=badge_rank)
    check(badges(shuffled, "ada") == once and rendered == ["tier:beginner", "tier:intermediate",
                                                           "tier:master", "streak:5", "solved:1"],
          f"badge computation must be order independent: {rendered}")

    print(f"ada_badges={rendered} ada_progress={progress(events, 'ada')}")
    print(f"streak_five={longest_streak(days_of(five, 'u'))} ladder_sizes="
          f"{[len(step) for step in ladder]} volume30={'solved:30' in badges(volume(30), 'v')}")
    print("PASS")


if __name__ == "__main__":
    main()
