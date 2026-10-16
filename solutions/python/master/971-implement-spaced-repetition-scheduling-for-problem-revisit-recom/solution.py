"""Task 971 - Implement spaced-repetition scheduling for problem revisit recommendations.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

An SM-2 scheduler over a small set of solved problems.  Each card keeps an ease factor (starts at 2.5,
floor 1.3), a repetition count and an interval; a grade 0-5 moves it: grades below 3 reset the repetition
chain and re-show the problem the next day, otherwise the interval becomes 1 day, then 6 days, then
round(interval * ease), and the ease factor is nudged by 0.1 - (5-q) * (0.08 + 0.02 * (5-q)).  Replaying a
learner's review history yields per-problem due dates, a due queue ordered by (due day, problem) and a
revisit plan for a 40-day window; overdue problems are flagged so the recommendation list can surface them.

Simplified versus Anki/FSRS: no learning steps, no fuzz, no lapses counter and no per-learner lapse
priors - one deterministic SM-2 chain per problem.  Verified: the published SM-2 reference sequence
(1, 6, 16, 45 days), an exact-rational oracle using Fractions and half-up rounding for the intervals,
failure resets and the 1.3 ease floor, the brand-new-card and empty-history boundaries, queue ordering,
the window plan against a day-by-day scan, grade monotonicity, and reproducibility.
"""
import re
from fractions import Fraction

DEFAULT_EF = 2.5
MIN_EF = 1.3
WINDOW = 40
HISTORIES = {
    101: ((0, 5), (1, 5), (7, 5), (23, 5)),
    102: ((0, 5), (1, 4), (7, 3), (21, 2)),
    103: ((0, 3), (1, 3)),
    104: ((0, 4),),
    105: ((0, 5), (1, 5)),
    106: ((0, 2), (1, 2), (2, 2)),
    107: (),
}
FRESH = (DEFAULT_EF, 0, 0)  # ease factor, repetitions, interval


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def grade_card(state, grade, day):
    """One SM-2 step.  Returns the new state and the day the problem becomes due again."""
    ease, repetitions, interval = state
    if grade < 3:
        repetitions, interval = 0, 1
    else:
        repetitions += 1
        interval = 1 if repetitions == 1 else (6 if repetitions == 2 else round(interval * ease))
    ease = max(MIN_EF, ease + 0.1 - (5 - grade) * (0.08 + 0.02 * (5 - grade)))
    return (ease, repetitions, interval), day + interval


def replay(history, start=0):
    """Review the card at each (day, grade) in order; each trace row is (ease, reps, interval, due)."""
    state, trace = FRESH, []
    for day, grade in history:
        state, due = grade_card(state, grade, day)
        trace.append(state + (due,))
    return trace


def half_up(value):
    return int(value + Fraction(1, 2))


def interval_oracle(grades):
    """Independent oracle: exact rational arithmetic with explicit half-up rounding."""
    ease, repetitions, interval = Fraction(5, 2), 0, 0
    intervals = []
    for grade in grades:
        if grade < 3:
            repetitions, interval = 0, 1
        else:
            repetitions += 1
            interval = 1 if repetitions == 1 else (6 if repetitions == 2 else half_up(interval * ease))
        ease = max(Fraction(13, 10), ease + Fraction(1, 10)
                   - (5 - grade) * (Fraction(8, 100) + Fraction(2, 100) * (5 - grade)))
        intervals.append(interval)
    return intervals


def due_queue(histories):
    """[(due day, problem, state)] ordered by due day, then by problem number."""
    queue = []
    for number, history in sorted(histories.items()):
        state, due = FRESH, 0
        for day, grade in history:
            state, due = grade_card(state, grade, day)
        queue.append((due, number, state))
    return sorted(queue)


def plan(histories, start=0, horizon=WINDOW):
    """day -> (problems due that day,) for every due day inside the window."""
    days = {}
    for due, number, _state in due_queue(histories):
        if start <= due < start + horizon:
            days.setdefault(due, []).append(number)
    return {day: tuple(numbers) for day, numbers in sorted(days.items())}


def plan_oracle(histories, start=0, horizon=WINDOW):
    """Independent oracle: scan every day of the window and ask which problems fall due there."""
    found = {}
    for day in range(start, start + horizon):
        due_today = []
        for number in sorted(histories):
            state, due = FRESH, 0
            for review_day, grade in histories[number]:
                state, due = grade_card(state, grade, review_day)
            if due == day:
                due_today.append(number)
        if due_today:
            found[day] = tuple(due_today)
    return found


def overdue(queue, today):
    return [(due, number) for due, number, _state in queue if due < today]


def perfect_chain_due(reviews, start=0):
    """Closed form via the rational oracle: the due day after `reviews` consecutive grade-5 reviews."""
    return start + sum(interval_oracle([5] * reviews))


def perfect_chain_history(reviews, grade=5):
    """Review days of a perfect chain: review i happens on the due day left by review i-1."""
    return tuple((sum(interval_oracle([5] * step)), grade) for step in range(reviews))


def render(history):
    """Compact human-readable form of a card's history, used by the diagnostic line."""
    return ",".join(f"{day}:{grade}" for day, grade in history) or "-"


def main():
    chain = replay([(0, 5), (1, 5), (7, 5), (23, 5)])
    # 1) published SM-2 reference chain for four perfect recalls: intervals 1, 6, 16, 45 days.
    check([row[2] for row in chain] == [1, 6, 16, 45] and [row[3] for row in chain] == [1, 7, 23, 68]
          and all(abs(row[0] - expected) < 1e-9 for row, expected in zip(chain, (2.6, 2.7, 2.8, 2.9)))
          and [row[1] for row in chain] == [1, 2, 3, 4],
          f"the SM-2 reference chain is wrong: {chain}")
    # 2) independent oracle: exact rational arithmetic with half-up rounding, on two grade sequences.
    for grades in ([5, 5, 5, 5], [5, 4, 5, 3, 5], [3, 3, 3], [4, 0, 5, 2, 5, 5]):
        trace = replay(list(zip(range(len(grades)), grades)))
        check([row[2] for row in trace] == interval_oracle(grades),
              f"intervals for {grades} disagree with the rational oracle: {[row[2] for row in trace]}")
    # 3) failure: a grade below 3 restarts the chain, and the ease factor bottom-clamps at 1.3.
    failed = replay([(0, 2)])[0]
    floored = replay([(index, 0) for index in range(6)])[-1]
    check((failed[1], failed[2], failed[3]) == (0, 1, 1) and abs(failed[0] - 2.18) < 1e-9
          and abs(floored[0] - MIN_EF) < 1e-12 and all(row[2] == 1 for row in replay([(i, 0) for i in range(4)]))
          and replay([(0, 5)])[0][0] > replay([(0, 3)])[0][0],
          f"failure handling is wrong: {failed} / {floored}")
    # 4) boundary: a brand-new card is due today with a one-day interval and the default ease.
    new_card = replay([])
    check(new_card == [] and grade_card(FRESH, 5, 0) == ((2.6, 1, 1), 1)
          and due_queue({}) == [] and plan({}) == {} and overdue([], 5) == []
          and due_queue({107: ()}) == [(0, 107, FRESH)],
          "the brand-new-card and empty-history boundaries are wrong")
    # 5) due queue: ordered by (due day, problem), one row per problem, and overdue rows are exact.
    queue = due_queue(HISTORIES)
    check([row[0] for row in queue] == [0, 1, 3, 7, 7, 22, 68]
          and [row[1] for row in queue] == [107, 104, 106, 103, 105, 102, 101]
          and queue[0][2] == FRESH and [due for due, _n, _s in queue] == sorted(due for due, _n, _s in queue)
          and overdue(queue, 8) == [(0, 107), (1, 104), (3, 106), (7, 103), (7, 105)],
          f"the due queue is wrong: {[(due, number) for due, number, _ in queue]}")
    # 6) window plan: equals the day-by-day scan oracle, never repeats a problem within a day.
    plan_window = plan(HISTORIES)
    check(plan_window == plan_oracle(HISTORIES)
          and plan_window == {0: (107,), 1: (104,), 3: (106,), 7: (103, 105), 22: (102,)}
          and all(len(numbers) == len(set(numbers)) for numbers in plan_window.values())
          and sorted(plan_window) == sorted({due for due, _n, _s in queue if due < WINDOW}),
          f"the revisit plan is wrong: {plan_window}")
    # 7) independent oracle: the due day of a perfect chain equals the rational interval prefix sum.
    for reviews in (1, 2, 3, 4):
        check(due_queue({1: perfect_chain_history(reviews)})[0][0] == perfect_chain_due(reviews),
              f"perfect chain of {reviews} reviews must be due on day {perfect_chain_due(reviews)}")
    # 8) property: better grades give a longer interval - monotone in grade quality, never decreasing.
    for reviews in (2, 3, 4):
        good = replay(perfect_chain_history(reviews))
        poor = replay(perfect_chain_history(reviews, 3))
        check(good[-1][3] >= poor[-1][3] and good[-1][0] > poor[-1][0]
              and all(good[i][2] >= good[i - 1][2] for i in range(1, len(good))),
              f"grade monotonicity fails at {reviews} reviews: {good[-1]} vs {poor[-1]}")
    # 9) determinism: replaying, queueing and planning the same input twice is byte-identical.
    check(replay([(0, 5), (1, 5), (7, 5), (23, 5)]) == chain and due_queue(HISTORIES) == queue
          and plan(HISTORIES) == plan_window and interval_oracle([5, 5, 5, 5]) == interval_oracle([5, 5, 5, 5])
          and re.sub(r" ", "", render(HISTORIES[101])) == "0:5,1:5,7:5,23:5",
          "the scheduler must be reproducible")
    due_soon = [number for _due, number, _state in queue if _due <= 8]
    print(f"problems={len(HISTORIES)} window={WINDOW} due_within_week={due_soon} "
          f"plan_days={sorted(plan_window)} overdue_at_day8={len(overdue(queue, 8))}")
    print(f"reference_intervals={[row[2] for row in chain]} reference_ease="
          f"{[round(row[0], 3) for row in chain]} perfect_due={[perfect_chain_due(n) for n in (1, 2, 3, 4)]}")
    print("PASS")


if __name__ == "__main__":
    main()
