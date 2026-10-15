"""Task 332 - Implement calendar booking (MyCalendar I/II/III style). (Tier: Intermediate, CTCI Ch 7)

Three interval-bookkeeping designs over half-open [start, end) bookings:
  * MyCalendar I   - accept only intervals that overlap nothing already accepted.
  * MyCalendarTwo - double booking allowed; reject only a booking that would create a
                     third simultaneous booking (classic singles + overlapping pairs).
  * MyCalendarK   - at most k simultaneous bookings at any instant (III generalisation),
                     maintained with a delta map and a max-prefix sweep.
Every variant is cross-checked against an independent sweep-line peak oracle on
deterministic LCG booking sequences.
"""


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def sweep_peak(intervals):
    """Maximum number of intervals covering any single instant (sweep line)."""
    events = {}
    for start, end in intervals:
        events[start] = events.get(start, 0) + 1
        events[end] = events.get(end, 0) - 1
    current = peak = 0
    for time in sorted(events):
        current += events[time]
        peak = max(peak, current)
    return peak


class MyCalendar:
    """Accepts a booking iff it overlaps no accepted booking."""

    def __init__(self):
        self.accepted = []

    def book(self, start, end):
        if any(overlaps((start, end), other) for other in self.accepted):
            return False
        self.accepted.append((start, end))
        return True


class MyCalendarTwo:
    """Allows double booking: rejects a booking that would create a triple overlap."""

    def __init__(self):
        self.singles = []
        self.pairs = []

    def book(self, start, end):
        for other in self.pairs:
            if overlaps((start, end), other):
                return False
        for other in self.singles:
            if overlaps((start, end), other):
                self.pairs.append((max(start, other[0]), min(end, other[1])))
        self.singles.append((start, end))
        return True

    def max_overlap(self):
        return sweep_peak(self.singles)


class MyCalendarK:
    """At most k simultaneous bookings at any instant."""

    def __init__(self, k):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.delta = {}

    def book(self, start, end):
        self.delta[start] = self.delta.get(start, 0) + 1
        self.delta[end] = self.delta.get(end, 0) - 1
        if self._peak() > self.k:
            self.delta[start] -= 1
            self.delta[end] += 1
            return False
        return True

    def _peak(self):
        current = peak = 0
        for time in sorted(self.delta):
            current += self.delta[time]
            peak = max(peak, current)
        return peak

    def max_overlap(self):
        return self._peak()


def lcg_bookings(count, seed, span=40):
    bookings = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        start = seed % span
        seed = (seed * 75 + 74) % 65537
        bookings.append((start, start + 1 + seed % 9))
    return bookings, seed


def main():
    # MyCalendar I, hand-checked: (10,20) accepted, (15,25) overlaps, (20,30) touches only.
    cal = MyCalendar()
    assert cal.book(10, 20) is True
    assert cal.book(15, 25) is False
    assert cal.book(20, 30) is True
    assert cal.book(5, 10) is True
    assert cal.book(19, 20) is False
    assert cal.book(10, 15) is False
    assert cal.accepted == [(10, 20), (20, 30), (5, 10)]

    # MyCalendar II, the classic LeetCode sequence.
    two = MyCalendarTwo()
    assert [two.book(s, e) for s, e in
            [(10, 20), (50, 60), (10, 40), (5, 15), (5, 10), (25, 55)]] == \
        [True, True, True, False, True, True]
    assert two.max_overlap() == 2
    assert two.book(30, 45) is False          # would triple-cover [30, 40)
    assert two.book(60, 70) is True

    # MyCalendar III with k = 3: the same sequence is fully accepted (peak is exactly 3).
    three = MyCalendarK(3)
    assert [three.book(s, e) for s, e in
            [(10, 20), (50, 60), (10, 40), (5, 15), (5, 10), (25, 55)]] == \
        [True] * 6
    assert three.max_overlap() == 3
    assert three.book(12, 13) is False        # [10, 20), [10, 40), [5, 15) already peak

    # k = 2 variant: the tail bookings never exceed two deep, but (12, 13) would sit on top
    # of (10, 20), (10, 40) and (5, 15) at t = 12.
    two_k = MyCalendarK(2)
    assert [two_k.book(s, e) for s, e in
            [(10, 20), (50, 60), (10, 40), (5, 15), (5, 10), (25, 55)]] == \
        [True, True, True, False, True, True]
    assert two_k.max_overlap() == 2
    assert two_k.book(12, 13) is False
    assert sweep_peak([(5, 15), (10, 20), (10, 40)]) == 3

    assert sweep_peak([]) == 0
    assert sweep_peak([(1, 2)]) == 1
    assert sweep_peak([(1, 5), (2, 6), (3, 7)]) == 3
    assert sweep_peak([(1, 2), (2, 3)]) == 1        # touching is not overlapping

    # LCG cross-check: every book() result must match the independent sweep oracle, and the
    # accept/reject decisions of MyCalendarTwo must equal MyCalendarK(2) on the same input.
    for k in (1, 2, 3):
        seed = 1717
        bookings, seed = lcg_bookings(240, seed)
        cal_k, oracle, k2 = MyCalendarK(k), [], MyCalendarTwo() if k == 2 else None
        for start, end in bookings:
            expected = sweep_peak(oracle + [(start, end)]) <= k
            actual = cal_k.book(start, end)
            assert actual == expected, (k, start, end, oracle)
            if actual:
                oracle.append((start, end))
            if k2 is not None:
                assert k2.book(start, end) == actual, (start, end)
            assert cal_k.max_overlap() == sweep_peak(oracle) <= k
        assert len(oracle) <= len(bookings)

    # MyCalendar I agrees with MyCalendarK(1) and the oracle on the same sequence.
    seed = 2626
    bookings, seed = lcg_bookings(200, seed)
    one, cal_one, oracle = MyCalendar(), MyCalendarK(1), []
    for start, end in bookings:
        expected = sweep_peak(oracle + [(start, end)]) <= 1
        assert one.book(start, end) == expected
        assert cal_one.book(start, end) == expected
        if expected:
            oracle.append((start, end))


if __name__ == "__main__":
    main()
    print("PASS")
