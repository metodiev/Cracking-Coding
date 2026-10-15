"""Task 325 - Minimum number of platforms for trains (sweep-line). (Tier: Intermediate, CTCI Beyond — Software Engineering & Tooling)

A train occupies its platform over [arrival, departure), so at an equal minute the
departing train releases the platform first.  Two-pointer sweep over the sorted
arrival and departure lists; the answer is the maximum number of trains present at
once, cross-checked by counting occupancy minute by minute.
"""


def min_platforms(arrivals, departures):
    """Two-pointer sweep; a train leaving at t frees its platform before any arrival at t."""
    arr = sorted(arrivals)
    dep = sorted(departures)
    i = j = 0
    needed = best = 0
    while i < len(arr):
        if arr[i] < dep[j]:
            needed += 1
            if needed > best:
                best = needed
            i += 1
        else:
            needed -= 1
            j += 1
    return best


def min_platforms_per_minute(arrivals, departures):
    """Independent oracle: count active trains at every minute of the schedule."""
    if not arrivals:
        return 0
    best = 0
    for minute in range(min(arrivals), max(departures) + 1):
        active = sum(1 for arrival, departure in zip(arrivals, departures)
                     if arrival <= minute < departure)
        if active > best:
            best = active
    return best


def lcg_schedules(count, seed):
    arrivals = []
    departures = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        arrival = seed % 60
        seed = (seed * 75 + 74) % 65537
        duration = 1 + seed % 20
        arrivals.append(arrival)
        departures.append(arrival + duration)
    return arrivals, departures, seed


def main():
    arrivals = [900, 940, 950, 1100, 1500, 1800]
    departures = [910, 1200, 1120, 1130, 1900, 2000]
    assert min_platforms(arrivals, departures) == 3
    assert min_platforms_per_minute(arrivals, departures) == 3

    assert min_platforms([1, 2], [2, 3]) == 1        # release before acquire at minute 2
    assert min_platforms_per_minute([1, 2], [2, 3]) == 1
    assert min_platforms([1, 1, 1], [2, 2, 2]) == 3
    assert min_platforms([5], [9]) == 1
    assert min_platforms([], []) == 0

    seed = 31415
    for trial in range(300):
        count = 1 + seed % 12
        arr, dep, seed = lcg_schedules(count, seed)
        expected = min_platforms_per_minute(arr, dep)
        assert min_platforms(arr, dep) == expected, (arr, dep)


if __name__ == "__main__":
    main()
    print("PASS")
