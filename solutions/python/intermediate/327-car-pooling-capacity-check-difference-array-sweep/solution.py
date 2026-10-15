"""Task 327 - Car pooling capacity check (difference array / sweep). (Tier: Intermediate, CTCI Ch 1)

Each trip adds `passengers` over [start, end): passengers get off before the next
ones get on at the same stop, so trips touch-don't-clash there.  A difference array
(pickup +p at start, drop -p at end) is swept in stop order; the running load must
never exceed the capacity.  A brute-force per-stop materialisation cross-checks it.
"""


def car_pooling(trips, capacity):
    """Difference-array sweep; True when the load never exceeds `capacity`."""
    delta = {}
    for passengers, start, end in trips:
        delta[start] = delta.get(start, 0) + passengers
        delta[end] = delta.get(end, 0) - passengers
    load = 0
    for stop in sorted(delta):
        load += delta[stop]
        if load > capacity:
            return False
    return True


def car_pooling_bruteforce(trips, capacity):
    """Independent oracle for small coordinates: materialise the load at every stop."""
    if not trips:
        return capacity >= 0
    lo = min(start for _, start, _ in trips)
    hi = max(end for _, _, end in trips)
    loads = [0] * (hi - lo)
    for passengers, start, end in trips:
        for stop in range(start, end):
            loads[stop - lo] += passengers
    return max(loads, default=0) <= capacity


def lcg_trips(count, seed):
    trips = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        passengers = 1 + seed % 6
        seed = (seed * 75 + 74) % 65537
        start = seed % 25
        seed = (seed * 75 + 74) % 65537
        end = start + 1 + seed % 8
        trips.append([passengers, start, end])
    return trips, seed


def main():
    assert car_pooling([[2, 1, 5], [3, 3, 7]], 4) is False
    assert car_pooling([[2, 1, 5], [3, 3, 7]], 5) is True
    assert car_pooling([[2, 1, 5], [3, 5, 7]], 3) is True
    assert car_pooling([[2, 1, 5], [3, 5, 7]], 2) is False
    assert car_pooling([], 0) is True
    assert car_pooling([[1, 0, 1]], 0) is False

    assert car_pooling_bruteforce([[2, 1, 5], [3, 3, 7]], 4) is False
    assert car_pooling_bruteforce([[2, 1, 5], [3, 5, 7]], 3) is True

    seed = 1618
    for trial in range(300):
        count = 1 + seed % 8
        trips, seed = lcg_trips(count, seed)
        capacity = seed % 21
        expected = car_pooling_bruteforce(trips, capacity)
        assert car_pooling(trips, capacity) is expected, (trips, capacity)


if __name__ == "__main__":
    main()
    print("PASS")
