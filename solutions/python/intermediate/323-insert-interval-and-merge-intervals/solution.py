"""Task 323 - Insert interval and merge intervals. (Tier: Intermediate, CTCI Ch 1)

Merge a list of closed intervals, and insert one more into an already sorted,
non-overlapping list.  Touching intervals merge too: [1, 2] and [2, 3] share the
endpoint 2 and become [1, 3] (a next interval is merged while next.start <= end).
The merge result is cross-checked against an independent endpoint-event sweep.
"""


def merge_intervals(intervals):
    """Sort by start and merge overlapping or touching intervals."""
    if not intervals:
        return []
    ordered = sorted((list(interval) for interval in intervals), key=lambda iv: (iv[0], iv[1]))
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:              # touching counts: <=
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def insert_interval(intervals, new_interval):
    """Insert into a sorted non-overlapping list, merging overlapping/touching parts."""
    start, end = new_interval
    result = []
    i = 0
    while i < len(intervals) and intervals[i][1] < start:
        result.append(list(intervals[i]))
        i += 1
    while i < len(intervals) and intervals[i][0] <= end:
        start = min(start, intervals[i][0])
        end = max(end, intervals[i][1])
        i += 1
    result.append([start, end])
    while i < len(intervals):
        result.append(list(intervals[i]))
        i += 1
    return result


def merge_intervals_sweep(intervals):
    """Independent oracle: endpoint events; starts sort before ends at the same x."""
    events = []
    for start, end in intervals:
        events.append((start, 0))
        events.append((end, 1))
    events.sort()
    merged = []
    depth = 0
    for x, kind in events:
        if kind == 0:
            if depth == 0:
                merged.append([x, 0])
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                merged[-1][1] = x
    return merged


def lcg_intervals(count, seed):
    intervals = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        start = seed % 25
        seed = (seed * 75 + 74) % 65537
        end = start + 1 + seed % 8
        intervals.append([start, end])
    return intervals, seed


def lcg_disjoint_sorted(count, seed):
    intervals = []
    cursor = 0
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        cursor += 1 + seed % 4
        seed = (seed * 75 + 74) % 65537
        end = cursor + 1 + seed % 5
        intervals.append([cursor, end])
        cursor = end + 1
    return intervals, seed


def main():
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert merge_intervals([[1, 2], [2, 3]]) == [[1, 3]]          # touching-merge policy
    assert merge_intervals([[1, 4], [4, 5], [7, 9]]) == [[1, 5], [7, 9]]
    assert merge_intervals([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]
    assert merge_intervals([[5, 5]]) == [[5, 5]]
    assert merge_intervals([]) == []
    assert merge_intervals_sweep([[1, 2], [2, 3]]) == [[1, 3]]

    base = [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]
    assert insert_interval(base, [9, 10]) == [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]
    assert insert_interval(base, [0, 0]) == [[0, 0], [1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]
    assert insert_interval(base, [4, 4]) == [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]
    assert insert_interval([[1, 2], [5, 7]], [2, 5]) == [[1, 7]]  # touches both sides
    assert insert_interval([[3, 4]], [1, 2]) == [[1, 2], [3, 4]]
    assert insert_interval([], [4, 6]) == [[4, 6]]

    seed = 20077
    for trial in range(200):
        intervals, seed = lcg_intervals(6, seed)
        assert merge_intervals(intervals) == merge_intervals_sweep(intervals), intervals

    seed = 4242
    for trial in range(200):
        base, seed = lcg_disjoint_sorted(5, seed)
        new_start = seed % 30
        seed = (seed * 75 + 74) % 65537
        new_interval = [new_start, new_start + 1 + seed % 8]
        assert insert_interval(base, new_interval) == merge_intervals(base + [new_interval]), (base, new_interval)


if __name__ == "__main__":
    main()
    print("PASS")
