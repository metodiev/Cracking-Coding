"""Task 326 - Meeting rooms required (interval partitioning). (Tier: Intermediate, CTCI Ch 1)

Minimum meeting rooms equals the maximum number of meetings alive at any instant.
Meetings are half-open ([start, end)), so a meeting ending exactly when another
starts reuses the same room.  The endpoint-event sweep is cross-checked against an
independent sort + min-heap sweep and against a per-minute brute force.
"""

import heapq


def min_meeting_rooms(intervals):
    """Maximum overlap via endpoint events; an end (-1) sorts before a start (+1)."""
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda event: (event[0], event[1]))
    rooms = best = 0
    for _, delta in events:
        rooms += delta
        if rooms > best:
            best = rooms
    return best


def min_rooms_heap(intervals):
    """Independent sweep with a min-heap of finish times."""
    heap = []
    for start, end in sorted(intervals):
        if heap and heap[0] <= start:
            heapq.heapreplace(heap, end)
        else:
            heapq.heappush(heap, end)
    return len(heap)


def min_rooms_per_minute(intervals):
    """Independent oracle for small coordinates: count active meetings at each minute."""
    if not intervals:
        return 0
    lo = min(start for start, _ in intervals)
    hi = max(end for _, end in intervals)
    return max(sum(1 for start, end in intervals if start <= t < end)
               for t in range(lo, hi + 1))


def lcg_intervals(count, seed):
    intervals = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        start = seed % 30
        seed = (seed * 75 + 74) % 65537
        end = start + 1 + seed % 10
        intervals.append([start, end])
    return intervals, seed


def main():
    assert min_meeting_rooms([[0, 30], [5, 10], [15, 20]]) == 2
    assert min_meeting_rooms([[7, 10], [2, 4]]) == 1
    assert min_rooms_heap([[0, 30], [5, 10], [15, 20]]) == 2
    assert min_rooms_heap([[7, 10], [2, 4]]) == 1

    assert min_meeting_rooms([[1, 2], [2, 3], [3, 4]]) == 1     # back-to-back shares a room
    assert min_meeting_rooms([[1, 5], [2, 6], [3, 7]]) == 3
    assert min_meeting_rooms([]) == 0
    assert min_rooms_heap([]) == 0

    seed = 2718
    for trial in range(300):
        count = 1 + seed % 10
        intervals, seed = lcg_intervals(count, seed)
        expected = min_rooms_per_minute(intervals)
        assert min_meeting_rooms(intervals) == expected, intervals
        assert min_rooms_heap(intervals) == expected, intervals


if __name__ == "__main__":
    main()
    print("PASS")
