"""Task 352 - Implement fork-join style parallel sum (threads). (Tier: Intermediate, CTCI Ch 15)

Fork-join outline: sum_range(lo, hi) returns the sum directly once the range holds at
most THRESHOLD = 256 elements (a leaf task); otherwise it forks the left half onto a
new thread while computing the right half inline, then joins. The self-test checks the
parallel sum against a sequential sum on LCG arrays of many sizes and asserts the task
accounting: leaves are counted under a lock, spawned threads == leaves - 1 (every
internal node of the binary split tree forks exactly once), and for power-of-two sizes
leaves == n / THRESHOLD. Every join carries a timeout, so a leaked worker fails an
assertion instead of hanging.
"""
import threading

THRESHOLD = 256


def sequential_sum(values):
    return sum(values)


def count_leaves(size):
    """Independent (iterative) oracle for the number of leaves the split policy creates."""
    leaves = 0
    stack = [(0, size)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo <= THRESHOLD:
            leaves += 1
        else:
            mid = (lo + hi) // 2
            stack.append((lo, mid))
            stack.append((mid, hi))
    return leaves


def parallel_sum(values, timeout=10.0):
    """Returns (total, stats) with stats['leaves'] and stats['spawned']."""
    stats = {"leaves": 0, "spawned": 0}
    lock = threading.Lock()

    def sum_range(lo, hi):
        if hi - lo <= THRESHOLD:                      # leaf task: sum without splitting
            with lock:
                stats["leaves"] += 1
            return sum(values[lo:hi])

        mid = (lo + hi) // 2
        forked = {}

        def fork_left():
            forked["value"] = sum_range(lo, mid)

        with lock:
            stats["spawned"] += 1
        thread = threading.Thread(target=fork_left, daemon=True)
        thread.start()
        right = sum_range(mid, hi)                    # compute one half inline
        thread.join(timeout)                          # join the forked half
        assert not thread.is_alive(), "forked worker did not finish in time"
        return forked["value"] + right

    return sum_range(0, len(values)), stats


def lcg_array(size, seed=24680):
    values = []
    for _ in range(size):
        seed = (seed * 75 + 74) % 65537
        values.append(seed)
    return values


def main():
    empty_total, empty_stats = parallel_sum([])
    assert empty_total == 0, "empty array sums to 0"
    assert empty_stats == {"leaves": 1, "spawned": 0}, "empty range is a single leaf"

    sizes = [1, 7, THRESHOLD, THRESHOLD + 1, 2 * THRESHOLD, 3 * THRESHOLD,
             1024, 2048, 4096, 8192]
    for size in sizes:
        values = lcg_array(size)
        total, stats = parallel_sum(values)
        assert total == sequential_sum(values), "parallel sum mismatch at size {}".format(size)
        assert stats["leaves"] == count_leaves(size), "leaf count at size {}".format(size)
        assert stats["spawned"] == stats["leaves"] - 1, "binary fork tree at size {}".format(size)

    # Documented threshold accounting for power-of-two sizes: leaves == n / THRESHOLD.
    for size in [512, 1024, 2048, 4096, 8192]:
        total, stats = parallel_sum(lcg_array(size))
        assert stats["leaves"] == size // THRESHOLD, "power-of-two leaves at {}".format(size)
        assert stats["spawned"] == size // THRESHOLD - 1, "power-of-two threads at {}".format(size)

    # Splitting really happens above the threshold, and 8192 splits into exactly 31 forks.
    assert parallel_sum(lcg_array(THRESHOLD))[1]["spawned"] == 0, "at threshold: no fork"
    assert parallel_sum(lcg_array(THRESHOLD + 1))[1]["spawned"] == 1, "just above: one fork"
    assert parallel_sum(lcg_array(8192))[1]["spawned"] == 31, "8192 -> 32 leaves, 31 forks"

    # LCG-driven sizes across the whole range, each cross-checked against the sequential sum.
    seed = 9753
    for _ in range(20):
        seed = (seed * 75 + 74) % 65537
        size = seed % 3000
        values = lcg_array(size)
        total, stats = parallel_sum(values)
        assert total == sequential_sum(values), "LCG size {}".format(size)
        assert stats["leaves"] == count_leaves(size), "LCG leaves at {}".format(size)
        assert stats["spawned"] == stats["leaves"] - 1, "LCG fork tree at {}".format(size)


if __name__ == "__main__":
    main()
    print("PASS")
