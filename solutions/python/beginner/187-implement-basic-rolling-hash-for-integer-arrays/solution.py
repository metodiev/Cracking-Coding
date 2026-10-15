"""Task 187 - Implement basic rolling hash for integer arrays. (Tier: Beginner, CTCI Beyond — Advanced Data Structures & Algorithms)

Hash every length-k window of an int array in O(1) per slide with a polynomial
rolling hash, bucket windows by hash, then confirm each bucket by comparing the
windows themselves so a hash collision cannot fake a duplicate. Returns
{first_index: [later_indices]} for every window that really repeats.
"""

DEFAULT_BASE = 257
DEFAULT_MOD = 1_000_000_007


def pow_mod(base, exponent, mod):
    result = 1
    for _ in range(exponent):
        result = result * base % mod
    return result


def find_duplicate_windows(values, k, base=DEFAULT_BASE, mod=DEFAULT_MOD):
    """Map each repeated window's first start index to its later occurrences."""
    if k <= 0:
        raise ValueError("window length must be positive")
    n = len(values)
    if k > n:
        return {}

    highest = pow_mod(base, k - 1, mod)
    window_hash = 0
    for value in values[:k]:
        window_hash = (window_hash * base + value % mod) % mod

    buckets = {}
    buckets.setdefault(window_hash, []).append(0)
    for start in range(1, n - k + 1):
        window_hash = (window_hash - (values[start - 1] % mod) * highest) % mod
        window_hash = (window_hash * base + values[start + k - 1] % mod) % mod
        buckets.setdefault(window_hash, []).append(start)

    # equal hashes are only candidates: compare the windows themselves
    duplicates = {}
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        groups = {}
        for index in indices:
            groups.setdefault(tuple(values[index:index + k]), []).append(index)
        for positions in groups.values():
            if len(positions) > 1:
                duplicates[positions[0]] = positions[1:]
    return dict(sorted(duplicates.items()))


def find_duplicate_windows_naive(values, k):
    """Reference implementation: group windows directly, no hashing."""
    if k <= 0 or k > len(values):
        return {}
    groups = {}
    for start in range(len(values) - k + 1):
        groups.setdefault(tuple(values[start:start + k]), []).append(start)
    result = {positions[0]: positions[1:] for positions in groups.values() if len(positions) > 1}
    return dict(sorted(result.items()))


def main():
    assert find_duplicate_windows([1, 2, 3, 1, 2, 3, 1, 2, 3], 3) == {0: [3, 6], 1: [4], 2: [5]}
    assert find_duplicate_windows([2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], 3) == {0: [3, 6], 1: [4, 7], 2: [5, 8]}
    assert find_duplicate_windows([5, 5, 5, 5], 2) == {0: [1, 2]}
    assert find_duplicate_windows([4, 7, 4, 7, 4], 2) == {0: [2], 1: [3]}          # overlaps count
    assert find_duplicate_windows([7, 7, 8, 7, 7], 1) == {0: [1, 3, 4]}
    assert find_duplicate_windows([1, 2, 3, 1, 2, 4], 3) == {}                    # near-duplicate only
    assert find_duplicate_windows([1, 2, 3, 4], 4) == {}                          # single window
    assert find_duplicate_windows([1, 2, 3, 4], 5) == {}                          # k larger than array
    assert find_duplicate_windows([], 2) == {}
    assert find_duplicate_windows([-5, -5, 3, -5, -5], 2) == {0: [3]}             # negative values
    try:
        find_duplicate_windows([1, 2], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for k = 0")

    # agreement with the direct grouping on pseudo-random arrays
    import random
    rng = random.Random(23)
    for _ in range(200):
        values = [rng.randrange(4) for _ in range(rng.randint(0, 30))]
        k = rng.randint(1, 4)
        assert find_duplicate_windows(values, k) == find_duplicate_windows_naive(values, k), (values, k)

    # a tiny modulus makes collisions common; results must still be exact
    for _ in range(200):
        values = [rng.randrange(200) for _ in range(rng.randint(0, 30))]
        k = rng.randint(1, 3)
        assert find_duplicate_windows(values, k, base=257, mod=97) == find_duplicate_windows_naive(values, k), (values, k)

    # concrete collision: the windows (5, 63) and (6, 0) share one hash under mod 97
    assert (5 * 257 + 63) % 97 == (6 * 257 + 0) % 97
    assert find_duplicate_windows([5, 63, 4, 6, 0], 2, base=257, mod=97) == {}
    assert find_duplicate_windows([5, 63, 4, 6, 0, 5, 63], 2, base=257, mod=97) == {0: [5]}


if __name__ == "__main__":
    main()
    print("PASS")
