"""Task 299 - DP over subsets (SOS DP simple task). (Tier: Intermediate, CTCI Beyond — Networking, OS & Real-Time)

Sum-over-subsets (zeta) transform: f[mask] = sum of a[sub] over every sub ⊆ mask, with
the array indexed by bitmask. The naive loop enumerates all submasks of every mask, so
the total work is sum over masks of 2^popcount(mask) = sum_k C(n,k) * 2^k = (1 + 2)^n =
3^n. The SOS DP instead runs n rounds of 2^n relaxations: round i adds f[mask ^ 2^i]
into f[mask] for every mask holding bit i, which propagates each (mask, sub) pair through
the highest bit where they differ, giving O(n * 2^n) instead of O(3^n). For n = 20 that
is about 20M operations versus 3^20 ≈ 3.5 billion for the naive enumeration.
"""


def sum_over_subsets(a):
    """f[mask] = sum of a[sub] for all sub ⊆ mask; len(a) must be a power of two."""
    n = len(a).bit_length() - 1
    assert len(a) == 1 << n, "length must be a power of two"
    f = list(a)
    for i in range(n):
        bit = 1 << i
        for mask in range(len(a)):
            if mask & bit:
                f[mask] += f[mask ^ bit]
    return f


def brute_sos(a):
    """Oracle: enumerate every submask of every mask explicitly (the O(3^n) loop)."""
    result = []
    for mask in range(len(a)):
        total = 0
        sub = mask
        while True:
            total += a[sub]
            if sub == 0:
                break
            sub = (sub - 1) & mask
        result.append(total)
    return result


def lcg_array(size, seed):
    values = []
    for _ in range(size):
        seed = (seed * 75 + 74) % 65537
        values.append(seed % 50)
    return values, seed


def main():
    assert sum_over_subsets([1, 2, 4, 8]) == [1, 3, 5, 15]
    assert sum_over_subsets([7]) == [7]

    seed = 606
    for n in (1, 2, 3, 4, 10):
        a, seed = lcg_array(1 << n, seed)
        f = sum_over_subsets(a)
        assert f == brute_sos(a), f"n={n}"
        # each a[sub] lands in exactly the 2^(n - popcount(sub)) supersets of sub
        assert sum(f) == sum(
            a[mask] * (1 << (n - bin(mask).count("1"))) for mask in range(len(a))
        ), f"n={n}"


if __name__ == "__main__":
    main()
    print("PASS")
