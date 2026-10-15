"""Task 039 - Simple linear congruential pseudorandom generator. (Tier: Beginner, CTCI Ch 6)"""

def lcg(seed, n, a=1103515245, c=12345, m=2 ** 31):
    """x_{k+1} = (a * x_k + c) mod m. Deterministic per seed."""
    x = seed
    out = []
    for _ in range(n):
        x = (a * x + c) % m
        out.append(x)
    return out


def main():
    seq = lcg(42, 5)
    assert len(seq) == 5
    assert seq == lcg(42, 5)                  # deterministic
    assert seq != lcg(43, 5)                  # seed-sensitive
    assert all(0 <= x < 2 ** 31 for x in seq)


if __name__ == "__main__":
    main()
    print("PASS")
