"""Task 379 - Binomial coefficients C(n, k) mod prime p via precomputed factorials. (Tier: Intermediate, CTCI Ch 6)

For a prime p = 1_000_000_007 factorials and inverse factorials are precomputed once:
inv_fact[n] = (fact[n])^(p-2) mod p (Fermat), then downward inv_fact[i] = inv_fact[i+1] * (i+1).
Then C(n, k) = fact[n] * inv_fact[k] * inv_fact[n-k] mod p in O(1) per query, valid for
0 <= k <= n < p and 0 < n < p (Lucas is not needed below p).

Hand-verified assertions: C(10,3) = 120, C(20,10) = 184756, C(n,0) = C(n,n) = 1,
k > n -> 0 (the convention used everywhere in this repo), k < 0 -> 0.

Independent verification paths:
  * a multiplicative loop C(n,k) = prod_{i=1..k} (n-k+i) * inv(i) mod p that never
    touches the factorial tables;
  * additive Pascal rows built by repeated addition mod p (no modular inverses at
    all) for n <= 140, cross-checked on LCG (n, k) pairs;
  * the known constants C(100,50) mod p = 538992043 and C(1000,500) mod p =
    159835829, each obtained independently (multiplicative loop + CAS-style big
    integer reduction) and asserted here.

The precomputation and all queries use exact integer modular arithmetic - no
floating point anywhere.
"""

MOD = 1_000_000_007


def mod_pow(base, exponent, modulus):
    """Fast modular exponentiation (binary)."""
    result = 1 % modulus
    base %= modulus
    while exponent > 0:
        if exponent & 1:
            result = result * base % modulus
        base = base * base % modulus
        exponent >>= 1
    return result


def build_factorials(limit):
    """(fact, inv_fact) arrays for 0..limit modulo MOD."""
    fact = [1] * (limit + 1)
    for i in range(1, limit + 1):
        fact[i] = fact[i - 1] * i % MOD
    inv_fact = [1] * (limit + 1)
    inv_fact[limit] = mod_pow(fact[limit], MOD - 2, MOD)
    for i in range(limit, 0, -1):
        inv_fact[i - 1] = inv_fact[i] * i % MOD
    return fact, inv_fact


class Binomial:
    """C(n, k) mod MOD with precomputed tables."""

    def __init__(self, limit):
        self.fact, self.inv_fact = build_factorials(limit)
        self.limit = limit

    def choose(self, n, k):
        if k < 0 or n < 0 or k > n:
            return 0
        if n > self.limit:
            raise ValueError("n above the precomputed limit")
        return self.fact[n] * self.inv_fact[k] % MOD * self.inv_fact[n - k] % MOD


def choose_multiplicative(n, k):
    """Independent path: running product with per-step modular inverses."""
    if k < 0 or n < 0 or k > n:
        return 0
    if k > n - k:
        k = n - k
    result = 1
    for i in range(1, k + 1):
        result = result * ((n - k + i) % MOD) % MOD
        result = result * mod_pow(i, MOD - 2, MOD) % MOD
    return result


def pascal_row(n):
    """Additive Pascal row mod MOD - no division or inversion used at all."""
    row = [1]
    for _ in range(n):
        nxt = [1] * (len(row) + 1)
        for i in range(1, len(row)):
            nxt[i] = (row[i - 1] + row[i]) % MOD
        row = nxt
    return row


def lcg_pair(seed, max_n):
    seed = (seed * 75 + 74) % 65537
    n = seed % (max_n + 1)
    seed = (seed * 75 + 74) % 65537
    k = seed % (n + 2) - 1           # includes -1, 0, n and n+1 -> edge conventions
    return n, k, seed


def main():
    table = Binomial(1000)

    assert table.choose(10, 3) == 120
    assert table.choose(20, 10) == 184756
    assert table.choose(5, 0) == 1 and table.choose(5, 5) == 1
    assert table.choose(0, 0) == 1
    assert table.choose(3, 5) == 0                    # k > n -> 0
    assert table.choose(3, -1) == 0                   # k < 0 -> 0
    assert table.choose(0, 1) == 0

    # symmetry C(n, k) == C(n, n-k)
    for n in (1, 2, 7, 40, 999, 1000):
        for k in (0, 1, n // 2, n - 1, n):
            assert table.choose(n, k) == table.choose(n, n - k), (n, k)

    # real (non-modular) checks where the exact value is known
    assert choose_multiplicative(10, 3) == 120
    assert choose_multiplicative(20, 10) == 184756
    assert choose_multiplicative(52, 5) == 2598960    # poker hands

    # known constants, each verified by the independent multiplicative path
    assert table.choose(100, 50) == 538992043
    assert choose_multiplicative(100, 50) == 538992043
    assert table.choose(1000, 500) == 159835829
    assert choose_multiplicative(1000, 500) == 159835829

    # additive Pascal rows: completely different arithmetic path for smaller n
    seed = 37913
    for _ in range(60):
        n, k, seed = lcg_pair(seed, 140)
        if k < 0 or k > n:
            expected = 0
        else:
            expected = pascal_row(n)[k]
        assert table.choose(n, k) == expected, (n, k)
        assert choose_multiplicative(n, k) == expected, (n, k)
    for n in (1, 2, 3, 10, 50, 140):
        row = pascal_row(n)
        for k in range(n + 1):
            assert table.choose(n, k) == row[k], (n, k)
            assert choose_multiplicative(n, k) == row[k], (n, k)

    # Pascal identity C(n, k) = C(n-1, k-1) + C(n-1, k) on the modular tables
    for _ in range(40):
        n, k, seed = lcg_pair(seed, 1000)
        if k < 1 or k > n:
            continue
        assert table.choose(n, k) == (table.choose(n - 1, k - 1) + table.choose(n - 1, k)) % MOD, (n, k)

    # binomial theorem sanity: sum_k C(n, k) = 2^n mod p
    for n in (5, 30, 200, 1000):
        total = 0
        for k in range(n + 1):
            total = (total + table.choose(n, k)) % MOD
        assert total == mod_pow(2, n, MOD), n


if __name__ == "__main__":
    main()
    print("PASS")
