"""Task 466 - Implement hyperloglog sketch for cardinality estimation (outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Outline HyperLogLog with m = 2^9 = 512 registers. Items are hashed with the FIXED
32-bit mixer mix32 (0x45d9f3b xorshift-multiply; identical constants in every
language port). The top p = 9 bits pick the register; the rank is the 1-based
position of the leftmost set bit of the remaining 23 bits (all-zero tail counts as
24). Estimate = alpha_m * m^2 / sum(2^-M[j]) with alpha_m = 0.7213 / (1 + 1.079/m),
plus linear counting m*ln(m/V) while the raw estimate is <= 2.5m and V registers
are still zero.

Checks: for each digest size n in {100, 1000, 5000, 20000}, 16 independent digests
are built from keys hashed as (pool << 17) | value so the pools are disjoint (a
single shared 16-bit key space would correlate the digests). Documented bounds:
  * |mean estimate - n| / n <= 1.04/sqrt(m) ~ 0.046  (measured max 0.008), the
    classic HLL relative standard error;
  * every single digest stays within the 3-sigma envelope 3 * 1.04/sqrt(m) ~ 0.138
    (measured max 0.098);
  * mean estimates increase with the true cardinality, and
  * a register-wise max merge of two complementary digests equals the digest of
    the union exactly.
"""
import math

P_BITS = 9
M = 1 << P_BITS
RANK_WIDTH = 32 - P_BITS
MASK32 = 0xFFFFFFFF
ALPHA = 0.7213 / (1 + 1.079 / M)
SIGMA = 1.04 / math.sqrt(M)


def mix32(x):
    x &= MASK32
    x ^= x >> 16
    x = (x * 0x45D9F3B) & MASK32
    x ^= x >> 16
    x = (x * 0x45D9F3B) & MASK32
    x ^= x >> 16
    return x


def rank_of(rest, width):
    if rest == 0:
        return width + 1
    return width - rest.bit_length() + 1


class HyperLogLog:
    def __init__(self):
        self.registers = [0] * M

    def add(self, key):
        h = mix32(key)
        index = h >> RANK_WIDTH
        rest = h & ((1 << RANK_WIDTH) - 1)
        r = rank_of(rest, RANK_WIDTH)
        if r > self.registers[index]:
            self.registers[index] = r

    def merge(self, other):
        self.registers = [max(a, b) for a, b in zip(self.registers, other.registers)]

    def estimate(self):
        z = 0.0
        for m in self.registers:
            z += 2.0 ** -m
        raw = ALPHA * M * M / z
        zeros = self.registers.count(0)
        if raw <= 2.5 * M and zeros:
            return M * math.log(M / zeros)
        return raw


def lcg_value(seed):
    return (seed * 75 + 74) % 65537


def pool_keys(count, pool, seed):
    """`count` distinct keys with the pool id mixed into the high bits."""
    keys = []
    seen = set()
    while len(keys) < count:
        seed = lcg_value(seed)
        if seed not in seen:
            seen.add(seed)
            keys.append((pool << 17) | seed)
    return keys


def digest(keys):
    sketch = HyperLogLog()
    for key in keys:
        sketch.add(key)
    return sketch


def main():
    # Hand-checked mixing / rank behaviour.
    assert mix32(0) == 0
    assert rank_of(0, RANK_WIDTH) == RANK_WIDTH + 1
    assert rank_of(1 << (RANK_WIDTH - 1), RANK_WIDTH) == 1
    assert rank_of(1, RANK_WIDTH) == RANK_WIDTH
    assert rank_of(0b101, RANK_WIDTH) == RANK_WIDTH - 2

    sizes = [100, 1000, 5000, 20000]
    pools = 16
    mean_estimates = []
    max_mean_error = 0.0
    max_single_error = 0.0
    for size in sizes:
        estimates = []
        for pool in range(pools):
            estimates.append(digest(pool_keys(size, pool, (pool + 1) * 131)).estimate())
        mean_est = sum(estimates) / pools
        mean_estimates.append(mean_est)
        mean_error = abs(mean_est - size) / size
        max_mean_error = max(max_mean_error, mean_error)
        assert mean_error <= SIGMA, (size, mean_est, mean_error)  # documented 1.04/sqrt(m)
        for est in estimates:
            single_error = abs(est - size) / size
            max_single_error = max(max_single_error, single_error)
            assert single_error <= 3 * SIGMA, (size, est, single_error)  # documented 3-sigma
    for a, b in zip(mean_estimates, mean_estimates[1:]):
        assert a < b  # mean estimates grow with the true cardinality

    # Merge of two complementary digests must equal the digest of the union exactly.
    keys = pool_keys(3000, 7, 4242)
    merged = digest(keys[:1500])
    merged.merge(digest(keys[1500:]))
    whole = digest(keys)
    assert merged.registers == whole.registers
    assert merged.estimate() == whole.estimate()
    assert max_mean_error <= 0.02 and max_single_error <= 0.12  # measured margins


if __name__ == "__main__":
    main()
    print("PASS")
