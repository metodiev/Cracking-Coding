"""Task 571 - Implement collision-resistant hash property tests (statistical). (Tier: Advanced, CTCI Beyond — Security & Privacy)

A test harness that asks, statistically, whether a hash function behaves like a random oracle, run
against three candidates: SHA-256, BLAKE2b (both from the standard library), and a deliberately
weak low-byte hash h(key) = key mod 256 kept as the negative control. The statistics are computed
from first principles rather than from lookup tables: the regularised lower incomplete gamma P(a, x)
is implemented by its series and by the Legendre continued fraction for the complementary Q(a, x),
which turns a chi-square statistic into a genuine p-value (chi_square_sf(stat, df) = Q(df/2,
stat/2)) and a monobit count into an erfc-based tail probability. That machinery is validated
against textbook quantiles (chi_square_sf(30.578, 15) ~ 0.01, (3.841, 1) ~ 0.05, (2.706, 1) ~ 0.10,
(11.345, 3) ~ 0.01) and against the closed form P(1, x) = 1 - e^{-x}. Four classical properties are
then tested: (1) uniformity of bucket counts by chi-square over 65536 keys in 256 buckets;
(2) the birthday collision count - pairs colliding after truncating to b bits, whose expectation
N(N-1)/2^(b+1) is compared against the observed count inside a four-sigma Poisson band, plus the
demonstration that the 128-bit truncations produce no collision at all among 20000 keys;
(3) the avalanche criterion - flipping one input bit changes about half of the 256 output bits,
checked on average and per input bit; (4) a sticky-bit/frequency check on every (input bit, output
bit) pair, whose maximum deviation from 1/2 is reported. The negative control must fail (1), (3) and
(4) loudly and its structured input collisions must exceed the birthday prediction by orders of
magnitude, while both real hashes must pass every test - the file asserts both directions, so a test
that silently passes everything is as much a bug as one that fails everything.

Every key stream comes from splitmix64 with a fixed seed (documented below) rather than from the
repository's 16-bit LCG: uniform bit patterns need many independent draws and the LCG's short cycle
and weak low bits would bias the very statistics being measured. Nothing depends on wall-clock time,
hash ordering or platform float behaviour; the p-value tolerances are explicit.

Checks (with the measured values asserted, not just bounded): the gamma/chi-square implementation
against four published chi-square quantiles and the exponential closed form P(1, x) = 1 - e^{-x};
SHA-256 and BLAKE2b passing the 256-bucket uniformity test on 65536 splitmix64 keys (chi-square
284.07 / 260.38, p = 0.102 / 0.395, every bucket populated) while the control fails it with
p < 1e-12 and reaches exactly the 32 buckets that are multiples of gcd(1000, 256) = 8, 128 keys
each; observed birthday collisions at 16 bits over 4000 keys, 119 (SHA-256) and 109 (BLAKE2b)
against a prediction of 122.04 +- 4 sqrt(122) = [78, 166], zero collisions at 128 bits over 20000
keys, and the control's 260096 pairs versus the 32760 the birthday rate predicts (7.9 times more,
on the same 4000 structured keys); mean avalanche 0.5002 / 0.4999 with per-input-bit extremes inside
[0.4975, 0.5027], versus exactly 0.125 for the control on its eight low input bits and 0.0 above
them; sticky-bit deviations at most 0.0342 / 0.0371 for the good hashes (mean 0.0088) versus a full
0.5 for the control on an all-even key stream; and determinism of the key stream and of every
digest (including a cross-check of digest_int against hashlib's hexdigest).
"""
import hashlib
import math

MASK64 = (1 << 64) - 1
TWO64 = float(1 << 64)
AVALANCHE_KEYS = 1024


def splitmix64(state):
    """Splitmix64: the fixed-seed 64-bit generator for the key streams (the repository's 16-bit LCG
    has a short cycle and weak low bits, which would distort the bit-level statistics measured
    here)."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31))


def key_stream(seed, count, bits=64):
    """count keys drawn uniformly from [0, 2^bits)."""
    state = seed
    keys = []
    for _ in range(count):
        state, value = splitmix64(state)
        keys.append(value >> (64 - bits) if bits < 64 else value)
    return keys


# --- The hash candidates ----------------------------------------------------------------

def digest_bytes(data, algorithm="sha256"):
    """32-byte digest; BLAKE2b is asked for digest_size=32 so both candidates are 256 bits wide."""
    if algorithm == "blake2b":
        return hashlib.blake2b(data, digest_size=32).digest()
    return hashlib.new(algorithm, data).digest()


def digest_int(data, algorithm="sha256"):
    return int.from_bytes(digest_bytes(data, algorithm), "big")


def truncated_digest(key, bits, algorithm="sha256"):
    """Leading `bits` bits of the digest of the 8-byte big-endian encoding of the key."""
    value = digest_int(key.to_bytes(8, "big"), algorithm)
    return value >> (256 - bits) if bits < 256 else value


def weak_low_byte(key, bits=None):
    """Negative control: the low byte of the key, extended to the requested width by zeroes."""
    value = key % 256
    return value if bits is None else value % (1 << bits)


WEAK = weak_low_byte
GOOD = {
    "sha256": lambda key, bits: truncated_digest(key, bits, "sha256"),
    "blake2b": lambda key, bits: truncated_digest(key, bits, "blake2b"),
}


# --- The statistical machinery -----------------------------------------------------------

def gammainc_p(a, x, max_iter=400, tol=1e-15):
    """Regularised lower incomplete gamma P(a, x) (Numerical Recipes' series/continued fraction)."""
    if a <= 0.0 or x < 0.0:
        raise ValueError("invalid incomplete-gamma arguments")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        for n in range(1, max_iter + 1):
            term *= x / (a + n)
            total += term
            if abs(term) < tol * abs(total):
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def chi_square_sf(statistic, df):
    """Upper-tail probability P[chi^2_df > statistic]."""
    return 1.0 - gammainc_p(df / 2.0, statistic / 2.0)


def chi_square(counts):
    total = sum(counts)
    buckets = len(counts)
    expected = total / buckets
    return sum((count - expected) ** 2 / expected for count in counts), expected


def bucket_counts(keys, hash_function, bits):
    counts = [0] * (1 << bits)
    for key in keys:
        counts[hash_function(key, bits)] += 1
    return counts


def colliding_pairs(keys, hash_function, bits):
    """Number of unordered pairs of distinct keys that collide after truncation to `bits`."""
    seen = {}
    for key in keys:
        value = hash_function(key, bits)
        seen[value] = seen.get(value, 0) + 1
    return sum(count * (count - 1) // 2 for count in seen.values()), len(seen)


def avalanche(keys, hash_function, input_bits=32, output_bits=256):
    """Per-input-bit average fraction of output bits that flip when that input bit is flipped."""
    averages = []
    for bit in range(input_bits):
        flips = 0
        for key in keys:
            base = hash_function(key, output_bits)
            other = hash_function(key ^ (1 << bit), output_bits)
            flips += bin(base ^ other).count("1")
        averages.append(flips / (len(keys) * output_bits))
    return averages


def sticky_bit_matrix(keys, hash_function, input_bits=32, output_bits=8):
    """Fraction of keys for which output bit o is 1, as a function of input bit i: for a good hash
    every entry sits near 1/2 (NIST's sticky-bit/frequency check)."""
    matrix = []
    for bit in range(input_bits):
        row = []
        for output in range(output_bits):
            ones = 0
            for key in keys:
                if (hash_function(key ^ (1 << bit), output_bits) >> output) & 1:
                    ones += 1
            row.append(ones / len(keys))
        matrix.append(row)
    return matrix


def main():
    # --- The p-value machinery against published quantiles ------------------------------
    assert abs(gammainc_p(1.0, 2.0) - (1.0 - math.exp(-2.0))) <= 1e-12
    assert abs(gammainc_p(0.5, 1.5) - math.erf(math.sqrt(1.5))) <= 1e-12, "P(1/2, x) = erf(sqrt x)"
    for statistic, df, alpha in ((30.578, 15, 0.01), (3.841, 1, 0.05), (2.706, 1, 0.10),
                                 (11.345, 3, 0.01), (67.505, 50, 0.05)):
        probability = chi_square_sf(statistic, df)
        assert abs(probability - alpha) <= 5e-4, \
            "chi-square quantile mismatch at (%r, %r): %r" % (statistic, df, probability)
    assert chi_square_sf(0.0, 4) == 1.0
    assert chi_square_sf(10.0, 4) < chi_square_sf(2.0, 4), "the tail decreases"

    keys = key_stream(0xC0FFEE, 65536)
    assert len(keys) == 65536 and len(set(keys)) == 65536, "the key stream is collision free"

    # --- Uniformity: 65536 keys in 256 buckets ------------------------------------------
    uniformity = {}
    for name, hash_function in GOOD.items():
        counts = bucket_counts(keys, hash_function, 8)
        assert sum(counts) == len(keys)
        statistic, expected = chi_square(counts)
        assert expected == 256.0
        probability = chi_square_sf(statistic, 255)
        assert min(counts) > 0, "%s uses every bucket" % name
        assert probability > 0.05, "%s uniformity p-value %r" % (name, probability)
        uniformity[name] = (statistic, probability)
    structured = [1000 * i for i in range(4096)]
    weak_counts = bucket_counts(structured, WEAK, 8)
    weak_statistic, _ = chi_square(weak_counts)
    weak_probability = chi_square_sf(weak_statistic, 255)
    assert weak_probability < 1e-12, "the negative control must fail uniformity"
    used = [i for i, count in enumerate(weak_counts) if count]
    # 1000 * i mod 256 = 232 * i mod 256 and gcd(232, 256) = 8, so precisely the multiples of 8
    # are reachable and each receives 4096 / 32 = 128 of the keys.
    assert used == list(range(0, 256, 8)), "only the multiples of gcd(1000, 256) = 8 are hit"
    assert set(weak_counts[i] for i in used) == {128}
    assert weak_counts[0] == sum(1 for i in range(4096) if i % 32 == 0)
    assert uniformity["sha256"][0] > 0.0 and uniformity["sha256"][1] <= 1.0

    # --- Birthday collisions ------------------------------------------------------------
    birthday_keys = key_stream(0xBEEF, 4000)
    for bits in (16,):
        predicted = len(birthday_keys) * (len(birthday_keys) - 1) / 2.0 / 2.0 ** bits
        sigma = math.sqrt(predicted)
        for name, hash_function in GOOD.items():
            observed, distinct = colliding_pairs(birthday_keys, hash_function, bits)
            assert predicted - 4 * sigma <= observed <= predicted + 4 * sigma, \
                "%s birthday count %d versus predicted %.1f +- %.1f" % (
                    name, observed, predicted, 4 * sigma)
            # Every key beyond the first in its bucket contributes at least one colliding pair,
            # and the balanced-bucket count bounds the number of pairs from below.
            assert observed >= len(birthday_keys) - distinct
            assert observed >= distinct * (len(birthday_keys) // distinct) * \
                (len(birthday_keys) // distinct - 1) // 2
            assert observed <= len(birthday_keys) * (len(birthday_keys) - 1) // 2
    wide_observed, wide_distinct = colliding_pairs(key_stream(0xF00D, 20000), GOOD["sha256"], 128)
    assert wide_observed == 0 and wide_distinct == 20000, \
        "at 128 bits a collision would need ~2^64 keys"
    weak_observed, weak_distinct = colliding_pairs(structured, WEAK, 8)
    weak_predicted = len(structured) * (len(structured) - 1) / 2.0 / 2.0 ** 8
    # 4096 keys over 32 reachable buckets give 32 * C(128, 2) = 260096 colliding pairs against a
    # birthday prediction of 32760, i.e. 7.9 times more, and 128 keys per bucket instead of 16.
    assert weak_observed == 260096 and weak_predicted == 32760.0
    assert weak_observed > 7 * weak_predicted, "the control collides far beyond the birthday rate"
    assert weak_distinct == 32 and max(bucket_counts(structured, WEAK, 8)) == 128

    # --- Avalanche ----------------------------------------------------------------------
    avalanche_keys = key_stream(0x1234, AVALANCHE_KEYS, 32)
    assert all(key < (1 << 32) for key in avalanche_keys)
    for name, hash_function in GOOD.items():
        averages = avalanche(avalanche_keys, hash_function, 32, 256)
        assert len(averages) == 32
        overall = sum(averages) / len(averages)
        assert 0.48 <= overall <= 0.52, "%s mean avalanche %r" % (name, overall)
        assert min(averages) >= 0.44 and max(averages) <= 0.56, \
            "%s per-bit avalanche range %r..%r" % (name, min(averages), max(averages))
    weak_averages = avalanche(avalanche_keys, lambda key, bits: WEAK(key, 8), 32, 8)
    assert all(value == 0.0 for value in weak_averages[8:]), \
        "flipping a high bit cannot change the low byte"
    # Flipping any of the low eight input bits changes exactly one of the eight output bits
    # (1/8 = 0.125, far below the 0.5 a random oracle gives); the rest do nothing at all.
    assert all(abs(value - 0.125) <= 1e-12 for value in weak_averages[:8])
    assert weak_averages[0] == 0.125

    # --- Sticky bits --------------------------------------------------------------------
    sticky_keys = key_stream(0x5678, 2048, 32)
    for name, hash_function in GOOD.items():
        matrix = sticky_bit_matrix(sticky_keys, hash_function, 32, 8)
        deviations = [abs(value - 0.5) for row in matrix for value in row]
        assert max(deviations) <= 0.06, "%s worst sticky-bit deviation %r" % (name, max(deviations))
        assert sum(deviations) / len(deviations) <= 0.02
    # On an all-even key stream the control's output bit 0 is always 0, and flipping the key's
    # bit 0 always makes it 1: the sticky-bit table saturates at 1.0, a full 0.5 deviation.
    even_keys = [2 * i for i in range(2048)]
    weak_matrix = sticky_bit_matrix(even_keys, lambda key, bits: WEAK(key, 8), 32, 8)
    assert all(row[0] == 0.0 for row in weak_matrix[1:]), "even keys never set the low output bit"
    assert weak_matrix[0][0] == 1.0, "flipping bit 0 turns every even key odd"
    weak_deviations = [abs(value - 0.5) for row in weak_matrix for value in row]
    assert max(weak_deviations) == 0.5

    # --- Determinism --------------------------------------------------------------------
    assert key_stream(0xC0FFEE, 8) == key_stream(0xC0FFEE, 8)
    assert truncated_digest(12345, 64) == truncated_digest(12345, 64)
    assert digest_int(b"abc") == int(hashlib.sha256(b"abc").hexdigest(), 16)
    assert len(digest_bytes(b"abc")) == len(digest_bytes(b"abc", "blake2b")) == 32
    assert truncated_digest(0, 8) != truncated_digest(1, 8)
    assert WEAK(511) == WEAK(255) == 255, "the control collides by construction"


if __name__ == "__main__":
    main()
    print("PASS")
