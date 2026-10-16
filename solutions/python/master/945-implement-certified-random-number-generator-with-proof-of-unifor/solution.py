"""Task 945 - Implement certified random number generator with proof of uniformity (small demo).  (Tier: Master, CTCI Beyond - Security & Privacy)

A generator plus a certificate that separates what can be *proved* from what is only *tested*.  The
proof part uses a reduced-width generator (a 12-bit counter with an odd increment, hence full period)
and a debiased range mapping with an explicit rejection rule: over the whole 2^12 input domain exactly
floor(2^12 / n) raw values map to each of the n outputs and exactly 2^12 mod n are rejected, so the
mapping is uniform by exhaustive enumeration rather than by sampling.  The statistical part then runs a
chi-square, a bit-balance and a serial-correlation battery on the 64-bit splitmix64 stream, and the
certificate stores both sections, re-derives the exact histograms on demand, and marks the statistical
results as evidence rather than proof.

Simplified versus NIST SP 800-90B: the exactness claim covers the reduced 12-bit domain only (the
64-bit stream cannot be enumerated), the statistical battery is minimal and no entropy estimator or
health test is implemented.  Verified: exhaustively exact uniformity and rejection counts for
power-of-two and non-power-of-two ranges; the n = 1 and n = 2^12 boundaries; full period of the reduced
generator; the chi-square, bit-balance and serial-correlation statistics; range coverage and mean of
the debiased mapping; reproducibility and seed independence; rejection of invalid ranges; and the
certificate verifying itself while rejecting a tampered copy.
"""
import math

WIDTH = 12
DOMAIN = 1 << WIDTH
MASK12 = DOMAIN - 1
MASK64 = (1 << 64) - 1
CHI_SQUARE_1PC = 21.666  # 9 degrees of freedom, 1% upper tail


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng64:
    """splitmix64: state advances by an odd constant, so the state cycle is the full 2^64."""

    INCREMENT = 0x9E3779B97F4A7C15

    def __init__(self, seed):
        self.state = seed & MASK64

    def next(self):
        self.state = (self.state + self.INCREMENT) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, bound):
        while True:
            accepted, value = debiased(self.next() >> (64 - WIDTH), bound)
            if accepted:
                return value


class Rng12:
    """Reduced-width generator whose whole state space can be enumerated (period 2^12)."""

    INCREMENT = 0x5BD  # odd -> coprime with 2^12 -> full period

    def __init__(self, seed):
        self.state = seed & MASK12

    def next(self):
        self.state = (self.state + self.INCREMENT) & MASK12
        return self.state


def debiased(raw, bound):
    """Map a uniform 12-bit value onto [0, bound); returns (accepted, value)."""
    if not 1 <= bound <= DOMAIN:
        raise ValueError(f"bound {bound} must lie in [1, {DOMAIN}]")
    limit = (DOMAIN // bound) * bound
    if raw < limit:
        return True, raw % bound
    return False, -1


def exhaustive_histogram(bound):
    """Enumerate the whole domain: exact per-bucket counts and exact rejection count."""
    counts = [0] * bound
    rejected = 0
    for raw in range(DOMAIN):
        accepted, value = debiased(raw, bound)
        if accepted:
            counts[value] += 1
        else:
            rejected += 1
    return counts, rejected


def draw_below(rng, bound):
    """The real rejection loop over the reduced generator."""
    while True:
        accepted, value = debiased(rng.next(), bound)
        if accepted:
            return value


class Certificate:
    """Holds the exact (provable) and statistical (evidence-only) sections of the audit."""

    def __init__(self, algorithm, width):
        self.algorithm, self.width = algorithm, width
        self.exact, self.statistical = {}, {}

    def record_exact(self, bound, counts, rejected):
        self.exact[bound] = (tuple(counts), rejected)

    def record_statistical(self, name, value):
        self.statistical[name] = value

    def verify(self):
        """Re-derive every exact claim; the statistical section is deliberately not re-judged."""
        if self.width != WIDTH or not self.exact:
            return False
        for bound, (counts, rejected) in self.exact.items():
            observed, observed_rejected = exhaustive_histogram(bound)
            if tuple(observed) != counts or observed_rejected != rejected:
                return False
        return True


def chi_square(counts):
    total = sum(counts)
    expected = total / len(counts)
    return sum((count - expected) ** 2 / expected for count in counts)


def serial_correlation(values):
    sample_mean = sum(values) / len(values)
    numerator = sum((values[i] - sample_mean) * (values[i + 1] - sample_mean)
                    for i in range(len(values) - 1))
    denominator = sum((value - sample_mean) ** 2 for value in values)
    return numerator / denominator


def main():
    # 1) exact uniformity by exhaustive enumeration, for power-of-two and other ranges.
    exact_claims = {}
    for bound in (2, 3, 5, 6, 7, 16):
        counts, rejected = exhaustive_histogram(bound)
        exact_claims[bound] = (counts, rejected)
        check(counts == [DOMAIN // bound] * bound and rejected == DOMAIN % bound,
              f"bound {bound}: counts must be exactly {DOMAIN // bound} and rejections "
              f"{DOMAIN % bound}, got {counts[:4]}.../{rejected}")
    check(exhaustive_histogram(3)[1] == DOMAIN % 3 == 1
          and exhaustive_histogram(5)[1] == DOMAIN % 5 == 1,
          "the rejection count must be exactly DOMAIN mod bound")

    # 2) boundaries: a single bucket accepts everything, the full width rejects nothing.
    single, single_rejected = exhaustive_histogram(1)
    every, every_rejected = exhaustive_histogram(DOMAIN)
    check(single == [DOMAIN] and single_rejected == 0 and every == [1] * DOMAIN
          and every_rejected == 0,
          "bound 1 and bound 2^12 must both be exactly uniform with no rejections")
    check(all(draw_below(Rng12(9), 1) == 0 for _ in range(100))
          and draw_below(Rng12(9), DOMAIN) == Rng12(9).next(),
          "bound 1 must always yield 0 and bound 2^12 must pass the raw value through")

    # 3) the reduced generator has full period: every state appears exactly once per cycle.
    cycle, state = [], Rng12(1)
    while True:
        state = Rng12(state.next())
        if state.state in cycle:
            break
        cycle.append(state.state)
    check(len(cycle) == DOMAIN and len(set(cycle)) == DOMAIN and Rng12.INCREMENT % 2 == 1
          and Rng64.INCREMENT % 2 == 1,
          f"the reduced generator must have period {DOMAIN}, observed {len(cycle)}")

    # 4) the 64-bit stream passes a chi-square over ten buckets, with the exact part as the proof.
    generator = Rng64(20240607)
    buckets = [0] * 10
    values = [generator.below(1000) for _ in range(40_000)]
    for value in values:
        buckets[value % 10] += 1
    statistic = chi_square(buckets)
    check(0 < statistic < CHI_SQUARE_1PC
          and all(abs(count - 4_000) < 400 for count in buckets),
          f"chi-square {statistic:.3f} must stay below {CHI_SQUARE_1PC} over 40k draws")

    # 5) bit balance and serial independence of the same stream.
    highs = sum(1 for value in values if value >= 500)
    correlation = serial_correlation(values)
    sigma = math.sqrt(40_000 * 0.25)
    check(abs(highs - 20_000) < 3 * sigma and abs(correlation) < 0.02,
          f"bit balance {highs} (3 sigma {3 * sigma:.0f}) and correlation {correlation:.4f}")

    # 6) the debiased mapping covers the whole range with the expected mean.
    spread = [draw_below(Rng12(seed), 1000) for seed in range(DOMAIN) for _ in range(1)]
    check(len(set(spread)) == 1000 and abs(sum(spread) / len(spread) - 499.5) < 60,
          f"range coverage {len(set(spread))}/1000, mean {sum(spread) / len(spread):.1f}")

    # 7) results are reproducible per seed and independent across seeds.
    twin = Rng64(20240607)
    check([twin.below(1000) for _ in range(50)] == values[:50]
          and [Rng64(1).next() for _ in range(3)] != [Rng64(2).next() for _ in range(3)],
          "the same seed must reproduce the stream and different seeds must diverge")

    # 8) invalid bounds are rejected instead of silently producing a biased value.
    for bound in (0, -3, DOMAIN + 1):
        try:
            debiased(0, bound)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bound {bound} must be rejected")

    # 9) the certificate verifies itself, records which claims are statistical, and rejects a
    #    tampered copy.
    certificate = Certificate("splitmix64+debiased12", WIDTH)
    for bound, (counts, rejected) in exact_claims.items():
        certificate.record_exact(bound, counts, rejected)
    certificate.record_statistical("chi_square", statistic)
    certificate.record_statistical("serial_correlation", correlation)
    tampered = Certificate("splitmix64+debiased12", WIDTH)
    tampered.exact = dict(certificate.exact)
    tampered.exact[3] = ((1, 2, 3), 0)
    check(certificate.verify() and certificate.exact and certificate.statistical
          and certificate.width == WIDTH and not tampered.verify(),
          "the certificate must verify the exact section and reject a tampered one")

    print(f"exact: bounds={sorted(exact_claims)} rejections="
          f"{ {bound: claim[1] for bound, claim in sorted(exact_claims.items())} }")
    print(f"statistical: chi_square={statistic:.3f} (<{CHI_SQUARE_1PC}) highs={highs} "
          f"correlation={correlation:.4f}")
    print("PASS")


if __name__ == "__main__":
    main()
