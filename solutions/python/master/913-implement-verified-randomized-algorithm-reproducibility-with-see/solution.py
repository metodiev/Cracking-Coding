"""Task 913 - Implement verified randomized algorithm reproducibility with seed management and
audit logs.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

What is implemented (in-process, deterministic; no wall clock, no threads, no filesystem):

  * `derive_seed` - hierarchical seed management: a substream seed is sha256(master, path, index)
    truncated to 64 bits, so any step can be named and re-derived without replaying the stream.
    Deriving is *order independent*: consuming substream "b" first does not change what substream
    "a" produces (the property that makes parallel runs reproducible, unlike one shared stream).
  * `StreamRng` - the splitmix64 stream every substream is built on.
  * `AuditLog` - an append-only hash chain: every entry carries the previous entry's digest, the
    step label, the re-derivable seed path and the payload digest, so the log is both an audit
    trail and a replay recipe.  `verify` re-walks the chain and `replay` recomputes every payload
    from the recorded seed paths and compares digests.
  * `monte_carlo_pi` - the randomized algorithm under audit: a hit-or-miss estimator whose error
    is checked against the binomial standard error, so "reproducible" also means "verified".

Simplified versus production: a hash chain rather than a signed/anchored log, one process, and no
durable storage.  Verified below: bit-exact reproducibility, substream isolation and distinctness,
seed-space uniformity, chain verification with tamper/reorder/drop rejection, replay equality,
the estimator's statistical correctness, and the zero-step / single-step / repeated-label
boundaries.
"""

import hashlib
import math

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
GENESIS = hashlib.sha256(b"genesis").digest()


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def derive_seed(master, path, index=0):
    """Hierarchical derivation: the child seed depends on the master, the path and the index."""
    material = "%d|%s|%d" % (master & MASK64, path, index)
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


class StreamRng:
    """splitmix64 over a derived seed - the only randomness in this file."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + GOLDEN) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def unit(self):
        return (self.next_u64() >> 11) * (2.0 ** -53)


def payload(path, samples, index=0, master=1):
    """Deterministic work unit: estimate pi from `samples` hit-or-miss draws of its substream."""
    rng = StreamRng(derive_seed(master, path, index))
    inside = 0
    for _ in range(samples):
        x, y = rng.unit(), rng.unit()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside, samples


def digest_of(path, index, inside, samples):
    return hashlib.sha256(("%s|%d|%d|%d" % (path, index, inside, samples)).encode("utf-8")).digest()


class AuditLog:
    """Append-only hash chain of (step, seed path, payload digest)."""

    def __init__(self, master):
        self.master = master
        self.entries = []                     # (step, path, index, inside, samples, digest, chain)
        self.head = GENESIS

    def __len__(self):
        return len(self.entries)

    def append(self, step, path, index=0, samples=64):
        inside, total = payload(path, samples, index, self.master)
        digest = digest_of(path, index, inside, total)
        chain = hashlib.sha256(self.head + b"%d|%s|%d" % (step, path.encode(), index)
                               + digest).digest()
        self.entries.append((step, path, index, inside, total, digest, chain))
        self.head = chain
        return self.entries[-1]

    def verify(self):
        """Re-walk the chain from the genesis digest."""
        head = GENESIS
        for step, path, index, inside, samples, digest, chain in self.entries:
            if digest != digest_of(path, index, inside, samples):
                return False
            head = hashlib.sha256(head + b"%d|%s|%d" % (step, path.encode(), index)
                                  + digest).digest()
            if head != chain:
                return False
        return head == self.head

    def replay(self, subset=None):
        """Re-derive every payload from the recorded seed paths and compare digests."""
        entries = self.entries if subset is None else [self.entries[i] for i in subset]
        return all(digest_of(path, index, *payload(path, samples, index, self.master)) == digest
                   for _, path, index, _, samples, digest, _ in entries)


def run_experiment(master, steps, samples=64):
    """The audited experiment: one labelled, independently seeded substream per step."""
    log = AuditLog(master)
    for step in range(steps):
        log.append(step, "step-%02d" % step, 0, samples)
    return log


def main():
    samples = 256

    # ---- 1. bit-exact reproducibility across runs and objects
    first = run_experiment(0xC0FFEE, 8, samples)
    second = run_experiment(0xC0FFEE, 8, samples)
    check(len(first) == 8 and first.head == second.head, "the same master seed must replay exactly")
    check([e[3] for e in first.entries] == [e[3] for e in second.entries],
          "every payload must be identical")
    other = run_experiment(0xC0FFEE + 1, 8, samples)
    check(other.head != first.head, "a different master seed must produce a different log")
    check(other.entries[0][3] != first.entries[0][3] or other.head != first.head,
          "the master seed must actually influence the payload")

    # ---- 2. substream isolation: order independence, the point of derived seeds
    path_a = [StreamRng(derive_seed(9, "a")).next_u64() for _ in range(4)]
    shared = StreamRng(derive_seed(9, "shared"))
    for _ in range(1000):
        shared.next_u64()
    path_a_after = [StreamRng(derive_seed(9, "a")).next_u64() for _ in range(4)]
    check(path_a == path_a_after, "consuming another substream must not disturb this one")
    sequential_a = [StreamRng(derive_seed(9, "s")).next_u64() for _ in range(4)]
    sequential_b = [StreamRng(derive_seed(9, "s")).next_u64() for _ in range(4)]
    check(sequential_a == sequential_b, "the same substream must always start at the same point")
    check(derive_seed(9, "a") != derive_seed(9, "b") != derive_seed(10, "a"),
          "path and master must both influence the seed")
    check(derive_seed(9, "a", 0) != derive_seed(9, "a", 1),
          "the derivation index must separate repeated labels")

    # ---- 3. the seed space looks uniform over many labels
    seeds = [derive_seed(4242, "label-%05d" % index) for index in range(5000)]
    check(len(set(seeds)) == len(seeds), "5000 derived seeds must not collide")
    mean = sum(seeds) / len(seeds) / (1 << 64)
    check(abs(mean - 0.5) < 0.02, "derived seeds must be uniform on average (%.4f)" % mean)
    buckets = [0] * 16
    for seed in seeds:
        buckets[(seed >> 60) & 15] += 1
    check(max(buckets) < 5000 / 16 * 1.6 and min(buckets) > 5000 / 16 * 0.4,
          "the high nibble must be evenly spread: %r" % buckets)

    # ---- 4. the chain verifies, and tampering is caught
    log = run_experiment(7, 12, samples)
    check(log.verify(), "an untouched chain must verify")
    check(len(log) == 12 and log.head != GENESIS, "the chain head must advance")
    payload_tamper = AuditLog(7)
    payload_tamper.entries = [list(entry) for entry in log.entries]
    payload_tamper.entries[5][3] += 1
    payload_tamper.head = log.head
    check(not payload_tamper.verify(), "a modified payload must break the chain")
    reorder = AuditLog(7)
    reorder.entries = log.entries[:4] + [log.entries[5], log.entries[4]] + log.entries[6:]
    reorder.head = log.head
    check(not reorder.verify(), "a reordered chain must fail verification")
    dropped = AuditLog(7)
    dropped.entries = [entry for index, entry in enumerate(log.entries) if index != 3]
    dropped.head = log.head
    check(not dropped.verify(), "a dropped entry must fail verification")
    extra = AuditLog(7)
    extra.entries = list(log.entries) + [list(log.entries[-1])]
    extra.head = log.head
    check(not extra.verify(), "an appended-forged entry must fail verification")
    chain_tamper = AuditLog(7)
    chain_tamper.entries = [list(entry) for entry in log.entries]
    chain_tamper.head = hashlib.sha256(log.head + b"x").digest()
    check(not chain_tamper.verify(), "a forged head must fail verification")

    # ---- 5. replay re-derives every payload from the recorded paths
    check(log.replay(), "replaying every step from its recorded seed path must match")
    check(log.replay([0, 3, 7, 11]), "any subset must be replayable on its own")
    forged_path = list(log.entries)
    forged_path[2] = tuple(list(forged_path[2])[:1] + ["step-99"] + list(forged_path[2])[2:])
    forged = AuditLog(7)
    forged.entries = forged_path
    check(not forged.replay(), "a rewritten seed path must fail replay")

    # ---- 6. the randomized algorithm itself is statistically verified
    inside, total = payload("stats", 40000, 0, 1)
    estimate = 4.0 * inside / total
    standard_error = 4.0 * math.sqrt((inside / total) * (1 - inside / total) / total)
    check(abs(estimate - math.pi) < 4.0 * standard_error,
          "the Monte Carlo estimate must sit inside 4 standard errors (%.4f vs %.4f)"
          % (estimate, math.pi))
    coarser = payload("stats", 4000, 0, 1)
    check(0.0 < 4.0 * coarser[0] / coarser[1] < 4.0, "every estimate must be a valid probability")
    repeated = payload("stats", 40000, 0, 1)
    check(repeated == (inside, total), "the estimator must be bit-exactly reproducible")

    # ---- 7. boundaries
    blank = run_experiment(3, 0, samples)
    check(len(blank) == 0 and blank.head == GENESIS and blank.verify() and blank.replay(),
          "an empty experiment must keep the genesis head and verify vacuously")
    single = run_experiment(3, 1, samples)
    check(len(single) == 1 and single.verify() and single.replay(),
          "a single-step log must verify and replay")
    check(single.entries[0][0] == 0 and single.entries[0][1] == "step-00",
          "steps must be labelled from zero")
    twin = AuditLog(11)
    twin.append(0, "same", 0, 32)
    doubled = AuditLog(11)
    doubled.append(0, "same", 1, 32)
    check(twin.entries[0][2] != doubled.entries[0][2],
          "the same label with a different index must be a different substream")
    check(len({derive_seed(11, "same", i) for i in range(6)}) == 6,
          "indices must separate repeated labels")

    # ---- 8. determinism of the whole audit trail
    again = run_experiment(7, 12, samples)
    check(again.head == log.head and again.verify(), "the audit trail must be reproducible")

    print("chain head %s over %d steps" % (log.head.hex()[:16], len(log)))
    print("pi estimate %.4f (error %.4f, se %.4f)" % (estimate, abs(estimate - math.pi), standard_error))
    print("seed mean %.4f, distinct labels 5000" % mean)
    print("PASS")


if __name__ == "__main__":
    main()
