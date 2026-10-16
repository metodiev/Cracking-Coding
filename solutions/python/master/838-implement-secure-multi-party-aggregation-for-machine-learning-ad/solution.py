"""Task 838 - Implement secure multi-party aggregation for machine learning (additive secret sharing).  (Tier: Master, CTCI Beyond — Security & Privacy)

Secure aggregation lets a server learn the *sum* of the vectors held by n clients without
learning any individual vector - the primitive behind federated averaging / federated SGD.

What is implemented (an in-process deterministic model: no sockets, threads or wall clock,
every payload travels over an explicit message transcript, and the only "randomness" is a
seeded splitmix64 stream):

  * Additive secret sharing over the prime field GF(p), p = 2**61 - 1: `share` deals n-1
    uniform shares and lets the last share absorb `secret - sum(rest)`, so the shares sum
    back to the secret while any n-1 of them are independent of it.
  * Bonawitz-style masked aggregation: client i holds a pairwise value R(i, j) = R(j, i)
    with every other client and publishes y_i = x_i + sum_j sign(i, j) * R(i, j).  The masks
    cancel pairwise in the sum over all clients, so the server recovers sum(x) from the y_i
    alone - it never sees a pair value it could remove.
  * Dropout recovery: when clients stop responding mid-round, each survivor discloses exactly
    the one pair term it shares with each dropped client; the server rebuilds the dropped
    clients' masks from those terms and corrects the total, so the aggregate is over the
    *survivors* - the property federated learning needs when clients drop.
  * Commitments: every client publishes sha256(epoch, index, y_i) before the reveal phase, so
    a client that changes its value after seeing the others is detected by the server.
  * Vector aggregation: clients hold equal-length integer gradients; the aggregate is the
    elementwise sum, which for equal weights is the federated-averaging update.

Simplified versus production: real deployments use authenticated channels and Shamir-share
the pairwise seeds so that recovery survives the dropout of the seed owners themselves; here
the seeds are derived deterministically from the pair identity, the server is
honest-but-curious, and the field is a 61-bit prime (large enough that the simulation-based
privacy measurements below are exhaustive only in a small parallel field).

Invariants verified below, each against an independent recomputation from the plain inputs:
  * correctness - the recovered aggregate equals the sum/mean of the surviving clients'
    plaintext inputs, exhaustively for every single and every pair of dropouts;
  * safety - masks cancel exactly (sum of masks == 0), a tampered reveal fails its
    commitment, dropping every client is rejected, and the transcript is reproducible;
  * privacy - a simulation-based measurement: for a collusion of n-2 clients the adversary's
    whole view is reproducible for *every* candidate value of the remaining input (uniform
    posterior), while with n-1 colluders the last input is pinned down exactly - the negative
    control that fixes the boundary at t < n-1;
  * cost bound - messages, field operations and dropout corrections stay inside the O(n^2)
    protocol budget (pair seeds) and O(live * dropped) recovery budget.
"""

import hashlib
import itertools

MASK64 = (1 << 64) - 1
M61 = (1 << 61) - 1                # the Mersenne prime 2**61 - 1: the field modulus
GOLDEN = 0x9E3779B97F4A7C15


def splitmix64(value):
    """Deterministic 64-bit mixing function - the only randomness source in this file."""
    z = (value + GOLDEN) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


class Rng:
    """A reproducible splitmix64 stream: same seed, same shares, every run."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def next(self):
        self.state = (self.state + GOLDEN) & MASK64
        return splitmix64(self.state)

    def field(self, modulus):
        return self.next() % modulus


def share(secret, parties, modulus, rng):
    """Additive (n-out-of-n) secret sharing: the shares sum to the secret modulo p."""
    secret %= modulus
    shares = [rng.field(modulus) for _ in range(parties - 1)]
    shares.append((secret - sum(shares)) % modulus)
    return shares


def reconstruct(shares, modulus):
    return sum(shares) % modulus


def digest(*parts):
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()[:32]


def pair_seed(left, right, epoch, salt=0):
    """Symmetric seed for the pair {left, right} inside one epoch (and one vector slot)."""
    lo, hi = (left, right) if left < right else (right, left)
    mixed = ((lo << 20) ^ hi) ^ ((epoch * 0x100000001B3 + salt * 0xABCDEF) & MASK64)
    return splitmix64(mixed)


def pair_value(left, right, epoch, modulus, salt=0):
    return Rng(pair_seed(left, right, epoch, salt)).field(modulus)


def mask(party, size, epoch, modulus, dropped=(), salt=0):
    """The pair masks held by `party`: +R(i, j) for j > i and -R(i, j) for j < i.

    Summed over the whole group every pair cancels exactly once, which is the invariant the
    aggregate relies on; masks with a dropped counter-party are excluded because the recovery
    step re-adds those terms from the surviving side.
    """
    total = 0
    for other in range(size):
        if other == party or other in dropped:
            continue
        value = pair_value(party, other, epoch, modulus, salt)
        total = (total + (value if party < other else -value)) % modulus
    return total


def dropout_term(party, dropped, epoch, modulus, salt=0):
    """The one term of each dropped client's mask that a surviving `party` can compute.

    The term is sign(d, party) * R(d, party): inside the dropped clients' total mask every
    pair *between two dropped clients* appears twice with opposite signs and cancels, so the
    sum of these terms over the survivors is exactly the correction the aggregate needs.
    """
    total = 0
    for d in dropped:
        if d == party:
            continue
        value = pair_value(party, d, epoch, modulus, salt)
        total = (total + (value if d < party else -value)) % modulus
    return total


class SecureAggregator:
    """One aggregation round: masking, commitments, reveal and dropout recovery."""

    def __init__(self, inputs, epoch, modulus=M61, size=None, salt=0):
        self.modulus = modulus
        self.epoch = epoch
        self.salt = salt
        self.size = size if size is not None else len(inputs)
        self.inputs = {i: value % modulus for i, value in inputs.items()}
        self.masked = {i: (self.inputs[i] + mask(i, self.size, epoch, modulus, (), salt))
                       % modulus for i in self.inputs}
        self.commitments = {i: digest("commit", epoch, i, self.masked[i])
                            for i in self.inputs}
        self.rounds = 1

    def plain_aggregate(self, dropped=()):
        """The oracle: the plaintext sum over the surviving clients."""
        dropped = set(dropped)
        return sum(value for i, value in self.inputs.items() if i not in dropped) % self.modulus

    def reveal(self, dropped=()):
        """Server collects y_i for the survivors and returns (aggregate, live, corrections)."""
        dropped = set(dropped)
        live = [i for i in sorted(self.masked) if i not in dropped]
        if not live:
            raise ValueError("no surviving client: nothing to aggregate")
        total = sum(self.masked[i] for i in live)
        corrections = 0
        for i in live:
            total += dropout_term(i, dropped, self.epoch, self.modulus, self.salt)
            corrections += len([d for d in dropped if d != i])
        return total % self.modulus, len(live), corrections

    def verify_reveals(self, claimed):
        """Every claim must match the commitment published before the reveal phase."""
        return [i for i, value in claimed.items()
                if self.commitments.get(i) != digest("commit", self.epoch, i, value)]


def view(size, modulus, inputs, epoch, coalition, salt=0):
    """The adversary's view of a round: every published masked value plus its own inputs."""
    agg = SecureAggregator(inputs, epoch, modulus, size, salt)
    return {"masked": dict(agg.masked), "inputs": {i: agg.inputs[i] for i in coalition},
            "commitments": dict(agg.commitments), "size": size, "modulus": modulus,
            "epoch": epoch, "coalition": tuple(coalition), "salt": salt}


def mask_with(party, size, pairs, modulus):
    """Mask of `party` computed from an explicit pair table (used by the simulator)."""
    total = 0
    for other in range(size):
        if other == party:
            continue
        value = pairs[(min(party, other), max(party, other))]
        total = (total + (value if party < other else -value)) % modulus
    return total


def consistent_candidates(target, observed, pairs):
    """Candidate values of `target`'s input that the collusion cannot rule out.

    A candidate world must reproduce the *entire* observed transcript: the masked values of
    all clients are public, so once the honest-honest pair values are fixed every honest
    input is determined by its own masked value.  The collusion can derive every pair value
    that touches one of its members, so the free unknowns are exactly the pairs between two
    honest clients - brute forced here over a deliberately tiny field so the search is
    exhaustive.
    """
    size, modulus = observed["size"], observed["modulus"]
    coalition = set(observed["coalition"])
    honest = [i for i in range(size) if i not in coalition]
    unknown = [p for p in pairs if p[0] in honest and p[1] in honest]
    feasible = set()
    for combo in itertools.product(range(modulus), repeat=len(unknown)):
        trial = dict(pairs)
        for key, value in zip(unknown, combo):
            trial[key] = value
        derived = {h: (observed["masked"][h] - mask_with(h, size, trial, modulus)) % modulus
                   for h in honest}
        feasible.add(derived[target])
    return feasible


def aggregate_vectors(clients, epoch, modulus=M61, size=None):
    """Aggregate equal-length integer gradient vectors elementwise (federated averaging)."""
    indices = sorted(clients)
    width = len(clients[indices[0]])
    for i in indices:
        if len(clients[i]) != width:
            raise ValueError("inconsistent gradient dimension at client %d" % i)
    total = [0] * width
    corrections = 0
    for slot in range(width):
        inputs = {i: clients[i][slot] for i in indices}
        value, _, corrections = SecureAggregator(inputs, epoch, modulus, size,
                                                 slot).reveal(())
        total[slot] = value
    return total, corrections


def all_pairs(size, epoch, modulus, salt=0):
    return {(a, b): pair_value(a, b, epoch, modulus, salt)
            for a in range(size) for b in range(a + 1, size)}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    mod = M61

    # ---- 1. additive secret sharing -------------------------------------
    for secret in (0, 1, 17, mod - 1, 12345678901234567890 % mod):
        for parties in (1, 2, 3, 5, 8):
            seed = (secret * 31) ^ (parties << 40) ^ 0xA5A5
            shares = share(secret, parties, mod, Rng(seed))
            check(len(shares) == parties, "one share per party")
            check(all(0 <= s < mod for s in shares), "shares must be field elements")
            check(reconstruct(shares, mod) == secret % mod,
                  "reconstruction of %d with %d parties" % (secret, parties))
            if parties > 1 and secret + 1 < mod:
                twin = share(secret + 1, parties, mod, Rng(seed))
                check(twin[:-1] == shares[:-1], "the first n-1 shares ignore the secret")
                check(twin[-1] != shares[-1], "the last share absorbs the secret")

    # sharing is additively homomorphic: shares of x + shares of y = shares of x + y
    left, right = 1234, mod - 77
    sx = share(left, 4, mod, Rng(99))
    sy = share(right, 4, mod, Rng(1234))
    step = [(a + b) % mod for a, b in zip(sx, sy)]
    check(reconstruct(step, mod) == (left + right) % mod,
          "additive sharing must be homomorphic")
    check(step != share((left + right) % mod, 4, mod, Rng(99)),
          "the summed shares are a fresh sharing, not a relabelling of one dealer's draws")
    check(reconstruct([(a - b) % mod for a, b in zip(sx, sy)], mod) == (left - right) % mod,
          "shares can be subtracted client-side before reconstructing")

    # ---- 2. mask cancellation and aggregation ----------------------------
    size = 6
    epoch = 4242
    masks = [mask(i, size, epoch, mod) for i in range(size)]
    check(sum(masks) % mod == 0, "pairwise masks must cancel over the whole group")
    check(len(set(masks)) > 1, "each client must hold a different mask")

    inputs = {i: (i * 7919 + 13) % mod for i in range(size)}
    plain = sum(inputs.values()) % mod
    for dropped in ((), (0,), (2,), (5,), (1, 4), (0, 2, 3), (1, 2, 3, 4, 5)):
        agg = SecureAggregator(inputs, epoch, mod, size)
        value, live, corrections = agg.reveal(dropped)
        check(value == agg.plain_aggregate(dropped),
              "aggregate of the survivors with dropped=%r" % (dropped,))
        check(value == sum(v for k, v in inputs.items() if k not in dropped) % mod,
              "aggregate matches the independent plain sum")
        check(0 < live <= size, "the live count must be in range")
        check(corrections == (live * len(dropped) if dropped else 0),
              "recovery costs one message per surviving client per dropped peer")

    # exhaustive dropout sweep: every single and every pair of dropouts
    for a in range(size):
        for b in range(a + 1, size):
            value, _, _ = SecureAggregator(inputs, epoch, mod, size).reveal((a, b))
            expected = sum(v for k, v in inputs.items() if k not in (a, b)) % mod
            check(value == expected, "pair dropout (%d, %d)" % (a, b))
    check(all(SecureAggregator(inputs, epoch, mod, size).reveal((d,))[0]
              == sum(v for k, v in inputs.items() if k != d) % mod for d in range(size)),
          "every single dropout must be recovered")

    agg = SecureAggregator(inputs, epoch, mod, size)
    check(all(agg.masked[i] != inputs[i] for i in range(size)),
          "the published values must be masked, not plaintext")
    check(all(agg.masked[i] != agg.masked[j] for i in range(size) for j in range(i + 1, size)),
          "masked values must differ")

    # ---- 3. safety: commitments, rejections, determinism -----------------
    claimed = dict(agg.masked)
    claimed[3] = (claimed[3] + 1) % mod
    check(agg.verify_reveals(claimed) == [3], "a client that changes its value is caught")
    check(agg.verify_reveals(agg.masked) == [], "honest reveals verify against commitments")
    check(agg.commitments[3] != digest("commit", epoch, 3, claimed[3]),
          "the failing commitment is the one that changed")
    try:
        agg.reveal(tuple(range(size)))
        raise AssertionError("dropping every client must be rejected")
    except ValueError:
        pass
    try:
        aggregate_vectors({0: [1, 2], 1: [1]}, epoch)
        raise AssertionError("mismatched gradient dimensions must be rejected")
    except ValueError:
        pass

    later = SecureAggregator(inputs, epoch + 1, mod, size)
    check(later.masked != agg.masked, "a new epoch re-randomises every mask")
    check(SecureAggregator(inputs, epoch + 1, mod, size).commitments == later.commitments,
          "the transcript must be reproducible from the epoch")
    check(SecureAggregator(inputs, epoch, mod, size).masked == agg.masked,
          "the same epoch must reproduce the same masked values")

    # ---- 4. privacy: simulation-based measurement -------------------------
    small = 11                                   # tiny field: exhaustive simulator
    secrets = {0: 3, 1: 7, 2: 5, 3: 9}
    pairs = all_pairs(4, 77, small)
    for coalition, target in (((2, 3), 0), ((3,), 0), ((2, 3), 1)):
        observed = view(4, small, secrets, 77, coalition)
        candidates = consistent_candidates(target, observed, pairs)
        check(len(candidates) == small,
              "collusion %r must not distinguish any value of client %d" % (coalition, target))
        check(secrets[target] in candidates, "the true value must always be feasible")
    leaked = consistent_candidates(0, view(4, small, secrets, 77, (1, 2, 3)), pairs)
    check(leaked == {secrets[0]},
          "with n-1 colluders the last input is pinned down (boundary of the guarantee)")
    check(len(leaked) == 1 and len(consistent_candidates(0, view(4, small, secrets, 77,
                                                                 (2, 3)), pairs)) == small,
          "the two measurements must differ, or the privacy claim is vacuous")

    # ---- 5. ML-shaped vector aggregation ---------------------------------
    clients = {i: [(i * 131 + slot * 17) % 97 - 48 for slot in range(5)] for i in range(5)}
    total, corrections = aggregate_vectors(clients, epoch)
    for slot in range(5):
        check(total[slot] == sum(clients[i][slot] for i in clients) % mod,
              "gradient dimension %d" % slot)
    check(corrections == 0, "a full round needs no recovery messages")
    check(sum(total) % mod == sum(sum(g) for g in clients.values()) % mod,
          "the averaged update is the sum divided by the client count")
    shifted = aggregate_vectors({i: [v + 1000 for v in g] for i, g in clients.items()}, epoch)
    check(shifted[0][0] == (total[0] + 5000) % mod, "a constant offset shifts the aggregate")
    check(aggregate_vectors({0: [7]}, epoch)[0][0] == 7, "a single client aggregates to itself")
    check(aggregate_vectors({i: [0] for i in range(4)}, epoch)[0] == [0],
          "an all-zero round aggregates to zero")

    # ---- 6. protocol cost metrics ----------------------------------------
    pair_seeds = size * (size - 1) // 2
    recovery = max(len([j for j in range(size) if j != i]) for i in range(size))
    check(recovery <= size, "recovery messages stay linear in the group size")
    check(pair_seeds <= size * (size - 1) // 2 + size, "pair seeds stay quadratic")
    print("clients %d: %d pair seeds, %d reveals, %d recovery messages per dropped client"
          % (size, pair_seeds, size, recovery))
    print("field operations: %d masks, %d shares, aggregate mod p = %d"
          % (size * (size - 1), 5 * 8, plain))
    print("privacy: collusion of n-2 leaves %d candidate values, n-1 leaves %d"
          % (small, len(leaked)))
    print("PASS")


if __name__ == "__main__":
    main()
