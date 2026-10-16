"""Task 713 - Implement secure aggregation protocol sketch for federated averaging.  (Tier: Expert, CTCI Beyond - Security & Privacy)

Secure aggregation (Bonawitz et al.) lets a server learn only the *sum* of a round's client
updates.  The trick is pairwise one-time pads that cancel by construction: clients i < j share a
pseudorandom mask vector r_ij, client i adds +r_ij for every j > i and subtracts r_ji for every
j < i, so

    mask_i = sum_{j>i} r_ij - sum_{j<i} r_ji     and     sum_i mask_i = 0.

Every client then uploads y_i = x_i + mask_i modulo a large prime; the server's sum of the y_i is
exactly the plaintext sum, while no individual y_i can be separated from its pad.  When clients
drop out mid-round the surviving clients reveal (only) the seeds they shared with the dropouts,
which lets the server subtract the masks that can no longer cancel -- the recovery step that makes
the protocol robust.

This implementation models the pairwise seeds directly (a real deployment agrees on them with
Diffie-Hellman, which is out of scope here) and keeps vectors as fixed-point integers modulo the
61-bit Mersenne prime 2^61 - 1, so the arithmetic is exact.  Simplifications, stated plainly: the
seeds are assumed pre-shared, there is no authentication or commitment of the uploads, and the
protocol hides individual updates from the *server* only -- any n-1 colluding clients can of
course recover the remaining one, which the checks document rather than paper over.

Verified: mask cancellation sum_i mask_i = 0 for every configuration, the aggregate equals the
plaintext sum for random and boundary inputs, masked uploads differ from the inputs and from each
other while their sum is unchanged, recovery after 1 and n-1 dropouts reproduces the survivor sum,
the fixed-point encode/decode round trip stays inside its rounding bound, and the degenerate
shapes (no clients, one client, zero vectors, negative values) behave.
"""
import math
import sys

MASK64 = (1 << 64) - 1
PRIME = (1 << 61) - 1                    # Mersenne prime: exact fixed-point arithmetic
SCALE_BITS = 16
SCALE = 1 << SCALE_BITS


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def hash_bytes(payload, seed=0x9E3779B97F4A7C15):
    """Deterministic 128-bit hash; only splitmix64's mixed output feeds the digest."""
    state = seed
    low = 0
    high = 0
    for index, byte in enumerate(payload):
        state, word = splitmix64(state + byte + index * 0x1F123BB5)
        low ^= word
        high = (high * 0x100000001B3 + word) & MASK64
    state, word = splitmix64(state ^ low)
    low = (low ^ word) & MASK64
    state, word = splitmix64(state ^ high)
    high = (high ^ word) & MASK64
    return (high << 64) | low


def encode(value):
    """Fixed-point encoding of a real number into [0, PRIME)."""
    return int(round(value * SCALE)) % PRIME


def decode(value):
    """Inverse of encode; callers must keep the total below PRIME/2 to avoid the wraparound."""
    value %= PRIME
    if value > PRIME // 2:
        value -= PRIME
    return value / SCALE


class SecureAggregation:
    """Pairwise-mask secure aggregation over d-dimensional fixed-point vectors."""

    def __init__(self, clients, dimension, master_seed=0xFEED):
        check(clients >= 1, "there must be at least one client")
        check(dimension >= 1, "the vector dimension must be positive")
        self.clients = clients
        self.dimension = dimension
        self.master_seed = master_seed
        self._pair_masks = {}

    def pair_mask(self, first, second):
        """The pad shared by two clients; symmetric because the ids are sorted."""
        low, high = min(first, second), max(first, second)
        key = (low, high)
        if key not in self._pair_masks:
            seed = hash_bytes(f"seed:{low}:{high}:{self.master_seed}".encode("ascii"))
            mask = [hash_bytes(f"mask:{low}:{high}:{index}".encode("ascii"), seed=seed) % PRIME
                    for index in range(self.dimension)]
            self._pair_masks[key] = mask
        return self._pair_masks[key]

    def mask(self, client):
        """mask_i = sum_{j>i} r_ij - sum_{j<i} r_ji, element-wise modulo PRIME."""
        total = [0] * self.dimension
        for other in range(self.clients):
            if other == client:
                continue
            contribution = self.pair_mask(client, other)
            sign = 1 if other > client else -1
            for index in range(self.dimension):
                total[index] = (total[index] + sign * contribution[index]) % PRIME
        return total

    def mask_sum(self, subset=None):
        total = [0] * self.dimension
        for client in (range(self.clients) if subset is None else subset):
            vector = self.mask(client)
            for index in range(self.dimension):
                total[index] = (total[index] + vector[index]) % PRIME
        return total

    def upload(self, vector, client):
        """y_i = x_i + mask_i mod PRIME."""
        check(len(vector) == self.dimension, "the vector must match the declared dimension")
        pad = self.mask(client)
        return [(value + pad[index]) % PRIME for index, value in enumerate(vector)]

    def recovery_share(self, client, dropped):
        """What a surviving client reveals for a dropped peer: exactly the pads they shared."""
        share = [0] * self.dimension
        for other in sorted(set(dropped)):        # a client can only drop once
            if other == client:
                continue
            contribution = self.pair_mask(client, other)
            sign = 1 if client < other else -1
            for index in range(self.dimension):
                share[index] = (share[index] + sign * contribution[index]) % PRIME
        return share

    def aggregate(self, vectors, dropped=()):
        """Run one round; returns (server total, individual uploads, corrections applied)."""
        check(len(vectors) == self.clients, "one vector per client is required")
        survivors = [client for client in range(self.clients) if client not in dropped]
        uploads = {client: self.upload(vectors[client], client) for client in survivors}
        total = [0] * self.dimension
        for upload in uploads.values():
            for index in range(self.dimension):
                total[index] = (total[index] + upload[index]) % PRIME
        correction = [0] * self.dimension
        for client in survivors:
            share = self.recovery_share(client, dropped)
            for index in range(self.dimension):
                correction[index] = (correction[index] + share[index]) % PRIME
        result = [(total[index] - correction[index]) % PRIME for index in range(self.dimension)]
        return result, uploads, correction

    def plaintext_sum(self, vectors, dropped=()):
        total = [0] * self.dimension
        for client in range(self.clients):
            if client in dropped:
                continue
            for index, value in enumerate(vectors[client]):
                total[index] = (total[index] + value) % PRIME
        return total


# --------------------------------------------------------------------------- checks
def test_mask_cancellation():
    for clients in (1, 2, 3, 7):
        aggregation = SecureAggregation(clients, 5, master_seed=0xC0FFEE)
        masks = [aggregation.mask(client) for client in range(clients)]
        total = [0] * 5
        for mask in masks:
            for index in range(5):
                total[index] = (total[index] + mask[index]) % PRIME
        check(total == [0] * 5, f"{clients} clients: the masks must cancel exactly, got {total}")
        check(aggregation.mask_sum() == [0] * 5, "the mask-sum helper agrees")
        if clients >= 2:
            for mask in masks:
                check(any(value != 0 for value in mask), "no client's own mask may be all zeros")
            check(masks[0] != masks[1], "different clients carry different masks")
    return aggregation


def test_pairwise_mask_symmetry():
    aggregation = SecureAggregation(6, 4)
    for first in range(6):
        for second in range(6):
            if first == second:
                continue
            check(aggregation.pair_mask(first, second) == aggregation.pair_mask(second, first),
                  f"the pad of ({first},{second}) must be symmetric")
    first = aggregation.pair_mask(0, 1)
    other = SecureAggregation(6, 4, master_seed=0xC0FFEE).pair_mask(0, 1)
    check(first != other, "a different master seed gives a different pad")
    check(all(0 <= value < PRIME for value in first), "pads live in the field")


def test_aggregate_matches_plaintext():
    rng = SecureAggregation(5, 6, master_seed=0x1234)
    for trial in range(8):
        vectors = []
        for client in range(5):
            vector = [encode(((hash_bytes(f"{trial}-{client}-{k}".encode("ascii")) % 2001) - 1000)
                             / 4.0) for k in range(6)]
            vectors.append(vector)
        total, uploads, correction = rng.aggregate(vectors)
        expected = rng.plaintext_sum(vectors)
        check(total == expected, f"trial {trial}: server total != plaintext sum")
        check(correction == [0] * 6, "with no dropouts no correction is applied")
        for client, upload in uploads.items():
            check(upload != vectors[client],
                  f"trial {trial}: client {client}'s upload must be masked")
            difference = [(upload[k] - vectors[client][k]) % PRIME for k in range(6)]
            check(difference == rng.mask(client), "the upload differs by exactly the mask")
    return rng


def test_hiding_property():
    aggregation = SecureAggregation(4, 3, master_seed=0xBEEF)
    vector = [encode(1.5), encode(-2.25), encode(0.125)]
    uploads = [aggregation.upload(vector, client) for client in range(4)]
    check(len({tuple(upload) for upload in uploads}) == 4,
          "two clients with identical inputs must still upload different values")
    check(all(upload != vector for upload in uploads), "no upload equals the input")
    others = SecureAggregation(4, 3, master_seed=0xFACE)
    check(others.upload(vector, 0) != uploads[0],
          "a different mask set gives a different upload")
    check(others.mask_sum() == [0] * 3, "and still cancels")
    # two runs with different masks produce different uploads but the same total
    vectors = [[encode(1.0)] * 3 for _ in range(4)]
    first, _, _ = aggregation.aggregate(vectors)
    second, _, _ = others.aggregate(vectors)
    check(first == second == [(4 * encode(1.0)) % PRIME] * 3, "the aggregate is mask-independent")
    # a single client's upload is the only thing that changes when its vector changes
    changed = [list(vector) for vector in vectors]
    changed[2] = [encode(2.0)] * 3
    moved, uploads_moved, _ = aggregation.aggregate(changed)
    check(moved != first, "changing a client changes the total")
    for client in range(4):
        moved_flag = uploads_moved[client] != aggregation.upload(vectors[client], client)
        check(moved_flag == (client == 2),
              f"only client 2's upload should move, but client {client} did "
              f"{'move' if moved_flag else 'stay'}")


def test_fixed_point_round_trip():
    values = [0.0, 1.0, -1.0, 0.5, -0.25, 123.456, -987.654, 1e-4]
    for value in values:
        recovered = decode(encode(value))
        check(abs(recovered - value) <= 0.5 / SCALE + 1e-12,
              f"the fixed-point round trip must stay inside half a step for {value}")
    aggregation = SecureAggregation(3, 2, master_seed=0x55)
    vectors = [[encode(1.25), encode(-0.5)],
               [encode(0.75), encode(2.0)],
               [encode(-3.0), encode(0.5)]]
    total, _, _ = aggregation.aggregate(vectors)
    decoded = [decode(value) for value in total]
    truth = [1.25 + 0.75 - 3.0, -0.5 + 2.0 + 0.5]
    for got, want in zip(decoded, truth):
        check(abs(got - want) <= 3 * 0.5 / SCALE + 1e-12,
              f"the decoded aggregate {got} should be {want} within the rounding bound")


def test_dropouts_recovery():
    aggregation = SecureAggregation(6, 4, master_seed=0xABC)
    vectors = [[encode((client + 1) * 0.5)] * 4 for client in range(6)]
    for dropped in ([], [0], [3, 4], [1, 2, 3, 4, 5]):
        dropped = tuple(dropped)
        total, uploads, correction = aggregation.aggregate(vectors, dropped=dropped)
        expected = aggregation.plaintext_sum(vectors, dropped=dropped)
        check(total == expected,
              f"dropouts {dropped}: the recovered aggregate must match the survivor sum")
        check(sorted(uploads) == sorted(set(range(6)) - set(dropped)),
              "only survivors upload")
        if not dropped:
            check(correction == [0] * 4, "no dropouts means no correction")
        else:
            check(any(value != 0 for value in correction),
                  f"dropouts {dropped} must require a non-trivial correction")

    # with every client but one gone, the aggregate is that client's own vector
    total, uploads, _ = aggregation.aggregate(vectors, dropped=tuple(range(5)))
    check(list(uploads) == [5], "only the last client survives")
    check(total == vectors[5], "and the server learns exactly its vector (nothing to hide)")
    expected = [decode(value) for value in total]
    check(all(abs(value - 3.0) < 1e-9 for value in expected),
          f"the surviving client's value is 3.0, got {expected}")

    # a dropped client that never uploaded cannot disturb the others
    total, _, _ = aggregation.aggregate(vectors, dropped=(0, 0, 1))
    check(total == aggregation.plaintext_sum(vectors, dropped=(0, 1)), "duplicate ids in the set")

    other_key = SecureAggregation(4, 2, master_seed=0x777)
    small = [[encode(client + 0.5)] * 2 for client in range(4)]
    total, _, _ = other_key.aggregate(small, dropped=(2,))
    check(total == other_key.plaintext_sum(small, dropped=(2,)),
          "a single dropout recovers correctly with a different key and dimension")


def test_boundaries_and_validation():
    single = SecureAggregation(1, 3, master_seed=0x1)
    check(single.mask(0) == [0, 0, 0], "a single client has no one to share a pad with")
    total, uploads, _ = single.aggregate([[encode(1.0), encode(2.0), encode(3.0)]])
    check(uploads[0] == [encode(1.0), encode(2.0), encode(3.0)],
          "with one client the upload is the plaintext vector")
    check([decode(value) for value in total] == [1.0, 2.0, 3.0], "and the aggregate is it")

    zero = SecureAggregation(3, 2, master_seed=0x2)
    total, _, _ = zero.aggregate([[0, 0] for _ in range(3)])
    check(total == [0, 0], "zero vectors aggregate to zero")

    total, _, _ = zero.aggregate([[encode(-4.0), encode(-0.5)]] * 3)
    check([decode(value) for value in total] == [-12.0, -1.5],
          "negative values survive the field arithmetic")

    total, _, _ = zero.aggregate([[PRIME - 1, 1], [1, PRIME - 1], [0, 0]])
    check(total == [0, 0], "values near the modulus wrap correctly")

    try:
        SecureAggregation(0, 3)
        raise AssertionError("zero clients must be rejected")
    except AssertionError as error:
        check("at least one client" in str(error), f"unexpected message: {error}")
    try:
        SecureAggregation(3, 0)
        raise AssertionError("a zero dimension must be rejected")
    except AssertionError as error:
        check("dimension" in str(error), f"unexpected message: {error}")
    try:
        zero.aggregate([[0, 0]])
        raise AssertionError("the wrong number of vectors must be rejected")
    except AssertionError as error:
        check("one vector per client" in str(error), f"unexpected message: {error}")
    try:
        zero.upload([1, 2, 3], 0)
        raise AssertionError("a vector of the wrong length must be rejected")
    except AssertionError as error:
        check("declared dimension" in str(error), f"unexpected message: {error}")

    first, _, _ = zero.aggregate([[encode(1.0), encode(2.0)]] * 3)
    second, _, _ = SecureAggregation(3, 2, master_seed=0x2).aggregate(
        [[encode(1.0), encode(2.0)]] * 3)
    check(first == second, "the protocol is deterministic for a fixed master seed")
    check(PRIME == (1 << 61) - 1, "the arithmetic field is the 61-bit Mersenne prime")
    check(math.log2(PRIME) > 60.9, "the field is large enough for the demo's pad space")


def test_federated_averaging_use_case():
    aggregation = SecureAggregation(8, 5, master_seed=0xFED)
    rng_state = 0x12345678
    vectors = []
    for _ in range(8):
        vector = []
        for _ in range(5):
            rng_state, word = splitmix64(rng_state)
            vector.append(encode((word % 20001 - 10000) / 10000.0))
        vectors.append(vector)
    total, _, _ = aggregation.aggregate(vectors)
    plain = aggregation.plaintext_sum(vectors)
    check(total == plain, "the averaged update must come from the true sum")
    decoded = [decode(value) for value in total]
    average = [value / 8 for value in decoded]
    truth = [0.0] * 5
    for vector in vectors:
        for index in range(5):
            truth[index] += decode(vector[index]) / 8
    for got, want in zip(average, truth):
        check(abs(got - want) < 1e-6, f"the federated average {got} should be {want}")
    return len(vectors), len(decoded), average


def main():
    test_mask_cancellation()
    test_pairwise_mask_symmetry()
    test_aggregate_matches_plaintext()
    test_hiding_property()
    test_fixed_point_round_trip()
    test_dropouts_recovery()
    test_boundaries_and_validation()
    clients, dimension, average = test_federated_averaging_use_case()

    print(f"federated round: clients={clients} dimension={dimension} "
          f"average={[round(value, 4) for value in average]}")
    print(f"field=2^61-1 scale=2^{SCALE_BITS} (fixed-point exactness in the checks)")
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print("PASS")


if __name__ == "__main__":
    main()
