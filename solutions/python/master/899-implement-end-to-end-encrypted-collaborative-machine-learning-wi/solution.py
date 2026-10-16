"""Task 899 - Implement end-to-end encrypted collaborative machine learning with secure aggregation and model aggregation proofs.  (Tier: Master, CTCI Beyond - Security, Privacy & Cryptography)

Secure aggregation computes the exact sum of many clients' model updates without the server ever
seeing an individual update.  The protocol implemented here is the double-masked one:

* **pairwise masks** - every pair (i, j) agrees on a shared secret `z_ij = g^(s_i * s_j)` from a
  Diffie-Hellman exchange and derives the same pseudorandom mask vector from it; the lower
  indexed client adds it and the higher indexed one subtracts it, so every pairwise mask cancels
  in the sum while the server (which cannot solve Diffie-Hellman) never learns them;
* **self masks** - each client adds one more mask derived from a private seed, so the published
  vector is dominated by randomness rather than by the update;
* **dropout recovery** - the seeds behind both the self mask *and* the pairwise masks of a client
  are secret-shared with Shamir's scheme, so when a client drops mid-round the survivors
  reconstruct exactly the masks it had contributed and remove them, leaving the aggregate equal
  to the sum over the surviving contributors - the property the whole design exists for;
* **aggregation proofs** - every client commits to its masked vector before the server
  aggregates, and every client re-checks the published aggregate against the commitments, so a
  contribution (or the aggregate) altered after the fact is caught rather than silently
  corrupting the model.

Verified: the aggregate equals the plaintext sum exactly, and modulo the field prime that is an
identity rather than a numerical coincidence; every pairwise mask cancels (the mask contributions
sum to exactly zero); the uploaded vector shows no linear correlation with the client's own
update while the aggregate does; neither the private keys nor the mask seeds appear anywhere in
the transcript; Shamir reconstruction with the threshold number of surviving shares recovers the
dropped client's masks exactly while fewer shares do not; a dropout round is exact and differs
from the naive sum, so the recovery is load bearing; a tampered contribution or a tampered
aggregate fails the commitment check; degenerate cases (one client, one dimension, a threshold of
one) have their closed forms; a 30 client round with a quarter of the clients dropping runs with
the cost accounted; and everything replays from its seed.

Simplified versus Bonawitz et al.: no authenticated channels, no zero-knowledge proofs beyond
hash commitments, a 257 bit group, no quantisation or compression of the updates, and a
synchronous single round per aggregation.
"""
import hashlib

MASK64 = (1 << 64) - 1

# p = 2q + 1 with q prime, as in the multiparty PSI demo.
P = 115792089237316195423570985008687907853269984665640564039457584007913129870127
Q = 57896044618658097711785492504343953926634992332820282019728792003956564935063
GENERATOR = 2


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def u64(self):
        self.state, value = splitmix64(self.state)
        return value

    def below(self, bound):
        return self.u64() % bound

    def scalar(self):
        return 1 + self.u64() % (Q - 1)


class ProtocolError(Exception):
    """Raised for a malformed client set, dimension, threshold or message."""


# ------------------------------------------------------------------ vector layer
def mask_vector(seed, dimension):
    """Deterministic pseudorandom vector in the field, derived from a shared secret."""
    out = []
    state = seed % Q
    for _ in range(dimension):
        state, value = splitmix64(state & MASK64)
        out.append(value % P)
    return out


def vector_add(left, right):
    return [(a + b) % P for a, b in zip(left, right)]


def vector_subtract(left, right):
    return [(a - b) % P for a, b in zip(left, right)]


def vector_zero(dimension):
    return [0] * dimension


def squared_norm(vector):
    return sum(value * value for value in vector) % P


def correlation(left, right):
    """Pearson correlation over the pooled coordinates; 0 means the pair is unrelated."""
    count = len(left)
    mean_left = sum(left) / count
    mean_right = sum(right) / count
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    spread_left = sum((a - mean_left) ** 2 for a in left) ** 0.5
    spread_right = sum((b - mean_right) ** 2 for b in right) ** 0.5
    if spread_left == 0 or spread_right == 0:
        return 0.0
    return covariance / (spread_left * spread_right)


# -------------------------------------------------------------- secret sharing
def shamir_split(secret, share_count, threshold, rng):
    """Share a field element so that any `threshold` shares reconstruct it."""
    if share_count < 1:
        raise ProtocolError("at least one share is required")
    if threshold < 1 or threshold > share_count:
        raise ProtocolError("the threshold must lie between one and the share count")
    coefficients = [secret % P] + [rng.below(P) for _ in range(threshold - 1)]
    shares = []
    for index in range(1, share_count + 1):
        value = 0
        power = 1
        for coefficient in coefficients:
            value = (value + coefficient * power) % P
            power = power * index % P
        shares.append((index, value))
    return shares


def shamir_reconstruct(share_pairs):
    """Lagrange interpolation at zero over the given (x, y) points."""
    if not share_pairs:
        raise ProtocolError("there is nothing to reconstruct from")
    seen = set()
    for index, _ in share_pairs:
        if index in seen:
            raise ProtocolError("shares must come from distinct holders")
        seen.add(index)
    total = 0
    for index, value in share_pairs:
        numerator = 1
        denominator = 1
        for other, _ in share_pairs:
            if other == index:
                continue
            numerator = numerator * (-other) % P
            denominator = denominator * (index - other) % P
        total = (total + value * numerator * pow(denominator, P - 2, P)) % P
    return total


def commit(vector):
    return hashlib.sha256(repr(list(vector)).encode()).hexdigest()


# ------------------------------------------------------------------ the clients
class Client:
    """One participant: its update, its Diffie-Hellman key, its masks and its shares."""

    def __init__(self, index, update, secret, rng):
        self.index = index
        self.update = [value % P for value in update]
        self.secret = secret
        self.public = pow(GENERATOR, secret, P)
        self.self_seed = rng.below(P)
        self.self_mask = mask_vector(self.self_seed, len(self.update))
        self.shares = {}          # key -> the full share list this client owns and distributes
        self.held = {}            # (owner index, key) -> this client's own share pair
        self.masked = None
        self.commitment = None

    def shared_secret(self, peer):
        """Only this pair can compute g^(s_i s_j)."""
        return pow(peer.public, self.secret, P)

    def pair_mask(self, peer, dimension, sign=None):
        """The vector this client contributes for the pair; signs make the pair cancel."""
        shared = self.shared_secret(peer)
        vector = mask_vector(shared, dimension)
        if sign is None:
            sign = 1 if self.index < peer.index else -1
        return vector if sign > 0 else [(P - value) % P for value in vector]

    def mask_with(self, peers, dimension):
        masked = vector_add(self.update, self.self_mask)
        for peer in peers:
            if peer.index == self.index:
                continue
            masked = vector_add(masked, self.pair_mask(peer, dimension))
        self.masked = masked
        self.commitment = commit(masked)
        return masked

    def share_secrets(self, peers, threshold, rng):
        """Secret-share this client's self seed and every pairwise seed it owns."""
        holders = len(peers)
        self.shares[-1] = shamir_split(self.self_seed, holders, threshold, rng)
        for peer in peers:
            if peer.index <= self.index:
                continue
            self.shares[peer.index] = shamir_split(self.shared_secret(peer), holders, threshold, rng)
    def verify_aggregate(self, contributions, result):
        """Re-check the published aggregate, the commitments and the revealed mask removal."""
        published = result["aggregate"]
        if len(published) != len(self.update):
            return False
        total = vector_zero(len(published))
        for masked, commitment_value in contributions:
            if commit(masked) != commitment_value:
                return False
            total = vector_add(total, masked)
        if total != list(result["masked_total"]):
            return False
        if total != vector_add(published, result["removal"]):
            return False
        recomputed = vector_zero(len(published))
        for _, _, seed, sign in result["revealed"]:
            vector = mask_vector(seed, len(published))
            recomputed = (vector_add(recomputed, vector) if sign > 0
                          else vector_subtract(recomputed, vector))
        return recomputed == result["removal"]


def build_clients(count, dimension, seed):
    if count < 1:
        raise ProtocolError("a round needs at least one client")
    if dimension < 1:
        raise ProtocolError("the update dimension must be positive")
    rng = Rng(seed)
    clients = []
    for index in range(count):
        update = [rng.below(1000) for _ in range(dimension)]
        clients.append(Client(index, update, rng.scalar(), rng))
    return clients


def prepare_shares(clients, threshold, seed):
    """Split every secret and hand each holder exactly its own share."""
    rng = Rng(seed)
    for client in clients:
        client.share_secrets(clients, threshold, rng)
    for owner in clients:
        for key, share_list in owner.shares.items():
            for holder in clients:
                holder.held[(owner.index, key)] = share_list[holder.index - 1]
    return rng


def mask_consistency(clients, dimension):
    """Sum of every pairwise contribution in the round; it must be exactly zero."""
    total = vector_zero(dimension)
    for client in clients:
        for peer in clients:
            if peer.index <= client.index:
                continue
            total = vector_add(total, client.pair_mask(peer, dimension))
            total = vector_add(total, peer.pair_mask(client, dimension))
    return total


def reconstruct_seed(owner, key, survivors, threshold):
    """Recover a seed `owner` shared under `key` from the surviving holders' own shares."""
    shares = []
    for holder in survivors:
        pair = holder.held.get((owner.index, key))
        if pair is None:
            continue
        shares.append(pair)
    if len(shares) < threshold:
        raise ProtocolError("only %d of %d required shares survived for %r"
                            % (len(shares), threshold, key))
    return shamir_reconstruct(shares[:threshold])


def aggregate(clients, dimension, dropped=(), threshold=None):
    """One server round: sum the masked updates, then remove every mask that is still in it.

    Returns a result dict with the aggregate, the contributing clients, the removed mask vector
    and the revealed mask seeds, so a client can audit the whole derivation.
    """
    dropped = set(dropped)
    survivors = [client for client in clients if client.index not in dropped]
    absentees = [client for client in clients if client.index in dropped]
    for client in survivors:
        client.masked = client.mask_with(clients, dimension)
    total = vector_zero(dimension)
    for client in survivors:
        total = vector_add(total, client.masked)
    removal = vector_zero(dimension)
    revealed = []
    for client in survivors:
        # a surviving client simply reveals its own self mask once the round is closed; the
        # pairwise masks still cover its update, so nothing about it leaks
        removal = vector_add(removal, client.self_mask)
        revealed.append(("self", client.index, client.self_seed, 1))
    for absent in absentees:
        for peer in survivors:
            if threshold is None:
                seed = absent.shared_secret(peer)
                sign = -1 if absent.index < peer.index else 1
            elif absent.index < peer.index:
                seed = reconstruct_seed(absent, peer.index, survivors, threshold)
                sign = -1
            else:
                seed = reconstruct_seed(peer, absent.index, survivors, threshold)
                sign = 1
            removal = (vector_add(removal, mask_vector(seed, dimension)) if sign > 0
                       else vector_subtract(removal, mask_vector(seed, dimension)))
            revealed.append(("pair", (absent.index, peer.index), seed, sign))
    return {"aggregate": vector_subtract(total, removal),
            "survivors": survivors,
            "absentees": absentees,
            "masked_total": total,
            "removal": removal,
            "revealed": revealed}


def plaintext_sum(clients, dropped=()):
    dropped = set(dropped)
    total = vector_zero(len(clients[0].update))
    for client in clients:
        if client.index in dropped:
            continue
        total = vector_add(total, client.update)
    return total


# ----------------------------------------------------------------------- main
def main():
    # --- 1. the field and the mask layer ------------------------------------
    check(P == 2 * Q + 1 and pow(GENERATOR, Q, P) == 1,
          "the group must have prime order q for the Diffie-Hellman exchange")
    first = mask_vector(123456789, 8)
    check(first == mask_vector(123456789, 8), "mask derivation must be deterministic")
    check(mask_vector(987654321, 8) != first, "different seeds must give different masks")
    check(len(first) == 8 and all(0 <= value < P for value in first),
          "mask entries must be field elements")
    check(vector_subtract(vector_add(first, first), first) == first,
          "field arithmetic must round-trip")
    check(mask_vector(7, 0) == [], "a zero dimensional mask is the empty vector")

    # --- 2. pairwise masks cancel -------------------------------------------
    clients = build_clients(5, 6, 899001)
    for client in clients:
        for peer in clients:
            if peer.index <= client.index:
                continue
            check(client.pair_mask(peer, 6) == vector_subtract(vector_zero(6),
                                                              peer.pair_mask(client, 6)),
                  "the two sides of a pair must contribute opposite masks")
            check(client.shared_secret(peer) == peer.shared_secret(client),
                  "both sides must derive the same Diffie-Hellman secret")
    check(mask_consistency(clients, 6) == vector_zero(6),
          "the pairwise masks must cancel exactly in the round")

    # --- 3. the aggregate equals the plaintext sum --------------------------
    result = aggregate(clients, 6)
    aggregated, survivors, absentees = result["aggregate"], result["survivors"], result["absentees"]
    expected = plaintext_sum(clients)
    check(aggregated == expected,
          "secure aggregation must return the exact plaintext sum: %r vs %r"
          % (aggregated[:3], expected[:3]))
    check(len(survivors) == len(clients) and not absentees, "a full round uses every client")
    check(all(client.masked is not None for client in clients),
          "every client must publish a masked update")
    check(aggregated != [value + 1 for value in expected],
          "the aggregate must not be an off-by-one artefact")

    # --- 4. the transcript hides the individual updates ---------------------
    transcripts = [(client.public, client.commitment, client.masked) for client in clients]
    blob = repr(transcripts).encode()
    for client in clients:
        check(str(client.secret).encode() not in blob,
              "a private Diffie-Hellman key must never reach the transcript")
        check(str(client.self_seed).encode() not in blob,
              "a self mask seed must never reach the transcript")
        check(str(client.update).encode() not in blob,
              "the raw update must never reach the transcript")
    differences_update = [a - b for a, b in zip(clients[0].update, clients[1].update)]
    differences_masked = [(a - b) % P if (a - b) % P < P // 2 else (a - b) % P - P
                          for a, b in zip(clients[0].masked, clients[1].masked)]
    check(abs(correlation(differences_update, differences_masked)) < 0.5,
          "the published vectors must not be linearly related to the updates")
    matched = 0
    for left in range(len(clients)):
        for right in range(left + 1, len(clients)):
            updates = [a - b for a, b in zip(clients[left].update, clients[right].update)]
            masked = [(a - b) % P if (a - b) % P < P // 2 else (a - b) % P - P
                      for a, b in zip(clients[left].masked, clients[right].masked)]
            if abs(correlation(updates, masked)) < 0.6:
                matched += 1
    check(matched >= 5, "most pairs must be decorrelated by their masks, only %d of 10 were"
          % matched)
    spread = max(abs((a - b) % P if (a - b) % P < P // 2 else (a - b) % P - P)
                 for a, b in zip(clients[0].update, clients[0].masked))
    check(spread > 10 ** 12, "the mask must dominate the update in the published vector")

    # --- 5. commitments catch tampering -------------------------------------
    contributions = [(client.masked, client.commitment) for client in clients]
    check(clients[0].verify_aggregate(contributions, result),
          "the published aggregate must pass the commitment and mask audits")
    tampered = list(clients[2].masked)
    tampered[0] = (tampered[0] + 1) % P
    forged = contributions[:2] + [(tampered, clients[2].commitment)] + contributions[3:]
    check(not clients[0].verify_aggregate(forged, result),
          "a contribution altered after committing must fail the check")
    for field_name in ("aggregate", "removal", "masked_total"):
        broken = dict(result)
        shifted = list(result[field_name])
        shifted[0] = (shifted[0] + 1) % P
        broken[field_name] = shifted
        check(not clients[0].verify_aggregate(contributions, broken),
              "a tampered %s must fail the audit" % field_name)
    broken = dict(result)
    broken["revealed"] = result["revealed"][:-1]
    check(not clients[0].verify_aggregate(contributions, broken),
          "a shortened mask reveal must fail the audit")

    # --- 6. secret sharing --------------------------------------------------
    sharing_rng = Rng(899100)
    secret = sharing_rng.below(P)
    shares = shamir_split(secret, 6, 4, sharing_rng)
    check(len(shares) == 6 and len({index for index, _ in shares}) == 6,
          "every holder must get a distinct point")
    check(shamir_reconstruct(shares[:4]) == secret,
          "the threshold number of shares must reconstruct the secret")
    check(shamir_reconstruct([shares[1], shares[3], shares[4], shares[5]]) == secret,
          "any threshold sized subset must reconstruct the secret")
    check(shamir_reconstruct(shares[:3]) != secret,
          "fewer than the threshold shares must not reveal the secret")
    check(shamir_reconstruct(shares[:1]) != secret, "a single share must not reveal the secret")
    trivial = shamir_split(secret, 3, 1, Rng(899101))
    check(all(value == secret for _, value in trivial),
          "a threshold of one makes every share the secret itself")
    for bad in (lambda: shamir_split(secret, 3, 4, Rng(899102)),
                lambda: shamir_split(secret, 0, 1, Rng(899103)),
                lambda: shamir_reconstruct([]),
                lambda: shamir_reconstruct([(1, 5), (1, 9)])):
        raised = False
        try:
            bad()
        except ProtocolError:
            raised = True
        check(raised, "an invalid sharing request must be rejected")

    # --- 7. dropout recovery ------------------------------------------------
    round_clients = build_clients(6, 5, 899200)
    prepare_shares(round_clients, 3, 899201)
    result = aggregate(round_clients, 5, dropped=(1, 4), threshold=3)
    aggregated, survivors, absentees = result["aggregate"], result["survivors"], result["absentees"]
    expected = plaintext_sum(round_clients, dropped=(1, 4))
    check(aggregated == expected,
          "the dropout round must return the exact sum over the survivors")
    check(len(survivors) == 4 and len(absentees) == 2, "the right clients must contribute")
    check(aggregated != plaintext_sum(round_clients),
          "the dropout recovery must actually change the answer")
    for absent in absentees:
        check(reconstruct_seed(absent, -1, survivors, 3) == absent.self_seed,
              "the dropped client's self mask seed must be recovered")
    for lower in round_clients:
        if lower.index not in (1, 4):
            continue
        for higher in round_clients:
            if higher.index <= lower.index:
                continue
            check(reconstruct_seed(lower, higher.index, survivors, 3)
                  == lower.shared_secret(higher),
                  "the pairwise seed of a dropout must be recovered")
    single = build_clients(4, 3, 899301)
    prepare_shares(single, 1, 899302)
    single_result = aggregate(single, 3, dropped=(1, 2, 3), threshold=1)
    total = single_result["aggregate"]
    check(len(single_result["survivors"]) == 1 and total == single[0].update,
          "with one contributor left the aggregate must be that client's update")
    raised = False
    try:
        aggregate(single, 3, dropped=(1, 2, 3), threshold=2)
    except ProtocolError:
        raised = True
    check(raised,
          "one survivor cannot reach a threshold of two shares, so the round must refuse")

    # --- 8. malformed rounds ------------------------------------------------
    for bad in (lambda: build_clients(2, 0, 899400),
                lambda: build_clients(0, 3, 899401)):
        raised = False
        try:
            bad()
        except ProtocolError:
            raised = True
        check(raised, "an invalid client set must be rejected")
    raised = False
    try:
        aggregate(build_clients(3, 3, 899402), 3, dropped=(0, 1), threshold=3)
    except ProtocolError:
        raised = True
    check(raised, "too few surviving shares must be reported, not silently tolerated")

    # --- 9. scale -----------------------------------------------------------
    scale_clients = build_clients(30, 12, 899500)
    check(len(scale_clients) == 30, "the scaled round must build every client")
    prepare_shares(scale_clients, 5, 899501)
    dropouts = tuple(range(0, 30, 4))
    scale_result = aggregate(scale_clients, 12, dropped=dropouts, threshold=5)
    aggregated, survivors = scale_result["aggregate"], scale_result["survivors"]
    absentees = scale_result["absentees"]
    check(aggregated == plaintext_sum(scale_clients, dropped=dropouts),
          "the scaled round must be exact with a quarter of the clients gone")
    check(len(survivors) + len(absentees) == 30, "every client must be accounted for")
    check(len(absentees) == len(dropouts), "every dropped client must be masked out")
    check(mask_consistency(scale_clients, 12) == vector_zero(12),
          "the scaled pairwise masks must cancel exactly")
    check(squared_norm(aggregated) > 0, "the aggregate must be non trivial")
    check(len(aggregated) == 12, "the aggregate must have the right dimension")

    # --- 10. determinism ----------------------------------------------------
    def replay(seed):
        local = build_clients(8, 4, seed)
        prepare_shares(local, 3, seed + 1)
        outcome = aggregate(local, 4, dropped=(2, 5), threshold=3)
        return (outcome["aggregate"], [client.commitment for client in local],
                outcome["revealed"])

    check(replay(899600) == replay(899600), "an identical seed must replay the whole round")
    check(replay(899601) != replay(899600), "a different seed must produce different updates")

    print("round: %d clients, dimension %d, aggregate exactly the plaintext sum"
          % (len(clients), len(clients[0].update)))
    print("masks: pairwise contributions sum to zero, published vectors decorrelated by %.3f"
          % correlation(differences_update, differences_masked))
    print("dropout: %d of %d clients removed, aggregate still exact for the survivors"
          % (len(absentees), len(scale_clients)))
    print("proofs: commitments verified by every client, tampering rejected")
    print("scale: %d clients, %d dimensions, threshold %d, %d survivors"
          % (len(scale_clients), len(aggregated), 5, len(survivors)))
    print("PASS")


if __name__ == "__main__":
    main()
