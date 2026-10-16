"""Task 889 - Implement secure multiparty computation for private set intersection at scale.  (Tier: Master, CTCI Beyond - Security, Privacy & Cryptography)

The protocol is the classical commutative-encryption PSI over a prime-order subgroup:

    p is a 257 bit safe prime, q = (p - 1) / 2, and every element is hashed into the
    subgroup of quadratic residues, which has order q.  Each party i owns a secret
    exponent k_i and blinds a value by raising it to k_i in the group.  Exponentiation
    commutes, so after a payload has been blinded by every party exactly once its value is
    h(x)^(k_1 * ... * k_n) whichever order the parties used.

The ring protocol makes that happen without any party seeing another party's exponent: every
payload starts at its owner and is handed around the ring; each recipient applies its own
exponent and forwards it.  After n hops each payload has returned home, blinded by all n
parties, and every party publishes its final list.  The blinded intersection - the values that
appear in every published list - can only be mapped back to plaintext by the party that owns
those elements, so each party learns exactly the (n-way) intersection and nothing else.

Cheating is constrained by the verifications a deployment of this protocol really performs:
every received and published token is checked for subgroup membership with `t^q = 1` (this
rejects an attacker who injects a small-order element, the classic leak attempt), each party
commits to the length of its list before the ring starts (so a relayer that drops or duplicates
a token is caught), and the intersection every party computes must agree (a conflicting
publication is rejected).  The remaining honest-but-curious boundary - a party that silently
replaces tokens while preserving the count - is documented rather than papered over.

Simplified versus a deployed PSI library: no OT or garbled circuits, no zero-knowledge proofs
of correct exponentiation, honest-but-curious security, exponents come from a seeded PRNG
rather than a CSPRNG, a 257 bit group instead of 2048+, no network (messages travel through an
in-process transcript with an injectable fault hook), and no offline/online phase separation.
Verified: the group is a genuine safe prime with a generator of order q, exponentiation
commutes, the map is a bijection on the subgroup, the protocol's intersection equals the
plaintext set intersection for every randomised configuration (including empty, full, single
shared element and empty-party cases), no plaintext element ever appears in the transcript,
permuting the party order does not change the blinded output, injected small-order tokens and
dropped or duplicated payloads are detected, the modexp count matches the analytic cost, and
every run is reproducible.
"""
import hashlib

MASK64 = (1 << 64) - 1

P = 115792089237316195423570985008687907853269984665640564039457584007913129870127
Q = 57896044618658097711785492504343953926634992332820282019728792003956564935063
GENERATOR = 2        # 2^q = 1 mod p and 2 != 1, so its order is exactly q
TOKEN_BYTES = 32


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

    def exponent(self):
        """A secret exponent in [2, q]; the exponents are independent generators of the map."""
        return 2 + self.below(Q - 3)


class ProtocolViolation(Exception):
    """Raised when a message fails one of the protocol's structural checks."""


def is_probable_prime(number, rng, rounds=24):
    if number < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if number % small == 0:
            return number == small
    odd = number - 1
    shift = 0
    while odd % 2 == 0:
        odd //= 2
        shift += 1
    for _ in range(rounds):
        base = 2 + rng.below(number - 3)
        witness = pow(base, odd, number)
        if witness in (1, number - 1):
            continue
        for _ in range(shift - 1):
            witness = witness * witness % number
            if witness == number - 1:
                break
        else:
            return False
    return True


# ------------------------------------------------------------ the group layer
def hash_to_group(element, pepper=b"psi-889"):
    """Deterministic hash into the subgroup: square a digest so the order divides q."""
    digest = hashlib.sha256(pepper + element.encode()).digest()
    candidate = int.from_bytes(digest, "big") % (P - 3) + 2
    return pow(candidate, 2, P)


def in_subgroup(value):
    return isinstance(value, int) and 1 < value < P and pow(value, Q, P) == 1


def blind_many(tokens, exponent):
    return [pow(token, exponent, P) for token in tokens]


# ---------------------------------------------------------------- the network
class Network:
    """Explicit in-process message bus: every send is logged, faults are injectable."""

    def __init__(self, fault=None):
        self.messages = []
        self.modexps = 0
        self.fault = fault
        self.delivered_bytes = 0

    def send(self, sender, recipient, payload):
        delivered = list(payload)
        if self.fault is not None:
            delivered = self.fault(sender, recipient, delivered)
        self.messages.append((sender, recipient, tuple(delivered)))
        self.delivered_bytes += len(delivered) * TOKEN_BYTES
        return delivered

    def broadcast(self, sender, payload):
        self.messages.append((sender, "*", payload))
        return payload

    def transcript_bytes(self):
        return repr(self.messages).encode()


class Party:
    """One PSI participant: holds its element set and its secret exponent."""

    def __init__(self, name, elements, exponent, network):
        self.name = name
        self.elements = sorted(set(elements))
        self.exponent = exponent
        self.network = network
        self.tokens = [hash_to_group(element) for element in self.elements]
        network.modexps += len(self.tokens)
        self.declared_length = len(self.tokens)
        self.published = ()
        self.final = ()

    def commit(self):
        """Publish the length of the list this party will feed into the ring."""
        return self.network.broadcast(self.name, ("commit", self.name, self.declared_length))

    def blind(self, tokens):
        self.network.modexps += len(tokens)
        return blind_many(tokens, self.exponent)

    def accept(self, sender, tokens):
        for token in tokens:
            if not in_subgroup(token):
                raise ProtocolViolation(
                    "%s received a token outside the order-q subgroup from %s" % (self.name, sender))
        return tokens

    def publish(self, tokens):
        self.final = tuple(tokens)
        self.published = tuple(tokens)
        self.network.broadcast(self.name, ("final", self.name, self.published))
        return self.published

    def recognise(self, blinded_intersection):
        return [element for index, element in enumerate(self.elements)
                if self.final[index] in blinded_intersection]


def run_psi(parties, network):
    """The ring: every payload is blinded by every party, then all final lists are published."""
    if len(parties) < 2:
        raise ValueError("private set intersection needs at least two parties")
    names = [party.name for party in parties]
    if len(set(names)) != len(names):
        raise ValueError("party names must be unique")
    for party in parties:
        party.network = network
        party.commit()

    count = len(parties)
    flight = {index: list(parties[index].tokens) for index in range(count)}
    for _ in range(count):
        outgoing = dict(flight)
        for index in range(count):
            sender = parties[index]
            recipient = parties[(index + 1) % count]
            delivered = network.send(sender.name, recipient.name, outgoing[index])
            flight[(index + 1) % count] = recipient.blind(
                recipient.accept(sender.name, delivered))

    declared = {party.name: party.declared_length for party in parties}
    for index, party in enumerate(parties):
        if len(flight[index]) != declared[party.name]:
            raise ProtocolViolation(
                "%s ended the ring with %d tokens, not the %d it committed to"
                % (party.name, len(flight[index]), declared[party.name]))
        party.publish(flight[index])

    for party in parties:
        for other in parties:
            if len(other.published) != declared[other.name]:
                raise ProtocolViolation(
                    "%s published %d tokens but committed to %d"
                    % (other.name, len(other.published), declared[other.name]))
            for token in other.published:
                if not in_subgroup(token):
                    raise ProtocolViolation("%s published a token outside the subgroup" % other.name)

    published_sets = [set(party.published) for party in parties]
    intersection = set.intersection(*published_sets) if published_sets else set()
    answers = [party.recognise(intersection) for party in parties]
    for answer in answers[1:]:
        if answer != answers[0]:
            raise ProtocolViolation("the parties disagree about the intersection")
    return {
        "blinded_intersection": intersection,
        "answer": answers[0],
        "modexps": network.modexps,
        "messages": len(network.messages),
    }


def plaintext_intersection(parties):
    """Brute-force oracle: what every party's element set has in common."""
    result = None
    for party in parties:
        current = set(party.elements)
        result = current if result is None else result & current
    return sorted(result or [])


def make_parties(rng, specification, network):
    """`specification` is a list of (name, elements) pairs; exponents come from the PRNG."""
    return [Party(name, elements, rng.exponent(), network) for name, elements in specification]


# ----------------------------------------------------------------------- main
def main():
    rng = Rng(889001)

    # --- 1. the group is what it claims to be -------------------------------
    check(P == 2 * Q + 1, "p must be one more than twice q")
    check(is_probable_prime(Q, Rng(889002)), "q must be prime so the subgroup has prime order")
    check(is_probable_prime(P, Rng(889003)), "p must be prime")
    check(pow(GENERATOR, Q, P) == 1, "the generator must lie in the order-q subgroup")
    check(GENERATOR != 1 and pow(GENERATOR, 2, P) != 1,
          "the generator must not be trivial, so its order is exactly q")

    # --- 2. hashing lands in the subgroup -----------------------------------
    universe = ["item%04d" % index for index in range(300)]
    tokens = {element: hash_to_group(element) for element in universe}
    check(all(in_subgroup(token) for token in tokens.values()),
          "every hashed element must be a quadratic residue of order q")
    check(len(set(tokens.values())) == len(universe),
          "the hash must not collide on the test universe")
    check(hash_to_group("item0007") == tokens["item0007"], "the hash must be deterministic")

    # --- 3. exponentiation commutes and is a bijection ----------------------
    left = rng.exponent()
    right = rng.exponent()
    base = hash_to_group("item0001")
    check(pow(pow(base, left, P), right, P) == pow(pow(base, right, P), left, P),
          "blinding must commute, which is what makes the ring order independent")
    check(blind_many(blind_many([base], left), right) == blind_many(blind_many([base], right), left),
          "the list level blinding must commute too")
    image = [pow(pow(GENERATOR, step, P), left, P) for step in range(1, 20)]
    check(len(set(image)) == len(image), "the exponent map must be injective on the subgroup")
    check(all(in_subgroup(value) for value in image),
          "blinding must keep every value inside the subgroup")

    # --- 4. correctness against the plaintext oracle ------------------------
    configuration_answers = []
    for trial in range(30):
        shared = universe[:3 + rng.below(4)]
        specification = []
        for party_index in range(2 + rng.below(3)):
            private = [universe[rng.below(len(universe))] for _ in range(4 + rng.below(5))]
            specification.append(("P%d" % party_index, list(shared) + private))
        network = Network()
        parties = make_parties(rng, specification, network)
        result = run_psi(parties, network)
        expected = plaintext_intersection(parties)
        check(result["answer"] == expected,
              "the PSI answer must equal the plaintext intersection: %r vs %r"
              % (result["answer"], expected))
        configuration_answers.append(len(expected))

    # --- 5. edge cases ------------------------------------------------------
    network = Network()
    disjoint = make_parties(rng, [("A", ["x1", "x2"]), ("B", ["y1", "y2"])], network)
    result = run_psi(disjoint, network)
    check(result["answer"] == [] and result["blinded_intersection"] == set(),
          "disjoint sets must yield an empty intersection")

    network = Network()
    identical = make_parties(rng, [("A", ["a", "b", "c"]), ("B", ["c", "b", "a"]),
                                   ("C", ["a", "b", "c"])], network)
    check(run_psi(identical, network)["answer"] == ["a", "b", "c"],
          "identical sets must intersect completely")

    network = Network()
    single = make_parties(rng, [("A", ["one", "two", "three"]), ("B", ["zero", "two", "four"]),
                                ("C", ["two", "five"])], network)
    check(run_psi(single, network)["answer"] == ["two"],
          "a single shared element must be found")

    network = Network()
    with_empty = make_parties(rng, [("A", ["a", "b"]), ("B", []), ("C", ["a", "b"])], network)
    check(run_psi(with_empty, network)["answer"] == [],
          "an empty party must force an empty intersection")

    network = Network()
    duplicated = make_parties(rng, [("A", ["d", "d", "e"]), ("B", ["d", "e", "e"])], network)
    check(run_psi(duplicated, network)["answer"] == ["d", "e"],
          "duplicate elements must be deduplicated before the protocol")

    network = Network()
    two_party = make_parties(rng, [("A", ["p", "q"]), ("B", ["q", "r"])], network)
    two = run_psi(two_party, network)
    exponents = [party.exponent for party in two_party]
    direct = pow(pow(hash_to_group("q"), exponents[0], P), exponents[1], P)
    check(direct in two["blinded_intersection"],
          "the ring's blinded value must equal the direct two-party composition")
    check(two["answer"] == ["q"], "the two-party intersection must be correct")

    # --- 6. nothing in the transcript reveals a plaintext element -----------
    network = Network()
    secret_terms = ["secret%04d" % index for index in range(40)]
    leaky = make_parties(rng, [("A", secret_terms[:25]), ("B", secret_terms[15:]), ("C",
                          secret_terms[10:30])], network)
    leak = run_psi(leaky, network)
    blob = network.transcript_bytes()
    for term in secret_terms:
        check(term.encode() not in blob,
              "the transcript must never contain the plaintext element %r" % term)
    for party in leaky:
        check(all(in_subgroup(token) for token in party.published),
              "only group elements may be published")
    raw_hashes = {hash_to_group(term) for term in secret_terms}
    for party in leaky:
        check(not (set(party.published) & raw_hashes),
              "published tokens must be blinded, never the bare hash of an element")
    check(leak["answer"] == plaintext_intersection(leaky),
          "the leakage test must still compute the right answer")
    check(len(leak["answer"]) == 10, "the crafted sets must share exactly ten elements")

    # --- 7. the ring is order independent -----------------------------------
    specification = [("P0", universe[:20]), ("P1", universe[10:35]), ("P2", universe[30:50])]
    forward = Network()
    forward_parties = make_parties(Rng(889500), specification, forward)
    forward_result = run_psi(forward_parties, forward)
    backward = Network()
    backward_parties = make_parties(Rng(889500), list(reversed(specification)), backward)
    backward_result = run_psi(list(reversed(backward_parties)), backward)
    check(forward_result["blinded_intersection"] == backward_result["blinded_intersection"],
          "reversing the ring order must not change the blinded intersection")
    check(forward_result["answer"] == backward_result["answer"],
          "the answer must be order independent as well")

    # --- 8. cost is exactly predictable -------------------------------------
    costed_network = Network()
    specification = [("P%d" % index, universe[index * 10:(index + 1) * 10 + 5])
                     for index in range(4)]
    costed_parties = make_parties(Rng(889600), specification, costed_network)
    costed_total = sum(len(party.elements) for party in costed_parties)
    costed_result = run_psi(costed_parties, costed_network)
    expected_modexps = costed_total * (len(costed_parties) + 1)
    check(costed_result["modexps"] == expected_modexps,
          "the modexp count must be |elements| * (n + 1): %d vs %d"
          % (costed_result["modexps"], expected_modexps))
    check(costed_result["messages"] == len(costed_parties) ** 2 + 2 * len(costed_parties),
          "the transcript must hold n^2 ring messages plus n commits and n publications")
    check(costed_network.delivered_bytes == len(costed_parties) * costed_total * TOKEN_BYTES,
          "bandwidth must be n ring hops times the total element count: %d vs %d"
          % (costed_network.delivered_bytes, len(costed_parties) * costed_total * TOKEN_BYTES))
    check(costed_result["answer"] == plaintext_intersection(costed_parties),
          "the costed run must be correct")

    # --- 9. fault injection: small-order tokens are rejected ----------------
    def inject_small_order(sender, recipient, payload):
        if payload and sender == "P0":
            return [P - 1] + payload[1:]        # p - 1 has order 2, a classic leak attempt
        return payload

    network = Network(fault=inject_small_order)
    parties = make_parties(Rng(889700), [("P0", ["a", "b"]), ("P1", ["b", "c"])], network)
    raised = False
    try:
        run_psi(parties, network)
    except ProtocolViolation:
        raised = True
    check(raised, "an injected element of order 2 must be rejected by the subgroup check")

    # --- 10. fault injection: dropped or duplicated payloads ----------------
    def drop_a_token(sender, recipient, payload):
        if payload and sender == "P1":
            return payload[:-1]
        return payload

    network = Network(fault=drop_a_token)
    parties = make_parties(Rng(889800), [("P0", ["a", "b", "c"]), ("P1", ["b", "c", "d"])],
                           network)
    raised = False
    try:
        run_psi(parties, network)
    except ProtocolViolation:
        raised = True
    check(raised, "a relayer that drops a token must be caught by the length commitment")

    def duplicate_a_token(sender, recipient, payload):
        if payload and sender == "P0":
            return payload + payload[:1]
        return payload

    network = Network(fault=duplicate_a_token)
    parties = make_parties(Rng(889801), [("P0", ["a", "b"]), ("P1", ["a", "b"])], network)
    raised = False
    try:
        run_psi(parties, network)
    except ProtocolViolation:
        raised = True
    check(raised, "a relayer that duplicates a token must be caught as well")

    # --- 11. malformed inputs ----------------------------------------------
    raised = False
    try:
        run_psi([Party("solo", ["a"], rng.exponent(), Network())], Network())
    except ValueError:
        raised = True
    check(raised, "a single party cannot run a set intersection")
    raised = False
    try:
        run_psi([Party("A", ["a"], rng.exponent(), Network()),
                 Party("A", ["b"], rng.exponent(), Network())], Network())
    except ValueError:
        raised = True
    check(raised, "duplicate party names must be rejected")
    check(not in_subgroup(P - 1) and not in_subgroup(1) and not in_subgroup(0),
          "trivial and order-two values must fail the subgroup test")

    # --- 12. scale ----------------------------------------------------------
    scale_network = Network()
    scale_rng = Rng(889900)
    scale_universe = ["node%05d" % index for index in range(4000)]
    scale_shared = scale_universe[:120]
    specification = []
    for party_index in range(5):
        private = [scale_universe[rng.below(len(scale_universe))] for _ in range(80)]
        specification.append(("S%d" % party_index, scale_shared + private))
    scale_parties = make_parties(scale_rng, specification, scale_network)
    scale_total = sum(len(party.elements) for party in scale_parties)
    scale_result = run_psi(scale_parties, scale_network)
    check(scale_result["answer"] == plaintext_intersection(scale_parties),
          "the scaled protocol must match the plaintext oracle")
    check(len(scale_result["answer"]) >= 120, "the scaled intersection must be substantial")
    check(scale_result["modexps"] == scale_total * (len(scale_parties) + 1),
          "the scaled modexp count must follow the same formula")
    check(scale_network.transcript_bytes().find(b"node0000") == -1,
          "the scaled transcript must not expose any plaintext element")

    # --- 13. reproducibility ------------------------------------------------
    def replay(seed):
        local = Rng(seed)
        network = Network()
        parties = make_parties(local, [("A", universe[:12]), ("B", universe[6:20]),
                                       ("C", universe[15:24])], network)
        result = run_psi(parties, network)
        return result["answer"], sorted(result["blinded_intersection"]), result["modexps"]

    check(replay(889950) == replay(889950), "an identical seed must replay the protocol")

    print("corpus: %d candidate elements, %d random configurations matched the oracle"
          % (len(universe), len(configuration_answers)))
    print("cost: %d modexps for %d elements over %d parties (%d in, 1 per hop)"
          % (costed_result["modexps"], costed_total, len(costed_parties), costed_total))
    print("transcript: %d messages, %d bytes of group elements, no plaintext"
          % (len(costed_network.messages), len(costed_network.transcript_bytes())))
    print("scale: %d parties, %d elements, %d shared, %d modexps"
          % (len(scale_parties), scale_total, len(scale_result["answer"]),
             scale_result["modexps"]))
    print("PASS")


if __name__ == "__main__":
    main()
