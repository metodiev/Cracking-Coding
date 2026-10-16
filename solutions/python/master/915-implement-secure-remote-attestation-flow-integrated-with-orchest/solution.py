"""Task 915 - Implement secure remote attestation flow integrated with orchestration for trust chain.
(Tier: Master, CTCI Beyond - Security & Privacy)

What is implemented (in-process, deterministic; HMAC-SHA256 stands in for the asymmetric
signatures of a real deployment, and there is no wall clock - every nonce and counter is part of
the explicitly driven schedule):

  * `issue` / `verify_chain` - a three-level trust chain: vendor root -> intermediate CA -> the
    node's endorsement key.  Every certificate is signed by its issuer's key, so a revoked or
    forged link invalidates everything below it.
  * `Node.quote` - the attested node measures each boot layer (firmware, bootloader, kernel,
    application), mixes in a fresh verifier nonce and a monotonic counter, and signs the whole
    quote with its endorsement key: measurements || counter || nonce -> signature.
  * `verify_quote` - the relying party's checks: the chain, the revocation list, the signature,
    the expected measurement policy, the freshness nonce and counter monotonicity ("no replay of
    an old, healthy quote after a downgrade").
  * `Orchestrator` - the orchestration loop: nodes are admitted only when their quote verifies,
    re-attested every round, and evicted the moment a check fails; the admitted set is
    re-verified from scratch each round, so an evicted node never sneaks back.

Simplified versus production: symmetric HMAC keys instead of TPM EK/AK keys and certificates, an
in-memory registry, no network protocol and no time-based certificate expiry.

Verified below: a healthy node attests and is admitted, measurement tampering, nonce replay and
counter rollback are rejected, unregistered and revoked keys are rejected, a broken chain link
invalidates the node, the orchestrator evicts and never re-admits a failing node, and the empty /
single-node / empty-policy boundaries.
"""

import hashlib
import hmac

LAYERS = ("firmware", "bootloader", "kernel", "application")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def digest(*parts):
    material = b"\x1f".join(part if isinstance(part, bytes) else str(part).encode("utf-8")
                            for part in parts)
    return hashlib.sha256(material).digest()


def sign(key, *parts):
    return hmac.new(key, b"\x1f".join(part if isinstance(part, bytes) else str(part).encode("utf-8")
                                        for part in parts), hashlib.sha256).digest()


class Certificate:
    __slots__ = ("subject", "public_key", "issuer", "signature")

    def __init__(self, issuer, issuer_key, subject, public_key):
        self.subject = subject
        self.public_key = public_key
        self.issuer = issuer
        self.signature = sign(issuer_key, subject, public_key)

    def valid(self, issuer_key):
        return hmac.compare_digest(self.signature, sign(issuer_key, self.subject, self.public_key))


def issue(issuer, issuer_key, subject, public_key):
    return Certificate(issuer, issuer_key, subject, public_key)


def verify_chain(chain, root_key, revoked=()):
    """chain is ordered root-first; each link must be signed by the key above it."""
    if not chain:
        return False
    head = chain[0]
    if head.issuer != head.subject or not head.valid(root_key) or head.subject in revoked:
        return False
    for index in range(1, len(chain)):
        previous, link = chain[index - 1], chain[index]
        if link.issuer != previous.subject or not link.valid(previous.public_key):
            return False
        if link.subject in revoked or link.issuer in revoked:
            return False
    return True


class Node:
    __slots__ = ("name", "key", "chain", "measurements", "counter")

    def __init__(self, name, key, chain, measurements, counter=1):
        self.name = name
        self.key = key
        self.chain = chain
        self.measurements = dict(measurements)
        self.counter = counter

    def boot(self, application=None):
        """Advance the counter and (optionally) load a different application image."""
        self.counter += 1
        if application is not None:
            self.measurements["application"] = digest(application)
        return self

    def quote(self, nonce):
        """The attestation quote binds the nonce, the counter and every layer measurement."""
        body = digest(nonce, self.counter, *[self.measurements[layer] for layer in LAYERS])
        return {"node": self.name, "key": self.key, "chain": self.chain, "nonce": nonce,
                "counter": self.counter, "measurements": dict(self.measurements),
                "signature": sign(self.key, nonce, self.counter,
                                  *[self.measurements[layer] for layer in LAYERS]), "body": body}


def make_node(name, key, registry, vendor_key, ca_key, measurements):
    ca = issue("vendor-root", vendor_key, "intermediate-ca", ca_key)
    endorsement = issue("intermediate-ca", ca_key, name, key)
    chain = [issue("vendor-root", vendor_key, "vendor-root", vendor_key), ca, endorsement]
    return Node(name, key, chain, measurements)


def verify_quote(quote, nonce, policy, registry, revoked=(), last_counter=None, root_key=None):
    """Returns (ok, reason).  Every check is independent, so the reason is precise."""
    if quote["chain"] is None or not verify_chain(quote["chain"], root_key, revoked):
        return False, "chain"
    if quote["key"] not in registry or registry[quote["key"]] != quote["node"]:
        return False, "unregistered"
    if quote["key"] in revoked or quote["node"] in revoked:
        return False, "revoked"
    expected = sign(quote["key"], quote["nonce"], quote["counter"],
                    *[quote["measurements"][layer] for layer in LAYERS])
    if not hmac.compare_digest(expected, quote["signature"]):
        return False, "signature"
    if quote["nonce"] != nonce:
        return False, "freshness"
    if last_counter is not None and quote["counter"] <= last_counter:
        return False, "rollback"
    for layer in LAYERS:
        if layer in policy and policy[layer] is not None \
                and quote["measurements"][layer] != policy[layer]:
            return False, "measurement:" + layer
    return True, "ok"


class Orchestrator:
    """Admits only nodes whose quote verifies; re-attests every round from scratch."""

    def __init__(self, registry, policy, root_key, revoked=()):
        self.registry = registry
        self.policy = policy
        self.root_key = root_key
        self.revoked = tuple(revoked)
        self.counter = {}
        self.admitted = []
        self.evicted = []
        self.reasons = {}

    def round(self, nodes, nonce_base):
        admitted, evicted = [], []
        for index, node in enumerate(sorted(nodes, key=lambda item: item.name)):
            nonce = digest("nonce", nonce_base, node.name)
            quote = node.quote(nonce)
            ok, reason = verify_quote(quote, nonce, self.policy, self.registry, self.revoked,
                                      self.counter.get(node.name), self.root_key)
            self.reasons[reason] = self.reasons.get(reason, 0) + 1
            if ok:
                admitted.append(node.name)
                self.counter[node.name] = quote["counter"]
            else:
                evicted.append((node.name, reason))
        self.admitted.append(admitted)
        self.evicted.append(evicted)
        return admitted, evicted


def main():
    vendor_key = b"vendor-root-key"
    ca_key = b"intermediate-ca-key"
    good = {"firmware": digest("fw-v1"), "bootloader": digest("bl-v1"),
            "kernel": digest("kernel-v3"), "application": digest("app-v2")}
    policy = dict(good)

    registry, nodes = {}, []
    for index in range(4):
        key = digest("ek", index)
        node = make_node("node-%d" % index, key, registry, vendor_key, ca_key, good)
        registry[key] = node.name
        nodes.append(node)
    check(len(registry) == 4, "four endorsement keys must be registered")

    # ---- 1. a healthy node attests and is admitted
    nonce = digest("nonce", "round-0", nodes[0].name)
    quote = nodes[0].quote(nonce)
    ok, reason = verify_quote(quote, nonce, policy, registry, (), None, vendor_key)
    check(ok and reason == "ok", "a healthy quote must verify (%s)" % reason)
    check(verify_chain(nodes[0].chain, vendor_key), "the trust chain must verify")
    check(all(len(link.signature) == 32 for link in nodes[0].chain),
          "every certificate must carry a signature")
    check(not verify_chain(nodes[0].chain, b"wrong-root"),
          "a chain must not verify under a foreign root")

    # ---- 2. measurement tampering, replay and rollback are all rejected
    downgraded = Node(nodes[1].name, nodes[1].key, nodes[1].chain,
                      dict(good, application=digest("app-v1")))
    tampered = downgraded.quote(nonce)
    check(verify_quote(tampered, nonce, policy, registry, (), None, vendor_key)
          == (False, "measurement:application"),
          "a downgraded application image must be rejected by the policy")
    forged = dict(quote)
    forged["measurements"] = dict(good, kernel=digest("evil-kernel"))
    check(verify_quote(forged, nonce, policy, registry, (), None, vendor_key)[0] is False,
          "re-measuring without a signature must fail")
    forged["signature"] = sign(nodes[0].key, nonce, quote["counter"],
                               *[forged["measurements"][layer] for layer in LAYERS])
    check(verify_quote(forged, nonce, policy, registry, (), None, vendor_key)
          == (False, "measurement:kernel"),
          "a correctly signed but non-policy measurement must still be rejected")
    check(verify_quote(quote, digest("other-nonce"), policy, registry, (), None, vendor_key)
          == (False, "freshness"), "a stale quote must fail the freshness check")
    check(verify_quote(quote, nonce, policy, registry, (), quote["counter"], vendor_key)
          == (False, "rollback"), "a counter that does not advance must be rejected as a replay")
    check(verify_quote(quote, nonce, policy, registry, (), quote["counter"] - 1, vendor_key)[0],
          "an advancing counter must pass")

    # ---- 3. key registration, revocation and chain integrity
    stranger = Node("node-9", digest("ek-stranger"), nodes[0].chain, good)
    raw = stranger.quote(nonce)
    check(verify_quote(raw, nonce, policy, registry, (), None, vendor_key)
          == (False, "unregistered"), "an unregistered endorsement key must be rejected")
    check(verify_quote(quote, nonce, policy, registry, (quote["key"],), None, vendor_key)
          == (False, "revoked"), "a revoked key must be rejected")
    broken = list(nodes[2].chain)
    broken[1] = Certificate("vendor-root", vendor_key, "intermediate-ca", b"attacker-ca-key")
    broken_node = Node(nodes[2].name, nodes[2].key, broken, good)
    check(verify_quote(broken_node.quote(nonce), nonce, policy, registry, (), None, vendor_key)
          == (False, "chain"), "a node whose chain terminates in a replaced CA must be rejected")
    check(verify_chain(broken, vendor_key) is False, "the replaced chain must not verify")
    check(verify_chain(nodes[0].chain, vendor_key, revoked=("intermediate-ca",)) is False,
          "revoking the intermediate CA must invalidate every node below it")

    # ---- 4. the orchestration loop: evict and never re-admit
    orchestrator = Orchestrator(registry, policy, vendor_key)
    for round_index in range(3):
        admitted, evicted = orchestrator.round([node.boot() for node in nodes],
                                               "base-%d" % round_index)
        check(sorted(admitted) == [node.name for node in nodes] and evicted == [],
              "healthy nodes must be admitted every round: %r" % evicted)
    rogue = Node("node-2", nodes[2].key, nodes[2].chain, dict(good, firmware=digest("fw-rogue")),
                 counter=nodes[2].counter)
    swapped = [rogue if node.name == "node-2" else node for node in nodes]
    admitted, evicted = orchestrator.round([node.boot() for node in swapped], "base-3")
    check(evicted == [("node-2", "measurement:firmware")],
          "a compromised node must be evicted with a precise reason (%r)" % evicted)
    check("node-2" not in admitted and len(admitted) == 3, "the cluster must shrink to three")
    for round_index in range(4, 7):
        admitted, evicted = orchestrator.round([node.boot() for node in swapped],
                                               "base-%d" % round_index)
        check("node-2" not in admitted, "an evicted node must never sneak back in")
    check(orchestrator.reasons.get("measurement:firmware") == 4,
          "the rogue node must be refused once per round: %r" % orchestrator.reasons)
    healed = Node("node-2", nodes[2].key, nodes[2].chain, good, counter=rogue.counter)
    healed.boot()
    healed_nodes = [healed if node.name == "node-2" else node for node in nodes]
    admitted, evicted = orchestrator.round([node.boot() for node in healed_nodes], "base-7")
    check(evicted == [] and len(admitted) == 4,
          "a node that fixes its measurements and advances its counter must return")

    # ---- 5. boundaries
    check(Orchestrator(registry, policy, vendor_key).round([], "empty") == ([], []),
          "an empty node set is a no-op")
    solo = [nodes[3]]
    check(Orchestrator(registry, policy, vendor_key).round(solo, "solo")[0] == ["node-3"],
          "a single healthy node must be admitted")
    strict = Orchestrator(registry, {layer: digest("never") for layer in LAYERS}, vendor_key)
    check(strict.round([node.boot() for node in nodes], "strict")[0] == [],
          "an impossible measurement policy must admit nobody")
    unattested = Node("node-5", digest("ek-5"), nodes[0].chain, good)
    registry[digest("ek-5")] = "node-5"
    check(verify_quote(unattested.quote(nonce), nonce, {}, registry, (), None, vendor_key)[0],
          "an empty policy accepts any measured value")
    empty_chain = Node("node-6", digest("ek-6"), [], good)
    check(verify_quote(empty_chain.quote(nonce), nonce, policy, registry, (), None, vendor_key)
          == (False, "chain"), "a node without a trust chain must be rejected")
    del registry[digest("ek-5")]
    check(verify_chain([], vendor_key) is False, "an empty chain cannot verify")
    check(verify_chain([nodes[1].chain[1]], vendor_key) is False,
          "a chain that starts below the root cannot verify")

    # ---- 6. determinism
    replay = Node("node-0", nodes[0].key, nodes[0].chain, good)
    check(replay.quote(nonce)["signature"] == quote["signature"],
          "quotes must be reproducible from the node state")

    print("reasons seen: %r" % orchestrator.reasons)
    print("admitted per round: %r" % [len(row) for row in orchestrator.admitted])
    print("evicted per round: %r" % [row for row in orchestrator.evicted if row])
    print("PASS")


if __name__ == "__main__":
    main()
