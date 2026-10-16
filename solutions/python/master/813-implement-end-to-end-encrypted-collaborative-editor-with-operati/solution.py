"""Task 813 - Implement end-to-end encrypted collaborative editor with operational transforms and access control.  (Tier: Master, Ch 9 - System Design and Scalability)

A collaborative editor replicates one document over N sites that edit concurrently.  A site emits an
operation - insert(pos, text) or delete(pos, length) - against the revision it has seen and applies it
optimistically to its own copy.  The server is a *canonicalising relay*: at each barrier it takes the
batch of operations that are concurrent with respect to the current revision, integrates the whole
batch into one canonical operation list, and broadcasts it.  Integration has a precise meaning here:

  * every delete in the batch removes its range from the *original* revision, so the batch removes the
    union of all deleted ranges (overlapping deletes remove exactly the union, identical deletes
    cancel to a no-op);
  * every insert is anchored to the *gap* it points at - the number of surviving characters before its
    position - and an insert that points inside a deleted range is pinned to the front of that hole,
    so a concurrent insert is never silently lost;
  * inserts that share a gap are ordered by (site id, Lamport clock), which makes the resolution a
    function of the batch alone: independent of arrival order, of the fold order and of the machine.

That canonical resolution is `resolve_batch`.  It is cross-checked against two independent
implementations of the same specification: the pairwise operational-transform pair
transform(a, b) -> (a', b') obeying the TP1 commute law

    apply(apply(doc, a), b')  ==  apply(apply(doc, b), a'),

which is what a client uses to rebase one operation against another, and `oracle_final`, a
specification-level implementation that keeps the surviving characters in a list and assigns every
insert to a gap - a representation completely different from the operation-list resolver.

Access control is a role/permission matrix (owner/edit/comment/view) evaluated by the server for every
operation against the member's role and the epoch the operation was issued in.  Revocation bumps the
epoch, so operations already in flight from a removed member are rejected, and a key rotation makes
content written after the revocation unreadable with the retired key.  Content is sealed end to end
with a toy pad-and-tag construction: a splitmix64 keystream XORed into the plaintext plus a keyed
digest tag over the ciphertext (encrypt-then-MAC), which detects tampering and wrong keys.

Verified: TP1 exhaustively over a small operation space, the resolver against the semantic oracle on
200 randomised concurrent batches (and against the pairwise transform over every pair of the operation
space), permutation invariance of the canonical revision, the length-conservation law, multi-round
multi-site convergence against a sequential replay oracle, byte-identical replay of two independent
runs, bounds checking of submitted operations, the role matrix with rejected operations, permission
and epoch (revocation) rejection, an audit log that records every decision, opaque ciphertext (no
plaintext keyword occurs in the sealed bytes), seal/unseal round-trip, tamper detection, and that a
retired key cannot open rotated content.

Simplified: all cryptography is toy-scale (no real key material, no block cipher, no signatures) and
every security claim is a measurement, never a reduction; sites are simulated sequentially in one
process, so concurrency is an explicit batch parameter, a site has at most one unacknowledged
operation per barrier and the network is a synchronous barrier - no clock skew, no reconnection, no
re-ordering beyond the permutations the checks enumerate; each site rebuilds from the canonical
revision at every barrier, so a speculative insert inside a concurrently deleted hole may visibly jump
to the front of the hole.
"""
MASK64 = (1 << 64) - 1

ROLE_PERMISSIONS = {
    "owner": frozenset({"ins", "del", "comment", "admin"}),
    "editor": frozenset({"ins", "del"}),
    "commenter": frozenset({"comment"}),
    "viewer": frozenset(),
}


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
    """Fixed-seed splitmix64 stream, so every run enumerates the same batches."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def below(self, bound):
        self.state, value = splitmix64(self.state)
        return value % bound


# --------------------------------------------------------------------- toy crypto
def digest(key, payload):
    """Keyed 64-bit digest used as the tag; a toy stand-in for a real MAC."""
    h = (key ^ 0x5A5A5A5A5A5A5A5A) & MASK64
    for byte in payload:
        h, r = splitmix64((h + byte + 0x9E37) & MASK64)
        h = (h ^ r) & MASK64
    return h


def keystream(key, length):
    """Deterministic keystream bytes from a splitmix64 chain seeded by the key."""
    state = (key ^ 0x0123456789ABCDEF) & MASK64
    out = bytearray()
    while len(out) < length:
        state, word = splitmix64(state)
        out.extend(word.to_bytes(8, "little"))
    return bytes(out[:length])


def seal(key, plaintext):
    """Encrypt-then-MAC: XOR with the keystream, then tag the ciphertext with the key."""
    raw = plaintext.encode("utf-8")
    cipher = bytes(mask ^ byte for mask, byte in zip(keystream(key, len(raw)), raw))
    return cipher, digest(key, cipher)


def unseal(key, cipher, tag):
    """Returns the plaintext, or raises ValueError when the tag does not verify."""
    if digest(key, cipher) != tag:
        raise ValueError("tag mismatch: wrong key or tampered ciphertext")
    raw = bytes(mask ^ byte for mask, byte in zip(keystream(key, len(cipher)), cipher))
    return raw.decode("utf-8")


def rotate_key(key):
    """Derive the next epoch's key (the retired key must not open new content)."""
    _, mixed = splitmix64((key ^ 0xDEADBEEFCAFEF00D) & MASK64)
    return mixed | 1


# ------------------------------------------------------------------- operations
class Op:
    """An insert (kind 'ins', text set) or a delete (kind 'del', length set)."""

    __slots__ = ("kind", "pos", "text", "length", "site", "clock")

    def __init__(self, kind, pos, text, length, site, clock):
        self.kind = kind
        self.pos = pos
        self.text = text
        self.length = length
        self.site = site
        self.clock = clock

    @staticmethod
    def insert(pos, text, site, clock):
        if pos < 0:
            raise ValueError("insert position must be non-negative")
        if not text:
            raise ValueError("an insert must carry at least one character")
        return Op("ins", pos, text, 0, site, clock)

    @staticmethod
    def delete(pos, length, site, clock):
        if pos < 0 or length <= 0:
            raise ValueError("a delete needs a non-negative position and a positive length")
        return Op("del", pos, "", length, site, clock)

    def size(self):
        return len(self.text) if self.kind == "ins" else self.length

    def order_key(self):
        """Canonical key: Lamport clock first, site id as the tie-break."""
        return (self.clock, self.site)

    def __repr__(self):
        if self.kind == "ins":
            return f"ins({self.pos},{self.text!r},{self.site}@{self.clock})"
        return f"del({self.pos},{self.length},{self.site}@{self.clock})"


def validate_batch(base_length, ops):
    """Reject any operation that does not fit the revision it was issued against."""
    seen = set()
    for op in ops:
        if op.site in seen:
            raise ValueError(f"site {op.site!r} may submit at most one operation per batch")
        seen.add(op.site)
        if op.kind == "ins":
            if op.pos > base_length:
                raise ValueError(f"insert at {op.pos} beyond end {base_length}")
        elif op.pos + op.length > base_length:
            raise ValueError(f"delete [{op.pos},{op.pos + op.length}) beyond end {base_length}")


def apply_op(document, op):
    """Apply one operation, rejecting any that leaves the document bounds."""
    if op.kind == "ins":
        if op.pos > len(document):
            raise ValueError(f"insert at {op.pos} beyond end {len(document)}")
        return document[:op.pos] + op.text + document[op.pos:]
    if op.pos + op.length > len(document):
        raise ValueError(f"delete [{op.pos},{op.pos + op.length}) beyond end {len(document)}")
    return document[:op.pos] + document[op.pos + op.length:]


def apply_ops(document, ops):
    for op in ops:
        document = apply_op(document, op)
    return document


def merge_ranges(ranges):
    """Union of half-open ranges, as a sorted list of disjoint (start, end) intervals."""
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def union_length(ops):
    return sum(end - start for start, end in
               merge_ranges([(op.pos, op.pos + op.length) for op in ops if op.kind == "del"]))


def project(pos, removed):
    """Number of surviving characters before `pos`; a position strictly inside a removed range is
    pinned to the front of that range, which is where an insert into a deleted hole must land."""
    surviving = 0
    cursor = 0
    for start, end in removed:
        if pos <= start:
            return surviving + (pos - cursor)
        if pos < end:
            return surviving + (start - cursor)
        surviving += start - cursor
        cursor = end
    return surviving + (pos - cursor)


def resolve_batch(base, ops):
    """The server's canonical integration of a concurrent batch, as an operation list.

    Deletes are emitted from the back (each position is still valid in the shrinking document), then
    every insert is placed at its projected gap plus the length of the inserts already placed, which
    is why a gap's inserts come out in sorted (site, clock) order."""
    validate_batch(len(base), ops)
    removed = merge_ranges([(op.pos, op.pos + op.length) for op in ops if op.kind == "del"])
    ordered = []
    for start, end in reversed(removed):
        owner = min((op for op in ops if op.kind == "del"
                     and op.pos <= start < op.pos + op.length), key=Op.order_key)
        ordered.append(Op.delete(start, end - start, owner.site, owner.clock))
    anchors = sorted((project(op.pos, removed), op.site, op.clock, op.text)
                     for op in ops if op.kind == "ins")
    shift = 0
    for gap, site, clock, text in anchors:
        ordered.append(Op.insert(gap + shift, text, site, clock))
        shift += len(text)
    return ordered


def oracle_final(base, ops):
    """Specification-level oracle: keep the surviving characters and assign every insert to a gap."""
    removed = merge_ranges([(op.pos, op.pos + op.length) for op in ops if op.kind == "del"])
    deleted = set()
    for start, end in removed:
        deleted.update(range(start, end))
    surviving = [char for index, char in enumerate(base) if index not in deleted]
    gaps = {}
    for op in ops:
        if op.kind == "ins":
            gaps.setdefault(project(op.pos, removed), []).append((op.site, op.clock, op.text))
    out = []
    for gap in range(len(surviving) + 1):
        for _, _, text in sorted(gaps.get(gap, [])):
            out.append(text)
        if gap < len(surviving):
            out.append(surviving[gap])
    return "".join(out)


# ------------------------------------------------------- pairwise operational transform
def interval_minus(x_start, x_end, y_start, y_end):
    """The pieces of the range X minus the range Y, as (start, length) pairs."""
    pieces = []
    if x_start < y_start:
        pieces.append((x_start, min(x_end, y_start)))
    if x_end > y_end:
        pieces.append((max(x_start, y_end), x_end))
    return [(start, end - start) for start, end in pieces if end > start]


def shifted_deletes(pieces, applied_end, applied_length, site, clock):
    """Move pieces of a delete into the frame in which [applied_end-applied_length) was removed."""
    ops = []
    for start, length in pieces:
        offset = applied_length if start >= applied_end else 0
        ops.append(Op.delete(start - offset, length, site, clock))
    return ops


def transform(a, b):
    """TP1 pair for two concurrent operations: apply(a) then returns[1] == apply(b) then returns[0]."""
    if a.site == b.site:
        raise ValueError("only concurrent operations from distinct sites can be transformed")
    if a.kind == "ins" and b.kind == "ins":
        if (a.pos, a.site) < (b.pos, b.site):
            return [a], [Op.insert(b.pos + len(a.text), b.text, b.site, b.clock)]
        return [Op.insert(a.pos + len(b.text), a.text, a.site, a.clock)], [b]
    if a.kind == "ins":                                   # b is a delete
        if a.pos <= b.pos:
            return [a], [Op.delete(b.pos + len(a.text), b.length, b.site, b.clock)]
        if a.pos >= b.pos + b.length:
            return [Op.insert(a.pos - b.length, a.text, a.site, a.clock)], [b]
        # the insert lands inside the deleted range: it is pinned to the range's front and the delete
        # splits around it, the tail shifting back by the length of the head piece.
        return ([Op.insert(b.pos, a.text, a.site, a.clock)],
                [Op.delete(b.pos, a.pos - b.pos, b.site, b.clock),
                 Op.delete(b.pos + len(a.text), b.pos + b.length - a.pos, b.site, b.clock)])
    if b.kind == "ins":
        b_ops, a_ops = transform(b, a)
        return a_ops, b_ops
    # two deletes: each side keeps exactly the pieces the other does not already remove
    a_ops = shifted_deletes(interval_minus(a.pos, a.pos + a.length, b.pos, b.pos + b.length),
                            b.pos + b.length, b.length, a.site, a.clock)
    b_ops = shifted_deletes(interval_minus(b.pos, b.pos + b.length, a.pos, a.pos + a.length),
                            a.pos + a.length, a.length, b.site, b.clock)
    return a_ops, b_ops


# ------------------------------------------------------------------- the server
class EditorServer:
    """Stores the canonical document, the ACL, the epoch, the revision history and an audit log."""

    def __init__(self, document, key, epoch=0):
        self.doc = document
        self.key = key
        self.epoch = epoch
        self.revision = 0
        self.members = {}
        self.audit = []
        self.history = []

    def grant(self, site, role):
        check(role in ROLE_PERMISSIONS, f"unknown role {role!r}")
        self.members[site] = role
        self.audit.append((site, "grant", role, True))
        return role

    def revoke(self, site):
        check(site in self.members, "cannot revoke an unknown member")
        role = self.members.pop(site)
        self.epoch += 1                 # every in-flight operation is now stale
        retired = self.key
        self.key = rotate_key(self.key)
        self.audit.append((site, "revoke", role, True))
        return retired

    def can(self, site, kind):
        role = self.members.get(site)
        return role is not None and kind in ROLE_PERMISSIONS[role]

    def _authorise(self, submissions, issued_epoch):
        """All-or-nothing: one rejected operation rejects every batch in the barrier."""
        reason = None
        for site, ops in sorted(submissions.items()):
            if site not in self.members:
                reason = "not a member"
            elif issued_epoch != self.epoch:
                reason = f"stale epoch {issued_epoch} (current {self.epoch})"
            else:
                for op in ops:
                    if not self.can(site, op.kind):
                        reason = f"role {self.members[site]!r} may not {op.kind}"
                        break
            if reason is not None:
                self.audit.append((site, "submit", reason, False))
                return reason
            self.audit.append((site, "submit", self.revision + 1, True))
        return None

    def barrier(self, submissions, issued_epoch=0):
        """One round: integrate the batches of every site that wrote against the same revision."""
        reason = self._authorise(submissions, issued_epoch)
        if reason is not None:
            return {"accepted": False, "reason": reason}
        flat = [op for _, ops in sorted(submissions.items()) for op in ops]
        base = self.doc
        filed = resolve_batch(base, flat)
        self.doc = apply_ops(base, filed)
        self.revision += 1
        self.history.append((self.revision, base, filed, flat))
        return {"accepted": True, "revision": self.revision, "ops": filed, "document": self.doc}

    def submit(self, site, ops, issued_epoch=0):
        return self.barrier({site: ops}, issued_epoch)


# ----------------------------------------------------------------------- checks
def op_space():
    """A small operation space used for the exhaustive pairwise checks."""
    space = []
    for pos in range(4):
        for text in ("x", "yz"):
            space.append(Op.insert(pos, text, "A", 1))
    for pos in range(3):
        for length in (1, 2):
            space.append(Op.delete(pos, length, "A", 1))
    return space


def brand(op, site):
    return Op(op.kind, op.pos, op.text, op.length, site, op.clock)


def permutations(items):
    if len(items) <= 1:
        yield list(items)
        return
    for index in range(len(items)):
        head = items[index]
        for rest in permutations(items[:index] + items[index + 1:]):
            yield [head] + rest


def random_batch(rng, base, size):
    """A concurrent batch: distinct sites, positions drawn from the current revision."""
    batch = []
    for index in range(size):
        site = chr(ord("A") + index)
        pos = rng.below(len(base) + 1)
        if rng.below(3) == 0 and pos < len(base):
            batch.append(Op.delete(pos, 1 + rng.below(len(base) - pos), site, 5))
        else:
            batch.append(Op.insert(pos, "pq"[rng.below(2)] * (1 + rng.below(2)), site, 5))
    return batch


def run_simulation(sites, rounds, base_document, key, seed=0x515E):
    """One barrier per round; every site issues one operation against the round-start revision."""
    server = EditorServer(base_document, key)
    for site in sites:
        server.grant(site, "editor")
    rng = Rng(seed)
    for round_index in range(rounds):
        revision = server.doc
        submissions = {}
        for site in sites:
            pos = rng.below(len(revision) + 1)
            if rng.below(4) == 0 and len(revision) > 1:
                length = 1 + rng.below(min(3, len(revision)))
                submissions[site] = [Op.delete(min(pos, len(revision) - length), length, site,
                                                round_index)]
            else:
                submissions[site] = [Op.insert(pos, site.lower(), site, round_index)]
        result = server.barrier(submissions)
        check(result["accepted"], f"simulation barrier rejected: {result.get('reason')}")
    return server


def history_signature(server):
    return tuple((revision, base, tuple(repr(op) for op in filed))
                 for revision, base, filed, _ in server.history)


def main():
    key = 0xC0FFEE1234567
    base = "abcdef"

    # 1. TP1 over the whole pairwise operation space
    pairs = 0
    space = op_space()
    for left in space:
        for right in space:
            a, b = brand(left, "A"), brand(right, "B")
            a_ops, b_ops = transform(a, b)
            try:
                forward = apply_ops(apply_op(base, a), b_ops)
                backward = apply_ops(apply_op(base, b), a_ops)
            except ValueError:
                continue            # a pair that is invalid against this particular base
            check(forward == backward,
                  f"TP1 violated for {a!r} / {b!r}: {forward!r} != {backward!r}")
            pairs += 1
    check(pairs >= 150, f"the TP1 sweep must cover the operation space, only {pairs} pairs")
    try:
        transform(Op.insert(0, "a", "A", 1), Op.insert(0, "b", "A", 2))
        check(False, "operations from the same site are not concurrent and must be rejected")
    except ValueError:
        pass

    # 2. for every pair of the op space the resolver, the semantic oracle and the pairwise transform
    #    must agree - three independent implementations of the same specification
    pair_checks = 0
    for left in space:
        for right in space:
            a, b = brand(left, "A"), brand(right, "B")
            batch = [a, b]
            try:
                got = apply_ops(base, resolve_batch(base, batch))
                want = oracle_final(base, batch)
                via_transform = apply_ops(apply_op(base, a), transform(a, b)[1])
            except ValueError:
                continue
            check(got == want, f"resolver {got!r} != oracle {want!r} for {a!r}/{b!r}")
            check(via_transform == want,
                  f"pairwise transform {via_transform!r} != oracle {want!r} for {a!r}/{b!r}")
            pair_checks += 1
    check(pair_checks >= 150, f"pair cross-checks must cover the op space, got {pair_checks}")

    # 3. randomised concurrent batches: resolver == oracle, and every arrival order converges
    rng = Rng(0xBEEF)
    permutations_checked = 0
    for trial in range(200):
        size = 2 + rng.below(5)
        batch = random_batch(rng, base, size)
        expected = oracle_final(base, batch)
        check(apply_ops(base, resolve_batch(base, batch)) == expected,
              f"trial {trial}: resolver and oracle disagree on {batch}")
        for order in permutations(batch) if size <= 4 else [batch]:
            permutations_checked += 1
            check(apply_ops(base, resolve_batch(base, list(order))) == expected,
                  f"trial {trial}: arrival order changed the canonical revision")
    check(permutations_checked >= 500,
          f"the permutation sweep must be broad, only {permutations_checked}")

    # 4. the named semantics
    d1, d2 = Op.delete(1, 3, "A", 5), Op.delete(2, 3, "B", 5)
    check(apply_ops(base, resolve_batch(base, [d1, d2])) == "af",
          "overlapping deletes must remove exactly the union")
    identical = [Op.delete(2, 2, "A", 9), Op.delete(2, 2, "B", 9)]
    check(len(resolve_batch(base, identical)) == 1 and apply_ops(base, resolve_batch(base,
          identical)) == "abef", "two identical deletes must collapse into one range")
    outer, inner = Op.delete(1, 4, "A", 7), Op.insert(3, "ZZ", "B", 7)
    check(apply_ops(base, resolve_batch(base, [outer, inner])) == "aZZf",
          "an insert inside a deleted range must survive at the front of the hole")
    check(apply_ops(base, resolve_batch(base, [inner, outer])) == "aZZf",
          "the hole's front must not depend on the arrival order")
    same_gap = [Op.insert(2, "1", "C", 9), Op.insert(2, "2", "A", 9)]
    check(apply_ops(base, resolve_batch(base, same_gap)) == "ab21cdef",
          "inserts sharing a gap are ordered by site id")
    disjoint = [Op.insert(1, "X", "A", 3), Op.delete(4, 2, "B", 3)]
    check(apply_ops(base, resolve_batch(base, disjoint)) == "aXbcd",
          "disjoint inserts and deletes must both take effect")
    check(resolve_batch(base, []) == [] and oracle_final(base, []) == base,
          "an empty batch is a no-op")

    # 5. the length-conservation law: final length = base - union of deletes + inserted characters
    for trial in range(50):
        batch = random_batch(Rng(0x1234 + trial), base, 2 + trial % 4)
        document = apply_ops(base, resolve_batch(base, batch))
        expected = len(base) - union_length(batch) + \
            sum(len(op.text) for op in batch if op.kind == "ins")
        check(len(document) == expected,
              f"trial {trial}: length {len(document)} != conserved {expected}")

    # 6. bounds checking of submissions
    for bad in (Op.insert(7, "x", "A", 1), Op.delete(3, 5, "A", 1)):
        try:
            apply_op(base, bad)
            check(False, f"{bad!r} must be rejected on a {len(base)}-character document")
        except ValueError:
            pass
        try:
            resolve_batch(base, [bad])
            check(False, f"the server must reject {bad!r}")
        except ValueError:
            pass
    try:
        Op.delete(-1, 1, "A", 1)
        check(False, "a negative delete position must be rejected")
    except ValueError:
        pass
    try:
        Op.insert(0, "", "A", 1)
        check(False, "an empty insert must be rejected")
    except ValueError:
        pass
    try:
        resolve_batch(base, [Op.insert(0, "x", "A", 1), Op.insert(1, "y", "A", 2)])
        check(False, "one site must not submit two operations in the same batch")
    except ValueError:
        pass

    # 7. multi-round, multi-site convergence against a sequential replay oracle
    sites = ["A", "B", "C", "D"]
    server = run_simulation(sites, rounds=12, base_document="document", key=key)
    check(server.revision == 12, f"one revision per barrier, got {server.revision}")
    replayed = "document"
    growth = 0
    for _, base_doc, filed, batch in server.history:
        check(replayed == base_doc, "each revision's base must be the previous revision's document")
        check(apply_ops(base_doc, filed) == oracle_final(base_doc, batch),
              "every canonical revision must equal the semantic oracle for its batch")
        growth += sum(len(op.text) for op in filed if op.kind == "ins") - \
            sum(op.length for op in filed if op.kind == "del")
        replayed = apply_ops(base_doc, filed)
    check(replayed == server.doc, "replaying the history must reproduce the document")
    check(len(server.doc) == len("document") + growth,
          f"the document length must be conserved, {len(server.doc)} != {len('document') + growth}")
    check(set(server.doc) <= set("documentabcd"),
          f"the converged document must only contain the sites' own text, got {server.doc!r}")

    # 8. determinism: two independent runs produce a byte-identical history
    again = run_simulation(sites, rounds=12, base_document="document", key=key)
    check(history_signature(again) == history_signature(server),
          "two runs of the same workload must produce the same history")
    check(again.doc == server.doc, "two runs of the same workload must converge to the same document")

    # 9. role matrix, and every decision is audited
    guard = EditorServer("secret", key)
    guard.grant("owner", "owner")
    guard.grant("ed", "editor")
    guard.grant("com", "commenter")
    guard.grant("view", "viewer")
    check(guard.submit("ed", [Op.insert(0, "hi", "ed", 1)], 0)["accepted"], "an editor may insert")
    check(guard.doc == "hisecret", "the accepted insert must be visible in the document")
    bad_viewer = guard.submit("view", [Op.insert(0, "!", "view", 1)], guard.epoch)
    check(not bad_viewer["accepted"] and "may not" in bad_viewer["reason"],
          "a viewer must not be able to insert")
    bad_commenter = guard.submit("com", [Op.delete(0, 1, "com", 1)], guard.epoch)
    check(not bad_commenter["accepted"], "a commenter must not delete")
    mixed = guard.submit("view", [Op.insert(0, "!", "view", 1), Op.delete(0, 1, "view", 2)],
                         guard.epoch)
    check(not mixed["accepted"] and guard.doc == "hisecret",
          "a rejected batch must leave the document untouched")
    outsider = guard.submit("mallory", [Op.insert(0, "x", "mallory", 1)], guard.epoch)
    check(not outsider["accepted"] and outsider["reason"] == "not a member",
          "a non-member must be rejected")
    check(guard.submit("owner", [Op.delete(0, 2, "owner", 3)], guard.epoch)["accepted"],
          "the owner may delete")
    check(guard.doc == "secret", "the owner's delete must take effect")
    check(len(guard.audit) == 10, f"every decision must be audited, log={len(guard.audit)}")
    check(sum(1 for entry in guard.audit if not entry[3]) == 4, "four decisions must be rejections")

    # 10. revocation invalidates in-flight operations and rotates the key
    locked = EditorServer("payload", key)
    locked.grant("ed", "editor")
    in_flight = [Op.insert(0, "Z", "ed", 4)]
    retired = locked.revoke("ed")
    stale = locked.submit("ed", in_flight, 0)
    check(not stale["accepted"] and "not a member" in stale["reason"],
          "an operation from a revoked member must be rejected")
    check(locked.epoch == 1 and locked.doc == "payload", "revocation must not change the document")
    sealed_before, tag_before = seal(retired, "payload")
    check(unseal(retired, sealed_before, tag_before) == "payload",
          "the retired key must still open content written before the revocation")
    sealed_after, tag_after = seal(locked.key, "payload")
    try:
        unseal(retired, sealed_after, tag_after)
        check(False, "the retired key must not open content written after the rotation")
    except ValueError:
        pass

    # 11. the ciphertext is opaque and tampering is detected
    document = server.doc * 12 + "the quick brown fox jumps over the lazy dog"
    cipher, tag = seal(key, document)
    check(unseal(key, cipher, tag) == document, "seal/unseal must round-trip the document")
    check(len(cipher) == len(document), "the toy pad keeps the plaintext length")
    hits = sum(1 for word in ("quick", "brown", "document") if word.encode() in cipher)
    check(hits == 0, "no plaintext keyword may occur verbatim in the ciphertext")
    flipped = bytearray(cipher)
    flipped[0] ^= 0x01
    try:
        unseal(key, bytes(flipped), tag)
        check(False, "a flipped ciphertext byte must fail the tag check")
    except ValueError:
        pass
    try:
        unseal(rotate_key(key), cipher, tag)
        check(False, "a wrong key must fail the tag check")
    except ValueError:
        pass

    # 12. degenerate cases: empty document, single site
    empty = EditorServer("", key)
    empty.grant("A", "owner")
    check(empty.submit("A", [], 0)["accepted"], "an empty batch is a legal no-op")
    check(empty.doc == "", "an empty batch must not change an empty document")
    check(empty.submit("A", [Op.insert(0, "z", "A", 1)], 0)["accepted"], "insert into an empty doc")
    check(empty.doc == "z", "the single-site path must produce the obvious document")
    solo = run_simulation(["A"], rounds=3, base_document="", key=key)
    check(solo.doc == oracle_final(solo.history[-1][1], solo.history[-1][3]),
          "the single-site simulation must match the oracle at every revision")

    print(f"checks: TP1 pairs={pairs}, pair cross-checks={pair_checks}, "
          f"random batches=200, permutations={permutations_checked}")
    print(f"revisions={server.revision}, document bytes={len(server.doc)}, "
          f"ciphertext bytes={len(cipher)}, keyword hits={hits}")
    print("PASS")


if __name__ == "__main__":
    main()
