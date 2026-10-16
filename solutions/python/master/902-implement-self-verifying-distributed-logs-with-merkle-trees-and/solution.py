"""Task 902 - Implement self-verifying distributed logs with Merkle trees and authenticated queries.
(Tier: Master, CTCI Beyond - Security & Privacy)

What is implemented (in-process, deterministic; sha256/hmac are the only "crypto", and the log
keeps its records so a verifier can be cross-checked against an independent tree construction):

  * `MerkleLog` - an append-only log whose hash tree follows the RFC 6962 shape: leaves are
    H(0x00 || record), internal nodes are H(0x01 || left || right), and an unpaired trailing
    node is *promoted* unchanged instead of duplicating the last leaf (the domain-separation
    prefix blocks second-preimage leaf/node confusion).
  * `audit_path` / `verify_inclusion` - an O(log n) inclusion proof from a leaf index to the
    root; the verifier only ever sees the record, its index, the sibling hashes and the root.
  * `reference_root` - a completely independent recursive construction (split at the largest
    power of two below the size); agreement of the two roots is the oracle for the tree shape.
  * `sign_tree_head` / `verify_tree_head` - a signed tree head (size + root) as an HMAC, plus
    `AuthenticatedMap`, a sorted-log authenticated map: membership is proven by an inclusion
    proof, absence by the two records adjacent to the probe key plus their log positions.

Simplified versus production: signatures are HMACs under a fixed in-file key rather than real
asymmetric signatures, and consistency between two tree heads is checked by re-deriving the
prefix root rather than by shipping an O(log n) consistency proof.  The verified invariant is
the real one: changing a record or a sibling hash changes the root, and every proof that
depended on it fails.
"""

import bisect
import hashlib
import hmac

EMPTY_ROOT = hashlib.sha256(b"").digest()
HEAD_KEY = b"cracking-coding/902/self-verifying-log"


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def as_bytes(record):
    return record.encode("utf-8") if isinstance(record, str) else bytes(record)


def leaf_hash(record):
    return hashlib.sha256(b"\x00" + as_bytes(record)).digest()


def node_hash(left, right):
    return hashlib.sha256(b"\x01" + left + right).digest()


class MerkleLog:
    """levels[k][i] is the hash of the 2**k leaves starting at leaf i * 2**k."""

    def __init__(self, records=()):
        self.records = [as_bytes(r) for r in records]
        self._build()

    def _build(self):
        self.levels = [[leaf_hash(r) for r in self.records]] if self.records else []
        while self.levels and len(self.levels[-1]) > 1:
            cur = self.levels[-1]
            nxt = [node_hash(cur[i], cur[i + 1]) for i in range(0, len(cur) - 1, 2)]
            if len(cur) % 2:
                nxt.append(cur[-1])          # promoted, never duplicated
            self.levels.append(nxt)

    @property
    def size(self):
        return len(self.records)

    @property
    def root(self):
        return self.levels[-1][0] if self.levels else EMPTY_ROOT

    @property
    def height(self):
        return len(self.levels)

    def append(self, record):
        if len(self.levels) == 0:
            self.records = [as_bytes(record)]
        else:
            self.records.append(as_bytes(record))
        self._build()

    def prefix_root(self, count):
        return MerkleLog(self.records[:count]).root

    def audit_path(self, index):
        check(0 <= index < self.size, "audit index %d out of range" % index)
        path = []
        for level in self.levels[:-1]:
            sibling = index ^ 1
            if sibling < len(level):          # otherwise the node was promoted
                path.append(("L" if index % 2 else "R", level[sibling]))
            index //= 2
        return path


def verify_inclusion(record, index, path, root):
    """Independent verifier: folds the sibling hashes onto the leaf in bottom-up order."""
    node = leaf_hash(record)
    for side, sibling in path:
        node = node_hash(sibling, node) if side == "L" else node_hash(node, sibling)
    return node == root


def reference_root(records):
    """Independent oracle for the tree shape: recursion split at the largest power of two."""
    leaves = [leaf_hash(r) for r in records]

    def build(lo, hi):
        if hi - lo == 1:
            return leaves[lo]
        span = 1
        while span * 2 < hi - lo:
            span *= 2
        return node_hash(build(lo, lo + span), build(lo + span, hi))

    return build(0, len(leaves)) if leaves else EMPTY_ROOT


def sign_tree_head(size, root):
    return hmac.new(HEAD_KEY, b"%d:" % size + root, hashlib.sha256).digest()


def verify_tree_head(size, root, signature):
    return hmac.compare_digest(sign_tree_head(size, root), signature)


class AuthenticatedMap:
    """A sorted-key log: keys are proved present by inclusion, absent by adjacency."""

    def __init__(self, pairs):
        self.pairs = sorted(pairs)
        self.log = MerkleLog(["%s=%s" % (k, v) for k, v in self.pairs])
        self.position = {k: i for i, (k, _) in enumerate(self.pairs)}

    @property
    def root(self):
        return self.log.root

    def member_proof(self, key):
        index = self.position.get(key)
        if index is None:
            return None
        return (self.pairs[index][1], index, self.log.audit_path(index))

    def absence_proof(self, key):
        """The two records adjacent to the insertion point, with their log positions."""
        spot = bisect.bisect_left([k for k, _ in self.pairs], key)
        window = []
        for index in (spot - 1, spot):
            if 0 <= index < len(self.pairs):
                window.append((index, self.pairs[index], self.log.audit_path(index)))
        return window


def verify_member(key, value, index, path, root):
    return verify_inclusion("%s=%s" % (key, value), index, path, root)
def verify_absence(key, window, root, size):
    """Both neighbours must be included, and their log positions must be adjacent - for a sorted
    log that is what certifies that no record lies between the predecessor and the successor.
    At the ends of the log the proof degenerates to the single record that sits at position 0
    (proving that nothing smaller exists) or at position size-1 (nothing larger)."""
    if not window:
        return False
    for index, (k, v), path in window:
        if not verify_member(k, v, index, path, root):
            return False
    lower = [entry for entry in window if entry[1][0] < key]
    upper = [entry for entry in window if entry[1][0] > key]
    if lower and upper:
        predecessor = max(lower, key=lambda entry: entry[1][0])
        successor = min(upper, key=lambda entry: entry[1][0])
        return successor[0] == predecessor[0] + 1
    if upper:
        return upper[0][0] == 0
    if lower:
        return lower[-1][0] == size - 1
    return False


def main():
    records = ["entry-%04d" % i for i in range(40)]

    # ---- 1. tree shape: the level construction and the recursive oracle must agree
    check(MerkleLog([]).root == EMPTY_ROOT and MerkleLog([]).height == 0, "empty log boundary")
    check(MerkleLog([]).root == reference_root([]), "empty log oracle agreement")
    for size in range(0, 34):
        tree = MerkleLog(records[:size])
        check(tree.root == reference_root(records[:size]),
              "root mismatch against the recursive oracle at size %d" % size)
    check(MerkleLog(records[:1]).root == leaf_hash(records[0]),
          "a single record must be its own leaf hash")

    # ---- 2. every inclusion proof of every index verifies; proof length is logarithmic
    for size in range(1, 26):
        tree = MerkleLog(records[:size])
        for index in range(size):
            path = tree.audit_path(index)
            check(verify_inclusion(records[index], index, path, tree.root),
                  "inclusion proof must verify (size=%d index=%d)" % (size, index))
            check(len(path) <= max(1, (size - 1).bit_length()),
                  "audit path must be O(log n) (size=%d index=%d)" % (size, index))
        if size & (size - 1) == 0:
            check(len(tree.audit_path(0)) == size.bit_length() - 1,
                  "a perfect tree has exactly log2(n) path steps (size=%d)" % size)
    check(MerkleLog(records[:1]).audit_path(0) == [], "a one-leaf tree has an empty audit path")

    # ---- 3. tampering: record, sibling, index, root
    tree = MerkleLog(records[:13])
    path = tree.audit_path(5)
    forged = MerkleLog(records[:13])
    forged.records[5] = as_bytes("entry-0005!")
    forged._build()
    check(forged.root != tree.root, "a changed record must change the root")
    check(not verify_inclusion("entry-0005!", 5, path, tree.root),
          "the old proof must reject the rewritten record")
    check(verify_inclusion("entry-0005!", 5, forged.audit_path(5), forged.root),
          "the rewritten record verifies against the rewritten tree's own root")
    flipped = [("L" if side == "R" else "R", sibling) for side, sibling in path]
    check(not verify_inclusion(records[5], 5, flipped, tree.root),
          "flipping the sibling side must invalidate the proof")
    check(not verify_inclusion(records[5], 6, tree.audit_path(6), tree.root),
          "a proof from another index must not verify")
    check(not verify_inclusion(records[5], 5, path, MerkleLog(records[:14]).root),
          "a proof must not verify under another tree's root")
    try:
        tree.audit_path(13)
        raise AssertionError("out-of-range audit index must be rejected")
    except AssertionError as error:
        check("out of range" in str(error), "the rejection must name the offending index")

    # ---- 4. signed tree heads
    head = sign_tree_head(tree.size, tree.root)
    check(verify_tree_head(tree.size, tree.root, head), "a valid tree head must verify")
    check(not verify_tree_head(tree.size + 1, tree.root, head), "a stale size must be rejected")
    check(not verify_tree_head(tree.size, forged.root, head), "a forged root must be rejected")
    check(not verify_tree_head(tree.size, tree.root, head[:-1] + bytes([head[-1] ^ 1])),
          "a flipped signature bit must be rejected")

    # ---- 5. authenticated queries over the sorted log
    table = AuthenticatedMap([("k%03d" % i, i * i) for i in range(24)])
    check(table.root == reference_root(["%s=%s" % kv for kv in table.pairs]),
          "the authenticated map must use the same tree construction")
    for key in ("k000", "k007", "k023"):
        value, index, path = table.member_proof(key)
        check(verify_member(key, value, index, path, table.root),
              "membership proof for %s must verify" % key)
        check(value == int(key[1:]) ** 2, "the proved value must be the stored one")
    check(table.member_proof("k999") is None, "an absent key has no membership proof")
    # boundary probes: inside the key space and past both ends, where the proof degenerates
    for probe in ("k007a", "a", "zzz"):
        window = table.absence_proof(probe)
        check(window, "an absence proof needs at least one neighbour")
        check(verify_absence(probe, window, table.root, table.log.size),
              "absence of %s must verify with adjacent records" % probe)
    check(len(table.absence_proof("a")) == 1 and table.absence_proof("a")[0][0] == 0,
          "a probe below every key must prove position 0")
    forged_window = table.absence_proof("k007a")
    forged_index, (k, v), forged_path = forged_window[-1]
    check(not verify_absence("k007a", [(forged_index, (k, v + 1), forged_path)],
                             table.root, table.log.size), "a forged absence value must fail verification")
    shifted = [(forged_index + 1, entry[1], entry[2]) for entry in table.absence_proof("k007a")]
    check(not verify_absence("k007a", shifted, table.root, table.log.size),
          "absence must fail when the bracketing positions are not adjacent")
    check(not verify_absence("k007a", [], table.root, table.log.size),
          "an empty window must not prove anything")
    crafted = table.absence_proof("k007a")
    check(not verify_absence("a", crafted, table.root, table.log.size),
          "neighbours of one key must not prove absence of another")

    # ---- 6. appends preserve complete subtrees (prefix commitment) and stay logarithmic
    growing = MerkleLog(records[:8])
    perfect = [growing.levels[level][0] for level in range(growing.height)]
    for extra in records[8:13]:
        growing.append(extra)
    check([growing.levels[level][0] for level in range(len(perfect))] == perfect,
          "appending must not change the hashes of complete subtrees")
    for index in range(13):
        check(verify_inclusion(records[index], index, growing.audit_path(index), growing.root),
              "old records must still prove after an append (index=%d)" % index)

    # ---- 7. duplicates and scale
    dup = MerkleLog(["same", "same", "same"])
    check(dup.levels[0][0] == dup.levels[0][1] == leaf_hash("same"),
          "identical records must hash identically")
    check(all(verify_inclusion("same", i, dup.audit_path(i), dup.root) for i in range(3)),
          "duplicate records must still prove at their own positions")
    check(dup.root != MerkleLog(["same", "same"]).root, "size must be part of the commitment")
    big = MerkleLog(["record-%05d" % i for i in range(257)])
    check(big.root == reference_root(big.records), "257 records must agree with the oracle")
    proofs = all(verify_inclusion("record-%05d" % i, i, big.audit_path(i), big.root)
                 for i in range(257))
    check(proofs, "all 257 inclusion proofs must verify")
    print("log size 257: height %d, max path %d" % (big.height, max(len(big.audit_path(i)) for i in range(257))))
    print("PASS")


if __name__ == "__main__":
    main()
