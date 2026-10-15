"""Task 474 - Implement Merkle tree for log integrity and proof generation. (Tier: Advanced, CTCI Beyond — Security & Privacy)

Merkle tree over an append-only log of records using FNV-1a-32 with domain separation
(leaves hashed with a 0x00 prefix, internal nodes over the two 4-byte big-endian child
hashes). Odd levels promote the last node unchanged. Inclusion proofs (audit paths) are
checked by recomputing the root; tampered leaves, tampered siblings and wrong roots must all
fail. Expected roots for the fixed records below were verified with a separate oracle run.
"""

MASK = 0xFFFFFFFF


def fnv1a(data):
    h = 2166136261
    for b in data:
        h = ((h ^ b) * 16777619) & MASK
    return h


def hash_leaf(value):
    return fnv1a(b"\x00" + value.encode())


def hash_node(left, right):
    return fnv1a(left.to_bytes(4, "big") + right.to_bytes(4, "big"))


class MerkleTree:
    def __init__(self, leaves):
        self.leaves = list(leaves)
        self.levels = [[hash_leaf(v) for v in self.leaves]]
        level = self.levels[0]
        while len(level) > 1:
            nxt = [hash_node(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
            if len(level) % 2 == 1:
                nxt.append(level[-1])       # promote the odd last node unchanged
            self.levels.append(nxt)
            level = nxt

    def root(self):
        return self.levels[-1][0]

    def proof(self, index):
        path = []
        idx = index
        for level in self.levels[:-1]:
            if idx % 2 == 1:
                path.append((level[idx - 1], "left"))
            elif idx + 1 < len(level):
                path.append((level[idx + 1], "right"))
            idx //= 2
        return path


def verify(root, value, path):
    h = hash_leaf(value)
    for sibling, side in path:
        h = hash_node(sibling, h) if side == "left" else hash_node(h, sibling)
    return h == root


def main():
    # Seven records: the odd level count exercises the promotion rule.
    leaves = ["tx%d" % i for i in range(7)]
    tree = MerkleTree(leaves)
    assert tree.root() == 818430076, "root of the 7-record log"
    for i in range(7):
        assert verify(tree.root(), leaves[i], tree.proof(i)), ("proof must verify", i)
    # Proof shape: index 0 walks three levels, promoted index 6 has no sibling at level 0.
    assert len(tree.proof(0)) == 3
    assert len(tree.proof(6)) == 2
    # Tampering is detected.
    assert not verify(tree.root(), "tx3x", tree.proof(3)), "tampered leaf must fail"
    assert not verify(tree.root() ^ 1, leaves[3], tree.proof(3)), "wrong root must fail"
    bad = list(tree.proof(3))
    bad[0] = (bad[0][0] ^ 1, bad[0][1])
    assert not verify(tree.root(), leaves[3], bad), "tampered sibling must fail"
    # Root equals a fresh recomputation, and proofs agree with it.
    assert MerkleTree(leaves).root() == tree.root()
    assert all(verify(MerkleTree(leaves).root(), v, tree.proof(i)) for i, v in enumerate(leaves))

    # Even count (power of two): every audit path is exactly log2(8) = 3 hashes long.
    leaves8 = ["tx%d" % i for i in range(8)]
    tree8 = MerkleTree(leaves8)
    assert tree8.root() == 955777637, "root of the 8-record log"
    for i in range(8):
        assert len(tree8.proof(i)) == 3
        assert verify(tree8.root(), leaves8[i], tree8.proof(i))
    assert not verify(tree8.root(), "tx0!", tree8.proof(0))
    assert tree8.root() != tree.root(), "different logs, different roots"

    # A single record is its own root with an empty proof.
    single = MerkleTree(["only"])
    assert single.root() == hash_leaf("only")
    assert single.proof(0) == []
    assert verify(single.root(), "only", [])


if __name__ == "__main__":
    main()
    print("PASS")
