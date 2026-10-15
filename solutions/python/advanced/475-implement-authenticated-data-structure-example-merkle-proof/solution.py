"""Task 475 - Implement authenticated data structure example (Merkle+proof). (Tier: Advanced, CTCI Beyond — Security & Privacy)

Authenticated log on top of task 474's Merkle tree: minimal *batch* (multi) proofs for a set of
records, plus update-path handling. A multi-proof carries one sibling hash per level per
frontier node, so proving two adjacent records costs fewer hashes than two separate audit
paths. Update paths recompute the new root from (old siblings + new record); stale proofs and
stale multi-proofs must fail against the updated root. All expected hash values were verified
with a separate oracle run.
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


class AuthenticatedLog:
    def __init__(self, records):
        self.records = list(records)
        self.levels = [[hash_leaf(v) for v in self.records]]
        level = self.levels[0]
        while len(level) > 1:
            nxt = [hash_node(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
            if len(level) % 2 == 1:
                nxt.append(level[-1])
            self.levels.append(nxt)
            level = nxt

    @property
    def size(self):
        return len(self.records)

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

    def multi_proof(self, targets):
        level = self.levels[0]
        needed = set(targets)
        out = []
        while len(level) > 1:
            siblings = []
            for i in sorted(needed):
                if i % 2 == 1 and (i - 1) not in needed:
                    siblings.append((i - 1, level[i - 1]))
                elif i % 2 == 0 and i + 1 < len(level) and (i + 1) not in needed:
                    siblings.append((i + 1, level[i + 1]))
            out.append(siblings)
            nxt = [hash_node(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
            if len(level) % 2 == 1:
                nxt.append(level[-1])
            level = nxt
            needed = {i // 2 for i in needed}
        return out


def verify(root, value, path):
    h = hash_leaf(value)
    for sibling, side in path:
        h = hash_node(sibling, h) if side == "left" else hash_node(h, sibling)
    return h == root


def verify_multi(root, n, targets, proof):
    known = {i: hash_leaf(v) for i, v in targets.items()}
    level_len = n
    for siblings in proof:
        sib_map = dict(siblings)
        parents = {}
        for i in range(0, level_len - 1, 2):
            have_left = i in known
            have_right = (i + 1) in known
            if not have_left and not have_right:
                continue
            if have_left and have_right:
                parents[i // 2] = hash_node(known[i], known[i + 1])
            else:
                pos = i + 1 if have_left else i
                if pos not in sib_map:
                    return False
                left = known[i] if have_left else sib_map[i]
                right = known[i + 1] if have_right else sib_map[i + 1]
                parents[i // 2] = hash_node(left, right)
        if level_len % 2 == 1 and (level_len - 1) in known:
            parents[(level_len - 1) // 2] = known[level_len - 1]
        known = parents
        level_len = (level_len + 1) // 2
    return known.get(0) == root


def updated_root(new_value, path):
    h = hash_leaf(new_value)
    for sibling, side in path:
        h = hash_node(sibling, h) if side == "left" else hash_node(h, sibling)
    return h


def main():
    leaves8 = ["tx%d" % i for i in range(8)]
    log = AuthenticatedLog(leaves8)
    assert log.root() == 955777637

    # Record 2's audit path (sibling hashes verified with the oracle).
    path2 = log.proof(2)
    assert [h for h, _side in path2] == [1947293384, 240787827, 3271786783]
    assert [side for _h, side in path2] == ["right", "left", "right"]

    # Batch proof for adjacent records 0,1: no level-0 sibling, two upper frontier hashes.
    mp01 = log.multi_proof([0, 1])
    assert [len(s) for s in mp01] == [0, 1, 1]
    assert [h for level in mp01 for _pos, h in level] == [1737906683, 3271786783]
    assert verify_multi(log.root(), 8, {0: "tx0", 1: "tx1"}, mp01)
    assert sum(len(s) for s in mp01) < len(log.proof(0)) + len(log.proof(1)), "batching must save hashes"

    mp03 = log.multi_proof([0, 3])
    assert [len(s) for s in mp03] == [2, 0, 1]
    assert verify_multi(log.root(), 8, {0: "tx0", 3: "tx3"}, mp03)

    mp_all = log.multi_proof(list(range(8)))
    assert [len(s) for s in mp_all] == [0, 0, 0]
    assert verify_multi(log.root(), 8, {i: "tx%d" % i for i in range(8)}, mp_all)

    mp246 = log.multi_proof([2, 4, 6])
    assert [len(s) for s in mp246] == [3, 1, 0]
    assert verify_multi(log.root(), 8, {2: "tx2", 4: "tx4", 6: "tx6"}, mp246)

    # Single-target multi-proof carries exactly the audit-path sibling hashes.
    mp5 = log.multi_proof([5])
    assert sorted(h for level in mp5 for _pos, h in level) == sorted(h for h, _side in log.proof(5))
    assert verify_multi(log.root(), 8, {5: "tx5"}, mp5)

    # Tampering the target or any provided sibling must fail.
    bad_sib = [list(level) for level in mp01]
    bad_sib[1][0] = (bad_sib[1][0][0] ^ 1, bad_sib[1][0][1])
    assert not verify_multi(log.root(), 8, {0: "tx0", 1: "tx1"}, bad_sib)
    assert not verify_multi(log.root(), 8, {0: "tx0x", 1: "tx1"}, mp01)

    # Odd leaf count (promotion): batch {5,6} has one sibling below and one at the top.
    leaves7 = ["tx%d" % i for i in range(7)]
    log7 = AuthenticatedLog(leaves7)
    mp56 = log7.multi_proof([5, 6])
    assert [len(s) for s in mp56] == [1, 0, 1]
    assert verify_multi(log7.root(), 7, {5: "tx5", 6: "tx6"}, mp56)
    mp5b = log7.multi_proof([5])
    assert [len(s) for s in mp5b] == [1, 1, 1]
    assert verify_multi(log7.root(), 7, {5: "tx5"}, mp5b)

    # Update path: replacing tx2 by tx2* recomputes the new root from the old siblings.
    updated = list(leaves8)
    updated[2] = "tx2*"
    log_new = AuthenticatedLog(updated)
    assert log_new.root() == 3100337682
    assert updated_root("tx2*", path2) == 3100337682, "update path must yield the new root"
    assert not verify(log_new.root(), "tx2", path2), "stale proof must fail after update"
    assert verify(log_new.root(), "tx2*", log_new.proof(2))
    assert not verify_multi(log_new.root(), 8, {0: "tx0", 1: "tx1"}, mp01), "stale batch must fail"
    mp25_new = log_new.multi_proof([2, 5])
    assert verify_multi(log_new.root(), 8, {2: "tx2*", 5: "tx5"}, mp25_new)
    assert not verify(log_new.root(), "tx5", log.proof(5)), "old proof of unaffected record fails too"


if __name__ == "__main__":
    main()
    print("PASS")
