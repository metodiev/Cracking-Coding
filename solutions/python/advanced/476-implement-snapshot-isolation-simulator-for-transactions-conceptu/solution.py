"""Task 476 - Implement snapshot isolation simulator for transactions (conceptual). (Tier: Advanced, CTCI Ch 14 — Databases)

MVCC snapshot isolation: every key keeps an append-only version list of (commit timestamp,
value). A transaction pins the latest commit timestamp at BEGIN and only ever reads versions
with ts <= its snapshot, plus its own buffered writes (read-your-writes); COMMIT is
first-committer-wins: if any key it wrote gained a version after the snapshot, the transaction
aborts. Asserted: no dirty reads, no dirty writes (lost updates are aborted), repeatable
snapshot reads, and the classic write-skew anomaly being *allowed* (each transaction checks
both x and y, then writes a disjoint key - both commit, breaking the "at least one on call"
invariant; documented as an SI limitation). A random workload is validated against an oracle
that replays snapshot copies and the same conflict rule.
"""

KEYS = ["a", "b", "c", "d"]
INITIAL = {"a": 0, "b": 1, "c": 2, "d": 3}


class Txn:
    def __init__(self, tid, snapshot):
        self.id = tid
        self.snapshot = snapshot
        self.writes = {}
        self.state = "active"


class MVCCStore:
    def __init__(self, initial):
        self.versions = {}
        self.commit_counter = 0
        self.next_tid = 0
        for key, value in initial.items():
            self._append_committed(key, value)

    def _append_committed(self, key, value):
        self.commit_counter += 1
        self.versions.setdefault(key, []).append((self.commit_counter, value))

    def begin(self):
        self.next_tid += 1
        return Txn("T%d" % self.next_tid, self.commit_counter)

    def read(self, txn, key):
        if key in txn.writes:
            return txn.writes[key]              # read-your-writes
        best = None
        for ts, value in self.versions.get(key, []):
            if ts <= txn.snapshot:              # only commit-time state as of BEGIN
                best = value
        return best

    def write(self, txn, key, value):
        txn.writes[key] = value

    def commit(self, txn):
        for key in txn.writes:
            for ts, _value in self.versions.get(key, []):
                if ts > txn.snapshot:           # someone committed this key after BEGIN
                    txn.state = "aborted"
                    return False
        self.commit_counter += 1
        ts = self.commit_counter
        for key, value in txn.writes.items():
            self.versions.setdefault(key, []).append((ts, value))
        txn.state = "committed"
        return True

    def materialize(self):
        out = {}
        for key, versions in self.versions.items():
            latest = None
            for _ts, value in versions:
                latest = value
            out[key] = latest
        return out


def lcg(seed):
    return (seed * 75 + 74) % 65537


def random_script(count, seed):
    script = []
    for _ in range(count):
        seed = lcg(seed)
        action = seed % 5
        seed = lcg(seed)
        pick = seed
        seed = lcg(seed)
        key = KEYS[seed % len(KEYS)]
        seed = lcg(seed)
        value = seed % 100
        script.append((action, pick, key, value))
    return script


def run_random_phase():
    store = MVCCStore(INITIAL)
    committed = dict(INITIAL)
    version_ts = {}
    ts = 1
    for key in INITIAL:
        version_ts[key] = ts
        ts += 1
    counter = store.commit_counter
    assert counter == 4
    live = {}
    next_tid = 0
    for action, pick, key, value in random_script(60, 97531):
        if action == 0 or not live:
            next_tid += 1
            txn = store.begin()
            assert txn.snapshot == counter
            live[txn.id] = (txn, {"snapshot": dict(committed), "writes": {}})
        else:
            tid = sorted(live)[pick % len(live)]
            txn, oracle = live[tid]
            if action in (1, 4):
                got = store.read(txn, key)
                expected = oracle["writes"].get(key, oracle["snapshot"].get(key))
                assert got == expected, (tid, key, got, expected)
            elif action == 2:
                store.write(txn, key, value)
                oracle["writes"][key] = value
            else:
                success = all(version_ts.get(k, -1) <= txn.snapshot for k in txn.writes)
                got = store.commit(txn)
                assert got == success, (tid, got, success)
                if got:
                    counter = store.commit_counter
                    for k, v in txn.writes.items():
                        version_ts[k] = counter
                        committed[k] = v
                del live[tid]
        assert store.materialize() == committed
        assert store.commit_counter == counter
    for txn, oracle in live.values():
        for key in KEYS:
            assert store.read(txn, key) == oracle["writes"].get(key, oracle["snapshot"].get(key))
    assert store.materialize() == committed


def main():
    store = MVCCStore(INITIAL)                 # versions a@1, b@2, c@3, d@4
    t1 = store.begin()                         # snapshot 4
    store.write(t1, "a", 99)
    t2 = store.begin()                         # snapshot 4
    t3 = store.begin()                         # snapshot 4
    assert store.read(t2, "a") == 0            # no dirty read of t1's uncommitted 99
    store.write(t2, "a", 100)                  # t1 and t2 both write "a"
    assert store.commit(t1)                    # first committer wins
    assert not store.commit(t2), "write-write conflict must abort the loser"
    assert store.read(t3, "a") == 0            # snapshot is pinned before t1's commit
    assert store.read(t3, "a") == 0            # repeatable read
    assert store.commit(t3)                    # read-only txn commits fine
    t4 = store.begin()
    assert store.read(t4, "a") == 99           # a fresh snapshot sees the commit

    t5 = store.begin()
    store.write(t5, "d", 6)
    assert store.read(t5, "d") == 6            # read-your-writes
    assert store.commit(t5)
    t6 = store.begin()
    assert store.read(t6, "d") == 6

    t7 = store.begin()
    t8 = store.begin()
    assert store.read(t7, "d") == 6 and store.read(t8, "d") == 6
    store.write(t7, "d", 7)
    store.write(t8, "d", 8)
    assert store.commit(t7)
    assert not store.commit(t8), "lost update must be prevented"
    assert store.read(store.begin(), "d") == 7
    assert store.materialize() == {"a": 99, "b": 1, "c": 2, "d": 7}

    # Write skew (documented SI anomaly): both txns see x=y="on", then each writes a
    # *disjoint* key, so first-committer-wins has nothing to reject and both commit.
    docs = MVCCStore({"x": "on", "y": "on"})
    t9 = docs.begin()
    t10 = docs.begin()
    assert docs.read(t9, "x") == "on" and docs.read(t9, "y") == "on"
    assert docs.read(t10, "x") == "on" and docs.read(t10, "y") == "on"
    docs.write(t9, "x", "off")
    docs.write(t10, "y", "off")
    assert docs.commit(t9) and docs.commit(t10), "write skew is allowed under SI"
    assert docs.materialize() == {"x": "off", "y": "off"}, "the on-call invariant is broken"
    fresh = docs.begin()
    assert docs.read(fresh, "x") == "off" and docs.read(fresh, "y") == "off"

    run_random_phase()


if __name__ == "__main__":
    main()
    print("PASS")
