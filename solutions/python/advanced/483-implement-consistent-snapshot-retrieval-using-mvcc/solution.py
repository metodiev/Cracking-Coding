"""Task 483 - Implement consistent snapshot retrieval using MVCC. (Tier: Advanced, CTCI Ch 14)

Multi-version store: every key holds the full list of (commit_ts, value) versions produced by
whole-transaction commits (all keys of one transaction share a single commit timestamp). A read
issued at snapshot timestamp s returns, per key, the newest version with commit_ts <= s, i.e. the
state as of a single point in time even if newer commits exist. Because one transaction stamps all
of its writes with the same ts, a snapshot never observes a torn transaction: it sees every key of
the transaction or none of it.

The scripted history is seeded with base versions at ts = 0, then 12 LCG-driven transactions at
ts = 1..12. A slow reader opens a snapshot at ts = 3 after commits 1..8 exist, reads all keys,
waits while 9..12 commit, and re-reads: both reads must equal the brute-force oracle (a linear scan
of the raw commit history for the last record with ts <= snapshot), while fresh snapshots keep
seeing the newest writers - writers are never blocked or hidden.
"""


class VersionedStore:
    def __init__(self, keys):
        self.versions = {key: [] for key in keys}
        self.commit_count = 0

    def commit(self, ts, writes):
        """Commit one transaction: writes is a list of (key, value) sharing timestamp ts."""
        for key, value in writes:
            self.versions[key].append((ts, value))
        self.commit_count += 1

    def get(self, key, snapshot_ts):
        entries = self.versions[key]
        lo, hi = 0, len(entries) - 1
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if entries[mid][0] <= snapshot_ts:
                best = entries[mid][1]
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def snapshot(self, ts):
        return Snapshot(self, ts)

    def head_ts(self):
        return max((ts for entries in self.versions.values() for ts, _ in entries), default=0)


class Snapshot:
    def __init__(self, store, ts):
        self.store = store
        self.ts = ts

    def get(self, key):
        return self.store.get(key, self.ts)


def oracle_value(history, key, snapshot_ts):
    """Brute force: last committed value with commit_ts <= snapshot_ts, scanning raw history."""
    value = None
    for ts, hkey, hvalue in history:
        if hkey == key and ts <= snapshot_ts:
            value = hvalue
    return value


def build_history(base, transactions, seed):
    """Flat record list [(ts, key, value)]; equal ts values form one transaction."""
    history = [(0, key, value) for key, value in sorted(base.items())]
    for t in range(1, transactions + 1):
        seed = (seed * 75 + 74) % 65537
        count = 1 + t % 2
        for _ in range(count):
            seed = (seed * 75 + 74) % 65537
            key = "k%d" % (seed % 5)
            seed = (seed * 75 + 74) % 65537
            value = 100 + seed % 900
            history.append((t, key, value))
    return history


def group_transactions(history):
    """[(ts, [(key, value), ...])] in commit order."""
    grouped = []
    for ts, key, value in history:
        if not grouped or grouped[-1][0] != ts:
            grouped.append((ts, []))
        grouped[-1][1].append((key, value))
    return grouped


def main():
    base = {"k%d" % i: 10 * i + 1 for i in range(5)}
    history = build_history(base, 12, 483)
    assert len(history) == 5 + 18, len(history)
    transactions = group_transactions(history)
    assert len(transactions) == 13 and transactions[0][0] == 0

    store = VersionedStore(base.keys())
    for ts, writes in transactions[:9]:  # base + commits 1..8
        store.commit(ts, writes)
    assert store.commit_count == 9

    # Long-range reader: opens at ts = 3 after commits 1..8 exist.
    reader = store.snapshot(3)
    first = {key: reader.get(key) for key in base}

    # Writers proceed while the slow reader is still open: commits 9..12 land now.
    for ts, writes in transactions[9:]:
        store.commit(ts, writes)
    second = {key: reader.get(key) for key in base}
    assert first == second
    for key in base:
        assert first[key] == oracle_value(history, key, 3), key
    assert all(v is not None for v in first.values())

    head = store.head_ts()
    assert head == 12
    fresh = store.snapshot(head)
    head_values = {key: fresh.get(key) for key in base}
    for key in base:
        assert head_values[key] == oracle_value(history, key, 12), key
    # Writers really did move the state past the reader's snapshot.
    assert any(first[key] != head_values[key] for key in base)

    # Snapshot isolation holds for every (key, ts) pair, including ts before any version (-1).
    for key in base:
        for ts in range(-1, 14):
            assert store.get(key, ts) == oracle_value(history, key, ts), (key, ts)

    # Read-your-writes: a snapshot exactly at a transaction's ts sees that whole transaction;
    # a snapshot one ts earlier sees the previous state for those same keys.
    for ts, writes in transactions[1:]:
        current = store.snapshot(ts)
        previous = store.snapshot(ts - 1)
        for key, value in writes:
            assert current.get(key) == value, (ts, key)
            assert previous.get(key) == oracle_value(history, key, ts - 1), (ts, key)

    # No torn transactions: each transaction writes 1..2 distinct keys at one shared ts, so a
    # snapshot either sees all of its writes or none of them (checked above per key).
    for ts, writes in transactions[1:]:
        keys_written = {key for key, _ in writes}
        assert len(keys_written) in (1, 2)

    versions = {key: len(entries) for key, entries in store.versions.items()}
    assert versions == {"k0": 3, "k1": 6, "k2": 4, "k3": 3, "k4": 7}, versions
    assert sum(versions.values()) == 5 + 18, versions
    assert min(versions.values()) == 3, versions
    assert max(versions.values()) == 7, versions

    # Writer progress is unabated: commit count grows monotonically as new transactions arrive.
    count_before = store.commit_count
    store.commit(13, [("k0", 777)])
    store.commit(14, [("k1", 888)])
    assert store.commit_count == count_before + 2
    assert store.get("k0", 13) == 777 and store.get("k0", 12) != 777
    assert store.get("k1", 14) == 888 and store.get("k1", 13) != 888
    print("PASS")


if __name__ == "__main__":
    main()
