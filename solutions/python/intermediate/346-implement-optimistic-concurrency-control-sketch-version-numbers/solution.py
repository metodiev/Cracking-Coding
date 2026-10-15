"""Task 346 - Optimistic concurrency control sketch with version numbers. (Tier: Intermediate, CTCI Ch 15)

Versioned records: a write commits only if the version it was based on is still
current, otherwise the caller re-reads and retries.  This file is a
single-threaded, fully scripted simulation of the interleaving - deterministic in
every language - instead of real threads, because the point is exact conflict
timing: two clients both read version 1, A commits first, B's stale commit is
rejected, B re-reads and commits on top of A.  A second script (three clients, a
round-robin scheduler) shows retry-until-success and rules out lost updates.
Production OCC would compare-and-swap the version field atomically.
"""


class VersionedStore:
    def __init__(self):
        self._records = {}                  # key -> (value, version)

    def create(self, key, value):
        if key in self._records:
            raise ValueError("key already exists")
        self._records[key] = (value, 1)     # new records start at version 1

    def read(self, key):
        """Returns (value, version); KeyError for a missing key."""
        value, version = self._records[key]
        return value, version

    def commit(self, key, new_value, expected_version):
        """True when the write wins; False when the version moved on (conflict)."""
        if key not in self._records:
            return False
        current_version = self._records[key][1]
        if current_version != expected_version:
            return False                    # stale read - the caller must retry
        self._records[key] = (new_value, current_version + 1)
        return True


class Client:
    """A client with a read snapshot; attempt() commits or reports a conflict."""

    def __init__(self, name, store, key):
        self.name = name
        self.store = store
        self.key = key
        self.snapshot = None
        self.retries = 0

    def read(self):
        self.snapshot = self.store.read(self.key)

    def attempt(self, new_value):
        ok = self.store.commit(self.key, new_value, self.snapshot[1])
        if not ok:
            self.retries += 1
        return ok


def main():
    store = VersionedStore()
    store.create("counter", 10)

    # scripted interleaving: A and B both read version 1 before anyone writes
    a = Client("A", store, "counter")
    b = Client("B", store, "counter")
    a.read()
    b.read()
    assert a.snapshot == (10, 1) and b.snapshot == (10, 1)

    assert a.attempt(11) is True             # A commits first
    assert store.read("counter") == (11, 2)
    assert b.attempt(11) is False            # B's stale commit is rejected
    assert b.retries == 1
    assert store.read("counter") == (11, 2)  # nothing changed on the rejected write

    b.read()                                 # B re-reads on top of A and retries
    assert b.snapshot == (11, 2)
    assert b.attempt(12) is True
    assert a.retries == 0 and b.retries == 1
    assert store.read("counter") == (12, 3)  # final state correct: 10 + 1 + 1

    # a stale commit long after the fact is still rejected
    assert store.commit("counter", 99, 1) is False
    assert store.read("counter") == (12, 3)
    assert store.commit("missing", 1, 1) is False     # unknown key never commits

    try:
        store.create("counter", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on duplicate create")

    # three clients, deterministic round-robin scheduler, retry until success
    store.create("total", 0)
    clients = [Client(name, store, "total") for name in "XYZ"]
    for client in clients:
        client.read()                        # all read version 1 before anyone writes
    pending = list(clients)
    rounds = 0
    while pending:
        rounds += 1
        still_pending = []
        for client in pending:
            if not client.attempt(client.snapshot[0] + 1):
                client.read()                # conflict -> refresh and retry later
                still_pending.append(client)
        pending = still_pending
    assert rounds == 3                       # X, then Y, then Z
    assert [client.retries for client in clients] == [0, 1, 2]
    assert store.read("total") == (3, 4)     # no lost update: 0 + 3 increments


if __name__ == "__main__":
    main()
    print("PASS")
