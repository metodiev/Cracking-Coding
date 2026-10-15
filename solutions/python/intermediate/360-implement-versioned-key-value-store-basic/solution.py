"""Task 360 - Implement versioned key value store basic. (Tier: Intermediate, CTCI Ch 14)

Versioned key-value store: every put or delete consumes exactly one monotonically
increasing version (1, 2, 3, ...). Per key a history of (version, value-or-TOMBSTONE)
entries is kept, so get(key, v) time-travels: it returns the value of the greatest
version <= v, or missing (None) when that entry is a tombstone. Tombstone policy
(documented): delete never erases history, it appends a TOMBSTONE; get(key) on a deleted
key reports missing while get(key, v) for v before the delete still returns the old
value; a later put resurrects the key at a fresh version. An explicit version must lie in
1..current (ValueError otherwise); the automatic latest read is allowed on an empty store.
Reads never consume a version, so they are idempotent. The canonical script is asserted
directly, and a random LCG operation script is cross-checked against a naive
per-version snapshot list (snapshot[v-1] is the whole map state after the v-th write).
"""
SCRIPT_OPS = 200
KEYSPACE = 8
TOMBSTONE = object()     # distinguishes "deleted at this version" from any stored value


class VersionedKV:
    def __init__(self):
        self._version = 0
        self._history = {}

    def latest_version(self):
        return self._version

    def put(self, key, value):
        self._version += 1
        self._history.setdefault(key, []).append((self._version, value))
        return self._version

    def delete(self, key):
        """Appends a tombstone at a fresh version; history (time travel) is preserved."""
        self._version += 1
        self._history.setdefault(key, []).append((self._version, TOMBSTONE))
        return self._version

    def get(self, key, version=None):
        """Value of key as of `version`, else the latest; None means missing (incl. tombstoned)."""
        if version is None:
            version = self._version
        elif not 1 <= version <= self._version:
            raise ValueError("version must be in 1..{}, got {}".format(self._version, version))
        if version < 1:
            return None
        latest_entry = None
        found = False
        for entry_version, value in self._history.get(key, ()):
            if entry_version <= version:
                latest_entry = value
                found = True
            else:
                break
        if not found or latest_entry is TOMBSTONE:
            return None
        return latest_entry


def main():
    # Canonical script from the task brief.
    store = VersionedKV()
    assert store.get("a") is None, "empty store reports missing"
    v1 = store.put("a", 1)
    v2 = store.put("a", 2)
    assert (v1, v2) == (1, 2), "versions increase by exactly one"
    assert store.get("a", v1) == 1, "get(a, v1) == 1"
    assert store.get("a", v2) == 2 and store.get("a") == 2, "latest value"
    v3 = store.delete("a")
    assert v3 == 3, "delete consumes a version"
    assert store.get("a") is None, "get(a, latest) == missing"
    assert store.get("a", 2) == 2, "old value visible before the tombstone"
    assert store.get("a", 3) is None, "tombstone visible at its own version"
    v4 = store.put("a", 5)
    assert v4 == 4, "put resurrects with a fresh version"
    assert store.get("a") == 5, "resurrected value"
    assert store.get("a", 3) is None and store.get("a", 1) == 1, "history unchanged by resurrection"
    assert store.get("missing") is None and store.get("missing", 2) is None, "missing keys"

    # Idempotent reads: reads never consume a version and repeat identically.
    before = store.latest_version()
    for _ in range(100):
        assert store.get("a") == 5
        assert store.get("a", 1) == 1
        assert store.get("a", 3) is None
    assert store.latest_version() == before == 4, "no version consumed by reads"

    # Versions outside 1..current are invalid for explicit reads.
    for bad in (0, -1, 5):
        try:
            store.get("a", bad)
            raise AssertionError("invalid version must raise")
        except ValueError:
            pass

    # Random LCG operation script cross-checked against a naive snapshot list.
    store = VersionedKV()
    snapshots = []                     # snapshots[v-1] == whole map state after write v
    reference = {}
    all_keys = set()
    seed = 20260
    for step in range(SCRIPT_OPS):
        seed = (seed * 75 + 74) % 65537
        key = "k{}".format(seed % KEYSPACE)
        seed = (seed * 75 + 74) % 65537
        if seed % 4 == 0:
            version = store.delete(key)
            reference.pop(key, None)
        else:
            version = store.put(key, seed % 100)
            reference[key] = seed % 100
        assert version == step + 1, "one version per write"
        snapshots.append(dict(reference))
        all_keys.add(key)

    current = store.latest_version()
    assert current == SCRIPT_OPS, "all writes counted"
    for key in sorted(all_keys) + ["absent"]:
        assert store.get(key) == reference.get(key), "latest view of {}".format(key)
        for version in range(1, current + 1):
            assert store.get(key, version) == snapshots[version - 1].get(key), \
                "time travel of {} at version {}".format(key, version)
    assert store.latest_version() == current, "the mass of reads consumed no version"


if __name__ == "__main__":
    main()
    print("PASS")
