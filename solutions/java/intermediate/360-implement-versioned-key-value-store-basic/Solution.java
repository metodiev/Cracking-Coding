/** Task 360 - Implement versioned key value store basic. (Tier: Intermediate, CTCI Ch 14) */
import java.util.*;

public class Solution {
    static final int SCRIPT_OPS = 200;
    static final int KEYSPACE = 8;

    /**
     * Versioned key-value store: every put or delete consumes exactly one monotonically
     * increasing version (1, 2, 3, ...). Per key a history of (version, value-or-TOMBSTONE)
     * entries is kept, so get(key, v) time-travels: it returns the value of the greatest
     * version <= v, or missing (null) when that entry is a tombstone. Tombstone policy
     * (documented): delete never erases history, it appends a TOMBSTONE; get(key) on a
     * deleted key reports missing while get(key, v) for v before the delete still returns
     * the old value; a later put resurrects the key at a fresh version. An explicit version
     * must lie in 1..current (IllegalArgumentException otherwise); the automatic latest
     * read is allowed on an empty store. Reads never consume a version, so they are
     * idempotent. A random LCG operation script is cross-checked against a naive
     * per-version snapshot list (snapshot[v-1] is the whole map state after the v-th write).
     */
    static class VersionedKV {
        private static final Object TOMBSTONE = new Object();
        private int version;
        private final Map<String, List<Object[]>> history = new HashMap<>();   // {version, value|TOMBSTONE}

        int latestVersion() { return version; }

        int put(String key, int value) {
            version++;
            history.computeIfAbsent(key, k -> new ArrayList<>()).add(new Object[]{version, value});
            return version;
        }

        int delete(String key) {
            version++;
            history.computeIfAbsent(key, k -> new ArrayList<>()).add(new Object[]{version, TOMBSTONE});
            return version;
        }

        /** Latest value; null means missing (never written or tombstoned). */
        Integer get(String key) { return getInternal(key, version, false); }

        /** Value as of an explicit version; IllegalArgumentException outside 1..current. */
        Integer get(String key, int atVersion) { return getInternal(key, atVersion, true); }

        private Integer getInternal(String key, int atVersion, boolean explicit) {
            if (explicit && (atVersion < 1 || atVersion > version)) {
                throw new IllegalArgumentException("version must be in 1.." + version + ", got " + atVersion);
            }
            if (atVersion < 1) return null;
            Object latestEntry = null;
            boolean found = false;
            for (Object[] entry : history.getOrDefault(key, Collections.emptyList())) {
                if ((Integer) entry[0] <= atVersion) {
                    latestEntry = entry[1];
                    found = true;
                } else {
                    break;
                }
            }
            if (!found || latestEntry == TOMBSTONE) return null;
            return (Integer) latestEntry;
        }
    }

    public static void main(String[] args) {
        // Canonical script from the task brief.
        VersionedKV store = new VersionedKV();
        check(store.get("a") == null, "empty store reports missing");
        int v1 = store.put("a", 1);
        int v2 = store.put("a", 2);
        check(v1 == 1 && v2 == 2, "versions increase by exactly one");
        check(store.get("a", v1) == 1, "get(a, v1) == 1");
        check(store.get("a", v2) == 2 && store.get("a") == 2, "latest value");
        int v3 = store.delete("a");
        check(v3 == 3, "delete consumes a version");
        check(store.get("a") == null, "get(a, latest) == missing");
        check(store.get("a", 2) == 2, "old value visible before the tombstone");
        check(store.get("a", 3) == null, "tombstone visible at its own version");
        int v4 = store.put("a", 5);
        check(v4 == 4, "put resurrects with a fresh version");
        check(store.get("a") == 5, "resurrected value");
        check(store.get("a", 3) == null && store.get("a", 1) == 1, "history unchanged by resurrection");
        check(store.get("missing") == null && store.get("missing", 2) == null, "missing keys");

        // Idempotent reads: reads never consume a version and repeat identically.
        int before = store.latestVersion();
        for (int i = 0; i < 100; i++) {
            check(store.get("a") == 5, "idempotent latest read");
            check(store.get("a", 1) == 1, "idempotent historical read");
            check(store.get("a", 3) == null, "idempotent tombstone read");
        }
        check(store.latestVersion() == before && before == 4, "no version consumed by reads");

        // Versions outside 1..current are invalid for explicit reads.
        int[] badVersions = {0, -1, 5};
        for (int bad : badVersions) {
            boolean raised = false;
            try {
                store.get("a", bad);
            } catch (IllegalArgumentException expected) {
                raised = true;
            }
            check(raised, "invalid version must raise: " + bad);
        }

        // Random LCG operation script cross-checked against a naive snapshot list.
        store = new VersionedKV();
        List<Map<String, Integer>> snapshots = new ArrayList<>();   // snapshots[v-1] == state after write v
        Map<String, Integer> reference = new HashMap<>();
        TreeSet<String> allKeys = new TreeSet<>();
        long seed = 20260;
        for (int step = 0; step < SCRIPT_OPS; step++) {
            seed = (seed * 75 + 74) % 65537;
            String key = "k" + (seed % KEYSPACE);
            seed = (seed * 75 + 74) % 65537;
            int version;
            if (seed % 4 == 0) {
                version = store.delete(key);
                reference.remove(key);
            } else {
                version = store.put(key, (int) (seed % 100));
                reference.put(key, (int) (seed % 100));
            }
            check(version == step + 1, "one version per write");
            snapshots.add(new HashMap<>(reference));
            allKeys.add(key);
        }

        int current = store.latestVersion();
        check(current == SCRIPT_OPS, "all writes counted");
        List<String> keysToCheck = new ArrayList<>(allKeys);
        keysToCheck.add("absent");
        for (String key : keysToCheck) {
            check(Objects.equals(store.get(key), reference.get(key)), "latest view of " + key);
            for (int version = 1; version <= current; version++) {
                check(Objects.equals(store.get(key, version), snapshots.get(version - 1).get(key)),
                      "time travel of " + key + " at version " + version);
            }
        }
        check(store.latestVersion() == current, "the mass of reads consumed no version");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
