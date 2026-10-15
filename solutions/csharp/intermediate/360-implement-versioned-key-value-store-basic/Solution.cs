// Task 360 - Implement versioned key value store basic. (Tier: Intermediate, CTCI Ch 14)
using System;
using System.Collections.Generic;

public static class Solution
{
    private const int ScriptOps = 200;
    private const int Keyspace = 8;

    /// <summary>
    /// Versioned key-value store: every put or delete consumes exactly one monotonically
    /// increasing version (1, 2, 3, ...). Per key a history of [version, value-or-TOMBSTONE]
    /// entries is kept, so Get(key, v) time-travels: it returns the value of the greatest
    /// version &lt;= v, or missing (null) when that entry is a tombstone. Tombstone policy
    /// (documented): delete never erases history, it appends a TOMBSTONE; Get(key) on a
    /// deleted key reports missing while Get(key, v) for v before the delete still returns
    /// the old value; a later put resurrects the key at a fresh version. An explicit version
    /// must lie in 1..current (ArgumentOutOfRangeException otherwise); the automatic latest
    /// read is allowed on an empty store. Reads never consume a version, so they are
    /// idempotent. A random LCG operation script is cross-checked against a naive
    /// per-version snapshot list (snapshot[v-1] is the whole map state after the v-th write).
    /// </summary>
    private class VersionedKV
    {
        private static readonly object Tombstone = new object();
        private readonly Dictionary<string, List<object[]>> history = new Dictionary<string, List<object[]>>();
        private int version;

        public int LatestVersion() { return version; }

        public int Put(string key, int value)
        {
            version++;
            Entries(key).Add(new object[] { version, value });
            return version;
        }

        public int Delete(string key)
        {
            version++;
            Entries(key).Add(new object[] { version, Tombstone });
            return version;
        }

        /// <summary>Latest value; null means missing (never written or tombstoned).</summary>
        public int? Get(string key) { return GetInternal(key, version, false); }

        /// <summary>Value as of an explicit version; ArgumentOutOfRangeException outside 1..current.</summary>
        public int? Get(string key, int atVersion) { return GetInternal(key, atVersion, true); }

        private int? GetInternal(string key, int atVersion, bool isExplicit)
        {
            if (isExplicit && (atVersion < 1 || atVersion > version))
            {
                throw new ArgumentOutOfRangeException(nameof(atVersion),
                    "version must be in 1.." + version + ", got " + atVersion);
            }
            if (atVersion < 1) return null;
            object? latestEntry = null;
            bool found = false;
            if (history.TryGetValue(key, out var entries))
            {
                foreach (var entry in entries)
                {
                    if ((int)entry[0] <= atVersion)
                    {
                        latestEntry = entry[1];
                        found = true;
                    }
                    else
                    {
                        break;
                    }
                }
            }
            if (!found || ReferenceEquals(latestEntry, Tombstone)) return null;
            return (int)latestEntry!;
        }

        private List<object[]> Entries(string key)
        {
            if (!history.TryGetValue(key, out var entries))
            {
                entries = new List<object[]>();
                history[key] = entries;
            }
            return entries;
        }
    }

    public static void Main()
    {
        // Canonical script from the task brief.
        var store = new VersionedKV();
        Check(store.Get("a") == null, "empty store reports missing");
        int v1 = store.Put("a", 1);
        int v2 = store.Put("a", 2);
        Check(v1 == 1 && v2 == 2, "versions increase by exactly one");
        Check(store.Get("a", v1) == 1, "Get(a, v1) == 1");
        Check(store.Get("a", v2) == 2 && store.Get("a") == 2, "latest value");
        int v3 = store.Delete("a");
        Check(v3 == 3, "delete consumes a version");
        Check(store.Get("a") == null, "Get(a, latest) == missing");
        Check(store.Get("a", 2) == 2, "old value visible before the tombstone");
        Check(store.Get("a", 3) == null, "tombstone visible at its own version");
        int v4 = store.Put("a", 5);
        Check(v4 == 4, "put resurrects with a fresh version");
        Check(store.Get("a") == 5, "resurrected value");
        Check(store.Get("a", 3) == null && store.Get("a", 1) == 1, "history unchanged by resurrection");
        Check(store.Get("missing") == null && store.Get("missing", 2) == null, "missing keys");

        // Idempotent reads: reads never consume a version and repeat identically.
        int before = store.LatestVersion();
        for (int i = 0; i < 100; i++)
        {
            Check(store.Get("a") == 5, "idempotent latest read");
            Check(store.Get("a", 1) == 1, "idempotent historical read");
            Check(store.Get("a", 3) == null, "idempotent tombstone read");
        }
        Check(store.LatestVersion() == before && before == 4, "no version consumed by reads");

        // Versions outside 1..current are invalid for explicit reads.
        foreach (int bad in new[] { 0, -1, 5 })
        {
            bool raised = false;
            try
            {
                store.Get("a", bad);
            }
            catch (ArgumentOutOfRangeException)
            {
                raised = true;
            }
            Check(raised, "invalid version must raise: " + bad);
        }

        // Random LCG operation script cross-checked against a naive snapshot list.
        store = new VersionedKV();
        var snapshots = new List<Dictionary<string, int>>();   // snapshots[v-1] == state after write v
        var reference = new Dictionary<string, int>();
        var allKeys = new SortedSet<string>();
        long seed = 20260;
        for (int step = 0; step < ScriptOps; step++)
        {
            seed = (seed * 75 + 74) % 65537;
            string key = "k" + (seed % Keyspace);
            seed = (seed * 75 + 74) % 65537;
            int version;
            if (seed % 4 == 0)
            {
                version = store.Delete(key);
                reference.Remove(key);
            }
            else
            {
                version = store.Put(key, (int)(seed % 100));
                reference[key] = (int)(seed % 100);
            }
            Check(version == step + 1, "one version per write");
            snapshots.Add(new Dictionary<string, int>(reference));
            allKeys.Add(key);
        }

        int current = store.LatestVersion();
        Check(current == ScriptOps, "all writes counted");
        var keysToCheck = new List<string>(allKeys) { "absent" };
        foreach (string key in keysToCheck)
        {
            Check(store.Get(key) == (reference.TryGetValue(key, out int expected) ? expected : (int?)null),
                  "latest view of " + key);
            for (int version = 1; version <= current; version++)
            {
                var snapshot = snapshots[version - 1];
                int? expectedAtVersion = snapshot.TryGetValue(key, out int value) ? value : (int?)null;
                Check(store.Get(key, version) == expectedAtVersion,
                      "time travel of " + key + " at version " + version);
            }
        }
        Check(store.LatestVersion() == current, "the mass of reads consumed no version");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
