/** Task 483 - Implement consistent snapshot retrieval using MVCC. (Tier: Advanced, CTCI Ch 14)
 *
 * Multi-version store: every key holds the full list of (commit_ts, value) versions produced by
 * whole-transaction commits (all keys of one transaction share a single commit timestamp). A read
 * issued at snapshot timestamp s returns, per key, the newest version with commit_ts <= s, i.e. the
 * state as of a single point in time even if newer commits exist. Because one transaction stamps
 * all of its writes with the same ts, a snapshot never observes a torn transaction.
 *
 * The scripted history is seeded with base versions at ts = 0, then 12 LCG-driven transactions at
 * ts = 1..12. A slow reader opens a snapshot at ts = 3 after commits 1..8 exist, reads all keys,
 * waits while 9..12 commit, and re-reads: both reads must equal the brute-force oracle (a linear
 * scan of raw history), while fresh snapshots keep seeing the newest writers.
 */
import java.util.*;

class VersionEntry {
    final int ts;
    final int value;

    VersionEntry(int ts, int value) {
        this.ts = ts;
        this.value = value;
    }
}

class HistoryRecord {
    final int ts;
    final String key;
    final int value;

    HistoryRecord(int ts, String key, int value) {
        this.ts = ts;
        this.key = key;
        this.value = value;
    }
}

class VersionedStore {
    final Map<String, List<VersionEntry>> versions = new LinkedHashMap<>();
    int commitCount;

    VersionedStore(List<String> keys) {
        for (String key : keys) versions.put(key, new ArrayList<>());
    }

    void commit(int ts, List<Object[]> writes) {
        for (Object[] write : writes) {
            versions.get((String) write[0]).add(new VersionEntry(ts, (Integer) write[1]));
        }
        commitCount++;
    }

    Integer get(String key, int snapshotTs) {
        List<VersionEntry> entries = versions.get(key);
        int lo = 0, hi = entries.size() - 1;
        Integer best = null;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (entries.get(mid).ts <= snapshotTs) {
                best = entries.get(mid).value;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return best;
    }

    Snapshot snapshot(int ts) {
        return new Snapshot(this, ts);
    }

    int headTs() {
        int best = 0;
        for (List<VersionEntry> entries : versions.values()) {
            for (VersionEntry entry : entries) best = Math.max(best, entry.ts);
        }
        return best;
    }
}

class Snapshot {
    final VersionedStore store;
    final int ts;

    Snapshot(VersionedStore store, int ts) {
        this.store = store;
        this.ts = ts;
    }

    Integer get(String key) {
        return store.get(key, ts);
    }
}

public class Solution {
    private static int seed = 483;

    private static Integer oracleValue(List<HistoryRecord> history, String key, int snapshotTs) {
        Integer value = null;
        for (HistoryRecord record : history) {
            if (record.key.equals(key) && record.ts <= snapshotTs) value = record.value;
        }
        return value;
    }

    private static List<HistoryRecord> buildHistory(Map<String, Integer> base, int transactions) {
        List<HistoryRecord> history = new ArrayList<>();
        List<String> sortedBase = new ArrayList<>(base.keySet());
        Collections.sort(sortedBase);
        for (String key : sortedBase) history.add(new HistoryRecord(0, key, base.get(key)));
        for (int t = 1; t <= transactions; t++) {
            seed = (seed * 75 + 74) % 65537;
            int count = 1 + t % 2;
            for (int w = 0; w < count; w++) {
                seed = (seed * 75 + 74) % 65537;
                String key = "k" + (seed % 5);
                seed = (seed * 75 + 74) % 65537;
                int value = 100 + seed % 900;
                history.add(new HistoryRecord(t, key, value));
            }
        }
        return history;
    }

    private static List<Object[]> groupTransactions(List<HistoryRecord> history) {
        List<Object[]> grouped = new ArrayList<>();
        for (HistoryRecord record : history) {
            if (grouped.isEmpty() || (Integer) grouped.get(grouped.size() - 1)[0] != record.ts) {
                grouped.add(new Object[]{record.ts, new ArrayList<Object[]>()});
            }
            @SuppressWarnings("unchecked")
            List<Object[]> writes = (List<Object[]>) grouped.get(grouped.size() - 1)[1];
            writes.add(new Object[]{record.key, record.value});
        }
        return grouped;
    }

    public static void main(String[] args) {
        Map<String, Integer> base = new LinkedHashMap<>();
        for (int i = 0; i < 5; i++) base.put("k" + i, 10 * i + 1);
        seed = 483;
        List<HistoryRecord> history = buildHistory(base, 12);
        check(history.size() == 5 + 18, "history size");
        List<Object[]> transactions = groupTransactions(history);
        check(transactions.size() == 13, "transaction count");
        check((Integer) transactions.get(0)[0] == 0, "base ts");

        VersionedStore store = new VersionedStore(new ArrayList<>(base.keySet()));
        for (int i = 0; i < 9; i++) {
            @SuppressWarnings("unchecked")
            List<Object[]> writes = (List<Object[]>) transactions.get(i)[1];
            store.commit((Integer) transactions.get(i)[0], writes);
        }
        check(store.commitCount == 9, "commits so far");

        Snapshot reader = store.snapshot(3);
        Map<String, Integer> first = new LinkedHashMap<>();
        for (String key : base.keySet()) first.put(key, reader.get(key));

        for (int i = 9; i < transactions.size(); i++) {
            @SuppressWarnings("unchecked")
            List<Object[]> writes = (List<Object[]>) transactions.get(i)[1];
            store.commit((Integer) transactions.get(i)[0], writes);
        }
        for (String key : base.keySet()) {
            check(Objects.equals(reader.get(key), first.get(key)), "repeatable read " + key);
            check(Objects.equals(first.get(key), oracleValue(history, key, 3)), "snapshot oracle " + key);
            check(first.get(key) != null, "visible key " + key);
        }

        check(store.headTs() == 12, "head ts");
        Snapshot fresh = store.snapshot(store.headTs());
        boolean moved = false;
        for (String key : base.keySet()) {
            Integer headValue = fresh.get(key);
            check(Objects.equals(headValue, oracleValue(history, key, 12)), "head oracle " + key);
            if (!Objects.equals(first.get(key), headValue)) moved = true;
        }
        check(moved, "writers moved state");

        for (String key : base.keySet()) {
            for (int ts = -1; ts <= 13; ts++) {
                check(Objects.equals(store.get(key, ts), oracleValue(history, key, ts)), "isolation " + key + " " + ts);
            }
        }

        for (int i = 1; i < transactions.size(); i++) {
            int ts = (Integer) transactions.get(i)[0];
            @SuppressWarnings("unchecked")
            List<Object[]> writes = (List<Object[]>) transactions.get(i)[1];
            Snapshot current = store.snapshot(ts);
            Snapshot previous = store.snapshot(ts - 1);
            Set<String> distinct = new LinkedHashSet<>();
            for (Object[] write : writes) {
                String key = (String) write[0];
                int value = (Integer) write[1];
                distinct.add(key);
                check(Objects.equals(current.get(key), value), "read your writes " + ts + " " + key);
                check(Objects.equals(previous.get(key), oracleValue(history, key, ts - 1)), "prior state " + ts + " " + key);
            }
            check(distinct.size() <= 2, "atomic key count " + ts);
        }

        Map<String, Integer> versions = new TreeMap<>();
        for (Map.Entry<String, List<VersionEntry>> entry : store.versions.entrySet()) {
            versions.put(entry.getKey(), entry.getValue().size());
        }
        check(versions.get("k0") == 3 && versions.get("k1") == 6 && versions.get("k2") == 4
                && versions.get("k3") == 3 && versions.get("k4") == 7, "version counts " + versions);
        int total = 0;
        for (int count : versions.values()) total += count;
        check(total == 5 + 18, "version total");

        int countBefore = store.commitCount;
        List<Object[]> k0Write = Collections.singletonList(new Object[]{"k0", 777});
        store.commit(13, k0Write);
        List<Object[]> k1Write = Collections.singletonList(new Object[]{"k1", 888});
        store.commit(14, k1Write);
        check(store.commitCount == countBefore + 2, "writer progress");
        check(store.get("k0", 13) == 777 && store.get("k0", 12) != 777, "k0 commit");
        check(store.get("k1", 14) == 888 && store.get("k1", 13) != 888, "k1 commit");
        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
