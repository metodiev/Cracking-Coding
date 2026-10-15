/** Task 484 - Implement simple key-value store with memtable + SSTable. (Tier: Advanced, CTCI Ch 14)
 *
 * LSM-style prototype with documented semantics:
 * - Writes land in an in-memory memtable (key -> value; deletes insert a null TOMBSTONE marker).
 * - When the memtable reaches `flush_threshold` entries it flushes to a new immutable SSTable:
 *   the records are written sorted by key and appended to the sstable list (oldest first).
 * - get(key) probes the memtable, then the SSTables from newest to oldest; the first record for the
 *   key wins. A tombstone means "deleted" (returns null) and must shadow older values in older
 *   SSTables - that is what makes deletes correct without rewriting data.
 * - compact() flushes the memtable, merges all SSTables newest-wins into a single one, and drops
 *   tombstones plus every record they shadow, so the result holds exactly the live keys.
 *
 * The test asserts get == oracle after every single operation of an LCG-driven 240-op script
 * (puts, gets, deletes) including tombstone shadowing across SSTables, then checks the
 * post-compaction layout and idempotence of a second compaction.
 */
import java.util.*;

class KVStore {
    private final int flushThreshold;
    final Map<String, Integer> memtable = new LinkedHashMap<>();
    final List<List<String[]>> sstables = new ArrayList<>();
    int flushes;
    int compactions;

    KVStore(int flushThreshold) {
        this.flushThreshold = flushThreshold;
    }

    void put(String key, Integer value) {
        memtable.put(key, value);
        maybeFlush();
    }

    void delete(String key) {
        memtable.put(key, null);
        maybeFlush();
    }

    private void maybeFlush() {
        if (memtable.size() >= flushThreshold) flush();
    }

    void flush() {
        if (memtable.isEmpty()) return;
        List<String[]> records = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : memtable.entrySet()) {
            records.add(new String[]{entry.getKey(), entry.getValue() == null ? null : String.valueOf(entry.getValue())});
        }
        records.sort(Comparator.comparing(r -> r[0]));
        sstables.add(records);
        memtable.clear();
        flushes++;
    }

    Integer get(String key) {
        if (memtable.containsKey(key)) return memtable.get(key);
        for (int i = sstables.size() - 1; i >= 0; i--) {
            for (String[] record : sstables.get(i)) {
                if (record[0].equals(key)) return record[1] == null ? null : Integer.valueOf(record[1]);
            }
        }
        return null;
    }

    void compact() {
        flush();
        Map<String, Integer> newest = new LinkedHashMap<>();
        for (int i = sstables.size() - 1; i >= 0; i--) {
            for (String[] record : sstables.get(i)) {
                if (!newest.containsKey(record[0])) {
                    newest.put(record[0], record[1] == null ? null : Integer.valueOf(record[1]));
                }
            }
        }
        List<String[]> live = new ArrayList<>();
        List<String> keys = new ArrayList<>(newest.keySet());
        Collections.sort(keys);
        for (String key : keys) {
            Integer value = newest.get(key);
            if (value != null) live.add(new String[]{key, String.valueOf(value)});
        }
        sstables.clear();
        if (!live.isEmpty()) sstables.add(live);
        compactions++;
    }
}

class Oracle {
    final Map<String, Integer> live = new LinkedHashMap<>();

    Oracle(List<String> keys) {
        for (String key : keys) live.put(key, null);
    }

    void put(String key, Integer value) {
        live.put(key, value);
    }

    void delete(String key) {
        live.remove(key);
    }

    Integer get(String key) {
        return live.get(key);
    }
}

public class Solution {
    private static int seed = 484;

    private static int[][] lcgOps(int count, int keyCount) {
        int[][] ops = new int[count][];
        for (int i = 0; i < count; i++) {
            seed = (seed * 75 + 74) % 65537;
            int kind = seed % 3;
            seed = (seed * 75 + 74) % 65537;
            int key = seed % keyCount;
            seed = (seed * 75 + 74) % 65537;
            int value = seed % 1000;
            ops[i] = new int[]{kind, key, value};
        }
        return ops;
    }

    private static String recordValue(List<List<String[]>> sstables, String key) {
        for (List<String[]> records : sstables) {
            for (String[] record : records) {
                if (record[0].equals(key)) return record[1];
            }
        }
        return "missing";
    }

    public static void main(String[] args) {
        KVStore store = new KVStore(2);
        store.put("a", 1);
        store.put("b", 2);
        check(store.flushes == 1 && store.memtable.isEmpty(), "first flush");
        check(store.get("a") == 1 && store.get("b") == 2, "values after flush");
        store.put("a", 5);
        store.flush();
        store.delete("b");
        check(store.get("b") == null, "tombstone visible immediately");
        store.flush();
        check(store.get("a") == 5 && store.get("b") == null, "read through tombstones");
        check(store.sstables.size() == 3, "three sstables");
        check(recordValue(Collections.singletonList(store.sstables.get(2)), "b") == null, "newest is tombstone");
        check("2".equals(recordValue(Collections.singletonList(store.sstables.get(0)), "b")), "oldest keeps value");
        store.compact();
        check(store.compactions == 1 && store.sstables.size() == 1, "compacted");
        check(store.sstables.get(0).size() == 1, "only live record");
        check(store.sstables.get(0).get(0)[0].equals("a") && store.sstables.get(0).get(0)[1].equals("5"), "live record");
        check(store.get("a") == 5 && store.get("b") == null, "post compact reads");
        store.compact();
        check(store.sstables.size() == 1 && store.sstables.get(0).get(0)[1].equals("5"), "idempotent compact");

        List<String> keys = new ArrayList<>();
        for (int i = 0; i < 12; i++) keys.add("k" + i);
        store = new KVStore(4);
        Oracle oracle = new Oracle(keys);
        int gets = 0, tombstones = 0;
        int[][] ops = lcgOps(240, 12);
        for (int step = 0; step < ops.length; step++) {
            int kind = ops[step][0];
            String key = "k" + ops[step][1];
            int value = ops[step][2];
            if (kind == 0) {
                store.put(key, value);
                oracle.put(key, value);
            } else if (kind == 1) {
                check(Objects.equals(store.get(key), oracle.get(key)), "direct get " + step);
                gets++;
            } else {
                store.delete(key);
                oracle.delete(key);
                tombstones++;
            }
            for (String probe : keys) {
                check(Objects.equals(store.get(probe), oracle.get(probe)), "probe " + step + " " + probe);
            }
            if (step % 60 == 59) {
                store.compact();
                for (String probe : keys) {
                    check(Objects.equals(store.get(probe), oracle.get(probe)), "post compact " + step + " " + probe);
                }
            }
        }
        check(gets == 84 && tombstones == 71, "op counts " + gets + " " + tombstones);
        check(store.flushes == 35, "flush count " + store.flushes);
        check(store.compactions == 4, "compaction count");
        store.compact();
        check(store.sstables.size() == 1, "final single sstable");
        List<String[]> records = store.sstables.get(0);
        List<String> liveKeys = new ArrayList<>();
        for (String key : oracle.live.keySet()) {
            if (oracle.live.get(key) != null) liveKeys.add(key);
        }
        Collections.sort(liveKeys);
        check(records.size() == liveKeys.size(), "record count " + records.size());
        for (int i = 0; i < records.size(); i++) {
            check(records.get(i)[0].equals(liveKeys.get(i)), "key order " + i);
            check(records.get(i)[1] != null && records.get(i)[1].equals(String.valueOf(oracle.live.get(liveKeys.get(i)))), "value " + i);
        }
        for (String probe : keys) {
            check(Objects.equals(store.get(probe), oracle.get(probe)), "final probe " + probe);
        }
        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
