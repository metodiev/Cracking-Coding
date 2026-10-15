/** Task 478 - Implement log-structured merge tree (LSM) basic flush/merge simulation. (Tier: Advanced, CTCI Ch 14 — Databases) */
import java.util.*;

class LsmEntry {
    final int seq;
    final Integer value; // null is a tombstone

    LsmEntry(int seq, Integer value) {
        this.seq = seq;
        this.value = value;
    }
}

class Sstable {
    final int id;
    final Map<Integer, LsmEntry> entries;

    Sstable(int id, Map<Integer, LsmEntry> entries) {
        this.id = id;
        this.entries = entries;
    }
}

class LSMStore {
    static final int MERGE_FANOUT = 4;

    final Map<Integer, LsmEntry> memtable = new LinkedHashMap<>();
    List<Sstable> sstables = new ArrayList<>();
    int seq = 0;
    int opsSinceFlush = 0;
    final int flushAfter;
    int flushCount = 0;
    int mergeCount = 0;
    int nextTableId = 0;

    LSMStore(int flushAfter) {
        this.flushAfter = flushAfter;
    }

    LSMStore() {
        this(4);
    }

    void write(int key, Integer value) {
        seq++;
        memtable.put(key, new LsmEntry(seq, value));
        opsSinceFlush++;
        if (opsSinceFlush >= flushAfter) flush();
    }

    void delete(int key) {
        write(key, null);
    }

    static Sstable mergeTables(List<Sstable> tables) {
        Map<Integer, LsmEntry> merged = new LinkedHashMap<>();
        for (Sstable table : tables) {
            for (Map.Entry<Integer, LsmEntry> e : table.entries.entrySet()) {
                LsmEntry current = merged.get(e.getKey());
                if (current == null || e.getValue().seq > current.seq) merged.put(e.getKey(), e.getValue());
            }
        }
        return new Sstable(-1, merged);
    }

    void flush() {
        if (memtable.isEmpty()) return;
        nextTableId++;
        sstables.add(new Sstable(nextTableId, new LinkedHashMap<>(memtable)));
        memtable.clear();
        opsSinceFlush = 0;
        flushCount++;
        while (sstables.size() >= MERGE_FANOUT) {
            Sstable merged = mergeTables(sstables.subList(0, MERGE_FANOUT));
            List<Sstable> rest = new ArrayList<>(sstables.subList(MERGE_FANOUT, sstables.size()));
            sstables = new ArrayList<>();
            sstables.add(merged);
            sstables.addAll(rest);
            mergeCount++;
        }
    }

    Integer entrySeq(int key) {
        LsmEntry best = memtable.get(key);
        for (Sstable table : sstables) {
            LsmEntry entry = table.entries.get(key);
            if (entry != null && (best == null || entry.seq > best.seq)) best = entry;
        }
        return best == null ? -1 : best.seq;
    }

    Integer get(int key) {
        LsmEntry best = memtable.get(key);
        for (Sstable table : sstables) {
            LsmEntry entry = table.entries.get(key);
            if (entry != null && (best == null || entry.seq > best.seq)) best = entry;
        }
        return best == null ? null : best.value;
    }

    List<int[]> scan() {
        Map<Integer, LsmEntry> merged = new LinkedHashMap<>();
        for (Sstable table : sstables) {
            for (Map.Entry<Integer, LsmEntry> e : table.entries.entrySet()) {
                LsmEntry current = merged.get(e.getKey());
                if (current == null || e.getValue().seq > current.seq) merged.put(e.getKey(), e.getValue());
            }
        }
        for (Map.Entry<Integer, LsmEntry> e : memtable.entrySet()) {
            LsmEntry current = merged.get(e.getKey());
            if (current == null || e.getValue().seq > current.seq) merged.put(e.getKey(), e.getValue());
        }
        List<Integer> keys = new ArrayList<>(merged.keySet());
        Collections.sort(keys);
        List<int[]> out = new ArrayList<>();
        for (int key : keys) {
            Integer value = merged.get(key).value;
            if (value != null) out.add(new int[]{key, value});
        }
        return out;
    }
}

public class Solution {
    static final int KEYS = 16;

    static int lcg(int seed) {
        return (seed * 75 + 74) % 65537;
    }

    static int[][] genOps(int count, int seed) {
        int[][] ops = new int[count][];
        for (int i = 0; i < count; i++) {
            seed = lcg(seed);
            int kind = seed % 5;
            seed = lcg(seed);
            int key = seed % KEYS;
            seed = lcg(seed);
            int value = seed % 100;
            ops[i] = new int[]{kind, key, value};
        }
        return ops;
    }

    static boolean sameScan(List<int[]> a, List<int[]> b) {
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); i++) {
            if (a.get(i)[0] != b.get(i)[0] || a.get(i)[1] != b.get(i)[1]) return false;
        }
        return true;
    }

    static List<int[]> expectedScan(Map<Integer, Integer> oracle) {
        List<Integer> keys = new ArrayList<>(oracle.keySet());
        Collections.sort(keys);
        List<int[]> out = new ArrayList<>();
        for (int key : keys) {
            if (oracle.get(key) != null) out.add(new int[]{key, oracle.get(key)});
        }
        return out;
    }

    static void runWorkload() {
        LSMStore store = new LSMStore();
        Map<Integer, Integer> oracle = new LinkedHashMap<>();
        int[][] ops = genOps(80, 31337);
        for (int i = 0; i < ops.length; i++) {
            int kind = ops[i][0], key = ops[i][1], value = ops[i][2];
            if (kind == 4) {
                store.delete(key);
                oracle.put(key, null);
            } else {
                store.write(key, value);
                oracle.put(key, value);
            }
            for (int k = 0; k < KEYS; k++) {
                checkObjects(store.get(k), oracle.get(k), "get " + i + " key " + k);
            }
            check(sameScan(store.scan(), expectedScan(oracle)), "scan " + i);
        }
        check(store.flushCount == 20, "one flush per four ops");
        check(store.mergeCount == 6, "merge at flushes 4,7,10,13,16,19");
        check(store.sstables.size() == 2, "one merged table plus the last flush");
        check(store.memtable.isEmpty(), "memtable drained by the last flush");
    }

    static void checkObjects(Integer got, Integer expected, String label) {
        check(java.util.Objects.equals(got, expected), label + ": " + got + " vs " + expected);
    }

    public static void main(String[] args) {
        // Flush policy: the first table appears exactly on the 4th operation.
        LSMStore s = new LSMStore();
        s.write(0, 10);
        s.write(1, 11);
        s.write(2, 12);
        check(s.sstables.isEmpty() && s.get(0) == 10, "no flush yet");
        s.write(3, 13);
        check(s.sstables.size() == 1 && s.memtable.isEmpty() && s.flushCount == 1, "flush on 4th op");
        check(sameScan(s.scan(), Arrays.asList(new int[]{0, 10}, new int[]{1, 11}, new int[]{2, 12}, new int[]{3, 13})), "scan after flush");
        s.delete(1);
        check(s.get(1) == null, "deleted");
        s.write(1, 99);
        check(s.get(1) == 99, "rewritten");
        check(sameScan(s.scan(), Arrays.asList(new int[]{0, 10}, new int[]{1, 99}, new int[]{2, 12}, new int[]{3, 13})), "scan after rewrite");

        // Tombstones survive flushing and merging: a deleted key stays deleted.
        LSMStore s2 = new LSMStore(2);
        s2.write(5, 1);
        s2.write(5, 2);
        s2.delete(5);
        s2.write(6, 3);
        check(s2.get(5) == null && s2.get(6) == 3, "tombstone visible");
        List<Sstable> mergedList = new ArrayList<>();
        mergedList.add(LSMStore.mergeTables(s2.sstables));
        s2.sstables = mergedList;
        check(s2.get(5) == null, "merge must not resurrect the tombstone");
        check(sameScan(s2.scan(), Arrays.asList(new int[]{6, 3})), "scan without the tombstone");
        s2.write(5, 7);
        s2.write(8, 8);
        check(s2.get(5) == 7 && s2.get(8) == 8, "reinserted key");
        check(sameScan(s2.scan(), Arrays.asList(new int[]{5, 7}, new int[]{6, 3}, new int[]{8, 8})), "final scan");

        runWorkload();
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
