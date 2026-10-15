/** Task 479 - Implement compaction strategies for LSM (levelled vs size-tiered). (Tier: Advanced, CTCI Ch 14 — Databases) */
import java.util.*;

class SizeTieredEngine {
    static final int MERGE_FANOUT = 4;
    static final int FACTOR = 2;

    List<Map<Integer, Integer>> runs = new ArrayList<>(); // chronological
    int totalWritten = 0;
    int mergeCount = 0;
    int peakRuns = 0;
    int peakEntries = 0;

    void flush(Map<Integer, Integer> memtable) {
        runs.add(new LinkedHashMap<>(memtable));
        totalWritten += memtable.size();
        compact();
        peakRuns = Math.max(peakRuns, runs.size());
        int entries = 0;
        for (Map<Integer, Integer> run : runs) entries += run.size();
        peakEntries = Math.max(peakEntries, entries);
    }

    void compact() {
        while (true) {
            List<int[]> sizes = new ArrayList<>(); // {size, index}
            for (int i = 0; i < runs.size(); i++) sizes.add(new int[]{runs.get(i).size(), i});
            sizes.sort((a, b) -> a[0] != b[0] ? Integer.compare(a[0], b[0]) : Integer.compare(a[1], b[1]));
            List<Integer> target = null;
            for (int i = 0; i + 3 < sizes.size(); i++) {
                if (sizes.get(i + 3)[0] <= FACTOR * sizes.get(i)[0]) {
                    target = new ArrayList<>();
                    for (int j = i; j < i + MERGE_FANOUT; j++) target.add(sizes.get(j)[1]);
                    Collections.sort(target);
                    break;
                }
            }
            if (target == null) return;
            Map<Integer, Integer> merged = new LinkedHashMap<>();
            for (int idx : target) merged.putAll(runs.get(idx)); // chronological: newer wins
            totalWritten += merged.size();
            mergeCount++;
            List<Map<Integer, Integer>> newRuns = new ArrayList<>();
            boolean inserted = false;
            for (int i = 0; i < runs.size(); i++) {
                if (target.contains(i)) {
                    if (!inserted) {
                        newRuns.add(merged);
                        inserted = true;
                    }
                } else {
                    newRuns.add(runs.get(i));
                }
            }
            runs = newRuns;
        }
    }

    Map<Integer, Integer> materialize() {
        Map<Integer, Integer> out = new LinkedHashMap<>();
        for (Map<Integer, Integer> run : runs) out.putAll(run);
        return out;
    }
}

class LevelledEngine {
    List<List<Map<Integer, Integer>>> levels = new ArrayList<>();
    int totalWritten = 0;
    int mergeCount = 0;
    int peakRuns = 0;
    int peakEntries = 0;

    void ensureLevel(int i) {
        while (levels.size() <= i) levels.add(new ArrayList<>());
    }

    void flush(Map<Integer, Integer> memtable) {
        ensureLevel(0);
        levels.get(0).add(new LinkedHashMap<>(memtable));
        totalWritten += memtable.size();
        compact();
        int runs = 0;
        int entries = 0;
        for (List<Map<Integer, Integer>> level : levels) {
            runs += level.size();
            for (Map<Integer, Integer> run : level) entries += run.size();
        }
        peakRuns = Math.max(peakRuns, runs);
        peakEntries = Math.max(peakEntries, entries);
    }

    void compact() {
        while (true) {
            boolean merged = false;
            for (int i = 0; i < levels.size(); i++) {
                List<Map<Integer, Integer>> level = levels.get(i);
                if (level.size() >= 2) {
                    Map<Integer, Integer> combined = new LinkedHashMap<>(level.get(0)); // older
                    combined.putAll(level.get(1));                                   // newer wins
                    totalWritten += combined.size();
                    mergeCount++;
                    level.remove(0);
                    level.remove(0);
                    ensureLevel(i + 1);
                    levels.get(i + 1).add(combined);
                    merged = true;
                    break;
                }
            }
            if (!merged) return;
        }
    }

    Map<Integer, Integer> materialize() {
        Map<Integer, Integer> out = new LinkedHashMap<>();
        for (int i = levels.size() - 1; i >= 0; i--) { // higher levels hold older data
            for (Map<Integer, Integer> run : levels.get(i)) out.putAll(run);
        }
        return out;
    }
}

public class Solution {
    static final int KEYS = 256;
    static final int OPS = 128;
    static final int FLUSH_EVERY = 8;

    static int lcg(int seed) {
        return (seed * 75 + 74) % 65537;
    }

    static int[][] genWrites(int count, int seed, int keyspace) {
        int[][] ops = new int[count][];
        for (int i = 0; i < count; i++) {
            seed = lcg(seed);
            int key = seed % keyspace;
            seed = lcg(seed);
            int value = seed % 1000;
            ops[i] = new int[]{key, value};
        }
        return ops;
    }

    public static void main(String[] args) {
        int[][] ops = genWrites(OPS, 24680, KEYS);
        Map<Integer, Integer> oracle = new LinkedHashMap<>();
        SizeTieredEngine st = new SizeTieredEngine();
        LevelledEngine lv = new LevelledEngine();
        for (int start = 0; start < OPS; start += FLUSH_EVERY) {
            Map<Integer, Integer> memtable = new LinkedHashMap<>();
            for (int i = start; i < start + FLUSH_EVERY; i++) {
                memtable.put(ops[i][0], ops[i][1]);
                oracle.put(ops[i][0], ops[i][1]);
            }
            st.flush(memtable);
            lv.flush(memtable);
            check(st.materialize().equals(oracle), "size-tiered lost a key at " + start);
            check(lv.materialize().equals(oracle), "levelled lost a key at " + start);
        }
        check(oracle.size() == 107, "live keys hand-counted to 107");
        check(st.materialize().equals(oracle) && lv.materialize().equals(oracle), "final materialized states");

        check(st.totalWritten == 357, "size-tiered entry writes");
        check(st.mergeCount == 5, "size-tiered compactions");
        check(st.peakRuns == 6, "size-tiered peak runs");
        check(st.peakEntries == 115, "size-tiered peak entries");
        check(st.runs.size() == 1 && st.runs.get(0).size() == 107, "size-tiered final run");

        check(lv.totalWritten == 600, "levelled entry writes");
        check(lv.mergeCount == 15, "levelled compactions (8 + 4 + 2 + 1)");
        check(lv.peakRuns == 4, "levelled peak runs");
        check(lv.peakEntries == 112, "levelled peak entries");
        check(lv.levels.size() == 5, "levelled depth");
        for (int i = 0; i < lv.levels.size() - 1; i++) check(lv.levels.get(i).isEmpty(), "lower levels drained");
        check(lv.levels.get(4).size() == 1 && lv.levels.get(4).get(0).size() == 107, "levelled final run");

        // Documented qualitative outcomes for this parameterisation.
        check(lv.totalWritten > st.totalWritten, "levelled rewrites more entries (write amplification)");
        check(lv.mergeCount > st.mergeCount, "levelled compacts more often, in smaller steps");
        check(st.peakRuns > lv.peakRuns, "size-tiered keeps more runs alive (read amplification)");
        check(st.peakEntries >= lv.peakEntries, "size-tiered temporarily stores more copies");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
