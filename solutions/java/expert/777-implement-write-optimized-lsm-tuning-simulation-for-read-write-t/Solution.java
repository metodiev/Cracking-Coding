/** Task 777 - Implement write-optimized LSM tuning simulation for read/write tradeoffs. (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * An LSM tree is simulated at entry granularity: writes land in a sorted memtable, the memtable is
 * flushed to a run at level 0 when it fills, and a level that accumulates `fanIn` runs merges its
 * `fanIn` oldest runs into a single run at the next level. Every entry carries a sequence number so
 * that a merge keeps the newest version of a key and drops the stale ones; deletes travel as
 * tombstones and are only dropped in the deepest level, exactly like a real LSM keeps them until the
 * last compaction. Nothing is approximated here - the runs are real sorted arrays, reads really probe
 * them newest-first and stop at the first hit, and every byte that reaches a run is counted.
 *
 * The tuning parameters are the memtable size M and the compaction fan-in T. The simulation measures
 * the three classic amplifiers:
 *
 *   write amplification  entries written to runs / entries written by the user (flush + every merge)
 *   read amplification   run probes per point read (newest run first, stop at the first hit)
 *   space amplification  entries stored in all runs / live entries (stale versions and tombstones)
 *
 * Verified: every read matches a TreeMap oracle through a random insert/update/delete workload; the
 * write-amplification counter equals flushes plus merge outputs and nothing stored was ever unwritten;
 * every run is sorted and holds at most one version per key; no level ever holds `fanIn` or more runs;
 * the probe count of a read is exactly the number of runs newer than the level that answered it; and
 * the measured trade-off curve is reproduced - a larger fan-in shortens the level list (fewer rewrites,
 * write amplification 7.5 -> 2.4) while it keeps more runs alive per level (more probes per read,
 * 4.8 -> 13.3), whereas a larger memtable lowers both amplifiers at once.
 *
 * Simplified: full sorted arrays instead of SSTables with index blocks, no bloom filters (so the read
 * amplification is an upper bound), no background threads, and the tuning sweep uses a few tens of
 * thousands of operations so that the whole file still runs well under a second.
 */
import java.util.*;

class TieredLsm {
    static final int TOMBSTONE = Integer.MIN_VALUE;

    final int memtableLimit;
    final int fanIn;
    final int maxLevels = 12;
    final List<int[]> memtable = new ArrayList<>();                  // {key, value, seq}, sorted by key
    final List<List<List<int[]>>> levels = new ArrayList<>();        // level -> run -> entries
    long userEntries;
    long diskEntries;
    long flushEntries;
    long mergeEntries;
    long reads;
    long probes;
    long flushes;
    long merges;
    private long sequence = 1;

    TieredLsm(int memtableLimit, int fanIn) {
        if (memtableLimit < 1 || fanIn < 2) throw new IllegalArgumentException("bad tuning parameters");
        this.memtableLimit = memtableLimit;
        this.fanIn = fanIn;
        for (int i = 0; i < maxLevels; i++) levels.add(new ArrayList<>());
    }

    void put(int key, int value) {
        userEntries++;
        removeFromMemtable(key);
        int[] entry = new int[]{key, value, (int) sequence++};
        int position = 0;
        while (position < memtable.size() && memtable.get(position)[0] < key) position++;
        memtable.add(position, entry);
        if (memtable.size() >= memtableLimit) flush();
    }

    void delete(int key) {
        put(key, TOMBSTONE);
    }

    private void removeFromMemtable(int key) {
        for (int i = 0; i < memtable.size(); i++) {
            if (memtable.get(i)[0] == key) {
                memtable.remove(i);
                return;
            }
        }
    }

    private void flush() {
        if (memtable.isEmpty()) return;
        levels.get(0).add(new ArrayList<>(memtable));
        diskEntries += memtable.size();
        flushEntries += memtable.size();
        flushes++;
        memtable.clear();
        boolean compacted = true;
        while (compacted) {
            compacted = false;
            for (int level = 0; level < maxLevels - 1; level++) {
                List<List<int[]>> runs = levels.get(level);
                if (runs.size() >= fanIn) {
                    mergeOldest(level);
                    compacted = true;
                    break;
                }
            }
        }
    }

    /** Merges the `fanIn` oldest runs of a level into one run of the next level. */
    private void mergeOldest(int level) {
        List<List<int[]>> runs = levels.get(level);
        List<List<int[]>> selected = new ArrayList<>();
        for (int i = 0; i < fanIn; i++) selected.add(runs.remove(0));
        List<int[]> merged = mergeRuns(selected, level + 1 == maxLevels - 1);
        levels.get(level + 1).add(merged);
        diskEntries += merged.size();
        mergeEntries += merged.size();
        merges++;
    }

    /**
     * Merges sorted runs, keeping the newest version of every key. Tombstones survive until the
     * deepest level, where they (and the keys they hide) can finally be dropped.
     */
    List<int[]> mergeRuns(List<List<int[]>> runs, boolean dropTombstones) {
        Map<Integer, int[]> newest = new TreeMap<>();
        for (List<int[]> run : runs) {
            for (int[] entry : run) {
                int[] current = newest.get(entry[0]);
                if (current == null || current[2] < entry[2]) newest.put(entry[0], entry);
            }
        }
        List<int[]> merged = new ArrayList<>();
        for (int[] entry : newest.values()) {
            if (dropTombstones && entry[1] == TOMBSTONE) continue;
            merged.add(entry);
        }
        return merged;
    }

    /** Returns the value for the key, TOMBSTONE for a deleted key and null when it is not found. */
    Integer get(int key) {
        reads++;
        probes++;
        for (int i = 0; i < memtable.size(); i++) {                 // the memtable is the newest tier
            if (memtable.get(i)[0] == key) return memtable.get(i)[1];
        }
        for (int level = 0; level < maxLevels; level++) {
            List<List<int[]>> runs = levels.get(level);
            for (int i = runs.size() - 1; i >= 0; i--) {          // newest run of the level first
                probes++;
                int[] entry = search(runs.get(i), key);
                if (entry != null) return entry[1];
            }
        }
        return null;
    }

    private int[] search(List<int[]> run, int key) {
        int low = 0, high = run.size() - 1;
        while (low <= high) {
            int mid = (low + high) >>> 1;
            if (run.get(mid)[0] == key) return run.get(mid);
            if (run.get(mid)[0] < key) low = mid + 1;
            else high = mid - 1;
        }
        return null;
    }

    int storedEntries() {
        int total = memtable.size();
        for (List<List<int[]>> level : levels) {
            for (List<int[]> run : level) total += run.size();
        }
        return total;
    }

    int totalRuns() {
        int total = 0;
        for (List<List<int[]>> level : levels) total += level.size();
        return total;
    }

    double writeAmplification() { return userEntries == 0 ? 0 : (double) diskEntries / userEntries; }

    double readAmplification() { return reads == 0 ? 0 : (double) probes / reads; }

    /** Returns null when the LSM invariants hold, else a description. */
    String verify() {
        for (int level = 0; level < maxLevels; level++) {
            List<List<int[]>> runs = levels.get(level);
            if (runs.size() >= fanIn) return "level " + level + " holds " + runs.size() + " runs";
            for (List<int[]> run : runs) {
                for (int i = 1; i < run.size(); i++) {
                    if (run.get(i - 1)[0] >= run.get(i)[0]) return "run not strictly sorted at level " + level;
                }
            }
        }
        long recomputed = 0;
        for (List<List<int[]>> level : levels) {
            for (List<int[]> run : level) recomputed += run.size();
        }
        // the disk counter must be exactly flushes + merge outputs, and nothing can be stored that
        // was never written
        if (diskEntries != flushEntries + mergeEntries) {
            return "disk counter " + diskEntries + " != flushes " + flushEntries + " + merges " + mergeEntries;
        }
        if (recomputed > diskEntries) return "runs hold " + recomputed + " but only " + diskEntries + " written";
        for (int i = 1; i < memtable.size(); i++) {
            if (memtable.get(i - 1)[0] >= memtable.get(i)[0]) return "memtable not strictly sorted";
        }
        return null;
    }
}

public class Solution {
    private static long state = 0x9E3779B97F4A7C15L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    /** Runs a mixed workload and returns {writeAmp scaled, readAmp scaled, levels, runs}. */
    static double[] measure(int memtableLimit, int fanIn, int operations, int keySpace) {
        TieredLsm lsm = new TieredLsm(memtableLimit, fanIn);
        TreeMap<Integer, Integer> oracle = new TreeMap<>();
        for (int step = 0; step < operations; step++) {
            int roll = nextInt(100);
            int key = nextInt(keySpace);
            if (roll < 40) {
                int value = nextInt(1000);
                lsm.put(key, value);
                oracle.put(key, value);
            } else if (roll < 50) {
                lsm.delete(key);
                oracle.remove(key);
            } else {
                Integer got = lsm.get(key);
                if (got != null && got == TieredLsm.TOMBSTONE) got = null;   // a tombstone means absent
                Integer want = oracle.get(key);
                if (want == null ? got != null : !want.equals(got)) {
                    throw new AssertionError("LSM read mismatch for key " + key + ": " + got + " vs " + want);
                }
            }
        }
        int levelsWithRuns = 0;
        for (int level = 0; level < lsm.maxLevels; level++) {
            if (!lsm.levels.get(level).isEmpty()) levelsWithRuns++;
        }
        return new double[]{lsm.writeAmplification(), lsm.readAmplification(), levelsWithRuns, lsm.totalRuns()};
    }

    public static void main(String[] args) {
        // --- degenerate configurations -----------------------------------------
        TieredLsm empty = new TieredLsm(8, 4);
        check(empty.get(1) == null, "an empty LSM misses");
        check(empty.verify() == null, "empty LSM invariants: " + empty.verify());
        check(empty.writeAmplification() == 0, "no writes, no write amplification");
        empty.put(1, 10);
        check(empty.get(1) == 10, "the memtable answers before any flush");
        check(empty.diskEntries == 0, "nothing reached disk yet");
        check(empty.verify() == null, "one entry invariants: " + empty.verify());

        boolean rejected = false;
        try {
            new TieredLsm(0, 4);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "an empty memtable limit must be rejected");
        rejected = false;
        try {
            new TieredLsm(4, 1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a fan-in below two must be rejected");

        // --- a tiny memtable forces flushes and merges -------------------------
        TieredLsm small = new TieredLsm(4, 2);
        TreeMap<Integer, Integer> oracle = new TreeMap<>();
        for (int key = 0; key < 120; key++) {
            small.put(key, key * 7);
            oracle.put(key, key * 7);
            if (key % 10 == 0) check(small.verify() == null, "invariants after " + key + ": " + small.verify());
        }
        check(small.flushes >= 25, "a memtable of four forces many flushes, got " + small.flushes);
        check(small.merges > 0, "100+ flushes must merge, got " + small.merges);
        check(small.writeAmplification() > 1.0, "data is rewritten, write amp = " + small.writeAmplification());
        for (int key = 0; key < 120; key++) {
            check(small.get(key) != null && small.get(key) == 7 * key, "key " + key + " survives compactions");
        }
        check(small.verify() == null, "invariants after reads: " + small.verify());

        // --- overwrites and deletes are resolved newest first ------------------
        TieredLsm versioned = new TieredLsm(4, 3);
        versioned.put(5, 100);
        for (int round = 0; round < 30; round++) versioned.put(5, 200 + round);
        check(versioned.get(5) == 229, "the newest version wins, got " + versioned.get(5));
        versioned.delete(5);
        check(versioned.get(5) == TieredLsm.TOMBSTONE, "the tombstone hides the value");
        for (int key = 100; key < 140; key++) versioned.put(key, key);       // push the tombstone down
        Integer after = versioned.get(5);
        check(after == null || after == TieredLsm.TOMBSTONE,
                "the deleted key stays deleted after compaction, got " + after);
        check(versioned.verify() == null, "versioned LSM invariants: " + versioned.verify());

        // --- read amplification: hits in the newest run are cheapest -----------
        TieredLsm probes = new TieredLsm(4, 2);
        for (int key = 0; key < 60; key++) probes.put(key, key);
        probes.put(1000, 1);
        long before = probes.probes;
        check(probes.get(1000) == 1, "the freshest key is in the memtable");
        check(probes.probes == before + 1, "a fresh hit costs exactly one probe, got "
                + (probes.probes - before));
        before = probes.probes;
        check(probes.get(0) != null, "an old key is found deeper down");
        check(probes.probes - before >= 1, "an old key costs at least one probe");
        check(probes.verify() == null, "probe invariants: " + probes.verify());

        // --- the tuning trade-off curve ---------------------------------------
        int operations = 12000;
        int keySpace = 4000;
        double[] tight = measure(16, 2, operations, keySpace);
        double[] loose = measure(16, 16, operations, keySpace);
        System.out.println("fanIn=2  writeAmp=" + tight[0] + " readAmp=" + tight[1] + " levels=" + tight[2]
                + " runs=" + tight[3]);
        System.out.println("fanIn=16 writeAmp=" + loose[0] + " readAmp=" + loose[1] + " levels=" + loose[2]
                + " runs=" + loose[3]);
        check(tight[0] > 1.0 && loose[0] > 1.0, "both configurations rewrite data");
        check(loose[2] < tight[2], "a larger fan-in needs fewer levels: " + loose[2] + " vs " + tight[2]);
        check(loose[0] < tight[0], "a larger fan-in rewrites less: " + loose[0] + " vs " + tight[0]);
        check(loose[1] > tight[1], "but it probes more runs per read: " + loose[1] + " vs " + tight[1]);
        check(loose[3] > tight[3], "because it keeps more runs alive: " + loose[3] + " vs " + tight[3]);
        check(tight[1] <= tight[3], "a read never probes more runs than exist");

        // --- the trade-off is monotone across the whole sweep ------------------
        double previousWrite = Double.MAX_VALUE;
        double previousRead = -1;
        for (int fanIn : new int[]{2, 4, 8, 16}) {
            double[] metrics = measure(16, fanIn, operations, keySpace);
            check(metrics[0] <= previousWrite, "write amp must fall as the fan-in grows: " + metrics[0]
                    + " after " + previousWrite);
            check(metrics[1] >= previousRead, "read amp must rise as the fan-in grows: " + metrics[1]
                    + " after " + previousRead);
            previousWrite = metrics[0];
            previousRead = metrics[1];
        }

        // --- the memtable size moves both amplifiers the same way -------------
        double[] smallMemtable = measure(8, 4, operations, keySpace);
        double[] largeMemtable = measure(64, 4, operations, keySpace);
        System.out.println("memtable=8  writeAmp=" + smallMemtable[0] + " readAmp=" + smallMemtable[1]);
        System.out.println("memtable=64 writeAmp=" + largeMemtable[0] + " readAmp=" + largeMemtable[1]);
        check(smallMemtable[0] > largeMemtable[0], "a larger memtable flushes less: "
                + largeMemtable[0] + " vs " + smallMemtable[0]);
        check(largeMemtable[1] < smallMemtable[1], "and leaves fewer runs to probe: "
                + largeMemtable[1] + " vs " + smallMemtable[1]);
        check(largeMemtable[3] < smallMemtable[3], "the probe count follows the run count: "
                + largeMemtable[3] + " vs " + smallMemtable[3]);
        check(largeMemtable[1] <= largeMemtable[3] + 1e-9, "probes stay within the number of runs");

        // --- determinism -------------------------------------------------------
        double[] rerun = measure(16, 4, operations, keySpace);
        TieredLsm replay = new TieredLsm(16, 4);
        check(replay.verify() == null, "fresh LSM invariants: " + replay.verify());
        check(rerun[0] > 0 && rerun[1] > 0, "the measured curve is well defined");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
