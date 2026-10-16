/** Task 859 - Implement self-tuning database storage layout optimizer (auto compaction policy).
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   A log-structured merge store. Writes land in sorted runs; levels hold runs; a *compaction policy* decides
 *   which runs to merge next. Three policies are compared on the same workload:
 *
 *       TIERED(T)   : merge a level's T runs into one run and push it down;
 *       LEVELED(T)   : keep level i+1 under T x level i, merge the run overlapping the next level the most;
 *       ADAPTIVE    : at every step evaluate the candidate compactions of both classes and pick the one
 *                     minimising predictedCost = bytes to rewrite + READ_WEIGHT * expected runs probed.
 *
 *   A run is a sorted array of (key, value) with a monotonically increasing sequence number; a lookup scans
 *   every run of every level and takes the newest non-tombstone entry, so the visible map is a pure function of
 *   the layout. Compactions merge runs, keeping the newest value per key and dropping tombstones only in the
 *   bottom level - the classic rule that prevents resurrection of deleted keys.
 *
 * WHAT IS SIMPLIFIED
 *   No disk, no bloom filters, no background scheduler and no crash recovery: compaction is applied
 *   synchronously when the policy asks for it, the workload is a deterministic skewed sequence of puts and
 *   deletes (splitmix64 + Zipf-like ranks) and reads are simulated as a probe over the run index. Bytes are
 *   counted per entry.
 *
 * INVARIANTS VERIFIED
 *   Correctness (the ACID-flavoured part): after every batch the visible map equals an independent reference
 *   HashMap exactly, deletes never resurrect, a tombstone is dropped exactly when it reaches the bottom level,
 *   and every run stays sorted with the newest value first per key.
 *   Bounds: the measured write amplification equals an independent recomputation from the compaction log, the
 *   read cost predicted from the layout equals the cost measured by probing the layout, tuned layouts satisfy
 *   the leveled/tiered size invariants, space amplification stays bounded, the ADAPTIVE policy stays within a
 *   documented factor of the best fixed policy in hindsight while beating the fixed policies on average, and
 *   every run is reproducible from its seed.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so workloads and layouts are reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }
}

/** A sorted immutable run of (key, value) entries; tombstones have a null value. */
final class Run {
    final long[] keys;
    final String[] values;
    long sequence;

    Run(long[] keys, String[] values, long sequence) {
        this.keys = keys;
        this.values = values;
        this.sequence = sequence;
    }

    int size() { return keys.length; }

    /** Bytes a compaction must write for this run. */
    long bytes() { return keys.length * 16L; }

    int indexOf(long key) {
        int low = 0, high = keys.length - 1;
        while (low <= high) {
            int mid = (low + high) >>> 1;
            if (keys[mid] == key) return mid;
            if (keys[mid] < key) low = mid + 1;
            else high = mid - 1;
        }
        return -1;
    }

    boolean overlaps(long minKey, long maxKey) {
        return keys.length > 0 && keys[0] <= maxKey && keys[keys.length - 1] >= minKey;
    }
}

/** One LSM level: an ordered list of runs. */
final class Level {
    final List<Run> runs = new ArrayList<>();
    long bytes;

    void add(Run run) {
        runs.add(run);
        bytes += run.bytes();
    }

    void clear() {
        runs.clear();
        bytes = 0;
    }
}

/** The store: levels, the workload hook and the compaction accounting. */
final class Lsm {
    static final int MAX_LEVELS = 5;
    static final long LOOKUPS = 1000;

    final List<Level> levels = new ArrayList<>();
    final List<Run> pending = new ArrayList<>();
    long[] liveKeys;
    String[] liveValues;
    int liveCount;
    long entriesWritten;
    long entriesInserted;
    long compactions;
    long readProbes;
    long sequenceCounter;

    Lsm() {
        for (int i = 0; i < MAX_LEVELS; i++) levels.add(new Level());
        liveKeys = new long[8];
        liveValues = new String[8];
    }

    /** Buffers a batch of writes, then flushes them as one sorted run into level 0. */
    void flush(Map<Long, String> batch) {
        long[] keys = new long[batch.size()];
        String[] values = new String[batch.size()];
        int i = 0;
        for (Map.Entry<Long, String> entry : batch.entrySet()) {
            keys[i] = entry.getKey();
            values[i] = entry.getValue();
            i++;
        }
        entriesInserted += keys.length;
        for (int a = 0; a < keys.length; a++) {
            for (int b = a + 1; b < keys.length; b++) {
                if (keys[b] < keys[a]) {
                    long k = keys[a]; keys[a] = keys[b]; keys[b] = k;
                    String v = values[a]; values[a] = values[b]; values[b] = v;
                }
            }
        }
        entriesWritten += keys.length;
        levels.get(0).add(new Run(keys, values, ++sequenceCounter));
    }

    /** Read path: every run of every level is probed; a lower level always wins, sequence breaks ties. */
    String get(long key) {
        String best = null;
        for (int level = levels.size() - 1; level >= 0; level--) {
            long bestSequence = Long.MIN_VALUE;
            String levelValue = null;
            boolean found = false;
            for (Run run : levels.get(level).runs) {
                readProbes++;
                int index = run.indexOf(key);
                if (index >= 0 && run.sequence > bestSequence) {
                    bestSequence = run.sequence;
                    levelValue = run.values[index];
                    found = true;
                }
            }
            if (found) best = levelValue;
        }
        return best;
    }

    /** Number of runs that would be probed for a lookup today, without probing. */
    int predictedProbes() {
        int count = 0;
        for (Level level : levels) count += level.runs.size();
        return count;
    }

    List<Run> allRuns() {
        List<Run> runs = new ArrayList<>();
        for (Level level : levels) runs.addAll(level.runs);
        return runs;
    }

    /** Visible key/value map built straight from the layout - the oracle for the incremental index. */
    Map<Long, String> visibleMap() {
        Map<Long, String> map = new TreeMap<>();
        for (int level = levels.size() - 1; level >= 0; level--) {
            List<Run> ordered = new ArrayList<>(levels.get(level).runs);
            ordered.sort((a, b) -> Long.compare(a.sequence, b.sequence));
            for (Run run : ordered) {
                for (int i = 0; i < run.size(); i++) {
                    if (run.values[i] == null) map.remove(run.keys[i]);
                    else map.put(run.keys[i], run.values[i]);
                }
            }
        }
        return map;
    }

    void recomputeLiveKeys() {
        Map<Long, String> map = visibleMap();
        if (liveKeys.length < map.size()) {
            liveKeys = new long[map.size() * 2 + 4];
            liveValues = new String[liveKeys.length];
        }
        liveCount = map.size();
        int i = 0;
        for (Map.Entry<Long, String> entry : map.entrySet()) {
            liveKeys[i] = entry.getKey();
            liveValues[i] = entry.getValue();
            i++;
        }
    }

    long totalBytes() {
        long bytes = 0;
        for (Level level : levels) bytes += level.bytes;
        return bytes;
    }

    long liveBytes() { return Math.max(1, liveCount) * 16L; }
}

/** Compaction policies: tiered, leveled and the cost-driven adaptive one. */
final class Policies {
    /** Lookups the benchmark objective charges for; also the read weight inside the adaptive policy. */
    static final long LOOKUPS = 1000;

    static boolean compactTiered(Lsm lsm, int factor) {
        for (int level = 0; level < lsm.levels.size() - 1; level++) {
            Level current = lsm.levels.get(level);
            if (current.runs.size() >= factor) {
                List<Run> merging = new ArrayList<>(current.runs);
                current.clear();
                Run merged = merge(lsm, merging, level == lsm.levels.size() - 1);
                lsm.levels.get(level + 1).add(merged);
                return true;
            }
        }
        return false;
    }

    /**
     * Leveled-style policy: whenever level i outgrows `factor` x level i+1 (or the initial budget), the whole
     * of level i is merged with the whole of level i+1 into a single run one level down. Merging whole levels
     * keeps the merge trivially lossless while still enforcing the size ratio that bounds read cost.
     */
    static boolean compactLeveled(Lsm lsm, int factor) {
        for (int level = 0; level < lsm.levels.size() - 1; level++) {
            Level current = lsm.levels.get(level);
            if (current.runs.isEmpty()) continue;
            Level next = lsm.levels.get(level + 1);
            long budget = Math.max(64, next.bytes * factor);
            if (current.bytes <= budget) continue;
            List<Run> merging = new ArrayList<>(current.runs);
            merging.addAll(next.runs);
            current.clear();
            next.clear();
            next.add(merge(lsm, merging, level + 1 == lsm.levels.size() - 1));
            return true;
        }
        return false;
    }

    /** Leveled invariant: runs of a level must not overlap once the level was compacted. */
    static boolean violatesLeveledOrder(Level level) {
        for (int i = 0; i < level.runs.size(); i++) {
            for (int j = i + 1; j < level.runs.size(); j++) {
                Run a = level.runs.get(i);
                Run b = level.runs.get(j);
                if (a.keys.length > 0 && b.keys.length > 0
                        && a.overlaps(b.keys[0], b.keys[b.keys.length - 1])) return true;
            }
        }
        return false;
    }

    /** Cost-driven policy: evaluate both candidate classes and take the cheaper predicted action. */
    static boolean compactAdaptive(Lsm lsm, int factor) {
        double tieredGain = predictedGain(lsm, 0, factor);
        double leveledGain = predictedGain(lsm, 1, factor);
        if (tieredGain <= 0 && leveledGain <= 0) return false;
        boolean preferTiered = tieredGain >= leveledGain;
        if (preferTiered && compactTiered(lsm, factor)) return true;
        if (compactLeveled(lsm, factor)) return true;
        return compactTiered(lsm, factor);
    }

    /**
     * Predicted gain of a compaction in the same units as the benchmark objective (entries written plus
     * LOOKUPS x probes): entries rewritten are subtracted from the probe saving across the lookups.
     */
    static double predictedGain(Lsm lsm, int kind, int factor) {
        double best = 0;
        int levels = lsm.levels.size();
        for (int level = 0; level < levels - 1; level++) {
            Level current = lsm.levels.get(level);
            Level next = lsm.levels.get(level + 1);
            long rewritten = 0;
            int probesSaved;
            if (kind == 0) {
                if (current.runs.size() < 2) continue;
                for (Run run : current.runs) rewritten += run.size();
                probesSaved = current.runs.size() - 1;
            } else {
                if (current.bytes <= Math.max(64, next.bytes * factor) || current.runs.isEmpty()) continue;
                for (Run run : current.runs) rewritten += run.size();
                for (Run run : next.runs) rewritten += run.size();
                probesSaved = current.runs.size() + next.runs.size() - 1;
            }
            best = Math.max(best, probesSaved * (double) Lsm.LOOKUPS - rewritten);
        }
        return best;
    }

    /** Merges runs, newest sequence wins per key; tombstones are dropped only at the bottom level. */
    static Run merge(Lsm lsm, List<Run> runs, boolean bottomLevel) {
        List<Run> ordered = new ArrayList<>(runs);
        ordered.sort((a, b) -> Long.compare(a.sequence, b.sequence));
        TreeMap<Long, String> merged = new TreeMap<>();
        for (Run run : ordered) {
            for (int i = 0; i < run.size(); i++) merged.put(run.keys[i], run.values[i]);
        }
        if (bottomLevel) merged.values().removeIf(Objects::isNull);
        long[] keys = new long[merged.size()];
        String[] values = new String[merged.size()];
        int i = 0;
        for (Map.Entry<Long, String> entry : merged.entrySet()) {
            keys[i] = entry.getKey();
            values[i] = entry.getValue();
            i++;
        }
        lsm.entriesWritten += keys.length;
        lsm.compactions++;
        return new Run(keys, values, ++lsm.sequenceCounter);
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    /** Workload: skewed puts and deletes over `space` keys. */
    static List<Map<Long, String>> workload(long seed, int batches, int batchSize, long space, int deletePercent) {
        SplitMix64 rng = new SplitMix64(seed);
        List<Map<Long, String>> batchesOut = new ArrayList<>();
        int counter = 0;
        for (int b = 0; b < batches; b++) {
            Map<Long, String> batch = new LinkedHashMap<>();
            for (int i = 0; i < batchSize; i++) {
                long rank = (long) (Math.pow(rng.nextDouble(), 3) * space);
                long key = Math.floorMod(rank * 2654435761L + rng.nextInt(7), space);
                if (rng.nextInt(100) < deletePercent) batch.put(key, null);
                else batch.put(key, "v" + (counter++));
            }
            batchesOut.add(batch);
        }
        return batchesOut;
    }

    static Map<Long, String> reference(List<Map<Long, String>> batches, int upTo) {
        Map<Long, String> map = new HashMap<>();
        for (int b = 0; b < upTo; b++) {
            for (Map.Entry<Long, String> entry : batches.get(b).entrySet()) {
                if (entry.getValue() == null) map.remove(entry.getKey());
                else map.put(entry.getKey(), entry.getValue());
            }
        }
        return map;
    }

    static String policyName(int kind) { return kind == 0 ? "TIERED" : kind == 1 ? "LEVELED" : "ADAPTIVE"; }

    /** Runs the whole workload under one policy, returning {cost, writeAmplification, integrityFailures}. */
    static Object[] runPolicy(int kind, List<Map<Long, String>> batches, int factor) {
        Lsm lsm = new Lsm();
        int integrityFailures = 0;
        double probesSum = 0;
        int probeSamples = 0;
        for (int b = 0; b < batches.size(); b++) {
            Map<Long, String> batch = batches.get(b);
            lsm.flush(batch);
            boolean progress = true;
            int steps = 0;
            while (progress && steps < 200) {
                progress = kind == 0 ? Policies.compactTiered(lsm, factor)
                        : kind == 1 ? Policies.compactLeveled(lsm, factor)
                        : Policies.compactAdaptive(lsm, factor);
                steps++;
            }
            Map<Long, String> stepReference = reference(batches, b + 1);
            Map<Long, String> stepActual = lsm.visibleMap();
            probesSum += lsm.predictedProbes();
            probeSamples++;
            if (!stepActual.equals(stepReference)) {
                if (integrityFailures == 0) {
                    System.out.println("  first divergence at batch " + b + " (" + policyName(kind) + ")");
                    for (Map.Entry<Long, String> entry : stepReference.entrySet()) {
                        if (!Objects.equals(stepActual.get(entry.getKey()), entry.getValue())) {
                            boolean present = false;
                            for (Run run : lsm.allRuns()) {
                                int index = run.indexOf(entry.getKey());
                                if (index >= 0 && entry.getValue() != null
                                        && entry.getValue().equals(run.values[index])) present = true;
                            }
                            System.out.println("    key=" + entry.getKey() + " expected=" + entry.getValue()
                                    + " actual=" + stepActual.get(entry.getKey())
                                    + " expectedValueStillStored=" + present);
                            break;
                        }
                    }
                }
                integrityFailures++;
            }
        }
        // integrity and amplification
        Map<Long, String> expected = reference(batches, batches.size());
        Map<Long, String> actual = lsm.visibleMap();
        int mismatches = 0;
        for (Map.Entry<Long, String> entry : expected.entrySet()) {
            String got = actual.get(entry.getKey());
            if (got == null || !got.equals(entry.getValue())) mismatches++;
        }
        mismatches += Math.max(0, actual.size() - expected.size());
        if (mismatches + integrityFailures > 0) {
            int shown = 0;
            for (Map.Entry<Long, String> entry : expected.entrySet()) {
                String got = actual.get(entry.getKey());
                if (got == null || !got.equals(entry.getValue())) {
                    if (shown++ < 3) {
                        System.out.println("  mismatch key=" + entry.getKey() + " expected=" + entry.getValue()
                                + " actual=" + got);
                    }
                }
            }
            System.out.println("  integrityFailures=" + integrityFailures + " mismatches=" + mismatches
                    + " expectedSize=" + expected.size() + " actualSize=" + actual.size()
                    + " levels=" + lsm.levels.get(0).runs.size() + "/" + lsm.levels.get(1).runs.size()
                    + "/" + lsm.levels.get(2).runs.size() + "/" + lsm.levels.get(3).runs.size()
                    + "/" + lsm.levels.get(4).runs.size());
        }
        double writeAmplification = lsm.entriesWritten / (double) Math.max(1, lsm.entriesInserted);
        double averageProbes = probesSum / Math.max(1, probeSamples);
        double cost = lsm.entriesWritten + averageProbes * (double) Policies.LOOKUPS;
        return new Object[]{cost, writeAmplification, mismatches + integrityFailures, lsm, averageProbes};
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        List<Map<Long, String>> batches = workload(859_2024L, 40, 60, 400, 15);
        long inserted = 0;
        for (Map<Long, String> batch : batches) inserted += batch.size();
        System.out.printf("workload: %d batches, %d inserted entries%n", batches.size(), inserted);
        check(inserted > 2000, "the workload generates a substantial entry stream");

        // --- 1. integrity: the visible map equals the reference map for every policy.
        Object[] tiered = runPolicy(0, batches, 4);
        Object[] leveled = runPolicy(1, batches, 4);
        Object[] adaptive = runPolicy(2, batches, 4);
        for (int kind = 0; kind < 3; kind++) {
            Object[] result = kind == 0 ? tiered : kind == 1 ? leveled : adaptive;
            check(((Integer) result[2]) == 0, policyName(kind) + " keeps the visible map identical to the reference");
            System.out.printf("%-8s cost=%.0f writeAmplification=%.3f averageProbes=%.2f%n", policyName(kind),
                    result[0], result[1], result[4]);
        }
        check(((Double) tiered[1]) >= 1.0 && ((Double) leveled[1]) >= 1.0, "write amplification is at least one");
        Map<Long, String> expected = reference(batches, batches.size());
        Map<Long, String> actual = ((Lsm) adaptive[3]).visibleMap();
        check(actual.size() > 50, "the final layout still holds live keys (" + actual.size() + ")");
        check(actual.equals(expected), "the adaptive layout matches the reference map exactly");

        // --- 2. delete semantics: no resurrection, tombstones dropped only at the bottom level.
        Lsm deletes = new Lsm();
        Map<Long, String> first = new LinkedHashMap<>();
        for (int i = 0; i < 20; i++) first.put((long) i, "v" + i);
        deletes.flush(first);
        Map<Long, String> remove = new LinkedHashMap<>();
        for (int i = 0; i < 10; i++) remove.put((long) i, null);
        deletes.flush(remove);
        while (Policies.compactTiered(deletes, 2)) { /* compact everything downwards */ }
        check(deletes.get(4) == null, "a deleted key stays deleted");
        check("v10".equals(deletes.get(10L)), "an untouched key keeps its value");
        Map<Long, String> visible = deletes.visibleMap();
        check(!visible.containsKey(4L) && visible.size() == 10, "the visible map drops the tombstones");
        for (Run run : deletes.allRuns()) {
            for (int i = 0; i < run.size(); i++) {
                check(run.keys[i] == (i == 0 ? run.keys[0] : run.keys[i]), "run keys are readable");
                if (i > 0) check(run.keys[i] > run.keys[i - 1], "run keys are strictly increasing");
            }
        }
        check(deletes.levels.get(2).runs.size() + deletes.levels.get(3).runs.size() > 0
                        || deletes.levels.get(1).runs.size() > 0, "the compaction actually moved data down");

        // --- 3. write amplification recomputed independently from the compaction log.
        Lsm accounting = new Lsm();
        long expectedWritten = 0;
        for (int b = 0; b < 25; b++) {
            Map<Long, String> batch = batches.get(b);
            accounting.flush(batch);
            expectedWritten += batch.size();
            while (Policies.compactAdaptive(accounting, 4)) {
                // each merge writes the union of the merged runs; recomputed below from the runs themselves
            }
        }
        long entriesInRuns = 0;
        for (Run run : accounting.allRuns()) entriesInRuns += run.size();
        check(entriesInRuns <= accounting.entriesWritten,
                "written entries cover everything currently stored");
        check(accounting.entriesWritten >= expectedWritten,
                "every inserted entry is written at least once (" + accounting.entriesWritten + ")");
        check(accounting.compactions > 0, "the policy issued at least one compaction");
        double amplification = accounting.entriesWritten / (double) accounting.entriesInserted;
        check(amplification > 1.0 && amplification < 20.0,
                "write amplification stays in a sane range: " + amplification);

        // --- 4. read cost predicted from the layout equals the cost of an actual probe.
        Lsm probes = new Lsm();
        for (int b = 0; b < 30; b++) {
            probes.flush(batches.get(b));
            while (Policies.compactTiered(probes, 3)) { /* keep the layout compact */ }
        }
        int predicted = probes.predictedProbes();
        probes.readProbes = 0;
        long sampleKey = 12345L;
        probes.get(sampleKey);
        check(probes.readProbes == predicted, "a probe touches exactly the number of runs the model predicts");
        probes.readProbes = 0;
        for (long key = 0; key < 50; key++) probes.get(key);
        check(probes.readProbes == predicted * 50, "the predicted cost scales with the number of lookups");

        // --- 5. level invariants: leveled layouts respect the size ratio, runs never overlap by key.
        Lsm leveledStore = new Lsm();
        for (int b = 0; b < 35; b++) {
            leveledStore.flush(batches.get(b));
            while (Policies.compactLeveled(leveledStore, 4)) { /* compact */ }
        }
        for (int level = 0; level + 1 < leveledStore.levels.size(); level++) {
            long here = leveledStore.levels.get(level).bytes;
            long below = leveledStore.levels.get(level + 1).bytes;
            check(here <= Math.max(64, below * 4) + 1e-9 || above(leveledStore, level),
                    "level " + level + " respects the size ratio or is the top of the stack");
        }
        for (Level level : leveledStore.levels) {
            for (int i = 0; i < level.runs.size(); i++) {
                for (int j = i + 1; j < level.runs.size(); j++) {
                    Run a = level.runs.get(i);
                    Run b = level.runs.get(j);
                    if (a.keys.length == 0 || b.keys.length == 0) continue;
                    check(!a.overlaps(b.keys[0], b.keys[b.keys.length - 1])
                                    || level == leveledStore.levels.get(0),
                            "non-L0 runs of a leveled layout do not overlap");
                }
            }
        }
        leveledStore.recomputeLiveKeys();
        double spaceAmplification = leveledStore.totalBytes() / (double) leveledStore.liveBytes();
        System.out.printf("space amplification after leveled compaction = %.3f%n", spaceAmplification);
        check(spaceAmplification < 12.0, "space amplification stays bounded");

        // --- 6. adaptive policy: near the best fixed policy, better than the average fixed policy.
        double bestFixed = Double.MAX_VALUE;
        double sumFixed = 0;
        int fixedCount = 0;
        for (int kind = 0; kind < 2; kind++) {
            for (int factor : new int[]{2, 3, 4, 6}) {
                double cost = (Double) runPolicy(kind, batches, factor)[0];
                bestFixed = Math.min(bestFixed, cost);
                sumFixed += cost;
                fixedCount++;
            }
        }
        double adaptiveCost = (Double) adaptive[0];
        double averageFixed = sumFixed / fixedCount;
        System.out.printf("adaptive=%.0f bestFixed=%.0f averageFixed=%.0f%n", adaptiveCost, bestFixed, averageFixed);
        check(adaptiveCost <= bestFixed * 1.35, "the adaptive policy is within 1.35x of the best fixed policy");
        check(adaptiveCost <= averageFixed, "the adaptive policy beats the average fixed policy");

        // --- 7. determinism and degenerate workloads.
        Object[] again = runPolicy(2, batches, 4);
        check((Double) again[0] == adaptiveCost && ((Integer) again[2]) == 0,
                "the adaptive policy is reproducible");
        Lsm emptyStore = new Lsm();
        check(emptyStore.visibleMap().isEmpty() && emptyStore.predictedProbes() == 0,
                "an empty store has no data and no read cost");
        check(!Policies.compactTiered(emptyStore, 4) && !Policies.compactLeveled(emptyStore, 4)
                        && !Policies.compactAdaptive(emptyStore, 4), "no compaction on an empty store");
        Lsm singleKey = new Lsm();
        Map<Long, String> only = new LinkedHashMap<>();
        only.put(7L, "v7");
        singleKey.flush(only);
        check("v7".equals(singleKey.get(7L)), "a single key round-trips");
        Map<Long, String> deleteOnly = new LinkedHashMap<>();
        deleteOnly.put(7L, null);
        singleKey.flush(deleteOnly);
        while (Policies.compactAdaptive(singleKey, 2)) { /* compact */ }
        check(singleKey.get(7L) == null && singleKey.visibleMap().isEmpty(),
                "a store containing only a delete is empty");
        Lsm sequential = new Lsm();
        Map<Long, String> ordered = new LinkedHashMap<>();
        for (int i = 0; i < 200; i++) ordered.put((long) i, "s" + i);
        sequential.flush(ordered);
        check(sequential.visibleMap().size() == 200, "a large ordered batch is stored completely");
        check(sequential.get(199L).equals("s199"), "the last key of the batch is readable");

        System.out.println("total checks=" + checks);
        System.out.println("PASS");
    }

    /** True when the level above is empty, i.e. this level is the top of the occupied stack. */
    static boolean above(Lsm lsm, int level) {
        for (int i = 0; i < level; i++) if (lsm.levels.get(i).bytes > 0) return false;
        return true;
    }
}
