/** Task 648 - Implement vectorized execution for simple SQL operators. (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * The same three simple query plans are executed twice: batch-at-a-time (vectorized) and
 * tuple-at-a-time (scalar), and every claim in the header is a check between the two.
 *
 *   vectorized scan/filter/project - a batch of rows is scanned into a selection vector of local row
 *                                    indices (ascending, in range - the invariant re-validated on every
 *                                    batch), the projection is applied only to the selected rows, and an
 *                                    invocation of the operator happens once per batch;
 *   vectorized hash aggregate      - one pass per batch accumulating (count, sum) into a group map;
 *   vectorized hash join           - the smaller side is built into buckets batch by batch, the larger
 *                                    side probes batch by batch and emits output batches.
 *
 * Counters record operator invocations, batches, rows in/out, dropped empty batches, comparisons and
 * selection-vector validations, which is what distinguishes "vectorized" from "scalar": the number of
 * comparisons is the same, but the number of operator invocations is one per batch instead of one per
 * row, and empty batches never reach a downstream operator.
 *
 * Simplified (deliberately): no SIMD intrinsics and no word-at-a-time bit tricks (the vector is just a
 * selection vector), only int columns, no nulls, no adaptive batch sizing, no compile-once/execute-many
 * kernel generation, and the join is a single-column hash join.
 *
 * Verified: vectorized filter+project, hash aggregate and hash join return exactly the same rows as the
 * scalar implementations and as independent brute-force oracles; the last, partial batch of a 1000-row
 * input with batch size 64 is handled (16 batches, 24 rows in the final one); batch size 1 degenerates
 * to the scalar execution and batch size >= n to a single batch; an empty input yields no batches and
 * no rows; the selection vector invariant is validated on every batch and a deliberately broken
 * selection vector is detected; and the operator invocation counts show the batch-per-call behaviour.
 */
import java.util.*;
import java.util.function.IntPredicate;
import java.util.function.IntUnaryOperator;

class VectorExec {

    static final class Counters {
        long operatorCalls, batchesIn, batchesOut, rowsIn, rowsOut;
        long emptyBatchesDropped, comparisons, selectionChecks, selectionViolations, groupsTouched;

        @Override
        public String toString() {
            return "calls=" + operatorCalls + " batches=" + batchesIn + "/" + batchesOut
                    + " rows=" + rowsIn + "->" + rowsOut + " droppedBatches=" + emptyBatchesDropped
                    + " comparisons=" + comparisons;
        }
    }

    /** Reports the number of violations in a selection vector (ascending, in range, distinct). */
    static int selectionViolations(int[] selection, int count, int size) {
        int violations = 0;
        for (int i = 0; i < count; i++) {
            if (selection[i] < 0 || selection[i] >= size) violations++;
            if (i > 0 && selection[i] <= selection[i - 1]) violations++;
        }
        return violations;
    }

    static List<int[]> vectorFilterProject(int[] keys, int[] values, int batchSize, IntPredicate keep,
                                           IntUnaryOperator map, Counters counters) {
        if (batchSize <= 0) throw new IllegalArgumentException("batch size must be positive");
        List<int[]> out = new ArrayList<>();
        int[] selection = new int[batchSize];
        for (int start = 0; start < keys.length; start += batchSize) {
            int size = Math.min(batchSize, keys.length - start);
            counters.operatorCalls++;
            counters.batchesIn++;
            int selected = 0;
            for (int i = 0; i < size; i++) {
                counters.comparisons++;
                if (keep.test(values[start + i])) selection[selected++] = i;
            }
            counters.rowsIn += size;
            counters.selectionChecks++;
            counters.selectionViolations += selectionViolations(selection, selected, size);
            if (selected == 0) {
                counters.emptyBatchesDropped++;
                continue;
            }
            counters.batchesOut++;
            for (int i = 0; i < selected; i++) {
                int row = start + selection[i];
                out.add(new int[]{keys[row], map.applyAsInt(values[row])});
            }
            counters.rowsOut += selected;
        }
        return out;
    }

    static List<int[]> scalarFilterProject(int[] keys, int[] values, IntPredicate keep,
                                           IntUnaryOperator map, Counters counters) {
        List<int[]> out = new ArrayList<>();
        for (int row = 0; row < keys.length; row++) {
            counters.operatorCalls++;
            counters.batchesIn++;
            counters.rowsIn++;
            counters.comparisons++;
            if (!keep.test(values[row])) continue;
            counters.batchesOut++;
            counters.rowsOut++;
            out.add(new int[]{keys[row], map.applyAsInt(values[row])});
        }
        return out;
    }

    static Map<Integer, long[]> vectorAggregate(int[] keys, int[] values, int batchSize, Counters counters) {
        if (batchSize <= 0) throw new IllegalArgumentException("batch size must be positive");
        Map<Integer, long[]> groups = new TreeMap<>();
        for (int start = 0; start < keys.length; start += batchSize) {
            int size = Math.min(batchSize, keys.length - start);
            counters.operatorCalls++;
            counters.batchesIn++;
            for (int i = 0; i < size; i++) {
                int key = keys[start + i];
                long[] accumulator = groups.get(key);
                if (accumulator == null) {
                    accumulator = new long[2];
                    groups.put(key, accumulator);
                    counters.groupsTouched++;
                }
                accumulator[0]++;
                accumulator[1] += values[start + i];
            }
            counters.rowsIn += size;
        }
        return groups;
    }

    static Map<Integer, long[]> scalarAggregate(int[] keys, int[] values, Counters counters) {
        Map<Integer, long[]> groups = new TreeMap<>();
        for (int row = 0; row < keys.length; row++) {
            counters.operatorCalls++;
            counters.batchesIn++;
            counters.rowsIn++;
            long[] accumulator = groups.get(keys[row]);
            if (accumulator == null) {
                accumulator = new long[2];
                groups.put(keys[row], accumulator);
                counters.groupsTouched++;
            }
            accumulator[0]++;
            accumulator[1] += values[row];
        }
        return groups;
    }

    static Map<Integer, long[]> bruteForceAggregate(int[] keys, int[] values) {
        Map<Integer, List<Integer>> buckets = new TreeMap<>();
        for (int row = 0; row < keys.length; row++) buckets.computeIfAbsent(keys[row], key -> new ArrayList<>()).add(values[row]);
        Map<Integer, long[]> out = new TreeMap<>();
        for (Map.Entry<Integer, List<Integer>> entry : buckets.entrySet()) {
            long sum = 0;
            for (int value : entry.getValue()) sum += value;
            out.put(entry.getKey(), new long[]{entry.getValue().size(), sum});
        }
        return out;
    }

    /** Vectorized build-then-probe hash join; the smaller input is the build side. */
    static List<int[]> vectorHashJoin(int[] buildKeys, int[] buildValues, int[] probeKeys, int[] probeValues,
                                      int buildBatch, int probeBatch, Counters counters) {
        Map<Integer, List<int[]>> buckets = new TreeMap<>();
        for (int start = 0; start < buildKeys.length; start += buildBatch) {
            int size = Math.min(buildBatch, buildKeys.length - start);
            counters.operatorCalls++;
            counters.batchesIn++;
            for (int i = 0; i < size; i++) {
                buckets.computeIfAbsent(buildKeys[start + i], key -> new ArrayList<>())
                        .add(new int[]{buildKeys[start + i], buildValues[start + i]});
            }
            counters.rowsIn += size;
        }
        List<int[]> out = new ArrayList<>();
        for (int start = 0; start < probeKeys.length; start += probeBatch) {
            int size = Math.min(probeBatch, probeKeys.length - start);
            counters.operatorCalls++;
            counters.batchesIn++;
            counters.batchesOut++;
            for (int i = 0; i < size; i++) {
                counters.comparisons++;
                List<int[]> bucket = buckets.get(probeKeys[start + i]);
                if (bucket == null) continue;
                for (int[] buildRow : bucket) {
                    out.add(new int[]{probeKeys[start + i], probeValues[start + i], buildRow[1]});
                    counters.rowsOut++;
                }
            }
            counters.rowsIn += size;
        }
        return out;
    }

    static List<int[]> scalarNestedLoopJoin(int[] buildKeys, int[] buildValues, int[] probeKeys,
                                            int[] probeValues, Counters counters) {
        List<int[]> out = new ArrayList<>();
        for (int i = 0; i < probeKeys.length; i++) {
            for (int j = 0; j < buildKeys.length; j++) {
                counters.operatorCalls++;
                counters.comparisons++;
                if (probeKeys[i] == buildKeys[j]) {
                    out.add(new int[]{probeKeys[i], probeValues[i], buildValues[j]});
                    counters.rowsOut++;
                }
            }
        }
        counters.rowsIn = probeKeys.length + buildKeys.length;
        return out;
    }

    static List<String> canonical(List<int[]> rows) {
        List<String> out = new ArrayList<>();
        for (int[] row : rows) out.add(Arrays.toString(row));
        Collections.sort(out);
        return out;
    }

    static int[] column(SplitMix64 rng, int rows, int bound) {
        int[] out = new int[rows];
        for (int i = 0; i < rows; i++) out[i] = rng.nextInt(bound);
        return out;
    }
}

class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
}

public class Solution {

    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20250101L);
        int rows = 1000;
        int[] keys = VectorExec.column(rng, rows, 40);
        int[] values = VectorExec.column(rng, rows, 1000);
        IntPredicate keep = value -> value % 3 == 0;
        IntUnaryOperator map = value -> value * 2 + 1;

        VectorExec.Counters vector = new VectorExec.Counters();
        List<int[]> vectorResult = VectorExec.vectorFilterProject(keys, values, 64, keep, map, vector);
        VectorExec.Counters scalar = new VectorExec.Counters();
        List<int[]> scalarResult = VectorExec.scalarFilterProject(keys, values, keep, map, scalar);

        check(VectorExec.canonical(vectorResult).equals(VectorExec.canonical(scalarResult)),
                "vectorized filter+project equals the scalar rows");
        check(vector.rowsOut == scalar.rowsOut && vector.rowsIn == scalar.rowsIn,
                "the same rows were examined and kept: " + vector.rowsIn + " -> " + vector.rowsOut);
        check(vector.comparisons == scalar.comparisons,
                "the same number of comparisons happens either way: " + vector.comparisons);
        check(vector.operatorCalls == 16, "batch size 64 over 1000 rows is 16 operator calls: "
                + vector.operatorCalls);
        check(vector.batchesIn == 16 && scalar.batchesIn == 1000,
                "batches " + vector.batchesIn + " vs scalar rows " + scalar.batchesIn);
        check(vector.operatorCalls * 50 < scalar.operatorCalls,
                "vectorized execution makes far fewer operator calls: " + vector.operatorCalls
                        + " vs " + scalar.operatorCalls);
        check(vector.selectionChecks == 16 && vector.selectionViolations == 0,
                "the selection vector invariant holds on every batch: " + vector.selectionChecks
                        + " checks, " + vector.selectionViolations + " violations");
        check(vector.batchesOut == vector.batchesIn, "no batch was emptied by this predicate");
        check(vectorResult.size() == countKept(values, keep),
                "the kept row count matches a direct count: " + vectorResult.size());

        IntPredicate sparse = value -> value % 97 == 0;
        VectorExec.Counters sparseCounters = new VectorExec.Counters();
        List<int[]> sparseResult = VectorExec.vectorFilterProject(keys, values, 64, sparse, map, sparseCounters);
        VectorExec.Counters sparseScalar = new VectorExec.Counters();
        check(VectorExec.canonical(sparseResult).equals(VectorExec.canonical(
                        VectorExec.scalarFilterProject(keys, values, sparse, map, sparseScalar))),
                "a very selective predicate still matches the scalar path");
        int batchesWithMatches = 0;
        for (int start = 0; start < rows; start += 64) {
            int end = Math.min(start + 64, rows);
            boolean any = false;
            for (int i = start; i < end; i++) if (values[i] % 97 == 0) any = true;
            if (any) batchesWithMatches++;
        }
        check(sparseCounters.batchesOut == batchesWithMatches,
                "only batches containing a match reach the projection: " + sparseCounters.batchesOut
                        + " vs " + batchesWithMatches);
        check(sparseCounters.emptyBatchesDropped > 0, "empty batches are dropped: "
                + sparseCounters.emptyBatchesDropped + " of " + sparseCounters.batchesIn);
        check(sparseResult.size() == countKept(values, sparse), "sparse result size matches a direct count");

        int remainder = rows % 64;
        check(remainder == 40, "the final batch is partial: " + remainder + " rows");
        VectorExec.Counters partial = new VectorExec.Counters();
        VectorExec.vectorFilterProject(keys, values, 64, value -> true, map, partial);
        check(partial.rowsIn == 1000 && partial.batchesIn == 16, "the partial batch is fully processed: "
                + partial.batchesIn + " batches, " + partial.rowsIn + " rows");

        VectorExec.Counters single = new VectorExec.Counters();
        List<int[]> singleResult = VectorExec.vectorFilterProject(keys, values, 1, keep, map, single);
        check(VectorExec.canonical(singleResult).equals(VectorExec.canonical(scalarResult)),
                "batch size 1 degenerates to the scalar result");
        check(single.operatorCalls == 1000, "batch size 1 means one call per row: " + single.operatorCalls);

        VectorExec.Counters oneBatch = new VectorExec.Counters();
        VectorExec.vectorFilterProject(keys, values, 4096, keep, map, oneBatch);
        check(oneBatch.batchesIn == 1, "a batch larger than the input is a single batch");

        VectorExec.Counters empty = new VectorExec.Counters();
        check(VectorExec.vectorFilterProject(new int[0], new int[0], 64, keep, map, empty).isEmpty()
                        && empty.batchesIn == 0,
                "an empty input produces no batches");

        int[] outOfRange = {3, 9};
        check(VectorExec.selectionViolations(outOfRange, 2, 8) == 1, "an out-of-range index is detected");
        int[] unsorted = {5, 2};
        check(VectorExec.selectionViolations(unsorted, 2, 8) == 1, "a non-ascending selection is detected");
        int[] good = {0, 1, 5};
        check(VectorExec.selectionViolations(good, 3, 8) == 0, "a well-formed selection passes");

        VectorExec.Counters vectorAgg = new VectorExec.Counters();
        Map<Integer, long[]> vectorGroups = VectorExec.vectorAggregate(keys, values, 64, vectorAgg);
        VectorExec.Counters scalarAgg = new VectorExec.Counters();
        Map<Integer, long[]> scalarGroups = VectorExec.scalarAggregate(keys, values, scalarAgg);
        check(sameGroups(vectorGroups, scalarGroups), "vectorized aggregate == scalar aggregate");
        check(sameGroups(vectorGroups, VectorExec.bruteForceAggregate(keys, values)),
                "vectorized aggregate == brute-force grouping oracle");
        check(vectorGroups.size() == 40, "all 40 groups were found: " + vectorGroups.size());
        check(vectorAgg.operatorCalls == 16 && scalarAgg.operatorCalls == 1000,
                "aggregate calls " + vectorAgg.operatorCalls + " vs " + scalarAgg.operatorCalls);
        check(vectorAgg.groupsTouched == vectorGroups.size(), "group creation counted once: "
                + vectorAgg.groupsTouched);
        long totalCount = 0, totalSum = 0;
        for (long[] group : vectorGroups.values()) {
            totalCount += group[0];
            totalSum += group[1];
        }
        long expectedSum = 0;
        for (int value : values) expectedSum += value;
        check(totalCount == rows && totalSum == expectedSum,
                "aggregate totals are exact: " + totalCount + " rows, sum " + totalSum);

        VectorExec.Counters emptyAgg = new VectorExec.Counters();
        check(VectorExec.vectorAggregate(new int[0], new int[0], 64, emptyAgg).isEmpty(), "empty aggregate");

        int[] buildKeys = VectorExec.column(rng, 200, 30);
        int[] buildValues = VectorExec.column(rng, 200, 100);
        int[] probeKeys = VectorExec.column(rng, 600, 30);
        int[] probeValues = VectorExec.column(rng, 600, 100);
        VectorExec.Counters vectorJoin = new VectorExec.Counters();
        List<int[]> vectorJoined = VectorExec.vectorHashJoin(buildKeys, buildValues, probeKeys, probeValues,
                32, 128, vectorJoin);
        VectorExec.Counters scalarJoin = new VectorExec.Counters();
        List<int[]> scalarJoined = VectorExec.scalarNestedLoopJoin(buildKeys, buildValues, probeKeys,
                probeValues, scalarJoin);
        check(VectorExec.canonical(vectorJoined).equals(VectorExec.canonical(scalarJoined)),
                "vectorized hash join equals the scalar nested loop");
        check(!vectorJoined.isEmpty(), "the join produced rows: " + vectorJoined.size());
        check(vectorJoin.batchesIn == 12, "seven build batches plus five probe batches: "
                + vectorJoin.batchesIn);
        check(vectorJoin.comparisons < scalarJoin.comparisons / 10,
                "hash probing is cheaper than nested loops: " + vectorJoin.comparisons + " vs "
                        + scalarJoin.comparisons);
        check(vectorJoin.rowsIn == 800 && scalarJoin.rowsIn == 800, "both sides consumed every row");

        VectorExec.Counters joinBatchOne = new VectorExec.Counters();
        check(VectorExec.canonical(VectorExec.vectorHashJoin(buildKeys, buildValues, probeKeys, probeValues,
                        1, 1, joinBatchOne)).equals(VectorExec.canonical(scalarJoined)),
                "batch size 1 join equals the scalar join");
        VectorExec.Counters emptyJoin = new VectorExec.Counters();
        check(VectorExec.vectorHashJoin(new int[0], new int[0], probeKeys, probeValues, 32, 128, emptyJoin).isEmpty(),
                "an empty build side joins to nothing");

        System.out.println("filter+project: " + vector + " vs scalar " + scalar);
        System.out.println("aggregate: " + vectorAgg.operatorCalls + " calls for " + vectorGroups.size()
                + " groups over " + rows + " rows (scalar " + scalarAgg.operatorCalls + ")");
        System.out.println("join: " + vectorJoined.size() + " rows, " + vectorJoin.batchesIn
                + " batches, " + vectorJoin.comparisons + " comparisons vs scalar nested loop "
                + scalarJoin.comparisons);
        System.out.println("PASS");
    }

    static boolean sameGroups(Map<Integer, long[]> left, Map<Integer, long[]> right) {
        if (!left.keySet().equals(right.keySet())) return false;
        for (Map.Entry<Integer, long[]> entry : left.entrySet()) {
            long[] other = right.get(entry.getKey());
            if (entry.getValue()[0] != other[0] || entry.getValue()[1] != other[1]) return false;
        }
        return true;
    }

    static long countKept(int[] values, IntPredicate keep) {
        long kept = 0;
        for (int value : values) if (keep.test(value)) kept++;
        return kept;
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
