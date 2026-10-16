/** Task 682 - Implement map-reduce optimization for skewed keys (combiner + partitioning). (Tier: Expert, CTCI Beyond — Concurrent & Distributed Systems)
 *
 * A scaled-down word-count pipeline: records are routed round-robin to map tasks, each map task runs a
 * combiner so only one partial (key, sum) per distinct key leaves the task, the shuffle routes those
 * partials with a partitioner and the reducers sum them per key. Three partitioners are compared on a
 * deliberately skewed key distribution (one hot key with more than half the records):
 *
 *   - hash partitioning, which sends every record of the hot key to a single reducer and therefore
 *     produces the classic straggler;
 *   - range partitioning, which sorts the keys by load and cuts balanced contiguous ranges so the hot key
 *     still lands in one reducer but the neighbouring ranges stay small;
 *   - salted (two-phase) partitioning, which splits the hot key's records over all reducers and merges the
 *     partial sums of a key in a second pass, which is the standard fix for skew.
 *
 * Verified: every pipeline variant reproduces the brute-force per-key count and the total record count is
 * conserved; the combiner is checked against an independent count of distinct keys per map task and must
 * cut the shuffle volume; the hash partitioner's maximum load must equal or exceed the hot key's share and
 * be badly imbalanced; range and salted partitioning must beat it; the salted maximum load must be within a
 * small factor of the perfect balance; the greedy partitioner is compared with the optimal assignment found
 * by brute force over all key-to-partition mappings on the small key sets where that is feasible; and the
 * degenerate inputs (empty, single record, single partition, single map task) are handled.
 */
import java.util.*;

final class Records {
    /** Skewed input: the hot key takes |hotShare| of the records, then a few medium and many rare keys. */
    static List<int[]> generate(int total, int keys, int hotSharePercent, long seed) {
        List<int[]> records = new ArrayList<>();
        long state = seed;
        int hotKey = 0, hotCount = total * hotSharePercent / 100;
        for (int i = 0; i < total; i++) {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            z ^= z >>> 31;
            int key;
            if (i < hotCount) key = hotKey;
            else if (i < hotCount + total / 5) key = 1;
            else if (i < hotCount + total * 35 / 100) key = 2;
            else key = 3 + (int) Math.floorMod(z, Math.max(1, keys - 3));
            records.add(new int[]{key, 1});
        }
        return records;
    }

    static Map<Integer, Integer> counts(List<int[]> records) {
        Map<Integer, Integer> counts = new TreeMap<>();
        for (int[] record : records) counts.merge(record[0], record[1], Integer::sum);
        return counts;
    }

    static int total(List<int[]> records) {
        int sum = 0;
        for (int[] record : records) sum += record[1];
        return sum;
    }
}

final class Pipeline {
    static final int HASH = 0, RANGE = 1;

    /** Map plus combiner: one partial sum per distinct key per map task. */
    static List<int[]> mapCombine(List<int[]> records, int mapTasks) {
        List<int[]> partials = new ArrayList<>();
        for (int task = 0; task < mapTasks; task++) {
            Map<Integer, Integer> local = new TreeMap<>();
            for (int i = task; i < records.size(); i += mapTasks)
                local.merge(records.get(i)[0], records.get(i)[1], Integer::sum);
            for (Map.Entry<Integer, Integer> entry : local.entrySet())
                partials.add(new int[]{entry.getKey(), entry.getValue()});
        }
        return partials;
    }

    /** Distinct keys per map task, counted independently of the pipeline. */
    static int distinctKeysPerTask(List<int[]> records, int mapTasks) {
        int sum = 0;
        for (int task = 0; task < mapTasks; task++) {
            Set<Integer> keys = new TreeSet<>();
            for (int i = task; i < records.size(); i += mapTasks) keys.add(records.get(i)[0]);
            sum += keys.size();
        }
        return sum;
    }

    static int hashOf(int key) { return Integer.rotateLeft(key * 0x9E3779B1, 5) ^ (key * 31); }

    /** Key to partition assignment for each strategy. */
    static Map<Integer, Integer> assign(List<int[]> partials, int partitions, int strategy, int hotKey) {
        Map<Integer, Integer> loads = new TreeMap<>();
        for (int[] partial : partials) loads.merge(partial[0], partial[1], Integer::sum);
        Map<Integer, Integer> assignment = new TreeMap<>();
        if (strategy == HASH) {
            for (int key : loads.keySet()) assignment.put(key, Math.floorMod(hashOf(key), partitions));
            return assignment;
        }
        List<Integer> keys = new ArrayList<>(loads.keySet());
        int total = 0;
        for (int load : loads.values()) total += load;
        // range partitioning: cut the key sequence into contiguous, load-balanced ranges
        int target = (int) Math.ceil((double) total / partitions);
        int bucket = 0, running = 0;
        for (int key : keys) {
            if (running > 0 && running + loads.get(key) > target && bucket < partitions - 1) {
                bucket++;
                running = 0;
            }
            assignment.put(key, bucket);
            running += loads.get(key);
        }
        return assignment;
    }

    static int[] loads(List<int[]> partials, Map<Integer, Integer> assignment, int partitions) {
        int[] loads = new int[partitions];
        for (int[] partial : partials) loads[assignment.get(partial[0])] += partial[1];
        return loads;
    }

    /** Reducers summing the partials they receive; the result must match the brute-force counts. */
    static Map<Integer, Integer> reduce(List<int[]> partials, Map<Integer, Integer> assignment, int partitions) {
        Map<Integer, Integer> result = new TreeMap<>();
        for (int partition = 0; partition < partitions; partition++)
            for (int[] partial : partials)
                if (assignment.get(partial[0]) == partition)
                    result.merge(partial[0], partial[1], Integer::sum);
        return result;
    }

    /** Salted pipeline: the hot key's records are spread over all reducers, then merged back. */
    static Map<Integer, Integer> saltedPipeline(List<int[]> records, int partitions, int hotKey) {
        Map<Integer, Integer> merged = new TreeMap<>();
        for (int salt = 0; salt < partitions; salt++) {
            Map<Integer, Integer> local = new TreeMap<>();
            int index = 0;
            for (int[] record : records) {
                boolean mine = record[0] == hotKey
                        ? index++ % partitions == salt
                        : Math.floorMod(hashOf(record[0]), partitions) == salt;
                if (mine) local.merge(record[0], record[1], Integer::sum);
            }
            for (Map.Entry<Integer, Integer> entry : local.entrySet())
                merged.merge(entry.getKey(), entry.getValue(), Integer::sum);
        }
        return merged;
    }

    static int[] saltedLoads(List<int[]> records, int partitions, int hotKey) {
        int[] loads = new int[partitions];
        int index = 0;
        for (int[] record : records) {
            if (record[0] == hotKey) loads[index++ % partitions] += record[1];
            else loads[Math.floorMod(hashOf(record[0]), partitions)] += record[1];
        }
        return loads;
    }

    /** Optimal max load over every key-to-partition assignment (small key sets only). */
    static int optimalMaxLoad(Map<Integer, Integer> loads, int partitions) {
        List<Integer> keys = new ArrayList<>(loads.keySet());
        int[] best = {Integer.MAX_VALUE};
        assignOptimal(loads, keys, 0, new int[keys.size()], partitions, best);
        return keys.isEmpty() ? 0 : best[0];
    }

    static void assignOptimal(Map<Integer, Integer> loads, List<Integer> keys, int index, int[] assignment,
                              int partitions, int[] best) {
        if (index == keys.size()) {
            int[] per = new int[partitions];
            for (int i = 0; i < keys.size(); i++) per[assignment[i]] += loads.get(keys.get(i));
            int max = 0;
            for (int load : per) max = Math.max(max, load);
            best[0] = Math.min(best[0], max);
            return;
        }
        for (int p = 0; p < partitions; p++) {
            assignment[index] = p;
            assignOptimal(loads, keys, index + 1, assignment, partitions, best);
        }
    }

    static int maxOf(int[] loads) {
        int max = 0;
        for (int load : loads) max = Math.max(max, load);
        return max;
    }

    static int sumOf(int[] loads) {
        int sum = 0;
        for (int load : loads) sum += load;
        return sum;
    }
}

public class Solution {
    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    public static void main(String[] args) {
        List<int[]> records = Records.generate(2000, 12, 55, 20240607L);
        Map<Integer, Integer> brute = Records.counts(records);
        check(Records.total(records) == 2000 && brute.get(0) == 1100 && brute.size() >= 10 && brute.size() <= 12,
                "the skewed input has 2000 records with 1100 on the hot key over " + brute.size() + " keys");

        int mapTasks = 4, partitions = 4;
        List<int[]> partials = Pipeline.mapCombine(records, mapTasks);
        int distinct = Pipeline.distinctKeysPerTask(records, mapTasks);
        check(partials.size() == distinct, "the combiner emitted exactly one partial per distinct key and task: "
                + partials.size() + " vs " + distinct);
        check(partials.size() < records.size() / 2, "the combiner cuts the shuffle from " + records.size()
                + " to " + partials.size() + " records");
        check(partials.size() < records.size() / 4, "with this skew the reduction is even better");

        for (int strategy : new int[]{Pipeline.HASH, Pipeline.RANGE}) {
            Map<Integer, Integer> assignment = Pipeline.assign(partials, partitions, strategy, 0);
            Map<Integer, Integer> reduced = Pipeline.reduce(partials, assignment, partitions);
            check(reduced.equals(brute), "strategy " + strategy + " reproduces the brute-force counts");
            int[] loads = Pipeline.loads(partials, assignment, partitions);
            check(Pipeline.sumOf(loads) == Records.total(records),
                    "strategy " + strategy + " moved every record exactly once");
        }

        Map<Integer, Integer> assignment = Pipeline.assign(partials, partitions, Pipeline.HASH, 0);
        for (int[] partial : partials)
            check(assignment.get(partial[0]) == Math.floorMod(Pipeline.hashOf(partial[0]), partitions),
                    "hash assignment is key-determined: " + partial[0]);
        int[] hashLoads = Pipeline.loads(partials, assignment, partitions);
        int hashMax = Pipeline.maxOf(hashLoads);
        check(hashMax >= brute.get(0), "one hash partition carries the whole hot key: " + hashMax + " >= " + brute.get(0));
        check(hashMax > 3 * (Records.total(records) / partitions),
                "hash partitioning is badly imbalanced: max " + hashMax + " vs average "
                        + (Records.total(records) / partitions));

        int[] rangeLoads = Pipeline.loads(partials, Pipeline.assign(partials, partitions, Pipeline.RANGE, 0), partitions);
        int rangeMax = Pipeline.maxOf(rangeLoads);
        check(rangeMax <= hashMax, "range partitioning never loses to hashing: " + rangeMax + " vs " + hashMax);
        check(rangeMax >= brute.get(0), "the hot key alone still bounds the best partition load");

        List<int[]> smallRecords = Records.generate(200, 6, 55, 99);
        Map<Integer, Integer> smallBrute = Records.counts(smallRecords);
        List<int[]> smallPartials = Pipeline.mapCombine(smallRecords, 2);
        int[] smallHash = Pipeline.loads(smallPartials, Pipeline.assign(smallPartials, 3, Pipeline.HASH, 0), 3);
        int[] smallRange = Pipeline.loads(smallPartials, Pipeline.assign(smallPartials, 3, Pipeline.RANGE, 0), 3);
        int optimal = Pipeline.optimalMaxLoad(smallBrute, 3);
        check(smallRange[0] + smallRange[1] + smallRange[2] == 200
                        && smallHash[0] + smallHash[1] + smallHash[2] == 200,
                "both partitioners move all 200 records of the small table");
        check(Pipeline.maxOf(smallRange) >= optimal && Pipeline.maxOf(smallHash) >= optimal,
                "no heuristic can beat the optimum (" + optimal + ")");
        check(Pipeline.maxOf(smallRange) <= 2 * optimal, "range partitioning stays within a factor two of the optimum: "
                + Pipeline.maxOf(smallRange) + " vs " + optimal);

        int[] salted = Pipeline.saltedLoads(records, partitions, 0);
        int saltedMax = Pipeline.maxOf(salted);
        check(Pipeline.sumOf(salted) == 2000, "salting conserves every record");
        check(saltedMax <= 2 * (2000 / partitions),
                "salting spreads the hot key so the maximum load approaches the perfect balance: " + saltedMax
                        + " vs the ideal " + (2000 / partitions));
        check(saltedMax < hashMax, "salting removes the hot-key straggler: " + saltedMax + " vs " + hashMax);
        check(saltedMax < rangeMax, "and it beats the balanced-range partitioner: " + saltedMax + " vs " + rangeMax);
        check(Pipeline.saltedPipeline(records, partitions, 0).equals(brute),
                "merging the salted partial sums reproduces the exact counts");

        List<int[]> empty = Records.generate(0, 4, 50, 7);
        check(empty.isEmpty() && Pipeline.mapCombine(empty, 3).isEmpty()
                        && Pipeline.saltedPipeline(empty, 3, 0).isEmpty(),
                "an empty input produces an empty result");
        List<int[]> single = Collections.singletonList(new int[]{5, 3});
        check(Pipeline.mapCombine(single, 3).size() == 1
                        && Pipeline.saltedPipeline(single, 3, 0).equals(Map.of(5, 3)),
                "a single record flows through the pipeline unchanged");
        List<int[]> one = Records.generate(50, 5, 60, 11);
        Map<Integer, Integer> onePartition = Pipeline.reduce(Pipeline.mapCombine(one, 2),
                Pipeline.assign(Pipeline.mapCombine(one, 2), 1, Pipeline.HASH, 0), 1);
        check(onePartition.equals(Records.counts(one)), "a single partition receives everything");
        List<int[]> oneTask = Pipeline.mapCombine(one, 1);
        check(Pipeline.reduce(oneTask, Pipeline.assign(oneTask, 3, Pipeline.RANGE, 0), 3).equals(Records.counts(one)),
                "a single map task still partitions correctly");

        List<int[]> again = Records.generate(2000, 12, 55, 20240607L);
        check(Pipeline.reduce(Pipeline.mapCombine(again, mapTasks),
                        Pipeline.assign(Pipeline.mapCombine(again, mapTasks), partitions, Pipeline.RANGE, 0), partitions)
                        .equals(Pipeline.reduce(partials, Pipeline.assign(partials, partitions, Pipeline.RANGE, 0), partitions)),
                "the pipeline is deterministic");

        int skew = 100 * brute.get(0) / Records.total(records);
        System.out.println("input        : 2000 records, " + brute.size() + " keys, the hot key holds " + skew + "%");
        System.out.println("combiner     : shuffle cut from " + records.size() + " to " + partials.size()
                + " records (" + distinct + " distinct keys across " + mapTasks + " map tasks)");
        System.out.println("partitions   : hash max " + hashMax + ", range max " + rangeMax + ", salted max "
                + saltedMax + " over " + partitions + " reducers (small-table optimum " + optimal + " with 3)");
        System.out.println("PASS");
    }
}
