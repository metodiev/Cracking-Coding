/** Task 657 - Implement distributed join shuffle simulation in map-reduce. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Two classic map-reduce equi-joins over (key, value) record streams, simulated end to end:
 *
 *   repartition join - every record of both inputs is emitted by a mapper, hash partitioned on the
 *                      key into R reducers (shuffle cost = |L| + |R| records) and the matching key
 *                      groups are joined inside each reducer;
 *   broadcast join   - the smaller side is replicated to every mapper (cost = |small| * mappers
 *                      copies) and only the large side is scanned, so nothing enters the shuffle;
 *                      the planner picks the smaller side automatically.
 *
 * The simulation accounts for the real cost counters - mapper emissions, records crossing the
 * shuffle, per-partition record counts (skew), and replication units - and is verified against an
 * in-memory nested loop join: result *multisets* must match exactly, for both plans, for every
 * reducer/mapper count. Further invariants: the result is independent of the number of reducers and
 * mappers; each mapper emits each input record exactly once; partition counts sum to the shuffled
 * total; a fully skewed key sends everything to one reducer while the join stays correct; duplicate
 * keys multiply out (|L_k| * |R_k|); empty inputs and non-matching keys produce an empty result; and
 * two runs of the same plan produce identical partition layouts (the hash is deterministic).
 */
import java.util.*;

final class Record {
    final int key;
    final int value;
    final int tag;        // 0 = left input, 1 = right input
    final int sequence;   // input order, keeps the output deterministic

    Record(int key, int value, int tag, int sequence) {
        this.key = key;
        this.value = value;
        this.tag = tag;
        this.sequence = sequence;
    }
}

final class ShuffleStats {
    int mapperEmissions;
    int shuffledRecords;
    int replicatedUnits;
    int[] partitionCounts = new int[0];

    @Override
    public String toString() {
        return "emissions=" + mapperEmissions + " shuffled=" + shuffledRecords
                + " replicated=" + replicatedUnits + " partitions=" + Arrays.toString(partitionCounts);
    }
}

final class MapReduceJoin {
    /**
     * Repartition (shuffle) join: map both sides, hash partition by key, then join per reducer.
     * Output rows are {key, leftValue, rightValue}.
     */
    static List<int[]> repartitionJoin(int[] leftKeys, int[] leftValues, int[] rightKeys, int[] rightValues,
                                       int reducers, ShuffleStats stats) {
        if (reducers < 1) throw new IllegalArgumentException("need at least one reducer");
        List<List<Record>> partitions = new ArrayList<>();
        for (int r = 0; r < reducers; r++) partitions.add(new ArrayList<>());
        int sequence = 0;
        for (int i = 0; i < leftKeys.length; i++) {
            Record record = new Record(leftKeys[i], leftValues[i], 0, sequence++);
            partitions.get(partitionOf(record.key, reducers)).add(record);
        }
        for (int i = 0; i < rightKeys.length; i++) {
            Record record = new Record(rightKeys[i], rightValues[i], 1, sequence++);
            partitions.get(partitionOf(record.key, reducers)).add(record);
        }
        stats.mapperEmissions = sequence;
        stats.shuffledRecords = sequence;
        stats.partitionCounts = new int[reducers];
        List<int[]> out = new ArrayList<>();
        for (int r = 0; r < reducers; r++) {
            List<Record> partition = partitions.get(r);
            stats.partitionCounts[r] = partition.size();
            // grouping by (key, input order) puts lefts before rights and preserves input order
            partition.sort((a, b) -> a.key != b.key ? Integer.compare(a.key, b.key)
                    : a.tag != b.tag ? Integer.compare(a.tag, b.tag) : Integer.compare(a.sequence, b.sequence));
            int at = 0;
            while (at < partition.size()) {
                int key = partition.get(at).key;
                int end = at;
                List<Integer> lefts = new ArrayList<>();
                List<Integer> rights = new ArrayList<>();
                while (end < partition.size() && partition.get(end).key == key) {
                    Record record = partition.get(end);
                    if (record.tag == 0) lefts.add(record.value); else rights.add(record.value);
                    end++;
                }
                for (int left : lefts) for (int right : rights) out.add(new int[]{key, left, right});
                at = end;
            }
        }
        return out;
    }

    /**
     * Broadcast join: the smaller table is replicated to every mapper and each mapper joins its split
     * of the large table locally, so no record enters the shuffle.
     */
    static List<int[]> broadcastJoin(int[] leftKeys, int[] leftValues, int[] rightKeys, int[] rightValues,
                                     int mappers, ShuffleStats stats) {
        if (mappers < 1) throw new IllegalArgumentException("need at least one mapper");
        boolean broadcastLeft = leftKeys.length <= rightKeys.length;
        int[] smallKeys = broadcastLeft ? leftKeys : rightKeys;
        int[] smallValues = broadcastLeft ? leftValues : rightValues;
        int[] bigKeys = broadcastLeft ? rightKeys : leftKeys;
        int[] bigValues = broadcastLeft ? rightValues : leftValues;
        Map<Integer, List<Integer>> smallTable = new LinkedHashMap<>();
        for (int i = 0; i < smallKeys.length; i++) {
            smallTable.computeIfAbsent(smallKeys[i], key -> new ArrayList<>()).add(smallValues[i]);
        }
        stats.mapperEmissions = bigKeys.length;
        stats.shuffledRecords = 0;                       // map side join: nothing crosses the shuffle
        stats.replicatedUnits = smallKeys.length * mappers;
        stats.partitionCounts = new int[mappers];
        List<int[]> out = new ArrayList<>();
        int perMapper = (bigKeys.length + mappers - 1) / mappers;
        for (int mapper = 0; mapper < mappers; mapper++) {
            int from = mapper * perMapper;
            int to = Math.min(bigKeys.length, from + perMapper);
            stats.partitionCounts[mapper] = Math.max(0, to - from);
            for (int i = from; i < to; i++) {
                List<Integer> matches = smallTable.get(bigKeys[i]);
                if (matches == null) continue;
                for (int match : matches) {
                    out.add(broadcastLeft ? new int[]{bigKeys[i], match, bigValues[i]}
                            : new int[]{bigKeys[i], bigValues[i], match});
                }
            }
        }
        out.sort(Comparator.comparingInt((int[] row) -> row[0]).thenComparingInt(row -> row[1]).thenComparingInt(row -> row[2]));
        return out;
    }

    static int partitionOf(int key, int reducers) {
        return Math.floorMod(mix(key), reducers);
    }

    /** Deterministic key hash (splitmix64 finaliser over the key). */
    static int mix(int key) {
        long z = key * 0x9E3779B97F4A7C15L + 0x2545F4914F6CDD1DL;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z = z ^ (z >>> 31);
        return (int) (z ^ (z >>> 32));
    }

    /** In-memory nested loop oracle; output sorted canonically. */
    static List<int[]> bruteForce(int[] leftKeys, int[] leftValues, int[] rightKeys, int[] rightValues) {
        List<int[]> out = new ArrayList<>();
        for (int i = 0; i < leftKeys.length; i++) {
            for (int j = 0; j < rightKeys.length; j++) {
                if (leftKeys[i] == rightKeys[j]) out.add(new int[]{leftKeys[i], leftValues[i], rightValues[j]});
            }
        }
        out.sort(Comparator.comparingInt((int[] row) -> row[0]).thenComparingInt(row -> row[1]).thenComparingInt(row -> row[2]));
        return out;
    }

    static String canonical(List<int[]> rows) {
        List<int[]> sorted = new ArrayList<>(rows);
        sorted.sort(Comparator.comparingInt((int[] row) -> row[0]).thenComparingInt(row -> row[1]).thenComparingInt(row -> row[2]));
        StringBuilder sb = new StringBuilder();
        for (int[] row : sorted) sb.append(row[0]).append(':').append(row[1]).append(':').append(row[2]).append(';');
        return sb.toString();
    }
}

public class Solution {
    private static long state = 0x6A09E667F3BCC909L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), bound);
    }

    static int[] randomInts(int size, int domain) {
        int[] out = new int[size];
        for (int i = 0; i < size; i++) out[i] = nextInt(domain);
        return out;
    }

    static int[] filled(int size, int value) {
        int[] out = new int[size];
        Arrays.fill(out, value);
        return out;
    }

    public static void main(String[] args) {
        // --- fixed dataset, both plans against the nested loop oracle -----------
        int[] leftKeys = {1, 2, 2, 4, 7, 9, 9, 9};
        int[] leftValues = {10, 20, 21, 40, 70, 90, 91, 92};
        int[] rightKeys = {2, 3, 9, 9, 11, 2};
        int[] rightValues = {200, 300, 900, 901, 1100, 201};
        String oracleRows = MapReduceJoin.canonical(MapReduceJoin.bruteForce(leftKeys, leftValues, rightKeys, rightValues));
        check(oracleRows.split(";").length == 10, "the oracle finds ten matching pairs");

        ShuffleStats repartitionStats = new ShuffleStats();
        List<int[]> repartition = MapReduceJoin.repartitionJoin(leftKeys, leftValues, rightKeys, rightValues, 4, repartitionStats);
        check(MapReduceJoin.canonical(repartition).equals(
                MapReduceJoin.canonical(MapReduceJoin.bruteForce(leftKeys, leftValues, rightKeys, rightValues))),
                "repartition join matches the oracle");
        check(repartitionStats.mapperEmissions == leftKeys.length + rightKeys.length,
                "every input record is emitted once, got " + repartitionStats.mapperEmissions);
        check(repartitionStats.shuffledRecords == leftKeys.length + rightKeys.length,
                "the repartition join shuffles every record");
        int partitionSum = 0;
        for (int count : repartitionStats.partitionCounts) partitionSum += count;
        check(partitionSum == repartitionStats.shuffledRecords, "partition counts account for every shuffled record");
        check(repartitionStats.partitionCounts.length == 4, "four reducers produce four partitions");

        ShuffleStats broadcastStats = new ShuffleStats();
        List<int[]> broadcast = MapReduceJoin.broadcastJoin(leftKeys, leftValues, rightKeys, rightValues, 3, broadcastStats);
        check(MapReduceJoin.canonical(broadcast).equals(
                MapReduceJoin.canonical(MapReduceJoin.bruteForce(leftKeys, leftValues, rightKeys, rightValues))),
                "broadcast join matches the oracle");
        check(broadcastStats.shuffledRecords == 0, "the broadcast join shuffles nothing");
        check(broadcastStats.replicatedUnits == rightKeys.length * 3,
                "the smaller side is replicated to every mapper, got " + broadcastStats.replicatedUnits);
        check(broadcastStats.mapperEmissions == leftKeys.length, "only the large side is scanned");
        ShuffleStats oneMapper = new ShuffleStats();
        MapReduceJoin.broadcastJoin(leftKeys, leftValues, rightKeys, rightValues, 1, oneMapper);
        check(oneMapper.replicatedUnits + oneMapper.shuffledRecords
                        < repartitionStats.shuffledRecords + repartitionStats.replicatedUnits,
                "with a single mapper the broadcast copies less than the shuffle moves");
        ShuffleStats manyMappers = new ShuffleStats();
        MapReduceJoin.broadcastJoin(leftKeys, leftValues, rightKeys, rightValues, 8, manyMappers);
        check(manyMappers.replicatedUnits > repartitionStats.shuffledRecords,
                "with eight mappers replication costs more than the shuffle: " + manyMappers.replicatedUnits
                        + " > " + repartitionStats.shuffledRecords);

        // --- the result must not depend on the parallelism ----------------------
        for (int reducers = 1; reducers <= 6; reducers++) {
            ShuffleStats stats = new ShuffleStats();
            check(MapReduceJoin.canonical(MapReduceJoin.repartitionJoin(leftKeys, leftValues, rightKeys, rightValues,
                    reducers, stats)).equals(oracleRows), "repartition result with " + reducers + " reducers");
            check(stats.partitionCounts.length == reducers, "partition count with " + reducers + " reducers");
        }
        for (int mappers = 1; mappers <= 5; mappers++) {
            ShuffleStats stats = new ShuffleStats();
            check(MapReduceJoin.canonical(MapReduceJoin.broadcastJoin(leftKeys, leftValues, rightKeys, rightValues,
                    mappers, stats)).equals(oracleRows), "broadcast result with " + mappers + " mappers");
        }

        // --- duplicate keys multiply out ---------------------------------------
        int[] dupLeftKeys = filled(4, 5);
        int[] dupRightKeys = filled(3, 5);
        ShuffleStats dupStats = new ShuffleStats();
        int dupRows = MapReduceJoin.repartitionJoin(dupLeftKeys, new int[]{1, 2, 3, 4},
                dupRightKeys, new int[]{7, 8, 9}, 3, dupStats).size();
        check(dupRows == 12, "|L_k| * |R_k| = 12 output rows, got " + dupRows);
        check(dupStats.partitionCounts[MapReduceJoin.partitionOf(5, 3)] == 7,
                "a single key lands in exactly one reducer");

        // --- total skew: everything in one partition, join still correct --------
        int[] skewLeft = filled(50, 42);
        int[] skewRight = filled(50, 42);
        ShuffleStats skewStats = new ShuffleStats();
        List<int[]> skewRows = MapReduceJoin.repartitionJoin(skewLeft, filled(50, 1), skewRight, filled(50, 2), 8, skewStats);
        check(skewRows.size() == 2500, "fully skewed join produces 2500 rows, got " + skewRows.size());
        int nonEmpty = 0, maxPartition = 0;
        for (int count : skewStats.partitionCounts) {
            if (count > 0) nonEmpty++;
            maxPartition = Math.max(maxPartition, count);
        }
        check(nonEmpty == 1 && maxPartition == 100, "all 100 records land in one of the eight partitions");

        // --- deterministic layout ----------------------------------------------
        ShuffleStats again = new ShuffleStats();
        MapReduceJoin.repartitionJoin(skewLeft, filled(50, 1), skewRight, filled(50, 2), 8, again);
        check(Arrays.equals(again.partitionCounts, skewStats.partitionCounts), "partition layout is deterministic");

        // --- empty and non-matching inputs -------------------------------------
        ShuffleStats emptyStats = new ShuffleStats();
        check(MapReduceJoin.repartitionJoin(new int[0], new int[0], new int[0], new int[0], 3, emptyStats).isEmpty(),
                "empty inputs -> empty result");
        check(emptyStats.shuffledRecords == 0 && emptyStats.partitionCounts.length == 3,
                "empty inputs still produce a partition layout");
        ShuffleStats noMatchStats = new ShuffleStats();
        check(MapReduceJoin.repartitionJoin(new int[]{1, 2}, new int[]{1, 2}, new int[]{3, 4}, new int[]{3, 4},
                4, noMatchStats).isEmpty(), "non-matching keys -> empty result");
        int[] onlyLeftKeys = {5};
        ShuffleStats oneSided = new ShuffleStats();
        check(MapReduceJoin.repartitionJoin(onlyLeftKeys, new int[]{9}, new int[0], new int[0], 2, oneSided).isEmpty(),
                "one sided input -> empty result");
        ShuffleStats oneSidedBroadcast = new ShuffleStats();
        check(MapReduceJoin.broadcastJoin(new int[0], new int[0], onlyLeftKeys, new int[]{9}, 2, oneSidedBroadcast)
                .isEmpty(), "broadcast with an empty side -> empty result");

        // --- random sweep against the oracle -----------------------------------
        int sweepRows = 0;
        for (int trial = 0; trial < 60; trial++) {
            int[] lk = randomInts(1 + nextInt(30), 12);
            int[] lv = randomInts(lk.length, 1000);
            int[] rk = randomInts(1 + nextInt(40), 12);
            int[] rv = randomInts(rk.length, 1000);
            String oracle = MapReduceJoin.canonical(MapReduceJoin.bruteForce(lk, lv, rk, rv));
            ShuffleStats stats = new ShuffleStats();
            String got = MapReduceJoin.canonical(MapReduceJoin.repartitionJoin(lk, lv, rk, rv, 1 + nextInt(6), stats));
            check(got.equals(oracle), "sweep trial " + trial + " repartition");
            check(stats.shuffledRecords == lk.length + rk.length, "sweep trial " + trial + " shuffle accounting");
            int sum = 0;
            for (int count : stats.partitionCounts) sum += count;
            check(sum == stats.shuffledRecords, "sweep trial " + trial + " partition accounting");
            ShuffleStats broadcastSweep = new ShuffleStats();
            int mappers = 1 + nextInt(4);
            String gotBroadcast = MapReduceJoin.canonical(
                    MapReduceJoin.broadcastJoin(lk, lv, rk, rv, mappers, broadcastSweep));
            check(gotBroadcast.equals(oracle), "sweep trial " + trial + " broadcast");
            check(broadcastSweep.shuffledRecords == 0, "sweep trial " + trial + " broadcast shuffles nothing");
            check(broadcastSweep.replicatedUnits == Math.min(lk.length, rk.length) * mappers,
                    "sweep trial " + trial + " replication is charged per mapper");
            check(broadcastSweep.mapperEmissions == Math.max(lk.length, rk.length),
                    "sweep trial " + trial + " scans only the large side");
            sweepRows += got.split(";").length - 1;
        }
        check(sweepRows > 0, "the sweep really produced joined rows: " + sweepRows);

        // --- skew metric on a uniform workload ---------------------------------
        int[] uniformLeft = randomInts(600, 64);
        int[] uniformRight = randomInts(600, 64);
        ShuffleStats uniformStats = new ShuffleStats();
        MapReduceJoin.repartitionJoin(uniformLeft, filled(600, 1), uniformRight, filled(600, 2), 8, uniformStats);
        int average = 1200 / 8;
        int worst = 0;
        for (int count : uniformStats.partitionCounts) worst = Math.max(worst, count);
        check(worst < 3 * average, "uniform keys stay balanced: worst " + worst + " vs average " + average);

        // --- rejected parallelism ----------------------------------------------
        boolean rejected = false;
        try {
            MapReduceJoin.repartitionJoin(leftKeys, leftValues, rightKeys, rightValues, 0, new ShuffleStats());
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "zero reducers must be rejected");
        rejected = false;
        try {
            MapReduceJoin.broadcastJoin(leftKeys, leftValues, rightKeys, rightValues, 0, new ShuffleStats());
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "zero mappers must be rejected");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
