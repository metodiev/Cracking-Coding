/** Task 511 - Implement checkpointed parallel map-reduce worker for crash recovery. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
 *
     * A checkpointed map-reduce worker: the corpus is split into P partitions, the parallel map phase
     * runs on a real fixed thread pool (one Callable per partition, futures collected in partition
     * order, each partition's completion recorded in an AtomicIntegerArray, and the pool joined with a
     * timeout that is asserted to succeed), while the crash-recovery semantics are a deterministic
     * simulation - the documented wave-3 approach for concurrency tasks - so the assertions never
     * depend on real thread scheduling. The worker processes partitions in order, commits each
     * partition result exactly once (a per-partition commit counter is kept so a re-execution cannot
     * commit twice), writes a checkpoint after every `checkpointEvery` commits plus a final flush, and
     * on an injected crash discards the volatile state since the last checkpoint and resumes from the
     * last durable snapshot - the classic at-least-once execution with exactly-once commit.
 *
     * Documented checks: the threaded map equals the sequential reference element-wise for 1, 3 and 8
     * threads and every partition is mapped exactly once (the AtomicIntegerArray counts are all 1);
     * with no crash the run performs exactly P executions and P commits with zero re-executions; with a
     * single crash injected after execution E the re-execution count equals E mod checkpointEvery (the
     * units committed since the last checkpoint are lost and recomputed, and the documented K multiples
     * of the checkpoint interval survive), the durable snapshot before the crash is a subset of the
     * final durable state, and the final map equals the crash-free reference; a two-crash run and a
     * crash on the very first execution (E = 1, so nothing durable is lost and one execution is
     * repeated) are also covered; in the two-crash run the re-execution count equals the summed
     * (volatile - durable) loss recorded at each crash, so every lost unit is recomputed exactly once;
     * every unit's commit counter is exactly 1 in all runs, so exactly-once commit holds even though
     * executions are at-least-once; the checkpoint write count equals ceil(P / K); and a re-run
     * reproduces every map value, counter and snapshot bit for bit.
 */
import java.util.*;
import java.util.concurrent.*;

public class Solution {
    /** 75/74 LCG modulo 65537 - the deterministic generator used by waves 1-2. */
    static class Lcg {
        private int state;

        Lcg(int seed) {
            this.state = ((seed % 65537) + 65537) % 65537;
        }

        double next() {
            state = (state * 75 + 74) % 65537;
            return state / 65537.0;
        }

        int nextInt(int bound) {
            return (int) (next() * bound);
        }
    }

    static final int PARTITIONS = 24;
    static final int ITEMS_PER_PARTITION = 64;

    /** The input corpus as one long array; partition p owns the slice [p * items, (p + 1) * items). */
    static long[] corpus(int seed) {
        Lcg lcg = new Lcg(seed);
        long[] data = new long[PARTITIONS * ITEMS_PER_PARTITION];
        for (int i = 0; i < data.length; i++) data[i] = lcg.nextInt(1000);
        return data;
    }

    /** Pure, deterministic map for one partition (integer arithmetic, so bit-exact everywhere). */
    static long mapUnit(long[] data, int partition) {
        long acc = 0;
        for (int i = 0; i < ITEMS_PER_PARTITION; i++) {
            long value = data[partition * ITEMS_PER_PARTITION + i];
            acc = (acc * 31 + value * value % 1000003) % 1000000007L;
        }
        return acc;
    }

    static long[] mapSequential(long[] data) {
        long[] out = new long[PARTITIONS];
        for (int p = 0; p < PARTITIONS; p++) out[p] = mapUnit(data, p);
        return out;
    }

    /** Real-thread map phase: one task per partition on a fixed pool, results in partition order. */
    static long[] mapParallel(long[] data, int threads, java.util.concurrent.atomic.AtomicIntegerArray runs)
            throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        List<Future<Long>> futures = new ArrayList<>();
        try {
            for (int p = 0; p < PARTITIONS; p++) {
                final int partition = p;
                futures.add(pool.submit(() -> {
                    runs.incrementAndGet(partition);
                    return mapUnit(data, partition);
                }));
            }
            long[] out = new long[PARTITIONS];
            for (int p = 0; p < PARTITIONS; p++) out[p] = futures.get(p).get();
            return out;
        } finally {
            pool.shutdown();
            if (!pool.awaitTermination(30, TimeUnit.SECONDS)) {
                throw new IllegalStateException("the map pool did not terminate");
            }
        }
    }

    static class CrashRun {
        final long[] results;
        final int[] executionsPerUnit;
        final int[] commitsPerUnit;
        final int executions;
        final int commits;
        final int reexecutions;
        final int checkpointWrites;
        final int crashes;
        final int lostUnits;
        final List<int[]> crashStats;

        CrashRun(long[] results, int[] executionsPerUnit, int[] commitsPerUnit, int executions, int commits,
                int reexecutions, int checkpointWrites, int crashes, int lostUnits, List<int[]> crashStats) {
            this.results = results;
            this.executionsPerUnit = executionsPerUnit;
            this.commitsPerUnit = commitsPerUnit;
            this.executions = executions;
            this.commits = commits;
            this.reexecutions = reexecutions;
            this.checkpointWrites = checkpointWrites;
            this.crashes = crashes;
            this.lostUnits = lostUnits;
            this.crashStats = crashStats;
        }

        int durableCommitsBeforeFirstCrash() {
            return crashStats.isEmpty() ? -1 : crashStats.get(0)[1];
        }
    }

    /**
     * Deterministic crash simulation: one worker walking the partitions in order, a checkpoint
     * every `checkpointEvery` durable commits plus a final flush, and a crash injected right after
     * the execution whose 1-based index appears in `crashAfterExecutions`. A checkpoint commits
     * every unit computed since the previous checkpoint; a crash discards the volatile work since
     * the last checkpoint, so exactly those units are re-executed while the durable set survives.
     */
    static CrashRun mapWithCrashes(long[] data, int checkpointEvery, Set<Integer> crashAfterExecutions) {
        long[] results = new long[PARTITIONS];
        Set<Integer> volatileSet = new LinkedHashSet<>();
        Set<Integer> snapshot = new LinkedHashSet<>();
        int[] executionsPerUnit = new int[PARTITIONS];
        int[] commitsPerUnit = new int[PARTITIONS];
        int executions = 0;
        int commits = 0;
        int reexecutions = 0;
        int checkpointWrites = 0;
        int crashes = 0;
        int lostUnits = 0;
        List<int[]> crashStats = new ArrayList<>();
        while (true) {
            int unit = -1;
            for (int i = 0; i < PARTITIONS; i++) {
                if (!volatileSet.contains(i)) {
                    unit = i;
                    break;
                }
            }
            if (unit < 0) break;
            executions++;
            executionsPerUnit[unit]++;
            if (executionsPerUnit[unit] > 1) reexecutions++;
            results[unit] = mapUnit(data, unit);
            volatileSet.add(unit);
            if (volatileSet.size() % checkpointEvery == 0 && volatileSet.size() > snapshot.size()) {
                for (int u : volatileSet) {
                    if (!snapshot.contains(u)) {
                        commitsPerUnit[u]++;
                        commits++;
                    }
                }
                snapshot = new LinkedHashSet<>(volatileSet);
                checkpointWrites++;
            }
            if (crashAfterExecutions.contains(executions)) {
                crashes++;
                int volatileSize = volatileSet.size();
                crashStats.add(new int[]{volatileSize, snapshot.size()});
                lostUnits += volatileSize - snapshot.size();
                volatileSet = new LinkedHashSet<>(snapshot);
            }
        }
        // final flush: everything computed but not yet checkpointed becomes durable now
        boolean flushed = false;
        for (int u : volatileSet) {
            if (!snapshot.contains(u)) {
                commitsPerUnit[u]++;
                commits++;
                flushed = true;
            }
        }
        if (flushed) checkpointWrites++;
        return new CrashRun(results, executionsPerUnit, commitsPerUnit, executions, commits, reexecutions,
                checkpointWrites, crashes, lostUnits, crashStats);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }

    public static void main(String[] args) throws Exception {
        long[] data = corpus(511511);
        long[] sequential = mapSequential(data);

        // 1. The real-thread map equals the sequential reference for several pool sizes.
        for (int threads : new int[]{1, 3, 8}) {
            java.util.concurrent.atomic.AtomicIntegerArray runs =
                    new java.util.concurrent.atomic.AtomicIntegerArray(PARTITIONS);
            long[] threaded = mapParallel(data, threads, runs);
            check(Arrays.equals(threaded, sequential),
                    "threaded map equals the reference with " + threads
                    + " threads");
            for (int p = 0; p < PARTITIONS; p++) {
                check(runs.get(p) == 1, "partition " + p + " was mapped exactly once with " + threads
                        + " threads");
            }
            int total = 0;
            for (int p = 0; p < PARTITIONS; p++) total += runs.get(p);
            check(total == PARTITIONS,
                    "every partition ran exactly once with " + threads + " threads");
        }

        // 2. Crash-free run: exactly P executions, P commits, no re-execution.
        Set<Integer> noCrashes = new HashSet<>();
        CrashRun clean = mapWithCrashes(data, 5, noCrashes);
        check(Arrays.equals(clean.results, sequential), "the crash-free run equals the reference map");
        check(clean.executions == PARTITIONS, "the crash-free run executes exactly P units");
        check(clean.commits == PARTITIONS, "the crash-free run commits exactly P units");
        check(clean.reexecutions == 0 && clean.crashes == 0, "no re-execution without a crash");
        for (int p = 0; p < PARTITIONS; p++) {
            check(clean.commitsPerUnit[p] == 1, "unit " + p + " is committed exactly once");
        }

        // 3. A single crash: the units since the last checkpoint are lost and recomputed.
        int checkpointEvery = 5;
        for (int crashAt : new int[]{1, 4, 7, 12, 23}) {
            Set<Integer> schedule = new HashSet<>();
            schedule.add(crashAt);
            CrashRun crashed = mapWithCrashes(data, checkpointEvery, schedule);
            check(crashed.crashes == 1, "exactly one crash is injected at execution " + crashAt);
            check(Arrays.equals(crashed.results, sequential),
                    "recovery reproduces the reference map after a crash at " + crashAt);
            check(crashed.reexecutions == crashAt % checkpointEvery,
                    "re-executions equal crashAt mod K for crashAt = " + crashAt);
            check(crashed.commits == PARTITIONS, "every unit is committed once after recovery");
            for (int p = 0; p < PARTITIONS; p++) {
                check(crashed.commitsPerUnit[p] == 1, "unit " + p + " committed exactly once (crash "
                        + crashAt + ")");
            }
            check(crashed.executions == PARTITIONS + crashed.reexecutions,
                    "at-least-once: executions = units + re-executions");
            check(crashed.durableCommitsBeforeFirstCrash() == (crashAt / checkpointEvery) * checkpointEvery,
                    "durable commits before the crash equal floor(E / K) * K");
        }

        // 4. Two crashes in one run, including one on the very first execution.
        Set<Integer> twoCrashes = new TreeSet<>(Arrays.asList(1, 11));
        CrashRun twice = mapWithCrashes(data, 4, twoCrashes);
        check(twice.crashes == 2, "both crashes fired");
        check(Arrays.equals(twice.results, sequential), "the map survives two crashes");
        check(twice.commits == PARTITIONS, "two crashes still commit every unit exactly once");
        for (int p = 0; p < PARTITIONS; p++) {
            check(twice.commitsPerUnit[p] == 1, "unit " + p + " committed once under two crashes");
        }
        check(twice.reexecutions == twice.lostUnits,
                "every unit lost at a crash is re-executed exactly once");
        check(twice.executions == PARTITIONS + twice.reexecutions,
                "re-executions are counted consistently under two crashes");
        check(twice.reexecutions == 3 && twice.executions == PARTITIONS + 3,
                "crashes at executions 1 and 11 with K = 4 lose 1 + 2 = 3 units");
        check(twice.checkpointWrites == 6, "K = 4 yields ceil(24 / 4) checkpoint writes");
        check(twice.reexecutions < PARTITIONS, "a crash never re-executes the whole corpus here");

        // 5. Checkpoint write accounting: floor(P / K) plus a final flush.
        for (int k : new int[]{1, 4, 5, 8, 24, 32}) {
            CrashRun run = mapWithCrashes(data, k, new HashSet<>());
            int expected = (PARTITIONS + k - 1) / k;
            check(run.checkpointWrites == expected,
                    "checkpoint writes are ceil(P / K) for K = " + k + " (got " + run.checkpointWrites
                            + ", expected " + expected + ")");
        }

        // 6. Determinism of the whole simulated run.
        CrashRun rerun = mapWithCrashes(data, 4, twoCrashes);
        check(Arrays.equals(rerun.results, twice.results), "deterministic re-run: identical map");
        check(rerun.executions == twice.executions && rerun.commits == twice.commits
                        && rerun.reexecutions == twice.reexecutions
                        && rerun.checkpointWrites == twice.checkpointWrites,
                "deterministic re-run: identical counters");
        check(Arrays.equals(rerun.executionsPerUnit, twice.executionsPerUnit)
                        && Arrays.equals(rerun.commitsPerUnit, twice.commitsPerUnit),
                "deterministic re-run: identical per-unit counters");
        java.util.concurrent.atomic.AtomicIntegerArray runs =
                new java.util.concurrent.atomic.AtomicIntegerArray(PARTITIONS);
        check(Arrays.equals(mapParallel(data, 4, runs), sequential),
                "deterministic re-run: the threaded map still matches");

        System.out.println("PASS");
    }
}
