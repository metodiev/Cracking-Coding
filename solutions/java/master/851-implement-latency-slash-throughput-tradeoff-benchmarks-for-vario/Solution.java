/** Task 851 - Implement latency-slash-throughput tradeoff benchmarks for various consistency levels.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   A replication group of N replicas becomes N independent single-threaded servers, each with a virtual
 *   clock (`freeAt`) and its own FIFO. A consistency level is a (readQuorum, writeQuorum) pair:
 *
 *       EVENTUAL      (1, 1)                stale reads possible
 *       QUORUM        (N/2+1, N/2+1)        R + W > N  =>  quorums always intersect
 *       ALL           (N, N)                every replica touched
 *       SINGLE-LEADER (1, 1) on replica 0   linearisable, but no scale-out
 *
 *   Every operation is dispatched to a rotating quorum of its level. A replica starts serving as soon as
 *   that replica is free (no global barrier: replicas pipeline), and the operation completes when the
 *   slowest member of its quorum finishes, so latency is a straggler (max-of-quorum) statistic. Service
 *   times and inter-arrival times are exponential deviates from fixed-seed splitmix64 streams keyed by
 *   (seed, operation, replica), i.e. all levels are benchmarked on the *same* sample set (a paired design).
 *
 * WHAT IS SIMPLIFIED
 *   No network, disk, consensus protocol, retries, timeouts, failure injection or quorum-formation cost,
 *   no wall-clock time, no threads and perfect clock synchronisation. Scale is expressed through typed
 *   parameters (N, offered rate, operation count); the model keeps exactly the mechanisms that create the
 *   real trade-off: fan-out, per-replica serialisation and straggler waiting.
 *
 * INVARIANTS VERIFIED
 *   Safety: no replica is ever double-booked (all service intervals of a replica are pairwise disjoint),
 *   replica busy time is exactly conserved, achieved throughput never exceeds the analytic capacity
 *   N/(q*mean), and every read quorum of a strong level intersects every write quorum (brute force over
 *   all quorum pairs) while the (1,1) level provably has a disjoint pair and produces stale reads.
 *   Liveness / bounds: throughput saturates within 12% of the analytic capacity; latency grows and
 *   throughput plateaus as offered load rises; mean latency at negligible load matches the closed form
 *   network + mean*H_q (E[max of q i.i.d. exponentials]); at equal utilisation the delay ordering is
 *   ANY < QUORUM < ALL; Little's law L = lambda*W holds to 5%; and the widest level is capped by
 *   E[max of N] while the (1,1) level scales linearly in N.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every benchmark run is bit-for-bit reproducible. */
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

    /** Exponential(mean) deviate: -mean * ln(1 - U), U uniform on (0,1). */
    double nextExponential(double mean) { return -mean * Math.log(1.0 - nextDouble()); }

    static long mix(long a, long b) {
        long h = a * 0x9E3779B97F4A7C15L + b * 0xC2B2AE3D27D4EB4FL;
        h ^= h >>> 29; h *= 0xBF58476D1CE4E5B9L; h ^= h >>> 32;
        return h;
    }
}

/** One replica: a single-threaded server with a virtual clock and an interval log for safety checking. */
final class Replica {
    final int id;
    double freeAt = 0.0;
    long version = 0;
    double busyTime = 0.0;
    double servedWork = 0.0;
    double maxOverlap = 0.0;
    final List<double[]> intervals = new ArrayList<>();

    Replica(int id) { this.id = id; }

    /** Records a service interval and checks it against the previous one: a replica may never be double-booked. */
    void occupy(double start, double finish, double work) {
        if (!intervals.isEmpty()) {
            double[] last = intervals.get(intervals.size() - 1);
            double overlap = last[1] - start;
            if (overlap > maxOverlap) maxOverlap = overlap;
        }
        intervals.add(new double[]{start, finish});
        busyTime += finish - start;
        servedWork += work;
        freeAt = finish;
    }
}

/** A consistency level: quorum sizes plus the "all traffic to one leader" special case. */
final class ConsistencyLevel {
    final String name;
    final int readQuorum;
    final int writeQuorum;
    final boolean leaderSerialized;

    ConsistencyLevel(String name, int readQuorum, int writeQuorum, boolean leaderSerialized) {
        this.name = name;
        this.readQuorum = readQuorum;
        this.writeQuorum = writeQuorum;
        this.leaderSerialized = leaderSerialized;
    }

    static List<ConsistencyLevel> standard(int replicas) {
        int majority = replicas / 2 + 1;
        List<ConsistencyLevel> levels = new ArrayList<>();
        levels.add(new ConsistencyLevel("EVENTUAL", 1, 1, false));
        levels.add(new ConsistencyLevel("QUORUM", majority, majority, false));
        levels.add(new ConsistencyLevel("ALL", replicas, replicas, false));
        levels.add(new ConsistencyLevel("SINGLE-LEADER", 1, 1, true));
        return levels;
    }

    /** Replicas busy per operation; the leader model funnels all traffic through one server. */
    double fanout() { return leaderSerialized ? 1.0 : writeQuorum; }

    @Override public String toString() { return name; }
}

/** Outcome of one benchmark run plus the safety numbers gathered while simulating. */
final class BenchResult {
    final ConsistencyLevel level;
    final double offeredRate;
    final int operations;
    final double throughput;
    final double meanLatency;
    final double p50Latency;
    final double p99Latency;
    final double maxLatency;
    final double utilization;
    final double inFlightMean;
    final double horizon;
    final double maxOverlap;
    final double conservationError;

    BenchResult(ConsistencyLevel level, double offeredRate, int operations, double throughput,
            double meanLatency, double p50Latency, double p99Latency, double maxLatency, double utilization,
            double inFlightMean, double horizon, double maxOverlap, double conservationError) {
        this.level = level;
        this.offeredRate = offeredRate;
        this.operations = operations;
        this.throughput = throughput;
        this.meanLatency = meanLatency;
        this.p50Latency = p50Latency;
        this.p99Latency = p99Latency;
        this.maxLatency = maxLatency;
        this.utilization = utilization;
        this.inFlightMean = inFlightMean;
        this.horizon = horizon;
        this.maxOverlap = maxOverlap;
        this.conservationError = conservationError;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    static void checkClose(double actual, double expected, double tol, String message) {
        double scale = Math.max(1e-9, Math.abs(expected));
        check(Math.abs(actual - expected) / scale <= tol,
                message + " (actual=" + actual + ", expected=" + expected + ", tol=" + tol + ")");
    }

    // ---------------------------------------------------------------- geometry of quorums

    /** All k-subsets of {0..n-1}, enumerated lexicographically so results are reproducible. */
    static List<int[]> combinations(int n, int k) {
        List<int[]> out = new ArrayList<>();
        if (k < 0 || k > n) return out;
        int[] idx = new int[k];
        for (int i = 0; i < k; i++) idx[i] = i;
        while (true) {
            out.add(idx.clone());
            int i = k - 1;
            while (i >= 0 && idx[i] == n - k + i) i--;
            if (i < 0) break;
            idx[i]++;
            for (int j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1;
        }
        return out;
    }

    static boolean intersects(int[] a, int[] b) {
        for (int x : a) for (int y : b) if (x == y) return true;
        return false;
    }

    /** Brute force: does *some* read quorum / write quorum pair fail to intersect? O(C(n,r)*C(n,w)). */
    static int[] firstDisjointPair(int n, int r, int w) {
        for (int[] rq : combinations(n, r)) {
            for (int[] wq : combinations(n, w)) {
                if (!intersects(rq, wq)) return new int[]{1}; // disjoint pair exists
            }
        }
        return null;
    }

    /** The central strong-consistency theorem, checked by exhaustive quorum-pair enumeration. */
    static boolean quorumsAlwaysIntersect(int n, int r, int w) { return firstDisjointPair(n, r, w) == null; }

    static int[] rotateQuorum(int n, int quorumSize, int operationIndex, int offset) {
        int[] group = new int[quorumSize];
        int base = (int) Math.floorMod((long) operationIndex * quorumSize + offset, (long) n);
        for (int i = 0; i < quorumSize; i++) group[i] = (base + i) % n;
        return group;
    }

    /**
     * Runs write-then-read pairs on real replica version stamps. When the level has a disjoint quorum pair
     * the adversarial schedule uses it, otherwise a rotating schedule is used. Returns stale read count.
     */
    static int[] stalenessExperiment(ConsistencyLevel level, int n, int operations) {
        long[] versions = new long[n];
        long counter = 0;
        int stale = 0;
        int[] forcedRead = null;
        int[] forcedWrite = null;
        boolean disjointPossible = !level.leaderSerialized
                && firstDisjointPair(n, level.readQuorum, level.writeQuorum) != null;
        if (level.leaderSerialized) {
            forcedRead = new int[]{0};
            forcedWrite = new int[]{0};
        } else if (disjointPossible) {
            search:
            for (int[] rq : combinations(n, level.readQuorum)) {
                for (int[] wq : combinations(n, level.writeQuorum)) {
                    if (!intersects(rq, wq)) { forcedRead = rq; forcedWrite = wq; break search; }
                }
            }
        }
        for (int op = 0; op < operations; op++) {
            int[] write = forcedWrite != null ? forcedWrite
                    : rotateQuorum(n, level.writeQuorum, op, 0);
            int[] read = forcedRead != null ? forcedRead
                    : rotateQuorum(n, level.readQuorum, op, level.readQuorum);
            counter++;
            for (int r : write) versions[r] = counter;
            long seen = Long.MIN_VALUE;
            for (int r : read) seen = Math.max(seen, versions[r]);
            if (seen < counter) stale++;
        }
        return new int[]{stale, disjointPossible ? 1 : 0};
    }

    // ---------------------------------------------------------------- workload generation

    /** Shared arrival process: exponential inter-arrivals with the requested offered rate. */
    static double[] arrivalProcess(int operations, double offeredRate, long seed) {
        double[] arrivals = new double[operations];
        SplitMix64 rng = new SplitMix64(seed);
        double t = 0.0;
        for (int i = 0; i < operations; i++) {
            t += rng.nextExponential(1.0 / offeredRate);
            arrivals[i] = t;
        }
        return arrivals;
    }

    /** Service time of operation op at replica r - identical across levels (paired benchmark design). */
    static double serviceTime(long seed, int operation, int replica, double mean) {
        SplitMix64 rng = new SplitMix64(SplitMix64.mix(seed + 0x51ED270BL, SplitMix64.mix(operation, replica)));
        return rng.nextExponential(mean);
    }

    /** Analytic capacity in operations per unit time: N/(q*mean) parallel replicas, 1/mean for a leader. */
    static double capacity(ConsistencyLevel level, int replicas, double meanService) {
        if (level.leaderSerialized) return 1.0 / meanService;
        return replicas / (level.fanout() * meanService);
    }

    /** Offered rate that makes every replica busy for the requested fraction of time. */
    static double offeredRateForUtilization(ConsistencyLevel level, int replicas, double meanService, double u) {
        return u * capacity(level, replicas, meanService);
    }

    // ---------------------------------------------------------------- the discrete-event benchmark

    static BenchResult run(ConsistencyLevel level, int replicas, int operations, double meanService,
            double networkOverhead, double offeredRate, long seed) {
        double[] arrivals = arrivalProcess(operations, offeredRate, seed);
        Replica[] reps = new Replica[replicas];
        for (int i = 0; i < replicas; i++) reps[i] = new Replica(i);
        double[] latency = new double[operations];
        double[] finishTime = new double[operations];
        double lastFinish = 0.0;
        for (int op = 0; op < operations; op++) {
            int[] group = level.leaderSerialized ? new int[]{0}
                    : rotateQuorum(replicas, level.writeQuorum, op, 0);
            double arrival = arrivals[op];
            double opFinish = arrival;
            for (int r : group) {
                double svc = serviceTime(seed, op, r, meanService);
                double start = Math.max(arrival, reps[r].freeAt);
                reps[r].occupy(start, start + svc, svc);
                opFinish = Math.max(opFinish, start + svc);
            }
            opFinish += networkOverhead;
            finishTime[op] = opFinish;
            latency[op] = opFinish - arrival;
            lastFinish = Math.max(lastFinish, opFinish);
        }
        double[] sorted = latency.clone();
        Arrays.sort(sorted);
        double sum = 0.0;
        for (double v : latency) sum += v;
        double horizon = lastFinish - arrivals[0];
        double busy = 0.0, maxOverlap = 0.0, work = 0.0;
        for (Replica r : reps) {
            busy += r.busyTime;
            work += r.servedWork;
            maxOverlap = Math.max(maxOverlap, r.maxOverlap);
        }
        // Conservation law: total replica busy time must equal the total service work issued.
        double issued = 0.0;
        for (int op = 0; op < operations; op++) {
            int[] group = level.leaderSerialized ? new int[]{0} : rotateQuorum(replicas, level.writeQuorum, op, 0);
            for (int r : group) issued += serviceTime(seed, op, r, meanService);
        }
        double conservationError = Math.abs(busy - issued) / Math.max(1e-12, issued);
        double throughput = operations / horizon;
        double meanLatency = sum / operations;
        double inFlight = meanInFlight(arrivals, finishTime, horizon);
        return new BenchResult(level, offeredRate, operations, throughput, meanLatency,
                percentile(sorted, 0.50), percentile(sorted, 0.99), sorted[operations - 1],
                busy / (replicas * horizon), inFlight, horizon, maxOverlap, conservationError);
    }

    /** Nearest-rank percentile over an ascending array. */
    static double percentile(double[] ascending, double p) {
        if (ascending.length == 0) throw new IllegalArgumentException("empty sample");
        int rank = (int) Math.ceil(p * ascending.length);
        rank = Math.max(1, Math.min(ascending.length, rank));
        return ascending[rank - 1];
    }

    /** Independent oracle for the percentile helper: sorts a fresh copy with insertion sort. */
    static double percentileOracle(double[] samples, double p) {
        double[] copy = samples.clone();
        for (int i = 1; i < copy.length; i++) {
            double key = copy[i];
            int j = i - 1;
            while (j >= 0 && copy[j] > key) { copy[j + 1] = copy[j]; j--; }
            copy[j + 1] = key;
        }
        int rank = (int) Math.ceil(p * copy.length);
        if (rank < 1) rank = 1;
        if (rank > copy.length) rank = copy.length;
        return copy[rank - 1];
    }

    /** Time-weighted mean number of in-flight operations, by a sweep over arrival/finish events. */
    static double meanInFlight(double[] arrivals, double[] finishes, double horizon) {
        int n = arrivals.length;
        double[][] events = new double[2 * n][2];
        for (int i = 0; i < n; i++) {
            events[2 * i][0] = arrivals[i];
            events[2 * i][1] = 1.0;
            events[2 * i + 1][0] = finishes[i];
            events[2 * i + 1][1] = -1.0;
        }
        Arrays.sort(events, (a, b) -> a[0] != b[0] ? Double.compare(a[0], b[0]) : Double.compare(b[1], a[1]));
        double area = 0.0, current = 0.0, previous = events[0][0];
        for (double[] e : events) {
            area += current * (e[0] - previous);
            current += e[1];
            previous = e[0];
        }
        return area / horizon;
    }

    /** H_q = 1 + 1/2 + ... + 1/q: E[max of q i.i.d. exponentials] = mean * H_q. */
    static double harmonic(int q) {
        double h = 0.0;
        for (int i = 1; i <= q; i++) h += 1.0 / i;
        return h;
    }

    // ---------------------------------------------------------------- main

    public static void main(String[] args) {
        final int replicas = 5;
        final double meanService = 2.0;
        final double network = 0.5;
        final long seed = 851_2024L;

        // --- 1. quorum intersection theorem, brute forced over every read/write quorum pair.
        for (int n : new int[]{1, 3, 5, 7}) {
            for (int r = 1; r <= n; r++) {
                for (int w = 1; w <= n; w++) {
                    check(quorumsAlwaysIntersect(n, r, w) == (r + w > n),
                            "intersection theorem for N=" + n + " R=" + r + " W=" + w);
                }
            }
        }
        check(quorumsAlwaysIntersect(5, 3, 3), "QUORUM(5,3,3) intersects");
        check(!quorumsAlwaysIntersect(5, 2, 3), "R+W<=N admits a disjoint pair");
        check(combinations(7, 4).size() == 35 && combinations(5, 1).size() == 5,
                "combination enumeration sizes");

        // --- 2. the level table is well formed and its quorums have the promised strengths.
        List<ConsistencyLevel> levels = ConsistencyLevel.standard(replicas);
        check(levels.size() == 4, "four consistency levels");
        for (ConsistencyLevel lv : levels) {
            check(lv.readQuorum >= 1 && lv.writeQuorum <= replicas && lv.readQuorum <= replicas
                    && lv.writeQuorum >= 1, "quorum sizes in range for " + lv.name);
        }
        check(levels.get(1).writeQuorum == replicas / 2 + 1, "majority definition");
        check(levels.get(3).leaderSerialized && levels.get(0).writeQuorum < levels.get(1).writeQuorum
                && levels.get(1).writeQuorum < levels.get(2).writeQuorum, "level strength ordering");

        // --- 3. stale reads: impossible for intersecting quorums, provable for the (1,1) level.
        for (ConsistencyLevel lv : levels) {
            int[] outcome = stalenessExperiment(lv, replicas, 300);
            int stale = outcome[0];
            boolean disjointPossible = outcome[1] == 1;
            if (lv.leaderSerialized) {
                check(stale == 0, "leader-pinned traffic never reads stale data, saw " + stale);
            } else if (disjointPossible) {
                check(stale == 300, lv.name + " with a disjoint quorum pair must produce stale reads");
            } else {
                check(stale == 0, lv.name + " (R+W>N) must never produce a stale read, saw " + stale);
            }
        }
        int[] staleQuorum = stalenessExperiment(levels.get(1), replicas, 300);
        check(staleQuorum[0] == 0 && staleQuorum[1] == 0,
                "QUORUM: intersecting quorums and zero stale reads");

        // --- 4. saturation: measured peak throughput against the analytic capacity N/(q*mean).
        double[] peak = new double[levels.size()];
        for (int i = 0; i < levels.size(); i++) {
            ConsistencyLevel lv = levels.get(i);
            double cap = capacity(lv, replicas, meanService);
            BenchResult saturated = run(lv, replicas, 3000, meanService, network, 3.0 * cap, seed);
            peak[i] = saturated.throughput;
            checkClose(saturated.throughput, cap, 0.12, lv.name + " saturates near capacity");
            check(saturated.throughput <= cap * 1.05, lv.name + " never exceeds capacity " + cap);
            check(saturated.maxOverlap <= 1e-9, lv.name + " never double-books a replica");
            check(saturated.conservationError < 1e-12, lv.name + " conserves replica busy time");
        }
        check(peak[0] > peak[1] && peak[1] > peak[2],
                "throughput falls as the write quorum widens: " + Arrays.toString(peak));
        check(peak[0] > 2.0 * peak[3], "the (1,1) level scales out beyond a single leader");
        checkClose(peak[3], 1.0 / meanService, 0.12, "leader is capped by one server");

        // --- 5. latency/throughput frontier at increasing fractions of each level's own capacity.
        System.out.printf("N=%d meanService=%.2f network=%.2f%n", replicas, meanService, network);
        System.out.printf("%-14s %8s %8s %8s %8s %8s %8s%n",
                "level", "load", "offer", "thrpt", "p50", "p99", "util");
        for (ConsistencyLevel lv : levels) {
            double cap = capacity(lv, replicas, meanService);
            double previousP99 = -1.0;
            double previousThroughput = -1.0;
            for (double u : new double[]{0.2, 0.6, 0.9}) {
                BenchResult res = run(lv, replicas, 3000, meanService, network, u * cap, seed);
                System.out.printf("%-14s %8.2f %8.3f %8.3f %8.3f %8.3f %8.3f%n",
                        lv.name, u, res.offeredRate, res.throughput, res.p50Latency, res.p99Latency,
                        res.utilization);
                check(res.p99Latency > previousP99, lv.name + " p99 latency grows with offered load");
                check(res.throughput > previousThroughput, lv.name + " throughput grows with offered load");
                check(res.utilization <= u * 1.15 + 1e-9, lv.name + " utilisation tracks the offered load");
                previousP99 = res.p99Latency;
                previousThroughput = res.throughput;
            }
            // Little's law: time-weighted in-flight work equals throughput times mean latency.
            BenchResult stable = run(lv, replicas, 3000, meanService, network,
                    offeredRateForUtilization(lv, replicas, meanService, 0.5), seed);
            checkClose(stable.inFlightMean, stable.throughput * stable.meanLatency, 0.05,
                    lv.name + " satisfies Little's law");
        }

        // --- 6. negligible load reproduces the closed form E[max of q exponentials] = mean * H_q.
        for (ConsistencyLevel lv : levels) {
            double offered = 0.05 * capacity(lv, replicas, meanService);
            BenchResult cold = run(lv, replicas, 4000, meanService, network, offered, seed);
            double expected = network + meanService * harmonic((int) lv.fanout());
            checkClose(cold.meanLatency, expected, 0.10, lv.name + " idle latency matches mean*H_q");
            check(lv.fanout() == lv.writeQuorum || lv.leaderSerialized, "fan-out equals the write quorum");
        }

        // --- 7. equal utilisation: wider quorums pay a straggler penalty.
        double meanEventual = 0.0, meanQuorum = 0.0, meanAll = 0.0;
        for (ConsistencyLevel lv : levels) {
            BenchResult res = run(lv, replicas, 4000, meanService, network,
                    offeredRateForUtilization(lv, replicas, meanService, 0.7), seed);
            System.out.printf("iso-utilisation u=0.70 %-14s mean=%7.3f p50=%7.3f p99=%7.3f thrpt=%6.3f%n",
                    lv.name, res.meanLatency, res.p50Latency, res.p99Latency, res.throughput);
            if (lv.name.equals("EVENTUAL")) meanEventual = res.meanLatency;
            if (lv.name.equals("QUORUM")) meanQuorum = res.meanLatency;
            if (lv.name.equals("ALL")) meanAll = res.meanLatency;
        }
        check(meanEventual < meanQuorum && meanQuorum < meanAll,
                "at equal utilisation ANY < QUORUM < ALL: " + meanEventual + " " + meanQuorum + " " + meanAll);

        // --- 8. percentile helper against an independent insertion-sort oracle, including edges.
        double[] sample = {5.0, 1.0, 4.0, 2.0, 3.0};
        double[] sortedSample = sample.clone();
        Arrays.sort(sortedSample);
        for (double p : new double[]{0.0, 0.25, 0.5, 0.9, 0.99, 1.0}) {
            check(percentile(sortedSample, p) == percentileOracle(sample, p), "percentile oracle at p=" + p);
        }
        check(percentile(sortedSample, 0.5) == 3.0, "single percentile value");
        check(percentile(new double[]{7.0}, 0.99) == 7.0, "single sample");
        check(percentile(new double[]{2.0, 2.0, 2.0}, 0.5) == 2.0, "constant sample");
        boolean threw = false;
        try {
            percentile(new double[0], 0.5);
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "empty sample rejected");

        // --- 9. determinism: identical seeds give identical numbers, and orderings survive a new seed.
        BenchResult a = run(levels.get(1), replicas, 500, meanService, network, 0.4, seed);
        BenchResult b = run(levels.get(1), replicas, 500, meanService, network, 0.4, seed);
        check(a.throughput == b.throughput && a.p99Latency == b.p99Latency && a.meanLatency == b.meanLatency,
                "identical seeds reproduce the benchmark exactly");
        for (long altSeed : new long[]{seed + 1, seed + 7}) {
            double[] alt = new double[3];
            for (int i = 0; i < 3; i++) {
                alt[i] = run(levels.get(i), replicas, 1500, meanService, network,
                        0.25 * capacity(levels.get(i), replicas, meanService), altSeed).meanLatency;
            }
            check(alt[0] < alt[1] && alt[1] < alt[2],
                    "mean latency ordering holds for seed " + altSeed + ": " + Arrays.toString(alt));
        }

        // --- 10. degenerate topologies: one replica collapses every level, and a single operation works.
        for (ConsistencyLevel lv : ConsistencyLevel.standard(1)) {
            BenchResult single = run(lv, 1, 200, meanService, network,
                    0.02 * capacity(lv, 1, meanService), seed);
            checkClose(single.meanLatency, network + meanService * harmonic(1), 0.10,
                    "N=1 latency for " + lv.name);
            check(single.maxOverlap <= 1e-9 && single.conservationError < 1e-12, "N=1 safety for " + lv.name);
        }
        BenchResult one = run(levels.get(2), replicas, 1, meanService, network, 0.5, seed);
        check(one.operations == 1 && one.throughput > 0 && Double.isFinite(one.p99Latency),
                "single operation run is finite");
        check(arrivalProcess(5, 2.0, seed)[0] > 0 && arrivalProcess(5, 2.0, seed)[4] > arrivalProcess(5, 2.0, seed)[0],
                "arrival process is increasing");

        System.out.printf("peak throughput by level: EVENTUAL=%.3f QUORUM=%.3f ALL=%.3f LEADER=%.3f%n",
                peak[0], peak[1], peak[2], peak[3]);
        System.out.printf("analytic capacities:       EVENTUAL=%.3f QUORUM=%.3f ALL=%.3f LEADER=%.3f%n",
                capacity(levels.get(0), replicas, meanService), capacity(levels.get(1), replicas, meanService),
                capacity(levels.get(2), replicas, meanService), capacity(levels.get(3), replicas, meanService));
        System.out.println("checks=" + checks);
        System.out.println("PASS");
    }
}
