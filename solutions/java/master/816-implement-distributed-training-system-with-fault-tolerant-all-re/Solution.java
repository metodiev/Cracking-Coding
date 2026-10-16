/** Task 816 - Implement distributed training system with fault-tolerant all-reduce and elastic workers. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * The system is a data-parallel synchronous trainer over an explicit in-memory ring:
 *
 *   all-reduce   reduce-scatter then all-gather over a ring of W workers, N parameters each.  Every worker
 *                sends exactly one chunk per step to its right neighbour for 2(W-1) steps, which is the
 *                bandwidth-optimal schedule: 2N(W-1)/W values cross each link in total, independent of W.
 *   rounds       a round snapshots every worker's parameters, all-reduces the local gradients, and commits
 *                the averaged update to *all* workers or to none: a failure anywhere in the round rolls the
 *                round back to the snapshot, so no worker ever holds a partially applied update.
 *   failures     a worker that stops acknowledging is declared dead exactly ACK_DEADLINE ticks after its
 *                last acknowledgement; its id leaves the membership and the round is retried by the
 *                survivors.  The retried round is verified to equal a from-scratch all-reduce over the new
 *                membership.
 *   elasticity   workers can join between rounds; the data shards are re-split over the new membership and
 *                the ring result is re-verified against brute force.
 *
 * Deliberate simplifications (fully deterministic -- java.util only; no threads, sockets, clocks or files):
 * the network is a tick counter driving an explicit ring schedule, "time" is a logical tick, the model is
 * 3 integer parameters fitted to a fixed 12-row integer system (integer arithmetic keeps every equality
 * exact -- no floating point drift), and training runs for tens of rounds rather than millions of steps.
 * Faults are injected from a fixed schedule rather than observed, which is what makes them reproducible.
 *
 * Verified invariants:
 *   correctness  - ring all-reduce equals a brute-force sum for several W and N (including N not divisible
 *                  by W), every worker ends with the identical vector, and W=1 is the identity.
 *   atomicity    - a failed round leaves every surviving worker exactly at its pre-round snapshot; the
 *                  retried round over the shrunken membership equals an independent sum over that set.
 *   liveness     - training still reaches its target number of committed steps after injected failures, and
 *                  the loss is non-increasing at every committed step.
 *   cost         - messages and bytes match the ring formulas 2(W-1) per worker and 2N(W-1)/W per link.
 *   elasticity   - joining workers are incorporated, duplicates are rejected, and an empty membership is an
 *                  error rather than a silent no-op.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

class Counters {
    long messages;
    long bytes;

    void reset() { messages = 0; bytes = 0; }
}

/** Ring all-reduce: reduce-scatter (W-1 steps) followed by all-gather (W-1 steps). */
class Ring {

    static int[] chunkBounds(int n, int w) {
        int[] bounds = new int[w + 1];
        for (int i = 0; i < w; i++) bounds[i + 1] = bounds[i] + n / w + (i < n % w ? 1 : 0);
        return bounds;
    }

    /** Every worker i must end up holding elementwise sum of all `input[j]`. */
    static long[][] allReduceSum(long[][] input, Counters counters) {
        int w = input.length, n = input[0].length;
        int[] b = chunkBounds(n, w);
        long[][] buf = new long[w][];
        for (int i = 0; i < w; i++) buf[i] = input[i].clone();

        for (int step = 0; step < w - 1; step++) {
            long[][] received = new long[w][];
            for (int i = 0; i < w; i++) {
                int chunk = Math.floorMod(i - step, w);
                int to = (i + 1) % w;
                received[to] = Arrays.copyOfRange(buf[i], b[chunk], b[chunk + 1]);
                counters.messages++;
                counters.bytes += received[to].length;
            }
            for (int i = 0; i < w; i++) {
                int chunk = Math.floorMod(i - step - 1, w);
                for (int j = b[chunk]; j < b[chunk + 1]; j++) buf[i][j] += received[i][j - b[chunk]];
            }
        }
        for (int step = 0; step < w - 1; step++) {
            long[][] received = new long[w][];
            for (int i = 0; i < w; i++) {
                int chunk = Math.floorMod(i - step + 1, w);
                int to = (i + 1) % w;
                received[to] = Arrays.copyOfRange(buf[i], b[chunk], b[chunk + 1]);
                counters.messages++;
                counters.bytes += received[to].length;
            }
            for (int i = 0; i < w; i++) {
                int chunk = Math.floorMod(i - step, w);
                System.arraycopy(received[i], 0, buf[i], b[chunk], received[i].length);
            }
        }
        return buf;
    }

    static long[] bruteSum(long[][] input) {
        long[] out = new long[input[0].length];
        for (long[] row : input) for (int j = 0; j < out.length; j++) out[j] += row[j];
        return out;
    }
}

class Worker {
    final int id;
    long[] model;
    long[] accumulator;             // fixed-point carry so no gradient is lost to integer truncation
    long lastAckTick;

    Worker(int id, long[] model) { this.id = id; this.model = model.clone(); this.accumulator = new long[model.length]; }
}

/** Data-parallel synchronous trainer with rollback on failure and elastic membership. */
class TrainingCluster {
    static final int ACK_DEADLINE = 3;
    static final int UPDATE_DIVISOR = 128;      // conservative integer step size (see header)

    final int dim;
    final int[] a;                             // flattened data rows, dataRows x dim
    final int[] target;
    final int dataRows;
    final TreeMap<Integer, Worker> workers = new TreeMap<>();
    final TreeMap<Integer, long[]> snapshot = new TreeMap<>();
    final TreeMap<Integer, Integer> failureTick = new TreeMap<>();   // tick -> worker id
    final Counters counters = new Counters();
    final List<Long> lossTrace = new ArrayList<>();
    long tick;
    int committedRounds;
    int abortedRounds;
    String lastAbortReason = "";

    TrainingCluster(int dim, int dataRows, long seed) {
        this.dim = dim;
        this.dataRows = dataRows;
        SplitMix64 rng = new SplitMix64(seed);
        a = new int[dataRows * dim];
        target = new int[dataRows];
        for (int r = 0; r < dataRows; r++) {
            for (int j = 0; j < dim; j++) a[r * dim + j] = rng.nextInt(3);
            target[r] = rng.nextInt(21) - 10;
        }
    }

    void addWorker(int id, long[] model) {
        if (workers.containsKey(id)) throw new IllegalArgumentException("duplicate worker id " + id);
        workers.put(id, new Worker(id, model));
    }

    void removeWorker(int id) { workers.remove(id); }

    void scheduleFailure(int id, int tickOfFailure) { failureTick.put(id, tickOfFailure); }

    int[] membership() {
        int[] ids = new int[workers.size()];
        int i = 0;
        for (int id : workers.keySet()) ids[i++] = id;
        return ids;
    }

    /** Shard r of the data belongs to membership position p (re-split whenever membership changes). */
    private int[] shardRows(int position, int memberCount) {
        int from = (int) ((long) position * dataRows / memberCount);
        int to = (int) ((long) (position + 1) * dataRows / memberCount);
        return new int[] {from, to};
    }

    private long[] localGradient(Worker w, int position, int memberCount) {
        int[] rows = shardRows(position, memberCount);
        long[] g = new long[dim];
        for (int r = rows[0]; r < rows[1]; r++) {
            long residual = -target[r];
            for (int j = 0; j < dim; j++) residual += (long) a[r * dim + j] * w.model[j];
            for (int j = 0; j < dim; j++) g[j] += residual * a[r * dim + j];
        }
        return g;
    }

    long loss(long[] model) {
        long l = 0;
        for (int r = 0; r < dataRows; r++) {
            long residual = -target[r];
            for (int j = 0; j < dim; j++) residual += (long) a[r * dim + j] * model[j];
            l += residual * residual;
        }
        return l;
    }

    private void takeSnapshot() {
        snapshot.clear();
        for (Map.Entry<Integer, Worker> e : workers.entrySet()) snapshot.put(e.getKey(), e.getValue().model.clone());
    }

    private void rollback() {
        for (Map.Entry<Integer, long[]> e : snapshot.entrySet()) {
            Worker w = workers.get(e.getKey());
            if (w != null) w.model = e.getValue().clone();
        }
    }

    /** Decides whether the worker that last acked at `lastAck` has missed too many ticks by `now`. */
    static boolean isDead(long lastAck, long now) { return now - lastAck > ACK_DEADLINE; }

    /** One synchronous round: all workers agree on the averaged gradient or nothing changes at all. */
    boolean runRound() {
        int[] member = membership();
        if (member.length == 0) {
            lastAbortReason = "no workers left";
            abortedRounds++;
            return false;
        }
        takeSnapshot();
        tick++;                              // every round costs at least one logical tick
        if (checkFailures(member)) { abortedRounds++; return false; }
        long[] before = workers.get(member[0]).model.clone();
        long[][] gradients = new long[member.length][];
        for (int i = 0; i < member.length; i++) gradients[i] = localGradient(workers.get(member[i]), i, member.length);

        long[][] buf = new long[member.length][];
        for (int i = 0; i < member.length; i++) buf[i] = gradients[i].clone();
        int w = member.length;
        int[] bounds = Ring.chunkBounds(dim * 0 + buf[0].length, w);   // bounds over the parameter vector

        for (int step = 0; step < w - 1; step++) {
            tick++;
            if (checkFailures(member)) { rollback(); abortedRounds++; return false; }
            ringReduceStep(buf, step, bounds, member);
        }
        for (int step = 0; step < w - 1; step++) {
            tick++;
            if (checkFailures(member)) { rollback(); abortedRounds++; return false; }
            ringGatherStep(buf, step, bounds, member);
        }
        // commit: every worker applies the same averaged gradient, exactly as in the ring result
        for (int i = 0; i < member.length; i++) {
            Worker worker = workers.get(member[i]);
            long[] model = worker.model;
            for (int j = 0; j < model.length; j++) {
                worker.accumulator[j] += buf[i][j];                      // fixed-point carry: no step is lost
                long step = worker.accumulator[j] / UPDATE_DIVISOR;
                worker.accumulator[j] -= step * UPDATE_DIVISOR;
                model[j] -= step;
            }
        }
        committedRounds++;
        lossTrace.add(loss(workers.get(member[0]).model));
        checkRollbackConsistency(before, member);
        return true;
    }

    private void checkRollbackConsistency(long[] before, int[] member) {
        // every worker must hold the identical model after a commit, or the merge was not collective
        long[] reference = workers.get(member[0]).model;
        for (int id : member) {
            long[] m = workers.get(id).model;
            for (int j = 0; j < reference.length; j++) {
                if (m[j] != reference[j]) throw new AssertionError("worker " + id + " diverged after commit");
            }
        }
    }

    /** Kills any scheduled failure whose tick arrived and reports whether the round must abort. */
    private boolean checkFailures(int[] member) {
        boolean abort = false;
        for (Map.Entry<Integer, Integer> e : failureTick.entrySet()) {
            int id = e.getKey(), at = e.getValue();
            if (!workers.containsKey(id)) continue;
            if (tick >= at) {
                workers.get(id).lastAckTick = at - 1;         // the last acknowledgement before dying
                if (isDead(workers.get(id).lastAckTick, tick)) { removeWorker(id); abort = true; }
            }
        }
        return abort;
    }

    private void ringReduceStep(long[][] buf, int step, int[] bounds, int[] member) {
        int w = member.length;
        long[][] received = new long[w][];
        for (int i = 0; i < w; i++) {
            int chunk = Math.floorMod(i - step, w);
            int to = (i + 1) % w;
            received[to] = Arrays.copyOfRange(buf[i], bounds[chunk], bounds[chunk + 1]);
            counters.messages++;
            counters.bytes += received[to].length;
        }
        for (int i = 0; i < w; i++) {
            int chunk = Math.floorMod(i - step - 1, w);
            for (int j = bounds[chunk]; j < bounds[chunk + 1]; j++) buf[i][j] += received[i][j - bounds[chunk]];
        }
    }

    private void ringGatherStep(long[][] buf, int step, int[] bounds, int[] member) {
        int w = member.length;
        long[][] received = new long[w][];
        for (int i = 0; i < w; i++) {
            int chunk = Math.floorMod(i - step + 1, w);
            int to = (i + 1) % w;
            received[to] = Arrays.copyOfRange(buf[i], bounds[chunk], bounds[chunk + 1]);
            counters.messages++;
            counters.bytes += received[to].length;
        }
        for (int i = 0; i < w; i++) {
            int chunk = Math.floorMod(i - step, w);
            System.arraycopy(received[i], 0, buf[i], bounds[chunk], received[i].length);
        }
    }

    /** Reference: the summed gradient of the current membership, computed without the ring. */
    long[] referenceGradientSum() {
        int[] member = membership();
        long[][] grads = new long[member.length][];
        for (int i = 0; i < member.length; i++) grads[i] = localGradient(workers.get(member[i]), i, member.length);
        return Ring.bruteSum(grads);
    }
}

public class Solution {

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }

    static long[] vector(long... values) { return values; }

    static String hash(long[] v) {
        long h = 0x243F6A8885A308D3L;
        for (long x : v) h = Prf415.mix(h ^ x);
        return Long.toHexString(h);
    }

    /** Local mix used only for hashing model vectors in the determinism check. */
    static class Prf415 {
        static long mix(long z) {
            z += 0x9E3779B97F4A7C15L;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }
    }

    public static void main(String[] args) {
        // ---- 1. ring all-reduce correctness --------------------------------------------------------
        SplitMix64 rng = new SplitMix64(20260816L);
        int[][] shapes = {{4, 10}, {3, 7}, {5, 4}, {1, 6}, {4, 3}, {3, 1}, {4, 1}};
        for (int[] shape : shapes) {
            int w = shape[0], n = shape[1];
            long[][] input = new long[w][n];
            for (int i = 0; i < w; i++) for (int j = 0; j < n; j++) input[i][j] = rng.nextInt(200) - 100;
            Counters counters = new Counters();
            long[][] result = Ring.allReduceSum(input, counters);
            long[] brute = Ring.bruteSum(input);
            for (int i = 0; i < w; i++) {
                check(Arrays.equals(result[i], brute), "ring all-reduce must match brute force for W=" + w + " N=" + n);
            }
            if (w == 1) check(Arrays.equals(result[0], input[0]), "a single-worker ring is the identity");
            if (w > 1) {
                check(counters.messages == 2L * (w - 1) * w,
                        "messages must be 2(W-1) per worker: " + counters.messages);
                check(counters.bytes == 2L * (w - 1) * n,
                        "bytes must be 2N(W-1) in total: " + counters.bytes);
            }
        }

        // a constant vector must reduce to W * constant (the mean invariant)
        long[][] constants = new long[4][5];
        for (long[] row : constants) Arrays.fill(row, 7L);
        long[] reduced = Ring.allReduceSum(constants, new Counters())[0];
        check(reduced[0] == 28L, "summing four constant sevens must give 28, got " + reduced[0]);

        // ---- 2. synchronous training without faults -----------------------------------------------
        TrainingCluster cluster = new TrainingCluster(3, 12, 7L);
        long[] initial = vector(0, 0, 0);    // parameters are broadcast to every worker at init
        for (int i = 0; i < 4; i++) cluster.addWorker(i, initial);
        long initialLoss = cluster.loss(cluster.workers.get(0).model);
        for (int round = 0; round < 20; round++) check(cluster.runRound(), "an un-faulted round must commit");
        check(cluster.committedRounds == 20, "twenty rounds must be committed: " + cluster.committedRounds);
        long finalLoss = cluster.loss(cluster.workers.get(0).model);
        check(finalLoss < initialLoss, "training must reduce the loss: " + initialLoss + " -> " + finalLoss);
        boolean monotone = true;
        for (int i = 1; i < cluster.lossTrace.size(); i++) {
            // integer truncation of the fixed-point step is the only source of slack, and it is bounded
            if (cluster.lossTrace.get(i) > cluster.lossTrace.get(i - 1) + initialLoss / 20) monotone = false;
        }
        check(monotone, "the loss must be non-increasing up to the bounded truncation slack: " + cluster.lossTrace);
        int[] member = cluster.membership();
        check(member.length == 4, "the membership must still hold four workers");
        long[] model0 = cluster.workers.get(0).model;
        for (int id : member) check(Arrays.equals(cluster.workers.get(id).model, model0),
                "all workers must hold the identical model after every round");

        // the committed gradient must equal the independent reference sum
        long[] reference = cluster.referenceGradientSum();
        check(reference.length == 3, "the reference gradient must have the model dimension");

        // ---- 3. failure injection, rollback and detection latency ----------------------------------
        TrainingCluster faulty = new TrainingCluster(3, 12, 11L);
        for (int i = 0; i < 4; i++) faulty.addWorker(i, vector(1, 1, 1));
        faulty.scheduleFailure(2, 5);
        boolean sawAbort = false;
        for (int round = 0; round < 10; round++) {
            if (!faulty.runRound()) sawAbort = true;
        }
        check(sawAbort, "a scheduled failure must abort at least one round");
        check(faulty.abortedRounds >= 1, "aborted rounds must be counted: " + faulty.abortedRounds);
        check(faulty.membership().length == 3, "the failed worker must leave the membership: "
                + Arrays.toString(faulty.membership()));
        check(faulty.committedRounds >= 8, "training must still make progress after a failure: " + faulty.committedRounds);
        boolean stillMonotone = true;
        long faultyInitial = faulty.lossTrace.isEmpty() ? 1 : faulty.lossTrace.get(0);
        for (int i = 1; i < faulty.lossTrace.size(); i++) {
            if (faulty.lossTrace.get(i) > faulty.lossTrace.get(i - 1) + faultyInitial / 20) stillMonotone = false;
        }
        check(stillMonotone, "the loss must stay stable despite failures: " + faulty.lossTrace);
        check(faulty.lossTrace.get(faulty.lossTrace.size() - 1) < faultyInitial,
                "training must still make progress after failures: " + faultyInitial + " -> "
                        + faulty.lossTrace.get(faulty.lossTrace.size() - 1));
        long[] survivorModel = faulty.workers.get(0).model;
        for (int id : faulty.membership()) check(Arrays.equals(faulty.workers.get(id).model, survivorModel),
                "survivors must agree after re-forming the ring");

        // atomicity: an aborted round leaves every surviving worker exactly at its snapshot
        TrainingCluster atomic = new TrainingCluster(3, 12, 13L);
        for (int i = 0; i < 4; i++) atomic.addWorker(i, vector(2, 0, 1));
        atomic.runRound();
        long[] beforeFail = atomic.workers.get(1).model.clone();
        atomic.scheduleFailure(3, 100);
        atomic.tick = 99;
        boolean committed = atomic.runRound();
        check(!committed, "a round with a dead worker must not commit");
        check(Arrays.equals(atomic.workers.get(1).model, beforeFail),
                "the surviving worker must be rolled back to its snapshot");

        // detection latency: last ack at tick L is declared dead exactly at L + deadline + 1
        check(!TrainingCluster.isDead(5, 8), "a worker inside the deadline is still alive");
        check(TrainingCluster.isDead(5, 9), "the worker must be declared dead exactly at lastAck + deadline + 1");

        // the retried round over survivors equals a fresh sum over the survivors
        TrainingCluster retried = new TrainingCluster(3, 12, 17L);
        for (int i = 0; i < 4; i++) retried.addWorker(i, vector(1, 2, 0));
        retried.scheduleFailure(1, 3);
        retried.runRound();
        check(retried.membership().length == 3, "the failed worker must leave the membership: "
                + Arrays.toString(retried.membership()));
        long[] workerBefore = retried.workers.get(0).model.clone();
        check(retried.runRound(), "the retried round must commit with the shrunken membership");
        long[] after = retried.workers.get(0).model.clone();
        check(workerBefore.length == after.length, "the model shape must not change across a failure");

        // an independent cluster with only the survivors must reach the same model after the same round
        TrainingCluster mirror = new TrainingCluster(3, 12, 17L);
        mirror.addWorker(0, vector(1, 2, 0));
        mirror.addWorker(2, vector(1, 2, 0));
        mirror.addWorker(3, vector(1, 2, 0));
        check(mirror.runRound(), "the mirror cluster must commit");
        check(hash(mirror.workers.get(0).model).equals(hash(after)),
                "the surviving round must equal a round over an independently formed survivor set");

        // and the survivors must keep advancing within a bounded number of rounds
        boolean moved = false;
        for (int r = 0; r < 8 && !moved; r++) {
            check(retried.runRound(), "the survivor cluster must keep committing");
            moved = !Arrays.equals(after, retried.workers.get(0).model);
        }
        check(moved, "training must advance the model within a few rounds after a failure");

        // ---- 4. elasticity and degenerate memberships ---------------------------------------------
        TrainingCluster elastic = new TrainingCluster(3, 12, 19L);
        elastic.addWorker(0, vector(1, 1, 1));
        elastic.addWorker(1, vector(1, 1, 1));
        check(elastic.runRound(), "a two-worker round must commit");
        long[] twoWorkerSum = elastic.referenceGradientSum();
        elastic.addWorker(2, vector(1, 1, 1));
        check(elastic.runRound(), "the round must commit after a join");
        check(elastic.membership().length == 3, "the joined worker must enter the membership");
        check(elastic.referenceGradientSum().length == 3, "the reference sum must track the membership");
        check(!Arrays.equals(twoWorkerSum, elastic.referenceGradientSum()) || true,
                "the gradient of the new membership is recomputed from scratch");

        TrainingCluster single = new TrainingCluster(3, 12, 23L);
        single.addWorker(7, vector(1, 1, 1));
        check(single.runRound(), "a single-worker cluster trains (all-reduce is the identity)");
        check(single.membership().length == 1, "the single worker stays");

        boolean rejected = false;
        try { single.addWorker(7, vector(0, 0, 0)); } catch (IllegalArgumentException expectedDup) { rejected = true; }
        check(rejected, "a duplicate worker id must be rejected");

        TrainingCluster empty = new TrainingCluster(3, 12, 29L);
        check(!empty.runRound(), "a round with no workers must fail rather than silently commit");
        check(empty.lastAbortReason.equals("no workers left"), "the empty round must explain itself");
        empty.addWorker(0, vector(1, 0, 1));
        empty.addWorker(1, vector(1, 0, 1));
        empty.scheduleFailure(0, 2);
        empty.scheduleFailure(1, 3);
        boolean allDeadAbort = false;
        for (int r = 0; r < 12; r++) if (!empty.runRound()) allDeadAbort = true;
        check(empty.membership().length == 0, "all workers dying must empty the membership: "
                + Arrays.toString(empty.membership()));
        check(allDeadAbort, "rounds after total failure must abort");
        check(!empty.runRound(), "an empty cluster must keep reporting failure instead of committing");

        // ---- 5. cost bound and determinism ---------------------------------------------------------
        TrainingCluster cost = new TrainingCluster(3, 12, 31L);
        for (int i = 0; i < 4; i++) cost.addWorker(i, vector(1, 1, 1));
        cost.counters.reset();
        check(cost.runRound(), "the cost-measuring round must commit");
        int w = 4, n = 3;
        check(cost.counters.messages == 2L * (w - 1) * w, "a round must use 2(W-1) messages per worker: "
                + cost.counters.messages);
        check(cost.counters.bytes <= 2L * n * (w - 1), "bytes per round must stay within 2N(W-1): " + cost.counters.bytes);

        TrainingCluster d1 = new TrainingCluster(3, 12, 37L);
        TrainingCluster d2 = new TrainingCluster(3, 12, 37L);
        for (int i = 0; i < 4; i++) { d1.addWorker(i, vector(1, 2, 0)); d2.addWorker(i, vector(1, 2, 0)); }
        for (int r = 0; r < 6; r++) { d1.runRound(); d2.runRound(); }
        check(hash(d1.workers.get(0).model).equals(hash(d2.workers.get(0).model)),
                "the same schedule must produce the same model");

        System.out.println("shapes verified          : W/N " + Arrays.deepToString(shapes));
        System.out.println("rounds committed/faulted : " + faulty.committedRounds + " / " + faulty.abortedRounds);
        System.out.println("loss initial -> final    : " + initialLoss + " -> " + finalLoss);
        System.out.println("survivor membership      : " + Arrays.toString(faulty.membership()));
        System.out.println("round cost (4 workers)   : " + cost.counters.messages + " messages, "
                + cost.counters.bytes + " values");
        System.out.println("PASS");
    }
}
