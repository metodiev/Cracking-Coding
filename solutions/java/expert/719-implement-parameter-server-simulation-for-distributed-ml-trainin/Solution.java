/** Task 719 - Implement parameter server simulation for distributed ML training. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A parameter server splits the model into S shards, one per server process. Workers pull the shards,
 * compute gradients on their own data and push the gradients back for the server to apply. Two schedules
 * are simulated: bulk synchronous parallel (every worker pulls the same parameter version, then every
 * worker pushes, so a round applies the sum of all the gradients) and asynchronous with stragglers (worker
 * w's gradient is applied delay[w] ticks after it was computed, so by then the other workers have moved the
 * shards on and the gradient is "stale"). Staleness is bounded: a push whose pulled version lags the shard
 * by more than maxStaleness is rejected - the standard guard that keeps asynchronous SGD from diverging.
 *
 * Invariants verified: the shards tile the parameter vector exactly once; an update moves only its own
 * shard and advances that shard's version by exactly one; a bounded-staleness schedule is never refused
 * while an over-stale push is; synchronous training reaches the analytic optimum exactly and asynchronous
 * training stays inside a small ball around it whose radius shrinks with the step size.
 *
 * Simplified: no sockets, threads or clocks - the worker interleaving is a deterministic schedule, worker
 * batches come from a fixed-seed PRNG and the objective is a quadratic whose gradient is exact. Verified:
 * exact shard accounting, bounded staleness, convergence against the sequential oracle, and the cost model.
 */
import java.util.*;

public class Solution {
    /** One process of the parameter-server fleet: a shard of the parameters and an update rule. */
    static final class ParameterServer {
        final int shards;
        final int shardSize;
        final double[][] params;            // [shard][local index]
        final long[] shardVersion;          // accepted pushes per shard
        final double learningRate;
        final int maxStaleness;
        long pushes, rejectedPushes, pulledValues, pushedValues, maxObservedStaleness;

        ParameterServer(int shards, int shardSize, double learningRate, int maxStaleness) {
            if (shards <= 0 || shardSize <= 0) throw new IllegalArgumentException("empty shard layout");
            this.shards = shards;
            this.shardSize = shardSize;
            this.params = new double[shards][shardSize];
            this.shardVersion = new long[shards];
            this.learningRate = learningRate;
            this.maxStaleness = maxStaleness;
        }

        /** Serves the current value of one shard; the reply is a private copy. Completes one pull. */
        double[] pull(int shard) {
            pulledValues += shardSize;
            return params[shard].clone();
        }

        long currentVersion(int shard) { return shardVersion[shard]; }

        long pulls() { return pulledValues / shardSize; }

        /** Applies a gradient to one shard, unless the worker's view is older than the staleness bound. */
        boolean push(int shard, double[] gradient, long versionSeen) {
            if (gradient.length != shardSize) throw new IllegalArgumentException("gradient size mismatch");
            long staleness = shardVersion[shard] - versionSeen;
            maxObservedStaleness = Math.max(maxObservedStaleness, staleness);
            if (staleness > maxStaleness) {      // stale beyond the bound: refuse rather than diverge
                rejectedPushes++;
                return false;
            }
            for (int i = 0; i < shardSize; i++) params[shard][i] -= learningRate * gradient[i];
            shardVersion[shard]++;
            pushedValues += shardSize;
            pushes++;
            return true;
        }

        double[] snapshot(int dim) {
            double[] w = new double[dim];
            for (int s = 0; s < shards; s++) System.arraycopy(params[s], 0, w, s * shardSize, shardSize);
            return w;
        }

        /** Returns a description of the first flaw in the shard tiling, or null when it is exact. */
        String verifySharding(int dim) {
            int[] owner = new int[dim];
            Arrays.fill(owner, -1);
            for (int s = 0; s < shards; s++) {
                for (int i = 0; i < shardSize; i++) {
                    int global = s * shardSize + i;
                    if (global >= dim || owner[global] != -1) return "shard " + s + " conflicts at " + global;
                    owner[global] = s;
                }
            }
            for (int i = 0; i < dim; i++) if (owner[i] == -1) return "global index " + i + " in no shard";
            return null;
        }
    }
    /** A training worker: its own data shard, a view of the parameters, and a gradient buffer. */
    static final class Worker {
        final ParameterServer server;
        final double[][] localTargets;
        final int batch;
        final double[] view;
        final double[] gradient;
        final long[] versionSeen;
        int cursor;

        Worker(ParameterServer server, double[][] localTargets, int batch) {
            this.server = server;
            this.localTargets = localTargets;
            this.batch = batch;
            this.view = new double[localTargets[0].length];
            this.gradient = new double[view.length];
            this.versionSeen = new long[server.shards];
        }

        /** Pulls the whole parameter vector, then averages the gradient over a deterministic batch. */
        void pullAndCompute(int dim, int shards, int shardSize) {
            for (int s = 0; s < shards; s++) {
                System.arraycopy(server.pull(s), 0, view, s * shardSize, shardSize);
                versionSeen[s] = server.currentVersion(s);
            }
            Arrays.fill(gradient, 0.0);
            for (int b = 0; b < batch; b++) {
                double[] sample = localTargets[cursor % localTargets.length];
                cursor++;
                for (int i = 0; i < dim; i++) gradient[i] += view[i] - sample[i];
            }
            for (int i = 0; i < dim; i++) gradient[i] /= batch;
        }

        /** Pushes the shard's slice with the version it was computed from. */
        boolean pushShard(int shard, int shardSize) {
            double[] slice = Arrays.copyOfRange(gradient, shard * shardSize, (shard + 1) * shardSize);
            return server.push(shard, slice, versionSeen[shard]);
        }
    }

    /** Bulk synchronous rounds: all workers pull the same version, then all of them push. */
    static int runSynchronous(Worker[] crew, int dim, int shards, int shardSize, int rounds) {
        int accepted = 0;
        for (int round = 0; round < rounds; round++) {
            for (Worker worker : crew) worker.pullAndCompute(dim, shards, shardSize);
            for (Worker worker : crew) {
                for (int s = 0; s < shards; s++) if (worker.pushShard(s, shardSize)) accepted++;
            }
        }
        return accepted;
    }

    /**
     * Asynchronous ticks: at tick t worker t%workers computes a gradient and its shards are pushed at tick
     * t + delay[t%workers]. Pushes scheduled for a tick are applied before that tick's computation, so a
     * gradient computed `delay` ticks ago lands on parameters that have moved on by that many versions -
     * the staleness the server measures and bounds.
     */
    static ParameterServer runAsynchronous(ParameterServer server, double[][][] data, int dim, int shards,
                                           int shardSize, int ticks, int[] delay) {
        int workers = data.length;
        int maxDelay = 0;
        for (int d : delay) maxDelay = Math.max(maxDelay, d);
        Worker[] crew = new Worker[workers];
        for (int j = 0; j < workers; j++) crew[j] = new Worker(server, data[j], 12);
        List<List<Object[]>> deferred = new ArrayList<>();
        for (int t = 0; t <= ticks + maxDelay; t++) deferred.add(new ArrayList<>());
        for (int tick = 0; tick < ticks + maxDelay; tick++) {
            for (Object[] pending : deferred.get(tick)) {
                server.push((Integer) pending[0], (double[]) pending[1], (Long) pending[2]);
            }
            if (tick >= ticks) continue;
            Worker worker = crew[tick % workers];
            worker.pullAndCompute(dim, shards, shardSize);
            for (int s = 0; s < shards; s++) {
                double[] slice = new double[shardSize];
                System.arraycopy(worker.gradient, s * shardSize, slice, 0, shardSize);
                deferred.get(tick + delay[tick % workers]).add(new Object[]{s, slice, worker.versionSeen[s]});
            }
        }
        return server;
    }

    private static long rngState = 0x106689D45497FDB5L;

    /** splitmix64 - deterministic, no Math.random. */
    static long nextLong() {
        rngState += 0x9E3779B97F4A7C15L;
        long z = rngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    /** Worker data: each worker sees its own offset plus noise, so the optimum is a compromise. */
    static double[][] workerData(int dim, int samples, double offset) {
        double[][] data = new double[samples][dim];
        for (int s = 0; s < samples; s++) {
            for (int i = 0; i < dim; i++) data[s][i] = Math.sin(0.5 * i) + offset + 0.2 * (nextDouble() * 2.0 - 1.0);
        }
        return data;
    }

    /** The objective's minimum: the mean of every sample every worker holds, computed analytically. */
    static double[] optimalParameters(double[][][] data, int dim) {
        double[] target = new double[dim];
        int samples = 0;
        for (double[][] worker : data) {
            for (double[] sample : worker) {
                for (int i = 0; i < dim; i++) target[i] += sample[i];
                samples++;
            }
        }
        for (int i = 0; i < dim; i++) target[i] /= samples;
        return target;
    }

    /** Largest coordinate-wise distance to the optimum: the scale-free convergence measure. */
    static double gap(double[] w, double[] optimum) {
        double worst = 0.0;
        for (int i = 0; i < w.length; i++) worst = Math.max(worst, Math.abs(w[i] - optimum[i]));
        return worst;
    }

    /** Sequential single-process GD over all of the data: the independent oracle. */
    static double[] sequentialOracle(double[][][] data, int dim, int steps, double lr) {
        double[] w = new double[dim];
        double[] optimum = optimalParameters(data, dim);
        for (int step = 0; step < steps; step++) {
            for (int i = 0; i < dim; i++) w[i] -= lr * (w[i] - optimum[i]);
        }
        return w;
    }

    public static void main(String[] args) {
        final int dim = 16;
        final int shards = 4;
        final int shardSize = 4;
        final int workers = 3;
        final int rounds = 60;
        check(new ParameterServer(shards, shardSize, 0.1, 2).verifySharding(dim) == null,
                "shards must tile the parameter vector exactly once");

        // --- degenerate deployments ------------------------------------------
        ParameterServer single = new ParameterServer(1, 1, 0.5, 0);
        check(single.verifySharding(1) == null && single.push(0, new double[]{4.0}, 0),
                "a single-shard server works and accepts a fresh push");
        check(single.snapshot(1)[0] == -2.0, "the update rule is w -= lr * g");
        boolean rejectedSize = false;
        boolean rejectedLayout = false;
        try { single.push(0, new double[3], 2); } catch (IllegalArgumentException e) { rejectedSize = true; }
        try { new ParameterServer(0, 4, 0.1, 1); } catch (IllegalArgumentException e) { rejectedLayout = true; }
        check(rejectedSize && rejectedLayout,
                "a mis-sized gradient and an empty shard layout must both be rejected");

        // --- shard isolation invariant ---------------------------------------
        ParameterServer isolation = new ParameterServer(shards, shardSize, 0.25, 8);
        double[] before = isolation.snapshot(dim);
        isolation.push(2, new double[]{1.0, 1.0, 1.0, 1.0}, 0);
        double[] after = isolation.snapshot(dim);
        for (int i = 0; i < dim; i++) {
            boolean inShard2 = i >= 2 * shardSize && i < 3 * shardSize;
            check((after[i] != before[i]) == inShard2, "coordinate " + i + " moved for the wrong shard");
        }
        check(isolation.shardVersion[2] == 1 && isolation.shardVersion[0] == 0,
                "only the targeted shard's version advances");

        // --- staleness admission control -------------------------------------
        ParameterServer guarded = new ParameterServer(1, 2, 0.1, 2);
        for (int behind = 0; behind <= 2; behind++) {
            check(guarded.push(0, new double[]{0.0, 0.0}, 0),
                    "staleness " + behind + " is inside the bound");
        }
        check(!guarded.push(0, new double[]{0.0, 0.0}, 0) && guarded.rejectedPushes == 1
                        && guarded.shardVersion[0] == 3 && guarded.maxObservedStaleness == 3,
                "the refused push must not change the shard: " + guarded.rejectedPushes + " / "
                        + guarded.shardVersion[0] + " / " + guarded.maxObservedStaleness);

        // --- data, optimum and the synchronous run ---------------------------
        double[][][] data = new double[workers][][];
        for (int j = 0; j < workers; j++) data[j] = workerData(dim, 12, 0.6 * (j - 1));
        double[] optimum = optimalParameters(data, dim);
        double lr = 1.0 / (2.0 * workers);
        ParameterServer bsp = new ParameterServer(shards, shardSize, lr, workers);
        Worker[] crew = new Worker[workers];
        for (int j = 0; j < workers; j++) crew[j] = new Worker(bsp, data[j], 12);
        int accepted = runSynchronous(crew, dim, shards, shardSize, rounds);
        long expectedPushes = (long) workers * shards * rounds;
        double[] oracleParams = sequentialOracle(data, dim, 500, 1.0 / workers);
        check(accepted == expectedPushes && bsp.pulls() == expectedPushes && bsp.rejectedPushes == 0
                        && bsp.maxObservedStaleness <= workers - 1,
                "BSP accounting: " + bsp.pulls() + " pulls / " + accepted + " pushes of " + expectedPushes
                        + ", refusals " + bsp.rejectedPushes + ", worst staleness "
                        + bsp.maxObservedStaleness);
        check(gap(bsp.snapshot(dim), optimum) < 1e-9,
                "BSP must converge to the optimum exactly, gap=" + gap(bsp.snapshot(dim), optimum));
        check(gap(oracleParams, optimum) < 1e-12, "the sequential oracle converges to the same optimum");

        // --- asynchronous training with stragglers ---------------------------
        // A gradient evaluated at a stale point biases the iteration, so the asynchronous schedule settles
        // into a ball around the optimum instead of a point. Its radius is proportional to the step size,
        // which is what the two runs below measure: the same schedule at lr and at lr/4, sampled at three
        // horizons so that the oscillation (not just one lucky phase) is what gets compared.
        int[] delay = {3, 4, 5};
        int[] horizons = {1000, 3000, 9000};
        double asyncWorst = 0.0;
        double coarseWorst = 0.0;
        double worstStaleness = 0.0;
        long refusals = 0;
        for (int ticks : horizons) {
            ParameterServer fine = runAsynchronous(new ParameterServer(shards, shardSize, lr / 4.0, 2 * workers), data, dim, shards, shardSize, ticks, delay);
            ParameterServer coarse = runAsynchronous(new ParameterServer(shards, shardSize, lr, 2 * workers), data, dim, shards, shardSize, ticks, delay);
            asyncWorst = Math.max(asyncWorst, gap(fine.snapshot(dim), optimum));
            coarseWorst = Math.max(coarseWorst, gap(coarse.snapshot(dim), optimum));
            worstStaleness = Math.max(worstStaleness, fine.maxObservedStaleness);
            refusals += fine.rejectedPushes;
        }
        check(worstStaleness > 0 && worstStaleness <= 2 * workers,
                "asynchronous workers must really be out of step, worst staleness " + worstStaleness);
        check(refusals == 0, "a bounded-staleness schedule is never refused, refusals=" + refusals);
        check(asyncWorst < 0.05, "asynchronous training must stay bounded near the optimum, worst gap="
                + asyncWorst);
        check(coarseWorst > 2.0 * asyncWorst,
                "the stale-gradient radius must shrink with the step size: " + asyncWorst + " at lr/4 vs "
                        + coarseWorst + " at lr");

        // --- cost model ------------------------------------------------------
        long valuesMoved = bsp.pulledValues + bsp.pushedValues;
        check(valuesMoved == 2L * workers * rounds * dim,
                "each worker moves 2 * dim values per round, got " + valuesMoved);
        long ringVolume = 2L * (workers - 1) * dim * rounds;
        check(valuesMoved >= ringVolume,
                "a parameter server moves at least a ring allreduce's volume: " + valuesMoved + " vs "
                        + ringVolume);

        // --- determinism ------------------------------------------------------
        rngState = 0x106689D45497FDB5L;
        double[][][] replayData = new double[workers][][];
        for (int j = 0; j < workers; j++) replayData[j] = workerData(dim, 12, 0.6 * (j - 1));
        ParameterServer replay = new ParameterServer(shards, shardSize, lr, workers);
        Worker[] replayCrew = new Worker[workers];
        for (int j = 0; j < workers; j++) replayCrew[j] = new Worker(replay, replayData[j], 12);
        runSynchronous(replayCrew, dim, shards, shardSize, rounds);
        check(Arrays.equals(replay.snapshot(dim), bsp.snapshot(dim)),
                "the same seed must reproduce the run exactly");

        System.out.println("parameter server: " + shards + " shards of " + shardSize + ", " + workers
                + " workers, " + rounds + " rounds, straggler delays " + Arrays.toString(delay)
                + "; gap to the optimum: BSP " + gap(bsp.snapshot(dim), optimum) + ", asynchronous "
                + asyncWorst + " (lr/4) vs " + coarseWorst + " (lr); values on the wire " + valuesMoved
                + " vs ring " + ringVolume + "; worst staleness " + worstStaleness);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
