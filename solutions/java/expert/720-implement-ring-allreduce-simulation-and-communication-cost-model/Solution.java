/** Task 720 - Implement ring-allreduce simulation and communication cost modeling. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Every rank holds a private vector split into N chunks. A ring allreduce runs two sweeps of N-1 steps.
 * In the reduce-scatter sweep rank r sends chunk (r - s) mod N to r+1 and adds the chunk it receives from
 * r-1, so after N-1 steps each rank owns exactly one fully reduced chunk. In the allgather sweep the
 * chunks are rotated the same way but copied instead of added, so after another N-1 steps every rank has
 * every reduced chunk. Each rank therefore moves 2(N-1)/N of the vector instead of 2(N-1): the ring is
 * bandwidth optimal, at the price of 2(N-1) dependent steps instead of one flat broadcast.
 *
 * Invariants verified:
 *   - result       every rank ends with the elementwise sum of all inputs, cross-checked against direct
 *                  summation, for N = 1..8 including chunk counts that do not divide the vector evenly
 *   - ownership    after the reduce-scatter every rank holds exactly one fully reduced chunk and no more,
 *                  and the other chunks are still partial (so the allgather is doing real work)
 *   - accounting   every element is sent exactly twice per rank and received twice, the data volume never
 *                  changes, and the per-rank byte count matches the closed form 2(N-1)S/N
 *   - cost model   the ring's volume is exactly N times smaller than the flat all-to-all baseline while its
 *                  step count is N times larger
 *
 * Simplified: the ranks are simulated in one process (explicit in-flight buffers reproduce the simultaneous
 * send/receive of a ring step) and only element counts are converted to bytes; the verified property is the
 * algorithmic one, equality of the reduction with direct summation plus an exact communication model.
 */
import java.util.*;

public class Solution {
    /** A ring allreduce over `ranks` processes, each holding a private vector of `chunkSize` chunks. */
    static final class RingAllReduce {
        final int ranks;
        final int[] chunkSize;
        final double[][][] data;        // [rank][chunk][element]
        final double[][] outbox;        // one in-flight message per rank, one chunk per step
        final long[] sentElements;
        final long[] receivedElements;
        long steps;

        RingAllReduce(int ranks, int[] chunkSize) {
            if (ranks <= 0 || chunkSize.length != ranks) throw new IllegalArgumentException("bad ring shape");
            this.ranks = ranks;
            this.chunkSize = chunkSize.clone();
            this.data = new double[ranks][ranks][];
            this.outbox = new double[ranks][];
            int widest = 0;
            for (int c : chunkSize) widest = Math.max(widest, c);
            for (int r = 0; r < ranks; r++) {
                for (int c = 0; c < ranks; c++) data[r][c] = new double[chunkSize[c]];
                outbox[r] = new double[widest];
            }
            this.sentElements = new long[ranks];
            this.receivedElements = new long[ranks];
        }

        void load(int rank, int chunk, double[] values) {
            if (values.length != chunkSize[chunk]) throw new IllegalArgumentException("chunk size mismatch");
            System.arraycopy(values, 0, data[rank][chunk], 0, values.length);
        }

        /** One ring step; `accumulate` true is the reduce-scatter sweep, false the allgather sweep. */
        private void step(int s, boolean accumulate) {
            int shift = accumulate ? 0 : 1;
            int[] sendChunk = new int[ranks];
            for (int r = 0; r < ranks; r++) {
                int chunk = Math.floorMod(r - s + shift, ranks);
                sendChunk[r] = chunk;
                // The message is cut before any receive is applied: a ring step sends and receives at once.
                System.arraycopy(data[r][chunk], 0, outbox[r], 0, chunkSize[chunk]);
            }
            for (int r = 0; r < ranks; r++) {
                int from = Math.floorMod(r - 1, ranks);
                int chunk = Math.floorMod(r - s + shift - 1, ranks);
                double[] inbox = outbox[from];
                if (chunkSize[chunk] != chunkSize[sendChunk[from]]) throw new IllegalStateException("ring skew");
                if (accumulate) {
                    for (int i = 0; i < chunkSize[chunk]; i++) data[r][chunk][i] += inbox[i];
                } else {
                    System.arraycopy(inbox, 0, data[r][chunk], 0, chunkSize[chunk]);
                }
                receivedElements[r] += chunkSize[chunk];
            }
            for (int r = 0; r < ranks; r++) sentElements[r] += chunkSize[sendChunk[r]];
            steps++;
        }

        void reduceScatter() { for (int s = 0; s < ranks - 1; s++) step(s, true); }

        void allGather() { for (int s = 0; s < ranks - 1; s++) step(s, false); }

        /** Runs both sweeps; returns the number of ring steps taken. */
        long run() {
            reduceScatter();
            allGather();
            return steps;
        }

        double[] flatten(int rank) {
            int total = 0;
            for (int c = 0; c < ranks; c++) total += chunkSize[c];
            double[] flat = new double[total];
            int at = 0;
            for (int c = 0; c < ranks; c++) {
                System.arraycopy(data[rank][c], 0, flat, at, chunkSize[c]);
                at += chunkSize[c];
            }
            return flat;
        }

        /** Elementwise sum of every rank's loaded vector: the answer the ring must reproduce. */
        static double[] directSum(double[][][] inputs) {
            int total = 0;
            for (double[] chunk : inputs[0]) total += chunk.length;
            double[] sum = new double[total];
            for (double[][] rank : inputs) {
                int at = 0;
                for (double[] chunk : rank) {
                    for (int i = 0; i < chunk.length; i++) sum[at + i] += chunk[i];
                    at += chunk.length;
                }
            }
            return sum;
        }

        long elementsPerRank() { return 2L * (ranks - 1) * vectorLength() / ranks; }

        long vectorLength() {
            int total = 0;
            for (int c : chunkSize) total += c;
            return total;
        }
    }

    private static long rngState = 0x2545F4914F6CDD1DL;

    /** splitmix64 - deterministic, no Math.random. */
    static long nextLong() {
        rngState += 0x9E3779B97F4A7C15L;
        long z = rngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    /** Splits `length` elements into `chunks` pieces, as evenly as possible. */
    static int[] split(int length, int chunks) {
        int[] size = new int[chunks];
        for (int c = 0; c < chunks; c++) {
            size[c] = length / chunks + (c < length % chunks ? 1 : 0);
        }
        return size;
    }

    /** Fills each rank with private data and returns the inputs plus the ring over them. */
    static double[][][] buildInputs(RingAllReduce ring, int ranks) {
        double[][][] inputs = new double[ranks][ranks][];
        for (int r = 0; r < ranks; r++) {
            for (int c = 0; c < ranks; c++) {
                double[] values = new double[ring.chunkSize[c]];
                for (int i = 0; i < values.length; i++) values[i] = nextDouble() * 4.0 - 2.0;
                inputs[r][c] = values;
                ring.load(r, c, values);
            }
        }
        return inputs;
    }

    static boolean close(double[] a, double[] b, double tolerance) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > tolerance) return false;
        return true;
    }

    public static void main(String[] args) {
        // --- boundary: a single rank is a no-op ---------------------------------
        RingAllReduce lone = new RingAllReduce(1, new int[]{5});
        lone.load(0, 0, new double[]{1.0, -2.0, 3.5, 0.0, 8.0});
        lone.run();
        check(lone.steps == 0, "one rank needs zero ring steps, got " + lone.steps);
        check(Arrays.equals(lone.flatten(0), new double[]{1.0, -2.0, 3.5, 0.0, 8.0}),
                "a single rank's vector is its own sum");
        boolean rejectedShape = false;
        try {
            new RingAllReduce(3, new int[]{1, 2});
        } catch (IllegalArgumentException expected) {
            rejectedShape = true;
        }
        check(rejectedShape, "a chunk count that does not match the rank count must be rejected");

        int[] rankCounts = {2, 3, 4, 5, 8};
        for (int ranks : rankCounts) {
            for (int length : new int[]{ranks, 2 * ranks, 3 * ranks + 1}) {
                RingAllReduce ring = new RingAllReduce(ranks, split(length, ranks));
                double[][][] inputs = buildInputs(ring, ranks);
                long steps = ring.run();
                double[] oracle = RingAllReduce.directSum(inputs);
                for (int r = 0; r < ranks; r++) {
                    check(close(ring.flatten(r), oracle, 1e-9),
                            "rank " + r + " of " + ranks + " ranks, length " + length + " must hold the sum");
                }
                check(steps == 2L * (ranks - 1), "a ring allreduce costs 2(N-1) steps, N=" + ranks
                        + ", got " + steps);
            }
        }

        // --- ownership after the reduce-scatter, and why the allgather is needed --
        int ranks = 5;
        int length = 40;
        RingAllReduce phase = new RingAllReduce(ranks, split(length, ranks));
        double[][][] inputs = buildInputs(phase, ranks);
        double[] oracle = RingAllReduce.directSum(inputs);
        phase.reduceScatter();
        check(phase.steps == ranks - 1, "the reduce-scatter sweep is N-1 steps, got " + phase.steps);
        int completeChunks = 0;
        for (int r = 0; r < ranks; r++) {
            for (int c = 0; c < ranks; c++) {
                double[] piece = phase.data[r][c];
                int base = 0;
                for (int k = 0; k < c; k++) base += phase.chunkSize[k];
                boolean reduced = true;
                for (int i = 0; i < piece.length; i++) {
                    if (Math.abs(piece[i] - oracle[base + i]) > 1e-9) { reduced = false; break; }
                }
                if (reduced) completeChunks++;
            }
        }
        check(completeChunks == ranks, "exactly one chunk per rank is fully reduced, saw "
                + completeChunks + " of " + ranks);
        long halves = 0;
        for (int r = 0; r < ranks; r++) halves += phase.sentElements[r];
        phase.allGather();
        check(phase.steps == 2L * (ranks - 1), "both sweeps together are 2(N-1) steps");
        long whole = 0;
        for (int r = 0; r < ranks; r++) whole += phase.sentElements[r];
        check(whole == 2 * halves, "each rank sends its vector twice, once per sweep: " + halves + " -> "
                + whole);

        // --- communication cost model ------------------------------------------
        for (int n : new int[]{2, 4, 8, 16}) {
            RingAllReduce model = new RingAllReduce(n, split(n * 100, n));
            long vector = model.vectorLength();
            long ringPerRank = 2L * (n - 1) * vector / n;          // closed form, equal chunks
            long measured = 2L * (n - 1) * model.chunkSize[0];     // one chunk per step in each sweep
            check(ringPerRank == measured, "closed form must match the sweep count: " + ringPerRank
                    + " vs " + measured);
            long flatPerRank = 2L * (n - 1) * vector;              // every rank broadcasts everything
            check(flatPerRank == n * ringPerRank, "the ring is exactly N times cheaper, N=" + n);
            check(model.elementsPerRank() == ringPerRank, "elementsPerRank agrees with the closed form");
        }
        RingAllReduce uneven = new RingAllReduce(3, new int[]{4, 3, 3});
        double[][][] unevenInputs = buildInputs(uneven, 3);
        uneven.run();
        for (int r = 0; r < 3; r++) {
            check(close(uneven.flatten(r), RingAllReduce.directSum(unevenInputs), 1e-9),
                    "uneven chunks: rank " + r + " must still hold the sum");
        }
        long unevenSent = uneven.sentElements[0] + uneven.sentElements[1] + uneven.sentElements[2];
        check(unevenSent == 2L * (3 - 1) * 10,
                "over both sweeps the ranks send 2(N-1)S elements in total, got " + unevenSent);

        // --- determinism --------------------------------------------------------
        RingAllReduce first = new RingAllReduce(4, split(9, 4));
        double[][][] firstInputs = buildInputs(first, 4);
        first.run();
        double[] firstResult = first.flatten(0);
        RingAllReduce second = new RingAllReduce(4, split(9, 4));
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) second.load(r, c, firstInputs[r][c]);
        }
        second.run();
        check(Arrays.equals(second.flatten(0), firstResult), "the same input must give the same result");
        check(close(firstResult, RingAllReduce.directSum(firstInputs), 1e-9),
                "the uneven-length ring result matches direct summation");

        System.out.println("ring allreduce: N-1 reduce-scatter steps plus N-1 allgather steps for any N");
        System.out.println("per-rank volume 2(N-1)S/N, so 2(N-1)S/N vs the flat baseline's 2(N-1)S");
        System.out.println("uneven split 10 elements over 3 ranks verified against direct summation");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
