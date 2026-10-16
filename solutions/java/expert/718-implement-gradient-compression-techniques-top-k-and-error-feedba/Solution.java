/** Task 718 - Implement gradient compression techniques (top-k) and error feedback. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * Distributed SGD is bandwidth bound, so a worker sends only the k largest-magnitude entries of its
 * gradient (top-k sparsification). Plain top-k destroys information: a coordinate that misses the top k
 * sees its gradient dropped for ever, so the update the model receives is a biased, ever-widening
 * approximation of the true gradient. Error feedback repairs that with one extra accumulator: the worker
 * folds the fresh gradient into a residual r, transmits s = topk(r + g) and keeps the remainder,
 * r' = (r + g) - s. Discarded mass is *deferred*, never destroyed. After a coordinate is transmitted its
 * residual is zero, so it drops out of the top k and the batch rotates through all coordinates.
 *
 * Invariants checked here (all exact, because the harness uses binary-representable numbers):
 *   - bookkeeping   r' == (r + g) - s, elementwise and bitwise (the update uses the same IEEE sequence,
 *                   so this is an identity, not a tolerance)
 *   - dominance     min |s| >= max |not transmitted|; the transmitted set is the brute-force top-k
 *   - conservation  sum of transmitted updates + final residual == sum of all gradients seen
 *   - bounded lag   ||residual||_1 stays inside one rotation window whatever the run length, while plain
 *                   top-k's shortfall is k/dim of the total and grows linearly
 *
 * Simplified: the transmitted vector is a dense staging array with the non-selected entries zeroed, not a
 * real sparse (index, value) wire format, and the payload is a deterministic gradient harness rather than
 * a neural network. Verified: no coordinate is starved and the integrated update equals the integrated
 * gradient up to a bounded lag, while plain top-k loses a fixed fraction of it for ever.
 */
import java.util.*;

public class Solution {
    /** Deterministic top-k selector: k largest magnitudes, ties broken by the smaller index. */
    static final class TopKSelector {
        final int dim;
        final int k;
        final int[] chosen;
        private final int[] order;

        TopKSelector(int dim, int k) {
            if (k < 0 || k > dim) throw new IllegalArgumentException("k out of range: " + k);
            this.dim = dim;
            this.k = k;
            this.order = new int[dim];
            this.chosen = new int[k];
        }

        /** Places the indices of the top-k entries of `vector` into the selection buffer. */
        int select(double[] vector) {
            if (vector.length != dim) throw new IllegalArgumentException("dimension mismatch");
            for (int i = 0; i < dim; i++) order[i] = i;
            for (int slot = 0; slot < k; slot++) {
                int best = slot;
                for (int i = slot + 1; i < dim; i++) {
                    if (better(vector, order[i], order[best])) best = i;
                }
                int swap = order[slot];
                order[slot] = order[best];
                order[best] = swap;
                chosen[slot] = order[slot];
            }
            return k;
        }

        /** Strict total order: larger magnitude first, smaller index first on exact ties. */
        private static boolean better(double[] vector, int a, int b) {
            double magnitudeA = Math.abs(vector[a]);
            double magnitudeB = Math.abs(vector[b]);
            if (magnitudeA != magnitudeB) return magnitudeA > magnitudeB;
            return a < b;
        }
    }

    /** Top-k sparsification with or without error feedback - the two variants the task contrasts. */
    static final class Sparsifier {
        final TopKSelector selector;
        final boolean feedback;
        final double[] residual;           // deferred mass, survives across rounds
        final double[] transmitted;        // dense staging buffer for one sparse update
        final double[] deliveredMass;      // cumulative value actually sent per coordinate
        final double[] seenGradient;       // cumulative gradient per coordinate
        final int[] transmitCount;
        final int[] firstTransmitRound;    // 1-based round of the first transmission, 0 if never
        long transmittedValues;
        long rounds;

        Sparsifier(int dim, int k, boolean feedback) {
            selector = new TopKSelector(dim, k);
            this.feedback = feedback;
            residual = new double[dim];  transmitted = new double[dim];
            deliveredMass = new double[dim];  seenGradient = new double[dim];
            transmitCount = new int[dim];  firstTransmitRound = new int[dim];
        }

        /** Collects a fresh gradient, transmits the top-k, and (with feedback) defers the remainder. */
        int step(double[] gradient) {
            int dim = residual.length;
            if (gradient.length != dim) throw new IllegalArgumentException("dimension mismatch");
            Arrays.fill(transmitted, 0.0);
            for (int i = 0; i < dim; i++) {
                seenGradient[i] += gradient[i];
                if (feedback) residual[i] += gradient[i];
            }
            int sent = selector.select(feedback ? residual : gradient);
            int[] index = selector.chosen;
            for (int slot = 0; slot < sent; slot++) {
                int i = index[slot];
                transmitted[i] = feedback ? residual[i] : gradient[i];
                if (feedback) residual[i] -= transmitted[i];   // the sent part stops being deferred
                deliveredMass[i] += transmitted[i];
                transmitCount[i]++;
                if (firstTransmitRound[i] == 0) firstTransmitRound[i] = (int) rounds + 1;
            }
            transmittedValues += sent;
            rounds++;
            return sent;
        }
    }

    private static long rngState = 0x243F6A8885A308D3L;

    /** splitmix64 - deterministic, no Math.random. */
    static long nextLong() {
        rngState += 0x9E3779B97F4A7C15L;
        long z = rngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Uniform in [0, 1). */
    static double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    /** Gradient whose coordinate magnitudes span decades: the regime that exposes top-k starvation. */
    static double[] heterogeneousGradient(int dim) {
        double[] g = new double[dim];
        for (int i = 0; i < dim; i++) g[i] = Math.pow(10.0, -((i % 10) / 3.0)) * (nextDouble() * 2.0 - 1.0);
        return g;
    }

    /** Reference selection: sort indices by (magnitude descending, index ascending) and cut. */
    static List<Integer> bruteForceTopK(double[] vector, int k) {
        Integer[] order = new Integer[vector.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> {
            int byMagnitude = Double.compare(Math.abs(vector[b]), Math.abs(vector[a]));
            return byMagnitude != 0 ? byMagnitude : Integer.compare(a, b);
        });
        List<Integer> top = new ArrayList<>(Arrays.asList(order).subList(0, Math.min(k, order.length)));
        Collections.sort(top);
        return top;
    }

    /**
     * SGD on the diagonal quadratic sum 0.5 * c_i (w_i - t_i)^2 with c_i = 1 + 3 (i mod 4) and
     * t_i = sin(0.7 i) (1 + 0.5 (i mod 3)), compressing each gradient with top-k. Returns the final loss,
     * which must approach zero: error feedback is lossless in the limit even at k/dim density. Minibatch
     * noise is out of scope here, the gradient is exact.
     */
    static double runSgd(int dim, int k, int steps, double lr, boolean feedback) {
        double[] w = new double[dim];
        Sparsifier compressor = k >= dim ? null : new Sparsifier(dim, k, feedback);
        for (int step = 0; step < steps; step++) {
            double[] g = new double[dim];
            for (int i = 0; i < dim; i++) {
                g[i] = (1.0 + 3.0 * (i % 4)) * (w[i] - Math.sin(0.7 * i) * (1.0 + 0.5 * (i % 3)));
            }
            double[] update = g;
            if (compressor != null) {
                compressor.step(g);
                update = compressor.transmitted;
            }
            for (int i = 0; i < dim; i++) w[i] -= lr * update[i];
        }
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            double d = w[i] - Math.sin(0.7 * i) * (1.0 + 0.5 * (i % 3));
            sum += 0.5 * (1.0 + 3.0 * (i % 4)) * d * d;
        }
        return sum;
    }

    public static void main(String[] args) {
        int dim = 32;
        int k = 3;

        // --- degenerate and rejected inputs ----------------------------------
        check(new TopKSelector(0, 0).select(new double[0]) == 0, "an empty gradient transmits nothing");
        boolean rejectedK = false;
        boolean rejectedDim = false;
        try { new TopKSelector(4, 5); } catch (IllegalArgumentException e) { rejectedK = true; }
        try { new TopKSelector(4, 2).select(new double[3]); } catch (IllegalArgumentException e) { rejectedDim = true; }
        check(rejectedK && rejectedDim, "k > dim and a wrong-sized gradient must both be rejected");

        // --- selection correctness against the brute-force oracle ------------
        TopKSelector tieCutter = new TopKSelector(4, 1);
        tieCutter.select(new double[]{0.5, -0.5, 1.0, 0.2});
        check(tieCutter.chosen[0] == 2, "the largest magnitude wins over a tie elsewhere");
        tieCutter.select(new double[]{0.25, 0.25, 0.1, 0.2});
        check(tieCutter.chosen[0] == 0, "an exact tie resolves to the smaller index");

        double[] sample = new double[dim];
        for (int i = 0; i < dim; i++) sample[i] = nextDouble() * 2.0 - 1.0;
        sample[7] = 0.25; sample[9] = -0.25;
        double[] untouched = sample.clone();
        TopKSelector selector = new TopKSelector(dim, 5);
        int sent = selector.select(sample);
        List<Integer> got = new ArrayList<>();
        for (int i = 0; i < sent; i++) got.add(selector.chosen[i]);
        Collections.sort(got);
        check(got.equals(bruteForceTopK(sample, 5)), "index set " + got + " != brute force");
        check(Arrays.equals(sample, untouched), "selection must not mutate the input");
        double minSent = Double.POSITIVE_INFINITY;
        double maxDropped = 0.0;
        for (int i = 0; i < dim; i++) {
            if (Collections.binarySearch(got, i) >= 0) minSent = Math.min(minSent, Math.abs(sample[i]));
            else maxDropped = Math.max(maxDropped, Math.abs(sample[i]));
        }
        check(minSent >= maxDropped, "a transmitted entry must dominate every dropped entry");

        // --- k = 0 and k = dim boundaries ------------------------------------
        Sparsifier silent = new Sparsifier(dim, 0, true);
        Sparsifier lossless = new Sparsifier(dim, dim, true);
        check(silent.step(sample.clone()) == 0 && allZero(silent.transmitted),
                "k = 0 produces an all-zero update");
        lossless.step(sample.clone());
        check(Arrays.equals(lossless.transmitted, sample) && allZero(lossless.residual),
                "k = dim reproduces the gradient exactly and defers nothing");

        // --- per-round invariants on heterogeneous gradients -----------------
        Sparsifier ef = new Sparsifier(dim, k, true);
        double largestAccumulator = 0.0;
        for (int round = 0; round < 200; round++) {
            double[] g = heterogeneousGradient(dim);
            double[] residualBefore = ef.residual.clone();
            check(ef.step(g) == k, "each round must transmit exactly k entries, round " + round);
            double smallestSent = Double.POSITIVE_INFINITY;
            for (int slot = 0; slot < k; slot++) {
                smallestSent = Math.min(smallestSent,
                        Math.abs(ef.transmitted[ef.selector.chosen[slot]]));
            }
            for (int i = 0; i < dim; i++) {
                double accumulated = residualBefore[i] + g[i];
                largestAccumulator = Math.max(largestAccumulator, Math.abs(accumulated));
                check(ef.residual[i] == accumulated - ef.transmitted[i],
                        "bookkeeping broken at " + i + " round " + round);
                check(ef.residual[i] == 0.0 || Math.abs(ef.residual[i]) <= smallestSent,
                        "a leftover sits above the transmitted threshold at " + i);
            }
        }
        check(ef.transmittedValues == 200L * k, "the transmission budget is exactly k per round");
        double maxResidual = 0.0;
        for (int i = 0; i < dim; i++) {
            maxResidual = Math.max(maxResidual, Math.abs(ef.residual[i]));
            check(Math.abs(ef.deliveredMass[i] + ef.residual[i] - ef.seenGradient[i]) < 1e-9,
                    "global conservation broken at coordinate " + i);
        }
        check(maxResidual <= largestAccumulator, "the residual stays within the accumulator range");
        // --- the mechanism: a stationary gradient, exactly representable -----
        // Every coordinate carries gradient 1.0, so a coordinate that is not transmitted is straight lost
        // update mass. Plain top-k re-selects the same k coordinates for ever; error feedback selects on
        // the accumulator, so a transmitted coordinate resets to zero and drops out of the top k, and the
        // batch rotates - k fresh coordinates per round, covering all of them within ceil(dim/k) rounds.
        final int pivotDim = 32;
        final int pivotK = 3;
        final int coverage = (pivotDim + pivotK - 1) / pivotK;
        double[] stationary = new double[pivotDim];
        Arrays.fill(stationary, 1.0);
        Sparsifier rotation = new Sparsifier(pivotDim, pivotK, true);
        Sparsifier starvation = new Sparsifier(pivotDim, pivotK, false);
        Sparsifier shortRun = new Sparsifier(pivotDim, pivotK, true);
        Sparsifier halfRun = new Sparsifier(pivotDim, pivotK, false);
        for (int round = 0; round < 4000; round++) {
            rotation.step(stationary);
            starvation.step(stationary);
            if (round < 2000) halfRun.step(stationary);
            if (round < 100) shortRun.step(stationary);
        }
        int lastFirstRound = 0;
        int reached = 0;
        int starved = 0;
        for (int i = 0; i < pivotDim; i++) {
            lastFirstRound = Math.max(lastFirstRound, rotation.firstTransmitRound[i]);
            if (rotation.transmitCount[i] > 0) reached++;
            if (starvation.transmitCount[i] == 0) starved++;
        }
        double totalSeen = sum(rotation.seenGradient);
        double residualMass = sumAbs(rotation.residual);
        double shortResidual = sumAbs(shortRun.residual);
        double plainDelivered = sum(starvation.deliveredMass);
        double efFraction = sum(rotation.deliveredMass) / totalSeen;
        double plainFraction = plainDelivered / totalSeen;
        double shortFraction = sum(shortRun.deliveredMass) / (totalSeen / 40.0);
        double fullShortfall = totalSeen - plainDelivered;
        double halfShortfall = sum(halfRun.seenGradient) - sum(halfRun.deliveredMass);
        check(reached == pivotDim && lastFirstRound == coverage,
                "error feedback covers all " + pivotDim + " coordinates in ceil(dim/k) = " + coverage
                        + " rounds, got " + reached + " coordinates and round " + lastFirstRound);
        check(starved == pivotDim - pivotK && plainDelivered == pivotK * 4000.0,
                "plain top-k feeds only its k favourites for ever: " + starved + " starved, delivered "
                        + plainDelivered + " of " + totalSeen);
        check(residualMass <= pivotDim * (coverage + 2) && shortResidual <= pivotDim * (coverage + 2),
                "the deferred mass stays inside one rotation window at 100 and at 4000 rounds: "
                        + shortResidual + " / " + residualMass);
        check(sum(rotation.deliveredMass) + residualMass == totalSeen,
                "the delivered mass plus the deferred mass is the whole gradient mass");
        check(efFraction > 0.995 && efFraction > shortFraction,
                "the delivered fraction improves with the run length: " + shortFraction + " -> " + efFraction);
        check(Math.abs(plainFraction - (double) pivotK / pivotDim) < 1e-12,
                "plain top-k delivers exactly k/dim of the mass: " + plainFraction);
        check(halfShortfall * 2.0 == fullShortfall && fullShortfall > 500.0 * residualMass,
                "the plain shortfall doubles with the rounds: " + halfShortfall + " -> " + fullShortfall
                        + " against a bounded lag of " + residualMass);

        // --- the compressed run still trains ---------------------------------
        // Error feedback hands over a batch of accumulated gradients at once, so its effective step is
        // roughly dim/k times larger than plain SGD's; the learning rate is scaled accordingly.
        double lr = 0.02;
        double exactLoss = runSgd(dim, dim, 4000, lr, false);
        double efLoss = runSgd(dim, k, 4000, lr, true);
        check(exactLoss < 1e-20, "full-precision SGD reaches the optimum, loss=" + exactLoss);
        check(efLoss < 1e-20 && efLoss <= exactLoss * 10.0,
                "compressed SGD with error feedback reaches it too, loss=" + efLoss);
        check(runSgd(dim, k, 4000, lr, true) == efLoss, "the same seed must repeat exactly");

        System.out.println("top-k sparsification, dim = " + dim + ", k = " + k + " (" + (100 * k / dim)
                + "% density: " + (dim / k) + "x fewer floats on the wire); stationary gradient: full coverage in "
                + lastFirstRound + " rounds with error feedback (ceil(dim/k) = " + coverage + "), " + starved
                + " coordinates never reached by plain top-k");
        System.out.println("delivered mass fraction: " + efFraction + " with error feedback vs "
                + plainFraction + " plain; deferred mass " + residualMass + " (bounded) vs shortfall "
                + fullShortfall + " (grows linearly)");
        System.out.println("final loss: full precision " + exactLoss + ", error feedback " + efLoss);
        System.out.println("PASS");
    }
    static boolean allZero(double[] vector) {
        for (double v : vector) if (v != 0.0) return false;
        return true;
    }

    static double sum(double[] vector) {
        double total = 0.0;
        for (double v : vector) total += v;
        return total;
    }

    static double sumAbs(double[] vector) {
        double total = 0.0;
        for (double v : vector) total += Math.abs(v);
        return total;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
