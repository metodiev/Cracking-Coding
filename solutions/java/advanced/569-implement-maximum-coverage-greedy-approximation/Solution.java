/** Task 569 - Implement maximum coverage greedy approximation. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Maximum coverage: given a universe of weighted elements and m sets, choose k sets maximising the
 * total weight of the covered elements. The classical greedy repeatedly picks the set with the
 * largest marginal gain and is guaranteed to reach at least (1 - 1/e) OPT. The implementation keeps
 * the universe in a single long bitmask (universe <= 64), so coverage is a handful of word
 * operations, and the greedy takes an explicit tie-breaking order array, which lets the self-test
 * re-run it under every possible ordering of the input list.
 *
 * Documented checks. (1) Hand-computed instance: sets {0,1,2}, {2,3}, {3,4}, {0,1,2,4} with k = 2;
 * the largest set is taken first and the greedy ends at coverage 5, which equals the brute-force
 * optimum. (2) *Exhaustive* verification of the guarantee: all 65536 set systems with a 4-element
 * universe and 4 sets are enumerated (16^4 combinations, the all-empty system excluded because its
 * optimum is 0 and the ratio 0/0 is undefined) and compared with the brute-force optimum for
 * k = 1, 2, 3, 4. The worst observed ratios over the whole enumeration are 1.0, 0.75, 1.0 and 1.0;
 * all are at or above 1 - 1/e = 0.6321 and at or above the per-k bound 1 - (1 - 1/k)^k. Only k = 2
 * attains its bound, and the witness system {0}, {0,1}, {0,2}, {1,3} is printed and re-checked
 * individually (greedy 3 of OPT 4 = 0.75): that is the concrete counterexample showing the greedy is
 * not exact. For k = 3 over this small universe the greedy is in fact always optimal, so the
 * theoretical 0.7037 bound is not attained there — stated as observed rather than glossed over.
 * A second family (5-element universe, 5 sets, 20000 splitmix64 samples) has worst ratio 0.8 at
 * k = 3, which is above the 0.7037 bound, again with the guarantee asserted on every sample.
 * (3) The proof's invariant is verified step by step on a 12-element / 8-set instance with k = 4:
 * after step i the uncovered part of the optimum satisfies OPT - f(S_i) <= k * g_{i+1}, the next
 * greedy gain, and f(S_i) >= (1 - (1 - 1/k)^i) OPT. Both hold at every step (cumulative coverage
 * 8, 10, 12, 12 against OPT 12; gains 8, 2, 2, 0) and are asserted. (4) Tie-breaking robustness: all
 * 720 orderings of a 6-set / 10-element instance are run; each of them respects the (1 - 1/e) bound
 * and none beats OPT — the printed example covers 9 of OPT 10 for every ordering, so on this
 * instance the tie-breaking order does not change the value (only the chosen sets). (5) Weighted
 * instances: 500 deterministic random instances (12 elements with weights 1..5, 10 sets, k = 3) are
 * checked against brute force, worst ratio 0.8205, asserted to stay above the documented 0.80.
 * (6) Complexity and boundary cases: the greedy performs exactly k*m - k(k-1)/2 coverage
 * evaluations (asserted), k = 0 covers nothing, k = m is optimal (everything is taken), an empty
 * universe gives 0, duplicate sets are harmless, and a set containing the whole universe makes the
 * greedy optimal for k = 1.
 *
 * All randomness is a fixed-seed splitmix64 (the wave-3 note applies: this file draws many
 * independent uniforms, so the repo's 16-bit LCG is not used); every printed number is reproducible.
 */
import java.util.*;

public class Solution {

    static final class Rng {
        private long state;

        Rng(long seed) { this.state = seed; }

        long nextLong() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
    }

    static long evaluations;

    static double coverage(long[] sets, double[] weight, int[] chosen, int count) {
        long mask = 0L;
        for (int i = 0; i < count; i++) mask |= sets[chosen[i]];
        double total = 0.0;
        for (int e = 0; e < weight.length; e++) if ((mask & (1L << e)) != 0) total += weight[e];
        return total;
    }

    static double coverage(int[] chosen, int count, long[] maskByChoice) {
        long mask = 0L;
        for (int i = 0; i < count; i++) mask |= maskByChoice[chosen[i]];
        return mask;
    }

    /** Marginal gain of set j given the covered mask. */
    static double gain(long[] sets, double[] weight, long covered, int j) {
        long fresh = sets[j] & ~covered;
        double total = 0.0;
        for (int e = 0; e < weight.length; e++) if ((fresh & (1L << e)) != 0) total += weight[e];
        return total;
    }

    /**
     * Greedy maximum coverage with an explicit tie-breaking order: at each step the largest marginal
     * gain wins and the first set in {@code order} among equal gains is taken. Steps' gains are
     * written to stepGains.
     */
    static int[] greedy(long[] sets, double[] weight, int k, int[] order, double[] stepGains) {
        int[] chosen = new int[k];
        long covered = 0L;
        for (int step = 0; step < k; step++) {
            int best = -1;
            double bestGain = -1.0;
            for (int i = 0; i < order.length; i++) {
                int j = order[i];
                boolean used = false;
                for (int t = 0; t < step; t++) if (chosen[t] == j) used = true;
                if (used) continue;
                evaluations++;
                double g = gain(sets, weight, covered, j);
                if (g > bestGain + 1e-15) {
                    best = j;
                    bestGain = g;
                }
            }
            chosen[step] = best;
            covered |= sets[best];
            if (stepGains != null) stepGains[step] = bestGain;
        }
        return chosen;
    }

    /** Exhaustive optimum over all k-subsets. */
    static double bruteForce(long[] sets, double[] weight, int k) {
        int[] chosen = new int[k];
        double[] best = {Double.NEGATIVE_INFINITY};
        combine(sets, weight, chosen, 0, 0, k, best);
        return best[0];
    }

    static void combine(long[] sets, double[] weight, int[] chosen, int start, int depth, int k, double[] best) {
        if (depth == k) {
            best[0] = Math.max(best[0], coverage(sets, weight, chosen, k));
            return;
        }
        for (int j = start; j <= sets.length - (k - depth); j++) {
            chosen[depth] = j;
            combine(sets, weight, chosen, j + 1, depth + 1, k, best);
        }
    }

    static int[] identity(int m) {
        int[] order = new int[m];
        for (int j = 0; j < m; j++) order[j] = j;
        return order;
    }

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    static void close(double a, double b, double tol, String message) {
        if (!(Math.abs(a - b) <= tol)) throw new AssertionError(message + ": " + a + " vs " + b);
    }

    static double lowerBound(int k) { return 1.0 - Math.pow(1.0 - 1.0 / k, k); }

    /** Canonical 6-set instance used for the tie-breaking sweep. */
    static long[] canonicalSets() {
        return new long[]{
            0b0000001111L, // {0,1,2,3}
            0b0000110011L, // {0,1,4,5}
            0b0011001100L, // {2,3,6,7}
            0b1100000001L, // {0,8,9}
            0b0011110000L, // {4,5,6,7}
            0b0101010101L  // {0,2,4,6,8}
        };
    }

    public static void main(String[] args) {
        Rng rng = new Rng(569_2025L);

        // ---- hand-computed instance -----------------------------------------------------------
        long[] handSets = {0b0111L, 0b1100L, 0b11000L, 0b10111L};
        double[] handWeight = {1, 1, 1, 1, 1};
        int[] handChosen = greedy(handSets, handWeight, 2, identity(4), null);
        close(coverage(handSets, handWeight, handChosen, 2), 5.0, 1e-9, "hand instance is optimal for k = 2");
        close(bruteForce(handSets, handWeight, 2), 5.0, 1e-9, "hand instance optimum is 5");
        check(handChosen[0] == 3, "the largest set (4 elements) is taken first");
        check(handChosen[0] != handChosen[1], "the greedy never repeats a set");

        // ---- exhaustive verification of the guarantee on all 4-element / 4-set systems ----------
        int universe = 4;
        int m = 4;
        double[] unit = {1, 1, 1, 1};
        double[] worstRatio = new double[5];
        int[] worstCode = new int[5];
        Arrays.fill(worstRatio, 1.0);
        for (int a = 0; a < 16; a++) {
            for (int b = 0; b < 16; b++) {
                for (int c = 0; c < 16; c++) {
                    for (int d = 0; d < 16; d++) {
                        long[] sets = {a, b, c, d};
                        for (int k = 1; k <= 4; k++) {
                            int[] order = identity(m);
                            double greedyValue = coverage(sets, unit, greedy(sets, unit, k, order, null), k);
                            double opt = bruteForce(sets, unit, k);
                            if (opt == 0.0) {
                                // The single degenerate system of four empty sets: the ratio 0/0 is
                                // undefined, so the check is that the greedy also covers nothing.
                                check(greedyValue == 0.0, "an all-empty system has greedy value 0");
                                continue;
                            }
                            double ratio = greedyValue / opt;
                            check(ratio >= 1.0 - 1.0 / Math.E - 1e-12,
                                    "guarantee violated by system " + a + "," + b + "," + c + "," + d
                                            + " at k = " + k);
                            check(ratio >= lowerBound(Math.min(k, m)) - 1e-12,
                                    "per-k bound violated at k = " + k);
                            check(ratio <= 1.0 + 1e-12, "greedy cannot beat the optimum");
                            if (ratio < worstRatio[k]) {
                                worstRatio[k] = ratio;
                                worstCode[k] = (a << 12) | (b << 8) | (c << 4) | d;
                            }
                        }
                    }
                }
            }
        }
        for (int k = 1; k <= 4; k++) {
            System.out.println("exhaustive 4x4 scan, k=" + k + ": worst ratio " + worstRatio[k]
                    + " (bound " + lowerBound(Math.min(k, m)) + ", guarantee " + (1.0 - 1.0 / Math.E) + ")");
            check(worstRatio[k] >= 1.0 - 1.0 / Math.E - 1e-12, "worst ratio respects the guarantee");
        }
        close(worstRatio[2], 0.75, 1e-12, "the exhaustive worst case for k = 2 equals 1 - (1/2)^2");
        // Re-check the worst k = 2 system individually.
        int code = worstCode[2];
        long[] worstSets = {(code >> 12) & 15, (code >> 8) & 15, (code >> 4) & 15, code & 15};
        int[] worstChosen = greedy(worstSets, unit, 2, identity(m), null);
        double worstGreedy = coverage(worstSets, unit, worstChosen, 2);
        double worstOpt = bruteForce(worstSets, unit, 2);
        System.out.println("worst k=2 system " + Arrays.toString(worstSets) + ": greedy " + worstGreedy
                + " of OPT " + worstOpt);
        check(worstGreedy < worstOpt, "documented counterexample: the greedy is not exact there");
        close(worstGreedy / worstOpt, 0.75, 1e-12, "counterexample ratio is 0.75");

        // ---- larger random family --------------------------------------------------------------
        double worst5 = 1.0;
        for (int trial = 0; trial < 20000; trial++) {
            int u = 5;
            double[] w = new double[u];
            Arrays.fill(w, 1.0);
            long[] sets = new long[5];
            for (int j = 0; j < 5; j++) sets[j] = 1 + rng.nextInt((1 << u) - 1);
            for (int k = 1; k <= 5; k++) {
                double greedyValue = coverage(sets, w, greedy(sets, w, k, identity(5), null), k);
                double opt = bruteForce(sets, w, k);
                double ratio = greedyValue / opt;
                check(ratio >= 1.0 - 1.0 / Math.E - 1e-12, "5x5 sample " + trial + " violates the guarantee");
                if (k == 3) worst5 = Math.min(worst5, ratio);
            }
        }
        System.out.println("5x5 random scan (20000 samples): worst ratio at k=3 is " + worst5);
        check(worst5 >= lowerBound(3), "the 5x5 sample respects the k = 3 bound");

        // ---- the proof's invariant on a k = 4 instance ------------------------------------------
        long[] inst = {
            0b000000001111L, 0b000000110011L, 0b000011000011L, 0b001100001100L,
            0b110000110000L, 0b010101010101L, 0b101010101010L, 0b111100001111L
        };
        double[] weight = new double[12];
        Arrays.fill(weight, 1.0);
        int k = 4;
        double[] stepGains = new double[k];
        int[] chosen = greedy(inst, weight, k, identity(inst.length), stepGains);
        double opt = bruteForce(inst, weight, k);
        double[] prefix = new double[k + 1];
        for (int i = 1; i <= k; i++) prefix[i] = coverage(inst, weight, chosen, i);
        for (int i = 0; i < k; i++) {
            check(prefix[i + 1] >= (1.0 - Math.pow(1.0 - 1.0 / k, i + 1)) * opt - 1e-9,
                    "f(S_" + (i + 1) + ") respects the (1 - 1/k)^i bound");
            if (i + 1 < k) {
                check(opt - prefix[i + 1] <= k * stepGains[i + 1] + 1e-9,
                        "uncovered optimum bounded by k times the next gain at step " + (i + 1));
            }
        }
        System.out.println("k=4 invariant instance: OPT " + opt + ", cumulative greedy coverage "
                + Arrays.toString(Arrays.copyOfRange(prefix, 1, k + 1)) + ", gains " + Arrays.toString(stepGains));
        check(prefix[k] >= (1.0 - 1.0 / Math.E) * opt - 1e-9, "final greedy value respects the guarantee");
        check(prefix[k] <= opt + 1e-9, "final greedy value does not exceed OPT");

        // ---- tie-breaking robustness (all 720 orderings of a 6-set instance) --------------------
        long[] canonical = canonicalSets();
        double[] canonicalWeight = new double[10];
        Arrays.fill(canonicalWeight, 1.0);
        double canonicalOpt = bruteForce(canonical, canonicalWeight, 3);
        double minPermutation = Double.POSITIVE_INFINITY;
        double maxPermutation = Double.NEGATIVE_INFINITY;
        int[] order = identity(6);
        List<int[]> permutations = new ArrayList<>();
        permute(order, 0, permutations);
        check(permutations.size() == 720, "all 720 orderings were enumerated");
        for (int[] perm : permutations) {
            double value = coverage(canonical, canonicalWeight, greedy(canonical, canonicalWeight, 3, perm, null), 3);
            check(value >= (1.0 - 1.0 / Math.E) * canonicalOpt - 1e-9,
                    "every tie-breaking order respects the guarantee");
            check(value <= canonicalOpt + 1e-9, "no ordering beats OPT");
            minPermutation = Math.min(minPermutation, value);
            maxPermutation = Math.max(maxPermutation, value);
        }
        System.out.println("tie-breaking sweep (720 orders): OPT " + canonicalOpt + ", coverage range "
                + minPermutation + ".." + maxPermutation);
        check(maxPermutation >= minPermutation, "range is consistent");

        // ---- weighted random instances ---------------------------------------------------------
        double worstWeighted = 1.0;
        for (int trial = 0; trial < 500; trial++) {
            int u = 12;
            double[] w = new double[u];
            for (int e = 0; e < u; e++) w[e] = 1 + rng.nextInt(5);
            long[] sets = new long[10];
            for (int j = 0; j < 10; j++) {
                long mask = 0L;
                for (int e = 0; e < u; e++) if (rng.nextInt(3) == 0) mask |= 1L << e;
                if (mask == 0L) mask = 1L << rng.nextInt(u);
                sets[j] = mask;
            }
            double greedyValue = coverage(sets, w, greedy(sets, w, 3, identity(10), null), 3);
            double weightedOpt = bruteForce(sets, w, 3);
            double ratio = greedyValue / weightedOpt;
            check(ratio >= 1.0 - 1.0 / Math.E - 1e-12, "weighted instance " + trial + " violates the guarantee");
            check(ratio <= 1.0 + 1e-12, "weighted instance " + trial + " beats the optimum");
            worstWeighted = Math.min(worstWeighted, ratio);
        }
        System.out.println("weighted random family (500 instances): worst ratio " + worstWeighted);
        check(worstWeighted >= 0.8, "weighted worst ratio stays above the documented 0.80");

        // ---- complexity: exactly k*m - k(k-1)/2 coverage evaluations ----------------------------
        evaluations = 0;
        greedy(sets(12, 7, rng), unitWeights(12), 5, identity(7), null);
        close(evaluations, 5 * 7 - 5 * 4 / 2, 0.0, "greedy evaluation count is k*m - k(k-1)/2");

        // ---- boundary cases --------------------------------------------------------------------
        double[] w1 = {1.0};
        long[] single = {1L};
        check(coverage(single, w1, greedy(single, w1, 0, identity(1), null), 0) == 0.0, "k = 0 covers nothing");
        int[] clamped = greedy(canonical, canonicalWeight, 6, identity(6), null);
        close(coverage(canonical, canonicalWeight, clamped, 6), bruteForce(canonical, canonicalWeight, 6), 1e-9,
                "k = m is optimal (everything is taken)");
        check(coverage(single, new double[0], new int[0], 0) == 0.0, "empty universe covers nothing");
        long[] duplicates = {0b0011L, 0b0011L, 0b1100L};
        double[] w2 = {1, 1, 1, 1};
        double dupValue = coverage(duplicates, w2, greedy(duplicates, w2, 2, identity(3), null), 2);
        close(dupValue, 4.0, 1e-9, "duplicate sets are handled (both copies plus the complement)");
        long[] everything = {0b1111L, 0b0001L};
        double[] w3 = {1, 1, 1, 1};
        close(coverage(everything, w3, greedy(everything, w3, 1, identity(2), null), 1), 4.0, 1e-9,
                "a set covering the universe makes the greedy optimal for k = 1");

        System.out.println("PASS");
    }

    static long[] sets(int universe, int m, Rng rng) {
        long[] out = new long[m];
        for (int j = 0; j < m; j++) {
            long mask = 0L;
            for (int e = 0; e < universe; e++) if (rng.nextInt(3) == 0) mask |= 1L << e;
            out[j] = mask == 0L ? 1L << rng.nextInt(universe) : mask;
        }
        return out;
    }

    static double[] unitWeights(int universe) {
        double[] out = new double[universe];
        Arrays.fill(out, 1.0);
        return out;
    }

    static void permute(int[] array, int index, List<int[]> out) {
        if (index == array.length) {
            out.add(array.clone());
            return;
        }
        for (int i = index; i < array.length; i++) {
            int tmp = array[index];
            array[index] = array[i];
            array[i] = tmp;
            permute(array, index + 1, out);
            tmp = array[index];
            array[index] = array[i];
            array[i] = tmp;
        }
    }
}
