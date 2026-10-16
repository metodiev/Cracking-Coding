/** Task 524 - Implement Hidden Markov Model Viterbi algorithm. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Viterbi decoding for a discrete HMM (start probabilities, a row-stochastic transition matrix and
 * a row-stochastic emission matrix over a finite alphabet). The recursion runs in the LOG domain,
 * delta_1(i) = log start_i + log e_i(o_1) and delta_t(j) = log e_j(o_t) + max_i (delta_{t-1}(i) +
 * log a_ij), with one backpointer per (t, j) and a final argmax over the states. Log-zero is
 * represented by a documented NEG_INF = -1e18 sentinel, so zero-probability transitions and
 * emissions are simply never chosen; a sequence whose probability is zero for every path makes the
 * whole column NEG_INF and returns a null result instead of a bogus path. Ties are broken by the
 * smaller state index because the argmax uses a strict 'greater than' comparison while iterating
 * states in increasing order - this is documented and checked. Documented model used for the
 * canonical example (the textbook weather HMM): states Rainy = 0 / Sunny = 1, start (0.6, 0.4),
 * transition rows Rainy (0.7, 0.3) and Sunny (0.4, 0.6), emissions Rainy (Walk 0.1, Shop 0.4,
 * Clean 0.5) and Sunny (Walk 0.6, Shop 0.3, Clean 0.1).
 *
 * Checks: the canonical observations (Walk, Shop, Clean) must decode to (Sunny, Rainy, Rainy) with
 * log-probability ln 0.01344; the reported score must equal the direct product
 * start * emission * transition * ... of that path recomputed independently; for LCG-generated
 * strictly positive HMMs (3 states x 4 observations and 5 states x 8 observations) the log-domain
 * answer must equal a full brute-force enumeration of all S^T paths in the probability domain, and
 * the probability-domain max-product recursion must return the same path and score; ties must pick
 * the smaller state index (a deliberately symmetric two-state model); a single-observation sequence
 * decodes to argmax_i start_i e_i(o) with the same tie rule; zero-probability emissions and
 * blocked transitions are avoided (verified against brute force on a model with injected zeros);
 * an impossible observation sequence (all emissions zero for some symbol) returns null; the
 * per-position posterior marginal of every timestep bounds the Viterbi posterior probability
 * (gamma_t(s*_t) >= delta* because the marginal sums over all paths through s*_t), with the
 * documented weaker bound that the product of the per-position maxima dominates (delta*)^T, and
 * the canonical example's Viterbi posterior is pinned at 0.399857193859336; scaling the whole model
 * by a positive constant does not change the path; repeated runs are identical.
 */
import java.util.*;

public class Solution {
    static final double NEG_INF = -1e18;

    static class Hmm {
        final int states;
        final int alphabet;
        final double[] start;
        final double[][] trans;
        final double[][] emit;

        Hmm(double[] start, double[][] trans, double[][] emit) {
            this.start = start;
            this.trans = trans;
            this.emit = emit;
            this.states = start.length;
            this.alphabet = emit[0].length;
        }
    }

    static class Decoding {
        final int[] path;
        final double logProbability;

        Decoding(int[] path, double logProbability) {
            this.path = path;
            this.logProbability = logProbability;
        }
    }

    static Hmm canonicalWeather() {
        return new Hmm(
                new double[]{0.6, 0.4},
                new double[][]{{0.7, 0.3}, {0.4, 0.6}},
                new double[][]{{0.1, 0.4, 0.5}, {0.6, 0.3, 0.1}});
    }

    /** Viterbi in log space; returns null when no path has positive probability. */
    static Decoding viterbi(Hmm hmm, int[] observations) {
        int states = hmm.states;
        int length = observations.length;
        if (length == 0) return new Decoding(new int[0], 0.0);
        double[] previous = new double[states];
        int[][] back = new int[length][states];
        for (int i = 0; i < states; i++) {
            previous[i] = log(hmm.start[i]) + log(hmm.emit[i][observations[0]]);
        }
        for (int t = 1; t < length; t++) {
            double[] current = new double[states];
            for (int j = 0; j < states; j++) {
                double best = NEG_INF;
                int arg = -1;
                for (int i = 0; i < states; i++) {
                    if (previous[i] <= NEG_INF / 2.0) continue;
                    double candidate = previous[i] + log(hmm.trans[i][j]);
                    if (candidate > best) {
                        best = candidate;
                        arg = i;
                    }
                }
                double emission = log(hmm.emit[j][observations[t]]);
                if (arg < 0 || emission <= NEG_INF / 2.0) {
                    current[j] = NEG_INF;
                    back[t][j] = -1;
                } else {
                    current[j] = best + emission;
                    back[t][j] = arg;
                }
            }
            previous = current;
        }
        double best = NEG_INF;
        int last = -1;
        for (int j = 0; j < states; j++) {
            if (previous[j] > best) {
                best = previous[j];
                last = j;
            }
        }
        if (last < 0 || best <= NEG_INF / 2.0) return null;
        int[] path = new int[length];
        path[length - 1] = last;
        for (int t = length - 1; t > 0; t--) {
            int previousState = back[t][path[t]];
            if (previousState < 0) return null;
            path[t - 1] = previousState;
        }
        return new Decoding(path, best);
    }

    /** Max-product recursion in the probability domain; usable only for short sequences. */
    static Decoding viterbiProbabilities(Hmm hmm, int[] observations) {
        int states = hmm.states;
        int length = observations.length;
        if (length == 0) return new Decoding(new int[0], 0.0);
        double[] previous = new double[states];
        int[][] back = new int[length][states];
        for (int i = 0; i < states; i++) {
            previous[i] = hmm.start[i] * hmm.emit[i][observations[0]];
        }
        for (int t = 1; t < length; t++) {
            double[] current = new double[states];
            for (int j = 0; j < states; j++) {
                double best = -1.0;
                int arg = -1;
                for (int i = 0; i < states; i++) {
                    double candidate = previous[i] * hmm.trans[i][j];
                    if (candidate > best) {
                        best = candidate;
                        arg = i;
                    }
                }
                if (arg < 0) {
                    current[j] = 0.0;
                    back[t][j] = -1;
                } else {
                    current[j] = best * hmm.emit[j][observations[t]];
                    back[t][j] = arg;
                }
            }
            previous = current;
        }
        double best = -1.0;
        int last = -1;
        for (int j = 0; j < states; j++) {
            if (previous[j] > best) {
                best = previous[j];
                last = j;
            }
        }
        if (last < 0 || best <= 0.0) return null;
        int[] path = new int[length];
        path[length - 1] = last;
        for (int t = length - 1; t > 0; t--) {
            int previousState = back[t][path[t]];
            if (previousState < 0) return null;
            path[t - 1] = previousState;
        }
        return new Decoding(path, Math.log(best));
    }

    /** Direct product of the model factors along a path. */
    static double pathProbability(Hmm hmm, int[] path, int[] observations) {
        double probability = hmm.start[path[0]] * hmm.emit[path[0]][observations[0]];
        for (int t = 1; t < path.length; t++) {
            probability *= hmm.trans[path[t - 1]][path[t]] * hmm.emit[path[t]][observations[t]];
        }
        return probability;
    }

    /** Exhaustive enumeration of every state path; returns the best path probability. */
    static double bestPathProbability(Hmm hmm, int[] observations) {
        int length = observations.length;
        int[] path = new int[length];
        return enumerate(hmm, observations, path, 0, hmm.states);
    }

    private static double enumerate(Hmm hmm, int[] observations, int[] path, int depth, int states) {
        if (depth == path.length) return pathProbability(hmm, path, observations);
        double best = 0.0;
        for (int state = 0; state < states; state++) {
            path[depth] = state;
            best = Math.max(best, enumerate(hmm, observations, path, depth + 1, states));
        }
        return best;
    }

    /** Sum of the joint probabilities of every path (the likelihood of the observations). */
    static double totalProbability(Hmm hmm, int[] observations) {
        int[] path = new int[observations.length];
        return enumerateTotal(hmm, observations, path, 0);
    }

    private static double enumerateTotal(Hmm hmm, int[] observations, int[] path, int depth) {
        if (depth == path.length) return pathProbability(hmm, path, observations);
        double total = 0.0;
        for (int state = 0; state < hmm.states; state++) {
            path[depth] = state;
            total += enumerateTotal(hmm, observations, path, depth + 1);
        }
        return total;
    }

    /** Per-position posterior marginals by brute-force enumeration over all paths. */
    static double[][] bruteForceMarginals(Hmm hmm, int[] observations) {
        int length = observations.length;
        int states = hmm.states;
        double[][] marginal = new double[length][states];
        int[] path = new int[length];
        enumerateMarginals(hmm, observations, path, 0, marginal);
        double total = 0.0;
        for (int i = 0; i < states; i++) total += marginal[0][i];
        for (int t = 0; t < length; t++) {
            for (int i = 0; i < states; i++) {
                marginal[t][i] = total == 0.0 ? 0.0 : marginal[t][i] / total;
            }
        }
        return marginal;
    }

    private static void enumerateMarginals(Hmm hmm, int[] observations, int[] path, int depth,
            double[][] marginal) {
        if (depth == path.length) {
            double probability = pathProbability(hmm, path, observations);
            for (int t = 0; t < path.length; t++) marginal[t][path[t]] += probability;
            return;
        }
        for (int state = 0; state < hmm.states; state++) {
            path[depth] = state;
            enumerateMarginals(hmm, observations, path, depth + 1, marginal);
        }
    }

    static Hmm lcgHmm(int states, int alphabet, int seed, double zeroDensity) {
        double[] start = new double[states];
        double[][] trans = new double[states][states];
        double[][] emit = new double[states][alphabet];
        int state = seed;
        for (int i = 0; i < states; i++) {
            state = (state * 75 + 74) % 65537;
            start[i] = zeroDensity > 0.0 && state % 7 == 0 ? 0.0 : state % 97 + 1;
        }
        for (int i = 0; i < states; i++) {
            for (int j = 0; j < states; j++) {
                state = (state * 75 + 74) % 65537;
                trans[i][j] = zeroDensity > 0.0 && state % 5 == 0 ? 0.0 : state % 89 + 1;
            }
            normalize(trans[i]);
        }
        for (int i = 0; i < states; i++) {
            for (int j = 0; j < alphabet; j++) {
                state = (state * 75 + 74) % 65537;
                emit[i][j] = zeroDensity > 0.0 && state % 6 == 0 ? 0.0 : state % 83 + 1;
            }
            normalize(emit[i]);
        }
        normalize(start);
        return new Hmm(start, trans, emit);
    }

    static void normalize(double[] row) {
        double sum = 0.0;
        for (double value : row) sum += value;
        if (sum == 0.0) {
            Arrays.fill(row, 1.0 / row.length);
            return;
        }
        for (int i = 0; i < row.length; i++) row[i] /= sum;
    }

    static double log(double value) {
        return value <= 0.0 ? NEG_INF : Math.log(value);
    }

    static Hmm scale(Hmm hmm, double factor) {
        double[] start = hmm.start.clone();
        for (int i = 0; i < start.length; i++) start[i] *= factor;
        double[][] trans = new double[hmm.states][hmm.states];
        for (int i = 0; i < hmm.states; i++) {
            for (int j = 0; j < hmm.states; j++) trans[i][j] = hmm.trans[i][j] * factor;
        }
        double[][] emit = new double[hmm.states][hmm.alphabet];
        for (int i = 0; i < hmm.states; i++) {
            for (int j = 0; j < hmm.alphabet; j++) emit[i][j] = hmm.emit[i][j] * factor;
        }
        return new Hmm(start, trans, emit);
    }

    public static void main(String[] args) {
        Hmm weather = canonicalWeather();
        int[] observations = {0, 1, 2};
        Decoding decoded = viterbi(weather, observations);
        check(Arrays.equals(decoded.path, new int[]{1, 0, 0}), "canonical Viterbi path Sunny/Rainy/Rainy");
        check(Math.abs(Math.exp(decoded.logProbability) - 0.01344) <= 1e-12,
                "canonical path probability 0.01344");
        check(Math.abs(pathProbability(weather, decoded.path, observations) - 0.01344) <= 1e-12,
                "independent product of the reported path factors");
        check(Math.abs(bestPathProbability(weather, observations) - 0.01344) <= 1e-12,
                "brute force agrees on the canonical example");
        double canonicalPosterior = 0.01344 / totalProbability(weather, observations);
        check(Math.abs(canonicalPosterior - 0.399857193859336) <= 1e-12,
                "the canonical Viterbi path carries posterior 0.39985719");
        check(Math.abs(viterbiProbabilities(weather, observations).logProbability
                - decoded.logProbability) <= 1e-12, "probability-domain recursion agrees");
        check(Arrays.equals(viterbiProbabilities(weather, observations).path, decoded.path),
                "probability-domain recursion returns the same path");

        int[] walkOnly = {0};
        Decoding single = viterbi(weather, walkOnly);
        check(single.path.length == 1 && single.path[0] == 1, "single observation picks Sunny");
        check(Math.abs(Math.exp(single.logProbability) - 0.4 * 0.6) <= 1e-12,
                "single observation probability = start * emission");
        check(Arrays.equals(viterbi(weather, new int[]{}).path, new int[0]),
                "empty observation sequence decodes to the empty path");

        Hmm symmetric = new Hmm(new double[]{0.5, 0.5},
                new double[][]{{0.5, 0.5}, {0.5, 0.5}},
                new double[][]{{0.5, 0.5}, {0.5, 0.5}});
        check(Arrays.equals(viterbi(symmetric, new int[]{0, 0, 0}).path, new int[]{0, 0, 0}),
                "ties break towards the smaller state index");

        int seed = 11;
        for (int trial = 0; trial < 6; trial++) {
            Hmm hmm = lcgHmm(3, 4, 1000 + trial * 97, 0.0);
            int length = 5;
            int[] obs = new int[length];
            for (int t = 0; t < length; t++) {
                seed = (seed * 75 + 74) % 65537;
                obs[t] = seed % 4;
            }
            Decoding logResult = viterbi(hmm, obs);
            check(logResult != null, "positive model always decodes");
            double bruteForce = bestPathProbability(hmm, obs);
            check(Math.abs(Math.exp(logResult.logProbability) - bruteForce) <= 1e-12,
                    "log-domain score equals brute force, trial " + trial);
            check(Math.abs(pathProbability(hmm, logResult.path, obs) - bruteForce) <= 1e-12,
                    "reported path is the best path, trial " + trial);
            Decoding probResult = viterbiProbabilities(hmm, obs);
            check(Arrays.equals(probResult.path, logResult.path),
                    "both domains return the same path, trial " + trial);

            double[][] marginal = bruteForceMarginals(hmm, obs);
            double mapProduct = 1.0;
            int[] mapPath = new int[length];
            for (int t = 0; t < length; t++) {
                double best = -1.0;
                for (int i = 0; i < hmm.states; i++) {
                    if (marginal[t][i] > best) {
                        best = marginal[t][i];
                        mapPath[t] = i;
                    }
                }
                mapProduct *= best;
            }
            check(mapPath.length == length, "pointwise MAP path has one state per observation");
            double total = totalProbability(hmm, obs);
            check(total >= bruteForce - 1e-18,
                    "the total probability dominates the best path, trial " + trial);
            double viterbiPosterior = bruteForce / total;
            check(viterbiPosterior <= 1.0 + 1e-12,
                    "the Viterbi posterior is a probability, trial " + trial);
            for (int t = 0; t < length; t++) {
                double marginalSum = 0.0;
                double maxMarginal = 0.0;
                for (int i = 0; i < hmm.states; i++) {
                    marginalSum += marginal[t][i];
                    maxMarginal = Math.max(maxMarginal, marginal[t][i]);
                }
                check(Math.abs(marginalSum - 1.0) <= 1e-12,
                        "posterior marginals sum to 1, trial " + trial);
                check(maxMarginal >= viterbiPosterior - 1e-12,
                        "max_i gamma_t(i) bounds the Viterbi posterior, trial " + trial + " t = " + t);
                check(marginal[t][logResult.path[t]] >= viterbiPosterior - 1e-12,
                        "the Viterbi state's own marginal bounds it too, trial " + trial);
            }
            check(mapProduct >= Math.pow(viterbiPosterior, length) - 1e-12,
                    "the pointwise MAP product dominates (delta*)^T, trial " + trial);
            Hmm scaled = scale(hmm, 3.5);
            check(Arrays.equals(viterbi(scaled, obs).path, logResult.path),
                    "scaling the model does not change the path, trial " + trial);
            check(Arrays.equals(viterbi(hmm, obs).path, logResult.path), "deterministic re-run");
        }

        Hmm large = lcgHmm(5, 3, 31337, 0.0);
        int[] longObs = new int[8];
        seed = 271;
        for (int t = 0; t < longObs.length; t++) {
            seed = (seed * 75 + 74) % 65537;
            longObs[t] = seed % 3;
        }
        Decoding largeDecoded = viterbi(large, longObs);
        check(Math.abs(Math.exp(largeDecoded.logProbability) - bestPathProbability(large, longObs))
                <= 1e-12, "5-state 8-step sequence matches the 390625-path brute force");
        check(Arrays.equals(viterbiProbabilities(large, longObs).path, largeDecoded.path),
                "5-state sequence agrees across domains");

        Hmm withZeros = lcgHmm(4, 3, 555, 1.0);
        int[] zeroObs = {0, 1, 2, 0, 1};
        Decoding zeroDecoded = viterbi(withZeros, zeroObs);
        double zeroBrute = bestPathProbability(withZeros, zeroObs);
        if (zeroBrute > 0.0) {
            check(zeroDecoded != null, "model with zeros still decodes a reachable sequence");
            check(Math.abs(Math.exp(zeroDecoded.logProbability) - zeroBrute) <= 1e-12,
                    "zeros handled exactly");
            for (int t = 0; t < zeroObs.length; t++) {
                check(withZeros.emit[zeroDecoded.path[t]][zeroObs[t]] > 0.0,
                        "no zero-probability emission is chosen");
            }
        } else {
            check(zeroDecoded == null, "unreachable sequence reports null");
        }

        Hmm blocked = new Hmm(new double[]{1.0, 0.0},
                new double[][]{{0.0, 1.0}, {0.0, 1.0}},
                new double[][]{{1.0, 0.0}, {1.0, 0.0}});
        Decoding blockedDecoded = viterbi(blocked, new int[]{0, 0});
        check(blockedDecoded != null && blockedDecoded.path[0] == 0 && blockedDecoded.path[1] == 1,
                "a blocked start state forces the single legal transition");
        check(viterbi(blocked, new int[]{1}) == null,
                "an emission that no state can produce returns null");

        Hmm impossible = new Hmm(new double[]{1.0},
                new double[][]{{1.0}},
                new double[][]{{1.0, 0.0}});
        check(viterbi(impossible, new int[]{1}) == null, "all-zero emission column returns null");
        check(viterbi(impossible, new int[]{0}) != null, "the reachable symbol still decodes");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
