/** Task 525 - Implement HMM forward-backward for probabilities. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Scaled forward-backward for a discrete HMM. The forward variables alpha_t(j) =
 * (sum_i alphaHat_{t-1}(i) a_ij) e_j(o_t) are rescaled by c_t = 1 / sum_j alpha_t(j) at every step
 * so the recursion cannot underflow, and the likelihood is recovered as log P(O) = -sum_t log c_t
 * (documented convention: a zero c_t means the observation sequence has probability 0, in which
 * case the posteriors are reported as 0 and the log-likelihood as NEG_INF = -1e18). The backward
 * variables are scaled with the SAME c_t factors, betaHat_T(i) = 1 and
 * betaHat_t(i) = sum_j a_ij e_j(o_{t+1}) betaHat_{t+1}(j) * c_{t+1}; then
 * gamma_t(i) = alphaHat_t(i) betaHat_t(i) / sum_j alphaHat_t(j) betaHat_t(j) and
 * xi_t(i, j) = alphaHat_t(i) a_ij e_j(o_{t+1}) betaHat_{t+1}(j) / (normalising sum). One Baum-Welch
 * update is also implemented from exactly these quantities: startHat_i = gamma_1(i),
 * aHat_ij = sum_t xi_t(i, j) / sum_t gamma_t(i), eHat_j(k) = sum_{t : o_t = k} gamma_t(j) /
 * sum_t gamma_t(j), with the documented fallback of keeping the old row when a denominator is 0.
 * The canonical weather HMM of the textbook example is used for the hand-checked values.
 *
 * Checks: the canonical observations (Walk, Shop, Clean) must give P(O) = 0.033612 and the Rainy
 * posteriors 0.231702963227 / 0.624062834702 / 0.863977151017 (tolerance 1e-9); sum_i gamma_t(i) = 1
 * for every t and sum_{i,j} xi_t(i,j) = 1, with the consistency identity sum_j xi_t(i, j) = gamma_t(i);
 * for LCG models and short sequences the scaled forward, the unscaled probability-domain forward,
 * the brute-force enumeration of all paths and the backward recursion must all agree to 1e-12, and
 * every gamma / xi entry must match the brute-force marginal / joint marginal; for a 300-step,
 * 6-state sequence the scaled log-likelihood must equal an independent log-sum-exp forward
 * recursion to 1e-9 (the two implementations differ in how they avoid underflow), the posteriors
 * must stay in [0, 1] and every column must sum to 1; a model with zero probabilities must produce
 * exactly zero posteriors for the impossible states and a NEG_INF likelihood for an impossible
 * sequence without any NaN; one Baum-Welch iteration must keep all three parameter families
 * row-stochastic and must not decrease the likelihood (checked over 5 iterations on 4 LCG models,
 * with the likelihood strictly increasing overall); repeated runs are identical.
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

        Hmm copy() {
            double[] s = start.clone();
            double[][] t = new double[states][];
            double[][] e = new double[states][];
            for (int i = 0; i < states; i++) {
                t[i] = trans[i].clone();
                e[i] = emit[i].clone();
            }
            return new Hmm(s, t, e);
        }
    }

    static class Posterior {
        final double[][] gamma;
        final double[][][] xi;
        final double[][] alpha;
        final double[][] beta;
        final double[] scales;
        final double logLikelihood;

        Posterior(double[][] gamma, double[][][] xi, double[][] alpha, double[][] beta,
                double[] scales, double logLikelihood) {
            this.gamma = gamma;
            this.xi = xi;
            this.alpha = alpha;
            this.beta = beta;
            this.scales = scales;
            this.logLikelihood = logLikelihood;
        }
    }

    static Hmm canonicalWeather() {
        return new Hmm(
                new double[]{0.6, 0.4},
                new double[][]{{0.7, 0.3}, {0.4, 0.6}},
                new double[][]{{0.1, 0.4, 0.5}, {0.6, 0.3, 0.1}});
    }

    /** Scaled forward-backward with all derived quantities. */
    static Posterior forwardBackward(Hmm hmm, int[] observations) {
        int states = hmm.states;
        int length = observations.length;
        double[][] alpha = new double[length][states];
        double[][] beta = new double[length][states];
        double[] scales = new double[length];
        boolean impossible = false;
        if (length == 0) {
            return new Posterior(new double[0][], new double[0][][], alpha, beta, scales, 0.0);
        }
        for (int i = 0; i < states; i++) {
            alpha[0][i] = hmm.start[i] * hmm.emit[i][observations[0]];
        }
        double sum = rowSum(alpha[0]);
        if (sum <= 0.0) {
            impossible = true;
            scales[0] = 0.0;
        } else {
            scales[0] = 1.0 / sum;
            for (int i = 0; i < states; i++) alpha[0][i] *= scales[0];
        }
        for (int t = 1; t < length; t++) {
            for (int j = 0; j < states; j++) {
                double acc = 0.0;
                for (int i = 0; i < states; i++) acc += alpha[t - 1][i] * hmm.trans[i][j];
                alpha[t][j] = acc * hmm.emit[j][observations[t]];
            }
            sum = rowSum(alpha[t]);
            if (sum <= 0.0) {
                impossible = true;
                scales[t] = 0.0;
            } else {
                scales[t] = 1.0 / sum;
                for (int j = 0; j < states; j++) alpha[t][j] *= scales[t];
            }
        }
        Arrays.fill(beta[length - 1], 1.0);
        for (int t = length - 2; t >= 0; t--) {
            for (int i = 0; i < states; i++) {
                double acc = 0.0;
                for (int j = 0; j < states; j++) {
                    acc += hmm.trans[i][j] * hmm.emit[j][observations[t + 1]] * beta[t + 1][j];
                }
                beta[t][i] = acc * scales[t + 1];
            }
        }
        double[][] gamma = new double[length][states];
        double[][][] xi = new double[Math.max(length - 1, 0)][states][states];
        for (int t = 0; t < length; t++) {
            double normalizer = 0.0;
            for (int i = 0; i < states; i++) normalizer += alpha[t][i] * beta[t][i];
            if (normalizer <= 0.0) continue;
            for (int i = 0; i < states; i++) {
                gamma[t][i] = alpha[t][i] * beta[t][i] / normalizer;
            }
            if (t == length - 1) continue;
            double pairNormalizer = 0.0;
            for (int i = 0; i < states; i++) {
                for (int j = 0; j < states; j++) {
                    pairNormalizer += alpha[t][i] * hmm.trans[i][j]
                            * hmm.emit[j][observations[t + 1]] * beta[t + 1][j];
                }
            }
            if (pairNormalizer <= 0.0) continue;
            for (int i = 0; i < states; i++) {
                for (int j = 0; j < states; j++) {
                    xi[t][i][j] = alpha[t][i] * hmm.trans[i][j]
                            * hmm.emit[j][observations[t + 1]] * beta[t + 1][j] / pairNormalizer;
                }
            }
        }
        double logLikelihood = NEG_INF;
        if (!impossible) {
            double logSum = 0.0;
            for (double scale : scales) logSum -= Math.log(scale);
            logLikelihood = logSum;
        }
        return new Posterior(gamma, xi, alpha, beta, scales, logLikelihood);
    }

    /** Unscaled forward recursion in the probability domain (short sequences only). */
    static double plainForward(Hmm hmm, int[] observations) {
        int states = hmm.states;
        double[] previous = new double[states];
        for (int i = 0; i < states; i++) {
            previous[i] = hmm.start[i] * hmm.emit[i][observations[0]];
        }
        for (int t = 1; t < observations.length; t++) {
            double[] current = new double[states];
            for (int j = 0; j < states; j++) {
                double acc = 0.0;
                for (int i = 0; i < states; i++) acc += previous[i] * hmm.trans[i][j];
                current[j] = acc * hmm.emit[j][observations[t]];
            }
            previous = current;
        }
        return rowSum(previous);
    }

    /** Independent log-sum-exp forward recursion for long sequences. */
    static double logForward(Hmm hmm, int[] observations) {
        int states = hmm.states;
        double[] previous = new double[states];
        for (int i = 0; i < states; i++) {
            previous[i] = logValue(hmm.start[i]) + logValue(hmm.emit[i][observations[0]]);
        }
        for (int t = 1; t < observations.length; t++) {
            double[] current = new double[states];
            for (int j = 0; j < states; j++) {
                double acc = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < states; i++) {
                    acc = logAdd(acc, previous[i] + logValue(hmm.trans[i][j]));
                }
                current[j] = acc + logValue(hmm.emit[j][observations[t]]);
            }
            previous = current;
        }
        double total = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < states; i++) total = logAdd(total, previous[i]);
        return total == Double.NEGATIVE_INFINITY ? NEG_INF : total;
    }

    static double logValue(double value) {
        return value <= 0.0 ? Double.NEGATIVE_INFINITY : Math.log(value);
    }

    static double logAdd(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) return b;
        if (b == Double.NEGATIVE_INFINITY) return a;
        double max = Math.max(a, b);
        return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
    }

    /** Exhaustive enumeration: likelihood, marginals and pair marginals. */
    static class Brute {
        double likelihood;
        double[][] gamma;
        double[][][] xi;
    }

    static Brute brute(Hmm hmm, int[] observations) {
        Brute result = new Brute();
        result.gamma = new double[observations.length][hmm.states];
        result.xi = new double[Math.max(observations.length - 1, 0)][hmm.states][hmm.states];
        int[] path = new int[observations.length];
        walk(hmm, observations, path, 0, result);
        for (int t = 0; t < observations.length; t++) {
            for (int i = 0; i < hmm.states; i++) {
                result.gamma[t][i] = result.likelihood == 0.0 ? 0.0
                        : result.gamma[t][i] / result.likelihood;
            }
        }
        for (int t = 0; t + 1 < observations.length; t++) {
            for (int i = 0; i < hmm.states; i++) {
                for (int j = 0; j < hmm.states; j++) {
                    result.xi[t][i][j] = result.likelihood == 0.0 ? 0.0
                            : result.xi[t][i][j] / result.likelihood;
                }
            }
        }
        return result;
    }

    private static void walk(Hmm hmm, int[] observations, int[] path, int depth, Brute result) {
        if (depth == path.length) {
            double probability = hmm.start[path[0]] * hmm.emit[path[0]][observations[0]];
            for (int t = 1; t < path.length; t++) {
                probability *= hmm.trans[path[t - 1]][path[t]] * hmm.emit[path[t]][observations[t]];
            }
            result.likelihood += probability;
            for (int t = 0; t < path.length; t++) {
                result.gamma[t][path[t]] += probability;
                if (t + 1 < path.length) result.xi[t][path[t]][path[t + 1]] += probability;
            }
            return;
        }
        for (int state = 0; state < hmm.states; state++) {
            path[depth] = state;
            walk(hmm, observations, path, depth + 1, result);
        }
    }

    /** One Baum-Welch (EM) update from the posterior statistics. */
    static Hmm baumWelchStep(Hmm hmm, int[] observations, Posterior posterior) {
        int states = hmm.states;
        int alphabet = hmm.alphabet;
        double[] start = posterior.gamma[0].clone();
        normalize(start, hmm.start);
        double[][] trans = new double[states][states];
        for (int i = 0; i < states; i++) {
            double denominator = 0.0;
            for (int t = 0; t + 1 < observations.length; t++) denominator += posterior.gamma[t][i];
            if (denominator <= 0.0) {
                trans[i] = hmm.trans[i].clone();
                continue;
            }
            for (int j = 0; j < states; j++) {
                double numerator = 0.0;
                for (int t = 0; t + 1 < observations.length; t++) numerator += posterior.xi[t][i][j];
                trans[i][j] = numerator / denominator;
            }
            normalize(trans[i], hmm.trans[i]);
        }
        double[][] emit = new double[states][alphabet];
        for (int j = 0; j < states; j++) {
            double denominator = 0.0;
            for (int t = 0; t < observations.length; t++) denominator += posterior.gamma[t][j];
            if (denominator <= 0.0) {
                emit[j] = hmm.emit[j].clone();
                continue;
            }
            for (int k = 0; k < alphabet; k++) {
                double numerator = 0.0;
                for (int t = 0; t < observations.length; t++) {
                    if (observations[t] == k) numerator += posterior.gamma[t][j];
                }
                emit[j][k] = numerator / denominator;
            }
            normalize(emit[j], hmm.emit[j]);
        }
        return new Hmm(start, trans, emit);
    }

    static void normalize(double[] row, double[] fallback) {
        double sum = 0.0;
        for (double value : row) sum += value;
        if (sum <= 0.0) {
            System.arraycopy(fallback, 0, row, 0, row.length);
            return;
        }
        for (int i = 0; i < row.length; i++) row[i] /= sum;
    }

    static double rowSum(double[] row) {
        double sum = 0.0;
        for (double value : row) sum += value;
        return sum;
    }

    static Hmm lcgHmm(int states, int alphabet, int seed) {
        double[] start = new double[states];
        double[][] trans = new double[states][states];
        double[][] emit = new double[states][alphabet];
        int state = seed;
        for (int i = 0; i < states; i++) {
            state = (state * 75 + 74) % 65537;
            start[i] = state % 97 + 1;
        }
        for (int i = 0; i < states; i++) {
            for (int j = 0; j < states; j++) {
                state = (state * 75 + 74) % 65537;
                trans[i][j] = state % 89 + 1;
            }
            normalize(trans[i], trans[i]);
        }
        for (int i = 0; i < states; i++) {
            for (int j = 0; j < alphabet; j++) {
                state = (state * 75 + 74) % 65537;
                emit[i][j] = state % 83 + 1;
            }
            normalize(emit[i], emit[i]);
        }
        normalize(start, start);
        return new Hmm(start, trans, emit);
    }

    public static void main(String[] args) {
        Hmm weather = canonicalWeather();
        int[] observations = {0, 1, 2};
        Posterior posterior = forwardBackward(weather, observations);
        check(Math.abs(Math.exp(posterior.logLikelihood) - 0.033612) <= 1e-9,
                "canonical likelihood 0.033612");
        check(Math.abs(posterior.gamma[0][0] - 0.23170296322741876) <= 1e-9,
                "gamma_1(Rainy) = 0.231702963");
        check(Math.abs(posterior.gamma[1][0] - 0.6240628347018923) <= 1e-9,
                "gamma_2(Rainy) = 0.624062835");
        check(Math.abs(posterior.gamma[2][0] - 0.863977151017494) <= 1e-9,
                "gamma_3(Rainy) = 0.863977151");
        check(Math.abs(posterior.scales[0] - 1.0 / (0.6 * 0.1 + 0.4 * 0.6)) <= 1e-12,
                "the first scale factor is 1 / sum_j alpha_1(j)");

        Brute weatherBrute = brute(weather, observations);
        check(Math.abs(weatherBrute.likelihood - Math.exp(posterior.logLikelihood)) <= 1e-12,
                "brute-force likelihood agrees");
        for (int t = 0; t < observations.length; t++) {
            double column = rowSum(posterior.gamma[t]);
            check(Math.abs(column - 1.0) <= 1e-12, "gamma columns sum to 1 at t = " + t);
            for (int i = 0; i < weather.states; i++) {
                check(Math.abs(posterior.gamma[t][i] - weatherBrute.gamma[t][i]) <= 1e-12,
                        "gamma matches brute force at t = " + t + ", state " + i);
            }
            check(posterior.gamma[t][0] >= 0.0 && posterior.gamma[t][0] <= 1.0,
                    "gamma inside [0, 1]");
        }
        for (int t = 0; t + 1 < observations.length; t++) {
            double total = 0.0;
            for (int i = 0; i < weather.states; i++) {
                double rowTotal = 0.0;
                for (int j = 0; j < weather.states; j++) {
                    total += posterior.xi[t][i][j];
                    rowTotal += posterior.xi[t][i][j];
                    check(Math.abs(posterior.xi[t][i][j] - weatherBrute.xi[t][i][j]) <= 1e-12,
                            "xi matches brute force at t = " + t);
                }
                check(Math.abs(rowTotal - posterior.gamma[t][i]) <= 1e-12,
                        "sum_j xi_t(i, j) = gamma_t(i) at t = " + t);
            }
            check(Math.abs(total - 1.0) <= 1e-12, "xi sums to 1 at t = " + t);
        }
        check(Math.abs(plainForward(weather, observations) - Math.exp(posterior.logLikelihood))
                <= 1e-12, "unscaled forward agrees on short sequences");
        check(Math.abs(logForward(weather, observations) - posterior.logLikelihood) <= 1e-9,
                "log-sum-exp forward agrees");

        int seed = 5;
        for (int trial = 0; trial < 6; trial++) {
            Hmm hmm = lcgHmm(4, 3, 700 + trial * 131);
            int length = 6;
            int[] obs = new int[length];
            for (int t = 0; t < length; t++) {
                seed = (seed * 75 + 74) % 65537;
                obs[t] = seed % 3;
            }
            Posterior p = forwardBackward(hmm, obs);
            Brute b = brute(hmm, obs);
            check(Math.abs(b.likelihood - Math.exp(p.logLikelihood)) <= 1e-12,
                    "likelihood matches brute force, trial " + trial);
            check(Math.abs(plainForward(hmm, obs) - b.likelihood) <= 1e-12,
                    "plain forward matches brute force, trial " + trial);
            check(Math.abs(logForward(hmm, obs) - p.logLikelihood) <= 1e-9,
                    "log forward matches scaled forward, trial " + trial);
            for (int t = 0; t < length; t++) {
                double column = 0.0;
                for (int i = 0; i < hmm.states; i++) {
                    column += p.gamma[t][i];
                    check(Math.abs(p.gamma[t][i] - b.gamma[t][i]) <= 1e-12,
                            "gamma matches brute force, trial " + trial);
                }
                check(Math.abs(column - 1.0) <= 1e-12, "gamma column sums to 1, trial " + trial);
            }
            for (int t = 0; t + 1 < length; t++) {
                for (int i = 0; i < hmm.states; i++) {
                    double rowTotal = 0.0;
                    for (int j = 0; j < hmm.states; j++) {
                        rowTotal += p.xi[t][i][j];
                        check(Math.abs(p.xi[t][i][j] - b.xi[t][i][j]) <= 1e-12,
                                "xi matches brute force, trial " + trial);
                    }
                    check(Math.abs(rowTotal - p.gamma[t][i]) <= 1e-12,
                            "xi row sums to gamma, trial " + trial);
                }
            }
            check(Arrays.equals(p.gamma[0], forwardBackward(hmm, obs).gamma[0]),
                    "deterministic re-run, trial " + trial);
        }

        Hmm longHmm = lcgHmm(6, 4, 20250916);
        int[] longObs = new int[300];
        seed = 919;
        for (int t = 0; t < longObs.length; t++) {
            seed = (seed * 75 + 74) % 65537;
            longObs[t] = seed % 4;
        }
        Posterior longPosterior = forwardBackward(longHmm, longObs);
        check(longPosterior.logLikelihood < 0.0 && longPosterior.logLikelihood != NEG_INF,
                "300-step likelihood stays finite");
        check(Math.abs(logForward(longHmm, longObs) - longPosterior.logLikelihood) <= 1e-9,
                "300-step scaled and log-sum-exp likelihoods agree");
        for (int t = 0; t < longObs.length; t++) {
            double column = 0.0;
            for (int i = 0; i < longHmm.states; i++) {
                column += longPosterior.gamma[t][i];
                check(longPosterior.gamma[t][i] >= -1e-15 && longPosterior.gamma[t][i] <= 1.0 + 1e-12,
                        "long-sequence gamma inside [0, 1]");
            }
            check(Math.abs(column - 1.0) <= 1e-9, "long-sequence gamma column sums to 1");
        }

        Hmm withZeros = new Hmm(new double[]{1.0, 0.0},
                new double[][]{{0.5, 0.5}, {0.5, 0.5}},
                new double[][]{{1.0, 0.0}, {0.0, 1.0}});
        Posterior zeroPosterior = forwardBackward(withZeros, new int[]{0, 0, 0});
        check(Math.abs(zeroPosterior.gamma[0][1]) <= 1e-15, "state 1 cannot emit symbol 0");
        check(Math.abs(zeroPosterior.gamma[1][1]) <= 1e-15, "state 1 stays impossible later too");
        check(Math.abs(rowSum(zeroPosterior.gamma[0]) - 1.0) <= 1e-12,
                "the reachable state carries all the mass");
        check(Math.abs(Math.exp(zeroPosterior.logLikelihood) - 0.25) <= 1e-12,
                "only the path that never enters state 1 survives: 0.5 * 0.5");
        for (int t = 0; t < 3; t++) {
            for (int i = 0; i < 2; i++) {
                check(!Double.isNaN(zeroPosterior.gamma[t][i]), "no NaN in the zero-probability model");
            }
        }
        Posterior impossible = forwardBackward(new Hmm(new double[]{1.0},
                new double[][]{{1.0}}, new double[][]{{1.0, 0.0}}), new int[]{0, 1});
        check(impossible.logLikelihood == NEG_INF, "impossible sequence reports NEG_INF");
        check(Math.abs(impossible.gamma[0][0]) <= 1e-15, "impossible sequence reports zero posteriors");

        for (int trial = 0; trial < 4; trial++) {
            Hmm hmm = lcgHmm(3, 3, 4242 + trial * 79);
            int[] obs = new int[12];
            seed = 31337 + trial;
            for (int t = 0; t < obs.length; t++) {
                seed = (seed * 75 + 74) % 65537;
                obs[t] = seed % 3;
            }
            double initial = forwardBackward(hmm, obs).logLikelihood;
            Hmm current = hmm;
            double previous = initial;
            for (int iteration = 0; iteration < 5; iteration++) {
                Posterior stats = forwardBackward(current, obs);
                Hmm updated = baumWelchStep(current, obs, stats);
                for (int i = 0; i < updated.states; i++) {
                    check(Math.abs(rowSum(updated.trans[i]) - 1.0) <= 1e-12,
                            "updated transition rows stay stochastic");
                    check(Math.abs(rowSum(updated.emit[i]) - 1.0) <= 1e-12,
                            "updated emission rows stay stochastic");
                }
                check(Math.abs(rowSum(updated.start) - 1.0) <= 1e-12,
                        "updated start vector stays stochastic");
                double likelihood = forwardBackward(updated, obs).logLikelihood;
                check(likelihood >= previous - 1e-9,
                        "Baum-Welch never decreases the likelihood, trial " + trial);
                current = updated;
                previous = likelihood;
            }
            check(previous > initial + 1e-9,
                    "five EM iterations strictly improve the likelihood, trial " + trial);
        }

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
