/** Task 714 - Implement federated averaging simulation with stragglers and compression.
 * (Tier: Expert, CTCI Beyond -- Machine Learning & AI)
 *
 * Federated averaging alternates two steps: every participating client takes local gradient steps on
 * its own data, and the server replaces the model by the *average* of what the clients produced. This
 * simulation keeps the model a linear regressor (so the loss, the gradient and the exact aggregated
 * delta are all computable in closed form) and adds the two realities of production federated
 * learning:
 *
 *   - stragglers: only a subset of clients reports in a given round, and the server must average over
 *     exactly those who reported (a client that does not answer simply is not in the denominator);
 *   - compression: each client sends only the k largest-magnitude coordinates of its update (top-k
 *     sparsification) instead of a dense vector. Naively throwing the rest away biases the update
 *     towards always the same coordinates, so the client keeps the discarded part as an *error
 *     feedback* residual that is added to the next round's update. With k equal to the dimension the
 *     compression is lossless and the residual stays exactly zero, which the checks exploit.
 *
 * Verified: the analytic gradient against central finite differences, the server update against a
 * brute-force average of the participants' deltas, convergence of the test loss (to a small fraction
 * of its initial value) under full participation, the structural properties of straggler rounds (a
 * round with no participants leaves the model untouched, one participant reproduces its own delta),
 * lossless top-k at k = dimension with a zero residual, the transmitted-value accounting for the
 * compression ratio, that error feedback converges at least as well as feedback-free sparsification,
 * and rejection of malformed parameters. Everything is driven by a fixed-seed splitmix64 stream.
 */
import java.util.*;

class FederatedAveragingSim {
    static final class RoundResult {
        final int participants;
        final double[] update;
        final int transmittedValues;

        RoundResult(int participants, double[] update, int transmittedValues) {
            this.participants = participants;
            this.update = update;
            this.transmittedValues = transmittedValues;
        }
    }

    private final int clients;
    private final int dimension;
    private final double learningRate;
    private final double[][][] features;
    private final double[][] targets;
    private final double[][] residuals;
    private double[] weights;
    private long transmittedTotal;

    FederatedAveragingSim(int clients, int dimension, int samplesPerClient,
                          double learningRate, long seed) {
        if (clients < 1) throw new IllegalArgumentException("need at least one client");
        if (dimension < 1) throw new IllegalArgumentException("dimension must be positive");
        if (samplesPerClient < 1) throw new IllegalArgumentException("need at least one sample");
        if (!(learningRate > 0)) throw new IllegalArgumentException("learning rate must be positive");
        this.clients = clients;
        this.dimension = dimension;
        this.learningRate = learningRate;
        this.features = new double[clients][samplesPerClient][dimension];
        this.targets = new double[clients][samplesPerClient];
        this.residuals = new double[clients][dimension];
        this.weights = new double[dimension];
        long state = seed;
        double[] truth = new double[dimension];
        for (int k = 0; k < dimension; k++) {
            state = mix(state + k);
            truth[k] = ((state >>> 11) * 0x1.0p-53) * 2 - 1;
        }
        for (int client = 0; client < clients; client++) {
            for (int sample = 0; sample < samplesPerClient; sample++) {
                double target = 0;
                for (int k = 0; k < dimension; k++) {
                    state = mix(state + client * 31L + sample * 17L + k);
                    features[client][sample][k] = (state >>> 11) * 0x1.0p-53;
                    target += truth[k] * features[client][sample][k];
                }
                state = mix(state + 7919L);
                targets[client][sample] = target + 0.01 * (((state >>> 11) * 0x1.0p-53) - 0.5);
            }
        }
    }

    private static long mix(long value) {
        long z = value + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int clients() {
        return clients;
    }

    int dimension() {
        return dimension;
    }

    double[] weights() {
        return weights.clone();
    }

    long transmittedTotal() {
        return transmittedTotal;
    }

    /** Mean squared error of one client. */
    double loss(int client) {
        double total = 0;
        for (int sample = 0; sample < features[client].length; sample++) {
            double error = predict(client, sample) - targets[client][sample];
            total += error * error;
        }
        return total / features[client].length;
    }

    double totalLoss() {
        double total = 0;
        for (int client = 0; client < clients; client++) total += loss(client);
        return total / clients;
    }

    private double predict(int client, int sample) {
        double value = 0;
        for (int k = 0; k < dimension; k++) value += weights[k] * features[client][sample][k];
        return value;
    }

    /** Gradient of the client's mean squared error. */
    double[] gradient(int client) {
        double[] gradient = new double[dimension];
        int samples = features[client].length;
        for (int sample = 0; sample < samples; sample++) {
            double error = predict(client, sample) - targets[client][sample];
            for (int k = 0; k < dimension; k++) gradient[k] += 2 * error * features[client][sample][k];
        }
        for (int k = 0; k < dimension; k++) gradient[k] /= samples;
        return gradient;
    }

    /** Central finite differences: the independent oracle for the analytic gradient. */
    double[] numericGradient(int client, double epsilon) {
        double[] numeric = new double[dimension];
        for (int k = 0; k < dimension; k++) {
            double saved = weights[k];
            weights[k] = saved + epsilon;
            double up = loss(client);
            weights[k] = saved - epsilon;
            double down = loss(client);
            weights[k] = saved;
            numeric[k] = (up - down) / (2 * epsilon);
        }
        return numeric;
    }

    /** One local step: the client's contribution before compression. */
    double[] localDelta(int client) {
        double[] gradient = gradient(client);
        double[] delta = new double[dimension];
        for (int k = 0; k < dimension; k++) delta[k] = -learningRate * gradient[k];
        return delta;
    }

    /** Top-k by magnitude; ties go to the lower coordinate index so the result is deterministic. */
    static double[] topK(double[] vector, int keep) {
        if (keep < 1 || keep > vector.length) throw new IllegalArgumentException("bad keep: " + keep);
        Integer[] order = new Integer[vector.length];
        for (int k = 0; k < vector.length; k++) order[k] = k;
        Arrays.sort(order, (left, right) -> {
            int byMagnitude = Double.compare(Math.abs(vector[right]), Math.abs(vector[left]));
            return byMagnitude != 0 ? byMagnitude : Integer.compare(left, right);
        });
        double[] sparse = new double[vector.length];
        for (int i = 0; i < keep; i++) sparse[order[i]] = vector[order[i]];
        return sparse;
    }

    static int nonZeroCount(double[] vector) {
        int count = 0;
        for (double value : vector) if (value != 0) count++;
        return count;
    }

    /** Full symmetric participation schedule. */
    boolean[] everyone() {
        boolean[] alive = new boolean[clients];
        Arrays.fill(alive, true);
        return alive;
    }

    /** Deterministic straggler pattern: a client reports unless round + client is divisible by stride. */
    boolean[] stragglers(int round, int stride) {
        if (stride < 1) throw new IllegalArgumentException("stride must be positive");
        boolean[] alive = new boolean[clients];
        for (int client = 0; client < clients; client++) {
            alive[client] = (round + client) % stride != 0;
        }
        return alive;
    }

    static boolean[] only(boolean[] alive, int client) {
        boolean[] single = new boolean[alive.length];
        single[client] = true;
        return single;
    }

    static int countTrue(boolean[] flags) {
        int count = 0;
        for (boolean flag : flags) if (flag) count++;
        return count;
    }

    /** One federated round with top-k compression and optional error feedback. */
    RoundResult round(boolean[] participants, int keep, boolean useErrorFeedback) {
        if (keep < 1 || keep > dimension) throw new IllegalArgumentException("bad keep: " + keep);
        int participating = countTrue(participants);
        if (participating == 0) return new RoundResult(0, new double[dimension], 0);
        double[] update = new double[dimension];
        int transmitted = 0;
        for (int client = 0; client < clients; client++) {
            if (!participants[client]) continue;
            double[] delta = localDelta(client);
            double[] message;
            if (keep == dimension) {
                message = delta;
                residuals[client] = new double[dimension];
            } else {
                double[] carried = new double[dimension];
                for (int k = 0; k < dimension; k++) {
                    carried[k] = delta[k] + (useErrorFeedback ? residuals[client][k] : 0);
                }
                message = topK(carried, keep);
                for (int k = 0; k < dimension; k++) {
                    residuals[client][k] = useErrorFeedback ? carried[k] - message[k] : 0;
                }
            }
            transmitted += nonZeroCount(message);
            for (int k = 0; k < dimension; k++) update[k] += message[k] / participating;
        }
        transmittedTotal += transmitted;
        return new RoundResult(participating, update, transmitted);
    }

    /** Brute-force average of the participants' uncompressed deltas. */
    double[] denseUpdate(boolean[] participants) {
        int participating = countTrue(participants);
        if (participating == 0) return new double[dimension];
        double[] update = new double[dimension];
        for (int client = 0; client < clients; client++) {
            if (!participants[client]) continue;
            double[] delta = localDelta(client);
            for (int k = 0; k < dimension; k++) update[k] += delta[k] / participating;
        }
        return update;
    }

    void apply(double[] update) {
        for (int k = 0; k < dimension; k++) weights[k] += update[k];
    }

    double[] residualOf(int client) {
        return residuals[client].clone();
    }

    double residualNorm(int client) {
        double total = 0;
        for (double value : residuals[client]) total += value * value;
        return Math.sqrt(total);
    }
}

public class Solution {
    public static void main(String[] args) {
        int clients = 6;
        int dimension = 4;
        int samples = 24;
        double learningRate = 0.5;

        FederatedAveragingSim sim = new FederatedAveragingSim(clients, dimension, samples,
                learningRate, 0x714DEEDL);
        check(sim.clients() == clients && sim.dimension() == dimension, "the simulation is configured");
        double initialLoss = sim.totalLoss();
        check(initialLoss > 0, "the initial loss is positive");

        // --- the gradient oracle --------------------------------------------------
        double[] analytic = sim.gradient(0);
        double[] numeric = sim.numericGradient(0, 1e-6);
        double worstGradientError = 0;
        for (int k = 0; k < dimension; k++) {
            worstGradientError = Math.max(worstGradientError,
                    Math.abs(analytic[k] - numeric[k]) / Math.max(1, Math.abs(analytic[k])));
        }
        check(worstGradientError < 1e-6, "analytic vs numeric gradient: " + worstGradientError);
        double[] delta = sim.localDelta(0);
        for (int k = 0; k < dimension; k++) {
            check(Math.abs(delta[k] + learningRate * analytic[k]) < 1e-12,
                    "a local delta is minus the learning rate times the gradient");
        }

        // --- top-k sparsification -------------------------------------------------
        double[] sample = {0.5, -3.0, 0.25, 2.0};
        double[] sparse = FederatedAveragingSim.topK(sample, 2);
        check(FederatedAveragingSim.nonZeroCount(sparse) == 2, "top-k keeps exactly k entries");
        check(sparse[1] == -3.0 && sparse[3] == 2.0, "top-k keeps the largest magnitudes");
        check(sparse[0] == 0.0 && sparse[2] == 0.0, "the discarded entries are zeroed");
        check(Arrays.equals(FederatedAveragingSim.topK(sample, 4), sample),
                "k equal to the dimension is lossless");
        check(rejects(() -> FederatedAveragingSim.topK(sample, 0)), "k = 0 rejected");
        check(rejects(() -> FederatedAveragingSim.topK(sample, 5)), "k above the dimension rejected");

        // --- full participation, lossless: the server update is the exact mean ----
        boolean[] everyone = sim.everyone();
        double[] expected = sim.denseUpdate(everyone);
        FederatedAveragingSim.RoundResult lossless = sim.round(everyone, dimension, true);
        check(lossless.participants == clients, "everyone reports");
        check(lossless.transmittedValues == clients * dimension,
                "a dense round transmits every coordinate: " + lossless.transmittedValues);
        for (int k = 0; k < dimension; k++) {
            check(Math.abs(lossless.update[k] - expected[k]) < 1e-12,
                    "the aggregated update must be the brute-force mean");
        }
        for (int client = 0; client < clients; client++) {
            check(sim.residualNorm(client) == 0.0, "a lossless round leaves no residual");
        }

        // --- convergence under full participation ---------------------------------
        for (int round = 0; round < 60; round++) {
            sim.apply(sim.round(everyone, dimension, true).update);
        }
        double convergedLoss = sim.totalLoss();
        check(convergedLoss < 0.05 * initialLoss,
                "full participation must converge: " + convergedLoss + " vs " + initialLoss);

        // --- stragglers ------------------------------------------------------------
        FederatedAveragingSim straggling = new FederatedAveragingSim(clients, dimension, samples,
                learningRate, 0x714DEEDL);
        Set<Integer> participationCounts = new TreeSet<>();
        for (int round = 0; round < 60; round++) {
            boolean[] alive = straggling.stragglers(round, 3);
            int participating = FederatedAveragingSim.countTrue(alive);
            check(participating >= 1, "the straggler pattern must leave somebody");
            participationCounts.add(participating);
            double[] dense = straggling.denseUpdate(alive);
            FederatedAveragingSim.RoundResult result = straggling.round(alive, dimension, true);
            check(result.participants == participating, "the round counts the participants");
            for (int k = 0; k < dimension; k++) {
                check(Math.abs(result.update[k] - dense[k]) < 1e-12,
                        "straggler rounds must average over the participants only");
            }
            straggling.apply(result.update);
        }
        check(participationCounts.size() == 1 && participationCounts.contains(clients - clients / 3),
                "the straggler schedule really drops one client in three: " + participationCounts);
        check(straggling.totalLoss() < 0.1 * initialLoss,
                "straggling must still learn: " + straggling.totalLoss());

        FederatedAveragingSim alone = new FederatedAveragingSim(clients, dimension, samples,
                learningRate, 0x714DEEDL);
        boolean[] onlySecond = FederatedAveragingSim.only(everyone, 1);
        double[] ownDelta = alone.localDelta(1);
        FederatedAveragingSim.RoundResult single = alone.round(onlySecond, dimension, true);
        for (int k = 0; k < dimension; k++) {
            check(Math.abs(single.update[k] - ownDelta[k]) < 1e-12,
                    "with one participant the update is that client's delta");
        }
        boolean[] nobody = new boolean[clients];
        FederatedAveragingSim.RoundResult idle = alone.round(nobody, dimension, true);
        check(idle.participants == 0 && FederatedAveragingSim.nonZeroCount(idle.update) == 0,
                "a round with no participants produces no update");
        double[] before = alone.weights();
        alone.apply(idle.update);
        check(Arrays.equals(alone.weights(), before), "an idle round leaves the model untouched");

        // --- compressed rounds: ratio, residuals, convergence ----------------------
        FederatedAveragingSim compressed = new FederatedAveragingSim(clients, dimension, samples,
                learningRate, 0x714DEEDL);
        FederatedAveragingSim unfiltered = new FederatedAveragingSim(clients, dimension, samples,
                learningRate, 0x714DEEDL);
        long compressedValues = 0;
        long denseValues = 0;
        for (int round = 0; round < 80; round++) {
            FederatedAveragingSim.RoundResult withFeedback = compressed.round(everyone, 1, true);
            FederatedAveragingSim.RoundResult withoutFeedback = unfiltered.round(everyone, 1, false);
            check(withFeedback.transmittedValues == clients * 1,
                    "a top-1 round transmits one value per client");
            compressedValues += withFeedback.transmittedValues;
            denseValues += clients * dimension;
            compressed.apply(withFeedback.update);
            unfiltered.apply(withoutFeedback.update);
        }
        check(compressedValues * 4 == denseValues,
                "compression should cut the traffic fourfold: " + compressedValues + " vs " + denseValues);
        double withFeedbackLoss = compressed.totalLoss();
        double withoutFeedbackLoss = unfiltered.totalLoss();
        check(withFeedbackLoss < 0.05 * initialLoss,
                "error feedback must converge: " + withFeedbackLoss);
        check(withoutFeedbackLoss < initialLoss,
                "even feedback-free sparsification must improve: " + withoutFeedbackLoss);
        check(withFeedbackLoss < withoutFeedbackLoss,
                "error feedback should beat plain sparsification: " + withFeedbackLoss + " vs "
                        + withoutFeedbackLoss);
        double residualBound = 0;
        for (int client = 0; client < clients; client++) {
            residualBound = Math.max(residualBound, compressed.residualNorm(client));
        }
        check(residualBound < 10, "the error-feedback residual must stay bounded: " + residualBound);

        // --- rejection and determinism --------------------------------------------
        check(rejects(() -> new FederatedAveragingSim(0, 4, 8, 0.1, 1L)), "zero clients rejected");
        check(rejects(() -> new FederatedAveragingSim(2, 0, 8, 0.1, 1L)), "zero dimension rejected");
        check(rejects(() -> new FederatedAveragingSim(2, 4, 8, -0.1, 1L)), "negative rate rejected");
        check(rejects(() -> new FederatedAveragingSim(2, 4, 8, 0.1, 1L).round(everyone, dimension + 1, true)),
                "bad keep rejected");
        check(rejects(() -> sim.stragglers(0, 0)), "zero stride rejected");
        FederatedAveragingSim rebuilt = new FederatedAveragingSim(clients, dimension, samples,
                learningRate, 0x714DEEDL);
        check(Math.abs(rebuilt.totalLoss() - initialLoss) < 1e-12, "the same seed gives the same data");
        for (int trial = 0; trial < 3; trial++) {
            double[] first = new FederatedAveragingSim(clients, dimension, samples, learningRate,
                    0x714DEEDL).round(everyone, 2, true).update;
            double[] second = new FederatedAveragingSim(clients, dimension, samples, learningRate,
                    0x714DEEDL).round(everyone, 2, true).update;
            check(Arrays.equals(first, second), "the round must be deterministic, trial " + trial);
        }

        System.out.println("loss " + String.format(Locale.ROOT, "%.5f", initialLoss) + " -> "
                + String.format(Locale.ROOT, "%.5f", convergedLoss) + " (full), "
                + String.format(Locale.ROOT, "%.5f", withFeedbackLoss) + " (top-1 with feedback), "
                + String.format(Locale.ROOT, "%.5f", withoutFeedbackLoss) + " (top-1 without)");
        System.out.println("PASS");
    }

    static boolean rejects(Runnable action) {
        try {
            action.run();
        } catch (IllegalArgumentException | IllegalStateException expected) {
            return true;
        }
        return false;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
