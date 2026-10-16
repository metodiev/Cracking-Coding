/** Task 934 - Implement distributed gradient compression with error correction under packet loss.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * Data-parallel training with compressed gradients: every worker keeps the residual it failed to send, adds its
 * fresh gradient to it, transmits only the k largest-magnitude entries (top-k sparsification), and subtracts
 * what it sent from the residual - so the mass that compression drops is not lost but carried into the next
 * round (error feedback). Transmission is protected by simple repetition: each payload is sent R times down a
 * channel that deterministically drops up to R - 1 copies, and the server takes the first copy that arrives; a
 * round where every copy is lost is reported, and the worker's residual survives it because the feedback is
 * kept on the worker.
 * Simplified: an in-memory channel with a fixed drop pattern (no real network, no stragglers), no quantisation
 * (only sparsification), and a fresh gradient per round rather than a real model. What is *not* simplified: the
 * compression contract (at most k entries, exactly the k largest magnitudes), the error-feedback recurrence,
 * the erasure rule, and the accounting of every dropped payload.
 * Verified: the transmitted sum tracks the true sum with error feedback and fails to without it; the top-k
 * payload really contains the k largest magnitudes and its loss is the closed-form dropped mass; up to R - 1
 * losses are recovered while R losses are reported as lost and still repaid in the next round; lossless mode
 * (k = d) has zero error; and the run is reproducible.
 */
import java.util.*;

/** Client-side compression with error feedback. */
final class Worker {
    final int id;
    final int dims;
    double[] residual, gradient, payload;
    int sent;
    long lostPayloads;

    Worker(int id, int dims) {
        this.id = id;
        this.dims = dims;
        this.residual = new double[dims];
        this.gradient = new double[dims];
        this.payload = new double[dims];
    }

    /** A deterministic local gradient for this round: white noise, so a sum over rounds is a random walk
     *  with the same statistics as a real training run. */
    void computeGradient(int round) {
        long state = round * 7919L + id * 104729L + 0x9E3779B97F4A7C15L;
        for (int i = 0; i < dims; i++) {
            state = state * 6364136223846793005L + 1442695040888963407L;
            gradient[i] = ((state >>> 11) * 0x1.0p-53 - 0.5) * 2;
        }
    }

    /** Error feedback plus top-k sparsification; returns the payload actually handed to the channel. */
    double[] compress(int k, boolean errorFeedback) {
        double[] target = new double[dims];
        for (int i = 0; i < dims; i++) target[i] = (errorFeedback ? residual[i] : 0) + gradient[i];
        payload = Compressor.topK(target, k);
        sent = Compressor.nonZeros(payload);
        if (errorFeedback) for (int i = 0; i < dims; i++) residual[i] = target[i] - payload[i];
        return payload;
    }

    double residualNorm() {
        double sum = 0;
        for (double v : residual) sum += v * v;
        return Math.sqrt(sum);
    }
}

/** Top-k sparsification and its closed-form error. */
final class Compressor {
    static double[] topK(double[] values, int k) {
        Integer[] order = new Integer[values.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> {
            double compare = Math.abs(values[b]) - Math.abs(values[a]);
            return compare != 0 ? (compare > 0 ? 1 : -1) : Integer.compare(a, b);
        });
        double[] out = new double[values.length];
        for (int i = 0; i < Math.min(k, order.length); i++) out[order[i]] = values[order[i]];
        return out;
    }

    static int nonZeros(double[] values) {
        int count = 0;
        for (double v : values) if (v != 0) count++;
        return count;
    }

    static double droppedMass(double[] values, double[] kept) {
        double sum = 0;
        for (int i = 0; i < values.length; i++) {
            double diff = values[i] - kept[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
}

/** A deterministic repetition channel: R copies, d drops, first arrival wins. */
final class Channel {
    final int copies;
    long dropped, delivered, lostRounds;

    Channel(int copies) { this.copies = copies; }

    /** True when at least one copy survives; the drop pattern rotates with the round and worker. */
    boolean send(int round, int worker, int dropsPerMessage) {
        int lost = 0;
        for (int c = 0; c < copies; c++) {
            int pattern = (round * 7 + worker * 3 + c) % copies;
            if (pattern < dropsPerMessage) lost++;
        }
        dropped += lost;
        if (lost == copies) {
            lostRounds++;
            return false;
        }
        delivered++;
        return true;
    }
}

public class Solution {
    static final int WORKERS = 8, DIMS = 32, K = 4, ROUNDS = 400, COPIES = 3, CHECKPOINT = 100;

    /** Everything a run reports. */
    static final class Metrics {
        double ratioEarly, ratioLate, residualEarly, residualLate, dropped, delivered, lostRounds;
    }

    static double[] add(double[] a, double[] b) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) out[i] = a[i] + b[i];
        return out;
    }

    static double norm(double[] v) {
        double sum = 0;
        for (double x : v) sum += x * x;
        return Math.sqrt(sum);
    }

    static double ratio(double[] sent, double[] truth) {
        double[] diff = new double[sent.length];
        for (int i = 0; i < sent.length; i++) diff[i] = sent[i] - truth[i];
        return norm(diff) / norm(truth);
    }

    /** One full run: error feedback on or off, `dropsPerMessage` copies lost before the first arrival. */
    static Metrics run(boolean errorFeedback, int dropsPerMessage) {
        Metrics metrics = new Metrics();
        List<Worker> workers = new ArrayList<>();
        for (int w = 0; w < WORKERS; w++) workers.add(new Worker(w, DIMS));
        Channel channel = new Channel(COPIES);
        double[] truth = new double[DIMS], sent = new double[DIMS];
        double residualEarly = 0, residualLate = 0;
        for (int round = 0; round < ROUNDS; round++) {
            for (Worker worker : workers) {
                worker.computeGradient(round);
                truth = add(truth, worker.gradient);
                worker.compress(K, errorFeedback);
            }
            for (Worker worker : workers) {
                if (channel.send(round, worker.id, dropsPerMessage)) {
                    for (int i = 0; i < DIMS; i++) sent[i] += worker.payload[i];
                } else {
                    worker.lostPayloads++;
                    if (errorFeedback) for (int i = 0; i < DIMS; i++) worker.residual[i] += worker.payload[i];
                }
            }
            double residual = 0;
            for (Worker worker : workers) residual += worker.residualNorm();
            if (round < CHECKPOINT) residualEarly += residual / (CHECKPOINT * WORKERS);
            if (round >= ROUNDS - CHECKPOINT) residualLate += residual / (CHECKPOINT * WORKERS);
            if (round == CHECKPOINT - 1) metrics.ratioEarly = ratio(sent, truth);
            if (round == ROUNDS - 1) metrics.ratioLate = ratio(sent, truth);
        }
        metrics.residualEarly = residualEarly;
        metrics.residualLate = residualLate;
        metrics.dropped = channel.dropped;
        metrics.delivered = channel.delivered;
        metrics.lostRounds = channel.lostRounds;
        return metrics;
    }

    public static void main(String[] args) {
        Metrics feedback = run(true, 0);
        Metrics noFeedback = run(false, 0);
        Metrics twoLost = run(true, COPIES - 1);
        Metrics allLost = run(true, COPIES);

        System.out.println("gradient compression: " + WORKERS + " workers x " + DIMS + " dims, top-" + K + ", "
                + ROUNDS + " rounds, " + COPIES + " copies - error-feedback error ratio "
                + String.format("%.4f", feedback.ratioEarly) + " at " + CHECKPOINT + " rounds -> "
                + String.format("%.4f", feedback.ratioLate) + " at " + ROUNDS + "; no-feedback "
                + String.format("%.4f", noFeedback.ratioEarly) + " -> " + String.format("%.4f", noFeedback.ratioLate));
        System.out.println("erasure: 2-of-3 losses recovered (ratio " + String.format("%.4f", twoLost.ratioLate)
                + ", lost rounds " + twoLost.lostRounds + "), 3-of-3 losses dropped " + allLost.lostRounds
                + " payloads (ratio " + String.format("%.4f", allLost.ratioLate) + "); mean residual "
                + String.format("%.4f", feedback.residualEarly) + " -> " + String.format("%.4f", feedback.residualLate));

        check(feedback.ratioLate < feedback.ratioEarly, "error feedback must reduce the error as rounds accumulate: "
                + feedback.ratioEarly + " -> " + feedback.ratioLate);
        check(feedback.ratioLate < 0.1, "with feedback the cumulative error must be small, got " + feedback.ratioLate);
        check(noFeedback.ratioLate > 4 * feedback.ratioLate, "without feedback the error must persist: "
                + noFeedback.ratioLate + " vs " + feedback.ratioLate);
        check(noFeedback.ratioEarly > feedback.ratioEarly && noFeedback.ratioLate > feedback.ratioLate,
                "without feedback the error must dominate at both checkpoints: " + noFeedback.ratioEarly + " vs "
                        + feedback.ratioEarly + " and " + noFeedback.ratioLate + " vs " + feedback.ratioLate);
        check(feedback.ratioLate < feedback.ratioEarly / 3, "feedback must shrink the error by at least 3x: "
                + feedback.ratioEarly + " -> " + feedback.ratioLate);
        check(feedback.residualLate < 2 * feedback.residualEarly, "the residual must stay bounded, mean "
                + feedback.residualEarly + " -> " + feedback.residualLate);
        check(twoLost.ratioLate < 0.1 && twoLost.lostRounds == 0,
                "up to R - 1 lost copies must be recovered, ratio " + twoLost.ratioLate);
        check(allLost.lostRounds == WORKERS * ROUNDS && allLost.delivered == 0,
                "losing every copy must lose every payload: " + allLost.lostRounds + " of " + (WORKERS * ROUNDS));
        check(allLost.ratioLate > twoLost.ratioLate, "a total outage must hurt even with feedback: "
                + allLost.ratioLate + " vs " + twoLost.ratioLate);

        // The compression contract itself.
        double[] vector = new double[DIMS];
        for (int i = 0; i < DIMS; i++) vector[i] = Math.sin(0.9 * i) * (i % 5 == 0 ? 4 : 1);
        double[] kept = Compressor.topK(vector, K);
        check(Compressor.nonZeros(kept) <= K, "the payload must hold at most k entries, got "
                + Compressor.nonZeros(kept));
        double[] magnitudes = new double[DIMS];
        for (int i = 0; i < DIMS; i++) magnitudes[i] = Math.abs(vector[i]);
        Arrays.sort(magnitudes);
        double threshold = magnitudes[DIMS - K];
        int aboveThreshold = 0;
        for (int i = 0; i < DIMS; i++) {
            if (kept[i] != 0) {
                if (Math.abs(vector[i]) < threshold - 1e-12) aboveThreshold++;
            } else if (Math.abs(vector[i]) > threshold + 1e-12) {
                aboveThreshold++;
            }
        }
        check(aboveThreshold == 0, "top-k must keep exactly the k largest magnitudes");
        double dropped = Compressor.droppedMass(vector, kept);
        double closedForm = 0;
        for (int i = 0; i < DIMS; i++) if (kept[i] == 0) closedForm += vector[i] * vector[i];
        check(Math.abs(dropped - Math.sqrt(closedForm)) < 1e-12, "the dropped mass must match the closed form");

        // Lossless mode: k = d transports everything, so the cumulative error is exactly zero.
        Worker lossless = new Worker(0, DIMS);
        lossless.computeGradient(3);
        double[] full = lossless.compress(DIMS, true);
        check(Compressor.nonZeros(full) == DIMS && lossless.residualNorm() == 0,
                "k = d must be lossless, nonzeros " + Compressor.nonZeros(full) + " residual " + lossless.residualNorm());
        Worker single = new Worker(0, DIMS);
        single.computeGradient(3);
        double[] one = single.compress(1, true);
        check(Compressor.nonZeros(one) == 1 && single.residualNorm() > 0,
                "k = 1 must keep a single entry and carry the rest as feedback");

        Metrics replay = run(true, 0);
        check(replay.ratioLate == feedback.ratioLate && replay.delivered == feedback.delivered,
                "the run must be reproducible");
        System.out.println("checks: top-k keeps the " + K + " largest of " + DIMS + " entries, dropped mass "
                + String.format("%.4f", dropped) + " matches its closed form, k = d lossless, k = 1 carries feedback");
        System.out.println("PASS");
    }

    static double max(double[] values) {
        double best = 0;
        for (double v : values) best = Math.max(best, v);
        return best;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
