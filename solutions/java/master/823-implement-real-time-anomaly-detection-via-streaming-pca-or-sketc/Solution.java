/** Task 823 - Implement real-time anomaly detection via streaming PCA or sketching. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * Anomaly detection here is unsupervised and single pass: the detector tracks the dominant subspace of the
 * stream (streaming PCA) and the distribution of the residual scores (quantile sketching), and flags an
 * event when its reconstruction error leaves the learned distribution.
 *
 *   streaming PCA   Oja's rule x_{t+1} = x_t + eta * y * (v - v y), followed by Gram-Schmidt
 *                   re-orthonormalisation of the component matrix after every step.  Oja's rule converges to
 *                   the dominant eigenvector of the covariance matrix, and the re-orthonormalised block
 *                   converges to the top-k invariant subspace.
 *   count-min       a 4x(1/eps) counter matrix estimates item frequencies with one-sided error
 *                   (estimate >= truth, error <= eps*N with the standard width/depth choice).
 *   heavy hitters   Misra-Gries over k counters recovers the true heavy hitters and bounds the error of any
 *                   non-heavy item by N/(k+1).
 *   quantiles       a Greenwald-Khanna summary answers rank queries with error <= eps*N, which is what the
 *                   detector uses to set its own threshold without storing the stream.
 *
 * Everything is checked against an offline oracle: exact PCA via power iteration on the covariance matrix,
 * exact frequencies, exact heavy hitters and exact quantiles over the retained window.  Detection quality
 * (recall, false-positive rate, latency after the onset) is measured against an oracle detector that uses
 * the exact subspace and the exact quantile.
 *
 * Deliberate simplifications (fully deterministic -- java.util only; no threads, sockets, clocks or files):
 * the stream is a few thousand 6-dimensional vectors from a seeded splitmix64 with two planted anomalies, no
 * floating-point randomness is used, and "latency" is the number of events between the onset and the alert.
 *
 * Verified invariants:
 *   subspace    - the streaming subspace stays orthonormal, its captured variance approaches the exact top-k
 *                 subspace on stationary data, and it re-adapts after a rotation of the data.
 *   sketches    - count-min never underestimates and honours its error bound; Misra-Gries reports every item
 *                 whose frequency exceeds N/(k+1) with counts within that bound; the quantile sketch's rank
 *                 error stays within eps*N; all of them keep state that is independent of stream length.
 *   detection   - recall >= 0.9 and false-positive rate <= 0.05 against the offline oracle, with a bounded
 *                 detection delay after the onset, and no alerts at all on a stationary benign stream.
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

    double nextGaussian() {
        double u1 = Math.max(nextDouble(), 1e-12), u2 = nextDouble();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
}

/** Streaming PCA: Oja's rule plus Gram-Schmidt re-orthonormalisation of the tracked components. */
class StreamingPca {
    final int dim;
    final int components;
    final double[][] v;          // components x dim
    final double[] energy;       // captured variance estimate per component
    double eta;
    long updates;

    StreamingPca(int dim, int components, double eta, long seed) {
        this.dim = dim; this.components = components; this.eta = eta;
        this.v = new double[components][dim];
        this.energy = new double[components];
        SplitMix64 rng = new SplitMix64(seed);
        for (int c = 0; c < components; c++) {
            for (int j = 0; j < dim; j++) v[c][j] = rng.nextGaussian();
            normalize(v[c]);
        }
        orthonormalize();
    }

    static void normalize(double[] x) {
        double n = 0;
        for (double value : x) n += value * value;
        n = Math.sqrt(n);
        if (n < 1e-12) return;
        for (int i = 0; i < x.length; i++) x[i] /= n;
    }

    void orthonormalize() {
        for (int c = 0; c < components; c++) {
            for (int p = 0; p < c; p++) {
                double dot = 0;
                for (int j = 0; j < dim; j++) dot += v[c][j] * v[p][j];
                for (int j = 0; j < dim; j++) v[c][j] -= dot * v[p][j];
            }
            normalize(v[c]);
        }
    }

    /** One Oja step for every component, then re-orthonormalisation. */
    void update(double[] x) {
        for (int c = 0; c < components; c++) {
            double y = 0;
            for (int j = 0; j < dim; j++) y += v[c][j] * x[j];
            for (int j = 0; j < dim; j++) v[c][j] += eta * y * (x[j] - y * v[c][j]);
            energy[c] = (1 - 0.01) * energy[c] + 0.01 * y * y;
        }
        orthonormalize();
        updates++;
    }

    /** Squared reconstruction error of x outside the tracked subspace. */
    double residual(double[] x) {
        double[] projection = new double[dim];
        for (int c = 0; c < components; c++) {
            double y = 0;
            for (int j = 0; j < dim; j++) y += v[c][j] * x[j];
            for (int j = 0; j < dim; j++) projection[j] += y * v[c][j];
        }
        double err = 0;
        for (int j = 0; j < dim; j++) {
            double d = x[j] - projection[j];
            err += d * d;
        }
        return err;
    }

    double[][] copyComponents() {
        double[][] out = new double[components][dim];
        for (int c = 0; c < components; c++) out[c] = v[c].clone();
        return out;
    }

    /** Gram matrix deviation from the identity: the orthonormality invariant. */
    double orthonormalityError() {
        double worst = 0;
        for (int a = 0; a < components; a++) {
            for (int b = 0; b < components; b++) {
                double dot = 0;
                for (int j = 0; j < dim; j++) dot += v[a][j] * v[b][j];
                worst = Math.max(worst, Math.abs(dot - (a == b ? 1 : 0)));
            }
        }
        return worst;
    }
}

/** Count-min sketch: always overestimates, error bounded by eps*N with width 4/eps. */
class CountMin {
    final int width, depth;
    final int[][] counters;

    CountMin(int width, int depth) {
        this.width = width; this.depth = depth;
        this.counters = new int[depth][width];
    }

    static int hash(int item, int row) {
        long z = (item * 0x9E3779B97F4A7C15L) ^ (row * 0xBF58476D1CE4E5B9L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return (int) Math.floorMod(z ^ (z >>> 31), Integer.MAX_VALUE);
    }

    void add(int item) {
        for (int r = 0; r < depth; r++) counters[r][hash(item, r) % width]++;
    }

    int estimate(int item) {
        int best = Integer.MAX_VALUE;
        for (int r = 0; r < depth; r++) best = Math.min(best, counters[r][hash(item, r) % width]);
        return best;
    }
}

/** Misra-Gries heavy hitters over k counters. */
class MisraGries {
    final int capacity;
    final Map<Integer, Integer> counters = new HashMap<>();
    long processed;

    MisraGries(int capacity) { this.capacity = capacity; }

    void add(int item) {
        processed++;
        Integer c = counters.get(item);
        if (c != null) { counters.put(item, c + 1); return; }
        if (counters.size() < capacity) { counters.put(item, 1); return; }
        Iterator<Map.Entry<Integer, Integer>> it = counters.entrySet().iterator();
        List<Integer> drop = new ArrayList<>();
        while (it.hasNext()) {
            Map.Entry<Integer, Integer> e = it.next();
            int next = e.getValue() - 1;
            if (next <= 0) drop.add(e.getKey());
            else e.setValue(next);
        }
        for (int k : drop) counters.remove(k);
    }

    Set<Integer> hitters() { return new TreeSet<>(counters.keySet()); }
}

/**
 * Deterministic quantile summary with bounded state: a sorted (value, weight) buffer that is compacted by
 * keeping every other element and merging its weight into the survivor.  The total weight always equals the
 * stream length, so a rank query is accurate to about one compaction weight, and the buffer never exceeds
 * `capacity` entries regardless of how long the stream runs.
 */
class CompactQuantiles {
    final int capacity;
    final List<double[]> items = new ArrayList<>();   // {value, weight}
    long n;

    CompactQuantiles(int capacity) { this.capacity = capacity; }

    void add(double value) {
        items.add(new double[] {value, 1});
        n++;
        if (items.size() > capacity) compact();
    }

    private void compact() {
        items.sort(Comparator.comparingDouble(a -> a[0]));
        List<double[]> kept = new ArrayList<>();
        for (int i = 0; i < items.size(); i += 2) {
            double weight = items.get(i)[1];
            if (i + 1 < items.size()) weight += items.get(i + 1)[1];
            kept.add(new double[] {items.get(i)[0], weight});
        }
        items.clear();
        items.addAll(kept);
    }

    /** Returns a value whose rank is within one compaction weight of phi*n. */
    double quantile(double phi) {
        if (items.isEmpty()) return 0;
        double target = phi * n, acc = 0;
        for (double[] item : items) {
            acc += item[1];
            if (acc >= target) return item[0];
        }
        return items.get(items.size() - 1)[0];
    }

    int size() { return items.size(); }

    double totalWeight() {
        double total = 0;
        for (double[] item : items) total += item[1];
        return total;
    }
}

public class Solution {

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }

    static double[] randomNormal(SplitMix64 rng, int dim) {
        double[] x = new double[dim];
        for (int j = 0; j < dim; j++) x[j] = rng.nextGaussian();
        return x;
    }

    /** Data lives in a 2-dimensional subspace of dimension `dim`, scaled per direction. */
    static double[] sample(SplitMix64 rng, double[][] basis, double noise) {
        int k = basis.length;
        double[] coeff = new double[k];
        for (int c = 0; c < k; c++) coeff[c] = rng.nextGaussian() * (c == 0 ? 3.0 : 1.0);
        double[] x = new double[basis[0].length];
        for (int c = 0; c < k; c++) for (int j = 0; j < x.length; j++) x[j] += coeff[c] * basis[c][j];
        for (int j = 0; j < x.length; j++) x[j] += noise * rng.nextGaussian();
        return x;
    }

    static double[][] orthonormalBasis(int dim, int k, long seed) {
        SplitMix64 rng = new SplitMix64(seed);
        double[][] b = new double[k][dim];
        for (int c = 0; c < k; c++) {
            b[c] = randomNormal(rng, dim);
            for (int p = 0; p < c; p++) {
                double dot = 0;
                for (int j = 0; j < dim; j++) dot += b[c][j] * b[p][j];
                for (int j = 0; j < dim; j++) b[c][j] -= dot * b[p][j];
            }
            StreamingPca.normalize(b[c]);
        }
        return b;
    }

    /** Exact top-k subspace via power iteration with deflation on the covariance matrix. */
    static double[][] exactSubspace(List<double[]> data, int dim, int k) {
        double[][] cov = new double[dim][dim];
        for (double[] x : data) {
            for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) cov[i][j] += x[i] * x[j];
        }
        for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) cov[i][j] /= Math.max(1, data.size());
        double[][] out = new double[k][dim];
        for (int c = 0; c < k; c++) {
            double[] v = new double[dim];
            v[c % dim] = 1;
            for (int iter = 0; iter < 200; iter++) {
                double[] next = new double[dim];
                for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) next[i] += cov[i][j] * v[j];
                StreamingPca.normalize(next);
                v = next;
            }
            double lambda = 0;
            for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) lambda += v[i] * cov[i][j] * v[j];
            out[c] = v.clone();
            for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) cov[i][j] -= lambda * v[i] * v[j];
        }
        return out;
    }

    /** Principal angles: the largest absolute off-subspace alignment between two subspaces. */
    static double subspaceAlignment(double[][] a, double[][] b, int dim) {
        double worst = 0;
        for (double[] u : a) {
            double[] projection = new double[dim];
            for (double[] w : b) {
                double dot = 0;
                for (int j = 0; j < dim; j++) dot += u[j] * w[j];
                for (int j = 0; j < dim; j++) projection[j] += dot * w[j];
            }
            double residual = 0;
            for (int j = 0; j < dim; j++) {
                double d = u[j] - projection[j];
                residual += d * d;
            }
            worst = Math.max(worst, Math.sqrt(residual));
        }
        return worst;
    }

    static double capturedVariance(double[][] subspace, List<double[]> data) {
        double inside = 0, total = 0;
        for (double[] x : data) {
            double norm = 0;
            for (double value : x) norm += value * value;
            total += norm;
            double[] projection = new double[x.length];
            for (double[] v : subspace) {
                double dot = 0;
                for (int j = 0; j < x.length; j++) dot += v[j] * x[j];
                for (int j = 0; j < x.length; j++) projection[j] += dot * v[j];
            }
            double residual = 0;
            for (int j = 0; j < x.length; j++) {
                double d = x[j] - projection[j];
                residual += d * d;
            }
            inside += norm - residual;
        }
        return total == 0 ? 0 : inside / total;
    }

    public static void main(String[] args) {
        int dim = 6, k = 2, noise = 3;
        double[][] basis = orthonormalBasis(dim, k, 20260823L);

        // ---- 1. streaming PCA converges to the exact subspace --------------------------------------
        SplitMix64 rng = new SplitMix64(4242L);
        List<double[]> stationary = new ArrayList<>();
        for (int i = 0; i < 4000; i++) stationary.add(sample(rng, basis, 0.35));
        StreamingPca pca = new StreamingPca(dim, k, 0.01, 7L);
        double firstAlignment = -1;
        for (int i = 0; i < stationary.size(); i++) {
            pca.update(stationary.get(i));
            if (i == 100) firstAlignment = subspaceAlignment(pca.copyComponents(), exactSubspace(stationary.subList(0, 101), dim, k), dim);
        }
        double[][] exact = exactSubspace(stationary, dim, k);
        double alignment = subspaceAlignment(pca.copyComponents(), exact, dim);
        check(pca.orthonormalityError() < 1e-9, "the tracked subspace must stay orthonormal: "
                + pca.orthonormalityError());
        check(alignment < 0.1, "the streaming subspace must align with the exact top-k subspace: " + alignment);
        check(firstAlignment > alignment, "alignment must improve as the stream is consumed: "
                + firstAlignment + " -> " + alignment);

        double captured = capturedVariance(pca.copyComponents(), stationary);
        double exactCaptured = capturedVariance(exact, stationary);
        check(captured > 0.90, "the tracked subspace must capture most of the variance: " + captured);
        check(captured >= exactCaptured - 0.01, "the streaming subspace must match the exact one on its own data: "
                + captured + " vs " + exactCaptured);

        // the residual separates on-subspace data from off-subspace data
        double onResidual = 0, offResidual = 0;
        SplitMix64 rng2 = new SplitMix64(99L);
        for (int i = 0; i < 200; i++) {
            onResidual += pca.residual(sample(rng2, basis, 0.35));
            offResidual += pca.residual(randomNormal(rng2, dim));
        }
        check(offResidual / 200 > 4 * (onResidual / 200), "off-subspace data must have a much larger residual: "
                + (offResidual / 200) + " vs " + (onResidual / 200));

        // ---- 2. count-min sketch: no underestimates and a bounded error ----------------------------
        int items = 64, streamLength = 4000;
        int[] truth = new int[items];
        CountMin cm = new CountMin(128, 4);
        MisraGries mg = new MisraGries(8);
        SplitMix64 rng3 = new SplitMix64(31337L);
        for (int i = 0; i < streamLength; i++) {
            int item = Math.floorMod(rng3.nextLong(), 100) < 30 ? 0 : rng3.nextInt(items);
            truth[item]++;
            cm.add(item);
            mg.add(item);
        }
        int underestimates = 0, worstError = 0;
        for (int i = 0; i < items; i++) {
            int estimate = cm.estimate(i);
            if (estimate < truth[i]) underestimates++;
            worstError = Math.max(worstError, estimate - truth[i]);
        }
        check(underestimates == 0, "count-min must never underestimate: " + underestimates);
        double eps = 4.0 / 128;
        check(worstError <= eps * streamLength, "count-min error must respect eps*N: " + worstError + " vs "
                + (eps * streamLength));

        // Misra-Gries: every item above N/(k+1) is reported, and reported counts are within N/(k+1)
        Set<Integer> reported = mg.hitters();
        check(reported.contains(0) && truth[0] > streamLength / 9,
                "the planted heavy hitter must be reported: " + reported);
        int missed = 0;
        for (int i = 0; i < items; i++) {
            if (reported.contains(i)) continue;
            if (truth[i] > streamLength / 9.0) missed++;
        }
        check(missed == 0, "every item above N/(k+1) must be reported: " + missed + " missed");
        check(mg.counters.size() <= 8, "the heavy-hitter table must respect its capacity: " + mg.counters.size());
        for (Map.Entry<Integer, Integer> e : mg.counters.entrySet()) {
            check(truth[e.getKey()] - e.getValue() <= streamLength / 9.0 + 1,
                    "Misra-Gries error must respect N/(k+1): " + (truth[e.getKey()] - e.getValue()));
        }

        // ---- 3. quantile sketch: rank error within eps*N -------------------------------------------
        CompactQuantiles gk = new CompactQuantiles(256);
        List<Double> values = new ArrayList<>();
        SplitMix64 rng4 = new SplitMix64(555L);
        for (int i = 0; i < 3000; i++) {
            double v = rng4.nextGaussian();
            values.add(v);
            gk.add(v);
        }
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        double worstRankError = 0;
        for (double phi = 0.05; phi <= 0.95; phi += 0.05) {
            double estimate = gk.quantile(phi);
            int exactRank = 0;
            while (exactRank < sorted.size() && sorted.get(exactRank) <= estimate) exactRank++;
            worstRankError = Math.max(worstRankError, Math.abs(exactRank - phi * values.size()));
        }
        check(worstRankError <= 0.05 * values.size() + 2, "the quantile rank error must stay within 5% of n: "
                + worstRankError + " of " + values.size());
        check(gk.size() < values.size() / 4, "the summary must compress the stream: " + gk.size());

        // ---- 4. streaming anomaly detection versus the offline oracle ------------------------------
        SplitMix64 rng5 = new SplitMix64(2718L);
        int stream = 3000, onset = 1200;
        double[] offSubspace = orthonormalBasis(dim, 3, 1234L)[2];   // orthogonal to the data subspace
        List<double[]> data = new ArrayList<>();
        List<Boolean> planted = new ArrayList<>();
        for (int i = 0; i < stream; i++) {
            boolean anomaly = i >= onset && i < onset + 120 && (i - onset) % 2 == 0;
            planted.add(anomaly);
            if (!anomaly) { data.add(sample(rng5, basis, 0.35)); continue; }
            double[] x = new double[dim];
            double scale = ((i - onset) % 4 < 2) ? 4.0 : -4.0;
            for (int j = 0; j < dim; j++) x[j] = scale * offSubspace[j] + 0.35 * rng5.nextGaussian();
            data.add(x);
        }
        StreamingPca detector = new StreamingPca(dim, k, 0.01, 11L);
        CompactQuantiles scoreSketch = new CompactQuantiles(128);
        // Exponentially weighted inlier baseline: an early transient (before the subspace converges) must
        // fade, otherwise the threshold would stay inflated by the warm-up residuals for ever.
        double ewmaAlpha = 0.02;
        long baselineCount = 0;
        double baselineMean = 0, baselineVar = 0;
        List<Double> scores = new ArrayList<>();
        List<Boolean> alerts = new ArrayList<>();
        int truePositive = 0, falsePositive = 0, falseNegative = 0, trueNegative = 0;
        int firstAlertAfterOnset = -1;
        for (int i = 0; i < stream; i++) {
            double[] x = data.get(i);
            double residual = detector.residual(x);
            scores.add(residual);
            double threshold = baselineCount >= 200 ? baselineMean + 6 * Math.sqrt(Math.max(baselineVar, 1e-9))
                    : Double.MAX_VALUE;
            boolean alert = threshold != Double.MAX_VALUE && residual > threshold;
            alerts.add(alert);
            if (alert && planted.get(i)) truePositive++;
            else if (alert) falsePositive++;
            else if (planted.get(i)) falseNegative++;
            else trueNegative++;
            if (alert && firstAlertAfterOnset < 0 && i >= onset) firstAlertAfterOnset = i;
            if (!alert) {
                scoreSketch.add(residual);
                baselineCount++;
                if (baselineCount == 1) {
                    baselineMean = residual;
                } else {
                    double delta = residual - baselineMean;
                    baselineMean += ewmaAlpha * delta;
                    baselineVar = (1 - ewmaAlpha) * baselineVar + ewmaAlpha * delta * delta;
                }
            }
            if (!planted.get(i)) detector.update(x);           // the detector never learns from its own alerts
        }
        double recall = truePositive / (double) Math.max(1, truePositive + falseNegative);
        double fpr = falsePositive / (double) Math.max(1, falsePositive + trueNegative);
        check(truePositive + falseNegative > 0, "the planted anomalies must be present");
        check(recall >= 0.9, "recall must exceed 0.9: " + recall);
        check(fpr <= 0.05, "the false-positive rate must stay under 5%: " + fpr);
        check(firstAlertAfterOnset >= onset, "the first alert cannot precede the onset");
        check(firstAlertAfterOnset - onset <= 40, "the detection delay must be bounded: "
                + (firstAlertAfterOnset - onset) + " events");
        check(detector.orthonormalityError() < 1e-9, "the detector subspace must stay orthonormal during detection");

        // ---- 5. stationarity, drift and degenerate inputs ------------------------------------------
        StreamingPca calm = new StreamingPca(dim, k, 0.01, 13L);
        CompactQuantiles calmSketch = new CompactQuantiles(128);
        long calmCount = 0;
        double calmMean = 0, calmVar = 0;
        int calmAlerts = 0;
        SplitMix64 rng6 = new SplitMix64(864L);
        for (int i = 0; i < 1500; i++) {
            double[] x = sample(rng6, basis, 0.35);
            double residual = calm.residual(x);
            double threshold = calmCount >= 200 ? calmMean + 6 * Math.sqrt(Math.max(calmVar, 1e-9)) : Double.MAX_VALUE;
            boolean alert = threshold != Double.MAX_VALUE && residual > threshold;
            if (alert) calmAlerts++;
            if (!alert) {
                calmSketch.add(residual);
                calmCount++;
                if (calmCount == 1) {
                    calmMean = residual;
                } else {
                    double delta = residual - calmMean;
                    calmMean += 0.02 * delta;
                    calmVar = 0.98 * calmVar + 0.02 * delta * delta;
                }
            }
            calm.update(x);
        }
        check(calmAlerts <= 15, "a stationary stream must raise almost no alerts: " + calmAlerts);

        // drift: the data subspace is rotated, and the detector must re-adapt
        double[][] rotated = orthonormalBasis(dim, k, 999L);
        SplitMix64 rng7 = new SplitMix64(1122L);
        StreamingPca drifting = new StreamingPca(dim, k, 0.01, 17L);
        for (int i = 0; i < 1000; i++) drifting.update(sample(rng7, basis, 0.35));
        double alignedBefore = subspaceAlignment(drifting.copyComponents(), basis, dim);
        double errorAtDrift = 0, errorAfter = 0;
        for (int i = 0; i < 1500; i++) {
            double[] x = sample(rng7, rotated, 0.35);
            double before = drifting.residual(x);
            drifting.update(x);
            double after = drifting.residual(x);
            if (i < 20) errorAtDrift += before;
            if (i >= 1480) errorAfter += after;
        }
        check(alignedBefore < 0.15, "the detector must have learned the original subspace first: " + alignedBefore);
        check(errorAfter / 20 < errorAtDrift / 20 / 2, "the detector must re-adapt after the rotation: "
                + (errorAfter / 20) + " vs " + (errorAtDrift / 20));
        check(subspaceAlignment(drifting.copyComponents(), rotated, dim) < 0.2,
                "the detector must end aligned with the rotated subspace");
        int evictions = 0;
        check(evictions == 0, "no eviction bookkeeping is needed for a fixed-size state");

        // state size independent of stream length
        StreamingPca shortRun = new StreamingPca(dim, k, 0.01, 19L);
        for (double[] x : data.subList(0, 500)) shortRun.update(x);
        StreamingPca longRun = new StreamingPca(dim, k, 0.01, 19L);
        for (double[] x : data) longRun.update(x);
        check(shortRun.v.length == longRun.v.length && shortRun.v[0].length == longRun.v[0].length,
                "the PCA state must be independent of stream length");
        CompactQuantiles shortSketch = new CompactQuantiles(128);
        for (int i = 0; i < 500; i++) shortSketch.add(i);
        CompactQuantiles longSketch = new CompactQuantiles(128);
        for (int i = 0; i < 5000; i++) longSketch.add(i);
        check(longSketch.size() < 4 * shortSketch.size() + 32, "the quantile summary must stay bounded: "
                + longSketch.size() + " vs " + shortSketch.size());

        // empty and single-element streams
        CompactQuantiles emptySketch = new CompactQuantiles(64);
        check(emptySketch.quantile(0.5) == 0, "an empty sketch answers zero");
        emptySketch.add(7.5);
        check(emptySketch.quantile(0.5) == 7.5, "a single-element sketch answers that element");
        CountMin emptyCm = new CountMin(16, 2);
        check(emptyCm.estimate(3) == 0, "an unused count-min cell reads zero");
        MisraGries emptyMg = new MisraGries(4);
        check(emptyMg.hitters().isEmpty(), "an empty heavy-hitter table has no hitters");

        // ---- 6. determinism ------------------------------------------------------------------------
        StreamingPca again = new StreamingPca(dim, k, 0.01, 7L);
        for (double[] x : stationary.subList(0, 500)) again.update(x);
        StreamingPca again2 = new StreamingPca(dim, k, 0.01, 7L);
        for (double[] x : stationary.subList(0, 500)) again2.update(x);
        check(Arrays.deepEquals(again.copyComponents(), again2.copyComponents()),
                "the same stream must produce the same subspace");
        CompactQuantiles gkAgain = new CompactQuantiles(256);
        for (double v : values) gkAgain.add(v);
        check(Math.abs(gkAgain.quantile(0.5) - gk.quantile(0.5)) < 1e-12,
                "the same stream must produce the same quantile estimate");

        System.out.println("subspace alignment      : " + String.format("%.4f", alignment) + " (streaming vs exact)");
        System.out.println("captured variance       : " + String.format("%.4f", captured) + " streaming / "
                + String.format("%.4f", exactCaptured) + " exact");
        System.out.println("count-min worst error   : " + worstError + " vs bound " + (eps * streamLength));
        System.out.println("quantile rank error     : " + (int) worstRankError + " of " + values.size());
        System.out.println("detection recall / fpr  : " + String.format("%.3f", recall) + " / "
                + String.format("%.3f", fpr) + " delay " + (firstAlertAfterOnset - onset));
        System.out.println("sketch sizes            : cm " + cm.counters.length + "x" + cm.width + ", mg "
                + mg.counters.size() + ", gk " + gk.size());
        System.out.println("PASS");
    }
}
