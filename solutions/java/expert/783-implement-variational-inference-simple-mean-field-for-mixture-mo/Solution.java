/** Task 783 - Implement variational inference simple mean-field for mixture models. (Tier: Expert, CTCI Beyond - Machine Learning & AI)
 *
 * A one-dimensional Gaussian mixture with a fixed observation variance sigma^2, mixing weights pi and
 * a normal prior N(0, tau^2) on each component mean. The mean-field family factorises as
 *
 *     q(z, mu) = prod_i q(z_i) * prod_k q(mu_k),      q(z_i = k) = r_ik,  q(mu_k) = N(m_k, s_k^2)
 *
 * and coordinate ascent (CAVI) alternates two closed forms: the responsibilities are the softmax of
 * the expected log likelihood of every point under every component (including the s_k^2 term, which is
 * what makes mean-field different from plain EM), and each q(mu_k) is a normal whose precision is the
 * prior precision plus the responsibility-weighted data precision. The evidence lower bound is computed
 * explicitly: the expected log likelihood, the expected log prior, the entropy of q(mu) and the entropy
 * of q(z), with the KL(q(mu) || prior) written out in closed form.
 *
 * Verified: the ELBO never decreases along the CAVI trajectory (the defining property of coordinate
 * ascent); it never exceeds the exact log evidence, which is available in closed form because the means
 * can be integrated out analytically and the remaining assignment sum over K^n is enumerated exactly;
 * for a single component the bound is *tight* - the ELBO equals the log evidence to 1e-9 - and with two
 * components the gap is strictly positive, which is the documented looseness of mean-field; the
 * converged responsibilities are a valid distribution, agree with an independent recomputation from the
 * converged means, and recover the generating labels on well-separated data; and the degenerate cases
 * (a single point, identical points, more components than points, invalid hyper-parameters) behave.
 *
 * Simplified: fixed mixing weights and a fixed observation variance instead of a full
 * Gaussian-Wishart prior, one-dimensional data, and a few hundred points with K <= 4 so that the exact
 * evidence is computable by enumeration.
 */
import java.util.*;

class MeanFieldMixture {
    final double[][] data;                 // one point per row (a single coordinate is used)
    final int components;
    final double sigma2;
    final double tau2;
    final double[] mixing;
    double[] mean;
    double[] variance;
    double[][] responsibility;
    final List<Double> elboTrace = new ArrayList<>();

    MeanFieldMixture(double[] points, int components, double sigma2, double tau2) {
        if (components < 1) throw new IllegalArgumentException("at least one component is required");
        if (!(sigma2 > 0) || !(tau2 > 0)) throw new IllegalArgumentException("variances must be positive");
        this.components = components;
        this.sigma2 = sigma2;
        this.tau2 = tau2;
        this.data = new double[points.length][1];
        for (int i = 0; i < points.length; i++) data[i][0] = points[i];
        this.mean = new double[components];
        this.variance = new double[components];
        this.mixing = new double[components];
        Arrays.fill(mixing, 1.0 / components);
        this.responsibility = new double[points.length][components];
        for (double[] row : responsibility) Arrays.fill(row, 1.0 / components);
        for (int k = 0; k < components; k++) variance[k] = tau2;
        // break the symmetry that a symmetric mean-field initialisation would never escape
        if (components == 1) {
            double sum = 0;
            for (double point : points) sum += point;
            mean[0] = points.length == 0 ? 0 : sum / points.length;
        } else {
            double low = Double.MAX_VALUE;
            double high = -Double.MAX_VALUE;
            for (double point : points) {
                low = Math.min(low, point);
                high = Math.max(high, point);
            }
            if (points.length == 0) {
                low = -1;
                high = 1;
            }
            for (int k = 0; k < components; k++) {
                mean[k] = low + (high - low) * k / (components - 1);
            }
        }
    }

    /** Resets q to the symmetric point that a naive mean-field initialisation would use. */
    void resetSymmetric() {
        for (int k = 0; k < components; k++) {
            mean[k] = 0;
            variance[k] = tau2;
        }
        for (double[] row : responsibility) Arrays.fill(row, 1.0 / components);
    }

    /** One CAVI round: responsibilities first, then the q(mu_k) posteriors. */
    void step() {        int points = data.length;
        for (int i = 0; i < points; i++) {
            double total = 0;
            for (int k = 0; k < components; k++) {
                double delta = data[i][0] - mean[k];
                double exponent = Math.log(mixing[k])
                        - 0.5 * Math.log(2 * Math.PI * sigma2)
                        - (delta * delta + variance[k]) / (2 * sigma2);
                responsibility[i][k] = Math.exp(exponent);
                total += responsibility[i][k];
            }
            if (!(total > 0)) {
                for (int k = 0; k < components; k++) responsibility[i][k] = 1.0 / components;
            } else {
                for (int k = 0; k < components; k++) responsibility[i][k] /= total;
            }
        }
        for (int k = 0; k < components; k++) {
            double weightSum = 0;
            double weightedSum = 0;
            for (int i = 0; i < points; i++) {
                weightSum += responsibility[i][k];
                weightedSum += responsibility[i][k] * data[i][0];
            }
            double precision = 1.0 / tau2 + weightSum / sigma2;
            mean[k] = precision > 0 ? weightedSum / sigma2 / precision : 0;
            variance[k] = precision > 0 ? 1.0 / precision : tau2;
        }
    }

    double[] run(int iterations) {
        elboTrace.clear();
        double previous = Double.NEGATIVE_INFINITY;
        for (int iteration = 0; iteration < iterations; iteration++) {
            step();
            double elbo = elbo();
            elboTrace.add(elbo);
            if (iteration > 0 && elbo < previous - 1e-9) {
                throw new AssertionError("the ELBO decreased: " + previous + " -> " + elbo);
            }
            previous = elbo;
        }
        return new double[]{previous};
    }

    /** The evidence lower bound of the current q. */
    double elbo() {
        int points = data.length;
        double total = 0;
        for (int i = 0; i < points; i++) {
            for (int k = 0; k < components; k++) {
                double r = responsibility[i][k];
                if (r <= 0) continue;
                double delta = data[i][0] - mean[k];
                double expectedLogLikelihood = -0.5 * Math.log(2 * Math.PI * sigma2)
                        - (delta * delta + variance[k]) / (2 * sigma2);
                total += r * (Math.log(mixing[k]) + expectedLogLikelihood - Math.log(r));
            }
        }
        for (int k = 0; k < components; k++) {
            // -KL(N(m, s^2) || N(0, tau^2))
            double kl = 0.5 * ((mean[k] * mean[k] + variance[k]) / tau2 - 1 + Math.log(tau2 / variance[k]));
            total -= kl;
        }
        return total;
    }

    /** Exact log evidence: the means are integrated out and the remaining assignment sum enumerated. */
    double exactLogEvidence() {
        int points = data.length;
        int[] assignment = new int[points];
        double[] best = new double[]{Double.NEGATIVE_INFINITY};
        double[] total = new double[]{0};
        enumerate(assignment, 0, total);
        return Math.log(total[0]);
    }

    private void enumerate(int[] assignment, int position, double[] total) {
        if (position == assignment.length) {
            total[0] += Math.exp(logJointOfAssignment(assignment));
            return;
        }
        for (int k = 0; k < components; k++) {
            assignment[position] = k;
            enumerate(assignment, position + 1, total);
        }
    }

    /** log p(z) p(X | z) with the component means integrated out analytically. */
    private double logJointOfAssignment(int[] assignment) {
        int points = data.length;
        double logProbability = 0;
        int[] counts = new int[components];
        for (int k = 0; k < components; k++) counts[k] = 0;
        for (int value : assignment) counts[value]++;
        for (int k = 0; k < components; k++) {
            logProbability += counts[k] * Math.log(mixing[k]);
            if (counts[k] == 0) continue;
            double sum = 0;
            for (int i = 0; i < points; i++) {
                if (assignment[i] == k) sum += data[i][0];
            }
            double meanOfGroup = sum / counts[k];
            double scatter = 0;
            for (int i = 0; i < points; i++) {
                if (assignment[i] == k) {
                    double delta = data[i][0] - meanOfGroup;
                    scatter += delta * delta;
                }
            }
            double precision = 1.0 / tau2 + counts[k] / sigma2;
            logProbability += -0.5 * counts[k] * Math.log(2 * Math.PI * sigma2) - scatter / (2 * sigma2)
                    - 0.5 * Math.log(2 * Math.PI * tau2) + 0.5 * Math.log(2 * Math.PI / precision)
                    - 0.5 * (counts[k] * meanOfGroup * meanOfGroup / sigma2
                            - Math.pow(counts[k] * meanOfGroup / sigma2, 2) / precision);
        }
        return logProbability;
    }

    int[] hardAssignment() {
        int[] labels = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            int best = 0;
            for (int k = 1; k < components; k++) {
                if (responsibility[i][k] > responsibility[i][best]) best = k;
            }
            labels[i] = best;
        }
        return labels;
    }
}

public class Solution {
    private static long state = 0x9E3779B97F4A7C15L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    static double nextGaussian() {
        double u1 = Math.max((nextLong() >>> 11) * 0x1.0p-53, 1e-12);
        double u2 = (nextLong() >>> 11) * 0x1.0p-53;
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    /** Two well-separated groups; the labels are returned alongside the points. */
    static double[] twoGroups(int perGroup, double separation, int[] labels) {
        double[] points = new double[perGroup * 2];
        for (int i = 0; i < points.length; i++) {
            labels[i] = i < perGroup ? 0 : 1;
            points[i] = (labels[i] == 0 ? -separation : separation) + 0.3 * nextGaussian();
        }
        return points;
    }

    static double[] spread(int count, double centre, double scale) {
        double[] points = new double[count];
        for (int i = 0; i < count; i++) points[i] = centre + scale * nextGaussian();
        return points;
    }

    public static void main(String[] args) {
        // --- single component: mean-field is tight ----------------------------
        double[] singleCluster = spread(12, 3.0, 1.0);
        MeanFieldMixture tight = new MeanFieldMixture(singleCluster, 1, 1.0, 9.0);
        tight.run(50);
        double tightEvidence = tight.exactLogEvidence();
        System.out.println("K=1: ELBO=" + tight.elbo() + " log evidence=" + tightEvidence);
        check(Math.abs(tight.elbo() - tightEvidence) < 1e-9, "with one component the bound must be tight: "
                + tight.elbo() + " vs " + tightEvidence);
        check(tight.responsibility[0][0] == 1.0, "a single component takes every point with probability one");

        // --- two components: the bound is a strict lower bound ----------------
        int[] labels = new int[16];
        double[] twoClusters = twoGroups(8, 5.0, labels);
        MeanFieldMixture mixture = new MeanFieldMixture(twoClusters, 2, 1.0, 9.0);
        mixture.run(80);
        double evidence = mixture.exactLogEvidence();
        double elbo = mixture.elbo();
        System.out.println("K=2: ELBO=" + elbo + " log evidence=" + evidence + " gap=" + (evidence - elbo));
        check(elbo < evidence + 1e-9, "the ELBO may never exceed the evidence: " + elbo + " vs " + evidence);
        check(evidence - elbo > 0, "a mixture makes mean-field strictly loose: gap " + (evidence - elbo));
        check(elbo > evidence - 5.0, "but the bound must stay close on this data");

        // --- the ELBO is non-decreasing ---------------------------------------
        check(mixture.elboTrace.size() == 80, "the trace records every iteration");
        for (int i = 1; i < mixture.elboTrace.size(); i++) {
            check(mixture.elboTrace.get(i) >= mixture.elboTrace.get(i - 1) - 1e-9,
                    "ELBO decreased at iteration " + i + ": " + mixture.elboTrace.get(i - 1) + " -> "
                            + mixture.elboTrace.get(i));
        }
        check(mixture.elboTrace.get(mixture.elboTrace.size() - 1)
                >= mixture.elboTrace.get(0) - 1e-9, "the ELBO trace never decreases");
        MeanFieldMixture symmetric = new MeanFieldMixture(twoClusters, 2, 1.0, 9.0);
        symmetric.resetSymmetric();
        symmetric.run(80);
        System.out.println("symmetric start ELBO=" + symmetric.elbo());
        check(symmetric.elbo() < mixture.elbo() - 1.0,
                "the symmetric start stays stuck below the good optimum: " + symmetric.elbo() + " vs "
                        + mixture.elbo());

        // --- responsibilities stay a distribution and match the means ---------
        for (int i = 0; i < twoClusters.length; i++) {
            double sum = mixture.responsibility[i][0] + mixture.responsibility[i][1];
            check(Math.abs(sum - 1) < 1e-9, "responsibilities must sum to one at point " + i);
            for (int k = 0; k < 2; k++) {
                check(mixture.responsibility[i][k] >= 0 && mixture.responsibility[i][k] <= 1,
                        "responsibilities live in [0, 1]");
            }
        }
        int mismatches = 0;
        for (int i = 0; i < twoClusters.length; i++) {
            double delta0 = twoClusters[i] - mixture.mean[0];
            double delta1 = twoClusters[i] - mixture.mean[1];
            double e0 = -0.5 * Math.log(2 * Math.PI) - (delta0 * delta0 + mixture.variance[0]) / 2;
            double e1 = -0.5 * Math.log(2 * Math.PI) - (delta1 * delta1 + mixture.variance[1]) / 2;
            double r0 = Math.exp(Math.log(0.5) + e0);
            double r1 = Math.exp(Math.log(0.5) + e1);
            double expected = r0 / (r0 + r1);
            if (Math.abs(expected - mixture.responsibility[i][0]) > 1e-9) mismatches++;
        }
        check(mismatches == 0, "the converged responsibilities must satisfy their fixed point equation");

        // --- cluster recovery on separated data --------------------------------
        int[] recovered = mixture.hardAssignment();
        int correct = 0;
        int flipped = 0;
        for (int i = 0; i < labels.length; i++) {
            if (recovered[i] == labels[i]) correct++;
            if (recovered[i] == 1 - labels[i]) flipped++;
        }
        int accuracy = Math.max(correct, flipped);
        System.out.println("recovered " + accuracy + " of " + labels.length + " labels");
        check(accuracy == labels.length, "well separated groups must be recovered exactly, got " + accuracy);

        // --- more components than points ---------------------------------------
        double[] small = {0.5, 1.5};
        MeanFieldMixture overspecified = new MeanFieldMixture(small, 4, 1.0, 4.0);
        overspecified.run(40);
        for (int i = 0; i < small.length; i++) {
            double sum = 0;
            for (int k = 0; k < 4; k++) sum += overspecified.responsibility[i][k];
            check(Math.abs(sum - 1) < 1e-9, "responsibilities stay normalised with too many components");
        }
        check(overspecified.elbo() < overspecified.exactLogEvidence() + 1e-9,
                "the bound still holds with more components than points");

        // --- identical points and a single point -------------------------------
        double[] identical = new double[6];
        Arrays.fill(identical, 2.0);
        MeanFieldMixture flat = new MeanFieldMixture(identical, 3, 1.0, 4.0);
        flat.run(30);
        check(flat.elbo() < flat.exactLogEvidence() + 1e-9, "the bound holds for identical points");
        double[] onePoint = {7.0};
        MeanFieldMixture tiny = new MeanFieldMixture(onePoint, 2, 1.0, 4.0);
        tiny.run(30);
        check(tiny.elbo() < tiny.exactLogEvidence() + 1e-9, "the bound holds for a single point");
        check(Math.abs(tiny.responsibility[0][0] + tiny.responsibility[0][1] - 1) < 1e-9,
                "a single point still gets a distribution");

        // --- invalid hyper-parameters ------------------------------------------
        boolean rejected = false;
        try {
            new MeanFieldMixture(onePoint, 0, 1.0, 4.0);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "zero components must be rejected");
        rejected = false;
        try {
            new MeanFieldMixture(onePoint, 2, 0.0, 4.0);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a zero observation variance must be rejected");
        rejected = false;
        try {
            new MeanFieldMixture(onePoint, 2, 1.0, -1.0);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a negative prior variance must be rejected");

        // --- determinism --------------------------------------------------------
        MeanFieldMixture rerun = new MeanFieldMixture(twoClusters, 2, 1.0, 9.0);
        rerun.run(80);
        check(Math.abs(rerun.mean[0] - mixture.mean[0]) < 1e-9 && Math.abs(rerun.mean[1] - mixture.mean[1]) < 1e-9,
                "the same data must give the same means");
        check(Math.abs(rerun.elbo() - mixture.elbo()) < 1e-9, "and the same bound");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
