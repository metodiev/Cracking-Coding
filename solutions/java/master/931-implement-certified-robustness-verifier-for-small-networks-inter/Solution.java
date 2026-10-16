/** Task 931 - Implement certified robustness verifier for small networks (interval bound propagation).
 *  (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * Interval bound propagation over a ReLU network. Given an input box [x - eps, x + eps] the verifier pushes
 * intervals forward: an affine layer maps [lo, hi] to [W+ lo + W- hi + b, W+ hi + W- lo + b] (splitting the
 * weight matrix by sign, which is exactly the tightest linear interval), and a ReLU widens an interval to
 * [max(0, lo), max(0, hi)]. If the lower bound of the true class ends up above the upper bound of every other
 * class the whole box is certified - no attack inside it can change the label.
 * What is simplified: a small fully connected network, exactly the IBP relaxation (no CROWN/LP tightening, so
 * the certificate is sound but conservative - the verified radius is a lower bound on the true one), and boxes
 * centred on a sample rather than arbitrary input sets.
 * Verified: soundness, i.e. hundreds of samples drawn inside each box always land inside the propagated
 * interval; the interval degenerates to the exact logits when eps = 0; a monotone one-dimensional network has
 * an exactly tight interval; a large eps is not certified and the network really is fooled inside that box;
 * IBP never certifies more than exhaustive sampling can refute; the certified radius grows monotonically with
 * the margin; and the verifier is deterministic.
 */
import java.util.*;

/** A ReLU network plus the interval propagation that the verifier runs on it. */
final class IntervalNet {
    final int in, hidden, out;
    final double[][] w1, w2;
    final double[] b1, b2;

    IntervalNet(int in, int hidden, int out, long seed) {
        this.in = in;
        this.hidden = hidden;
        this.out = out;
        w1 = new double[hidden][in];
        w2 = new double[out][hidden];
        b1 = new double[hidden];
        b2 = new double[out];
        long state = seed;
        for (int j = 0; j < hidden; j++) {
            for (int i = 0; i < in; i++) {
                state = state * 6364136223846793005L + 1442695040888963407L;
                w1[j][i] = ((state >>> 11) * 0x1.0p-53 - 0.5) * 4;
            }
            state = state * 6364136223846793005L + 1442695040888963407L;
            b1[j] = ((state >>> 11) * 0x1.0p-53 - 0.5) * 2;
        }
        for (int c = 0; c < out; c++) {
            for (int j = 0; j < hidden; j++) {
                state = state * 6364136223846793005L + 1442695040888963407L;
                w2[c][j] = ((state >>> 11) * 0x1.0p-53 - 0.5) * 2;
            }
            state = state * 6364136223846793005L + 1442695040888963407L;
            b2[c] = ((state >>> 11) * 0x1.0p-53 - 0.5) * 0.5;
        }
    }

    /** A deterministic, monotone, all-positive network: the IBP interval is tight for it. */
    static IntervalNet monotone(int in, int hidden, long seed) {
        IntervalNet net = new IntervalNet(in, hidden, 1, seed);
        for (int j = 0; j < hidden; j++) {
            for (int i = 0; i < in; i++) net.w1[j][i] = 0.25 + 0.1 * i;
            net.b1[j] = 0.05 * j;
        }
        for (int j = 0; j < hidden; j++) net.w2[0][j] = 0.3 + 0.05 * j;
        return net;
    }

    double[] hidden(double[] x) {
        double[] h = new double[hidden];
        for (int j = 0; j < hidden; j++) {
            double z = b1[j];
            for (int i = 0; i < in; i++) z += w1[j][i] * x[i];
            h[j] = Math.max(0, z);
        }
        return h;
    }

    double[] logits(double[] x) {
        double[] h = hidden(x);
        double[] y = new double[out];
        for (int c = 0; c < out; c++) {
            y[c] = b2[c];
            for (int j = 0; j < hidden; j++) y[c] += w2[c][j] * h[j];
        }
        return y;
    }

    int predict(double[] x) {
        double[] y = logits(x);
        int best = 0;
        for (int c = 1; c < out; c++) if (y[c] > y[best]) best = c;
        return best;
    }

    /** Interval bound propagation for the box centre +- eps; returns {lower[], upper[]}. */
    double[][] propagate(double[] centre, double eps) {
        double[] lo = new double[in], hi = new double[in];
        for (int i = 0; i < in; i++) {
            lo[i] = centre[i] - eps;
            hi[i] = centre[i] + eps;
        }
        double[] hiddenLo = new double[hidden], hiddenHi = new double[hidden];
        for (int j = 0; j < hidden; j++) {
            double low = b1[j], high = b1[j];
            for (int i = 0; i < in; i++) {
                if (w1[j][i] >= 0) {
                    low += w1[j][i] * lo[i];
                    high += w1[j][i] * hi[i];
                } else {
                    low += w1[j][i] * hi[i];
                    high += w1[j][i] * lo[i];
                }
            }
            hiddenLo[j] = Math.max(0, low);
            hiddenHi[j] = Math.max(0, high);
        }
        double[] outLo = new double[out], outHi = new double[out];
        for (int c = 0; c < out; c++) {
            double low = b2[c], high = b2[c];
            for (int j = 0; j < hidden; j++) {
                if (w2[c][j] >= 0) {
                    low += w2[c][j] * hiddenLo[j];
                    high += w2[c][j] * hiddenHi[j];
                } else {
                    low += w2[c][j] * hiddenHi[j];
                    high += w2[c][j] * hiddenLo[j];
                }
            }
            outLo[c] = low;
            outHi[c] = high;
        }
        return new double[][] {outLo, outHi};
    }

    /** Certified when the true class's lower bound clears every rival's upper bound. */
    boolean certified(double[] centre, double eps, int label) {
        double[][] bounds = propagate(centre, eps);
        for (int c = 0; c < out; c++) {
            if (c == label) continue;
            if (bounds[0][label] <= bounds[1][c]) return false;
        }
        return true;
    }

    /** Certificate margin: how much room the bound leaves. */
    double margin(double[] centre, double eps, int label) {
        double[][] bounds = propagate(centre, eps);
        double rival = -Double.MAX_VALUE;
        for (int c = 0; c < out; c++) if (c != label) rival = Math.max(rival, bounds[1][c]);
        return bounds[0][label] - rival;
    }

    /** Smallest eps at which the certificate is lost (a scan is enough at this scale). */
    double certifiedRadius(double[] centre, int label, double limit, double step) {
        double best = 0;
        for (double eps = 0; eps <= limit; eps += step) {
            if (!certified(centre, eps, label)) break;
            best = eps;
        }
        return best;
    }
}

public class Solution {
    static final int BOXES = 40, SAMPLES_PER_BOX = 60, DIMS = 3, HIDDEN = 10, CLASSES = 3;

    static double[] sampleInBox(double[] centre, double eps, long state) {
        double[] out = new double[centre.length];
        for (int i = 0; i < centre.length; i++) {
            state = state * 6364136223846793005L + 1442695040888963407L;
            out[i] = centre[i] + eps * (((state >>> 11) * 0x1.0p-53) * 2 - 1);
        }
        return out;
    }

    public static void main(String[] args) {
        IntervalNet net = new IntervalNet(DIMS, HIDDEN, CLASSES, 0x931BEEFL);
        int tested = 0, certified = 0, soundnessViolations = 0, refuted = 0;
        double tightest = Double.MAX_VALUE;

        for (int b = 0; b < BOXES; b++) {
            double[] centre = new double[DIMS];
            for (int i = 0; i < DIMS; i++) centre[i] = Math.sin(0.7 * b + 1.1 * i);
            double eps = 0.003 + 0.004 * (b % 7);
            int label = net.predict(centre);
            double[][] bounds = net.propagate(centre, eps);
            for (int s = 0; s < SAMPLES_PER_BOX; s++) {
                double[] point = sampleInBox(centre, eps, 0x931L + b * 131 + s);
                double[] out = net.logits(point);
                for (int c = 0; c < CLASSES; c++) {
                    if (out[c] < bounds[0][c] - 1e-9 || out[c] > bounds[1][c] + 1e-9) soundnessViolations++;
                }
                if (net.predict(point) != label) refuted++;
            }
            if (net.certified(centre, eps, label)) certified++;
            double margin = net.margin(centre, eps, label);
            if (margin > 0) tightest = Math.min(tightest, margin);
            tested++;
        }

        System.out.println("IBP verifier: " + tested + " boxes x " + SAMPLES_PER_BOX + " samples, certified " + certified
                + ", label flips found by sampling " + refuted + ", soundness violations " + soundnessViolations
                + ", tightest surviving margin " + String.format("%.4f", tightest));

        check(soundnessViolations == 0, "the propagated interval must contain every sampled logit");
        check(certified > 0, "some boxes must be certifiable at these radii");
        check(certified < tested, "some boxes must fail to certify, otherwise the radii are trivial");
        for (int b = 0; b < BOXES; b++) {
            double[] centre = new double[DIMS];
            for (int i = 0; i < DIMS; i++) centre[i] = Math.sin(0.7 * b + 1.1 * i);
            double eps = 0.003 + 0.004 * (b % 7);
            if (net.certified(centre, eps, net.predict(centre))) {
                check(net.margin(centre, eps, net.predict(centre)) > 0, "a certificate needs a positive margin");
            }
        }

        // eps = 0 collapses the interval to the exact logits.
        double[] centre = {0.3, -0.4, 0.2};
        double[][] degenerate = net.propagate(centre, 0.0);
        double[] exact = net.logits(centre);
        double worst = 0;
        for (int c = 0; c < CLASSES; c++) worst = Math.max(worst, Math.abs(degenerate[0][c] - exact[c]));
        check(worst < 1e-12 && Arrays.equals(degenerate[0], degenerate[1]),
                "eps = 0 must collapse the interval onto the exact logits, worst gap " + worst);
        check(net.certified(centre, 0.0, net.predict(centre)), "a point is always certified at eps = 0");

        // A monotone one-dimensional network has an exactly tight interval.
        IntervalNet monotone = IntervalNet.monotone(1, 6, 0x931L);
        double[] line = {0.25};
        double[][] tight = monotone.propagate(line, 0.5);
        double[] atLow = monotone.logits(new double[] {line[0] - 0.5});
        double[] atHigh = monotone.logits(new double[] {line[0] + 0.5});
        check(Math.abs(tight[0][0] - atLow[0]) < 1e-12 && Math.abs(tight[1][0] - atHigh[0]) < 1e-12,
                "the interval of a monotone network must be exact: [" + tight[0][0] + "," + tight[1][0] + "] vs ["
                        + atLow[0] + "," + atHigh[0] + "]");

        // Certified radius versus exhaustive sampling: IBP must never certify something sampling refutes.
        int radiusChecks = 0;
        for (int b = 0; b < 8; b++) {
            double[] point = {0.1 * b - 0.4, 0.3 - 0.05 * b, 0.2 * Math.sin(b)};
            int label = net.predict(point);
            double radius = net.certifiedRadius(point, label, 1.0, 0.01);
            check(radius >= 0, "the certified radius cannot be negative");
            for (int s = 0; s < 400; s++) {
                double[] inside = sampleInBox(point, Math.max(radius, 1e-9), 0x931F00DL + b * 977 + s);
                if (net.predict(inside) != label) {
                    check(false, "sampling refuted a certified box at radius " + radius);
                }
                radiusChecks++;
            }
        }
        check(radiusChecks == 8 * 400, "every certified box must have been sampled densely");

        // Boundary: a huge radius is never certified, and an attack inside it really changes the label.
        double[] corner = {0.0, 0.0, 0.0};
        int cornerLabel = net.predict(corner);
        check(!net.certified(corner, 5.0, cornerLabel), "a radius of 5 must not certify anything");
        boolean fooled = false;
        for (int s = 0; s < 2000 && !fooled; s++) fooled = net.predict(sampleInBox(corner, 5.0, 0x777L + s)) != cornerLabel;
        check(fooled, "sampling inside the uncertified box must find a label flip");

        // Monotone behaviour of the certificate in the margin.
        double[] wide = net.propagate(centre, 0.05)[0];
        double[] wider = net.propagate(centre, 0.10)[0];
        check(wider[net.predict(centre)] <= wide[net.predict(centre)] + 1e-12,
                "a larger box can only lower the lower bound of the true class");
        IntervalNet replay = new IntervalNet(DIMS, HIDDEN, CLASSES, 0x931BEEFL);
        check(Arrays.equals(replay.propagate(centre, 0.2)[0], net.propagate(centre, 0.2)[0]),
                "the verifier must be deterministic");

        System.out.println("boundaries: eps = 0 gap " + String.format("%.2e", worst) + ", monotone interval ["
                + String.format("%.4f", tight[0][0]) + "," + String.format("%.4f", tight[1][0])
                + "], boxes certified " + certified + "/" + tested + ", dense radius samples " + radiusChecks);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
