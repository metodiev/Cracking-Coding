/** Task 943 - Implement composable privacy guarantees across combined DP mechanisms.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 *  A differential-privacy accountant: pure (eps,0) mechanisms compose by summation, the advanced
 *  composition theorem gives the tighter sqrt(k)-ish bound for many mechanisms, disjoint datasets compose
 *  by parallel composition, and Gaussian mechanisms are priced through Renyi differential privacy whose
 *  RDP curve is converted back to (eps, delta) by minimising over the Renyi order alpha. Calibration runs
 *  the conversion backwards: a geometric bisection finds the smallest noise sigma that still delivers the
 *  requested (eps, delta), and subsampling amplification reports how much a q-fraction sample buys.
 *  Simplified: closed-form Gaussian/Laplace composition only, a fixed alpha grid instead of the analytic
 *  optimum, and no support for mechanisms whose RDP curve is only available numerically.
 *  Verified: basic composition is exact, parallel composition is at most its basic counterpart, advanced
 *  composition beats basic for many mechanisms, the RDP conversion matches a dense independent optimisation
 *  of the same curve, calibration round-trips to the requested epsilon, the classical sigma bound is sound,
 *  the sqrt(k) scaling law holds, and invalid privacy parameters are rejected.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    /** (eps, delta) pair. */
    static final class Budget {
        final double eps, delta;
        Budget(double eps, double delta) { this.eps = eps; this.delta = delta; }
    }

    static Budget basic(double[][] mechanisms) {
        double eps = 0, delta = 0;
        for (double[] m : mechanisms) {
            eps += m[0];
            delta += m[1];
        }
        return new Budget(eps, delta);
    }

    /** Advanced composition for k pure mechanisms: sum eps(e^eps - 1) + sqrt(2 ln(1/delta') sum eps^2). */
    static Budget advanced(double[] epsilons, double deltaPrime) {
        if (deltaPrime <= 0 || deltaPrime >= 1) {
            throw new IllegalArgumentException("delta' out of range: " + deltaPrime);
        }
        double linear = 0, quadratic = 0;
        for (double e : epsilons) {
            linear += e * (Math.exp(e) - 1);
            quadratic += e * e;
        }
        return new Budget(linear + Math.sqrt(2 * Math.log(1 / deltaPrime) * quadratic), deltaPrime);
    }

    /** Validating constructor for a privacy budget, used for the negative checks. */
    static Budget budget(double eps, double delta) {
        if (eps < 0) throw new IllegalArgumentException("negative epsilon: " + eps);
        if (delta < 0 || delta >= 1) throw new IllegalArgumentException("delta out of range: " + delta);
        return new Budget(eps, delta);
    }

    /** Parallel composition over disjoint datasets: the privacy loss is the worst mechanism. */
    static Budget parallel(double[][] mechanisms) {
        double eps = 0, delta = 0;
        for (double[] m : mechanisms) {
            eps = Math.max(eps, m[0]);
            delta = Math.max(delta, m[1]);
        }
        return new Budget(eps, delta);
    }

    /** RDP of the sensitivity-1 Gaussian mechanism at order alpha. */
    static double gaussianRdp(double sigma, double alpha) { return alpha / (2 * sigma * sigma); }

    static double rdpToDp(double rdp, double alpha, double delta) {
        return rdp + Math.log(1 / delta) / (alpha - 1);
    }

    static final int ALPHA_POINTS = 200;

    static double alphaAt(int index) { return 1 + 127.0 * Math.pow((index + 1.0) / ALPHA_POINTS, 3); }

    /** Best (eps, delta) the RDP curve of `count` identical Gaussian mechanisms supports. */
    static double composedGaussianEps(double sigma, int count, double delta) {
        double best = Double.MAX_VALUE;
        for (int i = 0; i < ALPHA_POINTS; i++) {
            double alpha = alphaAt(i);
            best = Math.min(best, rdpToDp(count * gaussianRdp(sigma, alpha), alpha, delta));
        }
        return best;
    }

    /** Independent dense scan over the same RDP curve, used as the optimisation oracle. */
    static double denseScanEps(double sigma, int count, double delta) {
        double best = Double.MAX_VALUE;
        for (int i = 0; i < 4000; i++) {
            double alpha = 1 + 127.0 * Math.pow((i + 1.0) / 4000, 2);
            best = Math.min(best, rdpToDp(count * gaussianRdp(sigma, alpha), alpha, delta));
        }
        return best;
    }

    /** Smallest sigma whose composed privacy loss is at most the requested epsilon. */
    static double calibrateSigma(double eps, double delta) {
        double lo = 0.05, hi = 1e5;
        for (int i = 0; i < 60; i++) {
            double mid = Math.sqrt(lo * hi);
            if (composedGaussianEps(mid, 1, delta) > eps) lo = mid;
            else hi = mid;
        }
        return hi;
    }

    /** The textbook closed form for the Gaussian mechanism, an independent (looser) calibration. */
    static double classicalSigma(double eps, double delta) {
        if (eps <= 0) return Double.POSITIVE_INFINITY;
        return Math.sqrt(2 * Math.log(1.25 / delta)) / eps;
    }

    /** Amplification by subsampling: only a q fraction of the dataset is used. */
    static double subsampledEpsilon(double q, double eps) {
        if (q < 0 || q > 1) throw new IllegalArgumentException("sample rate out of range: " + q);
        return Math.log(1 + q * (Math.exp(eps) - 1));
    }

    static String expectThrow(Runnable action) {
        try {
            action.run();
            return "";
        } catch (IllegalArgumentException e) {
            return e.getMessage();
        }
    }

    public static void main(String[] args) {
        double[][] mechanisms = {{0.1, 0}, {0.2, 0}, {0.05, 1e-7}, {0.3, 0}};
        Budget basic = basic(mechanisms);
        check(Math.abs(basic.eps - 0.65) < 1e-12 && Math.abs(basic.delta - 1e-7) < 1e-18,
                "basic composition must be exact: " + basic.eps + ", " + basic.delta);
        check(basic(new double[][] {{1.0, 0}}) .eps == 1.0 && basic(new double[0][]) .eps == 0
                        && basic(new double[0][]) .delta == 0,
                "a single mechanism must compose to itself and an empty list to nothing");

        Budget spread = parallel(new double[][] {{0.4, 1e-8}, {0.1, 1e-9}, {0.25, 1e-7}});
        check(spread.eps == 0.4 && spread.delta == 1e-7 && spread.eps <= basic(
                        new double[][] {{0.4, 1e-8}, {0.1, 1e-9}, {0.25, 1e-7}}).eps
                        && spread.eps >= 0.25,
                "parallel composition must be the worst mechanism and never exceed the basic sum: " + spread.eps);

        double[] many = new double[100];
        Arrays.fill(many, 0.05);
        double basicSum = 0;
        for (double e : many) basicSum += e;
        Budget adv = advanced(many, 1e-6);
        check(adv.eps < basicSum && adv.eps > 0 && adv.delta == 1e-6 && advanced(new double[] {0.1}, 1e-6).eps > 0,
                "advanced composition must beat basic composition for 100 mechanisms: " + adv.eps + " vs "
                        + basicSum);

        double sigma = calibrateSigma(1.0, 1e-6);
        double coarse = composedGaussianEps(sigma, 100, 1e-6);
        double dense = denseScanEps(sigma, 100, 1e-6);
        double atTwo = rdpToDp(100 * gaussianRdp(sigma, 2.0), 2.0, 1e-6);
        double atMax = rdpToDp(100 * gaussianRdp(sigma, 128.0), 128.0, 1e-6);
        check(coarse <= atTwo && coarse <= atMax && Math.abs(coarse - dense) < 0.05 && dense <= coarse + 1e-9,
                "the RDP conversion must be minimised over alpha: coarse " + coarse + ", dense " + dense
                        + ", alpha=2 " + atTwo + ", alpha=128 " + atMax);

        int roundTripViolations = 0;
        double[][] targets = {{0.5, 1e-5}, {1.0, 1e-6}, {2.0, 1e-7}};
        for (double[] target : targets) {
            double calibrated = calibrateSigma(target[0], target[1]);
            double achieved = composedGaussianEps(calibrated, 1, target[1]);
            if (Math.abs(achieved - target[0]) / target[0] > 0.03) roundTripViolations++;
        }
        check(roundTripViolations == 0, roundTripViolations + " of " + targets.length
                + " calibrations failed to round-trip to the requested epsilon");

        int soundnessViolations = 0;
        double[] classicalGap = new double[targets.length];
        for (int i = 0; i < targets.length; i++) {
            double classical = classicalSigma(targets[i][0], targets[i][1]);
            double achieved = composedGaussianEps(classical, 1, targets[i][1]);
            classicalGap[i] = achieved / targets[i][0];
            if (achieved > targets[i][0] * 1.1) soundnessViolations++;
        }
        check(soundnessViolations == 0 && classicalGap[1] > 0.8,
                "the classical sigma bound must be sound: achieved/target " + Arrays.toString(classicalGap));

        double eps100 = composedGaussianEps(sigma, 100, 1e-6);
        double eps400 = composedGaussianEps(sigma, 400, 1e-6);
        double law100 = Math.sqrt(2 * 100 * Math.log(1 / 1e-6)) / sigma;
        double law400 = Math.sqrt(2 * 400 * Math.log(1 / 1e-6)) / sigma;
        check(eps100 / law100 > 0.8 && eps100 / law100 < 1.4 && eps400 / law400 > 0.8 && eps400 / law400 < 1.4
                        && eps400 / eps100 > 2.0 && eps400 / eps100 < 2.6,
                "Gaussian composition must follow the sqrt(k) law: " + eps100 + " vs " + law100 + ", " + eps400
                        + " vs " + law400);

        double amplified = subsampledEpsilon(0.01, 1.0);
        double shrink = subsampledEpsilon(0.5, 1.0);
        double small = subsampledEpsilon(0.01, 0.1);
        check(amplified < shrink && shrink < 1.0 && Math.abs(subsampledEpsilon(1.0, 1.0) - 1.0) < 1e-12
                        && amplified <= 0.02 && amplified > 0.009 && small / (0.01 * 0.1) < 1.1
                        && subsampledEpsilon(0.25, 2.0) > subsampledEpsilon(0.1, 2.0),
                "subsampling must amplify privacy and vanish at q = 1: " + amplified + " / " + shrink + " / "
                        + small);

        String badDelta = expectThrow(() -> advanced(new double[] {0.1}, 1.0));
        String badRate = expectThrow(() -> subsampledEpsilon(1.5, 1.0));
        String badNeg = expectThrow(() -> budget(-0.1, 1e-6));
        String badBudgetDelta = expectThrow(() -> budget(0.1, 1.0));
        check(!badDelta.isEmpty() && !badRate.isEmpty() && !badNeg.isEmpty() && !badBudgetDelta.isEmpty()
                        && expectThrow(() -> advanced(new double[] {0.1}, 1e-6)).isEmpty(),
                "invalid privacy parameters must be rejected: " + badDelta + " | " + badRate + " | " + badNeg
                        + " | " + badBudgetDelta);

        double[] again = {calibrateSigma(1.0, 1e-6), composedGaussianEps(sigma, 100, 1e-6), adv.eps,
                denseScanEps(sigma, 100, 1e-6), subsampledEpsilon(0.01, 1.0)};
        check(again[0] == sigma && again[1] == coarse && again[2] == adv.eps && again[3] == dense
                        && again[4] == amplified,
                "the accountant must be fully reproducible");

        System.out.println("accountant: basic(5) " + String.format("%.4f", basic.eps) + ", parallel "
                + spread.eps + ", advanced(100) " + String.format("%.4f", adv.eps) + " vs basic "
                + String.format("%.4f", basicSum));
        System.out.println("sigma(" + 1.0 + ", 1e-6) = " + String.format("%.4f", sigma) + " (classical "
                + String.format("%.4f", classicalSigma(1.0, 1e-6)) + "), eps(100) "
                + String.format("%.3f", eps100) + " vs law " + String.format("%.3f", law100) + ", eps(400) "
                + String.format("%.3f", eps400));
        System.out.println("PASS");
    }
}
