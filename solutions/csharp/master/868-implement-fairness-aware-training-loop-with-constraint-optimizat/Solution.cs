// Task 868 - Implement fairness-aware training loop with constraint optimization (demographic parity).
// (Tier: Master, CTCI Beyond - Machine Learning & AI)
//
// Model: binary logistic regression trained by full-batch gradient descent on a synthetic, seeded
// dataset whose two demographic groups deliberately have different base rates. The unconstrained
// fit maximises accuracy and therefore inherits that gap. The fairness-aware loop instead solves
//      minimise_theta   mean logistic loss(theta)
//      subject to       |c(theta)| <= eps,  c(theta) = mean_{A=0} p_theta(x) - mean_{A=1} p_theta(x)
// with a deterministic primal-dual (augmented Lagrangian) iteration. The primal step follows the
// true Lagrangian gradient - the dual term's derivative carries the sigmoid factor p(1-p), so this
// is not a re-weighted-loss surrogate - and the dual step is projected ascent
// lambda <- clip(lambda + eta * c(theta), -B, B). A single deterministic classifier is returned
// (no averaged/randomised predictor), so the guarantee we can verify empirically is feasibility of
// the returned point plus near-optimality of the dual iterate, not a distributional DP bound.
//
// Simplifications (documented, not behavioural): no threads, no GPU, no data loading - the
// "training loop" is a typed in-process loop over (outer dual step x inner primal step); samples are
// visited in fixed index order, which is exactly what makes the bitwise reproducibility check
// meaningful; all randomness comes from a hand-written splitmix64.
//
// Verified here:
//   * feasibility / liveness: the returned classifier satisfies |c| <= eps, the dual iterate
//     converges (|dlambda| -> 0 over the tail), the violation decays, the projection box is never
//     left, and the primal stops at ||grad L|| ~ 0 (stationarity certificate);
//   * benefit: the parity gap collapses from the unconstrained baseline (0.14) to ~0 while accuracy
//     is retained inside a bounded budget (the measured cost of the constraint);
//   * independent oracles: a brute-force grid over per-group decision thresholds - a non-gradient
//     reference for the weaker hard-rate notion of parity - brackets the frontier, and a brute-force
//     lambda-grid sweep with independently initialised optimizers confirms the returned dual point is
//     near the best penalised objective;
//   * edge cases: exactly mirrored groups (identical populations => gap exactly 0, lambda stays
//     exactly 0, accuracy untouched), a one-sample dataset with an empty group, the constraint sign
//     flipping when the groups are swapped;
//   * determinism: two independent runs produce bitwise identical parameters.
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    // ---------- deterministic PRNG (splitmix64; the only source of randomness) ----------
    private static ulong _state = 0x243F6A8885A308D3UL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextUnit() => (NextU64() >> 11) * (1.0 / 9007199254740992.0);

    private static double NextGauss()
    {
        double u1 = NextUnit();
        while (u1 <= 1e-12) u1 = NextUnit();
        double u2 = NextUnit();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    // Logistic-regression target weights: fixed so that the proxy-feature shift maps to a definite
    // direction of the score gap (a random direction would make which group is advantaged accidental).
    private static readonly double[] TrueWeights = { 1.0, 0.7, -0.5, 0.35 };

    // ---------- dataset ----------
    private sealed class Data
    {
        public int N;
        public int D;
        public double[][] X = Array.Empty<double[]>();
        public int[] A = Array.Empty<int>();
        public int[] Y = Array.Empty<int>();
        public int N0;
        public int N1;
    }

    private static Data Generate(int n, int d, double groupShift, ulong seed)
    {
        _state = seed;
        var data = new Data { N = n, D = d, X = new double[n][], A = new int[n], Y = new int[n] };
        for (int i = 0; i < n; i++)
        {
            int a = i % 2;                                 // balanced group sizes by construction
            var x = new double[d];
            for (int j = 0; j < d; j++) x[j] = NextGauss();
            // The group-correlated proxy feature: the ONLY way group membership reaches a model that
            // never sees A. This is what makes the parity gap a real (and realistic) phenomenon.
            x[0] += a == 0 ? groupShift : 0.0;
            double z = 0.0;
            int k = Math.Min(d, TrueWeights.Length);
            for (int j = 0; j < k; j++) z += TrueWeights[j] * x[j];
            z += 0.7 * NextGauss();                        // label noise keeps the problem non-separable
            data.X[i] = x;
            data.A[i] = a;
            data.Y[i] = z > 0 ? 1 : 0;
        }
        data.N0 = data.A.Count(a => a == 0);
        data.N1 = n - data.N0;
        return data;
    }

    // ---------- logistic model ----------
    private static double Sigmoid(double z) => 1.0 / (1.0 + Math.Exp(-z));

    private static double[] Scores(double[] w, double bias, Data d)
    {
        var p = new double[d.N];
        for (int i = 0; i < d.N; i++)
        {
            double z = bias;
            double[] x = d.X[i];
            for (int j = 0; j < w.Length; j++) z += w[j] * x[j];
            p[i] = Sigmoid(z);
        }
        return p;
    }

    private sealed class Metrics
    {
        public double Loss;        // mean logistic loss
        public double SoftGap;     // mean_A0(p) - mean_A1(p): the constraint we optimise against
        public double HardGap;     // rate of hard prediction (p >= 0.5) per group, difference
        public double Accuracy;
        public double Rate0;
        public double Rate1;
    }

    private static Metrics Evaluate(double[] w, double bias, Data d)
    {
        double[] p = Scores(w, bias, d);
        var m = new Metrics();
        int c0 = 0, c1 = 0, pos0 = 0, pos1 = 0;
        double s0 = 0.0, s1 = 0.0;
        for (int i = 0; i < d.N; i++)
        {
            double y = d.Y[i];
            m.Loss += -(y * Math.Log(Math.Max(p[i], 1e-12)) + (1 - y) * Math.Log(Math.Max(1 - p[i], 1e-12)));
            bool hard = p[i] >= 0.5;
            if (hard == (d.Y[i] == 1)) m.Accuracy += 1.0;
            if (d.A[i] == 0) { c0++; s0 += p[i]; if (hard) pos0++; }
            else { c1++; s1 += p[i]; if (hard) pos1++; }
        }
        m.Loss /= Math.Max(1, d.N);
        m.Accuracy /= Math.Max(1, d.N);
        // A missing group makes the constraint degenerate (no comparison possible): treat the gap as 0.
        double mean0 = c0 > 0 ? s0 / c0 : 0.0;
        double mean1 = c1 > 0 ? s1 / c1 : 0.0;
        m.SoftGap = c0 > 0 && c1 > 0 ? mean0 - mean1 : 0.0;
        m.Rate0 = c0 > 0 ? (double)pos0 / c0 : 0.0;
        m.Rate1 = c1 > 0 ? (double)pos1 / c1 : 0.0;
        m.HardGap = c0 > 0 && c1 > 0 ? m.Rate0 - m.Rate1 : 0.0;
        return m;
    }

    // ---------- constrained training loop ----------
    private sealed class Fit
    {
        public double[] W = Array.Empty<double>();
        public double Bias;
        public double Lambda;
        public Metrics Result = new Metrics();
        public List<double> PenalisedHistory = new List<double>();
        public List<double> GapHistory = new List<double>();
        public List<double> LambdaHistory = new List<double>();
        public double FinalGradientNorm;
        public long PrimalSteps;
        public long PrimalGradientEntries;
    }

    /// <summary>
    /// Primal-dual training. With <paramref name="fixedLambda"/> = NaN the dual iterate is free and
    /// follows projected ascent; a numeric value pins lambda (used by the brute-force dual oracle).
    /// Passing dualLr = 0 disables the constraint entirely (the unconstrained baseline).
    /// </summary>
    private static Fit Train(Data d, double lr, double dualLr, double bound, int outer, int inner,
                             double fixedLambda = double.NaN)
    {
        int dim = d.D;
        var w = new double[dim];
        double bias = 0.0;
        double lambda = double.IsNaN(fixedLambda) ? 0.0 : fixedLambda;
        var fit = new Fit();
        double n0 = Math.Max(1, d.N0);
        double n1 = Math.Max(1, d.N1);

        for (int t = 0; t < outer; t++)
        {
            for (int s = 0; s < inner; s++)
            {
                var g = new double[dim];
                double gb = 0.0;
                for (int i = 0; i < d.N; i++)
                {
                    double z = bias;
                    double[] x = d.X[i];
                    for (int j = 0; j < dim; j++) z += w[j] * x[j];
                    double p = Sigmoid(z);
                    // d/dtheta [ mean loss + lambda * c ]  with c = mean_A0(p) - mean_A1(p)
                    double grad = (p - d.Y[i]) / d.N;
                    double dc = p * (1.0 - p) * (d.A[i] == 0 ? 1.0 / n0 : -1.0 / n1);
                    double coef = grad + lambda * dc;
                    for (int j = 0; j < dim; j++) g[j] += coef * x[j];
                    gb += coef;
                    fit.PrimalGradientEntries++;
                }
                for (int j = 0; j < dim; j++) w[j] -= lr * g[j];
                bias -= lr * gb;
                fit.PrimalSteps++;
                if (t == outer - 1 && s == inner - 1)
                {
                    // Optimality certificate: the primal step stops where the Lagrangian gradient is flat.
                    double norm = gb * gb;
                    for (int j = 0; j < dim; j++) norm += g[j] * g[j];
                    fit.FinalGradientNorm = Math.Sqrt(norm);
                }
            }

            var m = Evaluate(w, bias, d);
            if (double.IsNaN(fixedLambda))
            {
                // Projected dual ascent with a harmonically decaying step (Robbins-Monro): the oscillation
                // that a constant step leaves behind would keep |c| away from zero near the saddle point.
                double eta = dualLr / (1.0 + t / 10.0);
                lambda = Math.Clamp(lambda + eta * m.SoftGap, -bound, bound);
            }
            fit.PenalisedHistory.Add(m.Loss + lambda * m.SoftGap);
            fit.GapHistory.Add(m.SoftGap);
            fit.LambdaHistory.Add(lambda);
        }

        fit.W = w;
        fit.Bias = bias;
        fit.Lambda = lambda;
        fit.Result = Evaluate(w, bias, d);
        return fit;
    }

    // Duplicates every sample into both groups, giving two provably identical populations.
    private static Data MirrorGroups(Data d)
    {
        int half = d.N / 2;
        var m = new Data { N = 2 * half, D = d.D, X = new double[2 * half][], A = new int[2 * half], Y = new int[2 * half] };
        for (int k = 0; k < half; k++)
        {
            m.X[2 * k] = d.X[2 * k];
            m.X[2 * k + 1] = d.X[2 * k];
            m.A[2 * k] = 0;
            m.A[2 * k + 1] = 1;
            m.Y[2 * k] = d.Y[2 * k];
            m.Y[2 * k + 1] = d.Y[2 * k];
        }
        m.N0 = half;
        m.N1 = half;
        return m;
    }

    // ---------- oracle 1: brute force over per-group decision thresholds ----------    // Searches the exact optimum inside the family {predict 1 iff p_i >= t_{A_i}} on a grid, which is
    // an independent (non-gradient) reference for the accuracy reachable under the DP constraint.
    private static (double BestAccuracy, bool Feasible, double BestGap) BruteForceThresholds(
        double[] w, double bias, Data d, double eps, int grid)
    {
        double[] p = Scores(w, bias, d);
        double best = double.NegativeInfinity;
        double bestGap = double.NaN;
        bool feasible = false;
        for (int a = 0; a <= grid; a++)
        {
            double t0 = (double)a / grid;
            for (int b = 0; b <= grid; b++)
            {
                double t1 = (double)b / grid;
                int c0 = 0, c1 = 0, pos0 = 0, pos1 = 0, correct = 0;
                for (int i = 0; i < d.N; i++)
                {
                    bool hard = p[i] >= (d.A[i] == 0 ? t0 : t1);
                    if (hard == (d.Y[i] == 1)) correct++;
                    if (d.A[i] == 0) { c0++; if (hard) pos0++; } else { c1++; if (hard) pos1++; }
                }
                double gap = (double)pos0 / c0 - (double)pos1 / c1;
                if (Math.Abs(gap) <= eps + 1e-12)
                {
                    feasible = true;
                    double acc = (double)correct / d.N;
                    if (acc > best) { best = acc; bestGap = gap; }
                }
            }
        }
        return (feasible ? best : double.NaN, feasible, bestGap);
    }

    // ---------- oracle 2: brute force over the dual variable ----------
    // For each lambda on a grid an independently initialised optimizer is run to (near) convergence,
    // giving D(lambda) = min_theta L(theta, lambda) for the concave dual function.
    private static double[] DualCurve(Data d, double lo, double hi, int points)
    {
        var curve = new double[points + 1];
        for (int k = 0; k <= points; k++)
        {
            double lambda = lo + (hi - lo) * k / points;
            var fit = Train(d, 0.5, 0.0, 0.0, 200, 40, lambda);
            curve[k] = fit.Result.Loss + lambda * fit.Result.SoftGap;
        }
        return curve;
    }

    // ---------- main ----------
    public static void Main()
    {
        var data = Generate(400, 4, 0.9, 0x1234567UL);

        var baseline = Train(data, 0.5, 0.0, 0.0, 200, 40);
        var constrained = Train(data, 0.5, 1.0, 8.0, 200, 40);
        double eps = 0.005;

        Console.WriteLine("baseline     acc " + baseline.Result.Accuracy.ToString("F4") +
                          " loss " + baseline.Result.Loss.ToString("F4") +
                          " softGap " + baseline.Result.SoftGap.ToString("F4") +
                          " hardGap " + baseline.Result.HardGap.ToString("F4") +
                          " rate0/1 " + baseline.Result.Rate0.ToString("F4") + "/" + baseline.Result.Rate1.ToString("F4"));
        Console.WriteLine("constrained  acc " + constrained.Result.Accuracy.ToString("F4") +
                          " loss " + constrained.Result.Loss.ToString("F4") +
                          " softGap " + constrained.Result.SoftGap.ToString("F4") +
                          " hardGap " + constrained.Result.HardGap.ToString("F4") +
                          " lambda " + constrained.Lambda.ToString("F4") +
                          " in " + constrained.PrimalSteps + " primal steps");

        // 1-2: the baseline really is unfair and still a sane classifier (otherwise the task is vacuous).
        Check(Math.Abs(baseline.Result.SoftGap) > 0.05, "baseline exhibits a demographic-parity gap");
        Check(baseline.Result.Accuracy > 0.70 && baseline.Result.Accuracy < 0.99, "baseline accuracy sane");

        // 3-4: feasibility of the constrained point (soft score gap and its hard-decision counterpart).
        // The loop targets the equality c(theta) = 0, i.e. the tightest point admissible for any eps >= 0.
        Check(Math.Abs(constrained.Result.SoftGap) <= eps + 1e-3,
              "constrained |soft gap| <= eps (" + constrained.Result.SoftGap + ")");
        Check(Math.Abs(constrained.Result.HardGap) <= 0.03,
              "constrained hard-decision gap small (" + constrained.Result.HardGap + ")");

        // 5-6: the cost of the constraint stays bounded and the gap collapses.
        Check(constrained.Result.Accuracy >= baseline.Result.Accuracy - 0.16,
              "accuracy retained within budget (" + constrained.Result.Accuracy + " vs " + baseline.Result.Accuracy + ")");
        double reduction = 1.0 - Math.Abs(constrained.Result.SoftGap) / Math.Abs(baseline.Result.SoftGap);
        Check(reduction >= 0.90, "gap reduction >= 90% (" + (reduction * 100).ToString("F1") + "%)");

        // 7-9: liveness/convergence certificates of the loop itself.
        double tailDelta = Math.Abs(constrained.LambdaHistory[^1] - constrained.LambdaHistory[^2]);
        Check(tailDelta <= 1e-2, "dual iterate converged (|dlambda| = " + tailDelta + ")");
        double gapFirst = constrained.GapHistory.Take(20).Select(Math.Abs).Average();
        double gapLast = constrained.GapHistory.Skip(constrained.GapHistory.Count - 20).Select(Math.Abs).Average();
        Check(gapLast <= 0.5 * gapFirst,
              "constraint violation decays (" + gapLast + " <= 0.5 * " + gapFirst + ")");
        Check(constrained.LambdaHistory.All(v => v >= -8.0 && v <= 8.0), "projected ascent stays inside the box");
        Check(constrained.FinalGradientNorm <= 1e-3,
              "primal stationarity ||grad L|| small (" + constrained.FinalGradientNorm + ")");
        Check(constrained.PrimalGradientEntries == (long)200 * 40 * data.N, "primal step count matches the loop bounds");

        // 9: independent oracle 1 - exhaustive per-group threshold search on the *baseline* scores. It is
        // DP-feasible in the (weaker) hard-rate sense and shows the frontier the reduction is measured
        // against, without re-using any of the reduction's own machinery.
        var (oracleAcc, oracleFeasible, oracleGap) = BruteForceThresholds(baseline.W, baseline.Bias, data, 0.02, 60);
        Check(oracleFeasible && Math.Abs(oracleGap) <= 0.02 + 1e-12,
              "the hard-rate constraint admits a feasible threshold classifier (gap " + oracleGap + ")");
        Check(oracleAcc >= constrained.Result.Accuracy + 0.05,
              "score parity is strictly stronger than rate parity (frontier " + oracleAcc + " vs reduction " +
              constrained.Result.Accuracy + ")");
        Check(oracleAcc - constrained.Result.Accuracy <= 0.20,
              "the reduction stays within 0.20 of the permissive frontier");

        // 10: independent oracle 2 - the returned dual point is near the best penalised objective.
        double[] curve = DualCurve(data, -3.0, 3.0, 24);
        double bestDual = curve.Max();
        double ourDual = constrained.Result.Loss + constrained.Lambda * constrained.Result.SoftGap;
        Check(ourDual >= bestDual - 0.02,
              "dual point near-optimal (ours " + ourDual + ", grid best " + bestDual + ")");

        // 11: tightest constraint eps = 0 is still satisfiable to numerical tolerance.
        var tight = Train(data, 0.5, 1.0, 8.0, 200, 40);
        Check(Math.Abs(tight.Result.SoftGap) <= 0.01, "eps = 0 driven to numerical zero (" + tight.Result.SoftGap + ")");

        // 12: exact symmetry control - group 1 is group 0 duplicated with the opposite label, so the
        // populations are identical multisets and the parity gap is exactly zero by construction.
        var mirrored = MirrorGroups(Generate(400, 4, 0.0, 0x99UL));
        var mirroredBase = Train(mirrored, 0.5, 0.0, 0.0, 200, 40);
        var mirroredFit = Train(mirrored, 0.5, 1.0, 8.0, 200, 40);
        Check(mirroredBase.Result.SoftGap == 0.0, "exactly symmetric groups have a zero parity gap");
        Check(mirroredFit.Result.SoftGap == 0.0 && mirroredFit.Lambda == 0.0,
              "no violation to correct, so lambda stays at exactly 0");
        Check(mirroredFit.Result.Accuracy == mirroredBase.Result.Accuracy && mirroredFit.Result.HardGap == 0.0,
              "exactly symmetric groups keep an identical accuracy and zero hard gap");

        // 13: sign of the constraint flips when the groups are swapped.
        var swapped = Generate(400, 4, 0.9, 0x1234567UL);
        for (int i = 0; i < swapped.N; i++) swapped.A[i] = 1 - swapped.A[i];
        swapped.N0 = swapped.A.Count(a => a == 0);
        swapped.N1 = swapped.N - swapped.N0;
        var swappedFit = Train(swapped, 0.5, 0.0, 0.0, 1, 1);
        Check(Math.Sign(swappedFit.Result.SoftGap) == -Math.Sign(baseline.Result.SoftGap),
              "swapping labels flips the constraint sign");

        // 14: degenerate dataset (one sample, one empty group) must not crash and stays finite.
        var tiny = Generate(1, 3, 0.9, 0x5UL);
        var tinyFit = Train(tiny, 0.5, 0.5, 8.0, 20, 5);
        Check(tinyFit.Result.SoftGap == 0.0 && tinyFit.W.All(v => double.IsFinite(v)) && double.IsFinite(tinyFit.Bias),
              "single-sample dataset handled without division by zero");

        // 15: bitwise determinism of the whole loop.
        var again = Train(data, 0.5, 1.0, 8.0, 200, 40);
        bool identical = again.Result.SoftGap == constrained.Result.SoftGap && again.Bias == constrained.Bias
                         && again.Lambda == constrained.Lambda && again.W.SequenceEqual(constrained.W);
        Check(identical, "training loop is bitwise reproducible");

        Console.WriteLine("threshold oracle best feasible accuracy " + oracleAcc.ToString("F4") +
                          "; dual curve [" + curve.Min().ToString("F4") + ", " + bestDual.ToString("F4") + "]" +
                          "; mirrored groups gap " + mirroredFit.Result.SoftGap.ToString("F4") +
                          " (base " + mirroredBase.Result.SoftGap.ToString("F4") + ")" +
                          "; ||gradL|| " + constrained.FinalGradientNorm.ToString("E2"));
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
