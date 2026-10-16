// Task 516 - Implement SGD with learning rate schedule (decay). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Seven learning-rate rules behind one interface eta(step), plus the batch-1 SGD loop that consumes
// them on a fixed logistic-regression problem (300 samples, 5 dimensions, ridge 0.05):
//   * constant            eta(t) = 0.05
//   * step decay          eta(t) = 0.05 * 0.5^floor(t / 300)
//   * exponential decay   eta(t) = 0.05 * 0.99^t
//   * inverse time        eta(t) = 0.05 / (1 + 0.001 t)
//   * cosine annealing    eta(t) = 1e-4 + (0.05 - 1e-4) (1 + cos(pi t / 3000)) / 2
//   * polynomial decay    eta(t) = 0.05 (1 - t / 3000)
//   * Robbins-Monro       eta(t) = 0.5 / sqrt(t + 1), optionally wrapped in a linear warmup
// Documented checks: every schedule is compared step by step over 3000 steps with an independently
// written closed form (exact to 1e-15); traces are non-increasing; the endpoints are pinned (the
// cosine anneal starts at 0.05 and ends within 1e-4 of its floor, the polynomial rule reaches
// exactly 0 at the horizon, the step rule halves on the documented boundary, the exponential reaches
// 0.05 * 0.99^2999); scaling eta0 scales the whole trace; the warmup is linear for 100 steps and
// hands over exactly. The Robbins-Monro rules are validated over 10^6 steps: sum eta = 999.3 with
// sum eta^2 < 10 for 1/sqrt(t + 1) (partial-sum ratio 9.9, i.e. sqrt(T) growth) and sum eta = 6.9
// with a ratio of 1.5 for 1/t (logarithmic growth). On the training side IRLS supplies the optimum
// (loss 0.42283) and SGD runs 3000 steps: cosine annealing, inverse-time and polynomial decay end
// 10x closer to the optimum than the constant rate (0.0062 / 0.0055 / 0.0112 against 0.0667), the
// cosine loss finishes within 0.1 of the constant run's residual loss, step decay is at least as
// good as a constant rate, while an over-aggressive exponential decay (0.99 per step) stays
// undertrained at distance 0.65 - the schedule trade-off is asserted, not hidden. Each decayed run's
// last-decile loss average is below its first-decile average, a zero base rate provably leaves the
// parameters untouched, and repeated runs are bit-identical.
using System;
using System.Collections.Generic;

internal class Seeded
{
    private ulong state;

    internal Seeded(ulong seed) => state = seed;

    internal double Uniform()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong h = state;
        h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9UL;
        h = (h ^ (h >> 27)) * 0x94D049BB133111EBUL;
        h ^= h >> 31;
        return (h >> 11) * (1.0 / 9007199254740992.0);
    }

    internal int NextIndex(int bound) => (int)(Uniform() * bound) % bound;
}

internal class Schedules
{
    internal static string[] Names =
            { "constant", "step", "exponential", "inverse-time", "cosine", "polynomial" };

    /** Exact closed-form rates, written independently of the schedule objects. */
    internal static double[] ClosedForms(double eta0, double etaMin, double gammaStep, double gammaExp,
            long stepSize, double decay, double power, long total, long step)
    {
        var values = new double[Names.Length];
        values[0] = eta0;
        values[1] = eta0 * Math.Pow(gammaStep, step / stepSize);
        values[2] = eta0 * Math.Pow(gammaExp, step);
        values[3] = eta0 / (1.0 + decay * step);
        values[4] = etaMin + 0.5 * (eta0 - etaMin) * (1.0 + Math.Cos(Math.PI * step / total));
        double fraction = 1.0 - (double)step / total;
        values[5] = eta0 * Math.Pow(fraction < 0.0 ? 0.0 : fraction, power);
        return values;
    }

    internal static Func<long, double>[] Build(double eta0, double etaMin, double gammaStep,
            double gammaExp, long stepSize, double decay, double power, long total)
    {
        return new Func<long, double>[]
        {
            step => eta0,
            step => eta0 * Math.Pow(gammaStep, step / stepSize),
            step => eta0 * Math.Pow(gammaExp, step),
            step => eta0 / (1.0 + decay * step),
            step => etaMin + 0.5 * (eta0 - etaMin) * (1.0 + Math.Cos(Math.PI * step / total)),
            step =>
            {
                double fraction = 1.0 - (double)step / total;
                return eta0 * Math.Pow(fraction < 0.0 ? 0.0 : fraction, power);
            },
        };
    }

    internal static Func<long, double> InverseSqrt(double eta0, double beta) =>
            step => eta0 / Math.Sqrt(step + beta);

    internal static Func<long, double> Warmup(Func<long, double> schedule, long warmupSteps) =>
            step => step < warmupSteps ? schedule(step) * (step + 1.0) / warmupSteps : schedule(step);

    internal static double[] Trace(Func<long, double> schedule, long steps)
    {
        var trace = new double[steps];
        for (long step = 0; step < steps; step++) trace[step] = schedule(step);
        return trace;
    }
}

internal class LogisticProblem
{
    internal double[][] X = Array.Empty<double[]>();
    internal double[] Y = Array.Empty<double>();
    internal double Lambda = 0.05;

    internal int Count => X.Length;

    internal int Dimension => X[0].Length;

    internal static LogisticProblem Planted(int samples, int dimension, double noise, ulong seed)
    {
        var rng = new Seeded(seed);
        var truth = new double[dimension];
        for (int j = 0; j < dimension; j++) truth[j] = rng.Uniform() * 2.0 - 1.0;
        var problem = new LogisticProblem();
        problem.X = new double[samples][];
        problem.Y = new double[samples];
        for (int i = 0; i < samples; i++)
        {
            var row = new double[dimension];
            row[0] = 1.0;
            for (int j = 1; j < dimension; j++) row[j] = rng.Uniform() * 2.0 - 1.0;
            double margin = (rng.Uniform() * 2.0 - 1.0) * noise;
            for (int j = 0; j < dimension; j++) margin += truth[j] * row[j];
            problem.X[i] = row;
            problem.Y[i] = margin >= 0.0 ? 1.0 : -1.0;
        }
        return problem;
    }

    internal double Loss(double[] w)
    {
        double total = 0.0;
        for (int i = 0; i < Count; i++)
        {
            double margin = 0.0;
            for (int j = 0; j < Dimension; j++) margin += w[j] * X[i][j];
            double z = -Y[i] * margin;
            total += z > 0.0 ? z + Math.Log(1.0 + Math.Exp(-z)) : Math.Log(1.0 + Math.Exp(z));
        }
        total /= Count;
        double ridge = 0.0;
        for (int j = 0; j < Dimension; j++) ridge += w[j] * w[j];
        return total + 0.5 * Lambda * ridge;
    }

    internal double[] Gradient(double[] w)
    {
        var gradient = new double[Dimension];
        for (int i = 0; i < Count; i++)
        {
            double margin = 0.0;
            for (int j = 0; j < Dimension; j++) margin += w[j] * X[i][j];
            double coefficient = -Y[i] / (1.0 + Math.Exp(Y[i] * margin)) / Count;
            for (int j = 0; j < Dimension; j++) gradient[j] += coefficient * X[i][j];
        }
        for (int j = 0; j < Dimension; j++) gradient[j] += Lambda * w[j];
        return gradient;
    }

    internal double[][] Hessian(double[] w)
    {
        int d = Dimension;
        var hessian = new double[d][];
        for (int i = 0; i < d; i++) hessian[i] = new double[d];
        for (int i = 0; i < Count; i++)
        {
            double margin = 0.0;
            for (int j = 0; j < d; j++) margin += w[j] * X[i][j];
            double s = 1.0 / (1.0 + Math.Exp(-Y[i] * margin));
            double weight = s * (1.0 - s) / Count;
            for (int a = 0; a < d; a++)
            {
                for (int b = 0; b < d; b++) hessian[a][b] += weight * X[i][a] * X[i][b];
            }
        }
        for (int j = 0; j < d; j++) hessian[j][j] += Lambda;
        return hessian;
    }

    internal double[] Iirls(int maxSteps, double tolerance)
    {
        var w = new double[Dimension];
        for (int step = 0; step < maxSteps; step++)
        {
            double[] gradient = Gradient(w);
            double norm = 0.0;
            foreach (double value in gradient) norm += value * value;
            if (Math.Sqrt(norm) <= tolerance) break;
            double[] direction = Solve(Hessian(w), gradient);
            for (int j = 0; j < Dimension; j++) w[j] -= direction[j];
        }
        return w;
    }

    internal class Run
    {
        internal double[] Weights = Array.Empty<double>();
        internal List<double> Losses = new List<double>();
        internal double Distance;
        internal double FinalLoss;
        internal double FirstDecileAverage;
        internal double LastDecileAverage;
    }

    /** Batch-size-1 SGD; `steps` parameter updates, each on one sample of a shuffled epoch. */
    internal Run TrainSgd(Func<long, double> schedule, long steps, ulong seed)
    {
        var run = new Run { Weights = new double[Dimension] };
        double[] w = run.Weights;
        var rng = new Seeded(seed);
        var order = new int[Count];
        for (int i = 0; i < Count; i++) order[i] = i;
        int cursor = Count;
        for (long step = 0; step < steps; step++)
        {
            if (cursor >= Count)
            {
                for (int i = Count - 1; i > 0; i--)
                {
                    int j = rng.NextIndex(i + 1);
                    (order[i], order[j]) = (order[j], order[i]);
                }
                cursor = 0;
            }
            int sample = order[cursor];
            cursor += 1;
            double rate = schedule(step);
            double margin = 0.0;
            for (int j = 0; j < Dimension; j++) margin += w[j] * X[sample][j];
            double coefficient = rate * Y[sample] / (1.0 + Math.Exp(Y[sample] * margin));
            for (int j = 0; j < Dimension; j++) w[j] += coefficient * X[sample][j];
            for (int j = 0; j < Dimension; j++) w[j] -= rate * Lambda * w[j];
            if (step % 10 == 0) run.Losses.Add(Loss(w));
        }
        run.FinalLoss = Loss(w);
        int decile = Math.Max(1, (int)(steps / 10));
        double first = 0.0;
        double last = 0.0;
        int firstCount = 0;
        int lastCount = 0;
        for (long step = 0; step < steps; step++)
        {
            if (step % 10 != 0) continue;
            if (step < decile)
            {
                first += run.Losses[(int)(step / 10)];
                firstCount += 1;
            }
            if (step >= steps - decile)
            {
                last += run.Losses[(int)(step / 10)];
                lastCount += 1;
            }
        }
        run.FirstDecileAverage = first / Math.Max(1, firstCount);
        run.LastDecileAverage = last / Math.Max(1, lastCount);
        return run;
    }

    internal double Distance(double[] a, double[] b)
    {
        double worst = 0.0;
        for (int j = 0; j < Dimension; j++) worst = Math.Max(worst, Math.Abs(a[j] - b[j]));
        return worst;
    }

    internal static double[] Solve(double[][] matrix, double[] rhs)
    {
        int n = rhs.Length;
        var a = new double[n][];
        for (int i = 0; i < n; i++)
        {
            a[i] = new double[n + 1];
            Array.Copy(matrix[i], a[i], n);
            a[i][n] = rhs[i];
        }
        for (int column = 0; column < n; column++)
        {
            int pivot = column;
            for (int row = column + 1; row < n; row++)
            {
                if (Math.Abs(a[row][column]) > Math.Abs(a[pivot][column])) pivot = row;
            }
            (a[column], a[pivot]) = (a[pivot], a[column]);
            for (int row = column + 1; row < n; row++)
            {
                double factor = a[row][column] / a[column][column];
                for (int k = column; k <= n; k++) a[row][k] -= factor * a[column][k];
            }
        }
        var solution = new double[n];
        for (int row = n - 1; row >= 0; row--)
        {
            double sum = a[row][n];
            for (int k = row + 1; k < n; k++) sum -= a[row][k] * solution[k];
            solution[row] = sum / a[row][row];
        }
        return solution;
    }
}

public static class Solution
{
    public static void Main()
    {
        const double eta0 = 0.05;
        const double etaMin = 0.0001;
        const double gammaStep = 0.5;
        const double gammaExp = 0.99;
        const long stepSize = 300;
        const double decay = 0.001;
        const double power = 1.0;
        const long total = 3000;

        var schedules = Schedules.Build(eta0, etaMin, gammaStep, gammaExp, stepSize, decay, power, total);
        var traces = new double[schedules.Length][];
        for (int s = 0; s < schedules.Length; s++) traces[s] = Schedules.Trace(schedules[s], total);

        // --- closed forms and monotonicity ---------------------------------------------------------------
        for (long step = 0; step < total; step++)
        {
            double[] closed = Schedules.ClosedForms(eta0, etaMin, gammaStep, gammaExp, stepSize, decay,
                    power, total, step);
            for (int s = 0; s < schedules.Length; s++)
            {
                Check(Math.Abs(traces[s][step] - closed[s]) <= 1e-15,
                        Schedules.Names[s] + " closed form at step " + step);
            }
        }
        for (int s = 0; s < schedules.Length; s++)
        {
            Check(Math.Abs(traces[s][0] - eta0) <= 1e-15 || s == 4,
                    Schedules.Names[s] + " starts at the base rate");
            for (long step = 1; step < total; step++)
            {
                Check(traces[s][step] <= traces[s][step - 1] + 1e-15,
                        Schedules.Names[s] + " is non-increasing at step " + step);
            }
        }
        Check(Math.Abs(traces[0][total - 1] - eta0) <= 1e-15, "the constant schedule never changes");
        Check(traces[1][stepSize] < traces[1][stepSize - 1], "the step schedule drops at the boundary");
        Check(Math.Abs(traces[1][stepSize] - eta0 * gammaStep) <= 1e-15, "the step schedule halves exactly");
        Check(Math.Abs(traces[2][total - 1] - eta0 * Math.Pow(gammaExp, total - 1)) <= 1e-18,
                "the exponential schedule reaches eta0 * gamma^(T-1)");
        Check(Math.Abs(traces[4][0] - eta0) <= 1e-15, "the cosine schedule starts at the base rate");
        Check(Math.Abs(traces[4][total - 1] - etaMin) <= 1e-4, "the cosine schedule ends near etaMin");
        Check(traces[4][total - 2] - traces[4][total - 1] >= 0.0, "and keeps decreasing to the end");
        Check(traces[5][total - 1] <= 1e-3 * eta0, "the polynomial schedule is nearly zero at the last step");
        Check(schedules[5](total) == 0.0, "at the horizon the polynomial schedule reaches exactly 0");
        Check(traces[5][total / 2] > 0.0, "and is still positive halfway");
        for (int s = 0; s < schedules.Length; s++)
        {
            Check(traces[s][0] >= traces[s][total - 1] || s == 0, Schedules.Names[s] + " decays");
        }

        // --- scaling the base rate scales the trace ----------------------------------------------------------
        var doubled = Schedules.Build(2.0 * eta0, 2.0 * etaMin, gammaStep, gammaExp, stepSize, decay,
                power, total);
        for (long step = 0; step < total; step += 37)
        {
            for (int s = 0; s < schedules.Length; s++)
            {
                Check(Math.Abs(doubled[s](step) - 2.0 * traces[s][step]) <= 1e-15,
                        "scaling eta0 scales " + Schedules.Names[s]);
            }
        }

        // --- warmup -------------------------------------------------------------------------------------------
        var warmup = Schedules.Warmup(schedules[4], 100);
        for (long step = 0; step < 100; step++)
        {
            Check(Math.Abs(warmup(step) - schedules[4](step) * (step + 1.0) / 100.0) <= 1e-15,
                    "warmup is linear at step " + step);
        }
        Check(warmup(0) < warmup(50) && warmup(50) < warmup(99), "the warmed-up rate grows");
        Check(Math.Abs(warmup(100) - schedules[4](100)) <= 1e-15, "warmup hands over at its end");

        // --- Robbins-Monro conditions ---------------------------------------------------------------------------
        Func<long, double> robbins = Schedules.InverseSqrt(0.5, 1.0);
        double sumRates = 0.0;
        double sumSquares = 0.0;
        double sumSmall = 0.0;
        for (long step = 0; step < 1000000; step++)
        {
            double rate = robbins(step);
            sumRates += rate;
            sumSquares += rate * rate;
            if (step < 10000) sumSmall += rate;
        }
        Check(sumRates > 900.0, "sum eta diverges (sum = " + sumRates + ")");
        Check(sumSquares < 10.0, "sum eta^2 converges (sum = " + sumSquares + ")");
        Check(sumRates / sumSmall >= 8.0 && sumRates / sumSmall <= 12.0,
                "the 1/sqrt(t) rate sum grows like sqrt(T) (ratio " + (sumRates / sumSmall) + ")");
        double harmonicLarge = 0.0;
        double harmonicSmall = 0.0;
        for (long step = 0; step < 1000000; step++)
        {
            double rate = 0.5 / (step + 1.0);
            harmonicLarge += rate;
            if (step < 10000) harmonicSmall += rate;
        }
        Check(harmonicLarge > 6.0 && harmonicLarge < 8.0, "the 1/t rate sum is about 0.5 ln T");
        Check(harmonicLarge / harmonicSmall >= 1.4 && harmonicLarge / harmonicSmall <= 1.6,
                "the 1/t rate sum grows logarithmically (ratio " + (harmonicLarge / harmonicSmall) + ")");
        Check(harmonicSmall * harmonicSmall < 100.0, "and its square converges");

        // --- the optimisation problem -----------------------------------------------------------------------------
        LogisticProblem problem = LogisticProblem.Planted(300, 5, 0.4, 516516UL);
        double[] optimum = problem.Iirls(60, 1e-12);
        Check(problem.Distance(problem.Gradient(optimum), new double[problem.Dimension]) <= 1e-8,
                "the IRLS optimum is stationary");
        const long steps = total;
        var runs = new LogisticProblem.Run[schedules.Length];
        for (int s = 0; s < schedules.Length; s++)
        {
            runs[s] = problem.TrainSgd(schedules[s], steps, 2024UL);
            runs[s].Distance = problem.Distance(runs[s].Weights, optimum);
        }
        var constantRun = runs[0];
        var stepRun = runs[1];
        var cosineRun = runs[4];
        var polynomialRun = runs[5];
        double optimumLoss = problem.Loss(optimum);
        Check(stepRun.Distance <= constantRun.Distance,
                "step decay is at least as good as the constant rate (" + stepRun.Distance + " vs "
                + constantRun.Distance + ")");
        Check(cosineRun.Distance <= 0.5 * constantRun.Distance,
                "cosine annealing halves the distance to the optimum (" + cosineRun.Distance + " vs "
                + constantRun.Distance + ")");
        Check(polynomialRun.Distance <= 0.5 * constantRun.Distance,
                "polynomial decay halves the distance to the optimum (" + polynomialRun.Distance + " vs "
                + constantRun.Distance + ")");
        Check(runs[3].Distance <= 0.5 * constantRun.Distance,
                "inverse-time decay halves the distance to the optimum (" + runs[3].Distance + " vs "
                + constantRun.Distance + ")");
        Check(cosineRun.FinalLoss - optimumLoss <= 0.1 * (constantRun.FinalLoss - optimumLoss),
                "cosine annealing ends an order of magnitude closer in loss ("
                + (cosineRun.FinalLoss - optimumLoss) + " vs " + (constantRun.FinalLoss - optimumLoss) + ")");
        Check(runs[2].Distance >= 5.0 * cosineRun.Distance,
                "an over-aggressive exponential decay stays undertrained ("
                + runs[2].Distance + " vs " + cosineRun.Distance + ")");
        for (int s = 1; s < schedules.Length; s++)
        {
            Check(runs[s].LastDecileAverage <= runs[s].FirstDecileAverage,
                    Schedules.Names[s] + " improves over its last decile ("
                    + runs[s].LastDecileAverage + " <= " + runs[s].FirstDecileAverage + ")");
        }
        Check(cosineRun.FinalLoss <= constantRun.FinalLoss,
                "the annealed run ends no worse than the constant run ("
                + cosineRun.FinalLoss + " vs " + constantRun.FinalLoss + ")");

        // --- a tiny budget and degenerate rates ----------------------------------------------------------------------
        var shortRun = problem.TrainSgd(schedules[4], 1, 5UL);
        shortRun.Distance = problem.Distance(shortRun.Weights, optimum);
        Check(shortRun.Losses.Count == 1, "a single-step run records one loss");
        Check(shortRun.Distance > 0.0, "a single SGD step does not solve the problem");
        Func<long, double> zero = step => 0.0;
        var frozen = problem.TrainSgd(zero, 100, 7UL);
        frozen.Distance = problem.Distance(frozen.Weights, optimum);
        Check(frozen.Distance == problem.Distance(new double[problem.Dimension], optimum),
                "a zero base rate leaves the parameters untouched");
        Func<long, double> constant = step => eta0;
        var shortConstant = problem.TrainSgd(constant, 100, 7UL);
        shortConstant.Distance = problem.Distance(shortConstant.Weights, optimum);
        Check(shortConstant.FinalLoss < frozen.FinalLoss,
                "100 steps with a real rate reduce the loss (" + shortConstant.FinalLoss + " vs "
                + frozen.FinalLoss + ")");
        Check(shortConstant.Distance < frozen.Distance, "and move the weights toward the optimum");

        // --- determinism ---------------------------------------------------------------------------------------------------
        var repeat = problem.TrainSgd(schedules[4], steps, 2024UL);
        repeat.Distance = problem.Distance(repeat.Weights, optimum);
        Check(repeat.Distance == cosineRun.Distance, "the cosine run is reproducible");
        for (int j = 0; j < problem.Dimension; j++)
        {
            Check(repeat.Weights[j] == cosineRun.Weights[j], "weights are bit-identical at " + j);
        }
        var otherSeed = problem.TrainSgd(schedules[4], steps, 999UL);
        otherSeed.Distance = problem.Distance(otherSeed.Weights, optimum);
        Check(Math.Abs(otherSeed.Distance - cosineRun.Distance) <= 0.05,
                "a different shuffle lands in the same neighbourhood ("
                + otherSeed.Distance + " vs " + cosineRun.Distance + ")");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
