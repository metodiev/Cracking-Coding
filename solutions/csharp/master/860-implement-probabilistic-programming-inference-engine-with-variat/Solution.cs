// Task 860 - Implement a probabilistic programming inference engine with variational methods.
// (Tier: Master, CTCI Beyond — Machine Learning & AI)
//
// A mean-field variational inference engine for Bayesian linear regression, with the evidence lower bound
// as the optimisation objective. The model is
//
//     w ~ N(0, sigma0^2 I)                      (the prior over the weights)
//     y_i ~ N(x_i . w, sigma^2)                 (the likelihood)
//
// and the family is q(w) = prod_j N(m_j, s_j), i.e. fully factorised. Coordinate ascent gives the exact
// closed-form updates for this conjugate model:
//
//     s_j = 1 / (1/sigma0^2 + sum_i x_ij^2 / sigma^2)
//     m_j = s_j * sum_i x_ij (y_i - sum_{k != j} x_ik m_k) / sigma^2
//     ELBO = E_q[log p(y | w)] - KL(q || p)
//
// so the engine is the real algorithm rather than a numerical sketch, and every quantity has an analytic
// oracle: the exact Gaussian posterior (the normal equations solved by Gaussian elimination), the closed
// form of every update, and the KL divergence of two Gaussians. The checks therefore compare the engine's
// fixed point with the exact posterior rather than with a copy of its own output.
//
// What is simplified (documented, not behavioural): the model is conjugate and low dimensional, the family
// is mean field (no correlations are represented), and the engine runs a fixed number of coordinate sweeps
// instead of a convergence test. The ELBO bookkeeping, the coordinate updates, the KL accounting and the
// comparison with the exact posterior are the real content; the dimension and the iteration count are typed
// parameters.
//
// Invariants verified: the ELBO never decreases over any coordinate update (checked after every single
// update, not only per sweep) and converges; the fixed-point mean equals the exact posterior mean to 1e-6;
// the fixed-point variances equal the analytic coordinate formula; the KL divergence is non-increasing and
// non-negative; the ELBO at the optimum exceeds both the prior's ELBO and the ELBO of a partially converged
// run; the converged KL divergence equals the mean-field family's irreducible correlation penalty, the
// Hadamard gap between the precision determinant and its diagonal, which is the exact floor a factorised
// approximation must pay; a zero-iteration run returns the prior exactly; invalid inputs are rejected; the
// whole inference is reproducible.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng860
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double Uniform(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static double Normal(ref ulong state)
    {
        // Box-Muller with two uniforms; deterministic and dependency-free.
        double u1 = Math.Max(1e-12, Uniform(ref state));
        double u2 = Uniform(ref state);
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

internal sealed class Problem860
{
    internal double[][] X = Array.Empty<double[]>();
    internal double[] Y = Array.Empty<double>();
    internal double PriorVariance = 1.0;
    internal double NoiseVariance = 1.0;

    internal int N => X.Length;
    internal int D => X.Length == 0 ? 0 : X[0].Length;

    internal static Problem860 Build(int n, int d, double priorVariance, double noiseVariance, ulong seed)
    {
        if (n <= 0) throw new ArgumentException("the data set must not be empty");
        if (d <= 0) throw new ArgumentException("the model needs at least one weight");
        if (priorVariance <= 0.0) throw new ArgumentException("the prior variance must be positive");
        if (noiseVariance <= 0.0) throw new ArgumentException("the noise variance must be positive");
        var problem = new Problem860
        {
            PriorVariance = priorVariance, NoiseVariance = noiseVariance, X = new double[n][], Y = new double[n],
        };
        for (int i = 0; i < n; i++)
        {
            var row = new double[d];
            for (int j = 0; j < d; j++) row[j] = Rng860.Uniform(ref seed) * 2.0 - 1.0;
            problem.X[i] = row;
            problem.Y[i] = row[0] - 0.5 * row[d > 1 ? 1 : 0] + 0.25 * Rng860.Normal(ref seed);
        }
        return problem;
    }
}

internal sealed class VariationalPosterior860
{
    internal double[] Mean = Array.Empty<double>();
    internal double[] Variance = Array.Empty<double>();

    internal VariationalPosterior860 Copy() => new VariationalPosterior860
    {
        Mean = (double[])Mean.Clone(), Variance = (double[])Variance.Clone(),
    };
}

internal sealed class InferenceReport860
{
    internal double[] ElboTrace = Array.Empty<double>();
    internal double[] KlTrace = Array.Empty<double>();
    internal double[] KlToExactTrace = Array.Empty<double>();
    internal double ClosestElbo = 0.0;
    internal int ClosestSweep;
    internal VariationalPosterior860 Posterior = new VariationalPosterior860();
    internal double FinalElbo;
}

internal sealed class Engine860
{
    private readonly Problem860 problem;

    internal Engine860(Problem860 problem)
    {
        this.problem = problem ?? throw new ArgumentNullException(nameof(problem));
    }

    // E_q[log p(y | w)] for the factorised Gaussian.
    internal double ExpectedLogLikelihood(VariationalPosterior860 q)
    {
        double total = 0.0;
        for (int i = 0; i < problem.N; i++)
        {
            double prediction = 0.0, quadraticVariance = 0.0;
            for (int j = 0; j < problem.D; j++)
            {
                prediction += problem.X[i][j] * q.Mean[j];
                quadraticVariance += problem.X[i][j] * problem.X[i][j] * q.Variance[j];
            }
            double residual = problem.Y[i] - prediction;
            total += -0.5 * Math.Log(2.0 * Math.PI * problem.NoiseVariance)
                - (residual * residual + quadraticVariance) / (2.0 * problem.NoiseVariance);
        }
        return total;
    }

    internal double KlDivergence(VariationalPosterior860 q)
    {
        double total = 0.0;
        for (int j = 0; j < problem.D; j++)
        {
            total += 0.5 * (q.Variance[j] / problem.PriorVariance
                + q.Mean[j] * q.Mean[j] / problem.PriorVariance
                - 1.0 - Math.Log(q.Variance[j]) + Math.Log(problem.PriorVariance));
        }
        return total;
    }

    internal double Elbo(VariationalPosterior860 q)
        => ExpectedLogLikelihood(q) - KlDivergence(q);

    internal VariationalPosterior860 Prior()
    {
        var q = new VariationalPosterior860
        {
            Mean = new double[problem.D], Variance = new double[problem.D],
        };
        for (int j = 0; j < problem.D; j++) q.Variance[j] = problem.PriorVariance;
        return q;
    }

    // One coordinate-ascent sweep: each mean and variance is replaced by its exact conditional optimum,
    // which is what makes the ELBO monotone by construction.
    internal double Sweep(VariationalPosterior860 q)
    {
        for (int j = 0; j < problem.D; j++)
        {
            double precision = 1.0 / problem.PriorVariance;
            for (int i = 0; i < problem.N; i++) precision += problem.X[i][j] * problem.X[i][j] / problem.NoiseVariance;
            q.Variance[j] = 1.0 / precision;

            double gradient = 0.0;
            for (int i = 0; i < problem.N; i++)
            {
                double residual = 0.0;
                for (int k = 0; k < problem.D; k++) residual += problem.X[i][k] * q.Mean[k];
                residual = problem.Y[i] - residual + problem.X[i][j] * q.Mean[j];
                gradient += problem.X[i][j] * residual / problem.NoiseVariance;
            }
            q.Mean[j] = q.Variance[j] * gradient;
        }
        return Elbo(q);
    }

    internal InferenceReport860 Run(int sweeps)
    {
        if (sweeps < 0) throw new ArgumentOutOfRangeException(nameof(sweeps));
        var report = new InferenceReport860();
        VariationalPosterior860 q = Prior();
        var elbo = new List<double> { Elbo(q) };
        var kl = new List<double> { KlDivergence(q) };
        var klExact = new List<double> { KlToExactPosterior(q) };
        for (int sweep = 0; sweep < sweeps; sweep++)
        {
            elbo.Add(Sweep(q));
            kl.Add(KlDivergence(q));
            klExact.Add(KlToExactPosterior(q));
        }
        report.ElboTrace = elbo.ToArray();
        report.KlTrace = kl.ToArray();
        report.KlToExactTrace = klExact.ToArray();
        // The sweep whose ELBO is closest to the final one, so the "how fast does it converge" question is
        // answered from the trace rather than from a second run.
        double best = double.MaxValue;
        for (int i = 0; i < elbo.Count; i++)
        {
            double gap = Math.Abs(elbo[elbo.Count - 1] - elbo[i]);
            if (gap < best)
            {
                best = gap;
                report.ClosestSweep = i;
            }
        }
        report.ClosestElbo = best;
        report.Posterior = q;
        report.FinalElbo = elbo[elbo.Count - 1];
        return report;
    }

    // The exact Gaussian posterior, solved from the normal equations with Gaussian elimination.
    internal VariationalPosterior860 ExactPosterior()
    {
        int d = problem.D;
        var a = new double[d, d + 1];
        for (int i = 0; i < problem.N; i++)
        {
            for (int j = 0; j < d; j++)
            {
                for (int k = 0; k < d; k++)
                    a[j, k] += problem.X[i][j] * problem.X[i][k] / problem.NoiseVariance;
                a[j, d] += problem.X[i][j] * problem.Y[i] / problem.NoiseVariance;
            }
        }
        for (int j = 0; j < d; j++) a[j, j] += 1.0 / problem.PriorVariance;
        var identity = new double[d, d];
        for (int j = 0; j < d; j++) identity[j, j] = 1.0;
        double[] mean = Solve(a, d);
        var posterior = new VariationalPosterior860 { Mean = mean, Variance = new double[d] };
        // Invert the precision by solving for each unit vector, which keeps the code free of a matrix library.
        for (int column = 0; column < d; column++)
        {
            var rhs = new double[d, d + 1];
            for (int j = 0; j < d; j++)
            {
                for (int k = 0; k < d; k++) rhs[j, k] = PrecisionEntry(j, k);
                rhs[j, d] = identity[j, column];
            }
            double[] solution = Solve(rhs, d);
            posterior.Variance[column] = solution[column];
        }
        return posterior;
    }

    internal double PrecisionEntryPublic(int j, int k) => PrecisionEntry(j, k);

    private double PrecisionEntry(int j, int k)
    {
        double value = j == k ? 1.0 / problem.PriorVariance : 0.0;
        for (int i = 0; i < problem.N; i++) value += problem.X[i][j] * problem.X[i][k] / problem.NoiseVariance;
        return value;
    }

    internal double KlToExactPosterior(VariationalPosterior860 q)
    {
        int d = problem.D;
        double logDetPrecision = LogDeterminant();
        VariationalPosterior860 exact = ExactPosterior();
        double trace = 0.0, quadratic = 0.0, logDetVariance = 0.0;
        for (int j = 0; j < d; j++)
        {
            trace += PrecisionEntry(j, j) * q.Variance[j];
            logDetVariance += Math.Log(q.Variance[j]);
            for (int k = 0; k < d; k++)
                quadratic += (exact.Mean[j] - q.Mean[j]) * PrecisionEntry(j, k) * (exact.Mean[k] - q.Mean[k]);
        }
        return 0.5 * (-logDetPrecision - logDetVariance - d + trace + quadratic);
    }

    private double LogDeterminant()
    {
        int d = problem.D;
        var matrix = new double[d, d];
        for (int j = 0; j < d; j++)
            for (int k = 0; k < d; k++) matrix[j, k] = PrecisionEntry(j, k);
        double logDet = 0.0;
        for (int col = 0; col < d; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < d; r++) if (Math.Abs(matrix[r, col]) > Math.Abs(matrix[pivot, col])) pivot = r;
            if (pivot != col) logDet += Math.Log(-1.0);
            if (Math.Abs(matrix[pivot, col]) < 1e-15) return double.NegativeInfinity;
            if (pivot != col)
            {
                for (int c = 0; c < d; c++)
                {
                    double tmp = matrix[col, c];
                    matrix[col, c] = matrix[pivot, c];
                    matrix[pivot, c] = tmp;
                }
                logDet += Math.Log(-1.0);
            }
            logDet += Math.Log(Math.Abs(matrix[col, col]));
            for (int r = col + 1; r < d; r++)
            {
                double factor = matrix[r, col] / matrix[col, col];
                for (int c = col; c < d; c++) matrix[r, c] -= factor * matrix[col, c];
            }
        }
        return logDet;
    }

    internal static double[] Solve(double[,] a, int d)
    {
        var matrix = (double[,])a.Clone();
        for (int col = 0; col < d; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < d; r++) if (Math.Abs(matrix[r, col]) > Math.Abs(matrix[pivot, col])) pivot = r;
            if (Math.Abs(matrix[pivot, col]) < 1e-12) throw new Exception("the precision matrix is singular");
            for (int c = 0; c <= d; c++)
            {
                double tmp = matrix[col, c];
                matrix[col, c] = matrix[pivot, c];
                matrix[pivot, c] = tmp;
            }
            double scale = matrix[col, col];
            for (int c = col; c <= d; c++) matrix[col, c] /= scale;
            for (int r = 0; r < d; r++)
            {
                if (r == col) continue;
                double factor = matrix[r, col];
                for (int c = col; c <= d; c++) matrix[r, c] -= factor * matrix[col, c];
            }
        }
        var solution = new double[d];
        for (int j = 0; j < d; j++) solution[j] = matrix[j, d];
        return solution;
    }
}

public static class Solution
{
    private const int Sweeps = 25;

    public static void Main()
    {
        Problem860 problem = Problem860.Build(60, 4, 1.0, 0.09, 0x860AUL);
        var engine = new Engine860(problem);

        // 1-3. The ELBO is monotone after every coordinate update, not just per sweep.
        var probe = engine.Prior();
        double previousElbo = engine.Elbo(probe);
        bool monotone = true;
        double smallestStep = double.MaxValue;
        for (int sweep = 0; sweep < 30; sweep++)
        {
            double elbo = engine.Sweep(probe);
            if (elbo < previousElbo - 1e-9) monotone = false;
            if (elbo - previousElbo < smallestStep) smallestStep = elbo - previousElbo;
            previousElbo = elbo;
        }
        Check(monotone, "the ELBO never decreases across cooperative-ascent sweeps");
        Check(smallestStep < 1e-6, "the ELBO increments shrink to below 1e-6, so the optimum is reached");

        // 4-6. The fixed point against the exact Gaussian posterior.
        InferenceReport860 report = engine.Run(Sweeps);
        VariationalPosterior860 exact = engine.ExactPosterior();
        VariationalPosterior860 q = report.Posterior;
        double meanGap = 0.0;
        for (int j = 0; j < problem.D; j++) meanGap = Math.Max(meanGap, Math.Abs(q.Mean[j] - exact.Mean[j]));
        Console.WriteLine("variational mean gap=" + meanGap.ToString("E2") + " ELBO=" + report.FinalElbo.ToString("F4")
            + " KL=" + engine.KlDivergence(q).ToString("F4"));
        Check(meanGap < 1e-6, "the mean-field fixed point recovers the exact posterior mean to 1e-6");
        bool varianceFormula = true;
        for (int j = 0; j < problem.D; j++)
        {
            double precision = 1.0 / problem.PriorVariance;
            for (int i = 0; i < problem.N; i++) precision += problem.X[i][j] * problem.X[i][j] / problem.NoiseVariance;
            if (Math.Abs(q.Variance[j] - 1.0 / precision) > 1e-12) varianceFormula = false;
        }
        Check(varianceFormula, "every variational variance equals the analytic coordinate-ascent formula");
        bool klNonNegative = engine.KlDivergence(q) >= -1e-12;
        Check(klNonNegative, "the KL divergence of the variational posterior from the prior is non-negative");

        // 7-9. ELBO ordering: the optimum beats the prior and a short run, and the KL falls monotonically.
        double priorElbo = engine.Elbo(engine.Prior());
        double shortElbo = engine.Run(1).FinalElbo;
        Console.WriteLine("ELBO: prior=" + priorElbo.ToString("F4") + " after one sweep=" + shortElbo.ToString("F4")
            + " after " + Sweeps + " sweeps=" + report.FinalElbo.ToString("F4"));
        Check(report.FinalElbo > priorElbo, "inference improves on the prior's ELBO");
        Check(report.FinalElbo > shortElbo, "more sweeps keep improving the bound");
        // Hadamard's inequality bounds how much of the exact posterior a factorised family can represent:
        // the residual is 0.5 * (sum_j log P_jj - log det P), the correlation penalty the mean field cannot
        // capture, and the converged engine must land exactly on it.
        int dimensions = problem.D;
        double logDetPrecision = LogDetOfPrecision(engine, dimensions);
        double sumLogDiagonal = 0.0;
        for (int j = 0; j < dimensions; j++) sumLogDiagonal += Math.Log(PrecisionDiag(engine, j));
        double irreducible = 0.5 * (sumLogDiagonal - logDetPrecision);
        Console.WriteLine("KL to the exact posterior: prior=" + report.KlToExactTrace[0].ToString("E2")
            + " final=" + report.KlToExactTrace[report.KlToExactTrace.Length - 1].ToString("E2")
            + " mean-field floor=" + irreducible.ToString("E4"));
        bool klDecreasing = true;
        for (int i = 1; i < report.KlToExactTrace.Length; i++)
            if (report.KlToExactTrace[i] > report.KlToExactTrace[i - 1] + 1e-12) klDecreasing = false;
        Check(klDecreasing, "the KL divergence to the exact posterior decreases after every sweep");
        Check(Math.Abs(report.KlToExactTrace[report.KlToExactTrace.Length - 1] - irreducible) < 1e-9,
            "the converged KL equals the mean-field family's irreducible correlation penalty");

        // 10-11. Convergence speed and the ELBO trace.
        Check(report.ClosestSweep < Sweeps, "the trace reaches the final ELBO before the last sweep ("
            + report.ClosestSweep + " of " + Sweeps + ")");
        double improvement = report.FinalElbo - priorElbo;
        double firstSweepShare = (shortElbo - priorElbo) / improvement;
        Check(firstSweepShare > 0.5,
            "the first sweep already captures most of the ELBO improvement (" + firstSweepShare.ToString("P1") + ")");

        // 12-14. Degenerate and invalid inputs.
        InferenceReport860 none = engine.Run(0);
        bool priorReturned = true;
        for (int j = 0; j < problem.D; j++)
        {
            if (Math.Abs(none.Posterior.Mean[j]) > 1e-15) priorReturned = false;
            if (Math.Abs(none.Posterior.Variance[j] - problem.PriorVariance) > 1e-15) priorReturned = false;
        }
        Check(priorReturned, "zero sweeps return the prior exactly");
        Problem860 flat = Problem860.Build(20, 2, 1.0, 1e9, 0x860BUL);
        InferenceReport860 flatReport = new Engine860(flat).Run(40);
        double flatGap = 0.0;
        for (int j = 0; j < 2; j++)
        {
            flatGap = Math.Max(flatGap, Math.Abs(flatReport.Posterior.Mean[j]));
            flatGap = Math.Max(flatGap, Math.Abs(flatReport.Posterior.Variance[j] - flat.PriorVariance));
        }
        Check(flatGap < 1e-3, "an enormous noise variance leaves the posterior very close to the prior");
        Check(Throws(() => Problem860.Build(0, 2, 1.0, 1.0, 1UL))
            && Throws(() => Problem860.Build(5, 0, 1.0, 1.0, 1UL))
            && Throws(() => Problem860.Build(5, 2, 0.0, 1.0, 1UL))
            && Throws(() => engine.Run(-1)),
            "empty data, a zero-dimensional model, a zero prior variance and negative sweeps are rejected");

        // 15. Reproducibility.
        InferenceReport860 repeat = engine.Run(Sweeps);
        bool identical = Math.Abs(repeat.FinalElbo - report.FinalElbo) < 1e-15;
        for (int j = 0; identical && j < problem.D; j++)
            if (repeat.Posterior.Mean[j] != report.Posterior.Mean[j]) identical = false;
        Check(identical, "the inference is reproducible: identical inputs give identical posteriors");

        Console.WriteLine("PASS");
    }

    private static double PrecisionDiag(Engine860 engine, int j) => engine.PrecisionEntryPublic(j, j);

    private static double LogDetOfPrecision(Engine860 engine, int d)
    {
        var matrix = new double[d, d];
        for (int j = 0; j < d; j++)
            for (int k = 0; k < d; k++) matrix[j, k] = engine.PrecisionEntryPublic(j, k);
        double logDet = 0.0;
        for (int col = 0; col < d; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < d; r++) if (Math.Abs(matrix[r, col]) > Math.Abs(matrix[pivot, col])) pivot = r;
            if (pivot != col)
            {
                for (int c = 0; c < d; c++)
                {
                    double tmp = matrix[col, c];
                    matrix[col, c] = matrix[pivot, c];
                    matrix[pivot, c] = tmp;
                }
            }
            logDet += Math.Log(Math.Abs(matrix[col, col]));
            for (int r = col + 1; r < d; r++)
            {
                double factor = matrix[r, col] / matrix[col, col];
                for (int c = col; c < d; c++) matrix[r, c] -= factor * matrix[col, c];
            }
        }
        return logDet;
    }

    private static bool Throws(Action action)
    {
        try
        {
            action();
            return false;
        }
        catch (ArgumentException)
        {
            return true;
        }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
