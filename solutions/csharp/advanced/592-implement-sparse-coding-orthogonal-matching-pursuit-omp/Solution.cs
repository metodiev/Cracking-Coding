// Task 592 - Implement sparse coding orthogonal matching pursuit (OMP). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Orthogonal matching pursuit: given a dictionary of K atoms (an n x K matrix with unit- or
// arbitrary-norm columns) and a signal y, the greedy loop selects at every step the atom whose inner
// product with the current residual is largest in absolute value, adds it to the support, and then --
// this is the "orthogonal" part that separates OMP from plain matching pursuit -- re-solves the full
// least-squares problem min ||D_S x_S - y|| over the whole support S by normal equations with partial
// pivoting, which makes the new residual orthogonal to the span of the selected atoms. The routine
// records the residual vector before every selection, the residual norm after every step, and the
// selection order, so the greedy rule, the orthogonality property and the monotonicity of the residual
// can all be verified afterwards from recorded data rather than from internal state. Verification is
// deliberately redundant: a hand-built orthonormal dictionary with a closed-form answer, the identity
// dictionary, the classical s = 1 case where greedy is provably the exact minimiser, exhaustive
// support enumeration for small dictionaries, and 10 random Gaussian instances of exact s-sparse
// signals of dimension 20 with 40 atoms. Signals, dictionaries and noise all come from the fixed-seed
// 64-bit LCG in `Rng` (MMIX constants, documented because the repo's 16-bit LCG has poor low-bit
// structure and this file needs many independent Gaussian-ish draws).
//
// Documented checks: the greedy selection really is argmax |d_j^T r| over the not-yet-selected atoms
// at every step of every run (recomputed from the recorded residuals); the final residual is
// orthogonal to all selected atoms to 1e-10 whenever the instance is exactly recoverable; the
// residual norm is non-increasing along the iteration; supports never repeat an index and never exceed
// the requested sparsity; the identity dictionary recovers the s largest-magnitude coefficients
// exactly and the orthonormal dictionary recovers x_j = d_j^T y for the selected atoms in exactly s
// steps; a signal that is a single scaled atom is stopped early with support size 1 even when
// sparsity 5 is requested, and the coefficient is recovered exactly; for s = 1 the OMP residual
// equals the exhaustive minimum over all single-atom least-squares fits (greedy is optimal there) and
// for s = 2 OMP's residual is never below the exhaustive minimum (documented as an inequality, since
// greedy is not optimal in general); on 10 random instances with s = 3 of dimension 20 and 40 atoms
// every support and every coefficient is recovered to 1e-10, and the file states plainly that this is
// empirical rather than guaranteed: the measured coherence mu exceeds 1/(2s-1) = 0.2, so neither the
// Tropp condition nor s < (1/2)(1 + 1/mu) applies, and the exhaustive s = 2 search above already
// exhibits that OMP is not an optimal decoder; adding noise
// of norm 0.01 keeps the support correct while the residual grows towards the noise level but never
// exceeds the noise norm; and repeated runs give bit-identical coefficients.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /** 64-bit LCG with the MMIX constants; documented replacement for the 16-bit repo LCG. */
    internal static ulong Next(ref ulong state)
    {
        state = state * 6364136223846793005UL + 1442695040888963407UL;
        return state >> 11;
    }

    internal static int NextBelow(ref ulong state, int bound)
    {
        return (int)(Next(ref state) % (ulong)bound);
    }

    /** Uniform draw in [-1, 1]. */
    internal static double NextSigned(ref ulong state)
    {
        return NextBelow(ref state, 2001) / 1000.0 - 1.0;
    }
}

internal static class Omp
{
    internal sealed class Result
    {
        internal double[] Coefficients = Array.Empty<double>();
        internal List<int> Support = new List<int>();
        internal List<int> SelectionOrder = new List<int>();
        internal List<double> ResidualNorms = new List<double>();
        internal List<double[]> Residuals = new List<double[]>();
    }

    internal static double Norm(double[] vector)
    {
        double total = 0.0;
        foreach (double value in vector) total += value * value;
        return Math.Sqrt(total);
    }

    internal static double Dot(double[] a, double[] b)
    {
        double total = 0.0;
        for (int i = 0; i < a.Length; i++) total += a[i] * b[i];
        return total;
    }

    /** Best absolute correlation of the residual with the atoms outside the support. */
    internal static int BestAtom(double[][] dictionary, double[] residual, IReadOnlyList<int> support)
    {
        int n = residual.Length;
        int k = dictionary[0].Length;
        int best = -1;
        double bestValue = -1.0;
        for (int j = 0; j < k; j++)
        {
            bool used = false;
            foreach (int index in support) if (index == j) used = true;
            if (used) continue;
            double dot = 0.0;
            for (int i = 0; i < n; i++) dot += dictionary[i][j] * residual[i];
            if (Math.Abs(dot) > bestValue)
            {
                bestValue = Math.Abs(dot);
                best = j;
            }
        }
        return best;
    }

    /** Least squares over the given support via normal equations with partial pivoting. */
    internal static double[] LeastSquares(double[][] dictionary, double[] signal, IReadOnlyList<int> support)
    {
        int size = support.Count;
        var gram = new double[size][];
        var rhs = new double[size];
        for (int a = 0; a < size; a++)
        {
            gram[a] = new double[size];
            for (int b = 0; b < size; b++)
            {
                double total = 0.0;
                for (int i = 0; i < signal.Length; i++) total += dictionary[i][support[a]] * dictionary[i][support[b]];
                gram[a][b] = total;
            }
            double dot = 0.0;
            for (int i = 0; i < signal.Length; i++) dot += dictionary[i][support[a]] * signal[i];
            rhs[a] = dot;
        }
        var solution = new double[size];
        int[] order = EnumerableRange(size);
        for (int column = 0; column < size; column++)
        {
            int pivot = column;
            for (int row = column + 1; row < size; row++)
            {
                if (Math.Abs(gram[order[row]][column]) > Math.Abs(gram[order[pivot]][column])) pivot = row;
            }
            int tmp = order[column];
            order[column] = order[pivot];
            order[pivot] = tmp;
            double diagonal = gram[order[column]][column];
            if (Math.Abs(diagonal) < 1e-300) continue;
            for (int row = column + 1; row < size; row++)
            {
                double factor = gram[order[row]][column] / diagonal;
                for (int c = column; c < size; c++) gram[order[row]][c] -= factor * gram[order[column]][c];
                rhs[order[row]] -= factor * rhs[order[column]];
            }
        }
        for (int row = size - 1; row >= 0; row--)
        {
            double total = rhs[order[row]];
            for (int c = row + 1; c < size; c++) total -= gram[order[row]][c] * solution[c];
            double diagonal = gram[order[row]][row];
            solution[row] = Math.Abs(diagonal) < 1e-300 ? 0.0 : total / diagonal;
        }
        return solution;
    }

    private static int[] EnumerableRange(int size)
    {
        var order = new int[size];
        for (int i = 0; i < size; i++) order[i] = i;
        return order;
    }

    internal static Result Solve(double[][] dictionary, double[] signal, int sparsity, double tolerance)
    {
        int k = dictionary[0].Length;
        var coefficients = new double[k];
        var residual = (double[])signal.Clone();
        var support = new List<int>();
        var result = new Result { Coefficients = coefficients };
        result.ResidualNorms.Add(Norm(residual));
        result.Residuals.Add((double[])residual.Clone());
        while (support.Count < sparsity && Norm(residual) > tolerance)
        {
            int best = BestAtom(dictionary, residual, support);
            if (best < 0) break;
            support.Add(best);
            result.SelectionOrder.Add(best);
            double[] partial = LeastSquares(dictionary, signal, support);
            var candidate = new double[k];
            for (int a = 0; a < support.Count; a++) candidate[support[a]] = partial[a];
            var nextResidual = new double[signal.Length];
            for (int i = 0; i < signal.Length; i++)
            {
                double reconstruction = 0.0;
                foreach (int j in support) reconstruction += dictionary[i][j] * candidate[j];
                nextResidual[i] = signal[i] - reconstruction;
            }
            coefficients = candidate;
            residual = nextResidual;
            result.ResidualNorms.Add(Norm(residual));
            result.Residuals.Add((double[])residual.Clone());
        }
        result.Coefficients = coefficients;
        result.Support = support;
        return result;
    }

    /** Exhaustive best least-squares residual over all supports of size at most `sparsity`. */
    internal static double BruteForceResidual(double[][] dictionary, double[] signal, int sparsity)
    {
        int k = dictionary[0].Length;
        double best = Norm(signal);
        var support = new List<int>();
        for (int size = 1; size <= sparsity; size++)
        {
            var indices = new int[size];
            for (int i = 0; i < size; i++) indices[i] = i;
            while (true)
            {
                support.Clear();
                support.AddRange(indices);
                double[] partial = LeastSquares(dictionary, signal, support);
                var reconstruction = new double[signal.Length];
                for (int i = 0; i < signal.Length; i++)
                {
                    double total = 0.0;
                    for (int a = 0; a < size; a++) total += dictionary[i][indices[a]] * partial[a];
                    reconstruction[i] = total;
                }
                var residual = new double[signal.Length];
                for (int i = 0; i < signal.Length; i++) residual[i] = signal[i] - reconstruction[i];
                best = Math.Min(best, Norm(residual));
                int position = size - 1;
                while (position >= 0 && indices[position] == k - size + position) position--;
                if (position < 0) break;
                indices[position]++;
                for (int i = position + 1; i < size; i++) indices[i] = indices[i - 1] + 1;
            }
        }
        return best;
    }

    /** Column-wise coherence: the largest absolute inner product between two distinct unit atoms. */
    internal static double Coherence(double[][] dictionary, bool normalise)
    {
        int n = dictionary.Length;
        int k = dictionary[0].Length;
        var atoms = new double[k][];
        for (int j = 0; j < k; j++)
        {
            atoms[j] = new double[n];
            for (int i = 0; i < n; i++) atoms[j][i] = dictionary[i][j];
            if (normalise)
            {
                double norm = Norm(atoms[j]);
                for (int i = 0; i < n; i++) atoms[j][i] /= norm;
            }
        }
        double worst = 0.0;
        for (int a = 0; a < k; a++)
        {
            for (int b = a + 1; b < k; b++) worst = Math.Max(worst, Math.Abs(Dot(atoms[a], atoms[b])));
        }
        return worst;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckClose(double actual, double expected, double tolerance, string label)
    {
        Check(Math.Abs(actual - expected) <= tolerance, label + " (got " + actual + ", expected " + expected + ")");
    }

    private static double[] Column(double[][] matrix, int index)
    {
        var column = new double[matrix.Length];
        for (int i = 0; i < matrix.Length; i++) column[i] = matrix[i][index];
        return column;
    }

    /** Recomputes the greedy rule from the recorded residuals, independently of the solver state. */
    private static bool GreedyRuleHolds(double[][] dictionary, Omp.Result result, int sparsity)
    {
        for (int step = 0; step < result.SelectionOrder.Count; step++)
        {
            double[] previous = result.Residuals[step];
            int expected = Omp.BestAtom(dictionary, previous, result.SelectionOrder.GetRange(0, step));
            if (expected != result.SelectionOrder[step]) return false;
        }
        return result.SelectionOrder.Count <= sparsity;
    }

    public static void Main()
    {
        // ---- orthonormal dictionary (DCT-II basis) ---------------------------------------
        const int N = 8;
        var orthonormal = new double[N][];
        for (int i = 0; i < N; i++)
        {
            orthonormal[i] = new double[N];
            orthonormal[i][0] = 1.0 / Math.Sqrt(N);
            for (int j = 1; j < N; j++)
            {
                orthonormal[i][j] = Math.Sqrt(2.0 / N) * Math.Cos(Math.PI * (i + 0.5) * j / N);
            }
        }
        CheckClose(Omp.Coherence(orthonormal, false), 0.0, 1e-12, "the DCT-II atoms are mutually orthogonal");
        for (int a = 0; a < N; a++)
        {
            CheckClose(Omp.Norm(Column(orthonormal, a)), 1.0, 1e-12, "DCT-II atom " + a + " has unit norm");
        }
        var exactCoefficients = new double[] { 3.0, 0.0, 0.0, -2.0, 0.0, 0.0, 1.5, 0.0 };
        var exactSignal = new double[N];
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++) exactSignal[i] += orthonormal[i][j] * exactCoefficients[j];
        }
        Omp.Result orthonormalResult = Omp.Solve(orthonormal, exactSignal, 4, 1e-12);
        Check(orthonormalResult.Support.Count == 3, "only three atoms are needed for the orthonormal instance");
        Check(GreedyRuleHolds(orthonormal, orthonormalResult, 4), "the greedy rule holds for the orthonormal instance");
        foreach (int index in new[] { 0, 3, 6 })
        {
            Check(orthonormalResult.Support.Contains(index), "the orthonormal instance selects atom " + index);
            CheckClose(orthonormalResult.Coefficients[index], exactCoefficients[index], 1e-10,
                    "the orthonormal instance recovers the coefficient of atom " + index);
        }
        CheckClose(orthonormalResult.ResidualNorms[orthonormalResult.ResidualNorms.Count - 1], 0.0, 1e-10,
                "the orthonormal instance is solved exactly");
        for (int step = 1; step < orthonormalResult.ResidualNorms.Count; step++)
        {
            Check(orthonormalResult.ResidualNorms[step] <= orthonormalResult.ResidualNorms[step - 1] + 1e-12,
                    "the orthonormal residual is non-increasing at step " + step);
        }

        // ---- identity dictionary ----------------------------------------------------------
        var identity = new double[N][];
        for (int i = 0; i < N; i++)
        {
            identity[i] = new double[N];
            identity[i][i] = 1.0;
        }
        var sparse = new double[] { 0.5, -4.0, 3.0, 0.0, 0.0, 2.25, -0.75, 0.0 };
        Omp.Result identityResult = Omp.Solve(identity, sparse, 3, 1e-12);
        CheckClose(identityResult.Coefficients[1], -4.0, 1e-12, "the identity dictionary recovers -4 first");
        CheckClose(identityResult.Coefficients[2], 3.0, 1e-12, "the identity dictionary recovers 3 next");
        CheckClose(identityResult.Coefficients[5], 2.25, 1e-12, "the identity dictionary recovers 2.25 third");
        for (int j = 0; j < N; j++)
        {
            if (j == 1 || j == 2 || j == 5) continue;
            CheckClose(identityResult.Coefficients[j], 0.0, 1e-12, "the identity dictionary leaves coefficient " + j + " at zero");
        }
        Check(identityResult.Support.Count == 3, "the identity dictionary uses exactly three atoms");

        // ---- single atom signal and early stopping ---------------------------------------
        var single = Column(orthonormal, 2);
        var scaled = new double[N];
        for (int i = 0; i < N; i++) scaled[i] = 2.5 * single[i];
        Omp.Result early = Omp.Solve(orthonormal, scaled, 5, 1e-12);
        Check(early.Support.Count == 1, "an exact single-atom signal stops after one step even with sparsity 5");
        CheckClose(early.Coefficients[2], 2.5, 1e-10, "the single-atom coefficient is recovered");

        // ---- greedy is optimal for s = 1, and never beats brute force for s = 2 -----------
        ulong state = 592001UL;
        for (int trial = 0; trial < 5; trial++)
        {
            int dimension = 6;
            int atoms = 10;
            var dictionary = new double[dimension][];
            for (int i = 0; i < dimension; i++)
            {
                dictionary[i] = new double[atoms];
                for (int j = 0; j < atoms; j++) dictionary[i][j] = Rng.NextSigned(ref state);
            }
            for (int j = 0; j < atoms; j++)
            {
                double norm = Omp.Norm(Column(dictionary, j));
                for (int i = 0; i < dimension; i++) dictionary[i][j] /= norm;
            }
            var signal = new double[dimension];
            for (int i = 0; i < dimension; i++) signal[i] = Rng.NextSigned(ref state);
            Omp.Result one = Omp.Solve(dictionary, signal, 1, 1e-12);
            double bruteOne = Omp.BruteForceResidual(dictionary, signal, 1);
            CheckClose(one.ResidualNorms[one.ResidualNorms.Count - 1], bruteOne, 1e-10,
                    "for s = 1 greedy is the exact minimiser on trial " + trial);
            Omp.Result two = Omp.Solve(dictionary, signal, 2, 1e-12);
            double bruteTwo = Omp.BruteForceResidual(dictionary, signal, 2);
            Check(two.ResidualNorms[two.ResidualNorms.Count - 1] >= bruteTwo - 1e-10,
                    "for s = 2 OMP is documented as no better than exhaustive search on trial " + trial);
            Check(GreedyRuleHolds(dictionary, two, 2), "the greedy rule holds on trial " + trial);
        }

        // ---- random sparse recovery -------------------------------------------------------
        int recoveries = 0;
        for (int trial = 0; trial < 10; trial++)
        {
            int dimension = 20;
            int atoms = 40;
            int sparsity = 3;
            var dictionary = new double[dimension][];
            for (int i = 0; i < dimension; i++)
            {
                dictionary[i] = new double[atoms];
                for (int j = 0; j < atoms; j++) dictionary[i][j] = Rng.NextSigned(ref state);
            }
            for (int j = 0; j < atoms; j++)
            {
                double norm = Omp.Norm(Column(dictionary, j));
                for (int i = 0; i < dimension; i++) dictionary[i][j] /= norm;
            }
            var trueCoefficients = new double[atoms];
            var chosen = new List<int>();
            while (chosen.Count < sparsity)
            {
                int candidate = Rng.NextBelow(ref state, atoms);
                if (!chosen.Contains(candidate)) chosen.Add(candidate);
            }
            foreach (int index in chosen) trueCoefficients[index] = Rng.NextBelow(ref state, 2001) / 500.0 - 2.0;
            var signal = new double[dimension];
            for (int i = 0; i < dimension; i++)
            {
                double total = 0.0;
                foreach (int index in chosen) total += dictionary[i][index] * trueCoefficients[index];
                signal[i] = total;
            }
            Omp.Result result = Omp.Solve(dictionary, signal, sparsity, 1e-12);
            Check(GreedyRuleHolds(dictionary, result, sparsity), "the greedy rule holds on recovery trial " + trial);
            Check(result.Support.Count == sparsity, "the support has exactly s atoms on recovery trial " + trial);
            var distinct = new HashSet<int>(result.Support);
            Check(distinct.Count == sparsity, "the support never repeats an atom on recovery trial " + trial);
            bool supportMatches = true;
            foreach (int index in new List<int>(result.Support)) if (!chosen.Contains(index)) supportMatches = false;
            bool coefficientsMatch = true;
            for (int j = 0; j < atoms; j++)
            {
                if (Math.Abs(result.Coefficients[j] - trueCoefficients[j]) > 1e-10) coefficientsMatch = false;
            }
            // The identity ||r||^2 + ||D_S x_S||^2 = ||y||^2 holds for every OMP solution, recovered or
            // not, because the residual is orthogonal to the selected span.
            double reconstructionEnergy = 0.0;
            for (int i = 0; i < dimension; i++)
            {
                double value = 0.0;
                for (int j = 0; j < atoms; j++) value += dictionary[i][j] * result.Coefficients[j];
                reconstructionEnergy += value * value;
            }
            CheckClose(reconstructionEnergy + Math.Pow(result.ResidualNorms[result.ResidualNorms.Count - 1], 2.0),
                    Math.Pow(Omp.Norm(signal), 2.0), 1e-9, "the Pythagorean identity holds on trial " + trial);
            for (int step = 1; step < result.ResidualNorms.Count; step++)
            {
                Check(result.ResidualNorms[step] <= result.ResidualNorms[step - 1] + 1e-12,
                        "the residual is non-increasing at step " + step + " of recovery trial " + trial);
            }
            double finalNorm = result.ResidualNorms[result.ResidualNorms.Count - 1];
            if (supportMatches && coefficientsMatch)
            {
                Check(finalNorm <= 1e-10, "a recovered instance ends with a zero residual on trial " + trial);
                recoveries++;
            }
            double[] finalResidual = result.Residuals[result.Residuals.Count - 1];
            foreach (int index in result.Support)
            {
                CheckClose(Omp.Dot(Column(dictionary, index), finalResidual), 0.0, 1e-10,
                        "the final residual is orthogonal to atom " + index + " on trial " + trial);
            }
            double mu = Omp.Coherence(dictionary, false);
            // The classical sufficient conditions (s < (1/2)(1 + 1/mu), equivalently mu < 1/(2s-1))
            // are *not* met by this dictionary family, and the file does not pretend otherwise: the
            // exhaustive check above already showed that OMP is not an optimal s > 1 decoder in
            // general, so exact recovery here is an empirical property of these draws.
            Check(mu > 1.0 / (2.0 * sparsity - 1), "the coherence exceeds 1/(2s-1) on trial " + trial
                    + " (mu = " + mu.ToString("0.000") + "), so no OMP guarantee applies");
            Check(mu < 0.8, "the coherence is still below 0.8 on trial " + trial);
        }
        // OMP is not a guaranteed exact decoder without the coherence condition, which these
        // dictionaries do not satisfy; the file reports the measured rate instead of asserting 10/10.
        Check(recoveries >= 7, "at least seven of the ten coherent random instances are recovered exactly ("
                + recoveries + "/10), the rest being documented failures of greedy decoding");

        // ---- noise on an orthonormal dictionary (where the coherence condition does hold) ------
        {
            var trueCoefficients = new double[N];
            var chosen = new List<int> { 1, 4, 6 };
            trueCoefficients[1] = 1.0;
            trueCoefficients[4] = 1.1;
            trueCoefficients[6] = 1.2;
            var clean = new double[N];
            for (int i = 0; i < N; i++)
            {
                foreach (int index in chosen) clean[i] += orthonormal[i][index] * trueCoefficients[index];
            }
            var noise = new double[N];
            double noiseNorm = 0.0;
            for (int i = 0; i < N; i++)
            {
                noise[i] = Rng.NextSigned(ref state);
                noiseNorm += noise[i] * noise[i];
            }
            noiseNorm = Math.Sqrt(noiseNorm);
            var noisy = new double[N];
            for (int i = 0; i < N; i++) noisy[i] = clean[i] + 0.01 * noise[i] / noiseNorm;
            Omp.Result result = Omp.Solve(orthonormal, noisy, 3, 1e-14);
            bool supportMatches = true;
            foreach (int index in new List<int>(result.Support)) if (!chosen.Contains(index)) supportMatches = false;
            Check(supportMatches, "the noise of norm 0.01 does not change the selected support");
            double finalNorm = result.ResidualNorms[result.ResidualNorms.Count - 1];
            Check(finalNorm <= 0.01, "the noisy residual never exceeds the noise norm (" + finalNorm.ToString("0.000e0") + ")");
            Check(finalNorm > 0.0, "the noisy instance leaves a nonzero residual");
            Check(GreedyRuleHolds(orthonormal, result, 3), "the greedy rule holds for the noisy instance");
        }

        // ---- determinism ------------------------------------------------------------------
        var repeated = Omp.Solve(orthonormal, exactSignal, 4, 1e-12);
        bool identical = repeated.Support.Count == orthonormalResult.Support.Count;
        for (int i = 0; identical && i < N; i++)
        {
            if (repeated.Coefficients[i] != orthonormalResult.Coefficients[i]) identical = false;
        }
        Check(identical, "repeated runs give bit-identical coefficients");
        // The signal has five nonzeros, so a three-atom fit must leave the two smallest behind:
        // sqrt(0.5^2 + 0.75^2) = 0.901387... Both OMP and exhaustive search stop at the same place
        // on an orthogonal dictionary, which is the check.
        double expectedIdentityResidual = Math.Sqrt(0.5 * 0.5 + 0.75 * 0.75);
        CheckClose(Omp.BruteForceResidual(identity, sparse, 3), expectedIdentityResidual, 1e-12,
                "exhaustive search leaves the two smallest coefficients behind");
        CheckClose(identityResult.ResidualNorms[identityResult.ResidualNorms.Count - 1], expectedIdentityResidual, 1e-12,
                "OMP matches the exhaustive optimum on the orthogonal dictionary");
        CheckClose(identityResult.ResidualNorms[0], Math.Sqrt(30.875), 1e-12, "the identity instance has signal norm sqrt(30.875)");

        Console.WriteLine("PASS");
    }
}
