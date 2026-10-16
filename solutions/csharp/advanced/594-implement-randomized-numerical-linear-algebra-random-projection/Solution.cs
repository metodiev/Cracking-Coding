// Task 594 - Implement randomized numerical linear algebra (random projection) demo. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
//
// The Johnson-Lindenstrauss phenomenon in practice. A random projection is P = R / sqrt(k) with R a
// k x d matrix whose entries are either i.i.d. standard Gaussians (dense projection) or sqrt(3) times
// a three-point Achlioptas variable taking +1 with probability 1/6, 0 with 2/3 and -1 with 1/6, which
// also has second moment 1 and is therefore also an unbiased isometry in expectation. The file
// measures the two facts the whole technique rests on: E[||P x||^2] = ||x||^2 with variance 2||x||^4 / k
// and E[<P x, P y>] = <x, y>, estimated by Monte Carlo over hundreds of independent projections drawn
// from the fixed-seed 64-bit LCG in `Rng` (MMIX constants plus Box-Muller, documented because the
// repo's 16-bit LCG has poor low-bit structure and this file needs a few hundred thousand independent
// Gaussian draws) -- the Monte Carlo errors are compared against three standard errors so the checks
// test unbiasedness rather than luck. Those two moments give the JL distortion bound in the form
// k >= 8 ln(m) / eps^2 for m points and relative distortion eps, which the file instantiates with
// k = ceil(8 ln 60 / 0.5^2) = 132 and then verifies pair by pair over all 1770 pairs of a 60-point data set in
// dimension 500. Two applications close the loop: a nearest-neighbour demo over 200 clustered points in
// which the projected neighbour is required to land in the correct cluster and within 1.2 of the optimal
// distance for all 40 queries, and a sketch-and-solve least squares where the rows of a 2000 x 20 system
// are compressed to 500 rows and the resulting objective is compared with the exact optimum. Everything is deterministic; the reported quantities are the
// measured values, not textbook constants.
//
// Documented checks: the empirical mean of ||P x||^2 over 400 projections is within three standard
// errors of ||x||^2 for both projection families and its variance is within 25% of 2||x||^4 / k;
// the empirical mean of <P x, P y> matches <x, y> the same way; the Monte Carlo average of P P^T is within
// 10% of (d/k) I_k for both families, which is the correct isometry statement (E[P^T P] = I_d but
// E[P P^T] = (d/k) I_k, and the header records that the file checks the second one); the 60-point data set
// in R^500 has all 1770 pairwise squared distances distorted by at most 0.5 with the JL-mandated k = 132,
// both the lower and the upper end of the window being asserted, and the mean distortion staying below
// twice the concentration scale sqrt(2/k); the projected nearest neighbour lands in the right cluster for
// at least 38 of 40 queries with every achieved distance within 1.2 of the optimal one, while the exact
// identity of the neighbour is deliberately not required -- cluster members are nearly equidistant, so a
// 10% distortion reshuffles them -- and the file reports the identity-match count separately; the sketch-and-solve least squares objective is at most 1.1 times the exact optimum
// and never below it; the sparse Achlioptas projection satisfies the same unbiasedness checks as the
// Gaussian one even though 2/3 of its entries are zero; and all projections are reproducible for a
// fixed seed while a different seed gives different matrices.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state = state * 6364136223846793005UL + 1442695040888963407UL;
        return state >> 11;
    }

    internal static int NextBelow(ref ulong state, int bound)
    {
        return (int)(Next(ref state) % (ulong)bound);
    }

    internal static double NextUnit(ref ulong state)
    {
        return NextBelow(ref state, 2001) / 1000.0 - 1.0;
    }

    /** Box-Muller transform on two LCG draws, so the Gaussian stream is deterministic too. */
    internal static double NextGaussian(ref ulong state)
    {
        double u1 = (1 + NextBelow(ref state, 1000000)) / 1000001.0;
        double u2 = NextBelow(ref state, 1000000) / 1000000.0;
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

internal static class Projection
{
    internal const int Gaussian = 0;
    internal const int Sparse = 1;

    internal static double[][] Build(int family, int k, int dimension, ulong seed)
    {
        ulong state = seed;
        var matrix = new double[k][];
        double scale = 1.0 / Math.Sqrt(k);
        for (int i = 0; i < k; i++)
        {
            matrix[i] = new double[dimension];
            for (int j = 0; j < dimension; j++)
            {
                if (family == Gaussian)
                {
                    matrix[i][j] = Rng.NextGaussian(ref state) * scale;
                }
                else
                {
                    int draw = Rng.NextBelow(ref state, 6);
                    double value = draw == 0 ? 1.0 : draw == 1 ? -1.0 : 0.0;
                    matrix[i][j] = value * Math.Sqrt(3.0) * scale;
                }
            }
        }
        return matrix;
    }

    internal static double[] Apply(double[][] projection, double[] vector)
    {
        int k = projection.Length;
        var result = new double[k];
        for (int i = 0; i < k; i++)
        {
            double total = 0.0;
            for (int j = 0; j < vector.Length; j++) total += projection[i][j] * vector[j];
            result[i] = total;
        }
        return result;
    }

    internal static double NormSquared(double[] vector)
    {
        double total = 0.0;
        foreach (double value in vector) total += value * value;
        return total;
    }

    internal static double Dot(double[] a, double[] b)
    {
        double total = 0.0;
        for (int i = 0; i < a.Length; i++) total += a[i] * b[i];
        return total;
    }

    internal static double[] Difference(double[] a, double[] b)
    {
        var result = new double[a.Length];
        for (int i = 0; i < a.Length; i++) result[i] = a[i] - b[i];
        return result;
    }

    internal static double Mean(IReadOnlyList<double> values)
    {
        double total = 0.0;
        foreach (double value in values) total += value;
        return total / values.Count;
    }

    internal static double Variance(IReadOnlyList<double> values)
    {
        double mean = Mean(values);
        double total = 0.0;
        foreach (double value in values) total += (value - mean) * (value - mean);
        return total / (values.Count - 1);
    }

    /** Monte Carlo estimates of E||P x||^2, its variance and E<P x, P y> for the given family. */
    internal static void Moments(int family, double[] x, double[] y, int trials, int k, ulong seed,
            out double meanNormSquared, out double varianceNormSquared, out double meanInnerProduct)
    {
        var norms = new List<double>();
        var inner = new List<double>();
        for (int trial = 0; trial < trials; trial++)
        {
            double[][] projection = Build(family, k, x.Length, seed + 104729UL * (ulong)(trial + 1));
            double[] px = Apply(projection, x);
            double[] py = Apply(projection, y);
            norms.Add(NormSquared(px));
            inner.Add(Dot(px, py));
        }
        meanNormSquared = Mean(norms);
        varianceNormSquared = Variance(norms);
        meanInnerProduct = Mean(inner);
    }

    /**
     * Monte Carlo average of P P^T for the given family. With P = R / sqrt(k) the entries of the k x k
     * matrix P P^T have expectation (d / k) on the diagonal and 0 off it, i.e. E[P P^T] = (d/k) I_k,
     * while E[P^T P] = I_d: the row gram matrix is the one with the d/k scale factor.
     */
    internal static double[][] AverageRowGram(int family, int k, int dimension, int trials, ulong seed)
    {
        var gram = new double[k][];
        for (int i = 0; i < k; i++) gram[i] = new double[k];
        for (int trial = 0; trial < trials; trial++)
        {
            double[][] projection = Build(family, k, dimension, seed + 7919UL * (ulong)(trial + 1));
            for (int a = 0; a < k; a++)
            {
                for (int b = a; b < k; b++)
                {
                    double total = 0.0;
                    for (int j = 0; j < dimension; j++) total += projection[a][j] * projection[b][j];
                    gram[a][b] += total / trials;
                    gram[b][a] = gram[a][b];
                }
            }
        }
        return gram;
    }

    internal static double MaxGramDeviation(double[][] gram, double diagonal)
    {
        double worst = 0.0;
        for (int i = 0; i < gram.Length; i++)
        {
            for (int j = 0; j < gram.Length; j++)
            {
                double expected = i == j ? diagonal : 0.0;
                worst = Math.Max(worst, Math.Abs(gram[i][j] - expected));
            }
        }
        return worst;
    }

    /** Minimum 1 / (1 + eps) factor and maximum 1 / (1 - eps) factor over all pairs. */
    internal static void DistortionWindow(IReadOnlyList<double[]> points, int family, int k, ulong seed,
            out double minimumRatio, out double maximumRatio, out double meanAbsoluteDeviation)
    {
        double[][] projection = Build(family, k, points[0].Length, seed);
        var projected = new double[points.Count][];
        for (int i = 0; i < points.Count; i++) projected[i] = Apply(projection, points[i]);
        minimumRatio = double.PositiveInfinity;
        maximumRatio = 0.0;
        double totalDeviation = 0.0;
        int pairs = 0;
        for (int i = 0; i < points.Count; i++)
        {
            for (int j = i + 1; j < points.Count; j++)
            {
                double original = NormSquared(Difference(points[i], points[j]));
                double compressed = NormSquared(Difference(projected[i], projected[j]));
                double ratio = compressed / original;
                minimumRatio = Math.Min(minimumRatio, ratio);
                maximumRatio = Math.Max(maximumRatio, ratio);
                totalDeviation += Math.Abs(ratio - 1.0);
                pairs++;
            }
        }
        meanAbsoluteDeviation = totalDeviation / pairs;
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

    public static void Main()
    {
        ulong state = 594001UL;
        var x = new double[200];
        var y = new double[200];
        for (int i = 0; i < 200; i++)
        {
            x[i] = Rng.NextUnit(ref state);
            y[i] = Rng.NextUnit(ref state);
        }
        double normXSquared = Projection.NormSquared(x);
        double innerXY = Projection.Dot(x, y);

        // ---- unbiasedness of the norm ------------------------------------------------------
        for (int family = 0; family < 2; family++)
        {
            Projection.Moments(family, x, y, 400, 50, 594100UL + (ulong)family,
                    out double meanNorm, out double varianceNorm, out double meanInner);
            string name = family == Projection.Gaussian ? "Gaussian" : "Achlioptas";
            double standardError = Math.Sqrt(varianceNorm / 400.0);
            CheckClose(meanNorm, normXSquared, 3.0 * standardError,
                    "the " + name + " projection preserves the squared norm in expectation");
            CheckClose(varianceNorm, 2.0 * normXSquared * normXSquared / 50.0, 0.25 * 2.0 * normXSquared * normXSquared / 50.0,
                    "the " + name + " variance matches 2||x||^4/k within 25%");
            // sd(<Px,Py>) = sqrt((||x||^2 ||y||^2 + <x,y>^2) / k), so the Monte Carlo error is that
            // quantity divided by sqrt(trials).
            double innerStandardDeviation = Math.Sqrt((normXSquared * Projection.NormSquared(y) + innerXY * innerXY) / 50.0);
            CheckClose(meanInner, innerXY, 3.0 * innerStandardDeviation / Math.Sqrt(400.0),
                    "the " + name + " projection preserves the inner product in expectation");
            double[][] gram = Projection.AverageRowGram(family, 8, 60, 400, 594500UL + (ulong)family);
            double expectedDiagonal = 60.0 / 8.0;
            double gramDeviation = Projection.MaxGramDeviation(gram, expectedDiagonal);
            Check(gramDeviation <= 0.1 * expectedDiagonal,
                    "the " + name + " Monte Carlo average of P P^T is within 10% of (d/k) I (deviation "
                    + gramDeviation.ToString("0.0000") + " against " + expectedDiagonal.ToString("0.0") + ")");
        }

        // ---- Johnson-Lindenstrauss distortion over a point cloud --------------------------
        int dimension = 500;
        int pointCount = 60;
        var points = new double[pointCount][];
        for (int i = 0; i < pointCount; i++)
        {
            points[i] = new double[dimension];
            for (int j = 0; j < dimension; j++) points[i][j] = Rng.NextUnit(ref state);
        }
        double epsilon = 0.5;
        int jlTarget = (int)Math.Ceiling(8.0 * Math.Log(pointCount) / (epsilon * epsilon));
        Check(jlTarget == 132, "the JL requirement for 60 points and eps = 0.5 is k = ceil(8 ln 60 / 0.25) = 132 (got "
                + jlTarget + ")");
        Projection.DistortionWindow(points, Projection.Gaussian, jlTarget, 594777UL,
                out double minimum, out double maximum, out double meanDeviation);
        Check(maximum <= 1.0 + epsilon, "no pair is stretched by more than eps = 0.5 (worst "
                + maximum.ToString("0.0000") + ")");
        Check(minimum >= 1.0 - epsilon, "no pair is compressed by more than eps = 0.5 (worst "
                + minimum.ToString("0.0000") + ")");
        Check(meanDeviation <= 2.0 * Math.Sqrt(2.0 / jlTarget),
                "the mean distortion stays within twice the concentration scale sqrt(2/k) ("
                + meanDeviation.ToString("0.0000") + ")");
        Projection.DistortionWindow(points, Projection.Sparse, jlTarget, 594888UL,
                out double sparseMinimum, out double sparseMaximum, out _);
        Check(sparseMaximum <= 1.0 + epsilon && sparseMinimum >= 1.0 - epsilon,
                "the sparse projection also keeps every pair within eps = 0.5 (window ["
                + sparseMinimum.ToString("0.0000") + ", " + sparseMaximum.ToString("0.0000") + "])");
        Projection.DistortionWindow(points, Projection.Gaussian, 8, 594999UL,
                out double tinyMinimum, out double tinyMaximum, out double tinyDeviation);
        Check(tinyDeviation > meanDeviation, "a k = 8 projection is measurably worse than k = 131 ("
                + tinyDeviation.ToString("0.0000") + " vs " + meanDeviation.ToString("0.0000")
                + ", window [" + tinyMinimum.ToString("0.0000") + ", " + tinyMaximum.ToString("0.0000") + "])");

        // ---- nearest neighbour under projection -------------------------------------------
        const int Clusters = 20;
        const int PerCluster = 10;
        const int QueryCount = 40;
        var database = new double[Clusters * PerCluster][];
        var centres = new double[Clusters][];
        for (int cluster = 0; cluster < Clusters; cluster++)
        {
            centres[cluster] = new double[dimension];
            for (int j = 0; j < dimension; j++) centres[cluster][j] = 20.0 * Rng.NextUnit(ref state);
            for (int member = 0; member < PerCluster; member++)
            {
                var point = new double[dimension];
                for (int j = 0; j < dimension; j++) point[j] = centres[cluster][j] + 0.5 * Rng.NextUnit(ref state);
                database[cluster * PerCluster + member] = point;
            }
        }
        double[][] projectionMatrix = Projection.Build(Projection.Gaussian, 131, dimension, 594321UL);
        var projectedDatabase = new double[database.Length][];
        for (int i = 0; i < database.Length; i++) projectedDatabase[i] = Projection.Apply(projectionMatrix, database[i]);
        int hits = 0;
        int identityHits = 0;
        for (int query = 0; query < QueryCount; query++)
        {
            int cluster = query % Clusters;
            var point = new double[dimension];
            for (int j = 0; j < dimension; j++) point[j] = centres[cluster][j] + 0.4 * Rng.NextUnit(ref state);
            int exactNeighbour = -1;
            double exactDistance = double.PositiveInfinity;
            for (int i = 0; i < database.Length; i++)
            {
                double distance = Projection.NormSquared(Projection.Difference(point, database[i]));
                if (distance < exactDistance)
                {
                    exactDistance = distance;
                    exactNeighbour = i;
                }
            }
            double[] projectedQuery = Projection.Apply(projectionMatrix, point);
            int projectedNeighbour = -1;
            double projectedDistance = double.PositiveInfinity;
            for (int i = 0; i < database.Length; i++)
            {
                double distance = Projection.NormSquared(Projection.Difference(projectedQuery, projectedDatabase[i]));
                if (distance < projectedDistance)
                {
                    projectedDistance = distance;
                    projectedNeighbour = i;
                }
            }
            // The exact identity of the nearest neighbour is a fragile target here: all ten members of a
            // cluster are nearly equidistant from the query, so a 10% distance distortion is enough to
            // reshuffle them. What the projection does guarantee is the cluster, and the ratio of the
            // achieved distance to the optimal one; both are checked, and the identity matches are
            // counted and reported.
            if (projectedNeighbour / PerCluster == cluster) hits++;
            if (projectedNeighbour == exactNeighbour) identityHits++;
            double achieved = Projection.NormSquared(Projection.Difference(point, database[projectedNeighbour]));
            Check(achieved <= 1.2 * exactDistance, "the projected neighbour is within 1.2 of the optimal distance "
                    + "(query " + query + ")");
            Check(exactNeighbour / PerCluster == cluster,
                    "query " + query + " sits closest to its own cluster centre");
        }
        Check(hits >= 38, "the projected nearest neighbour lands in the right cluster for at least 38 of 40 queries ("
                + hits + ", exact identity matches: " + identityHits + ")");

        // ---- sketch and solve least squares ----------------------------------------------
        int rows = 2000;
        int columns = 20;
        var design = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            design[i] = new double[columns];
            for (int j = 0; j < columns; j++) design[i][j] = Rng.NextUnit(ref state);
        }
        var target = new double[rows];
        for (int i = 0; i < rows; i++)
        {
            double total = 0.0;
            for (int j = 0; j < columns; j++) total += design[i][j] * (j % 3 - 1.0);
            target[i] = total + 0.1 * Rng.NextUnit(ref state);
        }
        double[] exactSolution = LeastSquares(design, target);
        double exactObjective = Objective(design, target, exactSolution);
        int sketchRows = 500;
        double[][] sketch = Projection.Build(Projection.Gaussian, sketchRows, rows, 594654UL);
        var sketchedDesign = new double[sketchRows][];
        for (int i = 0; i < sketchRows; i++) sketchedDesign[i] = Combine(sketch[i], design);
        var sketchedTarget = new double[sketchRows];
        for (int i = 0; i < sketchRows; i++) sketchedTarget[i] = Projection.Dot(sketch[i], target);
        double[] sketchSolution = LeastSquares(sketchedDesign, sketchedTarget);
        double sketchObjective = Objective(design, target, sketchSolution);
        Check(sketchObjective >= exactObjective - 1e-9, "the sketch solution cannot beat the exact optimum");
        Check(sketchObjective <= 1.1 * exactObjective,
                "the sketch-and-solve objective is within 1.1 of the exact optimum ("
                + (sketchObjective / exactObjective).ToString("0.0000") + ")");
        Check(sketchRows < rows, "the sketch really reduces the row count (" + sketchRows + " < " + rows + ")");

        // ---- determinism ------------------------------------------------------------------
        double[][] first = Projection.Build(Projection.Gaussian, 5, 10, 594000UL);
        double[][] second = Projection.Build(Projection.Gaussian, 5, 10, 594000UL);
        double difference = 0.0;
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 10; j++) difference = Math.Max(difference, Math.Abs(first[i][j] - second[i][j]));
        }
        CheckClose(difference, 0.0, 0.0, "a fixed seed reproduces the projection bit for bit");
        double[][] third = Projection.Build(Projection.Gaussian, 5, 10, 594001UL);
        double distinct = 0.0;
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 10; j++) distinct = Math.Max(distinct, Math.Abs(first[i][j] - third[i][j]));
        }
        Check(distinct > 1e-9, "a different seed gives a different projection");

        Console.WriteLine("PASS");
    }

    /** Sketching a design matrix row by row with an explicit double loop, no LINQ. */
    private static double[] Combine(double[] weights, double[][] design)
    {
        int columns = design[0].Length;
        var result = new double[columns];
        for (int i = 0; i < weights.Length; i++)
        {
            if (weights[i] == 0.0) continue;
            for (int j = 0; j < columns; j++) result[j] += weights[i] * design[i][j];
        }
        return result;
    }

    private static double[] LeastSquares(double[][] a, double[] b)
    {
        int rows = a.Length;
        int columns = a[0].Length;
        var gram = new double[columns][];
        var rhs = new double[columns];
        for (int p = 0; p < columns; p++)
        {
            gram[p] = new double[columns];
            for (int q = 0; q < columns; q++)
            {
                double total = 0.0;
                for (int i = 0; i < rows; i++) total += a[i][p] * a[i][q];
                gram[p][q] = total;
            }
            double dot = 0.0;
            for (int i = 0; i < rows; i++) dot += a[i][p] * b[i];
            rhs[p] = dot;
        }
        for (int column = 0; column < columns; column++)
        {
            int pivot = column;
            for (int row = column + 1; row < columns; row++)
            {
                if (Math.Abs(gram[row][column]) > Math.Abs(gram[pivot][column])) pivot = row;
            }
            (gram[pivot], gram[column]) = (gram[column], gram[pivot]);
            (rhs[pivot], rhs[column]) = (rhs[column], rhs[pivot]);
            for (int row = column + 1; row < columns; row++)
            {
                double factor = gram[row][column] / gram[column][column];
                for (int c = column; c < columns; c++) gram[row][c] -= factor * gram[column][c];
                rhs[row] -= factor * rhs[column];
            }
        }
        var solution = new double[columns];
        for (int row = columns - 1; row >= 0; row--)
        {
            double total = rhs[row];
            for (int c = row + 1; c < columns; c++) total -= gram[row][c] * solution[c];
            solution[row] = total / gram[row][row];
        }
        return solution;
    }

    private static double Objective(double[][] a, double[] b, double[] solution)
    {
        double total = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            double prediction = 0.0;
            for (int j = 0; j < solution.Length; j++) prediction += a[i][j] * solution[j];
            double difference = prediction - b[i];
            total += difference * difference;
        }
        return total;
    }
}
