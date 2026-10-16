// Task 515 - Implement minibatch gradient descent training loop for logistic regression. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Binary logistic regression with an explicit intercept, trained three ways on the same
// deterministic dataset: full-batch gradient descent, minibatch/SGD with Fisher-Yates epoch
// shuffling driven by splitmix64, and Newton-Raphson (IRLS) as the oracle. Labels are the signs of
// a planted separator plus deterministic noise, so the data are almost linearly separable and the
// optimum is well defined for lambda > 0. The loss is the mean logistic log-loss
// (1/n) sum log(1 + exp(-y_i w.x_i)) + lambda/2 ||w||^2 with y in {-1, +1}, its gradient is
// -(1/n) sum y_i x_i sigma(-y_i w.x_i) + lambda w, and the Hessian is
// (1/n) sum sigma(1 - sigma) x x^T + lambda I, which makes IRLS exact and gives the gradient
// Lipschitz constant L (largest Hessian eigenvalue: (1/(4n)) lambda_max(X^T X) + lambda).
// Documented checks: the IRLS optimum has ||grad L|| <= 1e-8; the loss is numerically convex
// (Jensen holds for 400 random pairs); the average of the batch gradients over one shuffled epoch
// reproduces the full-batch gradient to 1e-12 (minibatches are unbiased and cover the data);
// minibatch GD (batch 20, lr = 0.3, 2000 epochs, 40000 updates) lands within 0.03 of the IRLS
// solution with ||grad L|| <= 0.01 and a loss within 2e-4 of the optimum (the constant step leaves
// the documented SGD noise floor, so 100 -> 500 epochs still lowers the loss but 2000 epochs no
// longer improves on 500); full-batch GD with lr = 0.9 / L is monotone (provable for 0 < lr < 2/L)
// while lr = 8 / L diverges (the loss grows by orders of magnitude); accuracy on the training set is
// at least 0.85 for both solutions, they agree within 0.02, and the recovered direction has cosine
// >= 0.98 with the planted one; L2 with lambda = 1 shrinks the weight norm and still drives the
// gradient to zero at its own optimum; the SGD gradient-estimate variance at batch size 1 exceeds
// the variance at batch size 16 by at least 3x; shuffling is reproducible; and no iterate ever
// becomes NaN.
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

internal class Logistic
{
    internal double[][] X = Array.Empty<double[]>();
    internal double[] Y = Array.Empty<double>();
    internal double Lambda;

    internal int Count => X.Length;

    internal int Dimension => X[0].Length;

    internal static Logistic Planted(int samples, int dimension, double noise, double lambda, ulong seed)
    {
        var rng = new Seeded(seed);
        var truth = new double[dimension];
        for (int j = 0; j < dimension; j++) truth[j] = rng.Uniform() * 2.0 - 1.0;
        var data = new Logistic { Lambda = lambda };
        data.X = new double[samples][];
        data.Y = new double[samples];
        for (int i = 0; i < samples; i++)
        {
            var row = new double[dimension];
            row[0] = 1.0;                       // intercept
            for (int j = 1; j < dimension; j++) row[j] = rng.Uniform() * 2.0 - 1.0;
            double margin = 0.0;
            for (int j = 1; j < dimension; j++) margin += truth[j] * row[j];
            margin += (rng.Uniform() * 2.0 - 1.0) * noise;
            data.X[i] = row;
            data.Y[i] = margin >= 0.0 ? 1.0 : -1.0;
        }
        data.Truth = truth;
        return data;
    }

    internal double[] Truth = Array.Empty<double>();

    internal double Loss(double[] w)
    {
        double total = 0.0;
        for (int i = 0; i < Count; i++)
        {
            double margin = 0.0;
            for (int j = 0; j < Dimension; j++) margin += w[j] * X[i][j];
            total += LogOnePlusExp(-Y[i] * margin);
        }
        total /= Count;
        double regulariser = 0.0;
        for (int j = 0; j < Dimension; j++) regulariser += w[j] * w[j];
        return total + 0.5 * Lambda * regulariser;
    }

    internal double[] Gradient(double[] w)
    {
        var gradient = new double[Dimension];
        for (int i = 0; i < Count; i++)
        {
            double margin = 0.0;
            for (int j = 0; j < Dimension; j++) margin += w[j] * X[i][j];
            double coefficient = -Y[i] * Sigmoid(-Y[i] * margin) / Count;
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
            double s = Sigmoid(-Y[i] * margin);
            double weight = s * (1.0 - s) / Count;
            for (int a = 0; a < d; a++)
            {
                for (int b = 0; b < d; b++) hessian[a][b] += weight * X[i][a] * X[i][b];
            }
        }
        for (int j = 0; j < d; j++) hessian[j][j] += Lambda;
        return hessian;
    }

    /** Newton-Raphson / IRLS; exact for this convex problem. */
    internal double[] Iirls(int maxSteps, double tolerance)
    {
        var w = new double[Dimension];
        for (int step = 0; step < maxSteps; step++)
        {
            double[] gradient = Gradient(w);
            double norm = 0.0;
            foreach (double value in gradient) norm += value * value;
            if (Math.Sqrt(norm) <= tolerance) break;
            double[][] hessian = Hessian(w);
            double[] direction = Solve(hessian, gradient);
            for (int j = 0; j < Dimension; j++) w[j] -= direction[j];
        }
        return w;
    }

    internal double[] FullBatchGradientDescent(double[] start, double learningRate, int steps,
            out List<double> trace)
    {
        var w = (double[])start.Clone();
        trace = new List<double>();
        trace.Add(Loss(w));
        for (int step = 0; step < steps; step++)
        {
            double[] gradient = Gradient(w);
            for (int j = 0; j < Dimension; j++) w[j] -= learningRate * gradient[j];
            trace.Add(Loss(w));
        }
        return w;
    }

    /** Minibatch / SGD with a fresh Fisher-Yates shuffle of the sample order each epoch. */
    internal double[] MinibatchGradientDescent(double[] start, int batchSize, double learningRate,
            int epochs, ulong shuffleSeed, out int updates)
    {
        var w = (double[])start.Clone();
        var rng = new Seeded(shuffleSeed);
        updates = 0;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            int[] order = ShuffledOrder(rng);
            for (int start2 = 0; start2 < Count; start2 += batchSize)
            {
                int size = Math.Min(batchSize, Count - start2);
                double[] gradient = BatchGradient(w, order, start2, size);
                for (int j = 0; j < Dimension; j++) w[j] -= learningRate * gradient[j];
                updates += 1;
            }
        }
        return w;
    }

    internal int[] ShuffledOrder(Seeded rng)
    {
        var order = new int[Count];
        for (int i = 0; i < Count; i++) order[i] = i;
        for (int i = Count - 1; i > 0; i--)
        {
            int j = rng.NextIndex(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
        return order;
    }

    /** Gradient of the loss restricted to one batch (mean over the batch, no 1/n scaling). */
    internal double[] BatchGradient(double[] w, int[] order, int start, int size)
    {
        var gradient = new double[Dimension];
        for (int k = start; k < start + size; k++)
        {
            int i = order[k];
            double margin = 0.0;
            for (int j = 0; j < Dimension; j++) margin += w[j] * X[i][j];
            double coefficient = -Y[i] * Sigmoid(-Y[i] * margin);
            for (int j = 0; j < Dimension; j++) gradient[j] += coefficient * X[i][j];
        }
        for (int j = 0; j < Dimension; j++) gradient[j] = gradient[j] / size + Lambda * w[j];
        return gradient;
    }

    /** Largest eigenvalue of X^T X / n by power iteration - used for the Lipschitz constant. */
    internal double LargestCurvature()
    {
        int d = Dimension;
        var direction = new double[d];
        for (int j = 0; j < d; j++) direction[j] = 1.0 / Math.Sqrt(d);
        double eigenvalue = 0.0;
        for (int iteration = 0; iteration < 500; iteration++)
        {
            var image = new double[d];
            for (int i = 0; i < Count; i++)
            {
                double projection = 0.0;
                for (int j = 0; j < d; j++) projection += X[i][j] * direction[j];
                for (int j = 0; j < d; j++) image[j] += X[i][j] * projection;
            }
            double norm = 0.0;
            foreach (double value in image) norm += value * value;
            norm = Math.Sqrt(norm);
            for (int j = 0; j < d; j++) image[j] /= norm;
            double shift = 0.0;
            for (int j = 0; j < d; j++) shift += Math.Abs(image[j] - direction[j]);
            direction = image;
            eigenvalue = norm / Count;
            if (shift <= 1e-14) break;
        }
        return eigenvalue;
    }

    internal double Accuracy(double[] w)
    {
        int correct = 0;
        for (int i = 0; i < Count; i++)
        {
            double margin = 0.0;
            for (int j = 0; j < Dimension; j++) margin += w[j] * X[i][j];
            double predicted = margin >= 0.0 ? 1.0 : -1.0;
            if (predicted == Y[i]) correct += 1;
        }
        return correct / (double)Count;
    }

    internal double Norm(double[] w)
    {
        double total = 0.0;
        foreach (double value in w) total += value * value;
        return Math.Sqrt(total);
    }

    internal static double Sigmoid(double z) => 1.0 / (1.0 + Math.Exp(-z));

    internal static double LogOnePlusExp(double z) => z > 0.0 ? z + Math.Log(1.0 + Math.Exp(-z))
            : Math.Log(1.0 + Math.Exp(z));

    /** Gaussian elimination with partial pivoting (the Newton step). */
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
        // --- the plain (unregularised) problem -----------------------------------------------------------
        Logistic data = Logistic.Planted(400, 6, 0.35, 0.05, 515515UL);
        Check(data.Count == 400 && data.Dimension == 6, "dataset shape");
        int positives = 0;
        foreach (double label in data.Y)
        {
            if (label > 0.0) positives += 1;
        }
        Check(positives >= 150 && positives <= 250, "the classes are roughly balanced (" + positives + ")");
        double[] optimum = data.Iirls(60, 1e-12);
        double[] optimumGradient = data.Gradient(optimum);
        double optimumNorm = data.Norm(optimumGradient);
        Check(optimumNorm <= 1e-8, "the IRLS optimum has a vanishing gradient (" + optimumNorm + ")");
        double optimumLoss = data.Loss(optimum);
        Check(optimumLoss > 0.0 && optimumLoss < 0.5, "the optimum loss is small (" + optimumLoss + ")");

        // --- convexity (Jensen) ----------------------------------------------------------------------------
        var rngCheck = new Seeded(99UL);
        for (int trial = 0; trial < 400; trial++)
        {
            var a = new double[data.Dimension];
            var b = new double[data.Dimension];
            for (int j = 0; j < data.Dimension; j++)
            {
                a[j] = rngCheck.Uniform() * 4.0 - 2.0;
                b[j] = rngCheck.Uniform() * 4.0 - 2.0;
            }
            var middle = new double[data.Dimension];
            for (int j = 0; j < data.Dimension; j++) middle[j] = 0.5 * (a[j] + b[j]);
            double jensen = data.Loss(middle);
            double chord = 0.5 * (data.Loss(a) + data.Loss(b));
            Check(jensen <= chord + 1e-12, "Jensen holds at trial " + trial);
        }

        // --- minibatches are unbiased and cover the data -----------------------------------------------------
        var orderRng = new Seeded(7UL);
        int[] order = data.ShuffledOrder(orderRng);
        var orderSeen = new HashSet<int>();
        foreach (int index in order) orderSeen.Add(index);
        Check(orderSeen.Count == data.Count, "the shuffle is a permutation");
        int[] repeatOrder = data.ShuffledOrder(new Seeded(7UL));
        for (int i = 0; i < order.Length; i++)
        {
            Check(repeatOrder[i] == order[i], "the shuffle is reproducible at position " + i);
        }
        double[] probePoint = { 0.1, -0.2, 0.3, 0.05, -0.15, 0.25 };
        double[] full = data.Gradient(probePoint);
        foreach (int batchSize in new[] { 1, 5, 20, 100 })
        {
            var averaged = new double[data.Dimension];
            for (int start = 0; start < data.Count; start += batchSize)
            {
                int size = Math.Min(batchSize, data.Count - start);
                double[] batch = data.BatchGradient(probePoint, order, start, size);
                for (int j = 0; j < data.Dimension; j++) averaged[j] += batch[j] * size;
            }
            for (int j = 0; j < data.Dimension; j++) averaged[j] /= data.Count;
            double error = 0.0;
            for (int j = 0; j < data.Dimension; j++) error = Math.Max(error, Math.Abs(averaged[j] - full[j]));
            Check(error <= 1e-12, "the batchSize " + batchSize + " partition is unbiased (" + error + ")");
        }

        // --- minibatch GD reaches the optimum ----------------------------------------------------------------
        var zero = new double[data.Dimension];
        int updates;
        double[] minibatch = data.MinibatchGradientDescent(zero, 20, 0.3, 2000, 2024UL, out updates);
        Check(updates == 2000 * (data.Count / 20), "20 minibatches per epoch for 2000 epochs");
        double distance = 0.0;
        for (int j = 0; j < data.Dimension; j++) distance = Math.Max(distance, Math.Abs(minibatch[j] - optimum[j]));
        double mbGradNorm = data.Norm(data.Gradient(minibatch));
        double mbLoss = data.Loss(minibatch);
        int shortUpdates;
        double[] mediumRun = data.MinibatchGradientDescent(zero, 20, 0.3, 500, 2024UL, out shortUpdates);
        int tinyUpdates;
        double[] tinyRun = data.MinibatchGradientDescent(zero, 20, 0.3, 100, 2024UL, out tinyUpdates);
        double lossTiny = data.Loss(tinyRun);
        double lossMedium = data.Loss(mediumRun);
        Check(mbGradNorm <= 0.01, "the minibatch gradient is nearly zero (" + mbGradNorm + ")");
        Check(mbLoss - optimumLoss <= 2e-4, "the minibatch loss is within 2e-4 of the optimum ("
                + (mbLoss - optimumLoss) + ")");
        Check(distance <= 0.03, "minibatch GD lands within 0.03 of the IRLS optimum (" + distance + ")");
        Check(lossTiny > lossMedium, "training from 100 to 500 epochs reduces the loss ("
                + lossTiny + " -> " + lossMedium + ")");
        Check(mbLoss <= lossMedium + 1e-4, "the loss stays at the noise floor by 2000 epochs");
        Check(data.Norm(data.Gradient(tinyRun)) > mbGradNorm, "the gradient norm shrinks during training");
        double minibatchAccuracy = data.Accuracy(minibatch);
        double optimumAccuracy = data.Accuracy(optimum);
        Check(minibatchAccuracy >= 0.85, "accuracy of the minibatch solution (" + minibatchAccuracy + ")");
        Check(optimumAccuracy >= 0.85, "accuracy of the IRLS solution (" + optimumAccuracy + ")");
        Check(Math.Abs(minibatchAccuracy - optimumAccuracy) <= 0.02,
                "the two solutions classify alike (" + minibatchAccuracy + " vs " + optimumAccuracy + ")");
        Check(regularisedAccuracyProbe(data), "the planted noise still allows a good fit");
        double cosine = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int j = 1; j < data.Dimension; j++)
        {
            cosine += minibatch[j] * data.Truth[j];
            normA += minibatch[j] * minibatch[j];
            normB += data.Truth[j] * data.Truth[j];
        }
        cosine /= Math.Sqrt(normA * normB);
        Check(cosine >= 0.98, "the recovered direction matches the planted one (cos " + cosine + ")");
        foreach (double value in minibatch)
        {
            Check(!double.IsNaN(value) && !double.IsInfinity(value), "the minibatch iterate stays finite");
        }

        // --- the learning-rate window for full-batch GD ----------------------------------------------------------
        double curvature = data.LargestCurvature();
        double lipschitz = 0.25 * curvature + data.Lambda;
        Check(curvature > 0.0, "the data curvature is positive");
        double stableRate = 0.9 / lipschitz;
        double[] stable = data.FullBatchGradientDescent(zero, stableRate, 200, out List<double> stableTrace);
        for (int i = 1; i < stableTrace.Count; i++)
        {
            Check(stableTrace[i] <= stableTrace[i - 1] + 1e-12,
                    "the loss is monotone with lr = 0.9/L at step " + i);
        }
        double[] stableOptimum = data.Gradient(stable);
        Check(data.Norm(stableOptimum) <= 1e-4, "0.9/L converges close to the optimum at 200 steps");
        double unstableRate = 8.0 / lipschitz;
        double[] unstable = data.FullBatchGradientDescent(zero, unstableRate, 200, out List<double> unstableTrace);
        Check(unstableTrace[unstableTrace.Count - 1] > 10.0 * unstableTrace[0],
                "lr = 8/L diverges (" + unstableTrace[0] + " -> " + unstableTrace[unstableTrace.Count - 1] + ")");

        // --- regularisation -----------------------------------------------------------------------------------------
        Logistic regularised = Logistic.Planted(400, 6, 0.35, 1.0, 515515UL);
        double[] ridge = regularised.Iirls(60, 1e-12);
        Check(regularised.Norm(regularised.Gradient(ridge)) <= 1e-8, "the ridge optimum has zero gradient");
        Check(regularised.Norm(ridge) < data.Norm(optimum), "L2 shrinks the weight norm");
        Check(regularised.Accuracy(ridge) >= 0.9, "the ridge solution still classifies well");

        // --- gradient-estimate variance shrinks with the batch size ---------------------------------------------------
        double[] variancePoint = { 0.0, 0.1, -0.1, 0.2, 0.0, -0.2 };
        double smallBatchVariance = GradientVariance(data, variancePoint, 1, 40);
        double largeBatchVariance = GradientVariance(data, variancePoint, 16, 40);
        Check(smallBatchVariance >= 3.0 * largeBatchVariance,
                "batch 1 has at least 3x the variance of batch 16 (" + smallBatchVariance + " vs "
                + largeBatchVariance + ")");
        Check(largeBatchVariance >= 0.0, "the variance is non-negative");

        // --- edge cases -----------------------------------------------------------------------------------------------
        Logistic oneClass = Logistic.Planted(120, 4, 0.2, 1.0, 321UL);
        for (int i = 0; i < oneClass.Count; i++) oneClass.Y[i] = 1.0;
        double[] oneClassOptimum = oneClass.Iirls(80, 1e-12);
        Check(oneClass.Norm(oneClass.Gradient(oneClassOptimum)) <= 1e-8,
                "a single-class dataset still has a stationary point (thanks to L2)");
        Check(oneClass.Accuracy(oneClassOptimum) == 1.0, "the single-class solution predicts that class");
        Logistic tiny = Logistic.Planted(1, 3, 0.0, 1.0, 5UL);
        double[] tinyOptimum = tiny.Iirls(80, 1e-12);
        Check(tiny.Norm(tiny.Gradient(tinyOptimum)) <= 1e-8, "a one-sample dataset is still solvable");

        // --- determinism -------------------------------------------------------------------------------------------------
        int repeatUpdates;
        double[] repeatMinibatch = data.MinibatchGradientDescent(zero, 20, 0.3, 2000, 2024UL, out repeatUpdates);
        for (int j = 0; j < data.Dimension; j++)
        {
            Check(repeatMinibatch[j] == minibatch[j], "minibatch training is reproducible at " + j);
        }
        Check(repeatUpdates == updates, "the update count is reproducible");
        Check(stableTrace[stableTrace.Count - 1] <= stableTrace[0], "the stable run ends below its start");

        Console.WriteLine("PASS");
    }

    private static bool regularisedAccuracyProbe(Logistic data) => data.Accuracy(data.Iirls(60, 1e-12)) >= 0.8;

    /** Variance of the minibatch gradient over 40 independent batches at a fixed point. */
    private static double GradientVariance(Logistic data, double[] point, int batchSize, int draws)
    {
        var rng = new Seeded(31337UL);
        var means = new double[data.Dimension];
        var squares = new double[data.Dimension];
        for (int draw = 0; draw < draws; draw++)
        {
            int[] order = data.ShuffledOrder(rng);
            double[] gradient = data.BatchGradient(point, order, 0, batchSize);
            for (int j = 0; j < data.Dimension; j++)
            {
                means[j] += gradient[j];
                squares[j] += gradient[j] * gradient[j];
            }
        }
        double total = 0.0;
        for (int j = 0; j < data.Dimension; j++)
        {
            double mean = means[j] / draws;
            double variance = squares[j] / draws - mean * mean;
            total += variance;
        }
        return total;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
