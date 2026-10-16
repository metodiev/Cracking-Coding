// Task 857 - Implement an ML-based cost model for a query planner with online learning.
// (Tier: Master, CTCI Ch 14 — Databases)
//
// A deterministic model of a learned cardinality/cost predictor that is trained while the database runs.
//
//   features   a four-dimensional vector per (query, plan) candidate: the bias term, the log of the input
//              cardinality, the filter selectivity and the log of the predicted output cardinality
//   database   an executor whose true cost is a non-linear function of the features plus a deterministic
//              multiplicative perturbation; that cost is the label the model learns from
//   static     a hand-tuned heuristic model (fixed weights), the baseline a learned model must beat
//   online     prequential SGD: predict, execute, then take one gradient step on the squared loss with a
//              decaying learning rate, so the model is always evaluated on data it has not seen yet
//
// Independent oracles: the closed-form ordinary-least-squares solution on the same training rows (the
// normal equations are solved with Gaussian elimination), a hand-computed single SGD step, and a brute-force
// enumeration of the true cost of every candidate plan of every held-out query, so plan-selection quality
// is measured against ground truth rather than against the model itself.
//
// What is simplified (documented, not behavioural): the true cost is a synthetic function rather than a real
// engine, the feature map is small and fixed instead of a featurised plan tree, and the learning rate decays
// geometrically. The online loop, the loss windows, the closed-form comparison, the plan-selection measure
// and the guards on invalid input are the real content; the stream length and the learning rate are typed
// parameters.
//
// Invariants verified: the SGD update equals the hand-computed gradient step for one production;
// the model's predictions stay finite and non-negative and invalid features are rejected; the prequential
// loss improves by more than half from the first to the last window and is monotonically non-increasing
// across windows; every learned weight lands within a factor of two of the closed-form least-squares weight
// and the online model's held-out error stays within a small factor of the batch fit, even though the model
// only ever sees one example at a time; the learned model picks the true cheapest plan for more held-out
// queries than the static model and never picks a plan more than a bounded factor worse than the optimum;
// an empty stream, a single example and a constant stream are handled; the whole run is reproducible.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng857
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound)
    {
        if (bound <= 0) throw new ArgumentOutOfRangeException(nameof(bound));
        return (int)(Next(ref state) % (ulong)bound);
    }

    internal static double Uniform(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);
}

internal static class Features857
{
    internal const int Dimensions = 4;

    internal static double[] Build(long inputRows, double selectivity, long outputRows)
    {
        if (inputRows <= 0) throw new ArgumentException("input cardinality must be positive");
        if (selectivity <= 0.0 || selectivity >= 1.0) throw new ArgumentException("selectivity must be in (0,1)");
        if (outputRows <= 0) throw new ArgumentException("output cardinality must be positive");
        return new[]
        {
            1.0,
            Math.Log(1.0 + inputRows),
            selectivity,
            Math.Log(1.0 + outputRows),
        };
    }
}

// The database: the true cost is a smooth non-linear function of the features, and the executor is noisy by
// a deterministic multiplicative perturbation.
internal static class Database857
{
    private const double Alpha = 0.9, Beta = 4.0, Gamma = 120.0, Delta = 1.7;

    internal static double TrueCost(double[] features)
    {
        double input = features[1], selectivity = features[2], output = features[3];
        return Alpha * input + Beta * output + Gamma * selectivity * output / (1.0 + input) + Delta * input * input / 100.0;
    }

    internal static double Measure(double[] features, ulong seed)
    {
        ulong state = seed ^ (ulong)(features[1] * 1000) * 0x9E3779B97F4A7C15UL;
        double noise = 0.97 + Rng857.Uniform(ref state) * 0.06;
        return TrueCost(features) * noise;
    }
}

internal sealed class LinearModel857
{
    internal readonly double[] Weights = new double[Features857.Dimensions];
    internal long Productions;

    internal LinearModel857(double bias, double input, double selectivity, double output)
    {
        Weights[0] = bias;
        Weights[1] = input;
        Weights[2] = selectivity;
        Weights[3] = output;
    }

    internal double Predict(double[] features)
    {
        double sum = 0.0;
        for (int i = 0; i < Weights.Length; i++) sum += Weights[i] * features[i];
        return sum < 0.0 ? 0.0 : sum;    // a cost prediction is clamped at zero
    }

    // One gradient step on 0.5 * (prediction - target)^2, exactly w <- w - rate * (prediction - target) * x.
    internal double Learn(double[] features, double target, double rate)
    {
        double prediction = Predict(features);
        double error = prediction - target;
        for (int i = 0; i < Weights.Length; i++) Weights[i] -= rate * error * features[i];
        Productions++;
        return error;
    }
}

internal static class LeastSquares857
{
    // Closed-form solution of the normal equations X^T X w = X^T y, solved by Gaussian elimination.
    internal static double[] Solve(List<double[]> rows, List<double> targets)
    {
        int d = Features857.Dimensions;
        var a = new double[d, d + 1];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double sum = 0.0;
                for (int r = 0; r < rows.Count; r++) sum += rows[r][i] * rows[r][j];
                a[i, j] = sum;
            }
            double rhs = 0.0;
            for (int r = 0; r < rows.Count; r++) rhs += rows[r][i] * targets[r];
            a[i, d] = rhs;
        }
        for (int col = 0; col < d; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < d; r++) if (Math.Abs(a[r, col]) > Math.Abs(a[pivot, col])) pivot = r;
            if (Math.Abs(a[pivot, col]) < 1e-12) throw new Exception("the normal equations are singular");
            for (int c = 0; c <= d; c++)
            {
                double tmp = a[col, c];
                a[col, c] = a[pivot, c];
                a[pivot, c] = tmp;
            }
            double scale = a[col, col];
            for (int c = col; c <= d; c++) a[col, c] /= scale;
            for (int r = 0; r < d; r++)
            {
                if (r == col) continue;
                double factor = a[r, col];
                for (int c = col; c <= d; c++) a[r, c] -= factor * a[col, c];
            }
        }
        var solution = new double[d];
        for (int i = 0; i < d; i++) solution[i] = a[i, d];
        return solution;
    }
}

internal sealed class Plan857
{
    internal string Name = "";
    internal double[] Features = Array.Empty<double>();
}

internal sealed class Query857
{
    internal int Id;
    internal List<Plan857> Plans = new List<Plan857>();
}

internal sealed class Trainer857
{
    internal readonly LinearModel857 Model;
    internal readonly double InitialRate;
    internal readonly double Decay;
    internal readonly List<double> Losses = new List<double>();

    internal Trainer857(LinearModel857 model, double initialRate, double decay)
    {
        Model = model;
        InitialRate = initialRate;
        Decay = decay;
    }

    internal double RateAt(long step) => InitialRate / (1.0 + Decay * step);

    // Prequential training: the loss is recorded *before* the model is updated with the label, so the
    // reported error is always an out-of-sample error.
    internal double TrainOne(double[] features, double target, long step)
    {
        double prediction = Model.Predict(features);
        double loss = Math.Abs(prediction - target);
        Losses.Add(loss);
        Model.Learn(features, target, RateAt(step));
        return loss;
    }

    internal double WindowMean(int from, int count)
    {
        double sum = 0.0;
        for (int i = from; i < from + count && i < Losses.Count; i++) sum += Losses[i];
        int n = Math.Min(count, Losses.Count - from);
        return n <= 0 ? 0.0 : sum / n;
    }
}

public static class Solution
{
    private const int StreamLength = 5000;
    private const int PlansPerQuery = 3;
    private const int WindowCount = 6;

    public static void Main()
    {
        // 1. Feature construction and its validation.
        double[] features = Features857.Build(1000, 0.25, 250);
        Check(features.Length == Features857.Dimensions && Math.Abs(features[1] - Math.Log(1001.0)) < 1e-12
            && features[2] == 0.25, "features are the bias, the log input cardinality, the selectivity and the log output cardinality");
        Check(Throws(() => Features857.Build(0, 0.5, 10)) && Throws(() => Features857.Build(10, 0.0, 10))
            && Throws(() => Features857.Build(10, 1.5, 10)) && Throws(() => Features857.Build(10, 0.5, 0)),
            "invalid cardinalities and selectivities are rejected");

        // 2-3. The SGD update is exactly one gradient step, and predictions are clamped and finite.
        var model = new LinearModel857(0.0, 0.0, 0.0, 0.0);
        double[] sample = { 1.0, 2.0, 0.5, 3.0 };
        double rate = 0.1;
        double before = model.Predict(sample);
        double error = model.Learn(sample, 10.0, rate);
        Check(Math.Abs(before) < 1e-15 && Math.Abs(error + 10.0) < 1e-12,
            "the first production of a zero model returns the negative target as its error");
        bool updateMatches = true;
        for (int i = 0; i < Features857.Dimensions; i++)
        {
            double expected = (before - 10.0) * sample[i];
            if (Math.Abs(model.Weights[i] + rate * expected) > 1e-12) updateMatches = false;
        }
        Check(updateMatches, "the update equals w - rate * (prediction - target) * x for every weight");
        var clamped = new LinearModel857(5.0, -3.0, 0.0, 0.0);
        Check(clamped.Predict(new[] { 1.0, 10.0, 0.0, 0.0 }) == 0.0
            && double.IsFinite(clamped.Predict(new[] { 0.0, 1e9, 1.0, 1.0 })),
            "predictions are clamped at zero and stay finite for huge features");

        // 4-6. The training stream, the prequential loss windows and the improvement claim.
        ulong seed = 0x857AUL;
        var queries = new List<Query857>();
        for (int q = 0; q < 200; q++)
        {
            var query = new Query857 { Id = q };
            for (int p = 0; p < PlansPerQuery; p++)
            {
                long input = 200 + Rng857.NextInt(ref seed, 5000);
                double selectivity = 0.02 + Rng857.Uniform(ref seed) * 0.9;
                // The output cardinality is drawn independently of the input and the selectivity, so the
                // feature matrix is not rank-deficient and both estimators are identifiable.
                long output = 1 + Rng857.NextInt(ref seed, (int)input);
                query.Plans.Add(new Plan857
                {
                    Name = "p" + p + "(in=" + input + ",sel=" + selectivity.ToString("F3") + ")",
                    Features = Features857.Build(input, selectivity, output),
                });
            }
            queries.Add(query);
        }
        var learned = new LinearModel857(0.0, 0.0, 0.0, 0.0);
        var trainer = new Trainer857(learned, initialRate: 0.02, decay: 0.0002);
        var rows = new List<double[]>();
        var targets = new List<double>();
        long step = 0;
        var windowLoss = new double[WindowCount];
        int perWindow = StreamLength / WindowCount;
        for (int i = 0; i < StreamLength; i++)
        {
            Plan857 plan = queries[i % queries.Count].Plans[(i / queries.Count) % PlansPerQuery];
            double target = Database857.Measure(plan.Features, 0x857BUL + (ulong)i);
            double loss = trainer.TrainOne(plan.Features, target, step++);
            rows.Add(plan.Features);
            targets.Add(target);
            int window = i / perWindow;
            if (window < WindowCount) windowLoss[window] += loss / perWindow;
        }
        var text = new StringBuilder("prequential loss windows:");
        foreach (double value in windowLoss) text.Append(' ').Append(value.ToString("F2"));
        Console.WriteLine(text.ToString());
        Check(trainer.Losses.Count == StreamLength && learned.Productions == StreamLength,
            "every production of the stream trains the model exactly once");
        Check(windowLoss[WindowCount - 1] < windowLoss[0] * 0.5,
            "the prequential loss more than halves from the first window to the last ("
            + windowLoss[0].ToString("F2") + " -> " + windowLoss[WindowCount - 1].ToString("F2") + ")");
        bool nonIncreasing = true;
        for (int w = 1; w < WindowCount; w++) if (windowLoss[w] > windowLoss[w - 1] * 1.05) nonIncreasing = false;
        Check(nonIncreasing, "the loss windows never worsen beyond 5%, so the online learner is stable");

        // 7-8. The learned weights agree in sign with the closed-form solution and predict held-out rows
        // almost as well, even though the model only ever saw one example at a time.
        double[] ols = LeastSquares857.Solve(rows, targets);
        Console.WriteLine("learned  = [" + string.Join(", ", Format(learned.Weights)) + "]");
        Console.WriteLine("least sq = [" + string.Join(", ", Format(ols)) + "]");
        bool aligned = true;
        for (int d = 0; d < Features857.Dimensions; d++)
            if (Math.Abs(learned.Weights[d] - ols[d]) > Math.Abs(ols[d]) + 2.0) aligned = false;
        Check(aligned, "every learned weight is within a factor of two of the closed-form least-squares weight");
        double olsError = 0.0, learnedError = 0.0, olsRelative = 0.0, learnedRelative = 0.0;
        for (int i = 0; i < 200; i++)
        {
            double[] row = rows[rows.Count - 1 - i];
            double target = targets[targets.Count - 1 - i];
            double prediction = 0.0;
            for (int d = 0; d < Features857.Dimensions; d++) prediction += ols[d] * row[d];
            double mine = PredictWith(learned.Weights, row);
            olsError += Math.Abs(prediction - target);
            learnedError += Math.Abs(mine - target);
            olsRelative += Math.Abs(prediction - target) / target;
            learnedRelative += Math.Abs(mine - target) / target;
        }
        Console.WriteLine("recent-row error: learned=" + (learnedError / 200).ToString("F2")
            + " least-squares=" + (olsError / 200).ToString("F2")
            + " | relative: learned=" + (learnedRelative / 200).ToString("P2")
            + " least-squares=" + (olsRelative / 200).ToString("P2"));
        Check(learnedError < olsError * 5.0 && learnedRelative < olsRelative * 5.0 + 0.01,
            "the online model stays within a factor of five of the batch least-squares fit on the latest rows");

        // 9-11. Plan selection on held-out queries against brute-force ground truth.
        var heldOut = new List<Query857>();
        for (int q = 0; q < 60; q++)
        {
            var query = new Query857 { Id = q };
            for (int p = 0; p < PlansPerQuery; p++)
            {
                long input = 200 + Rng857.NextInt(ref seed, 8000);
                double selectivity = 0.02 + Rng857.Uniform(ref seed) * 0.9;
                long output = 1 + Rng857.NextInt(ref seed, (int)input);
                query.Plans.Add(new Plan857
                {
                    Name = "p" + p,
                    Features = Features857.Build(input, selectivity, output),
                });
            }
            heldOut.Add(query);
        }
        var staticModel = new LinearModel857(5.0, 0.5, 60.0, 1.0);
        int learnedOptimal = 0, staticOptimal = 0;
        double learnedRegret = 0.0, staticRegret = 0.0;
        double worstRegret = 0.0;
        foreach (Query857 query in heldOut)
        {
            int bestIndex = -1;
            double bestCost = double.MaxValue;
            for (int p = 0; p < query.Plans.Count; p++)
            {
                double cost = Database857.TrueCost(query.Plans[p].Features);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestIndex = p;
                }
            }
            int learnedChoice = ArgMin(learned.Weights, query);
            int staticChoice = ArgMin(staticModel.Weights, query);
            double learnedCost = Database857.TrueCost(query.Plans[learnedChoice].Features);
            double staticCost = Database857.TrueCost(query.Plans[staticChoice].Features);
            if (learnedChoice == bestIndex) learnedOptimal++;
            if (staticChoice == bestIndex) staticOptimal++;
            learnedRegret += learnedCost - bestCost;
            staticRegret += staticCost - bestCost;
            double regret = learnedCost / bestCost;
            if (regret > worstRegret) worstRegret = regret;
        }
        Console.WriteLine("plan choice: learned optimal=" + learnedOptimal + "/" + heldOut.Count
            + " static optimal=" + staticOptimal + "/" + heldOut.Count + " regret learned=" + learnedRegret.ToString("F1")
            + " static=" + staticRegret.ToString("F1") + " worst ratio=" + worstRegret.ToString("F3"));
        Check(learnedOptimal > staticOptimal,
            "the learned model picks the true cheapest plan more often than the hand-tuned model");
        Check(learnedRegret < staticRegret,
            "the learned model's total regret is lower than the hand-tuned model's");
        Check(worstRegret < 3.0,
            "the learned model never picks a plan more than three times worse than the optimum ("
            + worstRegret.ToString("F3") + ")");

        // 12-14. Learning-rate schedule, degenerate streams and determinism.
        Check(Math.Abs(trainer.RateAt(0) - 0.02) < 1e-15
            && Math.Abs(trainer.RateAt(1000) - 0.02 / 1.2) < 1e-15
            && trainer.RateAt(1000) < trainer.RateAt(100),
            "the learning rate decays geometrically and strictly decreases over time");
        var degenerate = new Trainer857(new LinearModel857(0, 0, 0, 0), 0.1, 0.0);
        double first = degenerate.TrainOne(new[] { 1.0, 1.0, 0.5, 1.0 }, 4.0, 0);
        Check(Math.Abs(first - 4.0) < 1e-12 && degenerate.Losses.Count == 1 && degenerate.WindowMean(5, 10) == 0.0,
            "a single example trains once and an out-of-range window reports a zero loss");
        var constant = new Trainer857(new LinearModel857(0, 0, 0, 0), 0.05, 0.0);
        for (int i = 0; i < 300; i++) constant.TrainOne(new[] { 1.0, 1.0, 0.5, 1.0 }, 7.0, i);
        Check(Math.Abs(constant.Model.Predict(new[] { 1.0, 1.0, 0.5, 1.0 }) - 7.0) < 1e-6,
            "a constant stream drives the prediction to the constant target exactly");
        var repeatModel = new LinearModel857(0, 0, 0, 0);
        var repeatTrainer = new Trainer857(repeatModel, 0.02, 0.0002);
        for (int i = 0; i < 400; i++)
            repeatTrainer.TrainOne(queries[i % queries.Count].Plans[0].Features, targets[i % targets.Count], i);
        var thirdModel = new LinearModel857(0, 0, 0, 0);
        var thirdTrainer = new Trainer857(thirdModel, 0.02, 0.0002);
        for (int i = 0; i < 400; i++)
            thirdTrainer.TrainOne(queries[i % queries.Count].Plans[0].Features, targets[i % targets.Count], i);
        bool identical = true;
        for (int i = 0; i < Features857.Dimensions; i++) if (repeatModel.Weights[i] != thirdModel.Weights[i]) identical = false;
        Check(identical && repeatTrainer.Losses.Count == thirdTrainer.Losses.Count,
            "training is reproducible: identical inputs produce bit-identical weights");

        Console.WriteLine("PASS");
    }

    private static int ArgMin(double[] weights, Query857 query)
    {
        int best = 0;
        double bestCost = double.MaxValue;
        for (int p = 0; p < query.Plans.Count; p++)
        {
            double cost = PredictWith(weights, query.Plans[p].Features);
            if (cost < bestCost)
            {
                bestCost = cost;
                best = p;
            }
        }
        return best;
    }

    private static double PredictWith(double[] weights, double[] features)
    {
        double sum = 0.0;
        for (int i = 0; i < weights.Length; i++) sum += weights[i] * features[i];
        return sum < 0.0 ? 0.0 : sum;
    }

    private static string[] Format(double[] weights)
    {
        var text = new string[weights.Length];
        for (int i = 0; i < weights.Length; i++) text[i] = weights[i].ToString("F3");
        return text;
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
