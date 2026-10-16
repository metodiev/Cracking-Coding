// Task 519 - Implement confusion matrix + precision/recall metrics calculator. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// A single-label confusion matrix over an explicit or data-derived label set: Counts[t][p] counts the
// samples whose true class is Labels[t] and whose predicted class is Labels[p], so the row sums are
// the class supports, the column sums the predicted counts and the trace the number of correct
// predictions. Per class k we take tp = Counts[k][k], fp = Predicted[k] - tp, fn = Support[k] - tp and
// derive precision tp/(tp+fp), recall tp/(tp+fn) and their harmonic mean F1, with the documented
// convention that a ratio with a zero denominator (a class that is never predicted, or that never
// occurs but is listed explicitly) is reported as 0.0 rather than NaN. Aggregates: accuracy =
// trace / n; macro = the unweighted mean over the label set (classes with zero support count as 0);
// weighted = the support-weighted mean; micro = pooled tp / (tp + fp) and tp / (tp + fn), which for
// single-label multi-class data coincides with the accuracy. When no label set is supplied it is the
// sorted union of the observed truth and prediction values, and a truth or prediction value outside a
// supplied set is rejected.
//
// Documented checks. (1) Hand-checked 3-class example (truth 0,0,0,1,1,2,2 / prediction 0,2,0,1,0,2,1):
// the full 3x3 matrix, the per-class precision/recall/F1 values (2/3, 1/2, 1/2), accuracy 4/7, the
// macro averages 5/9 and the weighted averages 4/7 are compared with tolerance 1e-12; macro-F1
// is re-derived as the plain mean of the per-class F1s and weighted-F1 as the support-weighted mean,
// and every row sum / column sum is checked against the support / predicted counts. (2) Independent
// brute force: for five LCG data sets (200 samples, four classes) a nested-loop tp/fp/fn count feeds a
// second metric computation whose per-class and aggregated values must match the matrix-based ones to
// 1e-12. (3) Identities: micro precision = micro recall = micro F1 = accuracy on every data set.
// (4) Boundaries: a perfect classifier scores 1.0 everywhere, a systematically inverted binary
// classifier scores 0.0 everywhere, a class that is never predicted gets precision 0 (0/0
// convention) while the other class keeps its values, a listed-but-absent class gets all-zero metrics
// with the weighted aggregates unchanged and the macro averages diluted, and length mismatch, empty
// input, an unlisted label and duplicate labels each throw ArgumentException.
using System;
using System.Collections.Generic;

internal sealed class ClassMetrics
{
    internal readonly int TruePositives;
    internal readonly int FalsePositives;
    internal readonly int FalseNegatives;
    internal readonly double Precision;
    internal readonly double Recall;
    internal readonly double F1;

    internal ClassMetrics(int truePositives, int falsePositives, int falseNegatives)
    {
        TruePositives = truePositives;
        FalsePositives = falsePositives;
        FalseNegatives = falseNegatives;
        Precision = truePositives + falsePositives == 0
                ? 0.0 : (double)truePositives / (truePositives + falsePositives);
        Recall = truePositives + falseNegatives == 0
                ? 0.0 : (double)truePositives / (truePositives + falseNegatives);
        F1 = Precision + Recall == 0.0 ? 0.0 : 2.0 * Precision * Recall / (Precision + Recall);
    }

    internal static double Harmonic(double precision, double recall)
    {
        return precision + recall == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);
    }
}

internal sealed class AggregateMetrics
{
    internal double Accuracy;
    internal double MicroPrecision;
    internal double MicroRecall;
    internal double MicroF1;
    internal double MacroPrecision;
    internal double MacroRecall;
    internal double MacroF1;
    internal double WeightedPrecision;
    internal double WeightedRecall;
    internal double WeightedF1;
}

internal sealed class ConfusionMatrix
{
    internal int[] Labels { get; }
    internal int[][] Counts { get; }
    internal int[] Support { get; }
    internal int[] Predicted { get; }
    internal int Total { get; }
    internal int Correct { get; }

    internal ConfusionMatrix(int[] truth, int[] predicted, int[]? labelSet = null)
    {
        if (truth.Length != predicted.Length)
        {
            throw new ArgumentException("truth and prediction have different lengths");
        }
        if (truth.Length == 0) throw new ArgumentException("empty input");
        Labels = labelSet == null ? Union(truth, predicted) : (int[])labelSet.Clone();
        if (Labels.Length == 0) throw new ArgumentException("empty label set");
        var position = new Dictionary<int, int>();
        for (int i = 0; i < Labels.Length; i++)
        {
            if (position.ContainsKey(Labels[i])) throw new ArgumentException("duplicate label " + Labels[i]);
            position[Labels[i]] = i;
        }
        Counts = new int[Labels.Length][];
        for (int i = 0; i < Labels.Length; i++) Counts[i] = new int[Labels.Length];
        Support = new int[Labels.Length];
        Predicted = new int[Labels.Length];
        for (int i = 0; i < truth.Length; i++)
        {
            int actual = Lookup(position, truth[i], "truth");
            int guess = Lookup(position, predicted[i], "prediction");
            Counts[actual][guess]++;
            Support[actual]++;
            Predicted[guess]++;
            if (actual == guess) Correct++;
        }
        Total = truth.Length;
    }

    private static int Lookup(Dictionary<int, int> position, int label, string role)
    {
        if (!position.TryGetValue(label, out int index))
        {
            throw new ArgumentException(role + " label " + label + " is missing from the label set");
        }
        return index;
    }

    private static int[] Union(int[] truth, int[] predicted)
    {
        var seen = new SortedSet<int>();
        foreach (int value in truth) seen.Add(value);
        foreach (int value in predicted) seen.Add(value);
        var result = new int[seen.Count];
        seen.CopyTo(result);
        return result;
    }

    internal ClassMetrics MetricsOf(int index)
    {
        int truePositives = Counts[index][index];
        return new ClassMetrics(truePositives, Predicted[index] - truePositives,
                Support[index] - truePositives);
    }

    internal AggregateMetrics Aggregates()
    {
        var result = new AggregateMetrics();
        result.Accuracy = (double)Correct / Total;
        int pooledTp = 0;
        int pooledFp = 0;
        int pooledFn = 0;
        double macroP = 0.0;
        double macroR = 0.0;
        double macroF1 = 0.0;
        double weightedP = 0.0;
        double weightedR = 0.0;
        double weightedF1 = 0.0;
        for (int k = 0; k < Labels.Length; k++)
        {
            ClassMetrics metrics = MetricsOf(k);
            pooledTp += metrics.TruePositives;
            pooledFp += metrics.FalsePositives;
            pooledFn += metrics.FalseNegatives;
            macroP += metrics.Precision;
            macroR += metrics.Recall;
            macroF1 += metrics.F1;
            weightedP += Support[k] * metrics.Precision;
            weightedR += Support[k] * metrics.Recall;
            weightedF1 += Support[k] * metrics.F1;
        }
        int classes = Labels.Length;
        result.MacroPrecision = macroP / classes;
        result.MacroRecall = macroR / classes;
        result.MacroF1 = macroF1 / classes;
        result.WeightedPrecision = weightedP / Total;
        result.WeightedRecall = weightedR / Total;
        result.WeightedF1 = weightedF1 / Total;
        result.MicroPrecision = pooledTp + pooledFp == 0 ? 0.0 : (double)pooledTp / (pooledTp + pooledFp);
        result.MicroRecall = pooledTp + pooledFn == 0 ? 0.0 : (double)pooledTp / (pooledTp + pooledFn);
        result.MicroF1 = ClassMetrics.Harmonic(result.MicroPrecision, result.MicroRecall);
        return result;
    }
}

public static class Solution
{
    public static void Main()
    {
        int[] truth = { 0, 0, 0, 1, 1, 2, 2 };
        int[] predicted = { 0, 2, 0, 1, 0, 2, 1 };
        var matrix = new ConfusionMatrix(truth, predicted);
        Check(matrix.Labels.Length == 3, "labels derived from the data");
        Check(matrix.Labels[0] == 0 && matrix.Labels[1] == 1 && matrix.Labels[2] == 2, "labels sorted");
        Check(matrix.Total == 7 && matrix.Correct == 4, "total and correct counts");
        Check(matrix.Counts[0][0] == 2 && matrix.Counts[0][1] == 0 && matrix.Counts[0][2] == 1,
                "row 0 of the hand-checked example");
        Check(matrix.Counts[1][0] == 1 && matrix.Counts[1][1] == 1 && matrix.Counts[1][2] == 0,
                "row 1 of the hand-checked example");
        Check(matrix.Counts[2][0] == 0 && matrix.Counts[2][1] == 1 && matrix.Counts[2][2] == 1,
                "row 2 of the hand-checked example");

        ClassMetrics class0 = matrix.MetricsOf(0);
        ClassMetrics class1 = matrix.MetricsOf(1);
        ClassMetrics class2 = matrix.MetricsOf(2);
        Check(class0.TruePositives == 2 && class0.FalsePositives == 1 && class0.FalseNegatives == 1,
                "class 0 tp/fp/fn");
        CheckClose(class0.Precision, 2.0 / 3.0, "class 0 precision");
        CheckClose(class0.Recall, 2.0 / 3.0, "class 0 recall");
        CheckClose(class0.F1, 2.0 / 3.0, "class 0 F1");
        Check(class1.TruePositives == 1 && class1.FalsePositives == 1 && class1.FalseNegatives == 1,
                "class 1 tp/fp/fn");
        CheckClose(class1.Precision, 0.5, "class 1 precision");
        CheckClose(class1.Recall, 0.5, "class 1 recall");
        CheckClose(class1.F1, 0.5, "class 1 F1");
        Check(class2.TruePositives == 1 && class2.FalsePositives == 1 && class2.FalseNegatives == 1,
                "class 2 tp/fp/fn");
        CheckClose(class2.Precision, 0.5, "class 2 precision");
        CheckClose(class2.Recall, 0.5, "class 2 recall");
        CheckClose(class2.F1, 0.5, "class 2 F1");

        AggregateMetrics aggregates = matrix.Aggregates();
        CheckClose(aggregates.Accuracy, 4.0 / 7.0, "hand-checked accuracy");
        CheckClose(aggregates.MacroPrecision, 5.0 / 9.0, "hand-checked macro precision");
        CheckClose(aggregates.MacroRecall, 5.0 / 9.0, "hand-checked macro recall");
        CheckClose(aggregates.MacroF1, 5.0 / 9.0, "hand-checked macro F1");
        CheckClose(aggregates.MicroPrecision, 4.0 / 7.0, "hand-checked micro precision");
        CheckClose(aggregates.MicroRecall, 4.0 / 7.0, "hand-checked micro recall");
        CheckClose(aggregates.MicroF1, 4.0 / 7.0, "hand-checked micro F1");
        CheckClose(aggregates.WeightedPrecision, 4.0 / 7.0, "hand-checked weighted precision");
        CheckClose(aggregates.WeightedRecall, 4.0 / 7.0, "hand-checked weighted recall");
        CheckClose(aggregates.WeightedF1, 4.0 / 7.0, "hand-checked weighted F1");
        CheckClose(aggregates.Accuracy, (double)matrix.Correct / matrix.Total, "accuracy from the trace");
        CheckClose(aggregates.MacroF1, (class0.F1 + class1.F1 + class2.F1) / 3.0, "macro F1 is a plain mean");
        CheckClose(aggregates.WeightedF1, (3 * class0.F1 + 2 * class1.F1 + 2 * class2.F1) / 7.0,
                "weighted F1 is support-weighted");
        CheckRowAndColumnSums(matrix);

        foreach (int seed in new[] { 11, 29, 47, 71, 97 })
        {
            int[][] data = LcgData(200, seed, 4);
            var local = new ConfusionMatrix(data[0], data[1]);
            CompareWithBruteForce(local, data[0], data[1], "seed " + seed);
        }

        var perfect = new ConfusionMatrix(truth, truth);
        AggregateMetrics perfectAggregates = perfect.Aggregates();
        CheckClose(perfectAggregates.Accuracy, 1.0, "perfect accuracy");
        CheckClose(perfectAggregates.MacroPrecision, 1.0, "perfect macro precision");
        CheckClose(perfectAggregates.MacroRecall, 1.0, "perfect macro recall");
        CheckClose(perfectAggregates.MacroF1, 1.0, "perfect macro F1");
        CheckClose(perfectAggregates.WeightedF1, 1.0, "perfect weighted F1");
        Check(perfect.Correct == perfect.Total, "perfect classifier scores every sample right");

        int[] binaryTruth = { 0, 0, 1, 1 };
        var worst = new ConfusionMatrix(binaryTruth, new[] { 1, 1, 0, 0 });
        AggregateMetrics worstAggregates = worst.Aggregates();
        CheckClose(worstAggregates.Accuracy, 0.0, "inverted accuracy");
        CheckClose(worstAggregates.MacroPrecision, 0.0, "inverted macro precision");
        CheckClose(worstAggregates.MacroRecall, 0.0, "inverted macro recall");
        CheckClose(worstAggregates.MacroF1, 0.0, "inverted macro F1");
        CheckClose(worst.MetricsOf(0).Precision, 0.0, "inverted class 0 precision");

        var neverPredicted = new ConfusionMatrix(new[] { 0, 1, 1, 1 }, new[] { 0, 0, 0, 0 });
        CheckClose(neverPredicted.MetricsOf(1).Precision, 0.0, "never predicted class precision is 0/0 -> 0");
        CheckClose(neverPredicted.MetricsOf(1).Recall, 0.0, "never predicted class recall is 0");
        CheckClose(neverPredicted.MetricsOf(1).F1, 0.0, "never predicted class F1 is 0");
        CheckClose(neverPredicted.MetricsOf(0).Precision, 0.25, "never predicted: other class precision");
        CheckClose(neverPredicted.MetricsOf(0).Recall, 1.0, "never predicted: other class recall");
        CheckClose(neverPredicted.MetricsOf(0).F1, 0.4, "never predicted: other class F1");
        CheckClose(neverPredicted.Aggregates().Accuracy, 0.25, "never predicted: accuracy");
        CheckClose(neverPredicted.Aggregates().MacroPrecision, 0.125, "never predicted: macro precision");
        CheckClose(neverPredicted.Aggregates().MacroRecall, 0.5, "never predicted: macro recall");
        CheckClose(neverPredicted.Aggregates().MacroF1, 0.2, "never predicted: macro F1");

        var withAbsent = new ConfusionMatrix(new[] { 0, 0, 1, 1 }, new[] { 0, 1, 1, 1 }, new[] { 0, 1, 2 });
        var withoutAbsent = new ConfusionMatrix(new[] { 0, 0, 1, 1 }, new[] { 0, 1, 1, 1 });
        CheckClose(withAbsent.MetricsOf(2).Precision, 0.0, "listed but absent class precision");
        CheckClose(withAbsent.MetricsOf(2).Recall, 0.0, "listed but absent class recall");
        CheckClose(withAbsent.MetricsOf(2).F1, 0.0, "listed but absent class F1");
        CheckClose(withAbsent.MetricsOf(0).Precision, 1.0, "absent class leaves class 0 precision alone");
        CheckClose(withAbsent.MetricsOf(0).Recall, 0.5, "absent class leaves class 0 recall alone");
        CheckClose(withAbsent.MetricsOf(1).F1, 0.8, "absent class leaves class 1 F1 alone");
        CheckClose(withAbsent.Aggregates().WeightedF1, withoutAbsent.Aggregates().WeightedF1,
                "an absent class carries no support weight");
        Check(withAbsent.Aggregates().MacroF1 < withoutAbsent.Aggregates().MacroF1,
                "an absent class dilutes the macro averages");
        CheckClose(withAbsent.Aggregates().Accuracy, withoutAbsent.Aggregates().Accuracy,
                "an absent class does not change accuracy");

        CheckThrows(() => new ConfusionMatrix(new[] { 0, 1 }, new[] { 0 }), "length mismatch throws");
        CheckThrows(() => new ConfusionMatrix(new int[0], new int[0]), "empty input throws");
        CheckThrows(() => new ConfusionMatrix(new[] { 0, 1 }, new[] { 0, 1 }, new[] { 0, 3 }),
                "unlisted prediction throws");
        CheckThrows(() => new ConfusionMatrix(new[] { 0, 1 }, new[] { 0, 1 }, new[] { 0, 0 }),
                "duplicate labels throw");
        CheckThrows(() => new ConfusionMatrix(new[] { 0, 3 }, new[] { 0, 1 }, new[] { 0, 1 }),
                "unlisted truth throws");
        CheckThrows(() => new ConfusionMatrix(new[] { 0, 1 }, new[] { 0, 1 }, new int[0]),
                "empty label set throws");

        Console.WriteLine("PASS");
    }

    private static void CheckRowAndColumnSums(ConfusionMatrix matrix)
    {
        for (int i = 0; i < matrix.Labels.Length; i++)
        {
            int row = 0;
            int column = 0;
            for (int j = 0; j < matrix.Labels.Length; j++)
            {
                row += matrix.Counts[i][j];
                column += matrix.Counts[j][i];
            }
            Check(row == matrix.Support[i], "row sum equals the support of class " + i);
            Check(column == matrix.Predicted[i], "column sum equals the predictions of class " + i);
        }
    }

    private static void CompareWithBruteForce(ConfusionMatrix matrix, int[] truth, int[] predicted,
            string label)
    {
        foreach (int classLabel in matrix.Labels)
        {
            int position = Array.IndexOf(matrix.Labels, classLabel);
            int truePositives = 0;
            int falsePositives = 0;
            int falseNegatives = 0;
            for (int i = 0; i < truth.Length; i++)
            {
                if (truth[i] == classLabel && predicted[i] == classLabel) truePositives++;
                if (truth[i] != classLabel && predicted[i] == classLabel) falsePositives++;
                if (truth[i] == classLabel && predicted[i] != classLabel) falseNegatives++;
            }
            ClassMetrics brute = new ClassMetrics(truePositives, falsePositives, falseNegatives);
            ClassMetrics fast = matrix.MetricsOf(position);
            Check(fast.TruePositives == truePositives && fast.FalsePositives == falsePositives
                    && fast.FalseNegatives == falseNegatives, label + ": counts for class " + classLabel);
            CheckClose(fast.Precision, brute.Precision, label + ": precision for class " + classLabel);
            CheckClose(fast.Recall, brute.Recall, label + ": recall for class " + classLabel);
            CheckClose(fast.F1, brute.F1, label + ": F1 for class " + classLabel);
        }

        int correct = 0;
        for (int i = 0; i < truth.Length; i++)
        {
            if (truth[i] == predicted[i]) correct++;
        }
        AggregateMetrics aggregates = matrix.Aggregates();
        CheckClose(aggregates.Accuracy, (double)correct / truth.Length, label + ": accuracy");
        CheckClose(aggregates.MicroPrecision, aggregates.Accuracy, label + ": micro precision = accuracy");
        CheckClose(aggregates.MicroRecall, aggregates.Accuracy, label + ": micro recall = accuracy");
        CheckClose(aggregates.MicroF1, aggregates.Accuracy, label + ": micro F1 = accuracy");

        double macroP = 0.0;
        double macroR = 0.0;
        double macroF1 = 0.0;
        double weightedP = 0.0;
        double weightedR = 0.0;
        double weightedF1 = 0.0;
        int total = 0;
        for (int i = 0; i < matrix.Labels.Length; i++)
        {
            ClassMetrics metrics = matrix.MetricsOf(i);
            double support = matrix.Support[i];
            macroP += metrics.Precision;
            macroR += metrics.Recall;
            macroF1 += metrics.F1;
            weightedP += support * metrics.Precision;
            weightedR += support * metrics.Recall;
            weightedF1 += support * metrics.F1;
            total += matrix.Support[i];
        }
        Check(total == truth.Length, label + ": supports add up");
        CheckClose(aggregates.MacroPrecision, macroP / matrix.Labels.Length, label + ": macro precision");
        CheckClose(aggregates.MacroRecall, macroR / matrix.Labels.Length, label + ": macro recall");
        CheckClose(aggregates.MacroF1, macroF1 / matrix.Labels.Length, label + ": macro F1");
        CheckClose(aggregates.WeightedPrecision, weightedP / total, label + ": weighted precision");
        CheckClose(aggregates.WeightedRecall, weightedR / total, label + ": weighted recall");
        CheckClose(aggregates.WeightedF1, weightedF1 / total, label + ": weighted F1");
        Check(aggregates.Accuracy > 0.0 && aggregates.Accuracy < 1.0, label + ": accuracy strictly inside");
    }

    private static int[][] LcgData(int n, int seed, int classes)
    {
        var truth = new int[n];
        var predicted = new int[n];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            truth[i] = seed % classes;
            seed = (seed * 75 + 74) % 65537;
            predicted[i] = seed % classes;
        }
        return new[] { truth, predicted };
    }

    private static void CheckClose(double actual, double expected, string label)
    {
        if (!(Math.Abs(actual - expected) <= 1e-12))
        {
            throw new Exception("FAILED: " + label + " (" + actual + " != " + expected + ")");
        }
    }

    private static void CheckThrows(Action action, string label)
    {
        try
        {
            action();
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: " + label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
