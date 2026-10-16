// Task 520 - Implement ROC AUC estimator from predictions. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Three independent estimators of the ROC AUC for binary labels (0 = negative, 1 = positive) and
// real-valued scores, all documented and cross-checked against each other:
//   * Pairwise - the definition: the fraction of (positive, negative) pairs in which the positive
//     scores higher, counting a score tie as 1/2.
//   * ByRanks - the Mann-Whitney form (sum of the average ranks of the positives minus
//     np(np+1)/2) / (np*nn), where ranks are computed ascending with the average rank shared by all
//     equal scores.
//   * Curve - the empirical ROC curve built by walking the samples in descending score order and
//     moving each equal-score group diagonally through (fpr, tpr) space, then integrating with the
//     trapezoid rule. Because the ties of one group advance fpr and tpr simultaneously, the
//     trapezoid over these grouped points reproduces exactly the 1/2 credit of the pairwise
//     definition (verified below on all-tied and partially tied data).
// A second curve is built from scratch at each distinct threshold with the "score >= threshold"
// decision rule; its points must coincide with the group-end points of the sorted-path curve, which
// pins down the tie handling of both constructions.
//
// Documented checks. (1) Hand-checked examples: the four-sample set scores 0.1/0.4 (positive) and
// 0.35/0.8 (negative) has AUC 1/4 by all three methods, and a two-sample 1-vs-0 tie has AUC 1/2.
// (2) Degenerate cases: perfect separation gives 1.0, complete reversal gives 0.0, all-equal scores
// give 0.5, and a single positive surrounded by negatives gives the closed-form
// (number of negatives below it + 0.5 * ties) / nn. (3) Invariance: the AUC is unchanged by any
// strictly increasing transform of the scores (cube, affine map) and negating the scores replaces it
// by 1 - AUC. (4) Cross-checks: on five LCG data sets - two of them with heavily quantised scores so
// that tens of ties are present - the three methods agree to 1e-12, the curve has distinct+1 points,
// starts at (0,0), ends at (1,1) and is non-decreasing in both coordinates, both curves agree
// point-wise, and every estimate lies in [0,1]. (5) Errors: differing lengths, fewer than two
// samples, a label outside {0,1} and a single-class input each throw ArgumentException.
using System;
using System.Collections.Generic;

internal sealed class RocCurve
{
    internal readonly double[] Fpr;
    internal readonly double[] Tpr;
    internal readonly double[] Thresholds;

    internal RocCurve(double[] fpr, double[] tpr, double[] thresholds)
    {
        Fpr = fpr;
        Tpr = tpr;
        Thresholds = thresholds;
    }

    internal int Count => Fpr.Length;
}

internal static class RocAuc
{
    internal static int PositiveCount(int[] labels)
    {
        int positives = 0;
        foreach (int label in labels)
        {
            if (label == 1) positives++;
            else if (label != 0) throw new ArgumentException("labels must be 0 or 1, found " + label);
        }
        return positives;
    }

    internal static void Validate(double[] scores, int[] labels)
    {
        if (scores.Length != labels.Length)
        {
            throw new ArgumentException("scores and labels have different lengths");
        }
        if (scores.Length < 2) throw new ArgumentException("at least two samples are needed");
        int positives = PositiveCount(labels);
        if (positives == 0 || positives == labels.Length)
        {
            throw new ArgumentException("both classes must be present");
        }
    }

    /** The definition: the fraction of positive/negative pairs with the positive scoring higher. */
    internal static double Pairwise(double[] scores, int[] labels)
    {
        Validate(scores, labels);
        int positives = PositiveCount(labels);
        int negatives = scores.Length - positives;
        double credit = 0.0;
        for (int i = 0; i < scores.Length; i++)
        {
            if (labels[i] != 1) continue;
            for (int j = 0; j < scores.Length; j++)
            {
                if (labels[j] != 0) continue;
                if (scores[i] > scores[j]) credit += 1.0;
                else if (scores[i] == scores[j]) credit += 0.5;
            }
        }
        return credit / ((double)positives * negatives);
    }

    /** The Mann-Whitney form of the same quantity, using average ranks for tied scores. */
    internal static double ByRanks(double[] scores, int[] labels)
    {
        Validate(scores, labels);
        int n = scores.Length;
        var ranks = AverageRanks(scores);
        double rankSum = 0.0;
        int positives = 0;
        for (int i = 0; i < n; i++)
        {
            if (labels[i] == 1)
            {
                rankSum += ranks[i];
                positives++;
            }
        }
        int negatives = n - positives;
        return (rankSum - positives * (positives + 1) / 2.0) / ((double)positives * negatives);
    }

    internal static double[] AverageRanks(double[] scores)
    {
        int n = scores.Length;
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int cmp = scores[a].CompareTo(scores[b]);
            return cmp != 0 ? cmp : a.CompareTo(b);
        });
        var ranks = new double[n];
        int start = 0;
        while (start < n)
        {
            int end = start;
            while (end + 1 < n && scores[order[end + 1]] == scores[order[start]]) end++;
            double average = (start + end) / 2.0 + 1.0;
            for (int t = start; t <= end; t++) ranks[order[t]] = average;
            start = end + 1;
        }
        return ranks;
    }

    /** ROC from the score-sorted sample path; equal-score groups move diagonally. */
    internal static RocCurve SortedPathCurve(double[] scores, int[] labels)
    {
        Validate(scores, labels);
        int n = scores.Length;
        int positives = PositiveCount(labels);
        int negatives = n - positives;
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int cmp = scores[b].CompareTo(scores[a]);
            return cmp != 0 ? cmp : a.CompareTo(b);
        });
        var fpr = new List<double> { 0.0 };
        var tpr = new List<double> { 0.0 };
        var thresholds = new List<double>();
        int truePositives = 0;
        int falsePositives = 0;
        int start = 0;
        while (start < n)
        {
            int end = start;
            while (end + 1 < n && scores[order[end + 1]] == scores[order[start]]) end++;
            int groupPositives = 0;
            int groupNegatives = 0;
            for (int t = start; t <= end; t++)
            {
                if (labels[order[t]] == 1) groupPositives++;
                else groupNegatives++;
            }
            truePositives += groupPositives;
            falsePositives += groupNegatives;
            fpr.Add((double)falsePositives / negatives);
            tpr.Add((double)truePositives / positives);
            thresholds.Add(scores[order[start]]);
            start = end + 1;
        }
        return new RocCurve(fpr.ToArray(), tpr.ToArray(), thresholds.ToArray());
    }

    /** ROC built independently at every distinct threshold with the "score >= threshold" rule. */
    internal static RocCurve ThresholdCurve(double[] scores, int[] labels)
    {
        Validate(scores, labels);
        int positives = PositiveCount(labels);
        int negatives = scores.Length - positives;
        var distinct = new SortedSet<double>();
        foreach (double score in scores) distinct.Add(score);
        var descending = new List<double>(distinct);
        descending.Reverse();
        var fpr = new List<double> { 0.0 };
        var tpr = new List<double> { 0.0 };
        var thresholds = new List<double>();
        foreach (double threshold in descending)
        {
            int truePositives = 0;
            int falsePositives = 0;
            for (int i = 0; i < scores.Length; i++)
            {
                if (scores[i] >= threshold)
                {
                    if (labels[i] == 1) truePositives++;
                    else falsePositives++;
                }
            }
            tpr.Add((double)truePositives / positives);
            fpr.Add((double)falsePositives / negatives);
            thresholds.Add(threshold);
        }
        return new RocCurve(fpr.ToArray(), tpr.ToArray(), thresholds.ToArray());
    }

    internal static double Area(RocCurve curve)
    {
        double area = 0.0;
        for (int i = 1; i < curve.Count; i++)
        {
            double width = curve.Fpr[i] - curve.Fpr[i - 1];
            area += width * (curve.Tpr[i] + curve.Tpr[i - 1]) / 2.0;
        }
        return area;
    }

    internal static double CurveArea(double[] scores, int[] labels)
    {
        return Area(SortedPathCurve(scores, labels));
    }
}

public static class Solution
{
    public static void Main()
    {
        double[] scores = { 0.1, 0.4, 0.35, 0.8 };
        int[] labels = { 1, 1, 0, 0 };
        CheckClose(RocAuc.Pairwise(scores, labels), 0.25, "hand-checked pairwise AUC");
        CheckClose(RocAuc.ByRanks(scores, labels), 0.25, "hand-checked rank AUC");
        CheckClose(RocAuc.CurveArea(scores, labels), 0.25, "hand-checked trapezoid AUC");

        double[] tied = { 0.5, 0.5 };
        int[] tiedLabels = { 1, 0 };
        CheckClose(RocAuc.Pairwise(tied, tiedLabels), 0.5, "two-sample tie is worth 1/2");
        CheckClose(RocAuc.ByRanks(tied, tiedLabels), 0.5, "two-sample tie by ranks");
        CheckClose(RocAuc.CurveArea(tied, tiedLabels), 0.5, "two-sample tie by the trapezoid rule");

        double[] perfect = { 0.9, 0.7, 0.2, 0.1 };
        int[] perfectLabels = { 1, 1, 0, 0 };
        CheckClose(RocAuc.Pairwise(perfect, perfectLabels), 1.0, "perfect separation");
        CheckClose(RocAuc.ByRanks(perfect, perfectLabels), 1.0, "perfect separation by ranks");
        CheckClose(RocAuc.CurveArea(perfect, perfectLabels), 1.0, "perfect separation by the curve");

        double[] reversed = { 0.1, 0.2, 0.7, 0.9 };
        CheckClose(RocAuc.Pairwise(reversed, perfectLabels), 0.0, "complete reversal");
        CheckClose(RocAuc.ByRanks(reversed, perfectLabels), 0.0, "complete reversal by ranks");
        CheckClose(RocAuc.CurveArea(reversed, perfectLabels), 0.0, "complete reversal by the curve");

        double[] flat = { 0.3, 0.3, 0.3, 0.3 };
        CheckClose(RocAuc.Pairwise(flat, perfectLabels), 0.5, "all-equal scores give 1/2");
        CheckClose(RocAuc.ByRanks(flat, perfectLabels), 0.5, "all-equal scores by ranks");
        CheckClose(RocAuc.CurveArea(flat, perfectLabels), 0.5, "all-equal scores by the curve");

        double[] single = { 0.5, 0.1, 0.2, 0.5, 0.9 };
        int[] singleLabels = { 1, 0, 0, 0, 0 };
        CheckClose(RocAuc.Pairwise(single, singleLabels), (2.0 + 0.5) / 4.0, "single positive, closed form");
        CheckClose(RocAuc.ByRanks(single, singleLabels), (2.0 + 0.5) / 4.0, "single positive by ranks");
        CheckClose(RocAuc.CurveArea(single, singleLabels), (2.0 + 0.5) / 4.0, "single positive by the curve");

        var curve = RocAuc.SortedPathCurve(scores, labels);
        Check(curve.Count == 5, "the curve of four distinct scores has five points");
        CheckClose(curve.Fpr[0], 0.0, "curve starts at fpr 0");
        CheckClose(curve.Tpr[0], 0.0, "curve starts at tpr 0");
        CheckClose(curve.Fpr[curve.Count - 1], 1.0, "curve ends at fpr 1");
        CheckClose(curve.Tpr[curve.Count - 1], 1.0, "curve ends at tpr 1");
        CheckClose(curve.Thresholds[0], 0.8, "thresholds descend");

        int[] dataSeeds = { 101, 211, 313, 419, 521 };
        int[] levels = { 0, 0, 3, 3, 2 };
        for (int s = 0; s < dataSeeds.Length; s++)
        {
            double[][] data = LcgScores(120, dataSeeds[s], levels[s]);
            double[] localScores = data[0];
            int[] localLabels = ToLabels(data[1]);
            string tag = "seed " + dataSeeds[s];
            double pairwise = RocAuc.Pairwise(localScores, localLabels);
            double ranks = RocAuc.ByRanks(localScores, localLabels);
            double area = RocAuc.CurveArea(localScores, localLabels);
            CheckClose(pairwise, ranks, tag + ": pairwise = ranks");
            CheckClose(pairwise, area, tag + ": pairwise = trapezoid");
            Check(pairwise >= 0.0 && pairwise <= 1.0, tag + ": AUC in [0, 1]");
            CheckCurve(RocAuc.SortedPathCurve(localScores, localLabels), tag + " sorted curve");
            CheckCurve(RocAuc.ThresholdCurve(localScores, localLabels), tag + " threshold curve");
            CheckSamePoints(RocAuc.SortedPathCurve(localScores, localLabels),
                    RocAuc.ThresholdCurve(localScores, localLabels), tag);

            var transformed = new double[localScores.Length];
            for (int i = 0; i < transformed.Length; i++)
            {
                transformed[i] = localScores[i] * localScores[i] * localScores[i];
            }
            CheckClose(RocAuc.Pairwise(transformed, localLabels), pairwise,
                    tag + ": the cube is a strictly increasing transform");
            var affine = new double[localScores.Length];
            for (int i = 0; i < affine.Length; i++) affine[i] = 2.0 * localScores[i] + 1.0;
            CheckClose(RocAuc.Pairwise(affine, localLabels), pairwise,
                    tag + ": affine transforms do not change the AUC");
            var negated = new double[localScores.Length];
            for (int i = 0; i < negated.Length; i++) negated[i] = -localScores[i];
            CheckClose(RocAuc.Pairwise(negated, localLabels), 1.0 - pairwise,
                    tag + ": negation complements the AUC");
            if (levels[s] != 0) Check(TieCount(localScores) > 0, tag + ": the quantised dataset has tied scores");
            else Check(TieCount(localScores) == 0, tag + ": the continuous dataset has no tied scores");
        }

        double[][] signal = LcgSignal(200, 8080, 1.0, 1.0, 0.05);
        double[] signalScores = signal[0];
        int[] signalLabels = ToLabels(signal[1]);
        double signalAuc = RocAuc.Pairwise(signalScores, signalLabels);
        CheckClose(signalAuc, RocAuc.ByRanks(signalScores, signalLabels), "signal: pairwise = ranks");
        CheckClose(signalAuc, RocAuc.CurveArea(signalScores, signalLabels), "signal: pairwise = trapezoid");
        Check(signalAuc > 0.6 && signalAuc < 1.0, "an informative signal scores well above chance");
        Check(TieCount(signalScores) > 0, "the rounded signal dataset has tied scores");

        CheckThrows(() => RocAuc.Pairwise(new[] { 0.1, 0.2 }, new[] { 1 }), "length mismatch throws");
        CheckThrows(() => RocAuc.Pairwise(new[] { 0.1 }, new[] { 1 }), "single sample throws");
        CheckThrows(() => RocAuc.Pairwise(new[] { 0.1, 0.2 }, new[] { 1, 2 }), "label 2 throws");
        CheckThrows(() => RocAuc.Pairwise(new[] { 0.1, 0.2 }, new[] { 1, 1 }), "single class throws");
        CheckThrows(() => RocAuc.CurveArea(new[] { 0.1, 0.2 }, new[] { 0, 0 }), "single class throws for curves");

        Console.WriteLine("PASS");
    }

    private static int TieCount(double[] scores)
    {
        var distinct = new HashSet<double>();
        foreach (double score in scores) distinct.Add(score);
        return scores.Length - distinct.Count;
    }

    private static void CheckCurve(RocCurve curve, string tag)
    {
        CheckClose(curve.Fpr[0], 0.0, tag + ": starts at fpr 0");
        CheckClose(curve.Tpr[0], 0.0, tag + ": starts at tpr 0");
        CheckClose(curve.Fpr[curve.Count - 1], 1.0, tag + ": ends at fpr 1");
        CheckClose(curve.Tpr[curve.Count - 1], 1.0, tag + ": ends at tpr 1");
        for (int i = 1; i < curve.Count; i++)
        {
            Check(curve.Fpr[i] >= curve.Fpr[i - 1], tag + ": fpr is non-decreasing at " + i);
            Check(curve.Tpr[i] >= curve.Tpr[i - 1], tag + ": tpr is non-decreasing at " + i);
        }
        for (int i = 1; i < curve.Thresholds.Length; i++)
        {
            Check(curve.Thresholds[i - 1] > curve.Thresholds[i], tag + ": thresholds strictly descend");
        }
    }

    private static void CheckSamePoints(RocCurve a, RocCurve b, string tag)
    {
        Check(a.Count == b.Count, tag + ": both curves have the same number of points");
        for (int i = 0; i < a.Count; i++)
        {
            CheckClose(a.Fpr[i], b.Fpr[i], tag + ": fpr at point " + i);
            CheckClose(a.Tpr[i], b.Tpr[i], tag + ": tpr at point " + i);
        }
    }

    private static int[] ToLabels(double[] values)
    {
        var labels = new int[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            labels[i] = (int)values[i];
        }
        return labels;
    }

    /** LCG data; levels = 0 means continuous scores, otherwise them all is quantised to that many steps. */
    private static double[][] LcgScores(int n, int seed, int levels)
    {
        var scores = new double[n];
        var labels = new double[n];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            labels[i] = seed % 2;
            seed = (seed * 75 + 74) % 65537;
            scores[i] = levels == 0 ? seed / 65537.0 : seed % levels;
        }
        return new[] { scores, labels };
    }

    private static double[][] LcgSignal(int n, int seed, double signal, double noise, double quantum)
    {
        var scores = new double[n];
        var labels = new double[n];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            labels[i] = seed % 2;
            seed = (seed * 75 + 74) % 65537;
            double jitter = seed % 1001 / 1000.0 * noise;
            scores[i] = Math.Round(labels[i] * signal + jitter, 2);
        }
        return new[] { scores, labels };
    }

    private static void CheckClose(double actual, double expected, string label)
    {
        double tolerance = 1e-12 * Math.Max(1.0, Math.Abs(expected));
        if (!(Math.Abs(actual - expected) <= tolerance))
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
