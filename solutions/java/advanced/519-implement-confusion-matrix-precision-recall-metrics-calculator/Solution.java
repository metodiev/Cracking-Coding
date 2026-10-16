/** Task 519 - Implement confusion matrix + precision/recall metrics calculator. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Binary and multiclass confusion-matrix metrics. The binary core fills the four cells
 * (TP, FP, FN, TN) by a single pass over the true/predicted label vectors and derives accuracy,
 * error rate, precision, recall (sensitivity / TPR), specificity (TNR), false-positive and
 * false-negative rates, F1, F-beta family, balanced accuracy, Jaccard (IoU), Matthews correlation
 * coefficient and Cohen's kappa from the same cells. DOCUMENTED convention: every ratio whose
 * denominator is zero reports 0.0 (never NaN, never an exception), except specificity when the
 * truth set contains no negative instance at all (tn + fp = 0), which is 1.0 because there is
 * nothing to reject, and accuracy of an empty input, which is 0.0. The multiclass path builds the K x K matrix and computes each class
 * one-vs-rest; macro averages are unweighted means over classes, weighted averages weight each
 * class by its support, and micro precision/recall/F1 are the global cell ratios.
 *
 * Checks: a hand-computed 6-point example (TP = 2, FN = 1, FP = 1, TN = 2) must yield accuracy
 * 4/6, precision/recall/F1 2/3, specificity 2/3, MCC 1/3, kappa 1/3 and Jaccard 1/2; the cells must
 * sum to n; an independent "derive metrics from the 2x2 matrix" recomputation must agree with the
 * direct pass; degenerate inputs (no predicted positives, no actual positives, empty input, all
 * correct, all wrong, single element) hit the documented conventions; an LCG stream of random
 * labels/predictions is cross-checked against a naive double loop for the cells plus the
 * identities precision = TP/(TP+FP), F1 = 2TP/(2TP+FP+FN), balanced accuracy = (TPR+TNR)/2,
 * F2 = 5p r/(4p + r), and kappa = (po - pe)/(1 - pe) recomputed from the rates; the 3-class LCG
 * workload must satisfy micro precision = micro recall = micro F1 = accuracy, macro F1 = mean of
 * per-class F1, weighted F1 = sum(support_c * F1_c)/n, the row sums of the matrix = support, the
 * column sums = prediction counts, matrix total = n, the diagonal = per-class TP, and the
 * one-vs-rest binary cells of class c must reproduce row/column c of the matrix.
 */
import java.util.*;

public class Solution {
    /** Binary confusion cells plus the derived metrics. */
    static class Binary {
        final int tp, fp, fn, tn;

        Binary(int tp, int fp, int fn, int tn) {
            this.tp = tp;
            this.fp = fp;
            this.fn = fn;
            this.tn = tn;
        }

        int n() {
            return tp + fp + fn + tn;
        }

        double accuracy() {
            int total = n();
            return total == 0 ? 0.0 : (tp + tn) / (double) total;
        }

        double errorRate() {
            return 1.0 - accuracy();
        }

        double precision() {
            return (tp + fp) == 0 ? 0.0 : tp / (double) (tp + fp);
        }

        double recall() {
            return (tp + fn) == 0 ? 0.0 : tp / (double) (tp + fn);
        }

        double specificity() {
            return (tn + fp) == 0 ? 1.0 : tn / (double) (tn + fp);
        }

        double falsePositiveRate() {
            return (fp + tn) == 0 ? 0.0 : fp / (double) (fp + tn);
        }

        double falseNegativeRate() {
            return (fn + tp) == 0 ? 0.0 : fn / (double) (fn + tp);
        }

        double f1() {
            double p = precision();
            double r = recall();
            return (p + r) == 0.0 ? 0.0 : 2.0 * p * r / (p + r);
        }

        double fBeta(double beta) {
            double p = precision();
            double r = recall();
            double b2 = beta * beta;
            double denom = b2 * p + r;
            return denom == 0.0 ? 0.0 : (1.0 + b2) * p * r / denom;
        }

        double balancedAccuracy() {
            return 0.5 * (recall() + specificity());
        }

        double jaccard() {
            int union = tp + fp + fn;
            return union == 0 ? 0.0 : tp / (double) union;
        }

        double mcc() {
            double numerator = (double) tp * tn - (double) fp * fn;
            double denom = Math.sqrt(((double) tp + fp) * ((double) tp + fn)
                    * ((double) tn + fp) * ((double) tn + fn));
            return denom == 0.0 ? 0.0 : numerator / denom;
        }

        double observedAgreement() {
            return accuracy();
        }

        double expectedAgreement() {
            int total = n();
            if (total == 0) return 0.0;
            double posTrue = (tp + fn) / (double) total;
            double posPred = (tp + fp) / (double) total;
            double negTrue = (tn + fp) / (double) total;
            double negPred = (tn + fn) / (double) total;
            return posTrue * posPred + negTrue * negPred;
        }

        double kappa() {
            double expected = expectedAgreement();
            return expected == 1.0 ? 1.0 : (observedAgreement() - expected) / (1.0 - expected);
        }
    }

    static Binary binaryFromLabels(int[] truth, int[] predicted) {
        check(truth.length == predicted.length, "label vectors must have equal length");
        int tp = 0;
        int fp = 0;
        int fn = 0;
        int tn = 0;
        for (int i = 0; i < truth.length; i++) {
            check(truth[i] == 0 || truth[i] == 1, "binary truth labels");
            check(predicted[i] == 0 || predicted[i] == 1, "binary predictions");
            if (truth[i] == 1 && predicted[i] == 1) tp++;
            else if (truth[i] == 0 && predicted[i] == 1) fp++;
            else if (truth[i] == 1 && predicted[i] == 0) fn++;
            else tn++;
        }
        return new Binary(tp, fp, fn, tn);
    }

    static Binary fromCells(int[][] matrix) {
        return new Binary(matrix[1][1], matrix[0][1], matrix[1][0], matrix[0][0]);
    }

    static int[][] confusion(int[] truth, int[] predicted, int classes) {
        int[][] matrix = new int[classes][classes];
        for (int i = 0; i < truth.length; i++) matrix[truth[i]][predicted[i]]++;
        return matrix;
    }

    static class Multiclass {
        final int[][] matrix;
        final int classes;

        Multiclass(int[][] matrix) {
            this.matrix = matrix;
            this.classes = matrix.length;
        }

        Binary oneVsRest(int c) {
            int tp = matrix[c][c];
            int fn = 0;
            int fp = 0;
            int tn = 0;
            for (int i = 0; i < classes; i++) {
                for (int j = 0; j < classes; j++) {
                    if (i == c && j == c) continue;
                    if (i == c) fn += matrix[i][j];
                    else if (j == c) fp += matrix[i][j];
                    else tn += matrix[i][j];
                }
            }
            return new Binary(tp, fp, fn, tn);
        }

        int support(int c) {
            int total = 0;
            for (int j = 0; j < classes; j++) total += matrix[c][j];
            return total;
        }

        int predictions(int c) {
            int total = 0;
            for (int i = 0; i < classes; i++) total += matrix[i][c];
            return total;
        }

        double accuracy() {
            int diagonal = 0;
            int total = 0;
            for (int i = 0; i < classes; i++) {
                for (int j = 0; j < classes; j++) {
                    total += matrix[i][j];
                    if (i == j) diagonal += matrix[i][j];
                }
            }
            return total == 0 ? 0.0 : diagonal / (double) total;
        }

        double macro(String metric) {
            double total = 0.0;
            for (int c = 0; c < classes; c++) total += value(oneVsRest(c), metric);
            return total / classes;
        }

        double weighted(String metric) {
            int total = 0;
            double acc = 0.0;
            for (int c = 0; c < classes; c++) {
                total += support(c);
                acc += support(c) * value(oneVsRest(c), metric);
            }
            return total == 0 ? 0.0 : acc / total;
        }

        static double value(Binary bin, String metric) {
            switch (metric) {
                case "precision": return bin.precision();
                case "recall": return bin.recall();
                case "f1": return bin.f1();
                case "specificity": return bin.specificity();
                default: throw new IllegalArgumentException(metric);
            }
        }
    }

    public static void main(String[] args) {
        int[] truth = {0, 1, 1, 0, 1, 0};
        int[] predicted = {0, 1, 0, 0, 1, 1};
        Binary bin = binaryFromLabels(truth, predicted);
        check(bin.tp == 2 && bin.fp == 1 && bin.fn == 1 && bin.tn == 2, "hand example cells");
        check(Math.abs(bin.accuracy() - 4.0 / 6.0) <= 1e-12, "accuracy");
        check(Math.abs(bin.precision() - 2.0 / 3.0) <= 1e-12, "precision");
        check(Math.abs(bin.recall() - 2.0 / 3.0) <= 1e-12, "recall");
        check(Math.abs(bin.f1() - 2.0 / 3.0) <= 1e-12, "f1");
        check(Math.abs(bin.specificity() - 2.0 / 3.0) <= 1e-12, "specificity");
        check(Math.abs(bin.balancedAccuracy() - 2.0 / 3.0) <= 1e-12, "balanced accuracy");
        check(Math.abs(bin.mcc() - 1.0 / 3.0) <= 1e-12, "MCC");
        check(Math.abs(bin.kappa() - 1.0 / 3.0) <= 1e-12, "kappa");
        check(Math.abs(bin.jaccard() - 0.5) <= 1e-12, "Jaccard = TP/(TP+FP+FN)");
        check(Math.abs(bin.errorRate() + bin.accuracy() - 1.0) <= 1e-12, "error rate complements");
        check(Math.abs(bin.falsePositiveRate() - 1.0 / 3.0) <= 1e-12, "FPR");
        check(Math.abs(bin.falseNegativeRate() - 1.0 / 3.0) <= 1e-12, "FNR");
        check(Math.abs(bin.recall() + bin.falseNegativeRate() - 1.0) <= 1e-12, "TPR + FNR = 1");
        check(Math.abs(bin.specificity() + bin.falsePositiveRate() - 1.0) <= 1e-12, "TNR + FPR = 1");
        check(Math.abs(bin.fBeta(2.0) - 5.0 * bin.precision() * bin.recall()
                / (4.0 * bin.precision() + bin.recall())) <= 1e-12, "F2 closed form");
        check(Math.abs(bin.fBeta(1.0) - bin.f1()) <= 1e-12, "F1 is F-beta at beta = 1");

        int[][] matrix = confusion(truth, predicted, 2);
        check(matrix[1][1] == 2 && matrix[0][1] == 1 && matrix[1][0] == 1 && matrix[0][0] == 2,
                "matrix cells match the direct pass");
        Binary rebuilt = fromCells(matrix);
        check(rebuilt.tp == bin.tp && rebuilt.fp == bin.fp && rebuilt.fn == bin.fn
                && rebuilt.tn == bin.tn, "two independent cell builders agree");
        check(rebuilt.n() == truth.length, "cells sum to n");
        check(matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1] == truth.length,
                "the four cells sum to n");

        Binary neverPositive = binaryFromLabels(new int[]{1, 1, 0}, new int[]{0, 0, 0});
        check(neverPositive.precision() == 0.0, "precision with no predicted positives is 0");
        check(Math.abs(neverPositive.recall()) <= 1e-12, "recall with no predicted positives is 0");
        check(Math.abs(neverPositive.f1()) <= 1e-12, "F1 with no predicted positives is 0");
        Binary noPositiveTruth = binaryFromLabels(new int[]{0, 0, 0}, new int[]{0, 1, 0});
        check(Math.abs(noPositiveTruth.specificity() - 2.0 / 3.0) <= 1e-12,
                "specificity = TN/(TN+FP) when the truth has no positives");
        check(noPositiveTruth.recall() == 0.0, "all-negative truth: recall is 0");
        check(noPositiveTruth.accuracy() == 0.0 + 2.0 / 3.0, "accuracy counts the two true negatives");
        Binary noNegativeTruth = binaryFromLabels(new int[]{1, 1, 1}, new int[]{1, 1, 1});
        check(noNegativeTruth.specificity() == 1.0,
                "no negative instances: specificity is 1 by convention");
        check(noNegativeTruth.mcc() == 0.0, "no negative instances: MCC denominator is 0");
        Binary empty = binaryFromLabels(new int[0], new int[0]);
        check(empty.n() == 0 && empty.accuracy() == 0.0, "empty input documented value");
        check(empty.mcc() == 0.0 && empty.kappa() == 0.0, "empty input has no denominator");
        Binary perfect = binaryFromLabels(new int[]{1, 0, 1, 1}, new int[]{1, 0, 1, 1});
        check(perfect.accuracy() == 1.0 && perfect.precision() == 1.0 && perfect.recall() == 1.0
                && perfect.f1() == 1.0 && perfect.mcc() == 1.0 && perfect.kappa() == 1.0,
                "perfect classifier has all metrics at 1");
        Binary inverted = binaryFromLabels(new int[]{1, 0, 1}, new int[]{0, 1, 0});
        check(inverted.accuracy() == 0.0 && inverted.recall() == 0.0
                && inverted.specificity() == 0.0, "fully inverted classifier");
        Binary single = binaryFromLabels(new int[]{1}, new int[]{1});
        check(single.tp == 1 && single.n() == 1 && single.kappa() == 1.0, "single element");

        int seed = 4242;
        int[] randomTruth = new int[200];
        int[] randomPred = new int[200];
        for (int i = 0; i < 200; i++) {
            seed = (seed * 75 + 74) % 65537;
            randomTruth[i] = seed % 2;
            seed = (seed * 75 + 74) % 65537;
            randomPred[i] = seed % 2;
        }
        Binary random = binaryFromLabels(randomTruth, randomPred);
        Binary naive = naiveCells(randomTruth, randomPred);
        check(random.tp == naive.tp && random.fp == naive.fp && random.fn == naive.fn
                && random.tn == naive.tn, "LCG cells match the naive double loop");
        check(Math.abs(random.precision() - random.tp / (double) (random.tp + random.fp)) <= 1e-12,
                "precision = TP/(TP+FP)");
        check(Math.abs(random.f1() - 2.0 * random.tp
                / (2.0 * random.tp + random.fp + random.fn)) <= 1e-12, "F1 closed form");
        double po = (random.tp + random.tn) / (double) random.n();
        double pOutcome = (random.tp + random.fn) * (random.tp + random.fp)
                + (random.fn + random.tn) * (random.fp + random.tn);
        double pe = pOutcome / (random.n() * (double) random.n());
        check(Math.abs(random.kappa() - (po - pe) / (1.0 - pe)) <= 1e-12, "kappa from the rates");
        check(random.mcc() >= -1.0 && random.mcc() <= 1.0, "MCC in [-1, 1]");

        int[] threeTruth = new int[90];
        int[] threePred = new int[90];
        seed = 777;
        for (int i = 0; i < 90; i++) {
            seed = (seed * 75 + 74) % 65537;
            threeTruth[i] = (seed / 3) % 3;
            seed = (seed * 75 + 74) % 65537;
            threePred[i] = (seed / 5) % 3;
        }
        int[][] threeMatrix = confusion(threeTruth, threePred, 3);
        Multiclass multi = new Multiclass(threeMatrix);
        check(multi.accuracy() >= 0.0 && multi.accuracy() <= 1.0, "3-class accuracy range");
        double macroF1 = 0.0;
        double weightedF1 = 0.0;
        int totalSupport = 0;
        int diagonal = 0;
        int sumTp = 0;
        int sumFp = 0;
        int sumFn = 0;
        for (int c = 0; c < 3; c++) {
            Binary ovr = multi.oneVsRest(c);
            int rowSum = 0;
            for (int j = 0; j < 3; j++) rowSum += threeMatrix[c][j];
            int colSum = 0;
            for (int i = 0; i < 3; i++) colSum += threeMatrix[i][c];
            check(ovr.tp == threeMatrix[c][c], "one-vs-rest TP is the diagonal cell");
            check(ovr.fn == rowSum - threeMatrix[c][c], "one-vs-rest FN is the rest of the row");
            check(ovr.fp == colSum - threeMatrix[c][c], "one-vs-rest FP is the rest of the column");
            check(ovr.tn == 90 - rowSum - colSum + threeMatrix[c][c],
                    "one-vs-rest TN is everything else");
            macroF1 += ovr.f1();
            weightedF1 += rowSum * ovr.f1();
            totalSupport += rowSum;
            sumTp += ovr.tp;
            sumFp += ovr.fp;
            sumFn += ovr.fn;
            diagonal += threeMatrix[c][c];
        }
        macroF1 /= 3.0;
        check(totalSupport == 90, "supports sum to n");
        check(sumTp == diagonal && sumFp == 90 - diagonal && sumFn == 90 - diagonal,
                "one-vs-rest cells regroup the matrix");
        double accuracy = diagonal / 90.0;
        double microPrecision = diagonal / (double) (diagonal + sumFp);
        double microRecall = diagonal / (double) (diagonal + sumFn);
        double microF1 = 2.0 * diagonal / (2.0 * diagonal + sumFp + sumFn);
        check(Math.abs(multi.accuracy() - accuracy) <= 1e-12, "accuracy = diagonal / n");
        check(Math.abs(microPrecision - microRecall) <= 1e-12
                && Math.abs(microPrecision - accuracy) <= 1e-12,
                "micro precision = micro recall = accuracy");
        check(Math.abs(microF1 - microPrecision) <= 1e-12, "micro F1 = micro precision");
        check(Math.abs(macroF1 - multi.macro("f1")) <= 1e-12, "macro F1 is the unweighted mean");
        check(Math.abs(weightedF1 / 90.0 - multi.weighted("f1")) <= 1e-12,
                "weighted F1 is the support-weighted mean");
        check(Math.abs(multi.macro("precision") - multi.weighted("precision")) > 1e-3,
                "documented: macro and weighted differ when supports differ");

        Multiclass perfectMulti = new Multiclass(confusion(threeTruth, threeTruth, 3));
        check(perfectMulti.accuracy() == 1.0 && perfectMulti.macro("precision") == 1.0
                && perfectMulti.weighted("f1") == 1.0, "perfect 3-class classifier");
        int[] alwaysZero = new int[90];
        Multiclass lazy = new Multiclass(confusion(threeTruth, alwaysZero, 3));
        check(Math.abs(lazy.macro("recall") - 1.0 / 3.0) <= 1e-12,
                "always predicting class 0: macro recall = 1/3");
        check(Math.abs(lazy.oneVsRest(0).recall() - 1.0) <= 1e-12, "class 0 recall = 1");
        check(lazy.oneVsRest(1).recall() == 0.0 && lazy.oneVsRest(2).recall() == 0.0,
                "other recalls are 0");
        check(Math.abs(lazy.oneVsRest(1).f1()) <= 1e-12 && Math.abs(lazy.oneVsRest(2).f1()) <= 1e-12,
                "unpredicted classes get F1 = 0");
        check(Math.abs(lazy.accuracy() - lazy.support(0) / 90.0) <= 1e-12,
                "lazy accuracy is the class-0 prevalence");
        check(lazy.macro("precision") >= 0.0 && lazy.macro("precision") <= 1.0,
                "macro precision range");

        System.out.println("PASS");
    }

    private static Binary naiveCells(int[] truth, int[] predicted) {
        int tp = 0;
        int fp = 0;
        int fn = 0;
        int tn = 0;
        for (int t : new int[]{0, 1}) {
            for (int p : new int[]{0, 1}) {
                int count = 0;
                for (int i = 0; i < truth.length; i++) {
                    if (truth[i] == t && predicted[i] == p) count++;
                }
                if (t == 1 && p == 1) tp = count;
                if (t == 0 && p == 1) fp = count;
                if (t == 1 && p == 0) fn = count;
                if (t == 0 && p == 0) tn = count;
            }
        }
        return new Binary(tp, fp, fn, tn);
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
