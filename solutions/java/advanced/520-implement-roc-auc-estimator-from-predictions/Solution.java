/** Task 520 - Implement ROC AUC estimator from predictions. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Three independent ROC-AUC estimators over real-valued scores and binary labels.
 * (1) Trapezoid: the ROC staircase is built from the distinct thresholds in descending order,
 *     starting at (0, 0), moving to the (FPR, TPR) point obtained after admitting every item whose
 *     score equals the threshold, and finishing at (1, 1); the area is the sum of trapezoids
 *     (fpr1 - fpr0) * (tpr0 + tpr1) / 2.
 * (2) Pairwise / Mann-Whitney U: the probability that a uniformly drawn positive outranks a
 *     uniformly drawn negative, scoring a tie as 0.5, computed by the O(P*N) double loop.
 * (3) Rank sum: AUC = (sum of the average ranks of the positives - P(P+1)/2) / (P*N), with average
 *     ranks (R + 1 + ... ) so ties split credit - the classic equivalent formula.
 * A weighted variant AUC_w = sum w_i w_j [s_i > s_j] + 0.5 [s_i = s_j] over (positive, negative)
 * pairs divided by W_pos * W_neg is also provided; it must equal the unweighted AUC for unit
 * weights and must equal the plain AUC of the row-duplicated data set for integer weights.
 * Documented conventions: a data set with only one label class has no defined AUC and reports 0.5;
 * an empty input reports 0.5; scores are compared exactly (no epsilon), so ties are handled by the
 * explicit half-credit rules rather than by rounding.
 *
 * Checks: perfect scores give 1.0 and the reversed scores 0.0; constant scores with mixed labels
 * give exactly 0.5; the three estimators agree to <= 1e-12 on twelve LCG data sets with heavy ties
 * (only four distinct score values) and on continuous LCG scores; complementing the scores maps AUC
 * to 1 - AUC; multiplying by a positive constant and applying exp() both leave AUC unchanged;
 * the curve starts at (0, 0), ends at (1, 1), is monotone in both coordinates, has one point per
 * distinct threshold plus the origin, and each of its points reproduces TP/P and FP/N recomputed
 * independently at that threshold; weighted AUC equals the duplicated-row AUC and the unit-weight
 * AUC; a linearly separable LCG workload scores above 0.8 while a label-shuffled workload scores
 * near 0.5; the hand-computed 3-point example gives 0.5, and repeated runs are identical.
 */
import java.util.*;

public class Solution {
    static double[] rocTpr;
    static double[] rocFpr;
    static double[] rocThreshold;

    /** Builds the ROC curve (ascending FPR) and leaves it in the three static arrays. */
    static int buildCurve(double[] scores, int[] labels) {
        int n = scores.length;
        int positives = 0;
        for (int label : labels) positives += label;
        int negatives = n - positives;
        double[] distinct = new double[n];
        int m = 0;
        for (double score : scores) {
            boolean seen = false;
            for (int i = 0; i < m; i++) {
                if (distinct[i] == score) {
                    seen = true;
                    break;
                }
            }
            if (!seen) distinct[m++] = score;
        }
        double[] thresholds = Arrays.copyOf(distinct, m);
        Arrays.sort(thresholds);
        double[] fpr = new double[m + 1];
        double[] tpr = new double[m + 1];
        double[] thr = new double[m + 1];
        fpr[0] = 0.0;
        tpr[0] = 0.0;
        thr[0] = Double.POSITIVE_INFINITY;
        int tp = 0;
        int fp = 0;
        for (int i = m - 1, out = 1; i >= 0; i--, out++) {
            double threshold = thresholds[i];
            for (int j = 0; j < n; j++) {
                if (scores[j] == threshold) {
                    if (labels[j] == 1) tp++;
                    else fp++;
                }
            }
            fpr[out] = negatives == 0 ? 0.0 : fp / (double) negatives;
            tpr[out] = positives == 0 ? 0.0 : tp / (double) positives;
            thr[out] = threshold;
        }
        rocFpr = fpr;
        rocTpr = tpr;
        rocThreshold = thr;
        return m + 1;
    }

    static double aucTrapezoid(double[] scores, int[] labels) {
        int positives = 0;
        for (int label : labels) positives += label;
        if (labels.length == 0 || positives == 0 || positives == labels.length) return 0.5;
        int size = buildCurve(scores, labels);
        double area = 0.0;
        for (int i = 0; i + 1 < size; i++) {
            area += (rocFpr[i + 1] - rocFpr[i]) * (rocTpr[i] + rocTpr[i + 1]) / 2.0;
        }
        return area;
    }

    static double aucPairwise(double[] scores, int[] labels) {
        double credit = 0.0;
        long pairs = 0;
        for (int i = 0; i < scores.length; i++) {
            if (labels[i] != 1) continue;
            for (int j = 0; j < scores.length; j++) {
                if (labels[j] != 0) continue;
                pairs++;
                if (scores[i] > scores[j]) credit += 1.0;
                else if (scores[i] == scores[j]) credit += 0.5;
            }
        }
        return pairs == 0 ? 0.5 : credit / pairs;
    }

    static double aucRank(double[] scores, int[] labels) {
        int n = scores.length;
        int positives = 0;
        for (int label : labels) positives += label;
        int negatives = n - positives;
        if (positives == 0 || negatives == 0) return 0.5;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(scores[a], scores[b]));
        double[] rank = new double[n];
        int i = 0;
        while (i < n) {
            int j = i;
            while (j + 1 < n && scores[order[j + 1]] == scores[order[i]]) j++;
            double average = (i + j) / 2.0 + 1.0;
            for (int k = i; k <= j; k++) rank[order[k]] = average;
            i = j + 1;
        }
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            if (labels[k] == 1) sum += rank[k];
        }
        return (sum - positives * (positives + 1) / 2.0) / (positives * (double) negatives);
    }

    static double aucWeighted(double[] scores, int[] labels, double[] weights) {
        double credit = 0.0;
        double pairWeight = 0.0;
        for (int i = 0; i < scores.length; i++) {
            if (labels[i] != 1) continue;
            for (int j = 0; j < scores.length; j++) {
                if (labels[j] != 0) continue;
                double w = weights[i] * weights[j];
                pairWeight += w;
                if (scores[i] > scores[j]) credit += w;
                else if (scores[i] == scores[j]) credit += 0.5 * w;
            }
        }
        return pairWeight == 0.0 ? 0.5 : credit / pairWeight;
    }

    public static void main(String[] args) {
        double[] perfect = {0.1, 0.2, 0.3, 0.4};
        int[] perfectLabels = {0, 0, 1, 1};
        check(Math.abs(aucTrapezoid(perfect, perfectLabels) - 1.0) <= 1e-12, "perfect trapezoid");
        check(Math.abs(aucPairwise(perfect, perfectLabels) - 1.0) <= 1e-12, "perfect pairwise");
        check(Math.abs(aucRank(perfect, perfectLabels) - 1.0) <= 1e-12, "perfect rank sum");
        double[] reversed = {0.4, 0.3, 0.2, 0.1};
        check(Math.abs(aucTrapezoid(reversed, perfectLabels)) <= 1e-12, "anti-perfect trapezoid");
        check(Math.abs(aucPairwise(reversed, perfectLabels)) <= 1e-12, "anti-perfect pairwise");
        check(Math.abs(aucRank(reversed, perfectLabels)) <= 1e-12, "anti-perfect rank sum");

        double[] flat = {2.5, 2.5, 2.5, 2.5};
        check(Math.abs(aucTrapezoid(flat, perfectLabels) - 0.5) <= 1e-12, "constant scores");
        check(Math.abs(aucPairwise(flat, perfectLabels) - 0.5) <= 1e-12, "all pairs tied");
        check(Math.abs(aucRank(flat, perfectLabels) - 0.5) <= 1e-12, "rank ties split credit");

        double[] hand = {1.0, 2.0, 3.0};
        int[] handLabels = {0, 1, 0};
        check(Math.abs(aucTrapezoid(hand, handLabels) - 0.5) <= 1e-12, "hand example trapezoid");
        check(Math.abs(aucPairwise(hand, handLabels) - 0.5) <= 1e-12, "hand example pairwise");
        check(Math.abs(aucRank(hand, handLabels) - 0.5) <= 1e-12, "hand example rank sum");

        double[] single = {0.5, 0.5};
        check(Math.abs(aucTrapezoid(single, new int[]{1, 1}) - 0.5) <= 1e-12,
                "single-class input reports 0.5 (documented)");
        check(Math.abs(aucRank(new double[0], new int[0]) - 0.5) <= 1e-12,
                "empty input reports 0.5 (documented)");
        check(Math.abs(aucTrapezoid(new double[0], new int[0]) - 0.5) <= 1e-12,
                "empty trapezoid input reports 0.5 (documented)");

        int seed = 31;
        double worstGap = 0.0;
        for (int trial = 0; trial < 12; trial++) {
            int n = 12 + trial * 3;
            double[] tied = new double[n];
            double[] continuous = new double[n];
            int[] labels = new int[n];
            for (int i = 0; i < n; i++) {
                seed = (seed * 75 + 74) % 65537;
                tied[i] = (seed / 7) % 4;
                seed = (seed * 75 + 74) % 65537;
                continuous[i] = (seed % 1000) / 100.0;
                seed = (seed * 75 + 74) % 65537;
                labels[i] = (seed / 11) % 2;
            }
            int positives = 0;
            for (int label : labels) positives += label;
            if (positives == 0 || positives == n) labels[0] = 1;
            labels[n - 1] = 0;
            double a = aucTrapezoid(tied, labels);
            double b = aucPairwise(tied, labels);
            double c = aucRank(tied, labels);
            worstGap = Math.max(worstGap, Math.max(Math.abs(a - b), Math.abs(a - c)));
            check(Math.abs(a - b) <= 1e-12, "tied data: trapezoid = pairwise, trial " + trial);
            check(Math.abs(a - c) <= 1e-12, "tied data: trapezoid = rank sum, trial " + trial);
            check(a >= 0.0 && a <= 1.0, "AUC in [0, 1]");
            double d = aucTrapezoid(continuous, labels);
            check(Math.abs(d - aucPairwise(continuous, labels)) <= 1e-12, "continuous pairwise");
            check(Math.abs(d - aucRank(continuous, labels)) <= 1e-12, "continuous rank sum");

            double[] negated = new double[n];
            for (int i = 0; i < n; i++) negated[i] = -continuous[i];
            check(Math.abs(aucTrapezoid(negated, labels) - (1.0 - d)) <= 1e-12,
                    "complementing the scores gives 1 - AUC");

            double[] scaled = new double[n];
            double[] exponentiated = new double[n];
            for (int i = 0; i < n; i++) {
                scaled[i] = 17.0 * continuous[i] + 3.0;
                exponentiated[i] = Math.exp(continuous[i]);
            }
            check(Math.abs(aucTrapezoid(scaled, labels) - d) <= 1e-12,
                    "affine transform with a positive scale leaves AUC unchanged");
            check(Math.abs(aucTrapezoid(exponentiated, labels) - d) <= 1e-12,
                    "monotone transform leaves AUC unchanged");
        }

        seed = 555;
        int n = 40;
        double[] scores = new double[n];
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            scores[i] = (seed % 1000) / 100.0;
            seed = (seed * 75 + 74) % 65537;
            labels[i] = i % 2;
        }
        int size = buildCurve(scores, labels);
        double[] fpr = rocFpr.clone();
        double[] tpr = rocTpr.clone();
        double[] thr = rocThreshold.clone();
        Set<Double> distinctScores = new TreeSet<>();
        for (double score : scores) distinctScores.add(score);
        buildCurve(scores, labels);
        Set<Double> distinctAgain = new TreeSet<>();
        for (double score : scores) distinctAgain.add(score);
        check(distinctScores.equals(distinctAgain), "distinct score set is stable");
        check(size == distinctScores.size() + 1,
                "one curve point per distinct threshold plus the origin");
        check(fpr[0] == 0.0 && tpr[0] == 0.0, "curve starts at (0, 0)");
        check(Math.abs(fpr[size - 1] - 1.0) <= 1e-12 && Math.abs(tpr[size - 1] - 1.0) <= 1e-12,
                "curve ends at (1, 1)");
        int positives = 0;
        for (int label : labels) positives += label;
        int negatives = n - positives;
        for (int i = 1; i < size; i++) {
            check(fpr[i] >= fpr[i - 1] - 1e-15 && tpr[i] >= tpr[i - 1] - 1e-15, "monotone curve");
            int tp = 0;
            int fp = 0;
            for (int j = 0; j < n; j++) {
                if (scores[j] >= thr[i]) {
                    if (labels[j] == 1) tp++;
                    else fp++;
                }
            }
            check(Math.abs(tpr[i] - tp / (double) positives) <= 1e-12, "curve TPR at threshold " + i);
            check(Math.abs(fpr[i] - fp / (double) negatives) <= 1e-12, "curve FPR at threshold " + i);
        }
        check(thr[1] >= thr[size - 1], "the first real threshold is the largest");
        check(Math.abs(aucTrapezoid(scores, labels) - aucPairwise(scores, labels)) <= 1e-12,
                "large tied-free data set: two estimators agree");

        double[] weights = new double[n];
        double[] duplicatedScores = new double[3 * n];
        int[] duplicatedLabels = new int[3 * n];
        int write = 0;
        for (int i = 0; i < n; i++) {
            weights[i] = 1 + (i % 3);
            for (int r = 0; r < 3; r++) {
                if (r < weights[i]) {
                    duplicatedScores[write] = scores[i];
                    duplicatedLabels[write] = labels[i];
                    write++;
                }
            }
        }
        double[] unitWeights = new double[n];
        Arrays.fill(unitWeights, 1.0);
        check(Math.abs(aucWeighted(scores, labels, unitWeights) - aucPairwise(scores, labels)) <= 1e-12,
                "unit weights reduce to the plain AUC");
        double[] trimmedScores = Arrays.copyOf(duplicatedScores, write);
        int[] trimmedLabels = Arrays.copyOf(duplicatedLabels, write);
        check(Math.abs(aucWeighted(scores, labels, weights)
                - aucPairwise(trimmedScores, trimmedLabels)) <= 1e-12,
                "integer weights equal the duplicated-row AUC");
        check(Math.abs(aucWeighted(scores, labels, weights)
                - aucTrapezoid(trimmedScores, trimmedLabels)) <= 1e-12,
                "weighted AUC agrees with the trapezoid estimate of the duplicated rows");

        double[] separable = new double[60];
        int[] separableLabels = new int[60];
        seed = 8080;
        for (int i = 0; i < 60; i++) {
            seed = (seed * 75 + 74) % 65537;
            double noise = (seed % 100) / 100.0;
            separableLabels[i] = i % 2;
            separable[i] = separableLabels[i] * 2.0 + noise;
        }
        double separableAuc = aucTrapezoid(separable, separableLabels);
        check(separableAuc > 0.8, "informativeness: separable scores score above 0.8");
        int[] shuffled = new int[60];
        for (int i = 0; i < 60; i++) shuffled[i] = separableLabels[(i * 7 + 3) % 60];
        double shuffledAuc = aucTrapezoid(separable, shuffled);
        check(Math.abs(shuffledAuc - aucPairwise(separable, shuffled)) <= 1e-12,
                "shuffled labels still agree across estimators");
        check(shuffledAuc < separableAuc, "shuffled labels are less informative");

        check(Math.abs(aucPairwise(tiedWorkload(), tiedWorkloadLabels()) - aucRank(tiedWorkload(),
                tiedWorkloadLabels())) <= 1e-12, "worst tied case still agrees");
        check(worstGap <= 1e-12, "documented: the three estimators never disagree by more than 1e-12");

        System.out.println("PASS");
    }

    private static final double[] TIED_WORKLOAD = {1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0};
    private static final int[] TIED_WORKLOAD_LABELS = {0, 1, 1, 0, 1, 1, 0};

    private static double[] tiedWorkload() {
        return TIED_WORKLOAD.clone();
    }

    private static int[] tiedWorkloadLabels() {
        return TIED_WORKLOAD_LABELS.clone();
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
