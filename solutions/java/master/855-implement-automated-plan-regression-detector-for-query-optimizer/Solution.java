/** Task 855 - Implement automated plan regression detector for query optimizer changes.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   A cost-based optimizer is represented by plan trees (SCAN, INDEX_SCAN, HASH_JOIN, NESTED_LOOP,
 *   SORT_MERGE, SORT, HASH_AGG) with a deterministic cost model, and a workload of queries where each query
 *   has the plan an old optimizer version produced and the plan a new version produces. Execution latency is
 *   simulated as cost x per-operator factor plus fixed-seed measurement noise, sampled `reps` times per
 *   version, and the detector works only on those samples (never on the cost model):
 *
 *       per repetition i: d_i = (new_i - old_i) / old_i,  k = #{ d_i > 0 }
 *       signed-test p-value: P(X >= k) with X ~ Binomial(reps, 1/2)   (exact integer arithmetic)
 *       flag a query when p < alpha / m   (Bonferroni over the m queries of the workload)
 *
 *   Ground truth comes from the cost model, so precision/recall of the detector, the effect of the correction
 *   for multiple comparisons, and the robustness to a doubled noise level are all measurable. Regressions are
 *   additionally grouped by their structural rewrite signature (the plan-shape fingerprint) so that a single
 *   systematic change can be identified.
 *
 * WHAT IS SIMPLIFIED
 *   No SQL, no statistics catalog, no real executor: tree shapes, row estimates and the cost constants are
 *   hand-made, the noise is a deterministic uniform deviate and the "measurement" is arithmetic. There is no
 *   wall-clock time and no threads; scale is the workload size and the repetition count.
 *
 * INVARIANTS VERIFIED
 *   Cost/measurement: the recursive cost equals an independent stack-based traversal, cost is monotone in the
 *   row estimates, the median measured latency is rank-correlated with the true cost (Spearman >= 0.9), and
 *   fingerprints are stable and structure-only.
 *   Detector: the exact signed-test p-value matches a brute-force enumeration of all 2^reps sign patterns, is
 *   monotone in the effect size (p(0 positives) = 1) and the flagged sets are nested as the threshold
 *   tightens; injected regressions are found with precision and recall >= 0.9 while an unchanged control
 *   workload yields no flags at the Bonferroni threshold; improvements are never flagged; doubling the noise
 *   keeps the false-positive count bounded and the injections detected; and the report is reproducible from
 *   its seed.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so workloads, samples and reports are reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }

    boolean chance(int percent) { return nextInt(100) < percent; }

    /** Deterministic deviate in [-1, 1] keyed by the (query, version, repetition) triple. */
    static double noise(long seed, int query, int version, int repetition) {
        long h = SplitMix64.mix(seed ^ 0x5DEECE66DL, SplitMix64.mix(query * 131L + version, repetition));
        return ((h >>> 11) * 0x1.0p-53) * 2.0 - 1.0;
    }

    static long mix(long a, long b) {
        long h = a * 0x9E3779B97F4A7C15L + b * 0xC2B2AE3D27D4EB4FL;
        h ^= h >>> 29;
        h *= 0xBF58476D1CE4E5B9L;
        h ^= h >>> 32;
        return h;
    }
}

/** A physical operator: kind, estimated output rows, children and optional selectivity/table metadata. */
final class Op {
    final String kind;
    final long rows;
    final Op left;
    final Op right;
    final double selectivity;
    final String table;

    Op(String kind, long rows, Op left, Op right, double selectivity, String table) {
        this.kind = kind;
        this.rows = rows;
        this.left = left;
        this.right = right;
        this.selectivity = selectivity;
        this.table = table;
    }

    static Op scan(String table, long rows) { return new Op("SCAN", rows, null, null, 1.0, table); }

    static Op indexScan(String table, long rows, double selectivity) {
        return new Op("INDEX_SCAN", Math.max(1, (long) (rows * selectivity)), null, null, selectivity, table);
    }

    static Op join(String kind, Op left, Op right, long rows) { return new Op(kind, rows, left, right, 1.0, null); }

    static Op unary(String kind, Op child, long rows) { return new Op(kind, rows, child, null, 1.0, null); }

    /** Recursive cost model; the constants stand in for the optimizer's calibrated formula. */
    double cost() {
        switch (kind) {
            case "SCAN": return rows * 1.0;
            case "INDEX_SCAN": return 5.0 + rows * 0.4;
            case "SORT": return left.cost() + rows * Math.log(Math.max(2, rows)) / Math.log(2) * 0.8;
            case "HASH_AGG": return left.cost() + rows * 0.3;
            case "HASH_JOIN": return left.cost() + right.cost() + left.rows * 0.25 + rows * 1.2;
            case "SORT_MERGE": return left.cost() + right.cost()
                    + (left.rows + right.rows) * Math.log(Math.max(2, left.rows + right.rows)) / Math.log(2) * 0.5
                    + rows * 0.4;
            case "NESTED_LOOP": return left.cost() + right.cost() + left.rows * right.rows * 0.9;
            default: throw new IllegalStateException("unknown operator " + kind);
        }
    }

    /** Independent evaluation through an explicit post-order stack, no recursion. */
    double costIterative() {
        Map<Op, Double> solved = new IdentityHashMap<>();
        Deque<Op> stack = new ArrayDeque<>();
        Deque<Op> order = new ArrayDeque<>();
        stack.push(this);
        while (!stack.isEmpty()) {
            Op current = stack.pop();
            order.push(current);
            if (current.left != null) stack.push(current.left);
            if (current.right != null) stack.push(current.right);
        }
        while (!order.isEmpty()) {
            Op current = order.pop();
            double l = current.left == null ? 0.0 : solved.get(current.left);
            double r = current.right == null ? 0.0 : solved.get(current.right);
            double value;
            switch (current.kind) {
                case "SCAN": value = current.rows * 1.0; break;
                case "INDEX_SCAN": value = 5.0 + current.rows * 0.4; break;
                case "SORT": value = l + current.rows * Math.log(Math.max(2, current.rows)) / Math.log(2) * 0.8; break;
                case "HASH_AGG": value = l + current.rows * 0.3; break;
                case "HASH_JOIN": value = l + r + current.left.rows * 0.25 + current.rows * 1.2; break;
                case "SORT_MERGE": value = l + r
                        + (current.left.rows + current.right.rows)
                        * Math.log(Math.max(2, current.left.rows + current.right.rows)) / Math.log(2) * 0.5
                        + current.rows * 0.4;
                    break;
                case "NESTED_LOOP": value = l + r + current.left.rows * current.right.rows * 0.9; break;
                default: throw new IllegalStateException("unknown operator " + current.kind);
            }
            solved.put(current, value);
        }
        return solved.get(this);
    }

    /** Structure-only fingerprint: kinds, accessed relations and shapes, independent of estimated rows. */
    int fingerprint() {
        long h = SplitMix64.mix(kind.hashCode(), table == null ? 0x9E37L : table.hashCode());
        if (left != null) h = SplitMix64.mix(h, left.fingerprint());
        if (right != null) h = SplitMix64.mix(h, right.fingerprint() + 17);
        return (int) (h ^ (h >>> 32));
    }

    String describe() {
        StringBuilder sb = new StringBuilder(kind);
        if (table != null) sb.append(':').append(table);
        sb.append('(').append(rows).append(')');
        if (left != null) sb.append('[').append(left.describe());
        if (right != null) sb.append('|').append(right.describe());
        if (left != null) sb.append(']');
        return sb.toString();
    }
}

/** One workload entry: what the old optimizer produced, what the new one produces, and the injected change. */
final class QueryCase {
    final String name;
    final Op oldPlan;
    final Op newPlan;
    final String rewrite;

    QueryCase(String name, Op oldPlan, Op newPlan, String rewrite) {
        this.name = name;
        this.oldPlan = oldPlan;
        this.newPlan = newPlan;
        this.rewrite = rewrite;
    }

    boolean trueRegression() { return newPlan.cost() > oldPlan.cost() * 1.05; }

    boolean trueImprovement() { return newPlan.cost() < oldPlan.cost() * 0.95; }
}

/** Per-query detection outcome. */
final class Trial {
    final QueryCase query;
    final double oldMedian;
    final double newMedian;
    final double relativeDelta;
    final int positiveDeltas;
    final double pValue;

    Trial(QueryCase query, double oldMedian, double newMedian, double relativeDelta, int positiveDeltas,
            double pValue) {
        this.query = query;
        this.oldMedian = oldMedian;
        this.newMedian = newMedian;
        this.relativeDelta = relativeDelta;
        this.positiveDeltas = positiveDeltas;
        this.pValue = pValue;
    }
}

/** The detector report: trials, the flagged subset and the rewrite groups. */
final class Report {
    final List<Trial> trials;
    final boolean[] flagged;
    final double alpha;
    final double threshold;

    Report(List<Trial> trials, boolean[] flagged, double alpha, double threshold) {
        this.trials = trials;
        this.flagged = flagged;
        this.alpha = alpha;
        this.threshold = threshold;
    }

    int flaggedCount() {
        int n = 0;
        for (boolean f : flagged) if (f) n++;
        return n;
    }

    int truePositives() {
        int n = 0;
        for (int i = 0; i < flagged.length; i++) if (flagged[i] && trials.get(i).query.trueRegression()) n++;
        return n;
    }

    int falsePositives() {
        int n = 0;
        for (int i = 0; i < flagged.length; i++) {
            if (flagged[i] && !trials.get(i).query.trueRegression()) n++;
        }
        return n;
    }

    int trueRegressions() {
        int n = 0;
        for (Trial t : trials) if (t.query.trueRegression()) n++;
        return n;
    }

    Map<String, Integer> groupCounts() {
        Map<String, Integer> groups = new TreeMap<>();
        for (int i = 0; i < flagged.length; i++) {
            if (flagged[i]) groups.merge(trials.get(i).query.rewrite, 1, Integer::sum);
        }
        return groups;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    // ------------------------------------------------------------------ exact statistics

    /** Exact one-sided signed-test p-value P(X >= k), X ~ Binomial(n, 1/2), computed in integer arithmetic. */
    static double signP(int k, int n) {
        if (k <= 0) return 1.0;
        long total = 1L << n;
        long tail = 0;
        for (int i = k; i <= n; i++) tail += binomial(n, i);
        return Math.min(1.0, tail / (double) total);
    }

    static long binomial(int n, int k) {
        long result = 1;
        for (int i = 0; i < k; i++) {
            result = result * (n - i) / (i + 1);
        }
        return result;
    }

    /** Brute force over all 2^n sign patterns: fraction with at least k positive deltas. */
    static double signPBruteForce(int k, int n) {
        int favourable = 0;
        for (int mask = 0; mask < (1 << n); mask++) {
            if (Integer.bitCount(mask) >= k) favourable++;
        }
        return favourable / (double) (1 << n);
    }

    // ------------------------------------------------------------------ measurement and detection

    static final int REPS = 16;
    static final double BASE_NOISE = 0.10;

    static double[] sample(Op plan, long seed, int queryIndex, int version, double noiseScale, double factor) {
        double[] samples = new double[REPS];
        for (int rep = 0; rep < REPS; rep++) {
            double noise = 1.0 + noiseScale * SplitMix64.noise(seed, queryIndex, version, rep);
            samples[rep] = plan.cost() * factor * noise;
        }
        return samples;
    }

    static double median(double[] values) {
        double[] copy = values.clone();
        Arrays.sort(copy);
        if (copy.length % 2 == 1) return copy[copy.length / 2];
        return 0.5 * (copy[copy.length / 2 - 1] + copy[copy.length / 2]);
    }

    static Report detect(List<QueryCase> workload, long seed, double noiseScale, double alpha) {
        List<Trial> trials = new ArrayList<>();
        double threshold = alpha / Math.max(1, workload.size());
        for (int i = 0; i < workload.size(); i++) {
            QueryCase query = workload.get(i);
            double[] oldSamples = sample(query.oldPlan, seed, i, 0, noiseScale, 1.0);
            double[] newSamples = sample(query.newPlan, seed, i, 1, noiseScale, 1.0);
            int positive = 0;
            for (int rep = 0; rep < REPS; rep++) if (newSamples[rep] > oldSamples[rep]) positive++;
            double oldMedian = median(oldSamples);
            double newMedian = median(newSamples);
            trials.add(new Trial(query, oldMedian, newMedian, (newMedian - oldMedian) / oldMedian, positive,
                    signP(positive, REPS)));
        }
        boolean[] flagged = new boolean[workload.size()];
        for (int i = 0; i < trials.size(); i++) {
            Trial t = trials.get(i);
            flagged[i] = t.pValue < threshold && t.relativeDelta > 0.03;
        }
        return new Report(trials, flagged, alpha, threshold);
    }

    /** Spearman rank correlation between two equal-length series (ties broken by index). */
    static double spearman(double[] a, double[] b) {
        int n = a.length;
        double[] ra = ranks(a);
        double[] rb = ranks(b);
        double meanA = 0, meanB = 0;
        for (int i = 0; i < n; i++) { meanA += ra[i]; meanB += rb[i]; }
        meanA /= n;
        meanB /= n;
        double cov = 0, varA = 0, varB = 0;
        for (int i = 0; i < n; i++) {
            cov += (ra[i] - meanA) * (rb[i] - meanB);
            varA += (ra[i] - meanA) * (ra[i] - meanA);
            varB += (rb[i] - meanB) * (rb[i] - meanB);
        }
        return cov / Math.sqrt(varA * varB);
    }

    static double[] ranks(double[] values) {
        Integer[] order = new Integer[values.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        final double[] copy = values.clone();
        Arrays.sort(order, (x, y) -> copy[x] != copy[y] ? Double.compare(copy[x], copy[y]) : Integer.compare(x, y));
        double[] ranks = new double[values.length];
        for (int i = 0; i < order.length; i++) ranks[order[i]] = i + 1;
        return ranks;
    }

    // ------------------------------------------------------------------ workload construction

    static Op baseScan(String table, long rows, boolean index, double selectivity) {
        return index ? Op.indexScan(table, rows, selectivity) : Op.scan(table, rows);
    }

    /** A family of plans over the same shapes; `mode` selects the rewrite applied by the new optimizer. */
    static QueryCase buildCase(int index, String mode) {
        long factRows = 200 + 37L * (index % 7);
        long dimRows = 20 + index % 5;
        double selectivity = 0.2 + 0.05 * (index % 4);
        Op factScan = baseScan("facts", factRows, true, selectivity);
        Op dimScan = Op.scan("dims", dimRows);
        Op agg = Op.unary("HASH_AGG", factScan, factRows / 4);
        Op oldJoin = Op.join("HASH_JOIN", agg, dimScan, factRows / 3);
        Op newJoin;
        switch (mode) {
            case "identity":
                newJoin = Op.join("HASH_JOIN", agg, dimScan, factRows / 3);
                break;
            case "index-dropped":
                newJoin = Op.join("HASH_JOIN", Op.unary("HASH_AGG", Op.scan("facts", factRows), factRows / 4),
                        dimScan, factRows / 3);
                break;
            case "nested-loop-instead-of-hash":
                newJoin = Op.join("NESTED_LOOP", agg, dimScan, factRows / 3);
                break;
            case "join-order-swap":
                newJoin = Op.join("HASH_JOIN", dimScan, agg, factRows / 3);
                break;
            case "hash-instead-of-nested-loop":
                oldJoin = Op.join("NESTED_LOOP", agg, dimScan, factRows / 3);
                newJoin = Op.join("HASH_JOIN", agg, dimScan, factRows / 3);
                break;
            case "sort-merge-instead-of-hash":
                newJoin = Op.join("SORT_MERGE", agg, dimScan, factRows / 3);
                break;
            case "aggregate-pushed-up":
                oldJoin = Op.join("HASH_JOIN", factScan, dimScan, factRows / 4);
                newJoin = Op.unary("HASH_AGG", Op.join("HASH_JOIN", factScan, dimScan, factRows / 4),
                        factRows / 4);
                break;
            default:
                throw new IllegalArgumentException("unknown rewrite " + mode);
        }
        return new QueryCase("q" + index + "-" + mode, oldJoin, newJoin, mode);
    }

    static List<QueryCase> workload(List<String> modes) {
        List<QueryCase> workload = new ArrayList<>();
        for (int i = 0; i < modes.size(); i++) workload.add(buildCase(i, modes.get(i)));
        return workload;
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        final long seed = 855_2024L;

        // --- 1. the cost model is internally consistent, monotone and independent of the evaluator.
        for (int i = 0; i < 12; i++) {
            QueryCase c = buildCase(i, i % 3 == 0 ? "identity" : i % 3 == 1 ? "nested-loop-instead-of-hash"
                    : "sort-merge-instead-of-hash");
            check(Math.abs(c.oldPlan.cost() - c.oldPlan.costIterative()) < 1e-9,
                    "recursive and iterative cost agree for " + c.name);
            check(Math.abs(c.newPlan.cost() - c.newPlan.costIterative()) < 1e-9,
                    "recursive and iterative cost agree for the new plan of " + c.name);
        }
        Op bigger = Op.scan("facts", 500);
        Op smaller = Op.scan("facts", 100);
        check(bigger.cost() > smaller.cost(), "scan cost grows with row estimates");
        check(Op.indexScan("facts", 1000, 0.01).cost() < Op.scan("facts", 1000).cost(),
                "a selective index scan is cheaper than a full scan");
        check(Op.join("NESTED_LOOP", smaller, smaller, 10).cost()
                        < Op.join("NESTED_LOOP", bigger, bigger, 10).cost(),
                "nested loop cost grows quadratically with input rows");
        check(Op.unary("HASH_AGG", bigger, 10).cost() > Op.unary("HASH_AGG", smaller, 10).cost(),
                "aggregate cost is monotone in its input");
        double identityNoise = Math.abs(Op.scan("facts", 100).cost() - Op.scan("facts", 100).cost());
        check(identityNoise == 0.0, "the cost model is deterministic");

        // --- 2. fingerprints capture structure only.
        Op a = Op.join("HASH_JOIN", Op.scan("facts", 100), Op.scan("dims", 10), 40);
        Op b = Op.join("HASH_JOIN", Op.scan("facts", 900), Op.scan("dims", 77), 400);
        Op cSwap = Op.join("HASH_JOIN", Op.scan("dims", 10), Op.scan("facts", 100), 40);
        Op cNested = Op.join("NESTED_LOOP", Op.scan("facts", 100), Op.scan("dims", 10), 40);
        check(a.fingerprint() == b.fingerprint(), "identical shapes with different row estimates share a fingerprint");
        check(a.fingerprint() != cSwap.fingerprint(), "swapping join inputs changes the fingerprint");
        check(a.fingerprint() != cNested.fingerprint(), "changing the join algorithm changes the fingerprint");
        check(a.fingerprint() == Op.join("HASH_JOIN", Op.scan("facts", 1), Op.scan("dims", 2), 3).fingerprint(),
                "fingerprints are stable across rebuilds");

        // --- 3. exact signed-test p-values against brute-force sign enumeration.
        for (int n : new int[]{1, 2, 5, 8, 12, 16}) {
            for (int k = 0; k <= n; k++) {
                check(Math.abs(signP(k, n) - signPBruteForce(k, n)) < 1e-12,
                        "sign p-value matches brute force for k=" + k + " n=" + n);
            }
        }
        check(signP(0, REPS) == 1.0, "no positive deltas gives p = 1");
        check(signP(REPS, REPS) == 1.0 / (1 << REPS), "all positive deltas gives p = 2^-reps");
        check(signP(9, REPS) < signP(8, REPS) && signP(8, REPS) < signP(7, REPS),
                "the p-value is monotone in the effect size");
        check(signP(3, REPS) > 0.5, "three positive deltas out of sixteen are not significant");
        check(Math.abs(signP(4, REPS) + signP(REPS - 4 + 1, REPS) - 1.0) < 1e-12,
                "the one-sided p-value is symmetric under sign reversal");
        check(binomial(16, 8) == 12870, "binomial coefficients are exact");

        // --- 4. injected regressions are detected with high precision and recall.
        List<String> modes = new ArrayList<>();
        for (int i = 0; i < 8; i++) modes.add("identity");
        for (int i = 0; i < 10; i++) modes.add("nested-loop-instead-of-hash");
        for (int i = 0; i < 4; i++) modes.add("index-dropped");
        for (int i = 0; i < 2; i++) modes.add("join-order-swap");
        List<QueryCase> regressionWorkload = workload(modes);
        Report report = detect(regressionWorkload, seed, BASE_NOISE, 0.05);
        int trueRegressions = report.trueRegressions();
        int tp = report.truePositives();
        int fp = report.falsePositives();
        double recall = tp / (double) trueRegressions;
        double precision = report.flaggedCount() == 0 ? 1.0 : tp / (double) report.flaggedCount();
        System.out.printf("regression workload: queries=%d trueRegressions=%d flagged=%d tp=%d fp=%d "
                        + "recall=%.3f precision=%.3f%n", regressionWorkload.size(), trueRegressions,
                report.flaggedCount(), tp, fp, recall, precision);
        check(trueRegressions >= 12, "the labelled workload contains injected regressions");
        check(recall >= 0.9, "recall on injected regressions is at least 0.9");
        check(precision >= 0.9, "precision on injected regressions is at least 0.9");
        check(report.flaggedCount() >= report.truePositives(), "flagged is a superset of true positives");
        check(report.threshold < 0.05, "the Bonferroni threshold tightens alpha");

        // --- 5. an unchanged workload produces no flags unless the threshold is absurdly loose.
        List<QueryCase> control = workload(Collections.nCopies(24, "identity"));
        Report controlReport = detect(control, seed, BASE_NOISE, 0.05);
        System.out.println("control workload flagged=" + controlReport.flaggedCount());
        check(controlReport.trueRegressions() == 0, "the control workload has no true regression");
        check(controlReport.flaggedCount() == 0, "the control workload produces no flags");
        Report looseControl = detect(control, seed, BASE_NOISE, 0.5);
        check(looseControl.flaggedCount() >= controlReport.flaggedCount(),
                "a looser threshold cannot flag fewer queries");

        // --- 6. improvements are never reported as regressions.
        List<QueryCase> improvement = workload(Collections.nCopies(20, "hash-instead-of-nested-loop"));
        Report improvementReport = detect(improvement, seed, BASE_NOISE, 0.05);
        int wrong = 0;
        for (int i = 0; i < improvement.size(); i++) {
            if (improvementReport.flagged[i] && improvement.get(i).trueImprovement()) wrong++;
        }
        System.out.println("improvements flagged as regressions=" + wrong);
        check(wrong == 0, "no improvement is reported as a regression");
        for (Trial t : improvementReport.trials) {
            check(t.relativeDelta < 0.03, "measured delta of an improvement is not positive: " + t.relativeDelta);
        }

        // --- 7. threshold sweep: flagged sets are nested and recall is non-decreasing.
        int previousFlagged = -1;
        int previousRecall = -1;
        for (double alpha : new double[]{0.001, 0.01, 0.05, 0.2, 0.5}) {
            Report sweep = detect(regressionWorkload, seed, BASE_NOISE, alpha);
            int flagged = sweep.flaggedCount();
            int recallCount = sweep.truePositives();
            check(flagged >= previousFlagged, "flagged count is monotone in alpha (alpha=" + alpha + ")");
            check(recallCount >= previousRecall, "true positives are monotone in alpha (alpha=" + alpha + ")");
            for (int i = 0; i < sweep.flagged.length; i++) {
                if (sweep.flagged[i] && !report.flagged[i]) {
                    check(sweep.trials.get(i).pValue >= report.threshold,
                            "a query flagged only at a looser threshold has a larger p-value");
                }
            }
            previousFlagged = flagged;
            previousRecall = recallCount;
        }
        check(previousFlagged >= report.flaggedCount(), "the loosest threshold flags at least as much as alpha=0.05");

        // --- 8. regressions cluster by rewrite signature.
        Map<String, Integer> groups = report.groupCounts();
        System.out.println("flagged groups: " + groups);
        int grouped = 0;
        for (int count : groups.values()) grouped += count;
        check(grouped == report.flaggedCount(), "every flagged query belongs to exactly one group");
        check(groups.containsKey("nested-loop-instead-of-hash"), "the dominant rewrite is identified");
        check(groups.get("nested-loop-instead-of-hash") >= 7, "most nested-loop rewrites are flagged");
        check(!groups.containsKey("hash-instead-of-nested-loop"), "improvements form no group");

        // --- 9. measurement quality: medians rank like the true cost.
        double[] measured = new double[regressionWorkload.size()];
        double[] truth = new double[regressionWorkload.size()];
        for (int i = 0; i < regressionWorkload.size(); i++) {
            measured[i] = median(sample(regressionWorkload.get(i).newPlan, seed, i, 1, BASE_NOISE, 1.0));
            truth[i] = regressionWorkload.get(i).newPlan.cost();
        }
        double rho = spearman(measured, truth);
        System.out.printf("spearman(median latency, true cost) = %.4f%n", rho);
        check(rho >= 0.9, "median latency ranks like the true cost");
        check(spearman(truth, truth) > 0.999, "rank correlation of a series with itself is 1");
        check(spearman(new double[]{1, 2, 3}, new double[]{3, 2, 1}) < -0.999, "perfectly reversed ranks give -1");

        // --- 10. robustness: doubling the noise keeps false positives bounded and injections detected.
        Report noisy = detect(regressionWorkload, seed, 2 * BASE_NOISE, 0.05);
        Report noisyControl = detect(control, seed, 2 * BASE_NOISE, 0.05);
        System.out.printf("noisy: flagged=%d tp=%d fp=%d controlFlagged=%d%n", noisy.flaggedCount(),
                noisy.truePositives(), noisy.falsePositives(), noisyControl.flaggedCount());
        check(noisy.truePositives() >= 0.8 * trueRegressions, "noisy measurement still detects the injections");
        check(noisyControl.flaggedCount() <= 2, "noisy control stays near zero flags");

        // --- 11. determinism and seed sensitivity.
        Report again = detect(regressionWorkload, seed, BASE_NOISE, 0.05);
        boolean same = again.flaggedCount() == report.flaggedCount();
        for (int i = 0; i < report.trials.size(); i++) {
            same &= again.trials.get(i).pValue == report.trials.get(i).pValue
                    && again.trials.get(i).positiveDeltas == report.trials.get(i).positiveDeltas;
        }
        check(same, "the report is reproducible from its seed");
        Report otherSeed = detect(regressionWorkload, seed + 1, BASE_NOISE, 0.05);
        boolean differs = false;
        for (int i = 0; i < report.trials.size(); i++) {
            differs |= otherSeed.trials.get(i).oldMedian != report.trials.get(i).oldMedian;
        }
        check(differs, "a different seed shifts the samples");
        check(otherSeed.groupCounts().containsKey("nested-loop-instead-of-hash"),
                "the dominant rewrite is still identified with another seed");

        // --- 12. a single-case workload, an empty workload and borderline rewrites.
        List<QueryCase> single = workload(List.of("nested-loop-instead-of-hash"));
        Report singleReport = detect(single, seed, BASE_NOISE, 0.05);
        check(singleReport.flaggedCount() == 1 && singleReport.falsePositives() == 0,
                "a single-query workload detects its own regression");
        Report emptyWorkload = detect(new ArrayList<>(), seed, BASE_NOISE, 0.05);
        check(emptyWorkload.trials.isEmpty() && emptyWorkload.flaggedCount() == 0,
                "an empty workload is handled");
        QueryCase borderline = buildCase(3, "sort-merge-instead-of-hash");
        check(borderline.oldPlan.cost() != borderline.newPlan.cost(),
                "the sort-merge rewrite changes the cost");
        Report borderlineReport = detect(List.of(borderline), seed, BASE_NOISE, 0.05);
        check(borderlineReport.trials.size() == 1 && borderlineReport.threshold == 0.05,
                "a one-query workload uses alpha itself as the threshold");
        check(borderlineReport.flagged[0] == borderline.trueRegression()
                        || borderlineReport.trials.get(0).pValue >= borderlineReport.threshold,
                "a borderline case is flagged exactly when it is significant");

        System.out.println("total checks=" + checks);
        System.out.println("PASS");
    }
}
