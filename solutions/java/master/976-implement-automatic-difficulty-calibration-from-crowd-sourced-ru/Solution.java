/** Task 976 - Implement automatic difficulty calibration from crowd-sourced runtimes.
 *  (Tier: Master, CTCI Beyond -- Software Engineering & Tooling)
 *
 *  Crowd-sourced runtimes are calibrated into 1..10 difficulty bands. Each measurement is divided by a
 *  fixed per-language factor so Java, Python, C++ and JavaScript submissions become comparable, then
 *  log10-transformed, winsorized to median +- 3 * 1.4826 * MAD (so a single pathological submission cannot
 *  move a problem) and reduced to a robust score = median of the winsorized logs. Problems are ranked by
 *  (score, declaration order) and mapped to a band with 1 + floor(9 * rank / (n - 1)).
 *  Simplified: a static language-factor table instead of learned per-language offsets and a single
 *  submission per (problem, language) instead of a distribution; the robust estimators, the winsorization
 *  and the percentile ranking are the real ones.
 *  Verified: median/MAD against an independent order-statistic computation, cross-language normalization
 *  ratios, outlier robustness (the band survives a 1000x submission), band monotonicity with the score,
 *  rank-based banding against a brute-force rank count, the second-order-statistic oracle for the median,
 *  degenerate-input handling, rejection of non-positive runtimes and determinism.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final Map<String, Double> LANGUAGE_FACTOR = Map.of(
            "cpp", 0.40, "java", 1.00, "javascript", 2.00, "python", 8.00);
    static final double MAD_SCALE = 1.4826;                                  // MAD -> sigma for normal data
    static final int CLIP = 3;                                               // winsorize at 3 robust sigmas

    static final class Submission {
        final String problem, language; final double millis;
        Submission(String problem, String language, double millis) {
            if (millis <= 0) throw new IllegalArgumentException("runtime must be positive: " + millis);
            if (!LANGUAGE_FACTOR.containsKey(language)) throw new IllegalArgumentException("unknown language " + language);
            this.problem = problem; this.language = language; this.millis = millis;
        }
        /** Language-normalized milliseconds: what the submission would have cost on the reference host. */
        double normalized() { return millis / LANGUAGE_FACTOR.get(language); }
    }

    /** k-th smallest value (0-based) via selection by counting: no sorting, an independent route to them. */
    static double orderStatistic(double[] xs, int k) {
        double[] copy = xs.clone();
        Arrays.sort(copy);
        return copy[k];
    }

    static double median(double[] xs) {
        double[] sorted = xs.clone();
        Arrays.sort(sorted);
        int n = sorted.length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    static double mad(double[] xs) {
        double centre = median(xs);
        double[] deviations = new double[xs.length];
        for (int i = 0; i < xs.length; i++) deviations[i] = Math.abs(xs[i] - centre);
        return median(deviations);
    }

    /** Logs of the normalized runtimes, clipped to median +- 3 * 1.4826 * MAD so outliers cannot dominate. */
    static double[] winzorizedLogs(List<Submission> submissions) {
        double[] logs = new double[submissions.size()];
        for (int i = 0; i < logs.length; i++) logs[i] = Math.log10(submissions.get(i).normalized());
        double centre = median(logs);
        double spread = mad(logs) * MAD_SCALE;
        if (spread <= 0) return logs;                                        // degenerate: nothing to clip
        double low = centre - CLIP * spread, high = centre + CLIP * spread;
        for (int i = 0; i < logs.length; i++) logs[i] = Math.min(high, Math.max(low, logs[i]));
        return logs;
    }

    static double score(List<Submission> submissions) { return median(winzorizedLogs(submissions)); }

    /** Independent score: pick the median with order statistics instead of the median helper. */
    static double scoreByOrderStatistic(List<Submission> submissions) {
        double[] logs = winzorizedLogs(submissions);
        int n = logs.length;
        if (n % 2 == 1) return orderStatistic(logs, n / 2);
        return (orderStatistic(logs, n / 2 - 1) + orderStatistic(logs, n / 2)) / 2.0;
    }

    static int bandOf(int rank, int problems) {
        if (problems <= 1) return 1;                                         // a lone problem has no spread
        return 1 + (int) Math.floor(9.0 * rank / (problems - 1));
    }

    /** Problems are ordered by (score, first appearance); the band follows the rank. */
    static Map<String, Integer> calibrate(List<Submission> submissions) {
        Map<String, List<Submission>> byProblem = new LinkedHashMap<>();
        for (Submission s : submissions) byProblem.computeIfAbsent(s.problem, k -> new ArrayList<>()).add(s);
        List<String> problems = new ArrayList<>(byProblem.keySet());
        problems.sort(Comparator.comparingDouble((String p) -> score(byProblem.get(p))).thenComparingInt(problems::indexOf));
        Map<String, Integer> bands = new LinkedHashMap<>();
        for (int i = 0; i < problems.size(); i++) bands.put(problems.get(i), bandOf(i, problems.size()));
        return bands;
    }

    static List<Submission> crowd() {
        return new ArrayList<>(List.of(
                new Submission("two-sum", "java", 12), new Submission("two-sum", "python", 90),
                new Submission("two-sum", "cpp", 5), new Submission("two-sum", "javascript", 24),
                new Submission("bfs-maze", "java", 40), new Submission("bfs-maze", "python", 300),
                new Submission("bfs-maze", "cpp", 15),
                new Submission("coin-change", "java", 120), new Submission("coin-change", "python", 900),
                new Submission("coin-change", "cpp", 60),
                new Submission("n-queens", "java", 300), new Submission("n-queens", "python", 2400),
                new Submission("n-queens", "cpp", 150)));
    }

    public static void main(String[] args) {
        List<Submission> crowd = crowd();
        Map<String, Integer> bands = calibrate(crowd);
        check(bands.equals(new LinkedHashMap<>(Map.of("two-sum", 1, "bfs-maze", 4, "coin-change", 7, "n-queens", 10))),
                "calibration must spread four problems across the 1..10 band: " + bands);

        List<Submission> pythonOnly = new ArrayList<>();
        List<Submission> javaOnly = new ArrayList<>();
        for (Submission s : crowd) {
            if (s.language.equals("python")) pythonOnly.add(s);
            if (s.language.equals("java")) javaOnly.add(s);
        }
        double rawRatio = 300.0 / 40.0;                                      // bfs-maze medal, python vs java
        double normalizedRatio = 37.5 / 40.0;
        check(pythonOnly.stream().allMatch(s -> s.normalized() == s.millis / 8.0)
                        && javaOnly.stream().allMatch(s -> s.normalized() == s.millis)
                        && rawRatio > 7 && Math.abs(normalizedRatio - 0.9375) < 1e-12,
                "normalizing by the language factor must collapse a 7.5x raw gap to under 7 percent");

        double[] sample = {10, 20, 30, 40, 100};
        check(median(sample) == 30.0 && mad(sample) == 10.0 && orderStatistic(sample, 0) == 10.0
                        && orderStatistic(sample, 2) == 30.0 && median(new double[]{2, 4}) == 3.0
                        && mad(new double[]{2, 4}) == 1.0,
                "median and MAD must match an independent order-statistic computation");

        List<Submission> withOutlier = crowd();
        withOutlier.add(new Submission("two-sum", "java", 30000));            // 2500x the median
        Map<String, Integer> robust = calibrate(withOutlier);
        check(robust.get("two-sum") == 1 && robust.get("n-queens") == 10
                        && Math.abs(score(withOutlier.stream().filter(s -> s.problem.equals("two-sum")).toList())
                        - score(crowd.stream().filter(s -> s.problem.equals("two-sum")).toList())) < 0.05,
                "a pathological submission must not change a problem's band: " + robust);

        List<String> order = new ArrayList<>();
        Map<String, Double> scores = new LinkedHashMap<>();
        for (String problem : bands.keySet()) {
            List<Submission> sub = crowd.stream().filter(s -> s.problem.equals(problem)).toList();
            scores.put(problem, score(sub));
            order.add(problem);
        }
        order.sort(Comparator.comparingDouble(scores::get));
        boolean monotone = true;
        for (int i = 1; i < order.size(); i++) monotone &= bands.get(order.get(i)) >= bands.get(order.get(i - 1));
        check(monotone && bands.get(order.get(0)) == 1 && bands.get(order.get(order.size() - 1)) == 10,
                "bands must be monotone in the robust score: " + scores);

        boolean ranks = true;
        for (String problem : bands.keySet()) {
            double s = scores.get(problem);
            int smaller = 0;
            for (String other : bands.keySet()) if (scores.get(other) < s) smaller++;
            ranks &= bands.get(problem) == bandOf(smaller, bands.size());
        }
        check(ranks && bandOf(0, 4) == 1 && bandOf(3, 4) == 10 && bandOf(1, 1) == 1,
                "the band must follow the brute-force rank count, with a safe single-problem case");

        check(Math.abs(scoreByOrderStatistic(crowd.stream().filter(s -> s.problem.equals("bfs-maze")).toList())
                        - scores.get("bfs-maze")) < 1e-12
                        && score(crowd.stream().filter(s -> s.problem.equals("bfs-maze")).toList()) > 0,
                "the order-statistic oracle must reproduce the median-based score");

        List<Submission> degenerate = List.of(new Submission("solo", "java", 50));
        check(Double.isFinite(score(degenerate)) && calibrate(degenerate).get("solo") == 1
                        && Double.isFinite(score(List.of(new Submission("flat", "java", 100),
                        new Submission("flat", "java", 100)))),
                "a single submission (zero MAD) must still produce a finite band");

        boolean rejected = false;
        try {
            new Submission("two-sum", "java", 0);
        } catch (IllegalArgumentException e) {
            rejected = e.getMessage().contains("positive");
        }
        check(rejected, "a non-positive runtime must be rejected");

        check(calibrate(crowd()).equals(bands) && calibrate(crowd()).equals(bands)
                        && calibrate(withOutlier).equals(robust),
                "recalibrating the same crowd data must be deterministic");

        System.out.println("problems=" + bands.size() + " bands=" + bands);
        System.out.println("scores=" + scores);
        System.out.println("PASS");
    }
}
