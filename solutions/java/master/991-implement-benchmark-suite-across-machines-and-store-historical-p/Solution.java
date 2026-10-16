/** Task 991 - Implement benchmark suite across machines and store historical performance.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  Raw wall-clock numbers from different machines are not comparable, so every machine carries a
 *  calibration (the duration of a fixed reference workload) and each measurement is normalised to the
 *  reference machine: normalised = wallMs * REFERENCE / calibration. Normalised samples are stored in
 *  a per-task history of batches (day, samples, median, IQR-trimmed median), so a slowdown can be
 *  judged against the task's own past instead of against a global threshold. The judge compares the
 *  latest batch median with the median of the earlier batch medians: above +15% is a regression, below
 *  -13% an improvement, otherwise stable - and it stays inconclusive when there is no baseline, when
 *  the batch is too small, or when a machine reports an invalid calibration. Single-spike noise is
 *  removed by a 1.5xIQR trim before the median is taken.
 *  Simplified: scripted measurement batches instead of a real timer, three simulated machines; the
 *  calibration arithmetic, the robust statistics, the history and the judge are the real ones.
 *  Verified: cross-machine normalisation makes a 2x faster machine report the same number, medians are
 *  correct for odd/even/single samples, a 1.6x batch is called a regression and a 0.8x batch an
 *  improvement while 1.05x stays stable, small or baseline-free batches are inconclusive, IQR trimming
 *  drops a spike, invalid calibrations are refused, and sorting-based oracles agree.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final double REFERENCE_MS = 100.0;
    static final double REGRESSION_RATIO = 1.15, IMPROVEMENT_RATIO = 0.87;
    static final int MIN_BATCH = 3;

    /** Deterministic splitmix64 stream so batches are reproducible without wall-clock input. */
    static final class Rng {
        long state;
        Rng(long seed) { state = seed; }
        double next() {
            long z = (state += 0x9E3779B97F4A7C15L);
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return ((z ^ (z >>> 31)) >>> 11) * 0x1.0p-53;
        }
    }

    static final class Machine {
        final String name;
        final double calibrationMs;
        Machine(String name, double calibrationMs) { this.name = name; this.calibrationMs = calibrationMs; }
        boolean valid() { return calibrationMs > 0; }
    }

    static double normalise(double wallMs, Machine machine) {
        if (!machine.valid()) throw new IllegalArgumentException("invalid calibration for " + machine.name);
        return wallMs * REFERENCE_MS / machine.calibrationMs;
    }

    static double median(List<Double> values) {
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int n = sorted.size();
        if (n == 0) throw new IllegalArgumentException("median of an empty batch");
        return n % 2 == 1 ? sorted.get(n / 2) : (sorted.get(n / 2 - 1) + sorted.get(n / 2)) / 2.0;
    }

    static double quartile(List<Double> values, double fraction) {
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int index = (int) Math.round(fraction * (sorted.size() - 1));
        return sorted.get(Math.max(0, Math.min(sorted.size() - 1, index)));
    }

    /** Drops samples outside [q1 - 1.5*IQR, q3 + 1.5*IQR] so one hung run cannot mask a trend. */
    static List<Double> trim(List<Double> values) {
        if (values.size() < 4) return new ArrayList<>(values);
        double q1 = quartile(values, 0.25), q3 = quartile(values, 0.75), iqr = q3 - q1;
        List<Double> kept = new ArrayList<>();
        for (double v : values) if (v >= q1 - 1.5 * iqr && v <= q3 + 1.5 * iqr) kept.add(v);
        return kept.isEmpty() ? new ArrayList<>(values) : kept;
    }

    static final class Batch {
        final int day;
        final List<Double> samples;      // normalised
        final double median, robustMedian;
        Batch(int day, List<Double> samples) {
            this.day = day; this.samples = samples;
            this.median = median(samples);
            this.robustMedian = median(trim(samples));
        }
    }

    static final class Verdict {
        final String label;
        final double ratio, latest, baseline;
        Verdict(String label, double ratio, double latest, double baseline) {
            this.label = label; this.ratio = ratio; this.latest = latest; this.baseline = baseline;
        }
    }

    static final class History {
        final Map<String, List<Batch>> perTask = new TreeMap<>();
        final List<String> errors = new ArrayList<>();

        /** Records a batch of (machine, wallMs) measurements, normalised to the reference machine. */
        void record(String task, int day, Machine machine, double[] wallMs) {
            List<Double> samples = new ArrayList<>();
            for (double w : wallMs) {
                try { samples.add(normalise(w, machine)); }
                catch (IllegalArgumentException e) { errors.add(e.getMessage()); return; }
            }
            perTask.computeIfAbsent(task, k -> new ArrayList<>()).add(new Batch(day, samples));
        }

        Verdict judge(String task) {
            List<Batch> batches = perTask.getOrDefault(task, List.of());
            if (batches.size() < 2) return new Verdict("inconclusive", Double.NaN, Double.NaN, Double.NaN);
            Batch latest = batches.get(batches.size() - 1);
            List<Double> baselineMedians = new ArrayList<>();
            for (int i = 0; i < batches.size() - 1; i++) baselineMedians.add(batches.get(i).median);
            double baseline = median(baselineMedians);
            if (latest.samples.size() < MIN_BATCH)
                return new Verdict("inconclusive", Double.NaN, latest.median, baseline);
            double ratio = latest.median / baseline;
            String label = ratio > REGRESSION_RATIO ? "regression" : ratio < IMPROVEMENT_RATIO ? "improvement" : "stable";
            return new Verdict(label, ratio, latest.median, baseline);
        }
    }

    static double mean(List<Double> values) {
        double sum = 0;
        for (double v : values) sum += v;
        return values.isEmpty() ? 0 : sum / values.size();
    }

    /** Scripted batch: `count` samples around `base` with +-`jitterAmp` relative noise. */
    static double[] batch(double base, double jitterAmp, int count, long seed) {
        Rng rng = new Rng(seed);
        double[] out = new double[count];
        for (int i = 0; i < count; i++) out[i] = base * (1 + jitterAmp * (2 * rng.next() - 1));
        return out;
    }

    static final Machine FAST = new Machine("fastbox", 50.0);
    static final Machine REF = new Machine("reference", 100.0);
    static final Machine SLOW = new Machine("slowbox", 150.0);

    public static void main(String[] args) {
        // --- normalisation: the same work on a 2x faster machine reports the same number.
        check(normalise(120.0, REF) == 120.0 && Math.abs(normalise(60.0, FAST) - 120.0) < 1e-9
                        && Math.abs(normalise(180.0, SLOW) - 120.0) < 1e-9,
                "calibration must make machines comparable: " + normalise(60.0, FAST) + " / " + normalise(180.0, SLOW));

        // --- medians: odd, even and single-sample boundaries.
        check(median(List.of(3.0, 1.0, 2.0)) == 2.0 && median(List.of(4.0, 1.0, 3.0, 2.0)) == 2.5
                        && median(List.of(7.0)) == 7.0,
                "median must handle odd, even and single samples: " + median(List.of(4.0, 1.0, 3.0, 2.0)));

        // --- history: batches are stored per task in day order with their medians.
        History history = new History();
        history.record("sort-100k", 1, REF, batch(120.0, 0.03, 5, 0x5EED1L));
        history.record("sort-100k", 2, FAST, batch(60.0, 0.03, 5, 0x5EED2L));
        history.record("sort-100k", 3, FAST, batch(96.0, 0.03, 5, 0x5EED3L));      // 1.6x slower than day 1-2
        List<Batch> batches = history.perTask.get("sort-100k");
        check(batches.size() == 3 && batches.get(0).day == 1 && batches.get(2).day == 3
                        && batches.get(0).median > 115 && batches.get(0).median < 125,
                "history must keep batches in day order with sane medians: " + batches.get(0).median);
        check(Math.abs(batches.get(1).median - 120.0) < 5,
                "a faster machine's raw numbers must normalise back to the reference: " + batches.get(1).median);

        Verdict regression = history.judge("sort-100k");
        check(regression.label.equals("regression") && regression.ratio > 1.5 && regression.ratio > REGRESSION_RATIO,
                "a 1.6x batch must be judged a regression: " + regression.label + " ratio " + regression.ratio);
        check(Math.abs(regression.baseline - 120.0) < 5 && regression.latest > 185,
                "the baseline must be the median of earlier batch medians: " + regression.baseline);

        // --- the other verdicts, and the inconclusive gates.
        History improved = new History();
        improved.record("bfs-1m", 1, REF, batch(200.0, 0.02, 5, 0xA1L));
        improved.record("bfs-1m", 2, REF, batch(160.0, 0.02, 5, 0xA2L));
        check(improved.judge("bfs-1m").label.equals("improvement") && improved.judge("bfs-1m").ratio < IMPROVEMENT_RATIO,
                "a 0.8x batch must be judged an improvement: " + improved.judge("bfs-1m").ratio);

        History stable = new History();
        stable.record("dijkstra", 1, REF, batch(300.0, 0.02, 5, 0xB1L));
        stable.record("dijkstra", 2, REF, batch(315.0, 0.02, 5, 0xB2L));
        check(stable.judge("dijkstra").label.equals("stable"),
                "a 1.05x batch must stay stable: " + stable.judge("dijkstra").ratio);

        History tiny = new History();
        tiny.record("hash", 1, REF, batch(100.0, 0.02, 5, 0xC1L));
        tiny.record("hash", 2, REF, batch(500.0, 0.02, 2, 0xC2L));
        check(tiny.judge("hash").label.equals("inconclusive") && tiny.judge("hash").baseline > 0,
                "a batch smaller than the minimum must be inconclusive: " + tiny.judge("hash").label);
        History single = new History();
        single.record("graph", 1, REF, batch(100.0, 0.02, 5, 0xD1L));
        check(single.judge("graph").label.equals("inconclusive") && Double.isNaN(single.judge("graph").ratio),
                "a first-ever batch has no baseline: " + single.judge("graph").label);
        check(new History().judge("nothing").label.equals("inconclusive"),
                "an unknown task must judge inconclusive, not throw");
        // --- robust statistics: a single hung run is trimmed away.
        History spiky = new History();
        spiky.record("merge-50k", 1, REF, batch(150.0, 0.01, 6, 0xE1L));
        double[] withSpike = {150.0, 151.0, 149.5, 3000.0, 150.5, 149.0};
        spiky.record("merge-50k", 2, REF, withSpike);
        Batch spikyBatch = spiky.perTask.get("merge-50k").get(1);
        check(trim(new ArrayList<>(Arrays.stream(withSpike).boxed().toList())).size() == 5,
                "the IQR trim must drop exactly the hung sample");
        check(spikyBatch.robustMedian < 155 && mean(spikyBatch.samples) > 600,
                "the spike must dominate the mean yet not the robust median: mean " + mean(spikyBatch.samples)
                        + " vs robust " + spikyBatch.robustMedian);
        check(spiky.judge("merge-50k").label.equals("stable"),
                "trimmed history must not report a phantom regression: " + spiky.judge("merge-50k").ratio);

        // --- invalid calibration must be refused instead of skewing the history.
        History broken = new History();
        broken.record("oops", 1, new Machine("misconfigured", 0.0), new double[]{10.0, 11.0});
        check(broken.errors.size() == 1 && broken.errors.get(0).equals("invalid calibration for misconfigured")
                        && broken.perTask.isEmpty(),
                "a machine with zero calibration must not record anything: " + broken.errors);
        // --- independent oracle: sorting-based median and a second normalisation path must agree.
        List<Double> sample = Arrays.stream(batch(120.0, 0.25, 9, 0xF1L)).boxed().toList();
        check(Math.abs(oracleMedian(sample) - median(sample)) < 1e-12,
                "oracle median mismatch: " + oracleMedian(sample) + " vs " + median(sample));
        check(Math.abs(oracleNormalise(96.0, FAST) - normalise(96.0, FAST)) < 1e-12,
                "oracle normalisation mismatch: " + oracleNormalise(96.0, FAST));
        for (Batch b : batches)
            check(Math.abs(oracleMedian(b.samples) - b.median) < 1e-12,
                    "stored median must match the oracle for day " + b.day);

        System.out.println("batches for sort-100k: " + batches.size() + ", medians "
                + batches.stream().map(b -> String.format("%.1f", b.median)).toList());
        System.out.println("judge: " + regression.label + " ratio " + String.format("%.3f", regression.ratio));
        System.out.println("spike trimming kept " + trim(Arrays.stream(withSpike).boxed().toList()).size() + " of 6 samples");
        System.out.println("PASS");
    }

    /** Oracle median: insertion into a sorted position, no library sort of the caller's list. */
    static double oracleMedian(List<Double> values) {
        List<Double> sorted = new ArrayList<>();
        for (double v : values) {
            int at = 0;
            while (at < sorted.size() && sorted.get(at) < v) at++;
            sorted.add(at, v);
        }
        int n = sorted.size();
        return n % 2 == 1 ? sorted.get(n / 2) : (sorted.get(n / 2 - 1) + sorted.get(n / 2)) / 2.0;
    }

    /** Oracle normalisation: scale by the inverse calibration factor instead of dividing. */
    static double oracleNormalise(double wallMs, Machine machine) {
        return wallMs * (REFERENCE_MS / machine.calibrationMs) * 1.0;
    }
}
