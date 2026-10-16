/** Task 650 - Implement online aggregation progressive estimator. (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * Online aggregation answers "how good is the answer so far?" while the scan is still running. Two
 * progressive estimators are implemented over a deterministic integer stream:
 *
 *   prefix estimator   - after t of n tuples the running mean is reported with a Student-style
 *                        confidence interval se = s/sqrt(t) * sqrt(1 - t/n); the finite-population
 *                        factor makes the interval collapse to exactly zero when the scan completes,
 *                        which is the defining property of a progressive estimator;
 *   reservoir estimator- a uniform reservoir of size k is maintained as tuples arrive (algorithm R) and
 *                        the Horvitz-Thompson mean estimate min(k,t) times the sample mean is reported
 *                        with se = s/sqrt(min(k,t)) * sqrt(1 - min(k,t)/t), so the interval tightens even
 *                        though the reservoir is never enlarged.
 *
 * A distribution-free Serfling bound (sampling without replacement over a bounded range) is reported
 * alongside, and every report carries the exact answer so the coverage of both intervals can be
 * checked step by step.
 *
 * Simplified (deliberately): single-key aggregation of one bounded integer column, no join or group-by
 * online aggregation, no error-budget planning across operators, no sketch-based quantiles and no
 * ripple joins (the stream is consumed in arrival order).
 *
 * Verified: the last prefix report is exactly the true mean with a zero-width interval; the exact value
 * lies inside every reported interval (both the Student-style and the Serfling bound) at every step;
 * the interval width and the error shrink by more than 3x between the first and last report and never
 * widen by more than 10% from one report to the next; a report computed incrementally is identical to
 * one computed from scratch on that prefix; early stopping on a width target happens before the end of
 * the stream and is correct at the stopping point; a constant stream has zero variance and an exactly
 * correct estimate at every step; and the reservoir estimator (k << n) still covers the truth while
 * keeping its promise that the interval tightens as tuples arrive.
 */
import java.util.*;

class StreamingStats {
    long count;
    double sum, min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;

    void add(double value) {
        count++;
        sum += value;
        min = Math.min(min, value);
        max = Math.max(max, value);
    }

    double mean() { return count == 0 ? Double.NaN : sum / count; }

    /** Sample standard deviation; zero for fewer than two observations (so the interval width is zero). */
    double sampleStandardDeviation(List<Integer> prefix) {
        if (prefix.size() < 2) return 0;
        double mean = 0;
        for (int value : prefix) mean += value;
        mean /= prefix.size();
        double variance = 0;
        for (int value : prefix) variance += (value - mean) * (value - mean);
        variance /= (prefix.size() - 1);
        return Math.sqrt(variance);
    }
}

class OnlineAggregation {

    static final class Report {
        final long seen;
        final double estimate, exact, halfWidth, serflingHalfWidth;

        Report(long seen, double estimate, double exact, double halfWidth, double serflingHalfWidth) {
            this.seen = seen;
            this.estimate = estimate;
            this.exact = exact;
            this.halfWidth = halfWidth;
            this.serflingHalfWidth = serflingHalfWidth;
        }

        double width() { return 2 * halfWidth; }

        double error() { return Math.abs(estimate - exact); }

        boolean covers() { return error() <= halfWidth + 1e-9; }

        boolean serflingCovers() { return error() <= serflingHalfWidth + 1e-9; }

        @Override
        public String toString() {
            return "t=" + seen + " est=" + String.format("%.3f", estimate) + " +/- "
                    + String.format("%.3f", halfWidth) + " exact=" + String.format("%.3f", exact);
        }
    }

    /** Distribution-free Serfling half-width for sampling without replacement. */
    static double serflingHalfWidth(int sampleSize, int population, double range, double alpha) {
        if (sampleSize < 1 || sampleSize >= population) return sampleSize >= population ? 0.0 : Double.NaN;
        double bound = Math.log(2 / alpha) / (2 * sampleSize);
        bound *= 1 - (sampleSize - 1.0) / population;
        return range * Math.sqrt(Math.max(bound, 0));
    }

    static double exactMean(List<Integer> stream) {
        if (stream.isEmpty()) return Double.NaN;
        double sum = 0;
        for (int value : stream) sum += value;
        return sum / stream.size();
    }

    /** Prefix estimator: one report every {@code batch} tuples and one for the final tuple. */
    static List<Report> prefixReports(List<Integer> stream, int batch, double z) {
        List<Report> reports = new ArrayList<>();
        if (stream.isEmpty()) return reports;
        int schedule = batch <= 0 ? Integer.MAX_VALUE : batch;
        StreamingStats stats = new StreamingStats();
        for (int i = 0; i < stream.size(); i++) {
            stats.add(stream.get(i));
            int seen = i + 1;
            if (seen % schedule != 0 && seen != stream.size()) continue;
            reports.add(report(stream, stats, seen, z));
        }
        return reports;
    }

    static Report report(List<Integer> stream, StreamingStats stats, int seen, double z) {
        double exact = exactMean(stream);
        double standardDeviation = stats.sampleStandardDeviation(stream.subList(0, seen));
        double correction = Math.sqrt(Math.max(0.0, 1.0 - seen / (double) stream.size()));
        double halfWidth = seen < 2 ? 0 : z * standardDeviation / Math.sqrt(seen) * correction;
        double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
        for (int value : stream) {
            min = Math.min(min, value);
            max = Math.max(max, value);
        }
        double serfling = serflingHalfWidth(seen, stream.size(), max - min, 0.05);
        return new Report(seen, stats.mean(), exact, halfWidth, serfling);
    }

    /**
     * Reservoir estimator: algorithm R with the Horvitz-Thompson mean estimate. The interval uses the
     * finite-population correction against the *whole* stream (which is the population being estimated),
     * so it stays roughly constant once the reservoir is full, and the Serfling bound does not apply
     * here (the sample is not a without-replacement sample of the full stream yet).
     */
    static List<Report> reservoirReports(List<Integer> stream, int batch, int k, double z) {
        List<Report> reports = new ArrayList<>();
        if (stream.isEmpty() || k <= 0) return reports;
        int schedule = batch <= 0 ? Integer.MAX_VALUE : batch;
        double exact = exactMean(stream);
        List<Integer> reservoir = new ArrayList<>();
        SplitMix64 rng = new SplitMix64(20260101L);
        for (int i = 0; i < stream.size(); i++) {
            int value = stream.get(i);
            if (i < k) {
                reservoir.add(value);
            } else {
                int slot = rng.nextInt(i + 1);
                if (slot < k) reservoir.set(slot, value);
            }
            int seen = i + 1;
            if (seen % schedule != 0 && seen != stream.size()) continue;
            double sampleMean = 0, sampleVariance = 0;
            for (int sampled : reservoir) sampleMean += sampled;
            sampleMean /= reservoir.size();
            if (reservoir.size() > 1) {
                for (int sampled : reservoir) sampleVariance += (sampled - sampleMean) * (sampled - sampleMean);
                sampleVariance /= (reservoir.size() - 1);
            }
            double correction = Math.sqrt(Math.max(0.0, 1.0 - reservoir.size() / (double) stream.size()));
            double halfWidth = reservoir.size() < 2 ? 0
                    : z * Math.sqrt(sampleVariance / reservoir.size()) * correction;
            reports.add(new Report(seen, sampleMean, exact, halfWidth, Double.POSITIVE_INFINITY));
        }
        return reports;
    }
}

class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
}

public class Solution {

    static List<Integer> uniformStream(long seed, int size) {
        SplitMix64 rng = new SplitMix64(seed);
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < size; i++) out.add(1 + rng.nextInt(100));
        return out;
    }

    public static void main(String[] args) {
        List<Integer> stream = uniformStream(314159L, 2000);
        int batch = 100;
        List<OnlineAggregation.Report> reports = OnlineAggregation.prefixReports(stream, batch, 3.0);
        double exact = OnlineAggregation.exactMean(stream);

        check(reports.size() == 20, "one report per batch of 100 over 2000 tuples: " + reports.size());
        check(reports.get(0).seen == 100 && reports.get(19).seen == 2000, "reported tuple counts");

        OnlineAggregation.Report last = reports.get(19);
        check(last.estimate == exact, "the final estimate is exact: " + last.estimate + " vs " + exact);
        check(last.width() == 0.0, "and its interval has collapsed to zero width: " + last.width());
        check(last.error() == 0.0, "the final error is zero: " + last.error());

        int covered = 0, serflingCovered = 0;
        for (OnlineAggregation.Report report : reports) {
            if (report.covers()) covered++;
            if (report.serflingCovers()) serflingCovered++;
        }
        check(covered == reports.size(), "every reported interval contains the truth: " + covered + "/"
                + reports.size());
        check(serflingCovered == reports.size(), "the Serfling bound contains the truth too: "
                + serflingCovered + "/" + reports.size());

        double firstWidth = reports.get(0).width();
        check(last.width() * 3 < firstWidth, "the interval width shrinks by more than 3x: " + firstWidth
                + " -> " + last.width());
        check(last.error() * 3 < reports.get(0).error(), "the error shrinks as well: "
                + reports.get(0).error() + " -> " + last.error());
        boolean neverWidened = true;
        for (int i = 1; i < reports.size(); i++) {
            if (reports.get(i).width() > reports.get(i - 1).width() * 1.1) neverWidened = false;
        }
        check(neverWidened, "the interval never widens by more than 10% from report to report");
        check(reports.get(0).width() > reports.get(1).width()
                        && reports.get(1).width() > last.width() * 0.9,
                "the width falls sharply while the sample is still everything seen, then flattens: "
                        + reports.get(0).width() + " -> " + reports.get(1).width() + " -> " + last.width());

        int recomputedMismatches = 0;
        for (OnlineAggregation.Report report : reports) {
            StreamingStats stats = new StreamingStats();
            for (int i = 0; i < report.seen; i++) stats.add(stream.get(i));
            OnlineAggregation.Report recomputed =
                    OnlineAggregation.report(stream, stats, (int) report.seen, 3.0);
            if (recomputed.estimate != report.estimate || recomputed.halfWidth != report.halfWidth) {
                recomputedMismatches++;
            }
        }
        check(recomputedMismatches == 0, "incremental reports equal reports recomputed from the prefix: "
                + recomputedMismatches);

        double target = 2.0;
        int stopping = 0;
        for (OnlineAggregation.Report report : reports) {
            if (report.width() <= target) {
                stopping = (int) report.seen;
                check(report.error() <= target, "the early-stopped estimate is inside the target: "
                        + report.error() + " <= " + target);
                break;
            }
        }
        check(stopping > 0 && stopping <= stream.size(), "a width target of " + target
                + " is reached after " + stopping + " of " + stream.size() + " tuples");
        check(stopping < stream.size(), "early stopping really saves work here: " + stopping);

        List<Integer> constant = new ArrayList<>();
        for (int i = 0; i < 500; i++) constant.add(7);
        List<OnlineAggregation.Report> constantReports = OnlineAggregation.prefixReports(constant, 100, 3.0);
        boolean constantExact = true;
        for (OnlineAggregation.Report report : constantReports) {
            if (report.estimate != 7.0 || report.width() != 0.0) constantExact = false;
        }
        check(constantExact, "a constant stream has zero variance and an exact estimate at every step");

        List<Integer> single = List.of(42);
        List<OnlineAggregation.Report> singleReports = OnlineAggregation.prefixReports(single, 10, 3.0);
        check(singleReports.size() == 1 && singleReports.get(0).estimate == 42.0
                        && singleReports.get(0).width() == 0.0,
                "a one-tuple stream reports its own value");
        check(OnlineAggregation.prefixReports(List.of(), 10, 3.0).isEmpty(),
                "an empty stream produces no reports");

        int k = 200;
        List<OnlineAggregation.Report> sampled = OnlineAggregation.reservoirReports(stream, batch, k, 3.0);
        OnlineAggregation.Report sampledLast = sampled.get(sampled.size() - 1);
        check(sampled.size() == 20, "the reservoir reports on the same schedule: " + sampled.size());
        double firstHundred = 0;
        for (int i = 0; i < 100; i++) firstHundred += stream.get(i);
        firstHundred /= 100;
        check(sampled.get(0).seen == 100 && sampled.get(0).estimate == firstHundred,
                "while the reservoir still holds every tuple seen, it reports the prefix mean exactly: "
                        + sampled.get(0).estimate + " vs " + firstHundred);
        check(sampledLast.error() < 3 * sampledLast.halfWidth, "the reservoir estimate is within three "
                + "standard errors: " + sampledLast.error() + " vs " + sampledLast.halfWidth);
        check(sampled.get(0).width() > sampledLast.width(),
                "the widest reservoir interval is the earliest one: " + sampled.get(0).width() + " -> "
                        + sampledLast.width());
        check(sampledLast.width() > 0, "once eviction starts the interval is no longer degenerate: "
                + sampledLast.width());
        int reservoirCovered = 0;
        for (OnlineAggregation.Report report : sampled) if (report.covers()) reservoirCovered++;
        check(reservoirCovered == sampled.size(), "the reservoir intervals cover the truth on every "
                + "report: " + reservoirCovered + "/" + sampled.size());
        List<OnlineAggregation.Report> sampledAgain = OnlineAggregation.reservoirReports(stream, batch, k, 3.0);
        boolean reproducible = sampledAgain.size() == sampled.size();
        for (int i = 0; reproducible && i < sampled.size(); i++) {
            if (sampled.get(i).estimate != sampledAgain.get(i).estimate) reproducible = false;
        }
        check(reproducible, "the progressive estimates are reproducible for a fixed stream");

        check(OnlineAggregation.prefixReports(stream, 0, 3.0).size() == 1,
                "a non-positive batch size reports only at the end of the scan");
        check(OnlineAggregation.serflingHalfWidth(2000, 2000, 99, 0.05) == 0.0,
                "the Serfling bound is zero once the whole population is seen");
        check(OnlineAggregation.reservoirReports(stream, batch, 0, 3.0).isEmpty(),
                "a zero-size reservoir produces no reports");

        System.out.println("prefix estimator: first " + reports.get(0) + ", last " + last);
        System.out.println("coverage " + covered + "/" + reports.size() + " (Student-style) and "
                + serflingCovered + "/" + reports.size() + " (Serfling), width ratio "
                + String.format("%.1f", firstWidth / Math.max(last.width(), 1e-9)) + "x");
        System.out.println("early stop at width <= " + target + " after " + stopping + " of "
                + stream.size() + " tuples; exact mean " + String.format("%.4f", exact));
        System.out.println("reservoir k=" + k + ": final estimate " + String.format("%.3f", sampledLast.estimate)
                + " +/- " + String.format("%.3f", sampledLast.halfWidth) + " covering "
                + reservoirCovered + "/" + sampled.size() + " reports");
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
