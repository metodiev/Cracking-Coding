/** Task 937 - Implement deployment pipeline with canary analysis using online metrics and rollback.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 *  A Kayenta-style progressive rollout: a candidate version is exposed at 5%, 25%, 50% and 100% of the
 *  traffic, and after every online metric window the accumulated canary sample is judged against the
 *  baseline sample per metric (relative deviation vs the metric's allowance: <= 1 allowed, <= 2 marginal,
 *  > 2 fail). A fail or a weighted score above 1 rewinds the rollout to the last stable version; an
 *  indecisive rollout with no fresh data is blocked instead of promoted, so the pipeline fails closed.
 *  Simplified: window means instead of raw samples, a fixed window budget per stage and a scripted metric
 *  stream; the SLO judgement, the fail-fast abort and the blast-radius accounting are the real ones.
 *  Verified: healthy candidates reach 100%, hard regressions roll back at the 25% stage before the blast
 *  radius grows, marginal-but-acceptable candidates still promote, the judge agrees with an independent
 *  reference implementation of the same rule, and rollbacks keep the stable version live.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final double EPS = 1e-9;
    static final int[] SHARES = {5, 25, 50, 100};   // progressive traffic exposure
    static final int PER_STAGE = 6;                 // online metric windows analysed per stage

    /** A judged metric: `allowed` is the tolerated relative worsening (0.10 = 10% worse than baseline). */
    static final class Metric {
        final String name;
        final double weight, allowed;
        Metric(String name, double weight, double allowed) {
            this.name = name; this.weight = weight; this.allowed = allowed;
        }
    }

    /** One online window: the mean of each metric for the baseline and for the canary. */
    static final class Window {
        final double[] base, canary;
        Window(double[] base, double[] canary) { this.base = base; this.canary = canary; }
    }

    static final class Verdict {
        final String decision;
        final double score;
        final double[] norm;                        // per-metric deviation divided by its allowance
        final int failedMetric, marginal, windows;
        Verdict(String decision, double score, double[] norm, int failedMetric, int marginal, int windows) {
            this.decision = decision; this.score = score; this.norm = norm;
            this.failedMetric = failedMetric; this.marginal = marginal; this.windows = windows;
        }
    }

    static List<Metric> metrics() {
        return List.of(new Metric("error-rate", 0.4, 0.10),
                       new Metric("p99-latency", 0.4, 0.10),
                       new Metric("saturation", 0.2, 0.05));
    }

    /** The online judge: accumulated windows -> per-metric allowance-normalised deviation -> verdict. */
    static Verdict judge(List<Metric> metrics, List<Window> windows, int used) {
        double[] norm = new double[metrics.size()];
        int failed = -1, marginal = 0;
        double weighted = 0, weightSum = 0;
        for (int m = 0; m < metrics.size(); m++) {
            double base = 0, canary = 0;
            for (int w = 0; w < used; w++) { base += windows.get(w).base[m]; canary += windows.get(w).canary[m]; }
            double deviation = (canary / used - base / used) / Math.max(base / used, EPS);
            norm[m] = deviation / metrics.get(m).allowed;
            if (norm[m] > 2) { if (failed < 0) failed = m; }
            else if (norm[m] > 1) marginal++;
            weighted += metrics.get(m).weight * norm[m];
            weightSum += metrics.get(m).weight;
        }
        double score = weighted / weightSum;
        String decision = failed >= 0 || score > 1 ? "rollback" : "promote";
        return new Verdict(decision, score, norm, failed, marginal, used);
    }

    static final class Deployment {
        final String service;
        final List<String> log = new ArrayList<>();
        String result = "pending";
        int liveVersion = 12, candidateVersion = 13;
        int windowsUsed, stagesPassed, maxShare, rollbackShare = -1;
        Verdict last;

        Deployment(String service) { this.service = service; }

        void run(List<Metric> metrics, List<Window> windows) {
            for (int stage = 0; stage < SHARES.length; stage++) {
                maxShare = Math.max(maxShare, SHARES[stage]);
                Verdict verdict = null;
                for (int w = 0; w < PER_STAGE; w++) {
                    int available = Math.min(windows.size(), windowsUsed + 1);
                    if (available == windowsUsed) break;             // no fresh online data
                    windowsUsed = available;
                    verdict = judge(metrics, windows, windowsUsed);
                    if (verdict.decision.equals("rollback")) break;  // fail fast
                }
                if (windowsUsed == stage * PER_STAGE) {              // fail closed: never promote blind
                    result = "blocked (no data)";
                    log.add("no data at " + SHARES[stage] + "%");
                    return;
                }
                last = verdict;
                if (verdict.decision.equals("rollback")) {
                    rollbackShare = SHARES[stage];
                    result = "rolled back at " + SHARES[stage] + "%";
                    log.add("rollback at " + SHARES[stage] + "% after " + windowsUsed + " windows, score "
                            + String.format("%.3f", verdict.score));
                    return;
                }
                stagesPassed++;
                log.add("stage " + SHARES[stage] + "% passed after " + windowsUsed + " windows ("
                        + verdict.marginal + " marginal, score " + String.format("%.3f", verdict.score) + ")");
            }
            liveVersion = candidateVersion;
            result = "promoted";
            log.add("promoted v" + candidateVersion);
        }
    }

    /** Independent reference for the judge: reverse-order means, separate aggregation pass. */
    static Verdict reference(List<Metric> metrics, List<Window> windows, int used) {
        double[] norm = new double[metrics.size()];
        for (int m = 0; m < metrics.size(); m++) {
            double base = 0, canary = 0;
            for (int w = used - 1; w >= 0; w--) { base += windows.get(w).base[m]; canary += windows.get(w).canary[m]; }
            norm[m] = ((canary / used) / Math.max(base / used, EPS) - 1) / metrics.get(m).allowed;
        }
        double weighted = 0, weightSum = 0;
        int failed = -1, marginal = 0;
        for (int m = 0; m < metrics.size(); m++) {
            weighted += metrics.get(m).weight * norm[m];
            weightSum += metrics.get(m).weight;
            if (norm[m] > 2 && failed < 0) failed = m;
            else if (norm[m] > 1 && norm[m] <= 2) marginal++;
        }
        double score = weighted / weightSum;
        return new Verdict(failed >= 0 || score > 1 ? "rollback" : "promote", score, norm, failed, marginal, used);
    }

    static boolean sameVerdict(Verdict a, Verdict b) {
        if (!a.decision.equals(b.decision) || a.failedMetric != b.failedMetric || a.marginal != b.marginal) return false;
        if (Math.abs(a.score - b.score) > 1e-9) return false;
        for (int m = 0; m < a.norm.length; m++) if (Math.abs(a.norm[m] - b.norm[m]) > 1e-9) return false;
        return true;
    }

    /** Deterministic splitmix64 noise so every window differs slightly without wall-clock input. */
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

    /** 30 online windows: baseline 1% errors, 120 ms p99, 0.60 saturation; canary shape per `kind`. */
    static List<Window> stream(String kind) {
        Rng rng = new Rng(0x5DEECE66DL);
        List<Window> windows = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            double[] base = {0.0100, 120.0, 0.600};
            double[] canary = {0.0102, 121.0, 0.603};
            if (kind.equals("latency-regression") && i >= 9) canary[1] = 400.0;   // hard p99 blow-up
            if (kind.equals("marginal")) { canary[0] = 0.0112; canary[1] = 122.4; canary[2] = 0.606; }
            if (kind.equals("better")) { canary[0] = 0.0094; canary[1] = 117.0; canary[2] = 0.596; }
            if (kind.startsWith("zero-baseline")) { base[0] = 0.0; canary[0] = kind.endsWith("drift") ? 5e-5 : 0.0; }
            for (int m = 0; m < 3; m++) {                                       // +-1% window noise
                base[m] *= 1 + 0.02 * (rng.next() - 0.5);
                canary[m] *= 1 + 0.02 * (rng.next() - 0.5);
            }
            windows.add(new Window(base, canary));
        }
        return windows;
    }

    public static void main(String[] args) {
        List<Metric> metrics = metrics();

        Deployment healthy = new Deployment("checkout");
        healthy.run(metrics, stream("healthy"));
        check(healthy.result.equals("promoted") && healthy.liveVersion == 13 && healthy.stagesPassed == 4
                        && healthy.windowsUsed == 4 * PER_STAGE && healthy.maxShare == 100,
                "a healthy candidate must walk all stages to 100%: " + healthy.log);

        Deployment regressed = new Deployment("checkout");
        regressed.run(metrics, stream("latency-regression"));
        check(regressed.result.equals("rolled back at 25%") && regressed.liveVersion == 12
                        && regressed.rollbackShare == 25 && regressed.maxShare <= 25,
                "a p99 regression must roll back at 25% keeping the blast radius bounded: " + regressed.log);

        check(regressed.windowsUsed == 2 * PER_STAGE - 2 && regressed.last.failedMetric == 1
                        && regressed.last.norm[1] > 2 && regressed.windowsUsed < 4 * PER_STAGE,
                "the regression must abort as soon as the latency metric fails: " + regressed.windowsUsed
                        + " windows, norms " + Arrays.toString(regressed.last.norm));
        check(regressed.log.get(regressed.log.size() - 1).startsWith("rollback")
                        && regressed.log.stream().noneMatch(l -> l.startsWith("promoted")),
                "a rollback must log no promotion: " + regressed.log);

        Deployment marginal = new Deployment("search");
        marginal.run(metrics, stream("marginal"));
        check(marginal.result.equals("promoted") && marginal.last.marginal >= 1 && marginal.last.failedMetric < 0
                        && marginal.last.score <= 1 && marginal.last.norm[0] > 1 && marginal.last.norm[0] <= 2,
                "a marginal but acceptable candidate must still promote: " + Arrays.toString(marginal.last.norm));

        Deployment better = new Deployment("search");
        better.run(metrics, stream("better"));
        check(better.result.equals("promoted") && better.last.score < 0 && better.last.marginal == 0,
                "a candidate improving every metric must promote with a negative score: " + better.last.score);

        Deployment blocked = new Deployment("search");
        blocked.run(metrics, List.of());
        check(blocked.result.equals("blocked (no data)") && blocked.liveVersion == 12 && blocked.windowsUsed == 0,
                "an unmeasurable candidate must never be promoted: " + blocked.result);

        Deployment zero = new Deployment("search");
        zero.run(metrics, stream("zero-baseline"));
        Deployment zeroDrift = new Deployment("search");
        zeroDrift.run(metrics, stream("zero-baseline-drift"));
        check(zero.result.equals("promoted") && zeroDrift.result.equals("rolled back at 5%")
                        && Double.isFinite(zero.last.score) && Double.isFinite(zeroDrift.last.score),
                "a zero baseline must stay finite: " + zero.last.score + " / " + zeroDrift.last.score);

        Verdict online = judge(metrics, stream("healthy"), 24);
        Verdict oracle = reference(metrics, stream("healthy"), 24);
        Verdict onlineRegression = judge(metrics, stream("latency-regression"), 10);
        Verdict oracleRegression = reference(metrics, stream("latency-regression"), 10);
        check(sameVerdict(online, oracle) && sameVerdict(onlineRegression, oracleRegression),
                "the judge must agree with the reference implementation: " + online.score + " vs " + oracle.score);

        Deployment replay = new Deployment("checkout");
        replay.run(metrics, stream("latency-regression"));
        check(replay.log.equals(regressed.log) && replay.windowsUsed == regressed.windowsUsed
                        && replay.result.equals(regressed.result),
                "the rollout must be reproducible for the same metric stream");

        System.out.println("canary analysis: healthy promoted at stage " + healthy.stagesPassed
                + ", regression rolled back at " + regressed.rollbackShare + "% after " + regressed.windowsUsed
                + " of " + (4 * PER_STAGE) + " window slots");
        System.out.println("final scores: healthy " + String.format("%.3f", healthy.last.score) + ", regressed "
                + String.format("%.3f", regressed.last.score) + ", marginal "
                + String.format("%.3f", marginal.last.score) + ", better "
                + String.format("%.3f", better.last.score));
        System.out.println("live version after rollback v" + regressed.liveVersion + ", after promotion v"
                + healthy.liveVersion + ", blocked result \"" + blocked.result + "\"");
        System.out.println("PASS");
    }
}
