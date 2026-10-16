/** Task 938 - Implement safe experiment platform that enforces SLOs and budget on experiments.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 *  A/B platform running several concurrent experiments. Each round the platform apportions the shared
 *  traffic pool by largest-remainder proportional allocation (never above an experiment's cap nor above
 *  the global exposure cap), charges the impressions against the experiment's budget, then applies two
 *  safety mechanisms: an error-rate SLO guardrail that auto-pauses a variant whose error rate exceeds the
 *  control by more than a tolerance, and a group-sequential test that can declare a winner early against
 *  a power-family alpha spending boundary. An experiment stops when it is paused, out of budget or
 *  declared significant, and a variant with no measurable data is never promoted.
 *  Simplified: one treatment arm per experiment, scripted binomial streams from a fixed seed, one global
 *  traffic pool, and unused (capped) traffic is not redistributed.
 *  Verified: allocation and budget invariants hold after every round, the harmful variant is paused early
 *  with bounded exposure, the null experiment is never a false positive, the Fisher exact p-value matches
 *  an independent product-form enumeration and a normal approximation, and alpha is never overspent.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }
    static final int TRAFFIC = 1000;          // shared traffic units per round
    static final int GLOBAL_CAP = 600;        // experiments never hold more than 60% of the traffic
    static final double ALPHA = 0.05;
    static final int LOOKS = 4;
    static final double SPEND_THETA = 1.5;    // power-family alpha spending
    static final int MIN_USERS = 80;          // the guardrail needs evidence before it acts
    static final double ERROR_TOLERANCE = 0.5;
    static final double CONTROL_ERROR_RATE = 0.02;

    static final double[] LOG_FACT = logFactorials(4000);

    static double[] logFactorials(int n) {
        double[] f = new double[n + 1];
        for (int i = 1; i <= n; i++) f[i] = f[i - 1] + Math.log(i);
        return f;
    }
    static double logChoose(int n, int k) {
        if (k < 0 || k > n) return Double.NEGATIVE_INFINITY;
        return LOG_FACT[n] - LOG_FACT[k] - LOG_FACT[n - k];
    }
    /** Fisher exact one-sided p-value: P(X >= a | fixed margins), summed in log space. */
    static double fisherP(int a, int n1, int b, int n2) {
        int successes = a + b;
        if (n1 + n2 == 0) return 1.0;
        double p = 0;
        for (int x = a; x <= Math.min(successes, n1); x++) {
            p += Math.exp(logChoose(n1, x) + logChoose(n2, successes - x) - logChoose(n1 + n2, successes));
        }
        return Math.min(1.0, Math.max(0.0, p));
    }
    /** Oracle: the same hypergeometric sum written with explicit products instead of log factorials. */
    static double fisherPReference(int a, int n1, int b, int n2) {
        int successes = a + b;
        if (n1 + n2 == 0) return 1.0;
        double denom = choose(n1 + n2, successes);
        double p = 0;
        for (int x = a; x <= Math.min(successes, n1); x++) p += choose(n1, x) * choose(n2, successes - x) / denom;
        return Math.min(1.0, p);
    }
    static double choose(int n, int k) {
        if (k < 0 || k > n) return 0;
        k = Math.min(k, n - k);
        double r = 1;
        for (int i = 1; i <= k; i++) r = r * (n - k + i) / i;
        return r;
    }
    /** Independent normal-approximation cross-check for large tables (A&S 26.2.17 CDF). */
    static double zTestP(int a, int n1, int b, int n2) {
        if (n1 == 0 || n2 == 0) return 1.0;
        double pooled = (double) (a + b) / (n1 + n2);
        double se = Math.sqrt(pooled * (1 - pooled) * (1.0 / n1 + 1.0 / n2));
        if (se == 0) return 1.0;
        return 1 - normalCdf(((double) a / n1 - (double) b / n2) / se);
    }
    static double normalCdf(double z) {
        double t = 1 / (1 + 0.2316419 * Math.abs(z));
        double poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        double p = 1 - Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI) * poly;
        return z >= 0 ? p : 1 - p;
    }
    /** Cumulative alpha spent after `look` of `looks` looks: alpha * (look/looks)^theta. */
    static double spentAlpha(int look, int looks) {
        return ALPHA * Math.pow((double) look / looks, SPEND_THETA);
    }
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
    static final class Experiment {
        final String name;
        final int request, cap, budget;
        final double controlRate, treatmentRate, treatmentErrorRate;
        final Rng rng;
        int granted, spent, controlUsers, controlConv, controlErrors;
        int treatmentUsers, treatmentConv, treatmentErrors;
        String state = "running", reason = "";
        int stoppedAtLook = -1, pausedAtLook = -1;
        double lastP = 1.0;

        Experiment(String name, int request, int cap, int budget, double controlRate, double treatmentRate,
                   double treatmentErrorRate, long seed) {
            this.name = name; this.request = request; this.cap = cap; this.budget = budget;
            this.controlRate = controlRate; this.treatmentRate = treatmentRate;
            this.treatmentErrorRate = treatmentErrorRate; this.rng = new Rng(seed);
        }
        boolean running() { return state.equals("running"); }
        int hits(int users, double rate) {
            int count = 0;
            for (int i = 0; i < users; i++) if (rng.next() < rate) count++;
            return count;
        }
    }
    /** Largest-remainder apportionment of the capped demand over the global exposure cap. */
    static int allocate(List<Experiment> exps) {
        double[] exact = new double[exps.size()];
        int demand = 0;
        for (Experiment e : exps) {
            e.granted = 0;
            if (e.running()) demand += Math.min(e.request, e.cap);
        }
        int target = Math.min(demand, GLOBAL_CAP);
        int assigned = 0;
        for (int i = 0; i < exps.size(); i++) {
            Experiment e = exps.get(i);
            if (!e.running()) continue;
            exact[i] = demand == 0 ? 0 : (double) target * Math.min(e.request, e.cap) / demand;
            e.granted = (int) Math.floor(exact[i]);
            assigned += e.granted;
        }
        while (assigned < target) {
            int best = -1;
            for (int i = 0; i < exps.size(); i++) {
                double fraction = exact[i] - Math.floor(exact[i]);
                if (!exps.get(i).running() || fraction <= 0) continue;
                if (best < 0 || fraction > exact[best] - Math.floor(exact[best])) best = i;
            }
            if (best < 0) break;
            exact[best] = Math.floor(exact[best]) + 1.0;
            exps.get(best).granted++;
            assigned++;
        }
        return assigned;
    }
    static void guardrails(List<Experiment> exps, int look) {
        for (Experiment e : exps) {
            if (!e.running() || e.treatmentUsers < MIN_USERS) continue;
            double controlError = (double) e.controlErrors / e.controlUsers;
            double treatmentError = (double) e.treatmentErrors / e.treatmentUsers;
            if (treatmentError > controlError * (1 + ERROR_TOLERANCE)) {
                e.state = "paused";
                e.reason = "slo-error-rate";
                e.pausedAtLook = look;
            }
        }
    }
    static void sequential(List<Experiment> exps, int look) {
        double boundary = spentAlpha(look, LOOKS) - spentAlpha(look - 1, LOOKS);
        for (Experiment e : exps) {
            if (!e.running()) continue;
            e.lastP = fisherP(e.treatmentConv, e.treatmentUsers, e.controlConv, e.controlUsers);
            if (e.lastP <= boundary) {
                e.state = "promoted";
                e.reason = "significant";
                e.stoppedAtLook = look;
            }
        }
    }
    static void runRound(List<Experiment> exps, int look) {
        allocate(exps);
        for (Experiment e : exps) {
            if (!e.running()) continue;
            int users = Math.min(e.granted, e.budget - e.spent);
            if (users > 0) {
                e.spent += users;
                e.controlUsers += users;
                e.treatmentUsers += users;
                e.controlConv += e.hits(users, e.controlRate);
                e.treatmentConv += e.hits(users, e.treatmentRate);
                e.controlErrors += e.hits(users, CONTROL_ERROR_RATE);
                e.treatmentErrors += e.hits(users, e.treatmentErrorRate);
            }
            if (e.spent >= e.budget) {
                e.state = "exhausted";
                e.reason = "budget";
            }
        }
        guardrails(exps, look);
        sequential(exps, look);
    }
    static List<Experiment> platform() {
        List<Experiment> exps = new ArrayList<>();
        exps.add(new Experiment("cta-banner", 300, 300, 4000, 0.05, 0.10, 0.02, 11));
        exps.add(new Experiment("cta-null", 200, 200, 4000, 0.05, 0.05, 0.02, 22));
        exps.add(new Experiment("risky-model", 200, 200, 4000, 0.05, 0.05, 0.14, 33));
        exps.add(new Experiment("tiny-budget", 300, 300, 50, 0.05, 0.06, 0.02, 44));
        return exps;
    }
    static Experiment byName(List<Experiment> exps, String name) {
        for (Experiment e : exps) if (e.name.equals(name)) return e;
        throw new IllegalStateException(name);
    }
    public static void main(String[] args) {
        List<Experiment> exps = platform();
        int allocationViolations = 0, budgetViolations = 0, handedOut = 0;
        for (int look = 1; look <= LOOKS; look++) {
            int granted = allocate(exps);
            int demand = 0;
            for (Experiment e : exps) {
                if (e.running()) demand += Math.min(e.request, e.cap);
                if (e.granted > e.cap) allocationViolations++;
                if (!e.running() && e.granted != 0) allocationViolations++;
            }
            if (granted != Math.min(demand, GLOBAL_CAP) || granted > GLOBAL_CAP) allocationViolations++;
            handedOut += granted;
            runRound(exps, look);
            for (Experiment e : exps) {
                if (e.spent > e.budget) budgetViolations++;
                if (e.state.equals("exhausted") && e.spent != e.budget) budgetViolations++;
            }
        }
        check(allocationViolations == 0,
                "allocation must respect every cap and apportion exactly: " + allocationViolations + " violations");
        check(budgetViolations == 0, "budgets must never be overspent: " + budgetViolations + " violations");

        Experiment risky = byName(exps, "risky-model");
        Experiment winner = byName(exps, "cta-banner");
        Experiment nullExp = byName(exps, "cta-null");
        Experiment tiny = byName(exps, "tiny-budget");

        check(risky.state.equals("paused") && risky.reason.equals("slo-error-rate") && risky.pausedAtLook == 1
                        && risky.treatmentUsers <= 200 && risky.spent < risky.budget && risky.lastP == 1.0,
                "the harmful variant must be paused by the SLO guardrail in the first look: " + risky.state + "/"
                        + risky.reason + " at look " + risky.pausedAtLook + " after " + risky.treatmentUsers + " users");
        check(nullExp.running() && nullExp.controlUsers == nullExp.treatmentUsers && nullExp.treatmentUsers > 300,
                "a benign variant must keep running with balanced arms: " + nullExp.state + " users "
                        + nullExp.treatmentUsers);
        check(winner.state.equals("promoted") && winner.stoppedAtLook >= 2 && winner.stoppedAtLook < LOOKS
                        && winner.lastP <= spentAlpha(winner.stoppedAtLook, LOOKS)
                                - spentAlpha(winner.stoppedAtLook - 1, LOOKS),
                "a genuinely better variant must win before the last look: look " + winner.stoppedAtLook + " p="
                        + winner.lastP);
        check(tiny.state.equals("exhausted") && tiny.spent == tiny.budget && tiny.budget == 50
                        && tiny.stoppedAtLook < 0,
                "an experiment must stop exactly at its budget: spent " + tiny.spent + "/" + tiny.budget);
        check(nullExp.stoppedAtLook < 0 && nullExp.lastP > spentAlpha(LOOKS, LOOKS) - spentAlpha(LOOKS - 1, LOOKS),
                "the null experiment must never be declared significant: last p=" + nullExp.lastP);

        int spendingViolations = 0;
        double previous = 0;
        for (int look = 1; look <= LOOKS; look++) {
            double cumulative = spentAlpha(look, LOOKS);
            if (cumulative < previous || cumulative > ALPHA + 1e-12) spendingViolations++;
            previous = cumulative;
        }
        check(spendingViolations == 0 && Math.abs(spentAlpha(LOOKS, LOOKS) - ALPHA) < 1e-12
                        && spentAlpha(1, LOOKS) < ALPHA / LOOKS,
                "alpha spending must stay inside the budget and be stricter early: look 1 boundary "
                        + spentAlpha(1, LOOKS) + " < uniform " + (ALPHA / LOOKS));

        check(fisherP(9, 100, 2, 100) < fisherP(7, 100, 2, 100) && fisherP(2, 100, 2, 100) > 0.5
                        && fisherP(100, 100, 0, 100) < 1e-6 && fisherP(0, 0, 0, 0) == 1.0
                        && fisherP(0, 50, 0, 50) == 1.0,
                "the exact test must be monotone in the evidence and defined on degenerate tables");

        double worstOracle = 0;
        int[][] tables = {{9, 100, 2, 100}, {18, 180, 9, 180}, {5, 40, 1, 40}, {0, 12, 0, 12}, {40, 40, 1, 40}};
        for (int[] t : tables) {
            worstOracle = Math.max(worstOracle, Math.abs(fisherP(t[0], t[1], t[2], t[3])
                    - fisherPReference(t[0], t[1], t[2], t[3])));
        }
        double normalGap = Math.abs(fisherP(180, 720, 90, 720) - zTestP(180, 720, 90, 720)) / zTestP(180, 720, 90, 720);
        check(worstOracle < 1e-9,
                "the log-space exact test must match the product-form enumeration, worst gap " + worstOracle);
        check(normalGap < 0.25,
                "the exact test must agree with the normal approximation on a large table, relative gap " + normalGap);

        List<Experiment> replay = platform();
        for (int look = 1; look <= LOOKS; look++) runRound(replay, look);
        boolean same = true;
        for (int i = 0; i < exps.size(); i++) {
            Experiment a = exps.get(i), b = replay.get(i);
            same &= a.state.equals(b.state) && a.spent == b.spent && a.treatmentConv == b.treatmentConv
                    && a.controlConv == b.controlConv && a.stoppedAtLook == b.stoppedAtLook;
        }
        check(same, "the platform must be reproducible for the same seeded streams");

        System.out.println("experiments " + exps.size() + ", traffic pool " + TRAFFIC + " units, global cap "
                + GLOBAL_CAP + ", looks " + LOOKS + ", alpha " + ALPHA);
        for (Experiment e : exps) {
            System.out.println("  " + e.name + ": " + e.state + "/" + e.reason + " spent " + e.spent + "/" + e.budget
                    + ", users " + e.treatmentUsers + ", conversions " + e.treatmentConv + "/" + e.controlConv
                    + ", p=" + String.format("%.4f", e.lastP));
        }
        System.out.println("traffic units handed out: " + handedOut + ", oracle gap " + worstOracle
                + ", normal gap " + String.format("%.3f", normalGap));
        System.out.println("PASS");
    }
}
