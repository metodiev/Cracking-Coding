/** Task 940 - Implement high-fidelity load generator with correlated user behavior models.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 *  An open-loop workload model: session start times form a Poisson process (exponential inter-arrivals at
 *  the configured rate) and each session is a walk through two user classes' own 7-state Markov chain
 *  over landing/browse/search/product/cart/checkout/exit. Every non-exit state visit issues one request,
 *  service times and think times are exponential, and because the class drives both the chain and the
 *  think-time mean, session depth and think time come out correlated - the property that makes a load
 *  test realistic.
 *  Simplified: two classes, exponential think times instead of a fitted lognormal, no client-side
 *  concurrency limit, and sessions do not feed back into the arrival process (open loop).
 *  Verified: only allowed transitions occur and every session ends at exit, the empirical transition
 *  matrix matches the configured mixture and the analytic expected visit counts, the inter-arrival mean
 *  matches 1/rate, per-class think times match the configuration, the depth/think-time correlation is
 *  positive and agrees with an independent online estimator, and the stream is reproducible with a
 *  prefix property.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }
    static final int STATES = 7;      // landing browse search product cart checkout exit
    static final int LANDING = 0;
    static final int EXIT = 6;

    /** Deterministic splitmix64 - the only source of randomness in the generator. */
    static final class Rng {
        long state;
        Rng(long seed) { state = seed; }
        double next() {
            long z = (state += 0x9E3779B97F4A7C15L);
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return ((z ^ (z >>> 31)) >>> 11) * 0x1.0p-53;
        }
        double exponential(double mean) { return -Math.log(1 - next()) * mean; }
    }
    static double[][] parse(String[] rows) {
        double[][] p = new double[rows.length][STATES];
        for (int i = 0; i < rows.length; i++) {
            String[] cells = rows[i].trim().split("\\s+");
            for (int j = 0; j < STATES; j++) p[i][j] = Double.parseDouble(cells[j]);
        }
        return p;
    }
    static final double[][] BROWSER = parse(new String[] {
        "0.00 0.40 0.10 0.00 0.00 0.00 0.50",      // landing: shallow and impatient
        "0.00 0.20 0.20 0.10 0.00 0.00 0.50",
        "0.00 0.15 0.20 0.15 0.00 0.00 0.50",
        "0.00 0.20 0.05 0.05 0.20 0.00 0.50",
        "0.00 0.10 0.00 0.10 0.20 0.15 0.45",
        "0.00 0.00 0.00 0.05 0.15 0.20 0.60",
        "0.00 0.00 0.00 0.00 0.00 0.00 1.00",      // exit is absorbing
    });

    static final double[][] BUYER = parse(new String[] {
        "0.00 0.40 0.20 0.20 0.00 0.00 0.20",      // deep sessions, hardly ever abandons
        "0.00 0.20 0.25 0.35 0.00 0.00 0.20",
        "0.00 0.10 0.15 0.50 0.00 0.00 0.25",
        "0.00 0.05 0.05 0.25 0.60 0.00 0.05",
        "0.00 0.02 0.00 0.03 0.35 0.55 0.05",
        "0.00 0.00 0.00 0.00 0.02 0.30 0.68",
        "0.00 0.00 0.00 0.00 0.00 0.00 1.00",
    });

    static final class Model {
        final double[][] perClass;                 // [class * STATES + row][column]
        final double[] weight, thinkMean;

        Model(double[][] perClass, double[] weight, double[] thinkMean) {
            this.perClass = perClass; this.weight = weight; this.thinkMean = thinkMean;
        }
        double[][] classMatrix(int cls) {
            double[][] p = new double[STATES][STATES];
            for (int i = 0; i < STATES; i++) System.arraycopy(perClass[cls * STATES + i], 0, p[i], 0, STATES);
            return p;
        }
    }
    /**
     * Analytic reference for the pooled transition matrix. A pooled session is not a 50/50 mixture of the
     * classes at every state - deep states are enriched in buyers - so each class is weighted by its own
     * expected visit mass per state instead of by the prior.
     */
    static double[][] pooledTransitions(Model model) {
        double[] mass = new double[STATES];
        double[][] p = new double[STATES][STATES];
        for (int c = 0; c < 2; c++) {
            double[] visits = expectedVisits(model.classMatrix(c));
            for (int i = 0; i < STATES; i++) {
                mass[i] += model.weight[c] * visits[i];
                for (int j = 0; j < STATES; j++) {
                    p[i][j] += model.weight[c] * visits[i] * model.perClass[c * STATES + i][j];
                }
            }
        }
        for (int i = 0; i < STATES; i++) {
            for (int j = 0; j < STATES; j++) p[i][j] = mass[i] == 0 ? 0 : p[i][j] / mass[i];
        }
        return p;
    }
    static Model fixtureModel() {
        double[][] flat = new double[2 * STATES][];
        for (int i = 0; i < STATES; i++) flat[i] = BROWSER[i];
        for (int i = 0; i < STATES; i++) flat[STATES + i] = BUYER[i];
        return new Model(flat, new double[] {0.5, 0.5}, new double[] {2.0, 16.0});
    }
    /** A one-shot model where landing goes straight to exit, used as the degenerate boundary. */
    static Model oneShotModel() {
        double[][] flat = parse(new String[] {
            "0.00 0.00 0.00 0.00 0.00 0.00 1.00", "0.00 0.00 0.00 0.00 0.00 0.00 1.00",
            "0.00 0.00 0.00 0.00 0.00 0.00 1.00", "0.00 0.00 0.00 0.00 0.00 0.00 1.00",
            "0.00 0.00 0.00 0.00 0.00 0.00 1.00", "0.00 0.00 0.00 0.00 0.00 0.00 1.00",
            "0.00 0.00 0.00 0.00 0.00 0.00 1.00"});
        double[][] flat2 = new double[2 * STATES][];
        for (int i = 0; i < 2 * STATES; i++) flat2[i] = flat[i % STATES];
        return new Model(flat2, new double[] {0.5, 0.5}, new double[] {1.0, 1.0});
    }
    static final class Workload {
        final int sessions;
        final Map<Integer, Integer> transitions = new TreeMap<>();
        final double[] visits = new double[STATES];
        final double[] arrivals, sessionLength, sessionThink, thinkSum;
        final double[] thinkByClass = new double[2];
        final int[] sessionsByClass = new int[2], visitsByClass = new int[2];
        int requests, maxDepth, violations;
        long hash = 0xcbf29ce484222325L;

        Workload(int sessions) {
            this.sessions = sessions;
            arrivals = new double[sessions];
            sessionLength = new double[sessions];
            sessionThink = new double[sessions];
            thinkSum = new double[sessions];
        }
        int count(int from, int to) { return transitions.getOrDefault(from * STATES + to, 0); }
    }
    static Workload generate(long seed, int sessions, double rate, Model model) {
        Rng rng = new Rng(seed);
        Workload w = new Workload(sessions);
        double arrival = 0;
        for (int s = 0; s < sessions; s++) {
            arrival += rng.exponential(1.0 / rate);                  // Poisson session starts (open loop)
            w.arrivals[s] = arrival;
            int cls = rng.next() < model.weight[0] ? 0 : 1;
            double sessionClock = arrival, think = 0;
            int depth = 0, state = LANDING;
            while (state != EXIT) {
                w.visits[state]++;
                depth++;
                sessionClock += rng.exponential(0.05);               // server time for this request
                double u = rng.next(), cumulative = 0;
                int next = EXIT;
                for (int j = 0; j < STATES; j++) {
                    cumulative += model.perClass[cls * STATES + state][j];
                    if (u < cumulative) { next = j; break; }
                }
                w.transitions.merge(state * STATES + next, 1, Integer::sum);
                think += rng.exponential(model.thinkMean[cls]);
                state = next;
                if (depth > 500) {                                   // guard: a non-absorbing row
                    w.violations++;
                    break;
                }
            }
            if (state != EXIT) w.violations++;
            w.visits[EXIT]++;
            w.sessionLength[s] = depth;
            w.thinkSum[s] = think;
            w.sessionThink[s] = think / depth;
            w.thinkByClass[cls] += think;
            w.sessionsByClass[cls]++;
            w.visitsByClass[cls] += depth;
            w.requests += depth;
            w.maxDepth = Math.max(w.maxDepth, depth);
            w.hash = w.hash * 1000003L + (cls * 31L + depth) * 131 + Double.doubleToLongBits(think);
        }
        return w;
    }
    /** Analytic mean visits per state per session - an oracle that never runs the simulator. */
    static double[] expectedVisits(double[][] p) {
        double[] visits = new double[STATES];
        double[] mass = new double[STATES];
        mass[LANDING] = 1.0;
        for (int step = 0; step < 400; step++) {
            for (int i = 0; i < EXIT; i++) visits[i] += mass[i];      // exit is terminal, counted once below
            double[] next = new double[STATES];
            for (int i = 0; i < STATES; i++) for (int j = 0; j < STATES; j++) next[j] += mass[i] * p[i][j];
            mass = next;
        }
        visits[EXIT] = 1.0;                                           // every session ends exactly once
        return visits;
    }
    /** Two-pass Pearson correlation - the definition. */
    static double pearson(double[] x, double[] y) {
        int n = x.length;
        if (n < 2) return 0;
        double mx = 0, my = 0;
        for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
        mx /= n;
        my /= n;
        double cov = 0, vx = 0, vy = 0;
        for (int i = 0; i < n; i++) {
            double dx = x[i] - mx, dy = y[i] - my;
            cov += dx * dy;
            vx += dx * dx;
            vy += dy * dy;
        }
        return vx == 0 || vy == 0 ? 0 : cov / Math.sqrt(vx * vy);
    }
    /** Independent online (Welford-style) estimator of the same coefficient. */
    static double pearsonOnline(double[] x, double[] y) {
        int n = 0;
        double mx = 0, my = 0, cxy = 0, vx = 0, vy = 0;
        for (int i = 0; i < x.length; i++) {
            n++;
            double dx = x[i] - mx, dy = y[i] - my;
            mx += dx / n;
            my += dy / n;
            cxy += dx * (y[i] - my);
            vx += dx * (x[i] - mx);
            vy += dy * (y[i] - my);
        }
        return n < 2 || vx == 0 || vy == 0 ? 0 : cxy / Math.sqrt(vx * vy);
    }
    public static void main(String[] args) {
        Model model = fixtureModel();
        double rate = 5.0;
        int sessions = 4000;
        Workload w = generate(2024, sessions, rate, model);
        Workload replay = generate(2024, sessions, rate, model);
        Workload other = generate(2025, sessions, rate, model);

        check(w.violations == 0 && w.maxDepth < 200 && w.sessionsByClass[0] + w.sessionsByClass[1] == sessions,
                "every session must start at landing, follow allowed transitions and end at exit: "
                        + w.violations + " violations, max depth " + w.maxDepth);
        check(w.hash == replay.hash && replay.requests == w.requests && other.hash != w.hash
                        && replay.sessionLength[sessions - 1] == w.sessionLength[sessions - 1],
                "the stream must be reproducible for a seed and different across seeds");

        int rowViolations = 0;
        double worstDeviation = 0;
        double[][] pooled = pooledTransitions(model);
        for (int i = 0; i < STATES; i++) {
            int rowTotal = 0;
            for (int j = 0; j < STATES; j++) rowTotal += w.count(i, j);
            for (int j = 0; j < STATES; j++) {
                if (pooled[i][j] == 0 && w.count(i, j) > 0) rowViolations++;
                if (rowTotal == 0) continue;
                worstDeviation = Math.max(worstDeviation,
                        Math.abs((double) w.count(i, j) / rowTotal - pooled[i][j]));
            }
        }
        check(rowViolations == 0 && worstDeviation < 0.06,
                "the empirical transitions must match the class-weighted mixture, worst deviation " + worstDeviation);

        double[] expected = new double[STATES];
        for (int c = 0; c < 2; c++) {
            double[] visits = expectedVisits(model.classMatrix(c));
            for (int i = 0; i < STATES; i++) expected[i] += model.weight[c] * visits[i];
        }
        double worstVisits = 0;
        for (int i = 0; i < STATES; i++) {
            worstVisits = Math.max(worstVisits, Math.abs(w.visits[i] / sessions - expected[i]) / expected[i]);
        }
        check(worstVisits < 0.10 && w.visits[EXIT] == sessions && w.requests > sessions,
                "the realised visit counts must match the analytic expectation, worst relative gap " + worstVisits);

        double meanGap = 0;
        for (int i = 1; i < sessions; i++) meanGap += w.arrivals[i] - w.arrivals[i - 1];
        meanGap /= (sessions - 1);
        check(Math.abs(meanGap * rate - 1) < 0.05 && w.arrivals[0] > 0 && w.arrivals[sessions - 1] > w.arrivals[0],
                "the inter-arrival mean must match the configured rate: " + meanGap + " s vs " + (1.0 / rate));

        double browserMean = w.thinkByClass[0] / w.visitsByClass[0];      // mean think time per state visit
        double buyerMean = w.thinkByClass[1] / w.visitsByClass[1];
        check(Math.abs(browserMean / 2.0 - 1) < 0.1 && Math.abs(buyerMean / 16.0 - 1) < 0.1
                        && buyerMean > 4 * browserMean,
                "per-class think times must match the configuration: " + browserMean + " / " + buyerMean);

        double rho = pearson(w.sessionLength, w.sessionThink);
        double rhoOnline = pearsonOnline(w.sessionLength, w.sessionThink);
        check(rho > 0.3 && Math.abs(rho - rhoOnline) < 1e-9,
                "session depth and think time must be correlated and both estimators must agree: " + rho + " vs "
                        + rhoOnline);

        int rowSumViolations = 0;
        for (int c = 0; c < 2; c++) {
            double[][] p = model.classMatrix(c);
            for (int i = 0; i < STATES; i++) {
                double sum = 0;
                for (int j = 0; j < STATES; j++) sum += p[i][j];
                if (Math.abs(sum - 1.0) > 1e-12 || p[EXIT][i] != (i == EXIT ? 1.0 : 0.0)) rowSumViolations++;
            }
        }
        check(rowSumViolations == 0 && Math.abs(model.weight[0] + model.weight[1] - 1.0) < 1e-12,
                rowSumViolations + " configured rows are not stochastic");

        Workload shortRun = generate(2024, 500, rate, model);
        boolean prefix = true;
        for (int s = 0; s < 500 && prefix; s++) {
            prefix = shortRun.sessionLength[s] == w.sessionLength[s] && shortRun.thinkSum[s] == w.thinkSum[s];
        }
        check(prefix && shortRun.requests < w.requests,
                "a shorter run must reproduce the prefix of a longer run with the same seed");

        Workload none = generate(7, 0, rate, model);
        Workload single = generate(7, 1, rate, model);
        check(none.requests == 0 && none.arrivals.length == 0 && pearson(new double[0], new double[0]) == 0
                        && single.requests >= 1 && single.visits[EXIT] == 1
                        && pearson(single.sessionLength, single.sessionThink) == 0,
                "degenerate session counts must yield defined statistics: requests " + none.requests + "/"
                        + single.requests);

        Workload bouncing = generate(99, 200, rate, oneShotModel());
        check(bouncing.requests == 200 && bouncing.visits[EXIT] == 200 && bouncing.maxDepth == 1
                        && bouncing.violations == 0,
                "a one-shot model must emit exactly one request per session: " + bouncing.requests);

        System.out.println("workload: " + sessions + " sessions, " + w.requests + " requests, max depth "
                + w.maxDepth + ", classes " + w.sessionsByClass[0] + "/" + w.sessionsByClass[1] + " (rho "
                + String.format("%.3f", rho) + ")");
        System.out.println("transitions worst deviation " + String.format("%.4f", worstDeviation)
                + ", visits worst gap " + String.format("%.4f", worstVisits) + ", mean inter-arrival "
                + String.format("%.4f", meanGap) + " s at " + rate + " rps, think means "
                + String.format("%.2f", browserMean) + "/" + String.format("%.2f", buyerMean) + " s");
        System.out.println("PASS");
    }
}
