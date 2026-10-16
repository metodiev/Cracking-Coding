/** Task 691 - Implement additive-increase multiplicative-decrease (AIMD) simulation. (Tier: Expert, CTCI Beyond - Networking, OS & Real-Time)
 *
 * Several flows share one bottleneck of capacity C. Two update styles are simulated:
 *
 *   SYNCHRONOUS   every flow sees the same congestion event, so a round with total demand above
 *                 C multiplies *all* windows by beta (default 1/2); otherwise every flow adds
 *                 one segment. Neither operation changes the ordering of the windows and the
 *                 decrease shrinks every gap by beta, so the difference between any two flows
 *                 obeys the classic fairness result: gap <- beta * gap per loss event.
 *   ASYNCHRONOUS  each flow suffers its own loss with probability p per round (a fixed-seed
 *                 splitmix64 PRNG); a flow either halves or gains one segment. The window is
 *                 then a Markov chain with E[W] = E[W](1-p) + (1-p) + p E[W]/2 in steady state,
 *                 i.e. E[W] = 2(1-p)/p - an exact closed form the simulation is checked against.
 *
 * Verified: additive increase of exactly one segment per round, exact halving on a congestion
 * event, the geometric shrink of the inter-flow gap, convergence of Jain's fairness index to
 * one with the total demand parked just above the capacity, the 3/4 W_max average window of a
 * single flow (the standard AIMD throughput result), the E[W] = 2(1-p)/p law for two different
 * loss probabilities, and the instability of the beta = 1 variant that never backs off.
 */
import java.util.*;

/** One simulation run, recorded round by round. */
class AimdTrace {
    final double[][] windows;
    final double[] demand;
    final double[] served;
    final double[] jain;
    final boolean[] lossRound;

    AimdTrace(double[][] windows, double[] demand, double[] served, double[] jain, boolean[] lossRound) {
        this.windows = windows;
        this.demand = demand;
        this.served = served;
        this.jain = jain;
        this.lossRound = lossRound;
    }

    int rounds() {
        return demand.length;
    }

    double averageDemand(int from, int to) {
        double sum = 0;
        for (int i = from; i < to; i++) sum += demand[i];
        return sum / (to - from);
    }

    double averageJain(int from, int to) {
        double sum = 0;
        for (int i = from; i < to; i++) sum += jain[i];
        return sum / (to - from);
    }

    double utilization(int from, int to, int capacity) {
        double sum = 0;
        for (int i = from; i < to; i++) sum += served[i];
        return sum / ((to - from) * (double) capacity);
    }

    int lossEvents() {
        int count = 0;
        for (boolean lost : lossRound) if (lost) count++;
        return count;
    }
}

/** AIMD flow set over a single bottleneck. */
class AimdDomain {
    static final double BETA = 0.5;

    /** Synchronous AIMD: congestion is shared, so every flow backs off together. */
    static AimdTrace synchronous(double[] initial, int capacity, int rounds, double beta) {
        int flows = initial.length;
        double[] current = initial.clone();
        double[][] windows = new double[rounds][];
        double[] demand = new double[rounds];
        double[] served = new double[rounds];
        double[] jain = new double[rounds];
        boolean[] lossRound = new boolean[rounds];
        for (int round = 0; round < rounds; round++) {
            // Canonical AIMD per RTT: increase by one segment, then back off if congested.
            for (int i = 0; i < flows; i++) current[i] += 1;
            double total = 0;
            for (double value : current) total += value;
            demand[round] = total;
            served[round] = Math.min(total, capacity);
            jain[round] = Util.jain(current);
            lossRound[round] = total > capacity;
            if (lossRound[round]) {
                for (int i = 0; i < flows; i++) current[i] = Math.max(1, beta * current[i]);
            }
            windows[round] = current.clone();
        }
        return new AimdTrace(windows, demand, served, jain, lossRound);
    }

    /**
     * Asynchronous AIMD: every round each flow independently halves with probability p or gains
     * one segment. Returns the mean window per flow measured after `warmup` rounds.
     */
    static double[] asynchronous(int flows, double probability, int rounds, int warmup, long seed) {
        long state = seed;
        double[] current = new double[flows];
        Arrays.fill(current, 1);
        double[] totals = new double[flows];
        long samples = 0;
        for (int round = 0; round < rounds; round++) {
            for (int i = 0; i < flows; i++) {
                state += 0x9E3779B97F4A7C15L;
                long z = state;
                z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
                z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
                z ^= (z >>> 31);
                double uniform = (z >>> 11) * 0x1.0p-53;
                if (uniform < probability) {
                    current[i] = Math.max(1, current[i] * BETA);
                } else {
                    current[i] += 1;
                }
                if (round >= warmup) totals[i] += current[i];
            }
            if (round >= warmup) samples++;
        }
        for (int i = 0; i < flows; i++) totals[i] /= samples;
        return totals;
    }
}

/** Small numeric helpers shared by the checks. */
class Util {
    static double jain(double[] values) {
        double sum = 0;
        double squares = 0;
        for (double value : values) {
            sum += value;
            squares += value * value;
        }
        if (squares == 0) return 1;
        return sum * sum / (values.length * squares);
    }

    static double spread(double[] values) {
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        for (double value : values) {
            min = Math.min(min, value);
            max = Math.max(max, value);
        }
        return max - min;
    }

    static double mean(double[] values) {
        double sum = 0;
        for (double value : values) sum += value;
        return sum / values.length;
    }
}

public class Solution {
    /** Spread of the windows at each congestion event - the fairness signal. */
    static List<Double> spreadsAtLossEvents(AimdTrace trace) {
        List<Double> spreads = new ArrayList<>();
        for (int round = 0; round < trace.rounds(); round++) {
            if (trace.lossRound[round]) spreads.add(Util.spread(trace.windows[round]));
        }
        return spreads;
    }

    /** Coefficient of variation of the per-flow mean windows. */
    static double relativeDeviation(double[] values) {
        double mean = Util.mean(values);
        double squares = 0;
        for (double value : values) squares += (value - mean) * (value - mean);
        return Math.sqrt(squares / values.length) / mean;
    }

    public static void main(String[] args) {
        // 1. No congestion means exactly +1 per round for every flow.
        AimdTrace quiet = AimdDomain.synchronous(new double[]{1, 2, 3}, 1000, 10, 0.5);
        check(quiet.lossEvents() == 0, "a link wider than the demand has no congestion event");
        boolean additive = true;
        for (int round = 1; round < quiet.rounds(); round++) {
            for (int flow = 0; flow < 3; flow++) {
                if (quiet.windows[round][flow] - quiet.windows[round - 1][flow] != 1.0) additive = false;
            }
        }
        check(additive, "additive increase is exactly one segment per round");
        check(quiet.windows[0][0] == 2 && quiet.windows[9][2] == 13, "hand-checked window values");

        // 2. Congestion halves every window, exactly.
        AimdTrace halving = AimdDomain.synchronous(new double[]{59, 59}, 100, 1, 0.5);
        check(halving.lossRound[0] && halving.windows[0][0] == 30 && halving.windows[0][1] == 30,
                "one congested round halves both windows");

        // 3. Every congestion event shrinks the inter-flow gap by beta (AIMD fairness theorem).
        AimdTrace unfair = AimdDomain.synchronous(new double[]{1, 2, 4, 8}, 100, 4000, 0.5);
        List<Double> spreads = spreadsAtLossEvents(unfair);
        check(spreads.size() > 5, "several congestion events happened: " + spreads.size());
        boolean geometric = true;
        for (int i = 1; i < spreads.size() && i < 12; i++) {
            double expected = spreads.get(i - 1) * 0.5;
            if (Math.abs(spreads.get(i) - expected) > 1e-9) geometric = false;
        }
        check(geometric, "each congestion event halves the spread: " + spreads.subList(0, 6));
        check(spreads.get(Math.min(10, spreads.size() - 1)) < 0.1, "the flows end up nearly equal");

        // 4. Fairness on a longer horizon: Jain index near one, demand parked at the sawtooth
        //    average. Because every flow backs off together, the link is idle during the low
        //    part of each cycle - a queue in front of the bottleneck is what hides that cost.
        check(unfair.averageJain(2000, 4000) > 0.999,
                "Jain's index converges to one: " + unfair.averageJain(2000, 4000));
        check(unfair.utilization(2000, 4000, 100) > 0.70 && unfair.utilization(2000, 4000, 100) < 0.85,
                "synchronized AIMD uses about three quarters of the link: "
                        + unfair.utilization(2000, 4000, 100));
        check(unfair.averageDemand(2000, 4000) > 70 && unfair.averageDemand(2000, 4000) < 85,
                "the average total window sits between the trough and the peak: "
                        + unfair.averageDemand(2000, 4000));
        boolean boundedByCapacity = true;
        for (int round = 0; round < unfair.rounds(); round++) {
            if (unfair.served[round] > 100 + 1e-9) boundedByCapacity = false;
        }
        check(boundedByCapacity, "served traffic never exceeds the link capacity");
        check(unfair.averageJain(0, 200) < unfair.averageJain(2000, 4000),
                "fairness improves over the run");

        // 5. A single flow reproduces the classic 3/4 W_max average window.
        AimdTrace single = AimdDomain.synchronous(new double[]{1}, 100, 2000, 0.5);
        double peak = 0;
        double trough = Double.MAX_VALUE;
        for (int round = 1000; round < 2000; round++) {
            double window = single.windows[round][0];
            peak = Math.max(peak, window);
            if (single.lossRound[round]) trough = Math.min(trough, window);
        }
        check(peak > 98 && peak <= 101,
                "the peak window parks just below the capacity: " + peak);
        check(Math.abs(single.averageDemand(1000, 2000) / peak - 0.75) < 0.03,
                "the average window is 3/4 of the peak: " + single.averageDemand(1000, 2000) / peak);
        check(trough >= 49 && trough < 51, "the loss halves the window: " + trough);

        // 6. Without a decrease (beta = 1) the demand grows without bound: the negative case.
        AimdTrace unstable = AimdDomain.synchronous(new double[]{10}, 100, 400, 1.0);
        check(unstable.lossRound[90] && unstable.lossRound[399] && unstable.lossEvents() > 300,
                "the beta = 1 variant stays congested forever");
        check(unstable.demand[399] - unstable.demand[90] > 300,
                "demand grows linearly without the decrease: " + unstable.demand[399]);
        check(unstable.utilization(0, 400, 100) > 0.85,
                "the link is busy for almost the whole run: " + unstable.utilization(0, 400, 100));

        // 7. Asynchronous AIMD obeys the exact closed form E[W] = 2(1-p)/p.
        double[] lowLoss = AimdDomain.asynchronous(100, 0.01, 20000, 2000, 0x1234ABCDL);
        double[] highLoss = AimdDomain.asynchronous(100, 0.04, 20000, 2000, 0x1234ABCDL);
        double lowMean = Util.mean(lowLoss);
        double highMean = Util.mean(highLoss);
        check(Math.abs(lowMean - 2 * 0.99 / 0.01) / (2 * 0.99 / 0.01) < 0.10,
                "E[W] matches 2(1-p)/p for p = 0.01: " + lowMean);
        check(Math.abs(highMean - 2 * 0.96 / 0.04) / (2 * 0.96 / 0.04) < 0.10,
                "E[W] matches 2(1-p)/p for p = 0.04: " + highMean);
        check(Math.abs(lowMean / highMean - 4.0) < 0.4,
                "quadrupling the loss probability quarters the window: " + lowMean / highMean);
        check(relativeDeviation(lowLoss) < 0.10,
                "flows with identical loss rates stay balanced on average: " + relativeDeviation(lowLoss));
        check(lowMean > highMean, "heavier loss means a smaller window");
        double[] repeat = AimdDomain.asynchronous(100, 0.01, 20000, 2000, 0x1234ABCDL);
        check(Util.mean(repeat) == lowMean, "the PRNG makes the run reproducible");

        // 8. Boundary: a single flow under heavy loss stays at least one segment.
        double[] stressed = AimdDomain.asynchronous(1, 0.5, 5000, 0, 0x5EEDL);
        check(stressed[0] >= 1 && stressed[0] < 6,
                "with p = 0.5 the single flow hovers near two segments: " + stressed[0]);

        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
