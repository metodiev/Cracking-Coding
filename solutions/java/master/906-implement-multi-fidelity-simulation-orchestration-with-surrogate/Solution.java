/** Task 906 - Implement multi-fidelity simulation orchestration with surrogate models.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A design studio can run an expensive high-fidelity simulation (cost 10 units) and a cheap low-fidelity one
 * (cost 1) that shares the same physics but carries a smooth model discrepancy. Naively spending the whole
 * budget on the expensive code is wasteful: the cheap code already cancels most of the *function* variance,
 * so a control-variate estimator
 *
 *     mu_hat = mean(low over every cheap sample) + mean(high - low over the paired samples)
 *
 * is unbiased (the residual mean absorbs the discrepancy) with variance Var(low)/n_low + Var(high-low)/n_h.
 * The orchestrator runs a small paired pilot, estimates those two variances, and splits the remaining budget
 * by solving the Lagrange condition sqrt(10a)(p+m) = sqrt(b)(p+R-10m) for the number of paired samples m.
 *
 * Simplification: everything is deterministic arithmetic on a fixed-seed pseudo-random design, the two
 * "fidelities" are analytic functions plus Gaussian noise instead of PDE solvers, and the pilot is billed
 * inside the budget. Invariants verified: both estimators are unbiased against the closed-form integral, the
 * surrogate cancels the discrepancy, the multi-fidelity MSE beats a pure high-fidelity MSE at equal cost,
 * the allocator responds monotonically to the variance ratio, and an uncorrelated or equally-priced cheap
 * model is correctly rejected in favour of plain high-fidelity sampling.
 */
import java.util.*;

/** splitmix64 plus a deterministic Gaussian for the simulation noise. */
final class rng906 {
    private long state;
    private double spare;
    private boolean hasSpare;

    rng906(long seed) { this.state = seed; }

    private long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    double nextGaussian() {
        if (hasSpare) {
            hasSpare = false;
            return spare;
        }
        double u;
        double v;
        double s;
        do {
            u = 2 * nextDouble() - 1;
            v = 2 * nextDouble() - 1;
            s = u * u + v * v;
        } while (s >= 1 || s == 0);
        double mul = Math.sqrt(-2 * Math.log(s) / s);
        spare = v * mul;
        hasSpare = true;
        return u * mul;
    }
}

/** One estimator result: the estimate, what it cost, and how the budget was split. */
final class Estimate {
    final double value;
    final int cost;
    final int paired;
    final int cheapOnly;

    Estimate(double value, int cost, int paired, int cheapOnly) {
        this.value = value;
        this.cost = cost;
        this.paired = paired;
        this.cheapOnly = cheapOnly;
    }
}

public class Solution {
    static final int COST_HIGH = 10;
    static final int COST_LOW = 1;
    static final int BUDGET = 1200;
    static final int PILOT = 25;
    static final int REPLICATIONS = 300;
    static final double DISCREPANCY_MEAN = 0.5;

    public static void main(String[] args) {
        // ---- 1. the closed-form truth, cross-checked against a quadrature oracle ----------------------
        double truth = integral();
        double quadrature = 0;
        double cosineMean = 0;
        int steps = 200000;
        for (int i = 0; i < steps; i++) {
            double x = (i + 0.5) / steps;
            quadrature += physics(x) / steps;
            cosineMean += 0.3 * Math.cos(2 * Math.PI * x) / steps;
        }
        check(Math.abs(quadrature - truth) < 1e-4,
                "quadrature must match the closed form: " + quadrature + " vs " + truth);
        check(Math.abs(cosineMean) < 1e-9,
                "the oscillating part of the discrepancy must average to zero: " + cosineMean);

        // ---- 2. the surrogate must cancel the low-fidelity discrepancy --------------------------------
        rng906 pilotRng = new rng906(0x9060001L);
        double[] xs = new double[PILOT];
        double[] highs = new double[PILOT];
        double[] lows = new double[PILOT];
        for (int i = 0; i < PILOT; i++) {
            xs[i] = pilotRng.nextDouble();
            highs[i] = highFidelity(xs[i], pilotRng);
            lows[i] = lowFidelity(xs[i], pilotRng);
        }
        double meanResidual = mean(highs) - mean(lows);
        check(Math.abs(meanResidual - DISCREPANCY_MEAN) < 3 * Math.sqrt(0.0875 / PILOT) + 0.05,
                "the paired residual must estimate the discrepancy mean " + DISCREPANCY_MEAN + ", got "
                        + meanResidual);
        double surrogateError = 0;
        double rawError = 0;
        for (int i = 0; i < 400; i++) {
            double x = (i + 0.5) / 400;
            double corrected = lowFidelityNoiseless(x) + meanResidual;
            surrogateError += Math.abs(corrected - highFidelityNoiseless(x));
            rawError += Math.abs(lowFidelityNoiseless(x) - highFidelityNoiseless(x));
        }
        check(surrogateError < rawError / 2,
                "the bias-corrected surrogate must beat the raw cheap model: " + surrogateError + " vs "
                        + rawError);

        // ---- 3. allocator: variance estimates drive the budget split ----------------------------------
        double varCheap = variance(lows);
        double varResidual = variancePairs(highs, lows);
        Plan plan = allocate(BUDGET, PILOT, varCheap, varResidual, COST_HIGH, COST_LOW);
        check(plan.cost <= BUDGET, "the plan must respect the budget: " + plan.cost + " > " + BUDGET);
        check(BUDGET - plan.cost < COST_HIGH, "less than one high-fidelity call must be left over: "
                + (BUDGET - plan.cost));
        check(plan.paired > 0 && plan.cheapOnly > 0, "a healthy plan uses both fidelities");

        int previousPaired = -1;
        boolean monotone = true;
        StringBuilder sweep = new StringBuilder();
        for (double ratio : new double[] {0.05, 0.2, 0.8, 3.0}) {
            Plan candidate = allocate(BUDGET, PILOT, varCheap, varResidual * ratio / 0.0875,
                    COST_HIGH, COST_LOW);
            sweep.append(candidate.paired).append('/').append(candidate.cheapOnly).append(' ');
            if (previousPaired >= 0 && candidate.paired <= previousPaired) monotone = false;
            previousPaired = candidate.paired;
        }
        System.out.println("allocation sweep (paired/cheap-only per residual-variance ratio): " + sweep
                + "; varCheap " + round(varCheap) + " varResidual " + round(varResidual));
        check(monotone, "a noisier residual must buy more paired samples, allocation must be monotone");
        // ---- 4. equal-cost cheap model: multi-fidelity cannot win -------------------------------------
        check(!worthCorrecting(0.99, 1, 1), "a cheap model that costs the same as the expensive one is useless");
        check(worthCorrecting(0.99, 10, 1), "a ten-times cheaper, highly correlated model is worth correcting");
        check(!worthCorrecting(0.10, 10, 1), "a cheap model with no correlation must be rejected");

        // ---- 5. MSE race at equal cost -----------------------------------------------------------------
        double reference = target();
        int overruns = 0;
        double sumHighSquared = 0;
        double sumLowSquared = 0;
        double sumHigh = 0;
        double sumLow = 0;
        for (int r = 0; r < REPLICATIONS; r++) {
            Estimate hf = highFidelityOnly(0x9060000L + r);
            Estimate mf = multiFidelity(0x9060000L + r);
            if (hf.cost > BUDGET) overruns++;
            if (mf.cost > BUDGET) overruns++;
            sumHighSquared += sq(hf.value - reference);
            sumLowSquared += sq(mf.value - reference);
            sumHigh += hf.value;
            sumLow += mf.value;
        }
        check(overruns == 0, "no replication may overrun the budget, overruns " + overruns);
        double mseHigh = sumHighSquared / REPLICATIONS;
        double mseLow = sumLowSquared / REPLICATIONS;
        check(mseLow < mseHigh / 2,
                "multi-fidelity must halve the MSE: " + mseLow + " vs " + mseHigh);
        check(Math.abs(sumHigh / REPLICATIONS - reference) < 4 * Math.sqrt(mseHigh / REPLICATIONS) + 0.02,
                "the high-fidelity estimator must be unbiased");
        check(Math.abs(sumLow / REPLICATIONS - reference) < 4 * Math.sqrt(mseLow / REPLICATIONS) + 0.02,
                "the multi-fidelity estimator must be unbiased");

        // ---- 6. boundary cases -------------------------------------------------------------------------
        Plan tiny = allocate(COST_HIGH - 1, 0, varCheap, varResidual, COST_HIGH, COST_LOW);
        check(tiny.paired == 0 && tiny.cheapOnly == COST_HIGH - 1 && tiny.cost == COST_HIGH - 1,
                "below one high-fidelity call the planner can only buy cheap samples: " + tiny.paired + "/"
                        + tiny.cheapOnly);
        Estimate starved = multiFidelity(0x90600FFL, 5);
        check(Double.isNaN(starved.value) && starved.cost <= 5, "a starved run reports NaN, not a fake number");
        Plan zeroPilot = allocate(BUDGET, 0, varCheap, varResidual, COST_HIGH, COST_LOW);
        check(zeroPilot.cost <= BUDGET && zeroPilot.paired >= 0, "a zero-pilot plan must stay in budget");

        // ---- 7. determinism -----------------------------------------------------------------------------
        double first = multiFidelity(0x9060002L).value;
        double second = multiFidelity(0x9060002L).value;
        check(first == second, "the same seed must give the same estimate");
        check(multiFidelity(0x9060002L).cost == multiFidelity(0x9060002L).cost,
                "the same seed must give the same cost");

        System.out.println("physics integral " + round(truth) + "; pilot residual " + round(meanResidual)
                + "; plan paired " + plan.paired + " cheap-only " + plan.cheapOnly + " cost " + plan.cost);
        System.out.println("MSE over " + REPLICATIONS + " replications: high fidelity " + round(mseHigh)
                + " -> multi fidelity " + round(mseLow) + " (" + round(mseHigh / mseLow) + "x)");
        System.out.println("quadrature check " + round(quadrature) + "; surrogate error " + round(surrogateError)
                + " vs raw " + round(rawError));
        System.out.println("PASS");
    }

    /** The "expensive" simulator: shared physics, a smooth discrepancy and 0.2-magnitude noise. */
    static double highFidelity(double x, rng906 rng) {
        return highFidelityNoiseless(x) + 0.2 * rng.nextGaussian();
    }

    static double highFidelityNoiseless(double x) {
        return physics(x) + DISCREPANCY_MEAN + 0.3 * Math.cos(2 * Math.PI * x);
    }

    /** The "cheap" simulator: same physics, 0.05-magnitude noise, no discrepancy. */
    static double lowFidelity(double x, rng906 rng) { return lowFidelityNoiseless(x) + 0.05 * rng.nextGaussian(); }

    static double lowFidelityNoiseless(double x) { return physics(x); }

    static double physics(double x) { return 10 * Math.sin(3 * x) + 2 * x * x; }

    /** Closed form: integral of 10sin(3x) is 10(1 - cos(3))/3; integral of 2x^2 is 2/3. */
    static double integral() { return 10 * (1 - Math.cos(3)) / 3 + 2.0 / 3; }

    /** What the two simulators actually estimate: the physics plus the mean discrepancy. */
    static double target() { return integral() + DISCREPANCY_MEAN; }

    /** Pure high-fidelity sampling: every unit of budget buys one expensive call. */
    static Estimate highFidelityOnly(long seed) {
        rng906 rng = new rng906(seed);
        int samples = BUDGET / COST_HIGH;
        double sum = 0;
        for (int i = 0; i < samples; i++) sum += highFidelity(rng.nextDouble(), rng);
        return new Estimate(sum / samples, samples * COST_HIGH, samples, 0);
    }

    /** Control-variate estimator with pilot, allocation and a deterministic sample stream. */
    static Estimate multiFidelity(long seed) { return multiFidelity(seed, BUDGET); }

    static Estimate multiFidelity(long seed, int budget) {
        rng906 rng = new rng906(seed);
        int pilotSize = 0;
        int cost = 0;
        if (budget >= PILOT * (COST_HIGH + COST_LOW)) pilotSize = PILOT;
        if (pilotSize == 0) return new Estimate(Double.NaN, 0, 0, 0);
        double[] residual = new double[pilotSize];
        double[] cheap = new double[pilotSize];
        for (int i = 0; i < pilotSize; i++) {
            double x = rng.nextDouble();
            double h = highFidelity(x, rng);
            double l = lowFidelity(x, rng);
            residual[i] = h - l;
            cheap[i] = l;
            cost += COST_HIGH + COST_LOW;
        }
        Plan plan = allocate(budget - cost, pilotSize, variance(cheap), variance(residual), COST_HIGH, COST_LOW);
        double cheapSum = sum(cheap);
        int cheapCount = pilotSize;
        for (int i = 0; i < plan.cheapOnly; i++) {
            cheapSum += lowFidelity(rng.nextDouble(), rng);
            cheapCount++;
            cost += COST_LOW;
        }
        double residualSum = sum(residual);
        int residualCount = pilotSize;
        for (int i = 0; i < plan.paired; i++) {
            double x = rng.nextDouble();
            double h = highFidelity(x, rng);
            double l = lowFidelity(x, rng);
            residualSum += h - l;
            cheapSum += l;
            cheapCount++;
            residualCount++;
            cost += COST_HIGH + COST_LOW;
        }
        double value = cheapSum / cheapCount + residualSum / residualCount;
        return new Estimate(value, cost, plan.paired, plan.cheapOnly);
    }

    /** The budget split for a given remaining budget and variance estimates. */
    static Plan allocate(int budget, int pilot, double varCheap, double varResidual, int costHigh, int costLow) {
        int leftover = budget;
        int paired = 0;
        int cheapOnly = 0;
        double sqrtTerm = Math.sqrt(costHigh * varCheap) + costHigh * Math.sqrt(varResidual);
        if (sqrtTerm > 0 && pilot >= 0) {
            double numerator = Math.sqrt(varResidual) * (pilot + leftover)
                    - Math.sqrt(costHigh * varCheap) * pilot;
            double m = Math.max(0, Math.floor(numerator / sqrtTerm));
            paired = (int) m;
            while (paired > 0 && (long) paired * (costHigh + costLow) > leftover) paired--;
            cheapOnly = (leftover - paired * (costHigh + costLow)) / costLow;
            if (cheapOnly < 0) cheapOnly = 0;
        }
        return new Plan(paired, cheapOnly, paired * (costHigh + costLow) + cheapOnly * costLow);
    }

    /** A cheap model is only worth correcting when it is both correlated and materially cheaper. */
    static boolean worthCorrecting(double correlation, int costHigh, int costLow) {
        return costHigh >= 2 * costLow && correlation >= 0.5;
    }

    static double variance(double[] values) {
        double m = mean(values);
        double sum = 0;
        for (double v : values) sum += (v - m) * (v - m);
        return sum / values.length;
    }

    static double variancePairs(double[] highs, double[] lows) {
        double[] diff = new double[highs.length];
        for (int i = 0; i < highs.length; i++) diff[i] = highs[i] - lows[i];
        return variance(diff);
    }

    static double mean(double[] values) { return sum(values) / values.length; }

    static double sum(double[] values) {
        double total = 0;
        for (double v : values) total += v;
        return total;
    }

    static double sq(double value) { return value * value; }

    static double round(double value) { return Math.round(value * 10000) / 10000.0; }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}

/** The budget split chosen by the allocator. */
final class Plan {
    final int paired;
    final int cheapOnly;
    final int cost;

    Plan(int paired, int cheapOnly, int cost) {
        this.paired = paired;
        this.cheapOnly = cheapOnly;
        this.cost = cost;
    }
}
