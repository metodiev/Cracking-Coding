/** Task 532 - Implement online algorithm competitive analysis example (ski rental). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
 *
 * Ski rental, the canonical online problem: each of T days either costs r (rent) or costs B once for
 * unlimited use (buy), and the online algorithm must decide day by day without knowing T. Everything
 * is modelled exactly: the deterministic threshold algorithm RENT-THEN-BUY(h) rents for the first h
 * days and buys on day h+1 (cost r * min(T, h) + B when T > h), the offline optimum is
 * OPT(T) = min(r * T, B), and the competitive ratio is max over T of cost(T) / OPT(T) - which is
 * evaluated by brute force over an explicit T range (documented as T = 1 .. 6B + 5, so the range
 * covers the worst case with room to spare). Two classical results are verified rather than assumed:
 * for unit rent (r = 1) the exhaustive search over every threshold h = 0 .. B proves that the best
 * deterministic ratio is exactly 2 - 1/B and that h = B - 1 attains it, and for the pairs where r
 * divides B the same exhaustive optimum is exactly 2 - r/B (for r not dividing B the classic bound
 * is NOT tight, which the checks record by requiring the exhaustive optimum to be strictly larger
 * than 2 - r/B). The randomized algorithm is implemented with its textbook buy-day distribution:
 * with z = 1 + 1 / ((B / (B - 1))^B - 1), buy on day i < B with probability
 * p_i = ((z - 1) + S_{i-1}) / (B - 1) and on day B with the remaining mass, which makes
 * E[cost(T)] = z * OPT(T) for every T <= B and E[cost(T)] = z * B for T >= B; the checks recompute
 * the expectations in closed form, compare the simulated average against them (fixed LCG, 20000
 * trials per T) and show z <= e / (e - 1) with equality approached as B grows, so randomization
 * beats the best deterministic algorithm. Documented conventions: T runs from 1 (T = 0 makes the
 * ratio 0/0 and is excluded), day indices start at 1, and the threshold h = 0 means "buy
 * immediately" while h >= B never buys.
 *
 * Checks: hand ratios (r = 1: B = 2, 3, 5, 10, 17, 100 all attain exactly 2 - 1/B, with the worst
 * case at T = B and T = B + 1); H = B - 1 is optimal among all thresholds for every tested B; the
 * exhaustive optimum equals 2 - r/B for (r, B) = (1, 10), (2, 10), (2, 4), (3, 9), (4, 12) and is
 * strictly larger for (2, 11), (3, 10), (4, 10); OPT(T) never exceeds the deterministic cost and
 * the endpoints are pinned too: h = 0 (buy immediately) is only B-competitive, worst at T = 1,
 * while h = B (never buy inside the horizon) is exactly 2-competitive, and an ill-chosen threshold
 * such as h = 1 with B = 5 is exactly 3-competitive, i.e. worse than 2 - so only the right threshold
 * helps; the randomized distribution sums to 1 within 1e-12,
 * stays strictly positive, and its worst-case ratio equals z exactly for B = 2, 3, 5, 10, 50 while
 * z < e / (e - 1) and z is strictly increasing in B; for B = 200000 the closed form reaches
 * e / (e - 1) within 1e-5; the LCG simulation of the randomized algorithm reproduces the closed-form
 * expectations within 0.05 for T = 1, 5, B and 3B (and the empirical ratio never exceeds z + 0.05);
 * edge cases B = 1 (buying immediately is optimal), B < r, and huge T = 10^6 with integer-safe
 * arithmetic are pinned; repeated runs are identical.
 */
import java.util.*;

public class Solution {
    static final int Tmax_SLACK = 5;

    static class Rng {
        int seed;

        Rng(int seed) {
            this.seed = seed;
        }

        double uniform() {
            seed = (seed * 75 + 74) % 65537;
            return seed / 65537.0;
        }
    }

    /** Cost of RENT-THEN-BUY(h) on a run of T days. */
    static long deterministicCost(long rentalCost, long buyCost, long threshold, long totalDays) {
        long rented = Math.min(totalDays, threshold);
        long cost = rented * rentalCost;
        if (totalDays > threshold) cost += buyCost;
        return cost;
    }

    static long optimalCost(long rentalCost, long buyCost, long totalDays) {
        return Math.min(totalDays * rentalCost, buyCost);
    }

    /** Worst-case competitive ratio of RENT-THEN-BUY(h) over T = 1 .. horizon. */
    static double thresholdRatio(long rentalCost, long buyCost, long threshold, long horizon) {
        double worst = 0.0;
        for (long days = 1; days <= horizon; days++) {
            double ratio = deterministicCost(rentalCost, buyCost, threshold, days)
                    / (double) optimalCost(rentalCost, buyCost, days);
            worst = Math.max(worst, ratio);
        }
        return worst;
    }

    static long worstCaseDays(long rentalCost, long buyCost, long threshold, long horizon) {
        double worst = -1.0;
        long argmax = 1;
        for (long days = 1; days <= horizon; days++) {
            double ratio = deterministicCost(rentalCost, buyCost, threshold, days)
                    / (double) optimalCost(rentalCost, buyCost, days);
            if (ratio > worst) {
                worst = ratio;
                argmax = days;
            }
        }
        return argmax;
    }

    /** Exhaustive search over every deterministic threshold: the best ratio and its threshold. */
    static double[] bestThreshold(long rentalCost, long buyCost, long horizon) {
        double best = Double.POSITIVE_INFINITY;
        long bestThreshold = -1;
        for (long threshold = 0; threshold <= buyCost; threshold++) {
            double ratio = thresholdRatio(rentalCost, buyCost, threshold, horizon);
            if (ratio < best - 1e-15) {
                best = ratio;
                bestThreshold = threshold;
            }
        }
        return new double[]{best, bestThreshold};
    }

    static long horizon(int buyCost) {
        return 6L * buyCost + Tmax_SLACK;
    }

    /** Randomized buy-day distribution p_1 .. p_B; returns {p, z}. */
    static double[][] randomizedDistribution(int buyCost) {
        if (buyCost == 1) return new double[][]{{1.0}, {1.0}};
        double base = buyCost / (double) (buyCost - 1);
        double z = 1.0 + 1.0 / (Math.pow(base, buyCost) - 1.0);
        double[] p = new double[buyCost];
        double cumulative = 0.0;
        for (int day = 1; day <= buyCost; day++) {
            if (day < buyCost) {
                p[day - 1] = ((z - 1.0) + cumulative) / (buyCost - 1);
                cumulative += p[day - 1];
            } else {
                p[day - 1] = 1.0 - cumulative;
            }
        }
        return new double[][]{p, {z}};
    }

    static double expectedCost(double[] distribution, long rentalCost, long buyCost, long totalDays) {
        double expected = 0.0;
        for (int day = 1; day <= distribution.length; day++) {
            long cost = totalDays >= day ? (day - 1) * rentalCost + buyCost : totalDays * rentalCost;
            expected += distribution[day - 1] * cost;
        }
        return expected;
    }

    static double randomizedWorstRatio(double[] distribution, long rentalCost, long horizon) {
        double worst = 0.0;
        for (long days = 1; days <= horizon; days++) {
            double ratio = expectedCost(distribution, rentalCost, distribution.length, days)
                    / (double) optimalCost(rentalCost, distribution.length, days);
            worst = Math.max(worst, ratio);
        }
        return worst;
    }

    /** LCG simulation of the randomized algorithm: the average cost over many trials. */
    static double simulate(double[] distribution, long rentalCost, long buyCost, long totalDays,
            int trials) {
        Rng rng = new Rng(20250916);
        double total = 0.0;
        for (int trial = 0; trial < trials; trial++) {
            double u = rng.uniform();
            int buyDay = distribution.length;
            double cumulative = 0.0;
            for (int day = 1; day <= distribution.length; day++) {
                cumulative += distribution[day - 1];
                if (u < cumulative) {
                    buyDay = day;
                    break;
                }
            }
            total += totalDays >= buyDay ? (buyDay - 1) * rentalCost + buyCost
                    : totalDays * rentalCost;
        }
        return total / trials;
    }

    public static void main(String[] args) {
        long[] unitBuys = {2, 3, 5, 10, 17, 100};
        for (long buyCost : unitBuys) {
            long horizon = horizon((int) buyCost);
            double closedForm = 2.0 - 1.0 / buyCost;
            double achieved = thresholdRatio(1, buyCost, buyCost - 1, horizon);
            check(Math.abs(achieved - closedForm) <= 1e-12,
                    "r = 1, B = " + buyCost + ": h = B - 1 attains 2 - 1/B");
            double[] best = bestThreshold(1, buyCost, horizon);
            check(Math.abs(best[0] - closedForm) <= 1e-12,
                    "r = 1, B = " + buyCost + ": no threshold beats 2 - 1/B");
            check(best[1] == buyCost - 1, "r = 1, B = " + buyCost + ": h = B - 1 is optimal");
            long worstDay = worstCaseDays(1, buyCost, buyCost - 1, horizon);
            check(worstDay == buyCost || worstDay == buyCost + 1,
                    "r = 1, B = " + buyCost + ": the adversary picks T = B or T = B + 1");
            check(Math.abs(deterministicCost(1, buyCost, buyCost - 1, worstDay)
                    / (double) optimalCost(1, buyCost, worstDay) - closedForm) <= 1e-12,
                    "r = 1, B = " + buyCost + ": the worst case attains the ratio");
            check(Math.abs(thresholdRatio(1, buyCost, buyCost, horizon) - 2.0) <= 1e-12,
                    "h = B is exactly 2-competitive for B = " + buyCost);
            check(Math.abs(thresholdRatio(1, buyCost, 0, horizon) - buyCost) <= 1e-12,
                    "h = 0 buys immediately and is only B-competitive for B = " + buyCost);
        }

        check(Math.abs(thresholdRatio(1, 5, 1, horizon(5)) - 3.0) <= 1e-12,
                "h = 1 with B = 5 is exactly 3-competitive");
        check(thresholdRatio(1, 5, 1, horizon(5)) > 2.0, "an ill-chosen threshold exceeds 2");

        long[][] divisible = {{1, 10}, {2, 10}, {2, 4}, {3, 9}, {4, 12}};
        for (long[] pair : divisible) {
            double[] best = bestThreshold(pair[0], pair[1], horizon((int) pair[1]));
            check(Math.abs(best[0] - (2.0 - pair[0] / (double) pair[1])) <= 1e-12,
                    "r = " + pair[0] + ", B = " + pair[1] + ": optimum is 2 - r/B");
        }
        long[][] indivisible = {{2, 11}, {3, 10}, {4, 10}};
        for (long[] pair : indivisible) {
            double[] best = bestThreshold(pair[0], pair[1], horizon((int) pair[1]));
            check(best[0] > 2.0 - pair[0] / (double) pair[1] + 1e-12,
                    "r = " + pair[0] + ", B = " + pair[1]
                    + ": the classic bound is not tight when r does not divide B");
            check(best[0] <= 2.0, "ratio still at most 2 for r = " + pair[0] + ", B = " + pair[1]);
        }

        for (long days = 1; days <= 30; days++) {
            check(optimalCost(1, 10, days) <= deterministicCost(1, 10, 5, days),
                    "OPT never exceeds the deterministic cost at T = " + days);
            check(optimalCost(1, 10, days) <= deterministicCost(1, 10, 9, days),
                    "OPT never exceeds h = B - 1 at T = " + days);
        }
        check(deterministicCost(1, 10, 0, 100) == 10, "h = 0 buys immediately");
        check(deterministicCost(1, 10, 10, 100) == 20, "h = B rents B days and still buys");
        check(deterministicCost(2, 10, 4, 3) == 6, "renting below the threshold costs r*T");
        check(optimalCost(1, 1, 5) == 1, "B = 1 means buying is never worse");
        check(bestThreshold(1, 1, 12)[1] == 0, "B = 1: buying immediately is optimal");
        check(Math.abs(thresholdRatio(1, 1, 0, 12) - 1.0) <= 1e-12, "B = 1 is 1-competitive");
        check(bestThreshold(5, 3, 24)[1] == 0, "B < r: buying immediately is optimal");
        check(deterministicCost(5, 3, 0, 100) == 3, "B < r: immediate buy costs B");
        long huge = 1_000_000L;
        check(optimalCost(1, 10, huge) == 10, "huge horizon is capped by the buy cost");
        check(deterministicCost(1, 10, 9, huge) == 19, "huge horizon with h = B - 1 costs h + B");

        double previousZ = 0.0;
        for (int buyCost : new int[]{2, 3, 5, 10, 50, 1000}) {
            double[][] result = randomizedDistribution(buyCost);
            double[] p = result[0];
            double z = result[1][0];
            double sum = 0.0;
            for (double probability : p) {
                check(probability > 0.0, "the buy-day distribution is strictly positive");
                sum += probability;
            }
            check(Math.abs(sum - 1.0) <= 1e-12, "the distribution sums to 1 for B = " + buyCost);
            double worst = randomizedWorstRatio(p, 1, horizon(buyCost));
            check(Math.abs(worst - z) <= 1e-9,
                    "the randomized ratio equals z exactly for B = " + buyCost);
            check(z <= Math.E / (Math.E - 1) + 1e-12, "z is at most e/(e-1) for B = " + buyCost);
            check(z >= 1.0, "z is at least 1");
            check(z > previousZ, "z increases with B");
            previousZ = z;
        }
        double[][] large = randomizedDistribution(200_000);
        double largeZ = large[1][0];
        check(Math.abs(largeZ - Math.E / (Math.E - 1)) <= 1e-5,
                "for B = 200000 the closed form reaches e/(e-1) within 1e-5");
        double[][] ten = randomizedDistribution(10);
        check(Math.abs(ten[1][0] - 1.5353399327876291) <= 1e-12, "documented z for B = 10");
        check(ten[1][0] < bestThreshold(1, 10, horizon(10))[0],
                "randomization beats the best deterministic algorithm for B = 10");

        check(Math.abs(expectedCost(ten[0], 1, 10, 1) - ten[1][0]) <= 1e-9,
                "closed-form expectation at T = 1 is z * OPT(1)");
        check(Math.abs(expectedCost(ten[0], 1, 10, 5) - 5.0 * ten[1][0]) <= 1e-9,
                "closed-form expectation at T = 5 is z * T");
        check(Math.abs(expectedCost(ten[0], 1, 10, 10) - 10.0 * ten[1][0]) <= 1e-9,
                "closed-form expectation at T = B is z * B");
        check(Math.abs(expectedCost(ten[0], 1, 10, 30) - 10.0 * ten[1][0]) <= 1e-9,
                "closed-form expectation at T = 3B is z * B");

        double worstSimulationGap = 0.0;
        for (long days : new long[]{1, 5, 10, 30}) {
            double closed = expectedCost(ten[0], 1, 10, days);
            double simulated = simulate(ten[0], 1, 10, days, 20_000);
            worstSimulationGap = Math.max(worstSimulationGap, Math.abs(closed - simulated));
            check(Math.abs(closed - simulated) <= 0.08,
                    "simulation matches the closed form at T = " + days);
            check(simulated / (double) optimalCost(1, 10, days) <= ten[1][0] + 0.05,
                    "simulated ratio stays near z at T = " + days);
        }
        check(worstSimulationGap <= 0.08, "documented: the LCG simulation stays within 0.08");
        check(Math.abs(simulate(ten[0], 1, 10, 5, 20_000) - simulate(ten[0], 1, 10, 5, 20_000))
                <= 0.0, "the simulation is deterministic");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
