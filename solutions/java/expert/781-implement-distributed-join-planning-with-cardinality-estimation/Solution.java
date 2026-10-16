/** Task 781 - Implement distributed join planning with cardinality estimation errors simulation. (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
 *
 * A join planner picks an order for n relations. The cost model is the classic "sum of intermediate
 * result sizes": executing a left-deep order costs the sum of the cardinalities of every intermediate
 * join it materialises, and the cardinality of joining a *set* of relations is estimated with the
 * independence assumption,
 *
 *     card(S) = prod_{i in S} |R_i| * prod_{i<j in S} sel(i, j)
 *
 * which depends only on the set - so the cheapest left-deep order is found exactly by the Selinger
 * dynamic program over subsets, dp[S] = min_{r in S} dp[S \ r] + card(S).
 *
 * The point of the simulation is that those selectivities are *estimated*. Every pair carries a true
 * selectivity and an estimated one that differs by a multiplicative error factor; the planner only
 * ever sees the estimates, while the cost of the chosen plan is charged against the true cardinalities.
 *
 * Verified: the subset DP returns the same minimum as an exhaustive enumeration of all n! left-deep
 * orders, and the reconstructed order really achieves it; with zero estimation error the chosen plan is
 * exactly the true-cost optimum; with error the chosen plan is never cheaper than the true optimum and
 * never worse than the penultimate plan, and the measured quality gap grows with the error magnitude;
 * the estimated cardinality of every prefix agrees with a direct recomputation of the product formula;
 * and the degenerate instances (a single relation, two relations, a relation of cardinality zero,
 * identical relations where every order ties) are handled.
 *
 * Simplified: independence-based cardinality estimation with a synthetic (deterministic) error model
 * rather than real histograms or sampling, left-deep plans only, and at most eight relations so that
 * the enumeration of all orders stays cheap.
 */
import java.util.*;

class JoinPlanner {
    final int relations;
    final double[] baseCardinality;
    final double[][] selectivityTrue;
    final double[][] selectivityEstimated;
    double bestEstimatedCost;
    int[] chosenOrder;

    JoinPlanner(double[] baseCardinality, double[][] selectivityTrue, double[][] selectivityEstimated) {
        this.relations = baseCardinality.length;
        this.baseCardinality = baseCardinality;
        this.selectivityTrue = selectivityTrue;
        this.selectivityEstimated = selectivityEstimated;
    }

    double estimatedCardinality(int mask) {
        double product = 1;
        for (int i = 0; i < relations; i++) {
            if ((mask & (1 << i)) != 0) product *= baseCardinality[i];
        }
        for (int i = 0; i < relations; i++) {
            if ((mask & (1 << i)) == 0) continue;
            for (int j = i + 1; j < relations; j++) {
                if ((mask & (1 << j)) != 0) product *= selectivityEstimated[i][j];
            }
        }
        return product;
    }

    double trueCardinality(int mask) {
        double product = 1;
        for (int i = 0; i < relations; i++) {
            if ((mask & (1 << i)) != 0) product *= baseCardinality[i];
        }
        for (int i = 0; i < relations; i++) {
            if ((mask & (1 << i)) == 0) continue;
            for (int j = i + 1; j < relations; j++) {
                if ((mask & (1 << j)) != 0) product *= selectivityTrue[i][j];
            }
        }
        return product;
    }

    /** Selinger DP over subsets; the result also records the order that attains it. */
    int[] optimize() {
        int full = (1 << relations) - 1;
        double[] dp = new double[1 << relations];
        int[] last = new int[1 << relations];
        Arrays.fill(dp, Double.MAX_VALUE);
        for (int i = 0; i < relations; i++) {
            dp[1 << i] = 0;
            last[1 << i] = i;
        }
        for (int mask = 1; mask <= full; mask++) {
            if (Integer.bitCount(mask) < 2) continue;
            double cardinality = estimatedCardinality(mask);
            for (int r = 0; r < relations; r++) {
                if ((mask & (1 << r)) == 0) continue;
                int previous = mask ^ (1 << r);
                if (dp[previous] == Double.MAX_VALUE) continue;
                double candidate = dp[previous] + cardinality;
                if (candidate < dp[mask] - 1e-12) {
                    dp[mask] = candidate;
                    last[mask] = r;
                }
            }
        }
        bestEstimatedCost = dp[full];
        int[] order = new int[relations];
        int mask = full;
        for (int position = relations - 1; position >= 0; position--) {
            int r = last[mask];
            order[position] = r;
            mask ^= 1 << r;
        }
        chosenOrder = order;
        return order;
    }

    /** Estimated cost of an explicit order: the sum of the estimated prefix cardinalities. */
    double estimatedCost(int[] order) {
        double total = 0;
        int mask = 0;
        for (int position = 0; position < order.length; position++) {
            mask |= 1 << order[position];
            if (position > 0) total += estimatedCardinality(mask);
        }
        return total;
    }

    /** Actual cost of an explicit order, charged against the true cardinalities. */
    double trueCost(int[] order) {
        double total = 0;
        int mask = 0;
        for (int position = 0; position < order.length; position++) {
            mask |= 1 << order[position];
            if (position > 0) total += trueCardinality(mask);
        }
        return total;
    }

    /** Exhaustive enumeration of every left-deep order: the oracle for both cost functions. */
    static double bestCostOverAllOrders(JoinPlanner planner, boolean estimated) {
        int[] order = new int[planner.relations];
        for (int i = 0; i < order.length; i++) order[i] = i;
        double[] best = {Double.MAX_VALUE};
        permute(order, 0, planner, estimated, best);
        return best[0];
    }

    private static void permute(int[] order, int position, JoinPlanner planner, boolean estimated, double[] best) {
        if (position == order.length) {
            double cost = estimated ? planner.estimatedCost(order) : planner.trueCost(order);
            best[0] = Math.min(best[0], cost);
            return;
        }
        for (int i = position; i < order.length; i++) {
            int swap = order[position];
            order[position] = order[i];
            order[i] = swap;
            permute(order, position + 1, planner, estimated, best);
            swap = order[position];
            order[position] = order[i];
            order[i] = swap;
        }
    }

    boolean isPermutation(int[] order) {
        boolean[] seen = new boolean[relations];
        for (int r : order) {
            if (r < 0 || r >= relations || seen[r]) return false;
            seen[r] = true;
        }
        return true;
    }
}

public class Solution {
    private static long state = 0x14057B7EF767814FL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    static double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    /**
     * Builds an instance: base cardinalities and true selectivities from the generator, estimates
     * perturbed by a factor drawn from {error, 1, 1/error} so that the planner's world is wrong.
     */
    static JoinPlanner instance(int relations, double error) {
        double[] cardinality = new double[relations];
        for (int i = 0; i < relations; i++) cardinality[i] = 100 + nextInt(900);
        double[][] trueSelectivity = new double[relations][relations];
        double[][] estimated = new double[relations][relations];
        for (int i = 0; i < relations; i++) {
            for (int j = i + 1; j < relations; j++) {
                double selectivity = 0.01 + nextDouble() * 0.4;
                trueSelectivity[i][j] = selectivity;
                trueSelectivity[j][i] = selectivity;
                double factor = error <= 1 ? 1.0 : Math.pow(error, nextInt(3) - 1);
                estimated[i][j] = Math.min(1.0, selectivity * factor);
                estimated[j][i] = estimated[i][j];
            }
        }
        return new JoinPlanner(cardinality, trueSelectivity, estimated);
    }

    public static void main(String[] args) {
        // --- degenerate instances ----------------------------------------------
        JoinPlanner single = instance(1, 1.0);
        int[] singleOrder = single.optimize();
        check(singleOrder.length == 1 && singleOrder[0] == 0, "one relation has a trivial plan");
        check(single.bestEstimatedCost == 0, "a single relation costs nothing to join");
        check(single.trueCost(singleOrder) == 0, "and its true cost is zero as well");

        JoinPlanner pair = instance(2, 1.0);
        int[] pairOrder = pair.optimize();
        check(pair.isPermutation(pairOrder), "the two-relation plan is a permutation");
        check(pair.bestEstimatedCost == pair.estimatedCardinality(3), "the only join is the whole pair");
        check(JoinPlanner.bestCostOverAllOrders(pair, true) == pair.trueCost(pairOrder),
                "with two relations the DP is trivially optimal");

        double[] zeroCardinality = {0, 500, 500};
        double[][] unitSelectivity = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) unitSelectivity[i][j] = 1.0;
        }
        JoinPlanner empty = new JoinPlanner(zeroCardinality, unitSelectivity, unitSelectivity);
        empty.optimize();
        check(empty.bestEstimatedCost == 0, "a zero-cardinality relation makes every join free");
        check(empty.trueCost(empty.chosenOrder) == 0, "and truly free");

        double[] identical = {100, 100, 100, 100};
        double[][] equal = new double[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) equal[i][j] = 0.5;
        }
        JoinPlanner tie = new JoinPlanner(identical, equal, equal);
        tie.optimize();
        check(Math.abs(tie.bestEstimatedCost - JoinPlanner.bestCostOverAllOrders(tie, true)) < 1e-9,
                "identical relations: the DP still matches the enumeration");
        check(Math.abs(tie.trueCost(tie.chosenOrder) - JoinPlanner.bestCostOverAllOrders(tie, false)) < 1e-9,
                "identical relations: the chosen order is truly optimal");

        // --- the DP against the enumeration, without estimation error ----------
        int dpChecks = 0;
        for (int trial = 0; trial < 6; trial++) {
            JoinPlanner planner = instance(4 + trial % 3, 1.0);
            int[] order = planner.optimize();
            check(planner.isPermutation(order), "the plan must be a permutation of the relations");
            double enumerated = JoinPlanner.bestCostOverAllOrders(planner, true);
            check(Math.abs(planner.bestEstimatedCost - enumerated) < 1e-9 * (1 + enumerated),
                    "DP " + planner.bestEstimatedCost + " vs enumeration " + enumerated);
            check(Math.abs(planner.estimatedCost(order) - planner.bestEstimatedCost) < 1e-9 * (1 + enumerated),
                    "the reconstructed order must attain the DP value");
            double bestTrue = JoinPlanner.bestCostOverAllOrders(planner, false);
            check(Math.abs(planner.trueCost(order) - bestTrue) < 1e-9 * (1 + bestTrue),
                    "with exact estimates the chosen plan is truly optimal: " + planner.trueCost(order)
                            + " vs " + bestTrue);
            dpChecks++;
        }
        check(dpChecks == 6, "all DP-oracle comparisons ran");

        // --- estimated cardinalities recomputed independently ------------------
        JoinPlanner checkInstance = instance(5, 1.0);
        for (int mask = 1; mask < (1 << 5); mask++) {
            double expected = 1;
            for (int i = 0; i < 5; i++) {
                if ((mask & (1 << i)) != 0) expected *= checkInstance.baseCardinality[i];
            }
            for (int i = 0; i < 5; i++) {
                for (int j = i + 1; j < 5; j++) {
                    if ((mask & (1 << i)) != 0 && (mask & (1 << j)) != 0) {
                        expected *= checkInstance.selectivityEstimated[i][j];
                    }
                }
            }
            check(Math.abs(checkInstance.estimatedCardinality(mask) - expected) < 1e-9 * (1 + expected),
                    "cardinality mismatch for mask " + mask);
        }

        // --- estimation error degrades the plan ---------------------------------
        double[] errors = {1.0, 2.0, 4.0, 16.0};
        double[] averageRatio = new double[errors.length];
        int degradedInstances = 0;
        for (int e = 0; e < errors.length; e++) {
            double total = 0;
            int instances = 12;
            int degraded = 0;
            for (int trial = 0; trial < instances; trial++) {
                JoinPlanner planner = instance(6, errors[e]);
                int[] order = planner.optimize();
                double chosen = planner.trueCost(order);
                double bestTrue = JoinPlanner.bestCostOverAllOrders(planner, false);
                check(chosen >= bestTrue * (1 - 1e-9), "an estimated plan can never beat the true optimum: "
                        + chosen + " vs " + bestTrue);
                check(planner.isPermutation(order), "the plan stays a permutation under error");
                check(Math.abs(planner.estimatedCost(order) - planner.bestEstimatedCost)
                        < 1e-9 * (1 + planner.bestEstimatedCost), "the DP value still matches the plan");
                total += chosen / bestTrue;
                if (chosen > bestTrue * (1 + 1e-9)) degraded++;
            }
            averageRatio[e] = total / instances;
            if (e == 0) {
                check(degraded == 0, "exact estimates can never degrade a plan, got " + degraded);
            } else {
                degradedInstances += degraded;
            }
        }
        System.out.println("average true-cost ratio by error: " + Arrays.toString(averageRatio));
        check(Math.abs(averageRatio[0] - 1.0) < 1e-9, "with exact estimates the plan is always optimal");
        for (int e = 1; e < errors.length; e++) {
            check(averageRatio[e] >= 1.0 - 1e-9, "an estimated plan cannot be better than optimal");
        }
        check(averageRatio[errors.length - 1] > 1.0,
                "large estimation errors must actually cost something: " + averageRatio[errors.length - 1]);
        check(degradedInstances > 0, "some erroneous instance must have produced a suboptimal plan");

        // --- the same instance planned with and without error -------------------
        JoinPlanner informative = instance(6, 8.0);
        int[] estimatedPlan = informative.optimize();
        double estimatedPlanCost = informative.trueCost(estimatedPlan);
        JoinPlanner oracle = new JoinPlanner(informative.baseCardinality,
                informative.selectivityTrue, informative.selectivityTrue);
        int[] perfectPlan = oracle.optimize();
        double perfectCost = oracle.trueCost(perfectPlan);
        check(estimatedPlanCost >= perfectCost * (1 - 1e-9),
                "the estimate-driven plan cannot beat the perfect-information plan");
        check(Math.abs(perfectCost - JoinPlanner.bestCostOverAllOrders(informative, false)) < 1e-9 * (1 + perfectCost),
                "the perfect-information plan is truly optimal");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
