/** Task 856 - Implement runtime adaptive operator re-optimization and replanning.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   A query is a chain of S stages. Each stage has three physical alternatives - HASH with a linear cost,
 *   NLJ with a quadratic cost and SORT_MERGE with a steeper linear cost - and the cost of a stage depends on
 *   the *input* cardinality it actually sees. The optimizer starts from the alternative that looks cheapest
 *   for its estimate of the first stage; as the query runs, every stage reveals its true output cardinality and
 *   the executor may re-plan the remaining stages:
 *
 *       observe c_{i+1} -> rescale the remaining estimates by c_{i+1} / estimate_{i+1}   (error propagation)
 *       recompute the cheapest single plan for the whole remaining chain
 *       switch only if gain > 0, gain / stay > hysteresis, and i - lastSwitch >= minStable
 *
 *   Switches cost a fixed penalty, so re-planning is traded against switch overhead. Three workload families
 *   make the behaviour observable: a *collapse* (the true cardinality collapses while the estimate stays
 *   large, so the static plan keeps a quadratic operator that becomes ruinous), a *steady* workload with
 *   perfectly accurate estimates (nothing to fix) and an *oscillating* family whose per-stage estimates
 *   alternate between x8 and /8 around the truth (a naive re-optimizer thrashes, the stabilised one does not).
 *
 * WHAT IS SIMPLIFIED
 *   One query chain, no parallelism, no mid-stage cancellation, no statistics refresh and no rollback after a
 *   switch: "executing a stage" is evaluating its cost function on the revealed cardinality, and the plan the
 *   executor would install is decided at stage boundaries only. No wall-clock time, no threads; all
 *   cardinalities and estimate errors come from a fixed-seed splitmix64 stream.
 *
 * INVARIANTS VERIFIED
 *   Model: per-operator costs are deterministic and monotone, each linear operator crosses the quadratic one
 *   exactly once (located by brute-force scanning), and the observed cardinality chain equals an independent
 *   running product of selectivities.
 *   Optimality / bounds: exhaustive enumeration over all 3^S plan assignments equals an independent dynamic
 *   program; the adaptive executor never beats that optimum and stays within a documented factor of it; on a
 *   collapsing workload it re-plans and beats the static plan, while with accurate estimates it reproduces the
 *   static plan exactly and switches zero times; a prohibitive penalty also degenerates to the static plan and
 *   the total cost is non-decreasing in the penalty.
 *   Accounting / stability: the reported total always equals stage costs plus switches x penalty, the plan log
 *   replays to the reported stage cost independently, the stabilisation window bounds the switch count while
 *   the same workload without it thrashes and costs more, an empty or single-stage chain is handled, and every
 *   run is reproducible from its seed.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every instance and decision log is reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    /** Log-uniform jitter in [1/ratio, ratio]: the usual shape of a cardinality misestimate. */
    double jitter(double ratio) { return Math.exp((2 * nextDouble() - 1) * Math.log(ratio)); }
}

/** One physical alternative: cost = fixed + linear * rows + quad * rows^2. */
final class Alternative {
    final String name;
    final double fixed;
    final double linear;
    final double quad;

    Alternative(String name, double fixed, double linear, double quad) {
        this.name = name;
        this.fixed = fixed;
        this.linear = linear;
        this.quad = quad;
    }

    double cost(double rows) { return fixed + linear * rows + quad * rows * rows; }

    @Override public String toString() { return name; }
}

/** A stage: the true selectivity and the optimizer's estimate of it. */
final class Stage {
    final double trueSelectivity;
    final double estimatedSelectivity;

    Stage(double trueSelectivity, double estimatedSelectivity) {
        this.trueSelectivity = trueSelectivity;
        this.estimatedSelectivity = estimatedSelectivity;
    }
}

/** Result of one strategy: the plan used at each stage, the switch count and the fully accounted cost. */
final class RunResult {
    final int[] planByStage;
    final int switches;
    final double stageCost;
    final double penalty;
    final double totalCost;

    RunResult(int[] planByStage, int switches, double stageCost, double penalty) {
        this.planByStage = planByStage;
        this.switches = switches;
        this.stageCost = stageCost;
        this.penalty = penalty;
        this.totalCost = stageCost + switches * penalty;
    }
}

/** A workload instance: the stage chain, the alternatives and the cardinality entering the first stage. */
final class Instance {
    final String name;
    final Stage[] stages;
    final List<Alternative> alternatives;
    final double initialRows;

    Instance(String name, Stage[] stages, List<Alternative> alternatives, double initialRows) {
        this.name = name;
        this.stages = stages;
        this.alternatives = alternatives;
        this.initialRows = initialRows;
    }

    double[] trueCardinalities() {
        double[] cardinalities = new double[stages.length + 1];
        cardinalities[0] = initialRows;
        for (int i = 0; i < stages.length; i++) cardinalities[i + 1] = cardinalities[i] * stages[i].trueSelectivity;
        return cardinalities;
    }

    double[] estimatedCardinalities() {
        double[] cardinalities = new double[stages.length + 1];
        cardinalities[0] = initialRows;
        for (int i = 0; i < stages.length; i++) {
            cardinalities[i + 1] = cardinalities[i] * stages[i].estimatedSelectivity;
        }
        return cardinalities;
    }

    int argMin(double rows) {
        int best = 0;
        for (int i = 1; i < alternatives.size(); i++) {
            if (alternatives.get(i).cost(rows) < alternatives.get(best).cost(rows)) best = i;
        }
        return best;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    static List<Alternative> standardAlternatives() {
        return List.of(new Alternative("HASH", 20, 1.2, 0),
                new Alternative("NLJ", 5, 0.1, 0.05),
                new Alternative("SORT_MERGE", 15, 2.0, 0));
    }

    // ------------------------------------------------------------------ strategies

    /** Static strategy: pick the best alternative once, using the estimate, and never re-plan. */
    static RunResult runStatic(Instance instance, double penalty) {
        double[] trueCards = instance.trueCardinalities();
        int plan = instance.argMin(instance.estimatedCardinalities()[0]);
        int[] planByStage = new int[instance.stages.length];
        double cost = 0;
        for (int i = 0; i < instance.stages.length; i++) {
            planByStage[i] = plan;
            cost += instance.alternatives.get(plan).cost(trueCards[i]);
        }
        return new RunResult(planByStage, 0, cost, penalty);
    }

    /**
     * Adaptive executor with receding-horizon re-optimisation: after every stage the observed cardinality
     * rescales the remaining estimates, the cheapest single plan for the whole remaining chain is recomputed,
     * and the switch is taken only when it pays for the penalty, the relative dead-band and the stabilisation
     * window.
     */
    static RunResult runAdaptive(Instance instance, double penalty, double hysteresis, int minStable,
            boolean useStabilisation) {
        double[] trueCards = instance.trueCardinalities();
        double[] estimatedCards = instance.estimatedCardinalities();
        int stageCount = instance.stages.length;
        int altCount = instance.alternatives.size();
        int[] planByStage = new int[stageCount];
        int plan = instance.argMin(estimatedCards[0]);
        int switches = 0;
        int lastSwitch = -minStable;
        double cost = 0;
        for (int i = 0; i < stageCount; i++) {
            planByStage[i] = plan;
            cost += instance.alternatives.get(plan).cost(trueCards[i]);
            if (i + 1 >= stageCount) break;
            double scale = trueCards[i + 1] / Math.max(1e-9, estimatedCards[i + 1]);
            double stay = 0;
            double bestSum = Double.MAX_VALUE;
            int bestPlan = plan;
            for (int a = 0; a < altCount; a++) {
                double sum = 0;
                for (int j = i + 1; j < stageCount; j++) {
                    sum += instance.alternatives.get(a).cost(estimatedCards[j] * scale);
                }
                if (a == plan) stay = sum;
                if (sum < bestSum) {
                    bestSum = sum;
                    bestPlan = a;
                }
            }
            double gain = stay - (bestSum + penalty);
            boolean stable = !useStabilisation || (i - lastSwitch) >= minStable;
            boolean worthIt = gain > 0 && (!useStabilisation || gain / Math.max(1e-9, stay) > hysteresis);
            if (bestPlan != plan && worthIt && stable) {
                plan = bestPlan;
                switches++;
                cost += penalty;
                lastSwitch = i;
            }
        }
        return new RunResult(planByStage, switches, cost - switches * penalty, penalty);
    }

    /** Exhaustive offline optimum over all 3^S plan assignments, paying the penalty once per change. */
    static double offlineOptimalBruteForce(Instance instance, double penalty) {
        double[] trueCards = instance.trueCardinalities();
        int stageCount = instance.stages.length;
        int altCount = instance.alternatives.size();
        int total = 1;
        for (int i = 0; i < stageCount; i++) total *= altCount;
        double best = Double.MAX_VALUE;
        for (int code = 0; code < total; code++) {
            int[] assignment = new int[stageCount];
            int value = code;
            for (int i = 0; i < stageCount; i++) {
                assignment[i] = value % altCount;
                value /= altCount;
            }
            double cost = 0;
            for (int i = 0; i < stageCount; i++) cost += instance.alternatives.get(assignment[i]).cost(trueCards[i]);
            for (int i = 1; i < stageCount; i++) if (assignment[i] != assignment[i - 1]) cost += penalty;
            if (cost < best) best = cost;
        }
        return best;
    }

    /** Independent dynamic program over (stage, plan in use); an empty chain costs nothing. */
    static double offlineOptimalDp(Instance instance, double penalty) {
        int stageCount = instance.stages.length;
        if (stageCount == 0) return 0.0;
        double[] trueCards = instance.trueCardinalities();
        int altCount = instance.alternatives.size();
        double[] best = new double[altCount];
        for (int a = 0; a < altCount; a++) best[a] = instance.alternatives.get(a).cost(trueCards[0]);
        for (int i = 1; i < stageCount; i++) {
            double[] next = new double[altCount];
            for (int a = 0; a < altCount; a++) {
                double value = Double.MAX_VALUE;
                for (int previous = 0; previous < altCount; previous++) {
                    double candidate = best[previous] + instance.alternatives.get(a).cost(trueCards[i])
                            + (previous == a ? 0 : penalty);
                    value = Math.min(value, candidate);
                }
                next[a] = value;
            }
            best = next;
        }
        double optimum = Double.MAX_VALUE;
        for (double value : best) optimum = Math.min(optimum, value);
        return optimum;
    }

    /** Replays a plan log against the true cardinalities to recompute the stage cost independently. */
    static double replayStageCost(Instance instance, int[] planByStage) {
        double[] trueCards = instance.trueCardinalities();
        double cost = 0;
        for (int i = 0; i < planByStage.length; i++) {
            cost += instance.alternatives.get(planByStage[i]).cost(trueCards[i]);
        }
        return cost;
    }

    static int countChanges(int[] assignment) {
        int changes = 0;
        for (int i = 1; i < assignment.length; i++) if (assignment[i] != assignment[i - 1]) changes++;
        return changes;
    }

    // ------------------------------------------------------------------ workload families

    /** The true cardinality collapses almost to nothing while the estimate stays near the base value. */
    static Instance collapse(String name, int stageCount, long seed, double initialRows) {
        SplitMix64 rng = new SplitMix64(seed);
        Stage[] stages = new Stage[stageCount];
        for (int i = 0; i < stageCount; i++) {
            double truth = 0.04 + 0.03 * rng.nextDouble();
            double estimate = 0.85 + 0.1 * rng.nextDouble();
            stages[i] = new Stage(truth, estimate);
        }
        return new Instance(name, stages, standardAlternatives(), initialRows);
    }

    /** The true cardinality grows slowly while the estimates are exactly right: nothing needs re-planning. */
    static Instance steady(String name, int stageCount, long seed, double initialRows) {
        SplitMix64 rng = new SplitMix64(seed);
        Stage[] stages = new Stage[stageCount];
        for (int i = 0; i < stageCount; i++) {
            double truth = 0.9 + 0.7 * rng.nextDouble();
            stages[i] = new Stage(truth, truth);
        }
        return new Instance(name, stages, standardAlternatives(), initialRows);
    }

    /**
     * Per-stage estimates alternate between x ratio and / ratio around a truth that sits right at the
     * crossover cardinality, so the projected best plan for the remaining chain keeps flipping.
     */
    static Instance oscillating(String name, int stageCount, long seed, double initialRows, double ratio) {
        SplitMix64 rng = new SplitMix64(seed);
        Stage[] stages = new Stage[stageCount];
        for (int i = 0; i < stageCount; i++) {
            double truth = 0.9 + 0.1 * rng.nextDouble();
            double estimate = truth * (i % 2 == 0 ? 1.0 / ratio : ratio);
            stages[i] = new Stage(truth, Math.min(8.0, Math.max(0.001, estimate)));
        }
        return new Instance(name, stages, standardAlternatives(), initialRows);
    }

    /** Random log-uniform misestimation of the given ratio. */
    static Instance jittered(String name, int stageCount, long seed, double ratio, double initialRows) {
        SplitMix64 rng = new SplitMix64(seed);
        Stage[] stages = new Stage[stageCount];
        for (int i = 0; i < stageCount; i++) {
            double truth = 0.3 + 0.6 * rng.nextDouble();
            double estimate = Math.min(0.999, Math.max(0.0001, truth * rng.jitter(ratio)));
            stages[i] = new Stage(truth, estimate);
        }
        return new Instance(name, stages, standardAlternatives(), initialRows);
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        List<Alternative> alternatives = standardAlternatives();
        final double penalty = 15.0;

        // --- 1. cost functions: deterministic, monotone, with exactly one linear/quadratic crossover.
        for (Alternative a : alternatives) {
            check(a.cost(10) < a.cost(11) && a.cost(11) < a.cost(100), a.name + " cost is monotone in rows");
            check(a.cost(50) == a.cost(50), a.name + " cost is deterministic");
        }
        Alternative hash = alternatives.get(0);
        Alternative nestedLoop = alternatives.get(1);
        int crossings = 0;
        int crossing = -1;
        for (int rows = 1; rows < 400; rows++) {
            if ((nestedLoop.cost(rows) < hash.cost(rows)) != (nestedLoop.cost(rows + 1) < hash.cost(rows + 1))) {
                crossings++;
                if (crossing < 0) crossing = rows;
            }
        }
        check(crossings == 1, "the quadratic and linear operators cross exactly once");
        check(nestedLoop.cost(crossing) < hash.cost(crossing)
                        && hash.cost(crossing + 1) < nestedLoop.cost(crossing + 1),
                "the crossover is located at row " + crossing);
        check(nestedLoop.cost(5) < hash.cost(5) && hash.cost(3000) < nestedLoop.cost(3000),
                "small inputs prefer the quadratic plan, large inputs the linear one");

        // --- 2. cardinality chains against an independent running product.
        Instance collapsing = collapse("collapse", 12, 856_2024L, 400);
        double[] trueCards = collapsing.trueCardinalities();
        double product = collapsing.initialRows;
        boolean chainOk = true;
        for (int i = 0; i < collapsing.stages.length; i++) {
            product *= collapsing.stages[i].trueSelectivity;
            chainOk &= Math.abs(trueCards[i + 1] - product) < 1e-9;
        }
        check(chainOk, "the true cardinality chain equals the running product of selectivities");
        check(trueCards[0] == collapsing.initialRows, "the first stage sees the base cardinality");
        check(trueCards[3] < trueCards[0] / 100, "the collapsing workload really collapses the cardinality");

        // --- 3. the static strategy replays exactly.
        RunResult staticRun = runStatic(collapsing, penalty);
        check(Math.abs(staticRun.stageCost - replayStageCost(collapsing, staticRun.planByStage)) < 1e-9,
                "static stage cost replays exactly");
        check(staticRun.switches == 0, "the static strategy never switches");
        double staticDirect = 0;
        for (int i = 0; i < collapsing.stages.length; i++) {
            staticDirect += alternatives.get(staticRun.planByStage[0]).cost(trueCards[i]);
        }
        check(Math.abs(staticDirect - staticRun.totalCost) < 1e-9, "the static total equals the direct sum");

        // --- 4. the two offline oracles agree everywhere and bound the executor from below.
        for (Instance instance : List.of(collapsing, steady("steady", 8, 7L, 400),
                oscillating("osc", 12, 99L, 32, 6.0), jittered("jitter", 8, 5L, 2.0, 1000))) {
            double brute = offlineOptimalBruteForce(instance, penalty);
            double dp = offlineOptimalDp(instance, penalty);
            check(Math.abs(brute - dp) < 1e-9, "enumeration and DP agree on " + instance.name);
            RunResult adaptive = runAdaptive(instance, penalty, 0.15, 2, true);
            check(adaptive.totalCost >= brute - 1e-9, "the executor never beats the offline optimum");
        }
        check(offlineOptimalDp(new Instance("empty", new Stage[0], alternatives, 100), penalty) == 0.0,
                "the empty chain has zero optimum");
        check(offlineOptimalBruteForce(new Instance("empty", new Stage[0], alternatives, 100), penalty) == 0.0,
                "exhaustive enumeration handles the empty chain");

        // --- 5. on a collapsing workload the adaptive executor beats the static plan and stays near optimal.
        RunResult adaptive = runAdaptive(collapsing, penalty, 0.15, 2, true);
        double optimum = offlineOptimalDp(collapsing, penalty);
        System.out.printf("collapse: static=%.2f adaptive=%.2f optimum=%.2f switches=%d ratio=%.3f%n",
                staticRun.totalCost, adaptive.totalCost, optimum, adaptive.switches, adaptive.totalCost / optimum);
        check(adaptive.switches > 0, "the collapsing workload triggers a re-plan");
        check(adaptive.stageCost < staticRun.stageCost, "adaptive stage work is lower than the static plan");
        check(adaptive.totalCost < staticRun.totalCost, "adaptive is cheaper than static end to end");
        check(adaptive.totalCost <= optimum * 1.10, "the collapsing run is within 10% of the optimum");
        check(adaptive.planByStage[0] == staticRun.planByStage[0],
                "the first stage still uses the plan the optimizer picked");
        check(adaptive.planByStage[adaptive.planByStage.length - 1] != adaptive.planByStage[0],
                "the final stage uses a different operator than the first");

        // --- 6. with perfect estimates nothing is re-planned.
        Instance steady = steady("steady", 8, 7L, 400);
        RunResult steadyRun = runAdaptive(steady, penalty, 0.15, 2, true);
        RunResult steadyStatic = runStatic(steady, penalty);
        check(steadyRun.switches == 0, "perfect estimates produce no re-planning");
        check(Math.abs(steadyRun.totalCost - steadyStatic.totalCost) < 1e-9,
                "with perfect estimates the executor reproduces the static plan");
        check(Math.abs(steadyRun.totalCost - offlineOptimalDp(steady, penalty)) < 1e-9,
                "the static plan is optimal when the estimates are exact");

        // --- 7. the stabilisation window bounds switches; without it the same workload thrashes and costs more.
        Instance oscillating = oscillating("oscillating", 12, 856_777L, 32, 6.0);
        RunResult stabilised = runAdaptive(oscillating, penalty, 0.15, 2, true);
        RunResult thrashing = runAdaptive(oscillating, penalty, 0.0, 1, false);
        double oscillatingOptimum = offlineOptimalDp(oscillating, penalty);
        System.out.printf("oscillating: stabilisedSwitches=%d thrashingSwitches=%d stabilisedCost=%.2f "
                        + "thrashingCost=%.2f optimum=%.2f%n", stabilised.switches, thrashing.switches,
                stabilised.totalCost, thrashing.totalCost, oscillatingOptimum);
        check(thrashing.switches > stabilised.switches, "without stabilisation the executor thrashes");
        check(stabilised.switches <= oscillating.stages.length / 2, "the stabilisation window bounds switches");
        check(thrashing.totalCost > stabilised.totalCost, "thrashing pays more penalties than it saves");
        check(stabilised.totalCost <= oscillatingOptimum * 1.35 + 1e-9, "the stabilised run stays near optimal");
        check(stabilised.totalCost >= oscillatingOptimum - 1e-9, "the stabilised run respects the lower bound");

        // --- 8. the account is exact and the plan log replays.
        List<Instance> owners = List.of(collapsing, collapsing, oscillating, oscillating, steady);
        List<RunResult> runs = List.of(staticRun, adaptive, stabilised, thrashing, steadyRun);
        for (int i = 0; i < runs.size(); i++) {
            RunResult run = runs.get(i);
            check(Math.abs(run.stageCost + run.switches * run.penalty - run.totalCost) < 1e-9,
                    "switch penalty accounting is exact (" + owners.get(i).name + ")");
            check(Math.abs(run.stageCost - replayStageCost(owners.get(i), run.planByStage)) < 1e-9,
                    "the plan log replays to the reported stage cost (" + owners.get(i).name + ")");
            check(run.switches == countChanges(run.planByStage), "the switch count matches the plan log");
        }

        // --- 9. a prohibitive penalty degenerates to static; the cost is non-decreasing in the penalty.
        RunResult hugePenalty = runAdaptive(collapsing, 1e6, 0.15, 2, true);
        check(hugePenalty.switches == 0, "a prohibitive switch penalty disables re-planning");
        check(Math.abs(hugePenalty.totalCost - staticRun.totalCost) < 1e-9,
                "the degenerated run equals the static plan exactly");
        double previous = -1;
        for (double p : new double[]{0.0, 5.0, 15.0, 100.0, 5000.0}) {
            RunResult run = runAdaptive(collapsing, p, 0.15, 2, true);
            check(run.totalCost >= previous - 1e-9, "total cost is non-decreasing in the penalty (p=" + p + ")");
            check(run.switches <= collapsing.stages.length / 2, "switch count stays bounded for p=" + p);
            previous = run.totalCost;
        }
        RunResult freeSwitch = runAdaptive(collapsing, 0.0, 0.0, 0, false);
        check(freeSwitch.totalCost <= adaptive.totalCost + 1e-9, "a free switch cannot cost more than a paid one");

        // --- 10. robustness across misestimation ratios.
        for (double ratio : new double[]{2.0, 5.0, 20.0}) {
            Instance noisy = jittered("noisy-" + ratio, 10, 856_1234L, ratio, 400);
            RunResult noisyRun = runAdaptive(noisy, penalty, 0.15, 2, true);
            double noisyOptimum = offlineOptimalDp(noisy, penalty);
            System.out.printf("jitter x%.1f: adaptive=%.2f optimum=%.2f ratio=%.3f switches=%d%n",
                    ratio, noisyRun.totalCost, noisyOptimum, noisyRun.totalCost / noisyOptimum, noisyRun.switches);
            check(noisyRun.totalCost >= noisyOptimum - 1e-9, "noisy run respects the lower bound");
            check(noisyRun.totalCost <= noisyOptimum * 1.35 + 1e-9, "noisy run stays within 1.35x");
            check(noisyRun.totalCost <= runStatic(noisy, penalty).totalCost * 1.35 + 1e-9,
                    "noisy run is no worse than 1.35x the static plan");
        }

        // --- 11. determinism and seed sensitivity.
        Instance same = collapse("collapse", 12, 856_2024L, 400);
        RunResult again = runAdaptive(same, penalty, 0.15, 2, true);
        check(Arrays.equals(again.planByStage, adaptive.planByStage) && again.switches == adaptive.switches,
                "the same instance reproduces the same decisions");
        Instance different = collapse("collapse", 12, 856_2025L, 400);
        RunResult differentRun = runAdaptive(different, penalty, 0.15, 2, true);
        check(differentRun.totalCost != adaptive.totalCost, "a different seed produces a different workload");
        check(differentRun.totalCost <= offlineOptimalDp(different, penalty) * 1.10 + 1e-9,
                "the different instance still stays near its optimum");

        // --- 12. degenerate shapes: single stage, empty chain, single alternative.
        Instance singleStage = new Instance("single", new Stage[]{new Stage(0.5, 0.9)}, alternatives, 400);
        RunResult singleRun = runAdaptive(singleStage, penalty, 0.15, 2, true);
        check(singleRun.switches == 0 && singleRun.planByStage.length == 1,
                "a single-stage query has nothing to re-plan");
        check(Math.abs(singleRun.stageCost - alternatives.get(singleRun.planByStage[0]).cost(400)) < 1e-9,
                "the single stage pays its own cost");
        Instance none = new Instance("none", new Stage[0], alternatives, 400);
        RunResult noneRun = runAdaptive(none, penalty, 0.15, 2, true);
        check(noneRun.totalCost == 0 && noneRun.stageCost == 0 && noneRun.switches == 0,
                "an empty plan chain costs nothing");
        Instance oneAlt = new Instance("oneAlt", new Stage[]{new Stage(0.5, 0.02), new Stage(0.5, 0.9)},
                List.of(alternatives.get(0)), 400);
        RunResult oneAltRun = runAdaptive(oneAlt, penalty, 0.15, 2, true);
        check(oneAltRun.switches == 0, "with a single alternative no switch is possible");
        check(Math.abs(oneAltRun.stageCost - replayStageCost(oneAlt, oneAltRun.planByStage)) < 1e-9,
                "the single-alternative run replays");
        check(alternatives.get(0).name.equals("HASH") && alternatives.size() == 3, "the alternative set is stable");

        System.out.println("total checks=" + checks);
        System.out.println("PASS");
    }
}
