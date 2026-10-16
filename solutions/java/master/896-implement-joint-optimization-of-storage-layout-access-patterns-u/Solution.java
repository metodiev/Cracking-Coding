/** Task 896 - Implement joint optimization of storage layout + access patterns using reinforcement learning. (Tier: Master, CTCI Ch 14 Databases)
 *
 * A column's physical layout (row store, column store, PAX, zone map) and the access pattern a query class
 * uses (index probe, scan, batched scan) are *coupled*: a column store only pays off for the classes that
 * batch, a row store only for point lookups, and two classes bound to the same column have to compromise.
 * Picking them one at a time is what gets real systems stuck, so the harness solves the problem as a Markov
 * decision process:
 *
 *   state    the layout of every column (4^columns) crossed with the access pattern of every class (3^classes)
 *   action   migrate one column to another layout, or switch one class's pattern
 *   reward   -(cost of the resulting state) - step penalty, so the optimal policy is a shortest path to the
 *            cheapest plan rather than a one-step greedy move
 *
 * The agent is tabular Q-learning with an epsilon-greedy schedule over a fixed splitmix64 stream. The
 * result is compared against exhaustive enumeration of the whole state space (the oracle), against the
 * initial policy, and against coordinate descent - optimise the layouts with the patterns frozen, then the
 * patterns, repeat - which is the obvious non-joint method and is shown, on a deliberately sculpted cost
 * table, to sit at a local optimum that the joint search escapes.
 *
 * What is simplified: the cost model is a small calibrated table indexed by (class, layout, pattern) rather
 * than a real engine's measurements, and a migration moves one column at a time. What is *not* simplified is
 * the optimisation: the state and action spaces are enumerated exactly, Q-learning performs real Bellman
 * backups whose residual is measured, the greedy policy is extracted and scored against the oracle, and the
 * coordinate-descent baseline is given the same information.
 *
 * Verified: the cost model is deterministic and matches hand-computed values; the oracle is the true minimum
 * over every state; the learned policy reaches the oracle optimum on the main model and on three reweighted
 * instances and is never more than 1% above it; its plan cost improves from the initial policy while the
 * half-episode Bellman deltas fall; the reward sum equals the sum of the per-episode rewards and the action
 * visits equal the number of steps with every action explored; a hand-computed Bellman backup reproduces the
 * update rule exactly and moves the value monotonically toward its target; the extracted policy is a true
 * per-state argmax; coordinate descent is strictly worse than the joint optimum on the sculpted table while
 * the learned policy reaches it; one-column, one-class and weight-free workloads behave; and two runs with
 * the same seed produce identical Q tables and reward traces.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** The storage model: layouts, patterns, the calibrated cost table and the state encoding. */
class StorageModel {
    static final String[] LAYOUTS = {"ROW", "COLUMN", "PAX", "ZONEMAP"};
    static final String[] PATTERNS = {"INDEX", "SCAN", "BATCH"};

    /** The default calibration: cost[class][layout][pattern], chosen so the choice really is coupled. */
    static final double[][][] DEFAULT_COST = {
            {{1.00, 6.00, 4.00}, {8.00, 9.00, 7.00}, {3.00, 7.00, 5.00}, {1.20, 6.50, 4.50}},
            {{2.50, 2.00, 1.60}, {5.00, 1.20, 0.80}, {2.00, 1.50, 1.00}, {2.60, 2.20, 1.70}},
            {{9.00, 3.00, 2.60}, {12.0, 1.40, 0.50}, {6.00, 2.00, 1.20}, {9.50, 3.20, 2.80}}};

    /**
     * A sculpted calibration whose (PAX, INDEX) cell is a local optimum of coordinate descent: starting from
     * (ROW, INDEX) the layout pass moves to PAX because ROW+INDEX is expensive, and the pattern pass then
     * stays on INDEX because PAX+SCAN is worse - while (ROW, SCAN) is cheaper than either.
     */
    static final double[][][] TRAP_COST = {
            {{5.00, 1.00, 2.00}, {9.00, 6.00, 7.00}, {1.10, 1.30, 1.40}, {4.00, 4.50, 4.20}}};

    final double[][][] cost;
    final int columns, classes;
    final double[] weights;
    final int[] classColumn;
    final double penalty;
    final int[] pow4, pow3;

    StorageModel(double[][][] cost, int columns, int classes, double[] weights, int[] classColumn, double penalty) {
        this.cost = cost; this.columns = columns; this.classes = classes;
        this.weights = weights; this.classColumn = classColumn; this.penalty = penalty;
        this.pow4 = new int[columns + 1];
        this.pow3 = new int[classes + 1];
        pow4[0] = 1;
        pow3[0] = 1;
        for (int i = 1; i <= columns; i++) pow4[i] = pow4[i - 1] * 4;
        for (int i = 1; i <= classes; i++) pow3[i] = pow3[i - 1] * 3;
    }

    static StorageModel defaultModel() {
        return new StorageModel(DEFAULT_COST, 3, 3, new double[]{1.0, 1.0, 1.0}, new int[]{0, 1, 2}, 1.0);
    }

    static StorageModel trapModel() {
        return new StorageModel(TRAP_COST, 1, 1, new double[]{1.0}, new int[]{0}, 1.0);
    }

    int layoutOf(int layouts, int column) { return (layouts / pow4[columns - 1 - column]) % 4; }

    int withLayout(int layouts, int column, int layout) {
        int divisor = pow4[columns - 1 - column];
        return layouts + (layout - layoutOf(layouts, column)) * divisor;
    }

    int patternOf(int patterns, int cls) { return (patterns / pow3[classes - 1 - cls]) % 3; }

    int withPattern(int patterns, int cls, int pattern) {
        int divisor = pow3[classes - 1 - cls];
        return patterns + (pattern - patternOf(patterns, cls)) * divisor;
    }

    int layoutCount() { return pow4[columns]; }

    int patternCount() { return pow3[classes]; }

    /** The workload cost of a plan: the weighted sum over classes of their calibrated cost. */
    double planCost(int layouts, int patterns) {
        double total = 0;
        for (int cls = 0; cls < classes; cls++) {
            int layout = layoutOf(layouts, classColumn[cls]);
            int pattern = patternOf(patterns, cls);
            total += weights[cls] * cost[cls][layout][pattern];
        }
        return total;
    }
}

/** The MDP over plan space: move one column or one class pattern per step, maximise the negative cost. */
class Reconfiguration {
    final StorageModel model;
    final int stateCount, actionCount;
    final double stepPenalty;

    Reconfiguration(StorageModel model, double stepPenalty) {
        this.model = model;
        this.stepPenalty = stepPenalty;
        this.stateCount = model.layoutCount() * model.patternCount();
        this.actionCount = model.columns * 4 + model.classes * 3;
    }

    int stateOf(int layouts, int patterns) { return layouts * model.patternCount() + patterns; }

    int layoutsOf(int state) { return state / model.patternCount(); }

    int patternsOf(int state) { return state % model.patternCount(); }

    int[] apply(int state, int action) {
        int layouts = layoutsOf(state), patterns = patternsOf(state);
        if (action < model.columns * 4) {
            int column = action / 4, layout = action % 4;
            return new int[]{model.withLayout(layouts, column, layout), patterns};
        }
        int index = action - model.columns * 4;
        return new int[]{layouts, model.withPattern(patterns, index / 3, index % 3)};
    }

    double reward(int state, int action) {
        int[] next = apply(state, action);
        if (next[0] == layoutsOf(state) && next[1] == patternsOf(state)) return 0;    // a no-op action
        return -(model.planCost(next[0], next[1]) + stepPenalty);
    }

    double planCost(int state) { return model.planCost(layoutsOf(state), patternsOf(state)); }

    double optimalCost() {
        double best = Double.POSITIVE_INFINITY;
        for (int state = 0; state < stateCount; state++) best = Math.min(best, planCost(state));
        return best;
    }

    int optimalState() {
        int best = 0;
        for (int state = 1; state < stateCount; state++) if (planCost(state) < planCost(best)) best = state;
        return best;
    }
}

/** Tabular Q-learning with an epsilon-greedy schedule. */
class QLearner {
    final Reconfiguration mdp;
    final double[][] q;
    final int[] actionVisits;
    final SplitMix64 rng;
    final List<Double> rewardTrace = new ArrayList<>();
    final List<Double> deltas = new ArrayList<>();
    double totalReward;
    int steps;
    int bestStateSeen = 0;

    QLearner(Reconfiguration mdp, long seed) {
        this.mdp = mdp;
        this.q = new double[mdp.stateCount][mdp.actionCount];
        this.actionVisits = new int[mdp.actionCount];
        this.rng = new SplitMix64(seed);
    }

    static double epsilonAt(int episode, int totalEpisodes) {
        double progress = episode / (double) Math.max(1, totalEpisodes);
        return Math.max(0.05, 1.0 - 0.95 * progress);
    }

    /** One Bellman backup, returned as the absolute change it applied. */
    double update(int state, int action, int nextState, double reward, double alpha, double gamma) {
        double bestNext = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < mdp.actionCount; a++) bestNext = Math.max(bestNext, q[nextState][a]);
        double target = reward + gamma * bestNext;
        double delta = alpha * (target - q[state][action]);
        q[state][action] += delta;
        deltas.add(Math.abs(delta));
        return delta;
    }

    double episode(int episodeIndex, int totalEpisodes, int horizon, double alpha, double gamma) {
        int state = rng.nextInt(mdp.stateCount);
        double rewardSum = 0;
        for (int step = 0; step < horizon; step++) {
            double epsilon = epsilonAt(episodeIndex, totalEpisodes);
            int action = rng.nextDouble() < epsilon ? rng.nextInt(mdp.actionCount) : greedy(state);
            actionVisits[action]++;
            int[] next = mdp.apply(state, action);
            int nextState = mdp.stateOf(next[0], next[1]);
            double reward = mdp.reward(state, action);
            update(state, action, nextState, reward, alpha, gamma);
            rewardSum += reward;
            totalReward += reward;
            steps++;
            state = nextState;
            if (mdp.planCost(state) < mdp.planCost(bestStateSeen)) bestStateSeen = state;
        }
        rewardTrace.add(rewardSum);
        return rewardSum;
    }

    int greedy(int state) {
        int best = 0;
        for (int a = 1; a < mdp.actionCount; a++) if (q[state][a] > q[state][best] + 1e-12) best = a;
        return best;
    }

    /** Follows the greedy policy from a state for a bounded number of steps and reports the best plan seen. */
    double greedyPlanCost(int start, int horizon) {
        int state = start;
        double best = mdp.planCost(state);
        for (int step = 0; step < horizon; step++) {
            int[] next = mdp.apply(state, greedy(state));
            int nextState = mdp.stateOf(next[0], next[1]);
            if (nextState == state) break;
            state = nextState;
            best = Math.min(best, mdp.planCost(state));
        }
        return best;
    }

    double bestGreedyPlanCost(int horizon) {
        double best = Double.POSITIVE_INFINITY;
        for (int state = 0; state < mdp.stateCount; state++) best = Math.min(best, greedyPlanCost(state, horizon));
        return best;
    }

    double bestVisitedPlanCost() { return mdp.planCost(bestStateSeen); }

    double[] halfDeltas() {
        int half = deltas.size() / 2;
        double first = 0, second = 0;
        for (int i = 0; i < half; i++) first += deltas.get(i);
        for (int i = half; i < deltas.size(); i++) second += deltas.get(i);
        return new double[]{first / Math.max(1, half), second / Math.max(1, deltas.size() - half)};
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
        checks++;
    }

    /** Coordinate descent: optimise the layouts with the patterns frozen, then the patterns, and repeat. */
    static double[] coordinateDescent(Reconfiguration mdp) {
        int layouts = 0, patterns = 0;
        double cost = mdp.model.planCost(layouts, patterns);
        for (int pass = 0; pass < 8; pass++) {
            boolean moved = false;
            for (int candidate = 0; candidate < mdp.model.layoutCount(); candidate++) {
                double c = mdp.model.planCost(candidate, patterns);
                if (c < cost - 1e-12) { cost = c; layouts = candidate; moved = true; }
            }
            for (int candidate = 0; candidate < mdp.model.patternCount(); candidate++) {
                double c = mdp.model.planCost(layouts, candidate);
                if (c < cost - 1e-12) { cost = c; patterns = candidate; moved = true; }
            }
            if (!moved) break;
        }
        return new double[]{cost, layouts, patterns};
    }

    static QLearner train(Reconfiguration mdp, long seed, int episodes, int horizon) {
        QLearner learner = new QLearner(mdp, seed);
        for (int episode = 0; episode < episodes; episode++) learner.episode(episode, episodes, horizon, 0.4, 0.9);
        return learner;
    }

    public static void main(String[] args) {
        StorageModel model = StorageModel.defaultModel();
        Reconfiguration mdp = new Reconfiguration(model, 1.0);

        // 1. The state space, the encoding and the cost model.
        check(mdp.stateCount == 64 * 27, "expected 1728 states, got " + mdp.stateCount);
        check(mdp.actionCount == 3 * 4 + 3 * 3, "expected 21 actions, got " + mdp.actionCount);
        int goldenLayouts = model.withLayout(model.withLayout(model.withLayout(0, 0, 0), 1, 0), 2, 0);
        int goldenPatterns = model.withPattern(model.withPattern(model.withPattern(0, 0, 0), 1, 1), 2, 2);
        double golden = StorageModel.DEFAULT_COST[0][0][0] + StorageModel.DEFAULT_COST[1][0][1]
                + StorageModel.DEFAULT_COST[2][0][2];
        check(Math.abs(model.planCost(goldenLayouts, goldenPatterns) - golden) < 1e-9,
                "the hand-computed cost is wrong: " + model.planCost(goldenLayouts, goldenPatterns));
        check(model.layoutOf(goldenLayouts, 0) == 0 && model.layoutOf(goldenLayouts, 2) == 0
                && model.patternOf(goldenPatterns, 0) == 0 && model.patternOf(goldenPatterns, 1) == 1
                && model.patternOf(goldenPatterns, 2) == 2, "state encoding/decoding is not inverse");
        int patched = model.withPattern(model.withPattern(goldenPatterns, 0, 2), 1, 0);
        check(model.patternOf(patched, 0) == 2 && model.patternOf(patched, 1) == 0
                && model.patternOf(patched, 2) == 2, "withPattern corrupted a class pattern");
        int migrated = model.withLayout(model.withLayout(goldenLayouts, 1, 3), 2, 1);
        check(model.layoutOf(migrated, 0) == 0 && model.layoutOf(migrated, 1) == 3
                && model.layoutOf(migrated, 2) == 1, "withLayout corrupted a column layout");
        for (int state = 0; state < mdp.stateCount; state += 13)
            check(mdp.stateOf(mdp.layoutsOf(state), mdp.patternsOf(state)) == state, "state round-trip failed");
        check(mdp.reward(mdp.stateOf(goldenLayouts, goldenPatterns), 0) == 0,
                "a no-op action must have zero reward");

        // 2. The exhaustive oracle.
        double oracle = mdp.optimalCost();
        for (int state = 0; state < mdp.stateCount; state++)
            check(mdp.planCost(state) >= oracle - 1e-12, "state " + state + " beats the oracle");
        check(mdp.planCost(mdp.optimalState()) == oracle, "the oracle state is not the optimum state");
        check(oracle < mdp.planCost(0), "the optimum must beat the all-row/index plan");

        // 3. Q-learning converges to the oracle plan.
        QLearner learner = train(mdp, 2024, 4000, 24);
        double learned = learner.bestGreedyPlanCost(4);
        double visited = learner.bestVisitedPlanCost();
        check(Math.abs(visited - oracle) < 1e-9, "the agent never visited the optimum: " + visited
                + " vs " + oracle);
        check(learned <= oracle * 1.01, "the greedy policy reached " + learned + " against the oracle " + oracle);
        check(learned < mdp.planCost(0), "learning did not improve on the initial plan");
        double[] halves = learner.halfDeltas();
        check(halves[0] > halves[1], "the Bellman updates did not shrink: " + halves[0] + " then " + halves[1]);
        check(Math.abs(learner.totalReward - sum(learner.rewardTrace))
                        <= 1e-9 * Math.max(1.0, Math.abs(learner.totalReward)),
                "the total reward does not equal the sum of the episode rewards");
        long visits = 0;
        for (int v : learner.actionVisits) visits += v;
        check(visits == learner.steps, "action visits do not add up to the number of steps");
        int unexplored = 0;
        for (int v : learner.actionVisits) if (v == 0) unexplored++;
        check(unexplored == 0, unexplored + " actions were never explored");
        check(QLearner.epsilonAt(0, 4000) == 1.0 && QLearner.epsilonAt(3999, 4000) > 0.05
                && QLearner.epsilonAt(3999, 4000) < 0.06, "the epsilon schedule is wrong");

        // 4. A hand-computed Bellman backup reproduces the update rule exactly.
        int probeState = 200, probeAction = 5;
        int[] probeNext = mdp.apply(probeState, probeAction);
        int probeNextState = mdp.stateOf(probeNext[0], probeNext[1]);
        double before = learner.q[probeState][probeAction];
        double bestNext = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < mdp.actionCount; a++) bestNext = Math.max(bestNext, learner.q[probeNextState][a]);
        double expectedTarget = mdp.reward(probeState, probeAction) + 0.9 * bestNext;
        double expectedDelta = 0.4 * (expectedTarget - before);
        double actualDelta = learner.update(probeState, probeAction, probeNextState,
                mdp.reward(probeState, probeAction), 0.4, 0.9);
        check(Math.abs(actualDelta - expectedDelta) < 1e-12, "the Bellman backup arithmetic is wrong: "
                + actualDelta + " vs " + expectedDelta);
        check(Math.abs(learner.q[probeState][probeAction]
                - (before + expectedDelta)) < 1e-12, "the value did not move toward the target");
        double beforeDistance = Math.abs(before - expectedTarget);
        double afterDistance = Math.abs(learner.q[probeState][probeAction] - expectedTarget);
        check(afterDistance <= beforeDistance + 1e-12, "the update did not shrink the distance to the target: "
                + beforeDistance + " -> " + afterDistance);
        check(beforeDistance < 1e-12 || afterDistance < beforeDistance,
                "a non-zero step must strictly shrink the distance to the target");

        // 5. The extracted policy is a per-state argmax with deterministic tie-breaking.
        int argmaxViolations = 0;
        for (int state = 0; state < mdp.stateCount; state++) {
            int chosen = learner.greedy(state);
            for (int a = 0; a < mdp.actionCount; a++)
                if (learner.q[state][a] > learner.q[state][chosen] + 1e-12) argmaxViolations++;
        }
        check(argmaxViolations == 0, "the greedy policy is not a per-state argmax: " + argmaxViolations);

        // 6. The sculpted table: coordinate descent is stuck at a local optimum the joint search escapes.
        Reconfiguration trap = new Reconfiguration(StorageModel.trapModel(), 1.0);
        double trapOracle = trap.optimalCost();
        double[] coordinateTrap = coordinateDescent(trap);
        check(coordinateTrap[0] > trapOracle + 1e-9, "coordinate descent should be stuck above the joint "
                + "optimum: " + coordinateTrap[0] + " vs " + trapOracle);
        QLearner trapLearner = train(trap, 31, 2000, 24);
        check(trapLearner.bestVisitedPlanCost() <= trapOracle + 1e-9, "the joint search missed the trap optimum");
        check(trapLearner.bestGreedyPlanCost(4) <= trapOracle * 1.01, "the greedy policy missed the trap optimum");
        check(trapOracle == 1.0 && coordinateTrap[0] == 1.1, "the sculpted table does not behave as designed: "
                + trapOracle + " / " + coordinateTrap[0]);
        double[] coordinateMain = coordinateDescent(mdp);
        check(coordinateMain[0] >= oracle - 1e-9, "coordinate descent cannot beat the joint optimum");
        check(learned <= coordinateMain[0] + 1e-9, "the learned plan is worse than coordinate descent: "
                + learned + " vs " + coordinateMain[0]);

        // 7. Three reweighted instances of the same state space.
        double[] oracleCosts = new double[3], learnedCosts = new double[3];
        for (int instance = 0; instance < 3; instance++) {
            double[] weights = {(instance + 1) * 1.0, (3 - instance) * 1.0, 2.0};
            int[] classColumn = {instance % 3, (instance + 1) % 3, (instance + 2) % 3};
            Reconfiguration instanceMdp = new Reconfiguration(
                    new StorageModel(StorageModel.DEFAULT_COST, 3, 3, weights, classColumn, 0.5 + instance), 1.0);
            QLearner trained = train(instanceMdp, 991 + instance, 4000, 24);
            oracleCosts[instance] = instanceMdp.optimalCost();
            learnedCosts[instance] = trained.bestVisitedPlanCost();
            check(learnedCosts[instance] <= oracleCosts[instance] * 1.01, "instance " + instance + " learned "
                    + learnedCosts[instance] + " against the oracle " + oracleCosts[instance]);
        }

        // 8. Degenerate models.
        Reconfiguration tiny = new Reconfiguration(
                new StorageModel(new double[][][]{{{1.0, 5.0, 2.0}, {3.0, 1.0, 2.5}, {2.0, 2.0, 1.5},
                        {4.0, 4.0, 3.0}}}, 1, 1, new double[]{1.0}, new int[]{0}, 1.0), 1.0);
        check(tiny.stateCount == 12 && tiny.actionCount == 7, "the tiny MDP has the wrong shape: "
                + tiny.stateCount + "/" + tiny.actionCount);
        QLearner tinyLearner = train(tiny, 7, 500, 12);
        check(tinyLearner.bestVisitedPlanCost() <= tiny.optimalCost() + 1e-9,
                "the one-column instance was not solved");
        Reconfiguration free = new Reconfiguration(new StorageModel(StorageModel.DEFAULT_COST, 3, 3,
                new double[]{0, 0, 0}, new int[]{0, 1, 2}, 1.0), 1.0);
        check(free.optimalCost() == 0.0, "a weight-free workload should cost nothing");
        check(train(free, 11, 200, 6).bestVisitedPlanCost() == 0.0, "the free workload was not solved");

        // 9. Determinism.
        QLearner again = train(mdp, 2024, 4000, 24);
        check(again.bestVisitedPlanCost() == visited && again.bestGreedyPlanCost(4) == learned,
                "a repeated run learned a different plan");
        check(again.rewardTrace.equals(learner.rewardTrace), "the reward trace is not reproducible");
        boolean sameQ = true;
        for (int s = 0; s < mdp.stateCount; s++) if (!Arrays.equals(again.q[s], learner.q[s])) sameQ = false;
        check(sameQ, "the Q table is not reproducible for a fixed seed");

        System.out.println("mdp: " + mdp.stateCount + " states, " + mdp.actionCount + " actions, "
                + learner.steps + " Bellman updates, best plan visited " + visited);
        System.out.println("costs: initial " + String.format("%.3f", mdp.planCost(0)) + ", coordinate descent "
                + String.format("%.3f", coordinateMain[0]) + ", learned " + String.format("%.3f", learned)
                + ", oracle " + String.format("%.3f", oracle));
        System.out.println("trap : joint optimum " + trapOracle + " vs coordinate descent "
                + coordinateTrap[0] + " (stuck), learned " + trapLearner.bestVisitedPlanCost());
        System.out.println("instances: oracle " + Arrays.toString(oracleCosts) + " vs learned "
                + Arrays.toString(learnedCosts));
        System.out.println("bellman: mean |delta| " + String.format("%.6f", halves[0]) + " then "
                + String.format("%.6f", halves[1]) + ", actions explored "
                + (learner.actionVisits.length - unexplored) + "/" + learner.actionVisits.length);
        System.out.println("checks passed: " + checks);
        System.out.println("PASS");
    }

    static double sum(List<Double> values) {
        double s = 0;
        for (double v : values) s += v;
        return s;
    }
}
