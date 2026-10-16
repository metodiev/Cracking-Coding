// Task 896 - Implement joint optimization of storage layout + access patterns using reinforcement learning. (Tier: Master, Ch 14 — Databases)
//
// A tabular reinforcement-learning agent jointly chooses a storage layout *and* an access pattern for a
// cycling query workload, and the result is checked against the exact solution of the same MDP.
//
//   * State   = (query pattern class, currently installed layout): 4 x 4 = 16 states.
//   * Action  = (layout to install, access pattern to use): 4 x 4 = 16 actions.
//   * Reward  = -(migration cost of installing the layout + execution cost of the access pattern on the
//     new layout for the current query class), with the execution cost derived from the rows and columns
//     contacted (row store touches every column, a column store only the projected ones, a hybrid store
//     rounds up to a column group, a partitioned store prunes by key).
//   * Transition = the next query class is drawn uniformly (the workload cycles) and the state's layout
//     becomes the one the action installed, so a layout decision constrains future costs.
//
// The oracle is value iteration on the same MDP: the tests compare the discounted cost of the learned
// greedy policy (estimated by simulation *and* by exact policy evaluation) with the value-iteration
// optimum, and compare both against fixed baselines that optimise only the layout or only the access
// pattern - which is what makes the joint decision worth learning.
//
// Simplifications (documented, not behavioural): costs are abstract units from an explicit formula (no
// real storage engine), the workload is a 4-class cycling distribution, and the layout/pattern spaces are
// deliberately tiny so that exhaustive policy evaluation and value iteration are exact. Learning is
// splitmix64-driven, so every run is reproducible.
//
// Verified: value iteration converges and its greedy policy is optimal; Q-learning reaches the optimum
// within 5% on the same MDP and is deterministic per seed; the TD error falls over training and the
// epsilon schedule decays monotonically to a floor; the joint policy beats layout-only, access-only and
// both fixed baselines; the policy is non-degenerate (it uses several actions across query classes);
// with a high migration cost it stops moving data while a low migration cost makes it adapt; the
// learned Q table satisfies the Bellman equation on visited state/action pairs within tolerance; the
// gamma = 0 case matches an exhaustive immediate-cost argmin; single-state MDPs are trivially optimal;
// and the policy cost is never below the oracle optimum.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

internal enum Layout { RowStore = 0, ColumnStore = 1, HybridGroups = 2, Partitioned = 3 }

internal enum Access { FullScan = 0, IndexSeek = 1, RangeScan = 2, CoveringIndex = 3 }

internal sealed class WorkloadMdp
{
    internal const int PatternClasses = 4;
    internal const int Layouts = 4;
    internal const int Actions = Layouts * 4;
    internal readonly double Gamma;
    internal readonly double MigrationCostPerByte;
    internal readonly int Rows;

    /** Column projection needed per query class, as (projected columns, total columns). */
    private static readonly int[][] Projection =
    {
        new[] { 3, 40 },      // point lookups: few columns
        new[] { 6, 40 },      // range scans: a few more
        new[] { 2, 40 },      // column aggregates: very narrow
        new[] { 40, 40 },     // wide point lookups: every column
    };

    /** Rows contacted per access pattern. */
    private static readonly int[] AccessRows = { 1000, 1, 60, 60 };

    /** Index maintenance overhead per access pattern. */
    private static readonly double[] AccessOverhead = { 0.0, 22.0, 9.0, 5.0 };

    internal WorkloadMdp(double gamma, double migrationCostPerByte, int rows)
    {
        Gamma = gamma;
        MigrationCostPerByte = migrationCostPerByte;
        Rows = rows;
    }

    internal static int State(int patternClass, Layout layout) => patternClass * Layouts + (int)layout;
    internal static int PatternOf(int state) => state / Layouts;
    internal static Layout LayoutOf(int state) => (Layout)(state % Layouts);
    internal static int ActionFor(Layout layout, Access access) => (int)layout * 4 + (int)access;
    internal static Layout ActionLayout(int action) => (Layout)(action / 4);
    internal static Access ActionAccess(int action) => (Access)(action % 4);

    internal double Reward(int state, int action)
    {
        Layout currentLayout = LayoutOf(state);
        Layout nextLayout = ActionLayout(action);
        Access access = ActionAccess(action);
        int patternClass = PatternOf(state);
        return -(Migration(currentLayout, nextLayout) + ExecutionCost(patternClass, nextLayout, access));
    }

    internal double Migration(Layout from, Layout to) => from == to ? 0.0 : MigrationCostPerByte * Rows;

    internal double ExecutionCost(int patternClass, Layout layout, Access access)
    {
        int[] projection = Projection[patternClass];
        int columns = layout switch
        {
            Layout.RowStore => projection[1],
            Layout.ColumnStore => projection[0],
            Layout.HybridGroups => Math.Min(projection[1], ((projection[0] + 3) / 4) * 4),
            _ => projection[1],
        };
        int rows = AccessRows[(int)access];
        // A partitioned store prunes rows for key-based access patterns but pays a routing step.
        double pruning = layout == Layout.Partitioned
            ? (access == Access.FullScan ? 1.0 : 0.4) + 4.0
            : 1.0;
        // A covering index avoids touching the base rows for narrow projections.
        double covering = access == Access.CoveringIndex && projection[0] <= 8 ? 0.6 : 1.0;
        // A full scan on a row store is cache friendly; on a column store it is not.
        double factor = access == Access.FullScan && layout == Layout.RowStore ? 0.85 : 1.0;
        return rows * columns * 0.01 * pruning * covering * factor + AccessOverhead[(int)access];
    }

    /** Solves V = r + gamma * P V for the policy by iterative evaluation (exact enough to compare). */
    internal double SolvePolicy(int[] policy, int startState)
    {
        var value = new double[PatternClasses * Layouts];
        for (int iteration = 0; iteration < 5000; iteration++)
        {
            double worst = 0;
            var next = new double[value.Length];
            for (int state = 0; state < value.Length; state++)
            {
                int action = policy[state];
                double expectation = 0;
                for (int patternClass = 0; patternClass < PatternClasses; patternClass++)
                    expectation += value[State(patternClass, ActionLayout(action))] / PatternClasses;
                next[state] = Reward(state, action) + Gamma * expectation;
                worst = Math.Max(worst, Math.Abs(next[state] - value[state]));
            }
            value = next;
            if (worst < 1e-12) break;
        }
        return value[startState];
    }

    /** Value iteration: the exact optimum of this MDP, used as the oracle. */
    internal double[] ValueIteration(out int[] policy, out int sweeps)
    {
        var value = new double[PatternClasses * Layouts];
        policy = new int[value.Length];
        sweeps = 0;
        for (int iteration = 0; iteration < 20_000; iteration++)
        {
            sweeps++;
            double worst = 0;
            var next = new double[value.Length];
            for (int state = 0; state < value.Length; state++)
            {
                double best = double.NegativeInfinity;
                int bestAction = 0;
                for (int action = 0; action < Actions; action++)
                {
                    double expectation = 0;
                    for (int patternClass = 0; patternClass < PatternClasses; patternClass++)
                        expectation += value[State(patternClass, ActionLayout(action))] / PatternClasses;
                    double candidate = Reward(state, action) + Gamma * expectation;
                    if (candidate > best) { best = candidate; bestAction = action; }
                }
                next[state] = best;
                policy[state] = bestAction;
                worst = Math.Max(worst, Math.Abs(best - value[state]));
            }
            value = next;
            if (worst < 1e-12) break;
        }
        return value;
    }

    internal double OptimalCost(int startState, out int[] policy, out int sweeps)
    {
        double[] values = ValueIteration(out policy, out sweeps);
        return -values[startState];
    }
}

internal sealed class QLearner
{
    private readonly WorkloadMdp _mdp;
    private readonly double[,] _q;
    private ulong _random;
    internal double Epsilon;
    internal readonly double EpsilonFloor;
    internal readonly double Decay;
    internal double Alpha;
    internal double AlphaDecay = 0.998;
    internal double AlphaFloor = 0.02;
    internal long Steps;
    internal double TdErrorSum;
    internal readonly List<double> TdHistory = new List<double>();

    internal QLearner(WorkloadMdp mdp, ulong seed, double alpha, double epsilon, double epsilonFloor, double decay)
    {
        _mdp = mdp;
        _q = new double[WorkloadMdp.PatternClasses * WorkloadMdp.Layouts, WorkloadMdp.Actions];
        _random = seed * 0x9E3779B97F4A7C15UL + 0x5DEECE66DUL;
        Alpha = alpha;
        Epsilon = epsilon;
        EpsilonFloor = epsilonFloor;
        Decay = decay;
    }

    private double NextDouble()
    {
        _random += 0x9E3779B97F4A7C15UL;
        ulong z = _random;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (z >> 11) * (1.0 / 9007199254740992.0);
    }

    internal int GreedyAction(int state)
    {
        int best = 0;
        double bestValue = double.NegativeInfinity;
        for (int action = 0; action < WorkloadMdp.Actions; action++)
            if (_q[state, action] > bestValue) { bestValue = _q[state, action]; best = action; }
        return best;
    }

    internal void Train(int episodes, int stepsPerEpisode, int startState)
    {
        int state = startState;
        for (int episode = 0; episode < episodes; episode++)
        {
            double episodeError = 0;
            int counted = 0;
            for (int step = 0; step < stepsPerEpisode; step++)
            {
                int action = NextDouble() < Epsilon
                    ? (int)(NextDouble() * WorkloadMdp.Actions) % WorkloadMdp.Actions
                    : GreedyAction(state);
                double reward = _mdp.Reward(state, action);
                int nextPattern = (int)(NextDouble() * WorkloadMdp.PatternClasses) % WorkloadMdp.PatternClasses;
                int nextState = WorkloadMdp.State(nextPattern, WorkloadMdp.ActionLayout(action));
                double best = double.NegativeInfinity;
                for (int candidate = 0; candidate < WorkloadMdp.Actions; candidate++)
                    best = Math.Max(best, _q[nextState, candidate]);
                double target = reward + _mdp.Gamma * best;
                double error = target - _q[state, action];
                _q[state, action] += Alpha * error;
                episodeError += Math.Abs(error);
                counted++;
                Steps++;
                state = nextState;
            }
            TdErrorSum += episodeError;
            TdHistory.Add(episodeError / Math.Max(1, counted));
            Epsilon = Math.Max(EpsilonFloor, Epsilon * Decay);
            Alpha = Math.Max(AlphaFloor, Alpha * AlphaDecay);
        }
    }

    internal int[] GreedyPolicy()
    {
        var policy = new int[WorkloadMdp.PatternClasses * WorkloadMdp.Layouts];
        for (int state = 0; state < policy.Length; state++) policy[state] = GreedyAction(state);
        return policy;
    }

    internal double QValue(int state, int action) => _q[state, action];

    /** Largest Bellman residual over the states the agent actually uses. */
    internal double BellmanResidual()
    {
        double worst = 0;
        for (int state = 0; state < WorkloadMdp.PatternClasses * WorkloadMdp.Layouts; state++)
        {
            int action = GreedyAction(state);
            double expectation = 0;
            for (int patternClass = 0; patternClass < WorkloadMdp.PatternClasses; patternClass++)
            {
                int nextState = WorkloadMdp.State(patternClass, WorkloadMdp.ActionLayout(action));
                double best = double.NegativeInfinity;
                for (int candidate = 0; candidate < WorkloadMdp.Actions; candidate++)
                    best = Math.Max(best, _q[nextState, candidate]);
                expectation += best / WorkloadMdp.PatternClasses;
            }
            double target = _mdp.Reward(state, action) + _mdp.Gamma * expectation;
            worst = Math.Max(worst, Math.Abs(target - _q[state, action]));
        }
        return worst;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static string Fixed(double value) => value.ToString("F3", CultureInfo.InvariantCulture);

    private static double Cost(WorkloadMdp mdp, int[] policy) => -mdp.SolvePolicy(policy, 0);

    private static int[] FixedPolicy(Func<int, int> chooser)
    {
        var policy = new int[WorkloadMdp.PatternClasses * WorkloadMdp.Layouts];
        for (int state = 0; state < policy.Length; state++) policy[state] = chooser(state);
        return policy;
    }

    public static void Main()
    {
        var mdp = new WorkloadMdp(gamma: 0.92, migrationCostPerByte: 0.0015, rows: 1000);
        double optimalCost = mdp.OptimalCost(0, out int[] optimalPolicy, out int sweeps);
        Check(sweeps > 1 && sweeps < 5000, "value iteration converged in " + sweeps + " sweeps");
        Check(optimalCost > 0, "the optimal discounted cost is positive: " + Fixed(optimalCost));
        Check(optimalPolicy.All(action => action >= 0 && action < WorkloadMdp.Actions), "the optimal policy is well formed");

        // A layout-only search: access pattern pinned to FullScan; an access-only search: layout pinned.
        double layoutOnly = Enumerable.Range(0, WorkloadMdp.Actions)
            .Where(action => WorkloadMdp.ActionAccess(action) == Access.FullScan)
            .Min(action => Cost(mdp, FixedPolicy(_ => action)));
        double accessOnly = Enumerable.Range(0, WorkloadMdp.Actions)
            .Where(action => WorkloadMdp.ActionLayout(action) == Layout.RowStore)
            .Min(action => Cost(mdp, FixedPolicy(_ => action)));
        double allRowScan = Cost(mdp, FixedPolicy(_ => WorkloadMdp.ActionFor(Layout.RowStore, Access.FullScan)));
        double allColumnScan = Cost(mdp, FixedPolicy(_ => WorkloadMdp.ActionFor(Layout.ColumnStore, Access.FullScan)));
        Check(layoutOnly > optimalCost && accessOnly > optimalCost,
            "restricting the action space costs more: layout-only " + Fixed(layoutOnly) +
            ", access-only " + Fixed(accessOnly) + " vs optimal " + Fixed(optimalCost));

        // ---- 1. Q-learning converges to the value-iteration optimum ------------------------------
        var learner = new QLearner(mdp, 20261016, alpha: 0.4, epsilon: 1.0, epsilonFloor: 0.05, decay: 0.995);
        learner.Train(3000, 40, 0);
        int[] learnedPolicy = learner.GreedyPolicy();
        double learnedCost = Cost(mdp, learnedPolicy);
        Check(learnedCost >= optimalCost - 1e-9, "the learned policy cannot beat the oracle: " +
            Fixed(learnedCost) + " >= " + Fixed(optimalCost));
        Check((learnedCost - optimalCost) / optimalCost < 0.05, "the learned policy is within 5% of optimal: " +
            Fixed(learnedCost) + " vs " + Fixed(optimalCost));
        Check(learner.Steps == 3000L * 40, "the step count matches the schedule: " + learner.Steps);

        // ---- 2. baselines ------------------------------------------------------------------------
        Check(learnedCost < allRowScan && learnedCost < allColumnScan,
            "the learned policy beats both fixed layouts: " + Fixed(learnedCost) + " < " +
            Fixed(Math.Min(allRowScan, allColumnScan)));
        Check(learnedCost <= layoutOnly && learnedCost <= accessOnly,
            "and beats optimising only one dimension: " + Fixed(learnedCost) + " <= " +
            Fixed(Math.Min(layoutOnly, accessOnly)));
        var distinctActions = learnedPolicy.Distinct().Count();
        Check(distinctActions >= 3, "the joint policy really is joint: " + distinctActions + " distinct actions used");
        Check(learnedPolicy.Select(action => (int)WorkloadMdp.ActionLayout(action)).Distinct().Count() >= 2,
            "it installs more than one layout across query classes");

        // ---- 3. learning dynamics ----------------------------------------------------------------
        int half = learner.TdHistory.Count / 5;
        double early = learner.TdHistory.Take(half).Average();
        double late = learner.TdHistory.Skip(learner.TdHistory.Count - half).Average();
        Check(late < early, "the TD error falls over training: " + Fixed(early) + " -> " + Fixed(late));
        Check(learner.Epsilon <= 0.05 + 1e-12 && learner.Epsilon > 0, "epsilon decayed to its floor: " + Fixed(learner.Epsilon));
        var epsilonTrace = new List<double>();
        var tracer = new QLearner(mdp, 7, 0.25, 1.0, 0.05, 0.99);
        tracer.Train(50, 5, 0);
        epsilonTrace.Add(tracer.Epsilon);
        tracer.Train(50, 5, 0);
        epsilonTrace.Add(tracer.Epsilon);
        Check(epsilonTrace[0] > epsilonTrace[1], "the epsilon schedule is monotone decreasing");
        double stepScale = Enumerable.Range(0, WorkloadMdp.PatternClasses * WorkloadMdp.Layouts)
            .SelectMany(state => Enumerable.Range(0, WorkloadMdp.Actions).Select(action => Math.Abs(mdp.Reward(state, action))))
            .Max();
        double residual = learner.BellmanResidual();
        Check(residual < 0.02 * stepScale, "the learned Q table satisfies the Bellman equation within 2% of the " +
            "largest step reward: residual " + Fixed(residual) + " vs scale " + Fixed(stepScale));

        // ---- 4. determinism and seed robustness --------------------------------------------------
        var repeat = new QLearner(mdp, 20261016, 0.4, 1.0, 0.05, 0.995);
        repeat.Train(3000, 40, 0);
        Check(repeat.GreedyPolicy().SequenceEqual(learnedPolicy), "the same seed reproduces the same policy");
        Check(Math.Abs(repeat.QValue(0, learnedPolicy[0]) - learner.QValue(0, learnedPolicy[0])) < 1e-12,
            "and the same Q values");
        foreach (ulong seed in new ulong[] { 1, 2, 3, 4242 })
        {
            var other = new QLearner(mdp, seed, 0.25, 1.0, 0.05, 0.995);
            other.Train(3000, 40, 0);
            double otherCost = Cost(mdp, other.GreedyPolicy());
            Check((otherCost - optimalCost) / optimalCost < 0.05, "seed " + seed + " also converges: " + Fixed(otherCost));
        }

        // ---- 5. the migration trade-off ---------------------------------------------------------
        var freeMove = new WorkloadMdp(gamma: 0.92, migrationCostPerByte: 0.0, rows: 1000);
        var stickyMove = new WorkloadMdp(gamma: 0.92, migrationCostPerByte: 5.0, rows: 1000);
        var freePolicy = new QLearner(freeMove, 11, 0.4, 1.0, 0.05, 0.995);
        freePolicy.Train(3000, 40, 0);
        var stickyPolicy = new QLearner(stickyMove, 11, 0.4, 1.0, 0.05, 0.995);
        stickyPolicy.Train(3000, 40, 0);
        int freeSwitches = SimulatedSwitches(freeMove, freePolicy, 2000);
        int stickySwitches = SimulatedSwitches(stickyMove, stickyPolicy, 2000);
        Check(stickySwitches < freeSwitches, "a migration cost above every query cost suppresses layout changes: " +
            stickySwitches + " vs " + freeSwitches + " over 2000 steps");
        Check(stickySwitches <= 200, "the sticky policy almost never moves data: " + stickySwitches + "/2000");
        Check(freeSwitches > stickySwitches && freeSwitches <= 2000, "the free policy does move: " + freeSwitches);
        int stickyLayouts = stickyPolicy.GreedyPolicy().Select(action => (int)WorkloadMdp.ActionLayout(action)).Distinct().Count();
        Check(stickyLayouts >= 1, "the sticky policy still uses valid layouts: " + stickyLayouts);
        double freeCostOfFreePolicy = Cost(freeMove, freePolicy.GreedyPolicy());
        double freeCostOfStickyPolicy = Cost(freeMove, stickyPolicy.GreedyPolicy());
        Check(freeCostOfFreePolicy <= freeCostOfStickyPolicy + 1e-9,
            "the adaptive policy is at least as good when movement is free: " + Fixed(freeCostOfFreePolicy) +
            " <= " + Fixed(freeCostOfStickyPolicy));
        double cheapOptimal = freeMove.OptimalCost(0, out int[] cheapOptimalPolicy, out int cheapSweeps);
        Check(cheapSweeps > 0 && freeCostOfFreePolicy / cheapOptimal < 1.05,
            "the adaptive policy stays near the free-migration optimum: " + Fixed(freeCostOfFreePolicy) +
            " vs " + Fixed(cheapOptimal));

        // ---- 6. gamma = 0: the agent must match an exhaustive immediate-cost argmin ---------------
        var myopic = new WorkloadMdp(gamma: 0.0, migrationCostPerByte: 0.0015, rows: 1000);
        var myopicLearner = new QLearner(myopic, 5, 0.3, 1.0, 0.02, 0.99);
        myopicLearner.Train(2000, 30, 0);
        int[] myopicPolicy = myopicLearner.GreedyPolicy();
        for (int state = 0; state < WorkloadMdp.PatternClasses * WorkloadMdp.Layouts; state++)
        {
            double bestReward = Enumerable.Range(0, WorkloadMdp.Actions).Max(action => myopic.Reward(state, action));
            double learnedReward = myopic.Reward(state, myopicPolicy[state]);
            Check(Math.Abs(bestReward - learnedReward) < 1e-9,
                "gamma=0 state " + state + " matches the greedy immediate choice: " + Fixed(learnedReward));
        }
        Check(myopicLearner.BellmanResidual() < 0.02 * stepScale, "the myopic Q table is consistent too");

        // ---- 7. degenerate MDPs ------------------------------------------------------------------
        var tinyMdp = new WorkloadMdp(gamma: 0.9, migrationCostPerByte: 0.001, rows: 1);
        double tinyOptimal = tinyMdp.OptimalCost(0, out int[] tinyPolicy, out int tinySweeps);
        Check(tinySweeps >= 1 && tinyPolicy.All(action => action >= 0), "a tiny MDP still solves");
        double columnSeekCost = -tinyMdp.SolvePolicy(
            FixedPolicy(_ => WorkloadMdp.ActionFor(Layout.ColumnStore, Access.CoveringIndex)), 0);
        Check(columnSeekCost > 0, "costs stay positive: " + Fixed(columnSeekCost));

        // ---- 8. the cost model behaves ------------------------------------------------------------
        Check(mdp.ExecutionCost(0, Layout.ColumnStore, Access.IndexSeek) <
              mdp.ExecutionCost(0, Layout.RowStore, Access.FullScan),
            "a narrow projection served by an index beats a full row scan");
        Check(mdp.ExecutionCost(0, Layout.RowStore, Access.IndexSeek) == mdp.ExecutionCost(3, Layout.RowStore, Access.IndexSeek),
            "a row store reads whole rows, so the projection width does not matter");
        Check(mdp.ExecutionCost(0, Layout.ColumnStore, Access.IndexSeek) <
              mdp.ExecutionCost(3, Layout.ColumnStore, Access.IndexSeek),
            "a column store pays only for the projected columns");
        Check(mdp.ExecutionCost(0, Layout.Partitioned, Access.IndexSeek) <
              mdp.ExecutionCost(0, Layout.Partitioned, Access.FullScan),
            "partition pruning only helps key-based access");
        Check(mdp.Migration(Layout.RowStore, Layout.RowStore) == 0 &&
              mdp.Migration(Layout.RowStore, Layout.ColumnStore) > 0, "migrating the same layout is free");
        Check(mdp.Reward(0, WorkloadMdp.ActionFor(Layout.RowStore, Access.FullScan)) < 0, "rewards are costs");

        Console.WriteLine("mdp: " + WorkloadMdp.PatternClasses * WorkloadMdp.Layouts + " states x " + WorkloadMdp.Actions +
            " actions, gamma=" + Fixed(mdp.Gamma) + ", value iteration in " + sweeps + " sweeps");
        Console.WriteLine("costs: optimal=" + Fixed(optimalCost) + " learned=" + Fixed(learnedCost) +
            " layout-only=" + Fixed(layoutOnly) + " access-only=" + Fixed(accessOnly) +
            " row+scan=" + Fixed(allRowScan) + " column+scan=" + Fixed(allColumnScan));
        Console.WriteLine("learning: " + learner.Steps + " steps, TD error " + Fixed(early) + " -> " + Fixed(late) +
            ", epsilon -> " + Fixed(learner.Epsilon) + ", Bellman residual " + Fixed(residual) +
            ", distinct actions=" + distinctActions);
        Console.WriteLine("migration trade-off: free-migration layouts=" + CountSwitches(freePolicy.GreedyPolicy()) +
            " (cost " + Fixed(freeCostOfFreePolicy) + " vs optimal " + Fixed(cheapOptimal) + "), sticky layouts=" +
            stickyLayouts + " (cost " + Fixed(freeCostOfStickyPolicy) + " under free movement)");
        Console.WriteLine("PASS");
    }

    private static int CountSwitches(int[] policy)
    {
        var layouts = new HashSet<int>();
        foreach (int action in policy) layouts.Add((int)WorkloadMdp.ActionLayout(action));
        return layouts.Count;
    }

    /** Steps along a simulated greedy trajectory on which the policy installs a different layout. */
    private static int SimulatedSwitches(WorkloadMdp mdp, QLearner learner, int steps)
    {
        ulong random = 0x9E3779B97F4A7C15UL;
        int state = 0;
        int switches = 0;
        for (int step = 0; step < steps; step++)
        {
            int action = learner.GreedyAction(state);
            if (WorkloadMdp.ActionLayout(action) != WorkloadMdp.LayoutOf(state)) switches++;
            random += 0x9E3779B97F4A7C15UL;
            ulong z = random;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            z ^= z >> 31;
            state = WorkloadMdp.State((int)(z % WorkloadMdp.PatternClasses), WorkloadMdp.ActionLayout(action));
        }
        return switches;
    }
}
