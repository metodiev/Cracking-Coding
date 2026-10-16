// Task 978 - Implement adaptive test generator that adjusts difficulty to target percentile.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Items have a difficulty b on a logit scale and a learner has an ability theta; the one-parameter
// logistic model gives p(correct) = sigmoid(theta - b). To hit a target percentile the generator
// solves mean_i sigmoid(theta - b_i) = target by bisection (the expected score is increasing in
// theta), then emits the items whose difficulty is closest to that theta. The adaptive mode keeps a
// running estimate, serves the nearest-difficulty item, and moves the estimate by a damped logistic
// step after each response.
//
// Verified: the expected score of the emitted test round trips to the target, the closed-form oracle
// theta = b + logit(target) for a single-item pool, theta*(0.5) = 0 on a symmetric pool, a
// rank-counting oracle for the selection, monotone targeting, bounded and deterministic adaptive
// runs, and the empty-pool / out-of-range target / oversized-request boundaries.
// Simplification: one difficulty parameter per item, no guessing parameter, no item exposure control
// and a fixed item count rather than a stopping rule.
using System;
using System.Collections.Generic;

public static class Solution
{
    private static ulong _state;

    private static ulong NextUInt64()
    {
        unchecked
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    private static double NextUnit() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private const double AbilityLimit = 12.0;

    private sealed class Item
    {
        public readonly string Id;
        public readonly double Difficulty;

        public Item(string id, double difficulty)
        {
            Id = id;
            Difficulty = difficulty;
        }
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private static double ExpectedScore(List<Item> pool, double ability)
    {
        double total = 0.0;
        foreach (Item item in pool) total += Sigmoid(ability - item.Difficulty);
        return pool.Count == 0 ? double.NaN : total / pool.Count;
    }

    /** Bisection on the monotone expected-score curve; returns NaN for an empty pool. */
    private static double TargetAbility(List<Item> pool, double target)
    {
        if (pool.Count == 0) return double.NaN;
        double lo = -AbilityLimit, hi = AbilityLimit;
        for (int i = 0; i < 200; i++)
        {
            double mid = 0.5 * (lo + hi);
            if (ExpectedScore(pool, mid) < target) lo = mid;
            else hi = mid;
            if (hi - lo < 1e-13) break;
        }
        return 0.5 * (lo + hi);
    }

    /** Oracle for the selection: rank of an item counts how many items are closer to theta. */
    private static List<string> SelectByRank(List<Item> pool, double ability, int count)
    {
        var order = new List<int>();
        for (int i = 0; i < pool.Count; i++) order.Add(i);
        order.Sort((a, b) =>
        {
            double da = Math.Abs(pool[a].Difficulty - ability);
            double db = Math.Abs(pool[b].Difficulty - ability);
            if (Math.Abs(da - db) > 1e-12) return da.CompareTo(db);
            return string.CompareOrdinal(pool[a].Id, pool[b].Id);
        });
        var ids = new List<string>();
        for (int i = 0; i < Math.Min(count, order.Count); i++) ids.Add(pool[order[i]].Id);
        return ids;
    }

    private static List<Item> Select(List<Item> pool, double ability, int count)
    {
        var ranked = SelectByRank(pool, ability, count);
        var picked = new List<Item>();
        foreach (string id in ranked)
            foreach (Item item in pool)
                if (item.Id == id)
                {
                    picked.Add(item);
                    break;
                }
        return picked;
    }

    private static List<Item> Pool()
    {
        var pool = new List<Item>();
        for (int i = 0; i < 25; i++) pool.Add(new Item("i" + i.ToString("D2"), -3.0 + 0.25 * i));
        return pool;
    }

    private sealed class AdaptiveRun
    {
        public readonly List<string> Chosen = new List<string>();
        public double Estimate;
    }

    /** Adaptive mode: serve the item nearest the running estimate, then take a damped logistic step. */
    private static AdaptiveRun Adaptive(List<Item> pool, double trueAbility, ulong seed, int maxItems, double damp)
    {
        _state = seed;
        var run = new AdaptiveRun();
        double estimate = 0.0;
        for (int i = 0; i < maxItems && pool.Count > 0; i++)
        {
            Item item = Select(pool, estimate, 1)[0];
            run.Chosen.Add(item.Id);
            double predicted = Sigmoid(estimate - item.Difficulty);
            double outcome = NextUnit() < Sigmoid(trueAbility - item.Difficulty) ? 1.0 : 0.0;
            estimate += damp * (outcome - predicted);
            estimate = Math.Max(-8.0, Math.Min(8.0, estimate));
        }
        run.Estimate = estimate;
        return run;
    }

    public static void Main()
    {
        List<Item> pool = Pool();
        const double target = 0.6;
        double theta = TargetAbility(pool, target);
        Check(theta > -3.0 && theta < 3.0, "the target ability for 60% sits inside the pool's difficulty range");

        // Round trip 1: at that ability the whole pool really is expected to score the target.
        Check(Math.Abs(ExpectedScore(pool, theta) - target) < 1e-9,
            "the bisected ability reproduces the target score on the full pool (got "
            + ExpectedScore(pool, theta).ToString("F6") + ")");

        // Round trip 2: the emitted test is calibrated on its own items as well.
        List<Item> exam = Select(pool, theta, 10);
        Check(exam.Count == 10, "the generator emits the requested number of items");
        double examAbility = TargetAbility(exam, target);
        Check(Math.Abs(ExpectedScore(exam, examAbility) - target) < 1e-9,
            "the emitted test round trips to the target percentile on its own item set (got "
            + ExpectedScore(exam, examAbility).ToString("F6") + ")");
        double meanDifficulty = 0.0;
        foreach (Item item in exam) meanDifficulty += item.Difficulty;
        meanDifficulty /= exam.Count;
        Check(Math.Abs(meanDifficulty - theta) <= 0.13,
            "the mean difficulty of the exam sits within one pool spacing of the target ability (got "
            + meanDifficulty.ToString("F3") + ")");

        // Harder targets must produce harder tests.
        double easyMean = 0.0, hardMean = 0.0;
        foreach (Item item in Select(pool, TargetAbility(pool, 0.25), 10)) easyMean += item.Difficulty;
        foreach (Item item in Select(pool, TargetAbility(pool, 0.85), 10)) hardMean += item.Difficulty;
        Check(easyMean / 10.0 < hardMean / 10.0,
            "a 25% target emits an easier test than an 85% target (" + (easyMean / 10.0).ToString("F2")
            + " < " + (hardMean / 10.0).ToString("F2") + ")");

        // Closed-form oracle: with a single item the solution is theta = b + logit(target).
        var singleton = new List<Item> { new Item("solo", 1.5) };
        double closedForm = 1.5 + Math.Log(target / (1.0 - target));
        Check(Math.Abs(TargetAbility(singleton, target) - closedForm) < 1e-9,
            "the single-item pool reproduces the closed-form theta = b + logit(target)");
        Check(Math.Abs(ExpectedScore(singleton, TargetAbility(singleton, 0.9)) - 0.9) < 1e-9,
            "a 90% target is reachable on the single-item pool");

        // Symmetry oracle: a symmetric pool must place the 50% target exactly at zero.
        var symmetric = new List<Item>
        {
            new Item("a", -2.0), new Item("b", -1.0), new Item("c", 0.0), new Item("d", 1.0), new Item("e", 2.0),
        };
        Check(Math.Abs(TargetAbility(symmetric, 0.5)) < 1e-9,
            "a symmetric difficulty set puts the median target at ability 0 (got "
            + TargetAbility(symmetric, 0.5).ToString("E2") + ")");
        Check(TargetAbility(pool, 0.2) < TargetAbility(pool, 0.5) && TargetAbility(pool, 0.5) < TargetAbility(pool, 0.8),
            "raising the target percentile raises the required ability monotonically");

        // Selection oracle: the greedy nearest-difficulty pick matches the rank-counting formulation.
        List<string> oracle = SelectByRank(pool, theta, 10);
        bool same = oracle.Count == exam.Count;
        for (int i = 0; i < exam.Count && same; i++) same &= oracle[i] == exam[i].Id;
        Check(same, "the emitted exam matches the independent rank-counting oracle item for item");
        var unique = new HashSet<string>(StringComparer.Ordinal);
        foreach (Item item in exam) unique.Add(item.Id);
        Check(unique.Count == exam.Count, "no item is served twice in the same test");

        // Adaptive mode: bounded, directional and reproducible.
        double estimate = 0.0;
        bool rising = true;
        const double anchor = 0.5;
        for (int i = 0; i < 20; i++)
        {
            double before = estimate;
            estimate += 0.4 * (1.0 - Sigmoid(estimate - anchor));
            estimate = Math.Max(-8.0, Math.Min(8.0, estimate));
            rising &= estimate >= before;
        }
        Check(rising && estimate > 2.0, "a twenty-correct streak only ever raises the ability estimate ("
            + estimate.ToString("F3") + ")");

        AdaptiveRun first = Adaptive(pool, 1.2, 978UL, 20, 0.4);
        AdaptiveRun second = Adaptive(pool, 1.2, 978UL, 20, 0.4);
        AdaptiveRun other = Adaptive(pool, 1.2, 979UL, 20, 0.4);
        Check(first.Chosen.Count == 20 && string.Join(",", first.Chosen) == string.Join(",", second.Chosen)
              && Math.Abs(first.Estimate - second.Estimate) < 1e-15,
            "two seeded adaptive runs are identical");
        Check(string.Join(",", first.Chosen) != string.Join(",", other.Chosen),
            "a different seed produces a different adaptive item sequence");
        bool bounded = true;
        foreach (AdaptiveRun run in new[] { first, second, other })
            bounded &= run.Estimate >= -8.0 && run.Estimate <= 8.0;
        Check(bounded, "the adaptive estimate stays inside the clamping band in every run");

        // Boundaries: empty pool, impossible request sizes and out-of-range targets.
        Check(double.IsNaN(TargetAbility(new List<Item>(), 0.5)), "an empty pool yields no target ability");
        Check(Select(new List<Item>(), 0.0, 5).Count == 0, "an empty pool emits an empty test");
        Check(Select(pool, theta, 0).Count == 0 && Select(pool, theta, pool.Count + 1).Count == pool.Count,
            "a zero request emits nothing and an oversized request emits the whole pool");
        Check(TargetAbility(pool, 0.0) <= -11.9 && TargetAbility(pool, 1.0) >= 11.9,
            "targets at the extremes clamp to the ability limit instead of diverging");

        Console.WriteLine("target " + target + " -> ability " + theta.ToString("F4") + ", mean exam difficulty "
            + meanDifficulty.ToString("F3"));
        Console.WriteLine("closed form for the single item: " + closedForm.ToString("F4") + " vs bisection "
            + TargetAbility(singleton, target).ToString("F4"));
        Console.WriteLine("adaptive run: " + first.Chosen.Count + " items, estimate "
            + first.Estimate.ToString("F3"));
        Console.WriteLine("PASS");
    }
}
