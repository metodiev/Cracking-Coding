// Task 973 - Implement automated hint generator using solved-solution decomposition.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// A canonical solution is decomposed into steps whose dependencies form a DAG (read input, build the
// index, scan for the complement, ...). Each step carries its hint text. Steps are levelled with
// longest-path relaxation (level 0 for a root, else 1 + max(level of dependencies)) and the hint
// ladder is the step order sorted by (level, id); a learner with k failed attempts unlocks the first
// min(k, n) entries.
//
// Verified: the level chart of the two-sum decomposition, the prefix property (the ladder for k
// attempts is a prefix of the ladder for k+1), the safety property that no hint exposes a step whose
// dependencies are still hidden, exhaustiveness without duplicates, the empty decomposition and the
// negative attempt-count boundaries, cycle and dangling-dependency rejection, a recursive DFS level
// oracle, and stability under 4 seeded input permutations.
// Simplification: the decomposition is hand-authored in memory and "attempts" is just an integer
// counter, so no learner model, persistence or natural-language generation is involved.
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

    private sealed class Step
    {
        public readonly string Id;
        public readonly string Hint;
        public readonly string[] DependsOn;

        public Step(string id, string hint, string[] dependsOn)
        {
            Id = id;
            Hint = hint;
            DependsOn = dependsOn;
        }
    }

    private static Dictionary<string, Step> Index(List<Step> steps)
    {
        var map = new Dictionary<string, Step>(StringComparer.Ordinal);
        foreach (Step step in steps) map[step.Id] = step;
        return map;
    }

    /** null when the graph is cyclic or references an unknown step. */
    private static Dictionary<string, int> Levels(List<Step> steps)
    {
        var map = Index(steps);
        var level = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (Step step in steps) level[step.Id] = 0;
        for (int round = 0; round < steps.Count; round++)
        {
            bool changed = false;
            foreach (Step step in steps)
                foreach (string dependency in step.DependsOn)
                {
                    if (!map.ContainsKey(dependency)) return null;
                    if (level[step.Id] < level[dependency] + 1)
                    {
                        level[step.Id] = level[dependency] + 1;
                        changed = true;
                    }
                }
            if (!changed) return level;
        }
        // A DAG with n steps settles within n rounds; still growing means a cycle.
        foreach (Step step in steps)
            foreach (string dependency in step.DependsOn)
                if (level[step.Id] < level[dependency] + 1) return null;
        return level;
    }

    /** Oracle: the same levels computed by memoised recursion instead of relaxation. */
    private static Dictionary<string, int> DfsLevels(List<Step> steps)
    {
        var map = Index(steps);
        var levels = new Dictionary<string, int>(StringComparer.Ordinal);
        var visiting = new HashSet<string>(StringComparer.Ordinal);
        foreach (Step step in steps) DfsLevel(step.Id, map, levels, visiting);
        return levels;
    }

    private static int DfsLevel(string id, Dictionary<string, Step> map, Dictionary<string, int> levels, HashSet<string> visiting)
    {
        if (levels.TryGetValue(id, out int known)) return known;
        visiting.Add(id);
        int best = 0;
        foreach (string dependency in map[id].DependsOn)
        {
            int level = visiting.Contains(dependency) ? -1 : DfsLevel(dependency, map, levels, visiting);
            best = Math.Max(best, level + 1);
        }
        visiting.Remove(id);
        levels[id] = best;
        return best;
    }

    private static List<Step> Reveal(List<Step> steps, int attempts)
    {
        if (steps == null || steps.Count == 0) return new List<Step>();
        if (attempts < 0) attempts = 0;
        Dictionary<string, int> level = Levels(steps);
        if (level == null) return null;
        var order = new List<Step>(steps);
        order.Sort((a, b) => level[a.Id] != level[b.Id]
            ? level[a.Id].CompareTo(level[b.Id])
            : string.CompareOrdinal(a.Id, b.Id));
        return order.GetRange(0, Math.Min(attempts, order.Count));
    }

    private static string Ids(List<Step> steps)
    {
        if (steps == null) return "<rejected>";
        var ids = new List<string>();
        foreach (Step step in steps) ids.Add(step.Id);
        return string.Join(",", ids);
    }

    private static List<T> Shuffle<T>(List<T> items, ulong seed)
    {
        _state = seed;
        var copy = new List<T>(items);
        for (int i = copy.Count - 1; i > 0; i--)
        {
            int j = (int)(NextUnit() * (i + 1));
            T swap = copy[i];
            copy[i] = copy[j];
            copy[j] = swap;
        }
        return copy;
    }

    private static List<Step> TwoSumSteps()
    {
        return new List<Step>
        {
            new Step("read-input", "Parse n, the target and the array from stdin.", new string[0]),
            new Step("index-complement", "Store every value with its index in a hash map.", new[] { "read-input" }),
            new Step("scan-pair", "For each element look up target minus that element.", new[] { "read-input", "index-complement" }),
            new Step("handle-duplicates", "Reject reusing the same index twice.", new[] { "index-complement" }),
            new Step("order-output", "Sort the two indices before printing them.", new[] { "scan-pair" }),
            new Step("emit-result", "Print the pair of indices.", new[] { "order-output", "handle-duplicates" }),
        };
    }

    public static void Main()
    {
        List<Step> steps = TwoSumSteps();
        Dictionary<string, Step> map = Index(steps);
        Dictionary<string, int> levels = Levels(steps);
        Check(levels != null && levels["read-input"] == 0 && levels["index-complement"] == 1
              && levels["scan-pair"] == 2 && levels["handle-duplicates"] == 2
              && levels["order-output"] == 3 && levels["emit-result"] == 4,
            "the two-sum decomposition levels are 0,1,2,2,3,4");

        // Happy path: the ladder grows by exactly one hint per failed attempt.
        List<Step> twoHints = Reveal(steps, 2);
        Check(Ids(twoHints) == "read-input,index-complement",
            "two failed attempts unlock the two shallowest steps first (got " + Ids(twoHints) + ")");
        Check(Ids(Reveal(steps, 5)) == "read-input,index-complement,handle-duplicates,scan-pair,order-output",
            "the ladder orders equal-level steps by id");
        List<Step> complete = Reveal(steps, 99);
        Check(complete.Count == steps.Count, "enough attempts unlock every step, so hints are exhaustive");
        var distinct = new HashSet<string>(StringComparer.Ordinal);
        foreach (Step step in complete) distinct.Add(step.Id);
        Check(distinct.Count == steps.Count, "no step is hinted twice");

        // Prefix property across the whole ladder.
        bool prefix = true;
        for (int k = 0; k <= steps.Count; k++)
        {
            List<Step> shorter = Reveal(steps, k);
            List<Step> longer = Reveal(steps, k + 1);
            for (int i = 0; i < shorter.Count; i++) prefix &= shorter[i].Id == longer[i].Id;
        }
        Check(prefix, "the hints for k attempts are always a prefix of the hints for k+1 attempts");

        // Safety: a revealed hint never exposes a step whose dependencies are still hidden.
        int unsafeReveals = 0;
        for (int k = 0; k <= steps.Count; k++)
        {
            var revealed = new HashSet<string>(StringComparer.Ordinal);
            foreach (Step step in Reveal(steps, k)) revealed.Add(step.Id);
            foreach (string id in revealed)
                foreach (string dependency in map[id].DependsOn)
                    if (!revealed.Contains(dependency)) unsafeReveals++;
        }
        Check(unsafeReveals == 0, "no hint depends on a step that is still hidden at that attempt count");

        // Oracle: memoised recursion must produce the same level chart as relaxation.
        Dictionary<string, int> dfs = DfsLevels(steps);
        int drift = 0;
        foreach (Step step in steps) if (dfs[step.Id] != levels[step.Id]) drift++;
        Check(drift == 0, "the recursive DFS oracle agrees with the relaxation levels on all 6 steps");

        // Boundaries: nothing attempted, negative attempts, empty and malformed decompositions.
        Check(Reveal(steps, 0).Count == 0, "zero attempts generate no hint at all");
        Check(Reveal(steps, -4).Count == 0, "a negative attempt count is treated as zero attempts");
        Check(Reveal(new List<Step>(), 3) != null && Reveal(new List<Step>(), 3).Count == 0,
            "an empty decomposition yields an empty ladder rather than failing");
        Check(Reveal(new List<Step>
        {
            new Step("a", "first", new[] { "b" }),
            new Step("b", "second", new[] { "a" }),
        }, 5) == null, "a cyclic decomposition is rejected instead of hinted in an arbitrary order");
        Check(Reveal(new List<Step> { new Step("a", "first", new[] { "ghost" }) }, 5) == null,
            "a step that depends on an unknown step is rejected");

        // Stability: the ladder follows the graph, not the order the steps arrive in.
        bool stable = true;
        for (ulong seed = 1; seed <= 4; seed++)
            stable &= Ids(Reveal(Shuffle(steps, seed), 99)) == Ids(complete);
        Check(stable, "four seeded input permutations produce the identical hint ladder");

        Console.WriteLine("steps: " + steps.Count + ", levels " + levels["read-input"] + "/" + levels["index-complement"]
            + "/" + levels["scan-pair"] + "," + levels["handle-duplicates"] + "/" + levels["order-output"]
            + "/" + levels["emit-result"]);
        Console.WriteLine("ladder at k=2: " + Ids(twoHints));
        Console.WriteLine("unsafe reveals across all attempt counts: " + unsafeReveals);
        Console.WriteLine("PASS");
    }
}
