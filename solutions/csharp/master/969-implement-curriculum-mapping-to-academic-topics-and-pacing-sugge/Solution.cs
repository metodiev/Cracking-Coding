// Task 969 - Implement curriculum mapping to academic topics and pacing suggestions.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Problems carry topic tags and topics declare prerequisites. The mapper computes per-topic coverage,
// orders the topics so no prerequisite is ever scheduled after a topic that needs it (Kahn's algorithm
// with an id tie-break, so the schedule is reproducible), and fills fixed-minute week buckets along that
// order to suggest pacing.
//
// Verified: coverage counts, the prerequisite invariant across the whole order, the permutation property,
// buckets that never exceed the weekly budget and lose no minutes, gap reporting, cycle and
// dangling-prerequisite rejection, the oversized-topic and empty-curriculum boundaries, an independent
// prefix-closure oracle, a relaxation-based level oracle and stability under five seeded input shuffles.
// Simplification: a handful of in-memory topics and problems; pacing is a greedy bucket fill rather than
// a solver over a real multi-week academic calendar.
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

    private sealed class Topic
    {
        public readonly string Id; public readonly int Minutes; public readonly string[] Prerequisites;
        public Topic(string id, int minutes, params string[] prerequisites) { Id = id; Minutes = minutes; Prerequisites = prerequisites; }
    }

    private sealed class Problem
    {
        public readonly string[] Tags;
        public Problem(params string[] tags) { Tags = tags; }
    }

    private sealed class Week
    {
        public readonly int Index; public readonly List<string> Topics = new List<string>(); public int Minutes;

        public Week(int index) { Index = index; }
    }

    private static Dictionary<string, Topic> Index(List<Topic> topics)
    {
        var map = new Dictionary<string, Topic>(StringComparer.Ordinal);
        foreach (Topic topic in topics) map[topic.Id] = topic;
        return map;
    }

    /** Each problem contributes at most one point per topic even when it repeats a tag. */
    private static Dictionary<string, int> Coverage(List<Topic> topics, List<Problem> problems)
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (Topic topic in topics) counts[topic.Id] = 0;
        foreach (Problem problem in problems)
        {
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (string tag in problem.Tags) if (seen.Add(tag) && counts.ContainsKey(tag)) counts[tag]++;
        }
        return counts;    }

    private static List<string> TopologicalOrder(List<Topic> topics)
    {
        var map = Index(topics);
        var indegree = new Dictionary<string, int>(StringComparer.Ordinal);
        var dependents = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        foreach (Topic topic in topics) { indegree[topic.Id] = 0; dependents[topic.Id] = new List<string>(); }
        foreach (Topic topic in topics)
            foreach (string prerequisite in topic.Prerequisites)
            {
                if (!map.ContainsKey(prerequisite)) return null;
                indegree[topic.Id]++;
                dependents[prerequisite].Add(topic.Id);
            }
        var ready = new List<string>();
        foreach (Topic topic in topics) if (indegree[topic.Id] == 0) ready.Add(topic.Id);
        ready.Sort(StringComparer.Ordinal);
        var order = new List<string>();
        while (ready.Count > 0)
        {
            string next = ready[0];
            ready.RemoveAt(0);
            order.Add(next);
            foreach (string dependent in dependents[next]) if (--indegree[dependent] == 0) ready.Add(dependent);
            ready.Sort(StringComparer.Ordinal);
        }
        return order.Count == topics.Count ? order : null;
    }

    /** Oracle: level = 0 for a source, else 1 + max(level of prerequisites), computed by relaxation. */
    private static Dictionary<string, int> LongestPathLevels(List<Topic> topics)
    {
        var map = Index(topics);
        var level = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (Topic topic in topics) level[topic.Id] = 0;
        for (int round = 0; round < topics.Count; round++)
        {
            bool changed = false;
            foreach (Topic topic in topics)
                foreach (string prerequisite in topic.Prerequisites)
                    if (map.ContainsKey(prerequisite) && level[topic.Id] < level[prerequisite] + 1)
                    { level[topic.Id] = level[prerequisite] + 1; changed = true; }
            if (!changed) break;
        }
        return level;
    }

    private static List<Week> Pace(List<string> order, Dictionary<string, Topic> map, int minutesPerWeek)
    {
        var weeks = new List<Week>();
        if (order == null || order.Count == 0 || minutesPerWeek <= 0) return weeks;
        var current = new Week(1);
        weeks.Add(current);
        foreach (string id in order)
        {
            // A topic taller than one week is never split; it simply overflows its own week.
            if (current.Minutes > 0 && current.Minutes + map[id].Minutes > minutesPerWeek)
            { current = new Week(current.Index + 1); weeks.Add(current); }
            current.Topics.Add(id);
            current.Minutes += map[id].Minutes;
        }
        return weeks;
    }

    private static List<string> Gaps(Dictionary<string, int> coverage)
    {
        var gaps = new List<string>();
        foreach (KeyValuePair<string, int> pair in coverage) if (pair.Value == 0) gaps.Add(pair.Key);
        gaps.Sort(StringComparer.Ordinal);
        return gaps;
    }

    /** Reproduction guard: the schedule must not depend on the order topics arrive in. */
    private static List<Topic> Shuffle(List<Topic> topics, ulong seed)
    {
        _state = seed;
        var copy = new List<Topic>(topics);
        for (int i = copy.Count - 1; i > 0; i--)
        {
            int j = (int)(NextUnit() * (i + 1));
            Topic swap = copy[i]; copy[i] = copy[j]; copy[j] = swap;
        }
        return copy;
    }

    public static void Main()
    {
        var topics = new List<Topic>
        {
            new Topic("arrays", 90), new Topic("hashing", 120, "arrays"), new Topic("strings", 60),
            new Topic("trees", 150, "arrays"), new Topic("graphs", 180, "trees", "hashing"),
            new Topic("dp", 210, "graphs"),
        };
        var problems = new List<Problem>
        {
            new Problem("arrays"), new Problem("arrays", "hashing"), new Problem("trees"),
            new Problem("graphs"), new Problem("dp", "dp"), new Problem("arrays"),
        };
        Dictionary<string, Topic> map = Index(topics);
        Dictionary<string, int> coverage = Coverage(topics, problems);

        Check(coverage["arrays"] == 3 && coverage["hashing"] == 1 && coverage["trees"] == 1
              && coverage["graphs"] == 1 && coverage["dp"] == 1 && coverage["strings"] == 0
              && string.Join(",", Gaps(coverage)) == "strings",
            "coverage counts every problem once per topic, and only 'strings' is a gap");

        List<string> order = TopologicalOrder(topics);
        Check(order != null && order.Count == topics.Count
              && new HashSet<string>(order, StringComparer.Ordinal).Count == topics.Count,
            "the acyclic curriculum yields a permutation covering every topic exactly once");

        int violations = 0;
        var position = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < order.Count; i++) position[order[i]] = i;
        foreach (Topic topic in topics)
            foreach (string prerequisite in topic.Prerequisites)
                if (position[prerequisite] >= position[topic.Id]) violations++;
        Check(violations == 0, "every prerequisite is scheduled strictly before its dependent");

        // Oracle: every prefix of the order must be closed under prerequisites, and the relaxation
        // levels must agree with the edges without reusing Kahn's own bookkeeping.
        Dictionary<string, int> levels = LongestPathLevels(topics);
        int openPrefixes = 0;
        for (int cut = 1; cut <= order.Count; cut++)
        {
            var prefix = new HashSet<string>(order.GetRange(0, cut), StringComparer.Ordinal);
            bool closed = true;
            foreach (string id in prefix)
                foreach (string prerequisite in map[id].Prerequisites)
                    if (!prefix.Contains(prerequisite)) closed = false;
            if (closed) openPrefixes++;
        }
        Check(openPrefixes == order.Count && levels["arrays"] == 0 && levels["hashing"] == 1 && levels["trees"] == 1
              && levels["strings"] == 0 && levels["graphs"] == 2 && levels["dp"] == 3,
            "every prefix of the order is closed under prerequisites and the relaxation levels match the depth chart");
        const int minutesPerWeek = 300;
        List<Week> weeks = Pace(order, map, minutesPerWeek);
        int scheduled = 0, packed = 0, overflows = 0;
        foreach (Week week in weeks)
        {
            scheduled += week.Minutes;
            packed += week.Topics.Count;
            if (week.Minutes > minutesPerWeek) overflows++;
        }
        Check(weeks.Count == 4 && scheduled == 810 && packed == order.Count,
            "810 curriculum minutes at 300 minutes/week need 4 weeks and lose no minutes or topics");
        Check(overflows == 0 && weeks[0].Minutes == 270 && weeks[0].Topics.Count == 3,
            "no normal week exceeds the budget and the first week packs 270 of 300 minutes");

        // Boundary: an oversized topic owns its week, and the next topic starts a fresh one.
        var oversized = new List<Topic> { new Topic("monster", 500), new Topic("after", 10) };        List<Week> stretched = Pace(new List<string> { "monster", "after" }, Index(oversized), minutesPerWeek);
        Check(stretched.Count == 2 && stretched[0].Topics.Count == 1 && stretched[0].Minutes == 500
              && TopologicalOrder(new List<Topic> { new Topic("a", 10, "b"), new Topic("b", 10, "a") }) == null
              && TopologicalOrder(new List<Topic> { new Topic("a", 10, "ghost") }) == null
              && TopologicalOrder(new List<Topic>()).Count == 0
              && Pace(new List<string>(), map, minutesPerWeek).Count == 0,
            "an oversized topic, a prerequisite cycle, a dangling prerequisite and an empty curriculum are all handled");

        bool stable = true;
        for (ulong seed = 1; seed <= 5; seed++)
        {
            List<string> shuffled = TopologicalOrder(Shuffle(topics, seed));
            for (int i = 0; i < order.Count; i++) stable &= shuffled != null && shuffled[i] == order[i];
        }
        Check(stable, "five seeded input permutations still produce the identical schedule");

        Console.WriteLine("curriculum: " + topics.Count + " topics, " + problems.Count + " problems, order = "
            + string.Join(" > ", order) + ", gaps = " + Gaps(coverage).Count + ", levels " + levels["dp"]
            + " deep, " + weeks.Count + " weeks of " + minutesPerWeek + " min");
        Console.WriteLine("PASS");
    }
}
