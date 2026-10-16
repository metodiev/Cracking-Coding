// Task 979 - Implement certification exam generator and proctoring integration.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// The generator takes an exam blueprint (how many items per topic, how many minutes allowed) and a
// certified item pool. Candidates are sorted by id, shuffled with a splitmix64 stream seeded from the
// topic name and the blueprint seed, and the required number is drawn per topic; a blueprint that
// cannot be satisfied - unknown topic, too few items, or a total time over budget - is rejected
// instead of being patched up. The proctoring integration scores in-window integrity events and
// returns PASS, REVIEW or FAIL.
//
// Verified: per-topic counts against an independent count of the emitted exam, unique items, the time
// budget invariant, determinism and seed sensitivity, the shortfall/arithmetic/budget rejections, the
// proctoring verdict ladder including a critical event, the inclusive exam-window boundary with an
// independent recomputation of the integrity score, and the empty-pool edge case.
// Simplification: blueprints, items and integrity events are in-memory records; proctoring is a weighted
// event tally rather than a real biometric or browser-integrity signal.
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

    private sealed class Item
    {
        public readonly string Id, Topic;
        public readonly int Minutes;

        public Item(string id, string topic, int minutes)
        { Id = id; Topic = topic; Minutes = minutes; }
    }

    private sealed class Blueprint
    {
        public readonly Dictionary<string, int> PerTopic = new Dictionary<string, int>(StringComparer.Ordinal);
        public int TotalItems;
        public int MinutesBudget;

        public Blueprint With(string topic, int count)
        {
            PerTopic[topic] = count;
            return this;
        }
    }

    private sealed class Exam
    {
        public readonly List<Item> Items = new List<Item>();
        public int TotalMinutes;
    }

    private sealed class ProctorEvent
    {
        public readonly string Kind;
        public readonly int Second;

        public ProctorEvent(string kind, int second)
        { Kind = kind; Second = second; }
    }

    private sealed class ProctorReport
    {
        public int Score;
        public bool Critical;
        public string Verdict = "PASS";
    }

    private static readonly string[] CriticalKinds = { "multiple-faces", "screen-share-off" };

    private static readonly string[] Topics = { "arrays", "dp", "graphs" };

    private static ulong Hash(string text, ulong seed)
    {
        ulong hash = seed ^ 0xCBF29CE484222325UL;
        unchecked
        {
            foreach (char character in text)
            {
                hash = (hash ^ character) * 0x100000001B3UL;
                hash ^= hash >> 29;
            }
        }
        return hash;
    }

    private static Exam Generate(Blueprint blueprint, List<Item> pool, ulong seed)
    {
        if (blueprint == null || pool.Count == 0 || blueprint.TotalItems <= 0) return null;
        int required = 0;
        foreach (KeyValuePair<string, int> pair in blueprint.PerTopic) required += pair.Value;
        if (required != blueprint.TotalItems) return null;

        var exam = new Exam();
        var topics = new List<string>(blueprint.PerTopic.Keys);
        topics.Sort(StringComparer.Ordinal);
        foreach (string topic in topics)
        {
            var candidates = new List<Item>();
            foreach (Item item in pool) if (item.Topic == topic) candidates.Add(item);
            candidates.Sort((a, b) => string.CompareOrdinal(a.Id, b.Id));
            int want = blueprint.PerTopic[topic];
            if (want < 0 || candidates.Count < want) return null;
            _state = Hash(topic, seed);
            for (int i = candidates.Count - 1; i > 0; i--)
            {
                int j = (int)(NextUnit() * (i + 1));
                Item swap = candidates[i];
                candidates[i] = candidates[j];
                candidates[j] = swap;
            }
            for (int i = 0; i < want; i++)
            {
                exam.Items.Add(candidates[i]);
                exam.TotalMinutes += candidates[i].Minutes;
            }
        }
        return exam.TotalMinutes > blueprint.MinutesBudget ? null : exam;
    }

    private static int Weight(string kind)
        => kind == "gaze-away" ? 1 : kind == "clipboard" || kind == "identity-mismatch" ? 2 : 0;

    private static bool IsCritical(string kind) => kind == "multiple-faces" || kind == "screen-share-off";

    /** The exam window is inclusive on both ends; anything outside is not attributed to the exam. */
    private static ProctorReport Proctor(List<ProctorEvent> events, int startSecond, int endSecond)
    {
        var report = new ProctorReport();
        foreach (ProctorEvent item in events)
        {
            if (item.Second < startSecond || item.Second > endSecond) continue;
            report.Score += Weight(item.Kind);
            report.Critical |= IsCritical(item.Kind);
        }
        report.Verdict = report.Critical ? "FAIL" : report.Score >= 3 ? "REVIEW" : "PASS";
        return report;
    }

    private static List<Item> ItemPool()
    {
        return new List<Item>
        {
            new Item("a1", "arrays", 5), new Item("a2", "arrays", 7), new Item("a3", "arrays", 6),
            new Item("a4", "arrays", 9), new Item("g1", "graphs", 10), new Item("g2", "graphs", 12),
            new Item("g3", "graphs", 11), new Item("d1", "dp", 15), new Item("d2", "dp", 18),
        };
    }

    private static Blueprint SampleBlueprint()
    {
        return new Blueprint { TotalItems = 5, MinutesBudget = 60 }
            .With("arrays", 2).With("dp", 1).With("graphs", 2);
    }

    public static void Main()
    {
        List<Item> pool = ItemPool();
        Blueprint blueprint = SampleBlueprint();
        Exam exam = Generate(blueprint, pool, 979UL);
        Check(exam != null && exam.Items.Count == blueprint.TotalItems, "the blueprint's five items are emitted");

        // Oracle: per-topic counts recomputed by brute force over the emitted exam.
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (Item item in exam.Items) counts[item.Topic] = counts.TryGetValue(item.Topic, out int seen) ? seen + 1 : 1;
        int mismatches = 0;
        foreach (KeyValuePair<string, int> pair in blueprint.PerTopic)
            if (!counts.TryGetValue(pair.Key, out int got) || got != pair.Value) mismatches++;
        Check(mismatches == 0, "every topic is represented exactly as the blueprint demands");

        var unique = new HashSet<string>(StringComparer.Ordinal); var originalIds = new List<string>();
        foreach (Item item in exam.Items) { unique.Add(item.Id); originalIds.Add(item.Id); }
        Check(unique.Count == exam.Items.Count && exam.TotalMinutes <= blueprint.MinutesBudget,
            "no item is scheduled twice and the exam fits the " + blueprint.MinutesBudget + " minute budget");

        // Determinism and seed sensitivity.
        var repeatIds = new List<string>();
        foreach (Item item in Generate(blueprint, pool, 979UL).Items) repeatIds.Add(item.Id);
        var otherIds = new List<string>();
        foreach (Item item in Generate(blueprint, pool, 4242UL).Items) otherIds.Add(item.Id);
        Check(string.Join(",", repeatIds) == string.Join(",", originalIds)
              && string.Join(",", otherIds) != string.Join(",", originalIds),
            "the same seed reproduces the identical exam and a different seed does not");

        // Rejection paths: shortfall, blueprint arithmetic, an impossible budget and an empty pool.
        Check(Generate(new Blueprint { TotalItems = 3, MinutesBudget = 60 }.With("dp", 3), pool, 1UL) == null
              && Generate(new Blueprint { TotalItems = 6, MinutesBudget = 60 }.With("arrays", 2).With("dp", 1), pool, 1UL) == null
              && Generate(new Blueprint { TotalItems = 5, MinutesBudget = 20 }.With("arrays", 2).With("dp", 1)
                  .With("graphs", 2), pool, 1UL) == null && Generate(blueprint, new List<Item>(), 1UL) == null,
            "a topic shortfall, a mismatched blueprint, an impossible budget and an empty pool are all rejected");

        // Proctoring verdict ladder: clean passes, gaze spam reviews, a critical event fails.
        ProctorReport clean = Proctor(new List<ProctorEvent> { new ProctorEvent("identity-verified", 10) }, 0, 3600);
        Check(clean.Verdict == "PASS" && clean.Score == 0, "a clean session with only identity events passes");
        var gaze = new List<ProctorEvent>();
        for (int i = 0; i < 3; i++) gaze.Add(new ProctorEvent("gaze-away", 100 + i * 30));
        ProctorReport reviewed = Proctor(gaze, 0, 3600);
        ProctorReport borderline = Proctor(new List<ProctorEvent>
        {
            new ProctorEvent("gaze-away", 10), new ProctorEvent("clipboard", 20),
        }, 0, 3600);
        Check(reviewed.Verdict == "REVIEW" && reviewed.Score == 3
              && borderline.Verdict == "REVIEW" && borderline.Score == 3,
            "three gazes, and a gaze plus a clipboard event, both score exactly three and review");
        ProctorReport failed = Proctor(new List<ProctorEvent> { new ProctorEvent("multiple-faces", 500) }, 0, 3600);
        Check(failed.Verdict == "FAIL" && failed.Critical, "a critical event fails the session regardless of the score");

        // Window boundary and oracle: the window is inclusive and everything outside is ignored.
        var boundary = new List<ProctorEvent>
        {
            new ProctorEvent("gaze-away", -1), new ProctorEvent("gaze-away", 0),
            new ProctorEvent("clipboard", 3600), new ProctorEvent("clipboard", 3601),
        };
        ProctorReport windowed = Proctor(boundary, 0, 3600);
        int oracleScore = 0; bool oracleCritical = false;
        foreach (ProctorEvent item in boundary)
        {
            if (item.Second < 0 || item.Second > 3600) continue;
            oracleScore += item.Kind == "gaze-away" ? 1 : item.Kind == "clipboard" ? 2 : 0;
            oracleCritical |= item.Kind == "multiple-faces" || item.Kind == "screen-share-off";
        }
        Check(windowed.Score == 3 && Proctor(boundary, 5000, 5000).Score == 0
              && oracleScore == windowed.Score && oracleCritical == windowed.Critical,
            "only the two in-window events count, a zero-length window detects nothing, and the tally is reproduced");

        Console.WriteLine("exam: " + string.Join(",", originalIds) + " in " + exam.TotalMinutes + " of "
            + blueprint.MinutesBudget + " minutes | proctoring: clean=" + clean.Verdict + " gaze3=" + reviewed.Verdict
            + " critical=" + failed.Verdict + " windowed=" + windowed.Score);
        Console.WriteLine("PASS");
    }
}
