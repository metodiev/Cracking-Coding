// Task 961 - Implement mentoring workflow pairing novices with maintainers for PR review.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Preferences are derived deterministically: a novice ranks maintainers by how many interest areas they
// share (ties broken by id) and a maintainer ranks novices the same way.  Matching runs Gale-Shapley
// deferred acceptance with quotas - a maintainer tentatively keeps up to `Capacity` novices and displaces
// its least preferred one when a better proposal arrives, while the displaced novice keeps proposing.
//
// Verified: with enough capacity everyone is matched, nobody exceeds capacity, a brute-force blocking-pair
// scan finds nothing (the independent oracle), cardinality equals min(novices, total capacity) when the
// pool is too small, preference lists are permutations, an unmatched novice implies every maintainer is
// full, and the zero-novice / zero-maintainer / closed-maintainer boundaries hold.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private sealed class Novice
    {
        public readonly string Id;
        public readonly string[] Areas;
        public readonly List<string> Preferences = new List<string>();
        public string? Matched;
        public Novice(string id, params string[] areas) { Id = id; Areas = areas; }
    }

    private sealed class Maintainer
    {
        public readonly string Id;
        public readonly string[] Areas;
        public readonly int Capacity;
        public readonly List<string> Preferences = new List<string>();
        public readonly List<string> Assigned = new List<string>();
        public Maintainer(string id, int capacity, params string[] areas) { Id = id; Capacity = capacity; Areas = areas; }
    }

    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }

    private static int Shared(string[] left, string[] right)
    {
        int shared = 0;
        foreach (string a in left) { foreach (string b in right) { if (a == b) shared++; } }
        return shared;
    }

    private static Maintainer Find(List<Maintainer> maintainers, string id) => maintainers.Find(m => m.Id == id)!;

    private static Novice Find(List<Novice> novices, string id) => novices.Find(n => n.Id == id)!;

    private static int Compare(int leftScore, int rightScore, string leftId, string rightId) =>
        leftScore != rightScore ? rightScore.CompareTo(leftScore) : string.CompareOrdinal(leftId, rightId);

    /** Both sides rank the other by shared interest areas, breaking ties by id. */
    private static void BuildPreferences(List<Novice> novices, List<Maintainer> maintainers)
    {
        foreach (Novice novice in novices)
        {
            var order = new List<string>();
            foreach (Maintainer maintainer in maintainers) order.Add(maintainer.Id);
            order.Sort((x, y) => Compare(Shared(novice.Areas, Find(maintainers, x).Areas), Shared(novice.Areas, Find(maintainers, y).Areas), x, y));
            novice.Preferences.AddRange(order);
        }
        foreach (Maintainer maintainer in maintainers)
        {
            var order = new List<string>();
            foreach (Novice novice in novices) order.Add(novice.Id);
            order.Sort((x, y) => Compare(Shared(maintainer.Areas, Find(novices, x).Areas), Shared(maintainer.Areas, Find(novices, y).Areas), x, y));
            maintainer.Preferences.AddRange(order);
        }
    }

    private static int Preference(Maintainer maintainer, string noviceId) => maintainer.Preferences.IndexOf(noviceId);

    private static string Worst(Maintainer maintainer)
    {
        string worst = maintainer.Assigned[0];
        foreach (string candidate in maintainer.Assigned)
        {
            if (Preference(maintainer, candidate) > Preference(maintainer, worst)) worst = candidate;
        }
        return worst;
    }

    private static void Match(List<Novice> novices, List<Maintainer> maintainers)
    {
        var nextProposal = new Dictionary<string, int>(StringComparer.Ordinal);
        var queue = new Queue<Novice>();
        foreach (Novice novice in novices) { nextProposal[novice.Id] = 0; queue.Enqueue(novice); }
        while (queue.Count > 0)
        {
            Novice novice = queue.Dequeue();
            if (novice.Matched != null || nextProposal[novice.Id] >= novice.Preferences.Count) continue;
            Maintainer maintainer = Find(maintainers, novice.Preferences[nextProposal[novice.Id]]);
            nextProposal[novice.Id]++;
            if (maintainer.Assigned.Count < maintainer.Capacity) { maintainer.Assigned.Add(novice.Id); novice.Matched = maintainer.Id; continue; }
            if (maintainer.Capacity == 0) { queue.Enqueue(novice); continue; } // closed for business: nothing to displace
            string worst = Worst(maintainer);
            if (Preference(maintainer, novice.Id) >= Preference(maintainer, worst)) { queue.Enqueue(novice); continue; }
            maintainer.Assigned.Remove(worst);
            maintainer.Assigned.Add(novice.Id);
            novice.Matched = maintainer.Id;
            Novice displaced = Find(novices, worst);
            displaced.Matched = null;
            queue.Enqueue(displaced);
        }
    }

    /** Independent oracle: scan every (novice, maintainer) pair for a blocking pair. */
    private static List<string> BlockingPairs(List<Novice> novices, List<Maintainer> maintainers)
    {
        var blocking = new List<string>();
        foreach (Novice novice in novices)
        {
            foreach (Maintainer maintainer in maintainers)
            {
                if (novice.Matched == maintainer.Id) continue;
                if (novice.Matched != null && novice.Preferences.IndexOf(maintainer.Id) >= novice.Preferences.IndexOf(novice.Matched)) continue;
                bool prefers = maintainer.Assigned.Count < maintainer.Capacity ||
                    (maintainer.Capacity > 0 && Preference(maintainer, novice.Id) < Preference(maintainer, Worst(maintainer)));
                if (prefers) blocking.Add(novice.Id + "->" + maintainer.Id);
            }
        }
        blocking.Sort(StringComparer.Ordinal);
        return blocking;
    }

    private static List<Novice> Novices() => new List<Novice>
    {
        new Novice("n1", "sorting"), new Novice("n2", "strings"), new Novice("n3", "testing"),
        new Novice("n4", "algorithms"), new Novice("n5", "algorithms"), new Novice("n6", "sorting"),
    };

    private static List<Maintainer> Maintainers(int cap1, int cap2, int cap3) => new List<Maintainer>
    {
        new Maintainer("m1", cap1, "algorithms", "sorting"),
        new Maintainer("m2", cap2, "algorithms", "strings"),
        new Maintainer("m3", cap3, "algorithms", "testing"),
    };

    private static int MatchedCount(List<Novice> novices)
    {
        int matched = 0;
        foreach (Novice novice in novices) { if (novice.Matched != null) matched++; }
        return matched;
    }

    private static List<string> Assignment(List<Novice> novices)
    {
        var labels = new List<string>();
        foreach (Novice novice in novices) labels.Add(novice.Id + "->" + (novice.Matched ?? "none"));
        return labels;
    }

    public static void Main()
    {
        var novices = Novices();
        var maintainers = Maintainers(3, 3, 2);
        BuildPreferences(novices, maintainers);

        bool permutations = true;
        foreach (Novice novice in novices)
        {
            var seen = new HashSet<string>(StringComparer.Ordinal);
            permutations &= novice.Preferences.Count == maintainers.Count;
            foreach (string id in novice.Preferences) permutations &= seen.Add(id);
        }
        foreach (Maintainer maintainer in maintainers)
        {
            var seen = new HashSet<string>(StringComparer.Ordinal);
            permutations &= maintainer.Preferences.Count == novices.Count;
            foreach (string id in maintainer.Preferences) permutations &= seen.Add(id);
        }
        Check(permutations, "every preference list is a permutation of the other side");
        Check(novices[0].Preferences[0] == "m1", "n1 (sorting) ranks m1 (sorting) first by shared area");

        Match(novices, maintainers);
        Check(MatchedCount(novices) == novices.Count, "with enough capacity every novice is matched");
        bool capacityOk = true;
        foreach (Maintainer maintainer in maintainers) capacityOk &= maintainer.Assigned.Count <= maintainer.Capacity;
        Check(capacityOk, "no maintainer is assigned more novices than its capacity");
        List<string> blocking = BlockingPairs(novices, maintainers);
        Check(blocking.Count == 0, "the brute-force oracle finds no blocking pair (" + blocking.Count + " found)");
        bool listed = true;
        foreach (Novice novice in novices) listed &= novice.Preferences.Contains(Find(maintainers, novice.Matched!).Id);
        Check(listed, "every match came from the novice's own preference list");

        var tightNovices = Novices();
        var tightMaintainers = Maintainers(2, 2, 1);
        BuildPreferences(tightNovices, tightMaintainers);
        Match(tightNovices, tightMaintainers);
        int totalCapacity = 0, usedCapacity = 0;
        foreach (Maintainer maintainer in tightMaintainers) { totalCapacity += maintainer.Capacity; usedCapacity += maintainer.Assigned.Count; }
        Check(MatchedCount(tightNovices) == Math.Min(6, totalCapacity) && MatchedCount(tightNovices) == 5,
            "a constrained pool still reaches the cardinality optimum min(novices, capacity) = 5");
        Check(BlockingPairs(tightNovices, tightMaintainers).Count == 0, "the constrained matching is still stable");
        bool fullOrMatched = true;
        foreach (Novice novice in tightNovices)
        {
            if (novice.Matched != null) continue;
            foreach (Maintainer maintainer in tightMaintainers) fullOrMatched &= maintainer.Assigned.Count == maintainer.Capacity;
        }
        Check(fullOrMatched, "an unmatched novice implies every maintainer is already at capacity");

        var nobody = new List<Novice>();
        Match(nobody, Maintainers(1, 1, 1));
        Check(nobody.Count == 0, "zero novices is a no-op");
        var orphans = Novices();
        BuildPreferences(orphans, new List<Maintainer>());
        Match(orphans, new List<Maintainer>());
        bool allUnmatched = true;
        foreach (Novice novice in orphans) allUnmatched &= novice.Matched == null;
        Check(allUnmatched, "with zero maintainers every novice stays unmatched without crashing");

        var single = new List<Novice> { new Novice("only", "strings") };
        var closed = Maintainers(0, 1, 0); // m1 and m3 are closed, m2 has one slot
        BuildPreferences(single, closed);
        Match(single, closed);
        Check(single[0].Matched == "m2", "a single novice matches the only maintainer with an open slot");
        Check(BlockingPairs(single, closed).Count == 0, "closed maintainers never create a blocking pair");

        var againNovices = Novices();
        var againMaintainers = Maintainers(3, 3, 2);
        BuildPreferences(againNovices, againMaintainers);
        Match(againNovices, againMaintainers);
        bool deterministic = true;
        for (int i = 0; i < novices.Count; i++) deterministic &= novices[i].Matched == againNovices[i].Matched;
        Check(deterministic, "re-running the matching reproduces the identical assignment");

        Console.WriteLine("matching: " + MatchedCount(novices) + "/" + novices.Count + " novices placed, blocking pairs " + blocking.Count);
        Console.WriteLine("assignment: " + string.Join(", ", Assignment(novices)));
        Console.WriteLine("tight pool: " + MatchedCount(tightNovices) + " matched with capacity " + totalCapacity + " (" + usedCapacity + " slots used)");
        Console.WriteLine("PASS");
    }
}
