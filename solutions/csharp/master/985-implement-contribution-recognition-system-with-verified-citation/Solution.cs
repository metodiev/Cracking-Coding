// Task 985 - Implement contribution recognition system with verified citations for authors.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// An append-only contribution ledger chains every entry into the previous one
// (h_i = mix(h_{i-1} || id_i || author_i || kind_i)), so a retro-active edit breaks the chain from that
// index on. A recognition claim is honoured only when the entry exists, the claimed author matches, the
// chain still verifies and the claim is new; refusals carry a machine-readable reason. Verified: the
// incremental tally equals a brute-force rescan after every award (oracle), duplicate/unknown/misattributed
// claims refused, tampering located at the exact index, empty and single-entry boundaries, and inclusive
// tier thresholds. Simplification: in-memory ledger with a hand-written FNV-1a + splitmix64 mix instead of
// a real VCS object store; the checked chain invariant is the same.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public static class Solution
{
    private enum Kind { Commit, Review, Issue, Docs }

    private static readonly Dictionary<Kind, int> Weights = new Dictionary<Kind, int>
    {
        [Kind.Commit] = 3, [Kind.Review] = 2, [Kind.Issue] = 4, [Kind.Docs] = 1,
    };

    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }

    private static ulong Mix(ulong z)
    {
        unchecked
        {
            z += 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    /** FNV-1a over the UTF-8 bytes, seeded with the previous chain hash. */
    private static ulong Fnv(string text, ulong seed)
    {
        unchecked
        {
            ulong h = 14695981039346656037UL ^ seed;
            foreach (byte b in Encoding.UTF8.GetBytes(text)) { h ^= b; h *= 1099511628211UL; }
            return h;
        }
    }

    private sealed class Entry
    {
        public readonly string Id; public readonly string Author; public readonly Kind Kind; public readonly ulong Hash;
        public Entry(string id, string author, Kind kind, ulong hash) { Id = id; Author = author; Kind = kind; Hash = hash; }
    }

    private sealed class Ledger
    {
        private readonly List<Entry> _entries = new List<Entry>();
        private ulong _head;

        public IReadOnlyList<Entry> Entries => _entries;

        public static ulong AppendHash(ulong prev, string id, string author, Kind kind) =>
            Mix(Fnv(prev.ToString("x16", CultureInfo.InvariantCulture) + "|" + id + "|" + author + "|" + kind, 0));

        public void Add(string id, string author, Kind kind)
        {
            if (Find(id) != null) throw new Exception("duplicate contribution id: " + id);
            ulong hash = AppendHash(_head, id, author, kind);
            _entries.Add(new Entry(id, author, kind, hash));
            _head = hash;
        }

        /** Recomputes the chain; returns the first index whose stored hash disagrees, or -1 when intact. */
        public int FirstBrokenIndex()
        {
            ulong cursor = 0;
            for (int i = 0; i < _entries.Count; i++)
            {
                ulong expected = AppendHash(cursor, _entries[i].Id, _entries[i].Author, _entries[i].Kind);
                if (expected != _entries[i].Hash) return i;
                cursor = expected;
            }
            return -1;
        }

        public Entry Find(string id)
        {
            foreach (Entry entry in _entries) if (entry.Id == id) return entry;
            return null;
        }

        /** Simulates a history rewrite: the author changes while the stored hash does not. */
        public void Tamper(int index, string author)
        {
            Entry entry = _entries[index];
            _entries[index] = new Entry(entry.Id, author, entry.Kind, entry.Hash);
        }
    }

    private sealed class Recognition
    {
        private readonly Ledger _ledger;
        private readonly Dictionary<string, int> _points = new Dictionary<string, int>(StringComparer.Ordinal);
        private readonly List<KeyValuePair<string, string>> _awards = new List<KeyValuePair<string, string>>();
        private readonly HashSet<string> _claimed = new HashSet<string>(StringComparer.Ordinal);
        public readonly List<string> Refusals = new List<string>();

        public Recognition(Ledger ledger) { _ledger = ledger; }

        public IReadOnlyDictionary<string, int> Points => _points;
        public IReadOnlyList<KeyValuePair<string, string>> Awards => _awards;
        public int PointsOf(string author) => _points.TryGetValue(author, out int p) ? p : 0;

        public bool Cite(string author, string contributionId)
        {
            int broken = _ledger.FirstBrokenIndex();
            if (broken >= 0) { Refusals.Add("broken-chain@" + broken); return false; }
            Entry entry = _ledger.Find(contributionId);
            if (entry == null) { Refusals.Add("unknown-contribution:" + contributionId); return false; }
            if (entry.Author != author) { Refusals.Add("author-mismatch:" + contributionId); return false; }
            if (!_claimed.Add(contributionId)) { Refusals.Add("duplicate-claim:" + contributionId); return false; }
            _points[author] = PointsOf(author) + Weights[entry.Kind];
            _awards.Add(new KeyValuePair<string, string>(author, contributionId));
            return true;
        }

        /** Standings: points descending, then author name ordinal - a total deterministic order. */
        public string Standings()
        {
            var names = new List<string>(_points.Keys);
            names.Sort((a, b) =>
            {
                int byPoints = _points[b].CompareTo(_points[a]);
                return byPoints != 0 ? byPoints : string.CompareOrdinal(a, b);
            });
            var parts = new List<string>();
            foreach (string name in names) parts.Add(name + ":" + _points[name] + ":" + Tier(_points[name]));
            return string.Join(", ", parts);
        }
    }

    private static string Tier(int points)
    {
        if (points >= 25) return "Maintainer";
        if (points >= 12) return "Recognized";
        if (points >= 5) return "Contributor";
        return "Novice";
    }

    /** Oracle: rebuild the tally by rescanning the ledger for the awarded ids. */
    private static Dictionary<string, int> BruteForce(Ledger ledger, IReadOnlyList<KeyValuePair<string, string>> awards)
    {
        var result = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, string> award in awards)
        {
            if (ledger.Find(award.Value).Author != award.Key) throw new Exception("oracle: author drift on " + award.Value);
            result[award.Key] = (result.TryGetValue(award.Key, out int cur) ? cur : 0) + Weights[ledger.Find(award.Value).Kind];
        }
        return result;
    }

    private static Ledger Build(params (string Id, string Author, Kind Kind)[] entries)
    {
        var ledger = new Ledger();
        foreach ((string id, string author, Kind kind) in entries) ledger.Add(id, author, kind);
        return ledger;
    }

    public static void Main()
    {
        Ledger ledger = Build(("c1", "ada", Kind.Commit), ("c2", "ada", Kind.Review), ("c3", "bob", Kind.Issue),
                              ("c4", "cyd", Kind.Docs), ("c5", "bob", Kind.Commit), ("c6", "bob", Kind.Review));
        Check(ledger.FirstBrokenIndex() == -1 && ledger.Entries.Count == 6, "fresh ledger chain verifies");

        var recognition = new Recognition(ledger);
        string[] claims = { "ada/c1", "ada/c2", "bob/c3", "cyd/c4", "bob/c5", "bob/c6" };
        foreach (string claim in claims)
        {
            int slash = claim.IndexOf('/');
            Check(recognition.Cite(claim.Substring(0, slash), claim.Substring(slash + 1)), "valid claim accepted: " + claim);
        }
        Check(recognition.PointsOf("ada") == 5 && recognition.PointsOf("bob") == 9 && recognition.PointsOf("cyd") == 1,
              "closed-form weighted totals");

        bool tallyMatches = true;
        var replay = Build(("c1", "ada", Kind.Commit), ("c2", "ada", Kind.Review), ("c3", "bob", Kind.Issue),
                           ("c4", "cyd", Kind.Docs), ("c5", "bob", Kind.Commit), ("c6", "bob", Kind.Review));
        var stepwise = new Recognition(replay);
        var awards = new List<KeyValuePair<string, string>>();
        foreach (string claim in claims)
        {
            int slash = claim.IndexOf('/');
            stepwise.Cite(claim.Substring(0, slash), claim.Substring(slash + 1));
            awards.Add(new KeyValuePair<string, string>(claim.Substring(0, slash), claim.Substring(slash + 1)));
            Dictionary<string, int> oracle = BruteForce(replay, awards);
            if (oracle.Count != stepwise.Points.Count) tallyMatches = false;
            foreach (KeyValuePair<string, int> pair in oracle)
                if (stepwise.PointsOf(pair.Key) != pair.Value) tallyMatches = false;
        }
        Check(tallyMatches, "incremental tally equals a brute-force rescan after every award (oracle)");

        Check(!recognition.Cite("bob", "c3") && recognition.Refusals[recognition.Refusals.Count - 1] == "duplicate-claim:c3",
              "duplicate claim refused");
        Check(!recognition.Cite("bob", "c99") && recognition.Refusals[recognition.Refusals.Count - 1] == "unknown-contribution:c99"
              && !recognition.Cite("bob", "c1") && recognition.Refusals[recognition.Refusals.Count - 1] == "author-mismatch:c1"
              && recognition.PointsOf("bob") == 9,
              "unknown citation and misattributed claim refused, and refusals never change the tally");

        Ledger tampered = Build(("c1", "ada", Kind.Commit), ("c2", "bob", Kind.Issue));
        tampered.Tamper(1, "mallory");
        Check(tampered.FirstBrokenIndex() == 1 && !new Recognition(tampered).Cite("mallory", "c2"),
              "tampering located at the exact index and the claim against a broken chain is refused");

        var emptyRecognition = new Recognition(Build());
        Check(Build().FirstBrokenIndex() == -1 && emptyRecognition.Standings() == "", "empty ledger boundary");

        Ledger single = Build(("s1", "zoe", Kind.Docs));
        var singleRecognition = new Recognition(single);
        singleRecognition.Cite("zoe", "s1");
        Check(singleRecognition.Standings() == "zoe:1:Novice", "single-entry ledger boundary");

        Ledger tie = Build(("t1", "zeta", Kind.Issue), ("t2", "alpha", Kind.Issue));
        var tieRecognition = new Recognition(tie);
        tieRecognition.Cite("zeta", "t1");
        tieRecognition.Cite("alpha", "t2");
        Check(Tier(4) == "Novice" && Tier(5) == "Contributor" && Tier(11) == "Contributor" && Tier(12) == "Recognized"
              && Tier(24) == "Recognized" && Tier(25) == "Maintainer"
              && tieRecognition.Standings() == "alpha:4:Novice, zeta:4:Novice",
              "tier thresholds inclusive at the boundary and equal points break ties by name");

        Console.WriteLine("ledger entries: " + ledger.Entries.Count + ", awards: " + recognition.Awards.Count
                          + ", refusals: " + recognition.Refusals.Count);
        Console.WriteLine("standings: " + recognition.Standings());
        Console.WriteLine("PASS");
    }
}
