// Task 986 - Implement open dataset packaging and licensing automation for practice problems.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// SPDX-style license expressions are parsed with SPDX precedence (AND tighter than OR) and checked against
// a bundle policy: an atom must satisfy the policy, AND needs every child admissible, OR needs one child
// admissible and reports the smallest alternative (ties by canonical text). Packaging emits a deterministic
// manifest (problems by number, files by path, per-file FNV-1a checksum) and a bundle digest. Verified:
// precedence, refusals with positions, OR selection, share-alike propagation, manifest order independence
// and digest equality with an independent rebuild (oracle), and the empty / single-problem boundaries.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }

    private sealed class License
    {
        public readonly bool Commercial, Copyleft, Attribution;
        public License(bool commercial, bool copyleft, bool attribution)
        { Commercial = commercial; Copyleft = copyleft; Attribution = attribution; }
    }

    private static readonly Dictionary<string, License> Catalog = new Dictionary<string, License>(StringComparer.Ordinal)
    {
        ["CC0-1.0"] = new License(true, false, false),
        ["CC-BY-4.0"] = new License(true, false, true),
        ["CC-BY-SA-4.0"] = new License(true, true, true),
        ["CC-BY-NC-4.0"] = new License(false, false, true),
        ["MIT"] = new License(true, false, true),
        ["Apache-2.0"] = new License(true, false, true),
    };

    /** One node type: atoms have Id set, AND/OR nodes have Left/Right set. */
    private sealed class Node
    {
        public readonly string Id; public readonly Node Left, Right; public readonly bool IsAnd;
        public Node(string id) { Id = id; }
        public Node(bool isAnd, Node left, Node right) { IsAnd = isAnd; Left = left; Right = right; }
        public bool IsAtom => Id != null;
        public string Canonical() => IsAtom ? Id : "(" + Left.Canonical() + (IsAnd ? " AND " : " OR ") + Right.Canonical() + ")";
    }

    private static void Skip(string s, ref int pos) { while (pos < s.Length && s[pos] == ' ') pos++; }
    /** Keyword match: skips leading spaces and rewinds when it does not match. */
    private static bool Match(string s, ref int pos, string word)
    {
        int save = pos;
        Skip(s, ref pos);
        int after = pos + word.Length;
        bool hit = after <= s.Length && string.CompareOrdinal(s, pos, word, 0, word.Length) == 0
                   && (after == s.Length || s[after] == ' ' || s[after] == '(');
        pos = hit ? after : save;
        return hit;
    }

    /** Or := And ('OR' And)*, And := Atom ('AND' Atom)*, Atom := ID | '(' Or ')'. */
    private static Node Parse(string text)
    {
        int pos = 0;
        Node node = ParseOr(text, ref pos);
        Skip(text, ref pos);
        if (pos != text.Length) throw new Exception("trailing-input@" + pos);
        return node;
    }
    private static Node ParseOr(string s, ref int pos)
    {
        Node left = ParseAnd(s, ref pos);
        while (Match(s, ref pos, "OR")) left = new Node(false, left, ParseAnd(s, ref pos));
        return left;
    }
    private static Node ParseAnd(string s, ref int pos)
    {
        Node left = ParseAtom(s, ref pos);
        while (Match(s, ref pos, "AND")) left = new Node(true, left, ParseAtom(s, ref pos));
        return left;
    }
    private static Node ParseAtom(string s, ref int pos)
    {
        Skip(s, ref pos);
        if (pos < s.Length && s[pos] == '(')
        {
            pos++;
            Node inner = ParseOr(s, ref pos);
            Skip(s, ref pos);
            if (pos >= s.Length || s[pos] != ')') throw new Exception("unterminated-group@" + pos);
            pos++;
            return inner;
        }
        int start = pos;
        while (pos < s.Length && (char.IsLetterOrDigit(s[pos]) || s[pos] == '-' || s[pos] == '.')) pos++;
        if (pos == start) throw new Exception("expected-license@" + pos);
        string id = s.Substring(start, pos - start);
        if (!Catalog.ContainsKey(id)) throw new Exception("unknown-license:" + id + "@" + start);
        return new Node(id);
    }

    private sealed class Policy
    {
        public readonly bool Commercial, AllowCopyleft;
        public Policy(bool commercial, bool allowCopyleft) { Commercial = commercial; AllowCopyleft = allowCopyleft; }
    }

    private sealed class Outcome
    {
        public bool Admitted, ShareAlike, Attribution; public string Reason = "";
        public readonly List<string> Selected = new List<string>();
        public string EffectiveLicense()
        {
            var ids = new List<string>(Selected);
            ids.Sort(StringComparer.Ordinal);
            return string.Join(" AND ", ids);
        }
    }

    private static bool Admissible(Node node, Policy policy, out string reason)
    {
        if (node.IsAtom)
        {
            License license = Catalog[node.Id];
            reason = policy.Commercial && !license.Commercial ? "non-commercial-not-allowed:" + node.Id
                : !policy.AllowCopyleft && license.Copyleft ? "copyleft-not-allowed:" + node.Id : "";
            return reason.Length == 0;
        }
        bool leftOk = Admissible(node.Left, policy, out string leftReason);
        bool rightOk = Admissible(node.Right, policy, out string rightReason);
        if (node.IsAnd) { reason = leftOk ? rightReason : leftReason; return leftOk && rightOk; }
        reason = leftOk || rightOk ? "" : leftReason + "; " + rightReason;
        return leftOk || rightOk;
    }

    private static void Collect(Node node, Policy policy, Outcome outcome)
    {
        if (node.IsAtom)
        {
            License license = Catalog[node.Id];
            outcome.Selected.Add(node.Id);
            outcome.ShareAlike |= license.Copyleft;
            outcome.Attribution |= license.Attribution;
            return;
        }
        if (node.IsAnd)
        {
            Collect(node.Left, policy, outcome);
            Collect(node.Right, policy, outcome);
            return;
        }
        bool leftOk = Admissible(node.Left, policy, out _);
        bool rightOk = Admissible(node.Right, policy, out _);
        // Smallest admissible alternative first (by canonical text) - the same choice every run.
        bool takeLeft = leftOk && (!rightOk || string.CompareOrdinal(node.Left.Canonical(), node.Right.Canonical()) <= 0);
        Collect(takeLeft ? node.Left : node.Right, policy, outcome);
    }

    private static Outcome Evaluate(string expression, Policy policy)
    {
        var outcome = new Outcome();
        Node node;
        try { node = Parse(expression); }
        catch (Exception ex) { outcome.Reason = ex.Message; return outcome; }
        if (!Admissible(node, policy, out string reason)) { outcome.Reason = reason; return outcome; }
        outcome.Admitted = true;
        Collect(node, policy, outcome);
        return outcome;
    }

    private static ulong Fnv(string text, ulong seed)
    {
        unchecked
        {
            ulong h = 14695981039346656037UL ^ seed;
            foreach (byte b in Encoding.UTF8.GetBytes(text)) { h ^= b; h *= 1099511628211UL; }
            return h;
        }
    }

    /** Deterministic manifest: problems by number, files by path. */
    private static List<string> Manifest(List<(int Number, string Slug, string Expression, string[] Files)> problems,
                                         Policy policy, List<string> report)
    {
        var ordered = new List<(int Number, string Slug, string Expression, string[] Files)>(problems);
        ordered.Sort((a, b) => a.Number.CompareTo(b.Number));
        var lines = new List<string>();
        foreach ((int number, string slug, string expression, string[] files) in ordered)
        {
            Outcome outcome = Evaluate(expression, policy);
            string id = number.ToString("D3", CultureInfo.InvariantCulture);
            if (!outcome.Admitted) { report.Add("excluded " + id + ": " + outcome.Reason); continue; }
            report.Add("included " + id + ": " + outcome.EffectiveLicense() + (outcome.ShareAlike ? " (share-alike bundle)" : ""));
            var sorted = new List<string>(files);
            sorted.Sort(StringComparer.Ordinal);
            ulong fileHash = 0;
            foreach (string file in sorted) fileHash = Fnv(file, fileHash);
            lines.Add(id + "|" + slug + "|" + outcome.EffectiveLicense() + "|"
                      + sorted.Count.ToString(CultureInfo.InvariantCulture) + "|" + fileHash.ToString("x16", CultureInfo.InvariantCulture));
        }
        return lines;
    }
    private static ulong BundleDigest(List<string> manifest) => Fnv(string.Join("\n", manifest), 0);

    public static void Main()
    {
        Check(Parse("MIT OR Apache-2.0 AND CC0-1.0").Canonical() == "(MIT OR (Apache-2.0 AND CC0-1.0))"
              && Parse("(MIT OR Apache-2.0) AND CC0-1.0").Canonical() == "((MIT OR Apache-2.0) AND CC0-1.0)",
              "AND binds tighter than OR and explicit grouping overrides precedence");
        var commercial = new Policy(true, false);
        var nonCommercial = new Policy(false, false);
        var open = new Policy(true, true);
        Check(!Evaluate("CC-BY-NC-4.0", commercial).Admitted && Evaluate("CC-BY-NC-4.0", nonCommercial).Admitted
              && !Evaluate("MIT AND CC-BY-NC-4.0", commercial).Admitted
              && Evaluate("CC-BY-SA-4.0", commercial).Reason == "copyleft-not-allowed:CC-BY-SA-4.0"
              && Evaluate("BSD-3-Clause", commercial).Reason == "unknown-license:BSD-3-Clause@0"
              && Evaluate("(MIT", commercial).Reason == "unterminated-group@4"
              && Evaluate("", commercial).Reason == "expected-license@0",
              "policy refusals and malformed expressions reported with a precise reason");
        Outcome alternative = Evaluate("CC-BY-NC-4.0 OR CC-BY-4.0", commercial);
        Outcome shareAlike = Evaluate("MIT OR CC-BY-SA-4.0", open);
        Check(alternative.Admitted && alternative.EffectiveLicense() == "CC-BY-4.0"
              && shareAlike.ShareAlike && shareAlike.Attribution && shareAlike.EffectiveLicense() == "CC-BY-SA-4.0",
              "OR picks the smallest admissible alternative and share-alike propagates");
        var problems = new List<(int, string, string, string[])>
        {
            (7, "seven", "CC0-1.0", new[] { "tests.txt", "prompt.md" }),
            (3, "three", "CC-BY-NC-4.0", new[] { "prompt.md" }), (5, "five", "MIT OR CC-BY-SA-4.0", new[] { "solution.py" }),
        };
        var report = new List<string>();
        List<string> manifest = Manifest(problems, open, report);
        string five = "005|five|CC-BY-SA-4.0|1|" + Fnv("solution.py", 0).ToString("x16", CultureInfo.InvariantCulture);
        string seven = "007|seven|CC0-1.0|2|" + Fnv("tests.txt", Fnv("prompt.md", 0)).ToString("x16", CultureInfo.InvariantCulture);
        var shuffled = new List<(int, string, string, string[])> { problems[2], problems[0], problems[1] };
        Check(manifest.Count == 2 && report.Count == 3
              && report[0] == "excluded 003: non-commercial-not-allowed:CC-BY-NC-4.0"
              && report[1] == "included 005: CC-BY-SA-4.0 (share-alike bundle)",
              "compliance report lists exclusions and inclusions");
        // Oracle: the lines and the digest are rebuilt from literal metadata, not from the packager's loop.
        Check(manifest[0] == five && manifest[1] == seven
              && Manifest(shuffled, open, new List<string>())[0] == five
              && BundleDigest(manifest) == Fnv(five + "\n" + seven, 0),
              "manifest matches an independent rebuild and ignores insertion order");
        Check(BundleDigest(Manifest(new List<(int, string, string, string[])>(), open, new List<string>())) == Fnv("", 0)
              && Manifest(new List<(int, string, string, string[])> { (1, "one", "CC0-1.0", new string[0]) },
                          open, new List<string>()).Count == 1,
              "empty and single-problem bundle boundaries");
        Console.WriteLine("licenses: " + Catalog.Count + ", manifest lines: " + manifest.Count
                          + ", digest: " + BundleDigest(manifest).ToString("x16", CultureInfo.InvariantCulture));
        Console.WriteLine("PASS");
    }
}
