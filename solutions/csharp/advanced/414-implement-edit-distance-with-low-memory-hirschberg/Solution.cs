// Task 414 - Implement edit distance with low memory (Hirschberg). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Classic Levenshtein DP uses an O(n*m) table. Hirschberg's algorithm recovers an
// optimal alignment with O(n) auxiliary memory and O(n*m) time:
//   * split b in half (Myers-Miller variant of the split), compute the forward row
//     f[i] = dist(a[:i], b[:mid]) and the backward row g[i] = dist(a[i:], b[mid:])
//     with two rolling rows each — the backward row reverses both strings because
//     reversal preserves edit distance and turns suffixes into prefixes,
//   * pick the split k minimizing f[k] + g[k] (smallest k on ties), recurse on
//     (a[:k], b[:mid]) and (a[k:], b[mid:]); base cases solve directly.
// An alignment is a list of ops {"=", x, y} match; {"!", x, y} substitution;
// {"-", x, ""} deletion; {"+", "", y} insertion. Replaying the ops from a rebuilds b.
using System;
using System.Collections.Generic;
using System.Linq;

internal class AlignResult
{
    public readonly int Dist;
    public readonly List<string[]> Ops;

    public AlignResult(int dist, List<string[]> ops)
    {
        Dist = dist;
        Ops = ops;
    }
}

public static class Solution
{
    private const int Base = 16;

    /// <summary>Oracle: full O(n*m) table over rolling rows.</summary>
    public static int FullDpDist(string a, string b)
    {
        int m = b.Length;
        var prev = new int[m + 1];
        for (int j = 0; j <= m; j++) prev[j] = j;
        for (int i = 1; i <= a.Length; i++)
        {
            var cur = new int[m + 1];
            cur[0] = i;
            for (int j = 1; j <= m; j++)
            {
                int sub = a[i - 1] == b[j - 1] ? 0 : 1;
                cur[j] = Math.Min(Math.Min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + sub);
            }
            prev = cur;
        }
        return prev[m];
    }

    /// <summary>f[i] = dist(a[:i], prefix) for every i; one rolling row over the prefix.</summary>
    public static int[] ForwardRow(string a, string prefix)
    {
        int m = prefix.Length;
        var prev = new int[m + 1];
        for (int j = 0; j <= m; j++) prev[j] = j;
        var outp = new int[a.Length + 1];
        outp[0] = prev[m];
        for (int i = 1; i <= a.Length; i++)
        {
            var cur = new int[m + 1];
            cur[0] = i;
            for (int j = 1; j <= m; j++)
            {
                int sub = a[i - 1] == prefix[j - 1] ? 0 : 1;
                cur[j] = Math.Min(Math.Min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + sub);
            }
            prev = cur;
            outp[i] = prev[m];
        }
        return outp;
    }

    /// <summary>g[i] = dist(a[i:], suffix) for every i (reversal turns suffixes into prefixes).</summary>
    public static int[] BackwardRow(string a, string suffix)
    {
        string ra = new string(a.Reverse().ToArray());
        string rs = new string(suffix.Reverse().ToArray());
        int m = rs.Length;
        var prev = new int[m + 1];
        for (int j = 0; j <= m; j++) prev[j] = j;
        var collected = new int[a.Length + 1];
        collected[0] = prev[m];
        for (int p = 1; p <= ra.Length; p++)
        {
            var cur = new int[m + 1];
            cur[0] = p;
            for (int j = 1; j <= m; j++)
            {
                int sub = ra[p - 1] == rs[j - 1] ? 0 : 1;
                cur[j] = Math.Min(Math.Min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + sub);
            }
            prev = cur;
            collected[p] = prev[m];
        }
        int n = a.Length;
        var outp = new int[n + 1];
        for (int i = 0; i <= n; i++) outp[i] = collected[n - i];
        return outp;
    }

    /// <summary>Full-table DP with backtracking; used for the recursion base cases.</summary>
    internal static AlignResult DirectAlign(string a, string b)
    {
        int n = a.Length, m = b.Length;
        var dp = new int[n + 1, m + 1];
        for (int i = 1; i <= n; i++) dp[i, 0] = i;
        for (int j = 1; j <= m; j++) dp[0, j] = j;
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                int sub = a[i - 1] == b[j - 1] ? 0 : 1;
                dp[i, j] = Math.Min(Math.Min(dp[i - 1, j] + 1, dp[i, j - 1] + 1), dp[i - 1, j - 1] + sub);
            }
        }
        var ops = new LinkedList<string[]>();
        int x = n, y = m;
        while (x > 0 || y > 0)
        {
            if (x > 0 && y > 0 && a[x - 1] == b[y - 1] && dp[x, y] == dp[x - 1, y - 1])
            {
                ops.AddFirst(new[] { "=", a[x - 1].ToString(), b[y - 1].ToString() });
                x--; y--;
            }
            else if (x > 0 && y > 0 && dp[x, y] == dp[x - 1, y - 1] + 1)
            {
                ops.AddFirst(new[] { "!", a[x - 1].ToString(), b[y - 1].ToString() });
                x--; y--;
            }
            else if (x > 0 && dp[x, y] == dp[x - 1, y] + 1)
            {
                ops.AddFirst(new[] { "-", a[x - 1].ToString(), "" });
                x--;
            }
            else
            {
                ops.AddFirst(new[] { "+", "", b[y - 1].ToString() });
                y--;
            }
        }
        return new AlignResult(dp[n, m], ops.ToList());
    }

    internal static AlignResult HirschbergAlign(string a, string b, int baseSize)
    {
        if (a.Length == 0)
        {
            return new AlignResult(b.Length, b.Select(c => new[] { "+", "", c.ToString() }).ToList());
        }
        if (b.Length == 0)
        {
            return new AlignResult(a.Length, a.Select(c => new[] { "-", c.ToString(), "" }).ToList());
        }
        if (Math.Min(a.Length, b.Length) <= Math.Max(baseSize, 1)) return DirectAlign(a, b);
        int mid = b.Length / 2;
        var f = ForwardRow(a, b.Substring(0, mid));
        var g = BackwardRow(a, b.Substring(mid));
        int split = 0, best = int.MaxValue;
        for (int i = 0; i <= a.Length; i++)
        {
            if (f[i] + g[i] < best) { best = f[i] + g[i]; split = i; }
        }
        var left = HirschbergAlign(a.Substring(0, split), b.Substring(0, mid), baseSize);
        var right = HirschbergAlign(a.Substring(split), b.Substring(mid), baseSize);
        var ops = new List<string[]>(left.Ops);
        ops.AddRange(right.Ops);
        return new AlignResult(left.Dist + right.Dist, ops);
    }

    /// <summary>Apply an alignment to a; returns the produced string.</summary>
    public static string Replay(string a, List<string[]> ops)
    {
        var sb = new System.Text.StringBuilder();
        int ai = 0;
        foreach (var op in ops)
        {
            if (op[0] == "=" || op[0] == "!" || op[0] == "-")
            {
                if (a[ai] != op[1][0]) throw new Exception("bad alignment column");
                ai++;
            }
            if (op[0] == "=" || op[0] == "!" || op[0] == "+") sb.Append(op[2]);
        }
        if (ai != a.Length) throw new Exception("source not fully consumed");
        return sb.ToString();
    }

    public static int CountKind(List<string[]> ops, string kind) => ops.Count(op => op[0] == kind);

    private static int seed = 4242;

    private static string LcgString(int length, string alphabet)
    {
        var sb = new System.Text.StringBuilder();
        for (int k = 0; k < length; k++)
        {
            seed = (seed * 75 + 74) % 65537;
            sb.Append(alphabet[seed % alphabet.Length]);
        }
        return sb.ToString();
    }

    public static void Main()
    {
        // hand-derived distances (Levenshtein, substitution cost 1)
        Check(FullDpDist("kitten", "sitting") == 3, "kitten sitting oracle");
        Check(FullDpDist("flaw", "lawn") == 2, "flaw lawn oracle");
        Check(FullDpDist("sunday", "saturday") == 3, "sunday saturday oracle");
        Check(FullDpDist("", "") == 0, "empty oracle");
        Check(FullDpDist("a", "b") == 1, "single sub");
        Check(FullDpDist("abc", "") == 3, "deletions oracle");
        Check(FullDpDist("abc", "abc") == 0, "equal oracle");
        Check(FullDpDist("abc", "yabd") == 2, "abc yabd oracle");

        var res = HirschbergAlign("kitten", "sitting", Base);
        Check(res.Dist == 3, "kitten sitting hirschberg");
        Check(Replay("kitten", res.Ops) == "sitting", "kitten replay");
        Check(CountKind(res.Ops, "=") == 4, "kitten matches forced");
        Check(CountKind(res.Ops, "!") == 2, "kitten subs forced");
        Check(CountKind(res.Ops, "+") == 1, "kitten insert forced");
        Check(CountKind(res.Ops, "-") == 0, "kitten deletes none");

        res = HirschbergAlign("flaw", "lawn", 1);
        Check(res.Dist == 2, "flaw lawn hirschberg");
        Check(Replay("flaw", res.Ops) == "lawn", "flaw replay");
        Check(CountKind(res.Ops, "!") + CountKind(res.Ops, "-") + CountKind(res.Ops, "+") == 2, "flaw edits");
        Check(CountKind(res.Ops, "=") == 3, "flaw matches law");

        // base-case paths: empty strings, pure insertions/deletions, equal strings
        var ins = HirschbergAlign("", "abc", Base);
        Check(ins.Dist == 3 && ins.Ops.Count == 3, "pure insertions");
        var del = HirschbergAlign("abc", "", Base);
        Check(del.Dist == 3 && del.Ops.Count == 3, "pure deletions");
        var none = HirschbergAlign("", "", Base);
        Check(none.Dist == 0 && none.Ops.Count == 0, "empty pair");
        var same = HirschbergAlign("same", "same", Base);
        Check(same.Dist == 0 && Replay("same", same.Ops) == "same", "equal alignment");
        Check(CountKind(same.Ops, "=") == 4, "equal matches");

        // both base settings must agree with the full table on hand and random pairs
        var pairs = new[]
        {
            new[] { "kitten", "sitting" }, new[] { "flaw", "lawn" },
            new[] { "sunday", "saturday" }, new[] { "abc", "yabd" }
        };
        var bases = new[] { 0, 1, 3, 16, 1000 };
        foreach (var pair in pairs)
        {
            foreach (int baseSize in bases)
            {
                var r2 = HirschbergAlign(pair[0], pair[1], baseSize);
                Check(r2.Dist == FullDpDist(pair[0], pair[1]), "pair " + pair[0] + "/" + pair[1] + " base " + baseSize);
                Check(Replay(pair[0], r2.Ops) == pair[1], "pair replay " + pair[0] + " base " + baseSize);
            }
        }

        const string alphabet = "abcd";
        for (int round = 0; round < 10; round++)
        {
            int la = 30 + seed % 60;
            string a = LcgString(la, alphabet);
            int lb = 25 + seed % 75;
            string b = LcgString(lb, alphabet);
            int want = FullDpDist(a, b);
            foreach (int baseSize in new[] { 2, 16, 400 })
            {
                var r2 = HirschbergAlign(a, b, baseSize);
                Check(r2.Dist == want, "lcg round " + round + " base " + baseSize);
                Check(Replay(a, r2.Ops) == b, "lcg replay " + round + " base " + baseSize);
                Check(CountKind(r2.Ops, "=") + CountKind(r2.Ops, "!") + CountKind(r2.Ops, "-") == a.Length, "lcg consume a");
                Check(CountKind(r2.Ops, "=") + CountKind(r2.Ops, "!") + CountKind(r2.Ops, "+") == b.Length, "lcg consume b");
                Check(CountKind(r2.Ops, "!") + CountKind(r2.Ops, "-") + CountKind(r2.Ops, "+") == want, "lcg cost");
                foreach (var op in r2.Ops)
                {
                    if (op[0] == "=") Check(op[1] == op[2], "match equal");
                    if (op[0] == "!") Check(op[1] != op[2], "sub unequal");
                }
            }
        }

        // recurrence sanity: mid rows of the recursion match the full-table boundary values
        string sa = "alignment", sb = "alignmentx";
        int mid = sb.Length / 2;
        var f = ForwardRow(sa, sb.Substring(0, mid));
        var g = BackwardRow(sa, sb.Substring(mid));
        Check(f[sa.Length] == FullDpDist(sa, sb.Substring(0, mid)), "f whole");
        Check(g[0] == FullDpDist(sa, sb.Substring(mid)), "g whole");
        for (int i = 0; i <= sa.Length; i++)
        {
            Check(f[i] == FullDpDist(sa.Substring(0, i), sb.Substring(0, mid)), "f row " + i);
            Check(g[i] == FullDpDist(sa.Substring(i), sb.Substring(mid)), "g row " + i);
        }
        int sp = 0, bestSplit = int.MaxValue;
        for (int i = 0; i <= sa.Length; i++)
        {
            if (f[i] + g[i] < bestSplit) { bestSplit = f[i] + g[i]; sp = i; }
        }
        Check(bestSplit == FullDpDist(sa, sb), "split optimal");
        Check(sp >= 0 && sp <= sa.Length, "split range");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
