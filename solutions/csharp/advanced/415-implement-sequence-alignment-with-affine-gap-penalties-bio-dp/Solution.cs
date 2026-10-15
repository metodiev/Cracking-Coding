// Task 415 - Implement sequence alignment with affine gap penalties (bio-DP). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Gotoh's algorithm aligns two strings under affine gap costs. Cost model (documented):
//   * substitution: MISMATCH = 2, matched characters cost 0,
//   * a gap of length L costs OPEN + (L - 1) * EXTEND, with OPEN = 2 and EXTEND = 1:
//     opening the run costs OPEN, every extra character costs EXTEND.
// Three DP states drive the recurrence:
//   * X - the column ends with a (mis)match of the last two characters,
//   * Y - the column ends with a gap in b (a character of a is deleted),
//   * Z - the column ends with a gap in a (a character of b is inserted).
// Gap-to-gap switches are allowed so that with EXTEND == OPEN the optimum equals
// Needleman-Wunsch with a linear gap cost of OPEN per character (asserted below).
// An alignment is a list of ops {"=", x, y} match; {"!", x, y} substitution;
// {"-", x, ""} deletion; {"+", "", y} insertion; replaying the ops rebuilds b.
using System;
using System.Collections.Generic;
using System.Text;

internal class GotohResult
{
    public readonly int Cost;
    public readonly List<string[]> Ops;

    public GotohResult(int cost, List<string[]> ops)
    {
        Cost = cost;
        Ops = ops;
    }
}

public static class Solution
{
    private const int Mismatch = 2;
    private const int OpenCost = 2;
    private const int Extend = 1;
    private const int Inf = int.MaxValue / 4;

    private static int ArgMin3(int a, int b, int c)
    {
        if (a <= b && a <= c) return 0;
        if (b <= c) return 1;
        return 2;
    }

    /// <summary>Returns (cost, ops) for the optimal affine-gap global alignment.</summary>
    internal static GotohResult Gotoh(string a, string b, int mismatch, int openCost, int extend)
    {
        int n = a.Length, m = b.Length;
        var x = new int[n + 1, m + 1];
        var y = new int[n + 1, m + 1];
        var z = new int[n + 1, m + 1];
        for (int i = 0; i <= n; i++)
        {
            for (int j = 0; j <= m; j++) { x[i, j] = Inf; y[i, j] = Inf; z[i, j] = Inf; }
        }
        x[0, 0] = 0;
        for (int i = 1; i <= n; i++) y[i, 0] = openCost + (i - 1) * extend;
        for (int j = 1; j <= m; j++) z[0, j] = openCost + (j - 1) * extend;
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                int sub = a[i - 1] == b[j - 1] ? 0 : mismatch;
                x[i, j] = Math.Min(Math.Min(x[i - 1, j - 1], y[i - 1, j - 1]), z[i - 1, j - 1]) + sub;
                y[i, j] = Math.Min(Math.Min(x[i - 1, j] + openCost, z[i - 1, j] + openCost), y[i - 1, j] + extend);
                z[i, j] = Math.Min(Math.Min(x[i, j - 1] + openCost, y[i, j - 1] + openCost), z[i, j - 1] + extend);
            }
        }
        int state = ArgMin3(x[n, m], y[n, m], z[n, m]);
        int cost = state == 0 ? x[n, m] : state == 1 ? y[n, m] : z[n, m];
        var ops = new LinkedList<string[]>();
        int ia = n, jb = m;
        while (ia > 0 || jb > 0)
        {
            if (state == 0)
            {
                int sub = a[ia - 1] == b[jb - 1] ? 0 : mismatch;
                ops.AddFirst(new[] { sub == 0 ? "=" : "!", a[ia - 1].ToString(), b[jb - 1].ToString() });
                int k = ArgMin3(x[ia - 1, jb - 1], y[ia - 1, jb - 1], z[ia - 1, jb - 1]);
                int cand = k == 0 ? x[ia - 1, jb - 1] : k == 1 ? y[ia - 1, jb - 1] : z[ia - 1, jb - 1];
                if (cand + sub != x[ia, jb]) throw new Exception("X backtrack");
                state = k;
                ia--;
                jb--;
            }
            else if (state == 1)
            {
                ops.AddFirst(new[] { "-", a[ia - 1].ToString(), "" });
                int k = ArgMin3(x[ia - 1, jb] + openCost, z[ia - 1, jb] + openCost, y[ia - 1, jb] + extend);
                int cand = k == 0 ? x[ia - 1, jb] + openCost : k == 1 ? z[ia - 1, jb] + openCost : y[ia - 1, jb] + extend;
                if (cand != y[ia, jb]) throw new Exception("Y backtrack");
                state = k == 0 ? 0 : k == 1 ? 2 : 1;
                ia--;
            }
            else
            {
                ops.AddFirst(new[] { "+", "", b[jb - 1].ToString() });
                int k = ArgMin3(x[ia, jb - 1] + openCost, y[ia, jb - 1] + openCost, z[ia, jb - 1] + extend);
                int cand = k == 0 ? x[ia, jb - 1] + openCost : k == 1 ? y[ia, jb - 1] + openCost : z[ia, jb - 1] + extend;
                if (cand != z[ia, jb]) throw new Exception("Z backtrack");
                state = k;
                jb--;
            }
        }
        return new GotohResult(cost, new List<string[]>(ops));
    }

    /// <summary>Linear gap cost: every gap character costs `gap` regardless of run length.</summary>
    public static int NeedlemanWunsch(string a, string b, int gap, int mismatch)
    {
        int n = a.Length, m = b.Length;
        var dp = new int[n + 1, m + 1];
        for (int i = 1; i <= n; i++) dp[i, 0] = i * gap;
        for (int j = 1; j <= m; j++) dp[0, j] = j * gap;
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                int sub = a[i - 1] == b[j - 1] ? 0 : mismatch;
                dp[i, j] = Math.Min(Math.Min(dp[i - 1, j] + gap, dp[i, j - 1] + gap), dp[i - 1, j - 1] + sub);
            }
        }
        return dp[n, m];
    }

    public static string Replay(string a, List<string[]> ops)
    {
        var sb = new StringBuilder();
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

    /// <summary>Recompute the cost from the op pattern alone, validating the cost model.</summary>
    public static int OpsCost(List<string[]> ops, int mismatch, int openCost, int extend)
    {
        int total = 0, i = 0;
        while (i < ops.Count)
        {
            string kind = ops[i][0];
            if (kind == "!")
            {
                total += mismatch;
                i++;
            }
            else if (kind == "-" || kind == "+")
            {
                int run = 0;
                while (i < ops.Count && ops[i][0] == kind) { run++; i++; }
                total += openCost + (run - 1) * extend;
            }
            else
            {
                i++;
            }
        }
        return total;
    }

    private static int seed = 20270;

    private static string LcgString(int length, string alphabet)
    {
        var sb = new StringBuilder();
        for (int k = 0; k < length; k++)
        {
            seed = (seed * 75 + 74) % 65537;
            sb.Append(alphabet[seed % alphabet.Length]);
        }
        return sb.ToString();
    }

    public static void Main()
    {
        // equal strings align for free
        var res = Gotoh("GATTACA", "GATTACA", Mismatch, OpenCost, Extend);
        Check(res.Cost == 0 && res.Ops.Count == 7, "equal strings");
        foreach (var op in res.Ops) Check(op[0] == "=", "equal all matches");

        // a single substitution (2) beats delete+insert (2 + 2)
        res = Gotoh("a", "b", Mismatch, OpenCost, Extend);
        Check(res.Cost == 2 && res.Ops.Count == 1 && res.Ops[0][0] == "!", "single sub");

        // one insertion: a single gap of length 1 costs OPEN = 2
        res = Gotoh("ACGT", "ACCGT", Mismatch, OpenCost, Extend);
        Check(res.Cost == 2 && OpsCost(res.Ops, Mismatch, OpenCost, Extend) == 2, "one insertion");
        Check(Replay("ACGT", res.Ops) == "ACCGT", "one insertion replay");

        // one gap of length 2 costs OPEN + EXTEND = 3; two separate gaps would cost 4
        res = Gotoh("acgt", "acgggt", Mismatch, OpenCost, Extend);
        Check(res.Cost == 3, "gap 2 cost");
        Check(Replay("acgt", res.Ops) == "acgggt", "gap 2 replay");
        Check(OpsCost(res.Ops, Mismatch, OpenCost, Extend) == 3, "gap 2 ops cost");
        Check(NeedlemanWunsch("acgt", "acgggt", 2, Mismatch) == 4, "linear two gaps");
        Check(NeedlemanWunsch("ab", "aZZb", 2, Mismatch) == 4, "linear ab");
        var ab = Gotoh("ab", "aZZb", Mismatch, OpenCost, Extend);
        Check(ab.Cost == 3 && OpsCost(ab.Ops, Mismatch, OpenCost, Extend) == 3, "affine ab");
        Check(Gotoh("acgt", "acgggt", 2, 2, 2).Cost == NeedlemanWunsch("acgt", "acgggt", 2, 2), "extend equal open");

        // empty-string cases: a gap of length L costs OPEN + (L-1)*EXTEND
        var empty = Gotoh("", "", Mismatch, OpenCost, Extend);
        Check(empty.Cost == 0 && empty.Ops.Count == 0, "empty pair");
        res = Gotoh("abc", "", Mismatch, OpenCost, Extend);
        Check(res.Cost == OpenCost + 2 * Extend && res.Cost == 4 && res.Ops.Count == 3, "pure deletion");
        foreach (var op in res.Ops) Check(op[0] == "-", "deletion ops");
        res = Gotoh("", "abc", Mismatch, OpenCost, Extend);
        Check(res.Cost == 4 && res.Ops.Count == 3, "pure insertion");
        Check(Gotoh("", "AAAAA", Mismatch, OpenCost, Extend).Cost == OpenCost + 4 * Extend, "long insertion");
        Check(Gotoh("", "AAAAA", Mismatch, OpenCost, Extend).Cost == 6, "long insertion value");
        Check(NeedlemanWunsch("", "AAAAA", 2, Mismatch) == 10, "linear long insertion");

        // longer hand case: 3-character insertion in the middle
        res = Gotoh("ABCD", "ABXYZCD", Mismatch, OpenCost, Extend);
        Check(res.Cost == OpenCost + 2 * Extend && res.Cost == 4, "middle insertion cost");
        Check(Replay("ABCD", res.Ops) == "ABXYZCD", "middle insertion replay");
        Check(OpsCost(res.Ops, Mismatch, OpenCost, Extend) == 4, "middle insertion ops cost");

        // replay + cost-model consistency on the hand pairs
        var hand = new[] { new[] { "kitten", "sitting" }, new[] { "sunday", "saturday" }, new[] { "flaw", "lawn" } };
        foreach (var pair in hand)
        {
            var r2 = Gotoh(pair[0], pair[1], Mismatch, OpenCost, Extend);
            Check(Replay(pair[0], r2.Ops) == pair[1], "hand replay " + pair[0]);
            Check(OpsCost(r2.Ops, Mismatch, OpenCost, Extend) == r2.Cost, "hand ops cost " + pair[0]);
        }

        // with EXTEND == OPEN the affine optimum equals Needleman-Wunsch
        const string alphabet = "acgt";
        for (int round = 0; round < 8; round++)
        {
            int la = 25 + seed % 40;
            string a = LcgString(la, alphabet);
            int lb = 20 + seed % 45;
            string b = LcgString(lb, alphabet);
            var affine = Gotoh(a, b, 2, 3, 3);
            int linear = NeedlemanWunsch(a, b, 3, 2);
            Check(affine.Cost == linear, "affine == linear round " + round);
            Check(Replay(a, affine.Ops) == b, "lcg replay " + round);
            Check(OpsCost(affine.Ops, 2, 3, 3) == affine.Cost, "lcg ops cost " + round);
            // a cheaper affine extension can never lose against the linear model
            Check(Gotoh(a, b, 2, 2, 1).Cost <= linear, "affine bound round " + round);
        }

        // gap-run counting sanity: a long deletion is one open plus many extends
        res = Gotoh("ABCDEFGH", "AB", Mismatch, OpenCost, Extend);
        Check(res.Cost == OpenCost + 5 * Extend && res.Cost == 7, "long deletion cost");
        int deletes = 0;
        foreach (var op in res.Ops) if (op[0] == "-") deletes++;
        Check(deletes == 6 && OpsCost(res.Ops, Mismatch, OpenCost, Extend) == 7, "long deletion run");
        Check(Replay("ABCDEFGH", res.Ops) == "AB", "long deletion replay");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
