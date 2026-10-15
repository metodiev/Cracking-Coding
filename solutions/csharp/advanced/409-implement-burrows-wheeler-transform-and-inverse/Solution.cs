// Task 409 - Implement the Burrows-Wheeler transform and its inverse. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Cyclic BWT: sort all n cyclic rotations of s, take the last column, and record the *primary
// index* - the row holding the rotation that starts at position 0. Hand anchors: banana ->
// ("nnbaaa", 3), abab -> ("bbaa", 0), aaaa -> ("aaaa", 0). Reversal is only possible with the
// primary index: "ab" and "ba" both transform to "ba" but use primary 0 and 1.
//
// Inverse: with F = sorted(L) (the first column) the LF-mapping row_of(L[i]) -> position in F
// (counting equal characters in order) maps a rotation to the rotation starting one position
// earlier; walking n steps from the primary row and reversing the collected characters
// rebuilds the string. Ties between equal rotations are resolved by row index (stable sort).
//
// Sentinel variant: BWT(s + "$") with a unique smallest character needs no primary index - the
// row whose last character is "$" is the rotation starting at 0, so the same LF walk recovers
// s exactly ("banana" -> "annb$aa").
//
// Checks: hand values, LF-mapping is a permutation, multiset preserved, both inverses
// round-trip on LCG strings including the empty string, and the medium case (300 characters).
using System;
using System.Collections.Generic;
using System.Text;

internal class BwtResult
{
    public readonly string Last;
    public readonly int Primary;

    public BwtResult(string last, int primary)
    {
        Last = last;
        Primary = primary;
    }
}

internal static class BurrowsWheeler
{
    public const char Sentinel = '$';

    public static BwtResult BwtCyclic(string s)
    {
        int n = s.Length;
        if (n == 0) return new BwtResult("", 0);
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            for (int k = 0; k < n; k++)
            {
                char ca = s[(a + k) % n];
                char cb = s[(b + k) % n];
                if (ca != cb) return ca < cb ? -1 : 1;
            }
            return 0;                                // equal rotations keep index order (stable)
        });
        var last = new StringBuilder();
        int primary = -1;
        for (int i = 0; i < n; i++)
        {
            last.Append(s[(order[i] - 1 + n) % n]);
            if (order[i] == 0) primary = i;
        }
        return new BwtResult(last.ToString(), primary);
    }

    public static int[] LfMapping(string last)
    {
        int n = last.Length;
        var lf = new int[n];
        var starts = new SortedDictionary<char, int>();
        int total = 0;
        foreach (char ch in SortedUnique(last))
        {
            starts[ch] = total;
            total += Count(last, ch);
        }
        var seen = new Dictionary<char, int>();
        for (int i = 0; i < n; i++)
        {
            char ch = last[i];
            seen.TryGetValue(ch, out int rank);
            lf[i] = starts[ch] + rank;
            seen[ch] = rank + 1;
        }
        return lf;
    }

    private static List<char> SortedUnique(string s)
    {
        var set = new SortedSet<char>();
        foreach (char ch in s) set.Add(ch);
        return new List<char>(set);
    }

    private static int Count(string s, char ch)
    {
        int total = 0;
        foreach (char c in s) if (c == ch) total++;
        return total;
    }

    public static string InverseCyclic(string last, int primary)
    {
        int n = last.Length;
        if (n == 0) return "";
        var lf = LfMapping(last);
        int row = primary;
        var collected = new StringBuilder();
        for (int step = 0; step < n; step++)
        {
            collected.Append(last[row]);
            row = lf[row];
        }
        if (row != primary) throw new Exception("LF walk must return to start");
        var chars = collected.ToString().ToCharArray();
        Array.Reverse(chars);
        return new string(chars);
    }

    public static string BwtSentinel(string s)
    {
        if (s.IndexOf(Sentinel) != -1) throw new ArgumentException("text must not contain $");
        return BwtCyclic(s + Sentinel).Last;
    }

    public static string InverseSentinel(string last)
    {
        int n = last.Length - 1;
        int row = last.IndexOf(Sentinel);
        var lf = LfMapping(last);
        var collected = new StringBuilder();
        for (int step = 0; step < n + 1; step++)
        {
            collected.Append(last[row]);
            row = lf[row];
        }
        var chars = collected.ToString().ToCharArray();
        Array.Reverse(chars);
        return new string(chars, 0, n);              // reversed join is s + "$"
    }
}

public static class Solution
{
    private static int seed = 2718;

    private static string LcgString(int length, string alphabet)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < length; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            sb.Append(alphabet[seed % alphabet.Length]);
        }
        return sb.ToString();
    }

    private static string SortedChars(string s)
    {
        var chars = s.ToCharArray();
        Array.Sort(chars);
        return new string(chars);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckBwt(string s, string expectedLast, int expectedPrimary)
    {
        var result = BurrowsWheeler.BwtCyclic(s);
        Check(result.Last == expectedLast, "last " + s);
        Check(result.Primary == expectedPrimary, "primary " + s);
    }

    public static void Main()
    {
        CheckBwt("banana", "nnbaaa", 3);
        CheckBwt("abab", "bbaa", 0);
        CheckBwt("aaaa", "aaaa", 0);
        CheckBwt("", "", 0);
        CheckBwt("a", "a", 0);
        CheckBwt("ba", "ba", 1);
        CheckBwt("ab", "ba", 0);

        // same last column, different primary index -> different original
        Check(BurrowsWheeler.InverseCyclic("ba", 0) == "ab", "inverse ab");
        Check(BurrowsWheeler.InverseCyclic("ba", 1) == "ba", "inverse ba");
        var banana = BurrowsWheeler.BwtCyclic("banana");
        Check(BurrowsWheeler.InverseCyclic(banana.Last, banana.Primary) == "banana", "inverse banana");
        var miss = BurrowsWheeler.BwtCyclic("mississippi");
        Check(BurrowsWheeler.InverseCyclic(miss.Last, miss.Primary) == "mississippi", "inverse mississippi");

        Check(BurrowsWheeler.BwtSentinel("banana") == "annb$aa", "sentinel banana");
        Check(BurrowsWheeler.BwtSentinel("") == "$", "sentinel empty");
        Check(BurrowsWheeler.BwtSentinel("a") == "a$", "sentinel a");
        Check(BurrowsWheeler.InverseSentinel("$") == "", "inverse sentinel empty");
        Check(BurrowsWheeler.InverseSentinel("annb$aa") == "banana", "inverse sentinel banana");
        Check(BurrowsWheeler.InverseSentinel(BurrowsWheeler.BwtSentinel("abracadabra")) == "abracadabra",
            "inverse sentinel abracadabra");

        Check(SortedChars(miss.Last) == SortedChars("mississippi"), "multiset");
        int[] lf = BurrowsWheeler.LfMapping(miss.Last);
        var sortedLf = (int[])lf.Clone();
        Array.Sort(sortedLf);
        for (int i = 0; i < lf.Length; i++) Check(sortedLf[i] == i, "lf permutation " + i);

        foreach (string alphabet in new[] { "ab", "abc" })
        {
            for (int trial = 0; trial < 40; trial++)
            {
                string s = LcgString(trial % 41, alphabet);
                var result = BurrowsWheeler.BwtCyclic(s);
                Check(SortedChars(result.Last) == SortedChars(s), "lcg multiset " + s);
                Check(BurrowsWheeler.InverseCyclic(result.Last, result.Primary) == s, "lcg inverse " + s);
                Check(BurrowsWheeler.InverseSentinel(BurrowsWheeler.BwtSentinel(s)) == s, "lcg sentinel " + s);
            }
        }

        string medium = LcgString(300, "abc");
        var mediumResult = BurrowsWheeler.BwtCyclic(medium);
        Check(mediumResult.Last.Length == 300, "medium length");
        Check(BurrowsWheeler.InverseCyclic(mediumResult.Last, mediumResult.Primary) == medium, "medium inverse");
        Check(BurrowsWheeler.InverseSentinel(BurrowsWheeler.BwtSentinel(medium)) == medium, "medium sentinel");

        bool threw = false;
        try
        {
            BurrowsWheeler.BwtSentinel("bad$text");
        }
        catch (ArgumentException)
        {
            threw = true;
        }
        Check(threw, "sentinel validation");
        Console.WriteLine("PASS");
    }
}
