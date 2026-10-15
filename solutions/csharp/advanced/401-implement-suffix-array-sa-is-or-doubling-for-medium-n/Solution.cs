// Task 401 - Implement suffix array (SA-IS or doubling) for medium N. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// `SuffixArrays.Doubling` sorts the suffixes by their first 2^k characters, doubling k until
// every rank is unique: O(n log^2 n) with comparison sorting. No sentinel is appended - a
// position past the end of the string ranks below every character, which resolves "one suffix
// is a prefix of the other" correctly because the shorter suffix must win.
//
// Correctness is pinned by two independent oracles: a naive suffix sort and the hand-written
// character-by-character comparator. Property tests include medium N (a few hundred
// characters) so the doubling loop really runs several rounds.
using System;
using System.Text;

internal class SuffixArrays
{
    public static int[] Doubling(string s)
    {
        int n = s.Length;
        if (n == 0) return Array.Empty<int>();
        var sa = new int[n];
        for (int i = 0; i < n; i++) sa[i] = i;
        var rank = new int[n];
        for (int i = 0; i < n; i++) rank[i] = s[i];
        for (int k = 1; ; k <<= 1)
        {
            int step = k;
            Array.Sort(sa, (a, b) =>
            {
                if (rank[a] != rank[b]) return rank[a].CompareTo(rank[b]);
                int ra = a + step < n ? rank[a + step] : -1;
                int rb = b + step < n ? rank[b + step] : -1;
                return ra.CompareTo(rb);
            });
            var next = new int[n];
            for (int i = 1; i < n; i++)
            {
                int prev = sa[i - 1], cur = sa[i];
                int pa = prev + step < n ? rank[prev + step] : -1;
                int pb = cur + step < n ? rank[cur + step] : -1;
                next[cur] = next[prev] + (rank[prev] != rank[cur] || pa != pb ? 1 : 0);
            }
            Array.Copy(next, rank, n);
            if (rank[sa[n - 1]] == n - 1) break;
        }
        return sa;
    }

    public static int Compare(string s, int i, int j)
    {
        int n = s.Length;
        while (i < n && j < n)
        {
            if (s[i] != s[j]) return s[i] < s[j] ? -1 : 1;
            i++;
            j++;
        }
        if (i == j) return 0;
        return i == n ? -1 : 1;
    }

    public static int[] Naive(string s)
    {
        int n = s.Length;
        var idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => Compare(s, a, b));
        return idx;
    }

    public static bool IsValid(string s, int[] order)
    {
        int n = s.Length;
        if (order.Length != n) return false;
        var seen = new bool[n];
        foreach (int start in order)
        {
            if (start < 0 || start >= n || seen[start]) return false;
            seen[start] = true;
        }
        for (int i = 0; i + 1 < n; i++)
        {
            if (Compare(s, order[i], order[i + 1]) >= 0) return false;
        }
        return true;
    }

    public static string Join(int[] values) => string.Join(",", values);
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

    private static string Repeat(string unit, int times)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < times; i++) sb.Append(unit);
        return sb.ToString();
    }

    public static void Main()
    {
        Check(SuffixArrays.Join(SuffixArrays.Doubling("banana")) == "5,3,1,0,4,2", "banana");
        Check(SuffixArrays.Join(SuffixArrays.Doubling("mississippi")) == "10,7,4,1,0,9,8,6,3,5,2", "mississippi");

        Check(SuffixArrays.Doubling("").Length == 0, "empty");
        Check(SuffixArrays.Join(SuffixArrays.Doubling("a")) == "0", "single");
        Check(SuffixArrays.Join(SuffixArrays.Doubling("aaaa")) == "3,2,1,0", "aaaa");
        Check(SuffixArrays.Join(SuffixArrays.Doubling("abab")) == "2,0,3,1", "abab");
        Check(SuffixArrays.Join(SuffixArrays.Doubling("abracadabra")) == SuffixArrays.Join(SuffixArrays.Naive("abracadabra")),
            "abracadabra");

        var small = new[] { "", "a", "b", "aa", "ab", "ba", "aaaa", "abab", "banana", "mississippi", "abracadabra" };
        foreach (string s in small)
        {
            var order = SuffixArrays.Doubling(s);
            Check(SuffixArrays.Join(order) == SuffixArrays.Join(SuffixArrays.Naive(s)), "small naive " + s);
            Check(SuffixArrays.IsValid(s, order), "small valid " + s);
        }

        foreach (string alphabet in new[] { "ab", "abc", "abcdefghijklmnopqrstuvwxyz" })
        {
            for (int trial = 0; trial < 40; trial++)
            {
                string s = LcgString(28, alphabet);
                var order = SuffixArrays.Doubling(s);
                Check(SuffixArrays.Join(order) == SuffixArrays.Join(SuffixArrays.Naive(s)), "lcg naive " + s);
                Check(SuffixArrays.IsValid(s, order), "lcg valid " + s);
            }
        }

        var medium = new[]
        {
            LcgString(500, "ab"),
            LcgString(500, "abcdefghijklmnopqrstuvwxyz"),
            Repeat("a", 400),
            Repeat("abcab", 100),
        };
        foreach (string s in medium)
        {
            var order = SuffixArrays.Doubling(s);
            Check(SuffixArrays.Join(order) == SuffixArrays.Join(SuffixArrays.Naive(s)), "medium naive");
            Check(SuffixArrays.IsValid(s, order), "medium comparator");
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
