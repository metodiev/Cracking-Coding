// Task 402 - Build the LCP array from the suffix array (Kasai). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Kasai's algorithm walks the suffixes in text order, reusing the previous longest common
// prefix: if suffix i shares h characters with its suffix-array predecessor, then suffix i+1
// shares at least h-1 with its own predecessor, so each position is advanced amortized once
// and the whole scan is O(n). lcp[i] is the LCP of suffixes sa[i-1] and sa[i], lcp[0] = 0.
//
// Hand-checked anchors: banana -> [0, 1, 3, 0, 0, 2] and mississippi -> [0, 1, 1, 4, 0, 0,
// 1, 0, 2, 1, 3]. A pairwise char-by-char oracle and the identity "distinct substrings =
// n(n+1)/2 - sum(lcp)" cross-check it on LCG strings.
using System;
using System.Collections.Generic;
using System.Text;

internal class LcpArrays
{
    public static int[] Kasai(string s, int[] sa)
    {
        int n = s.Length;
        var lcp = new int[n];
        var rank = new int[n];
        for (int i = 0; i < n; i++) rank[sa[i]] = i;
        int h = 0;
        for (int i = 0; i < n; i++)
        {
            int r = rank[i];
            if (r == 0)
            {
                h = 0;
                continue;
            }
            int j = sa[r - 1];
            while (i + h < n && j + h < n && s[i + h] == s[j + h]) h++;
            lcp[r] = h;
            if (h > 0) h--;
        }
        return lcp;
    }

    public static int PairLcp(string s, int i, int j)
    {
        int n = s.Length;
        int h = 0;
        while (i + h < n && j + h < n && s[i + h] == s[j + h]) h++;
        return h;
    }

    public static int[] Naive(string s, int[] sa)
    {
        int n = sa.Length;
        if (n == 0) return Array.Empty<int>();
        var lcp = new int[n];
        for (int i = 1; i < n; i++) lcp[i] = PairLcp(s, sa[i - 1], sa[i]);
        return lcp;
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

    public static int[] SuffixArrayNaive(string s)
    {
        int n = s.Length;
        var idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => Compare(s, a, b));
        return idx;
    }

    public static int DistinctSubstringsBrute(string s)
    {
        var seen = new HashSet<string>();
        for (int i = 0; i < s.Length; i++)
            for (int j = i + 1; j <= s.Length; j++)
                seen.Add(s.Substring(i, j - i));
        return seen.Count;
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

    private static int Sum(int[] values)
    {
        int total = 0;
        foreach (int value in values) total += value;
        return total;
    }

    public static void Main()
    {
        var banana = LcpArrays.Kasai("banana", new[] { 5, 3, 1, 0, 4, 2 });
        Check(LcpArrays.Join(banana) == "0,1,3,0,0,2", "banana");
        Check(LcpArrays.Join(banana) == LcpArrays.Join(LcpArrays.Naive("banana", new[] { 5, 3, 1, 0, 4, 2 })),
            "banana naive");

        var mississippi = LcpArrays.Kasai("mississippi", new[] { 10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2 });
        Check(LcpArrays.Join(mississippi) == "0,1,1,4,0,0,1,0,2,1,3", "mississippi");

        Check(LcpArrays.Kasai("", Array.Empty<int>()).Length == 0, "empty");
        Check(LcpArrays.Join(LcpArrays.Kasai("a", new[] { 0 })) == "0", "single");
        Check(LcpArrays.Join(LcpArrays.Kasai("aaaa", new[] { 3, 2, 1, 0 })) == "0,1,2,3", "aaaa");
        Check(LcpArrays.Join(LcpArrays.Kasai("abab", new[] { 2, 0, 3, 1 })) == "0,2,0,1", "abab");

        var small = new[] { "", "a", "aaa", "abab", "banana", "mississippi", "abracadabra", "aaaaabaaaab" };
        foreach (string s in small)
        {
            var sa = LcpArrays.SuffixArrayNaive(s);
            var lcp = LcpArrays.Kasai(s, sa);
            Check(LcpArrays.Join(lcp) == LcpArrays.Join(LcpArrays.Naive(s, sa)), "small " + s);
            Check(lcp.Length == s.Length, "length " + s);
            Check(lcp.Length == 0 || lcp[0] == 0, "first " + s);
        }

        Check(LcpArrays.DistinctSubstringsBrute("banana") == 15, "banana distinct");
        Check(6 * 7 / 2 - Sum(banana) == 15, "banana identity");

        foreach (string alphabet in new[] { "ab", "abc" })
        {
            for (int trial = 0; trial < 40; trial++)
            {
                string s = LcgString(24, alphabet);
                var sa = LcpArrays.SuffixArrayNaive(s);
                var lcp = LcpArrays.Kasai(s, sa);
                Check(LcpArrays.Join(lcp) == LcpArrays.Join(LcpArrays.Naive(s, sa)), "lcg " + s);
                Check(s.Length * (s.Length + 1) / 2 - Sum(lcp) == LcpArrays.DistinctSubstringsBrute(s),
                    "identity " + s);
            }
        }

        string medium = LcgString(400, "ab");
        var mediumSa = LcpArrays.SuffixArrayNaive(medium);
        var mediumLcp = LcpArrays.Kasai(medium, mediumSa);
        Check(LcpArrays.Join(mediumLcp) == LcpArrays.Join(LcpArrays.Naive(medium, mediumSa)), "medium");
        Check(mediumLcp[0] == 0, "medium first");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
