// Task 403 - Use the suffix array + LCP array for the longest repeated substring. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A substring occurring at least twice is a common prefix of two different suffixes, and those
// two suffixes are adjacent somewhere in the suffix array, so the answer length is max(lcp)
// and one witness is the prefix of that length of sa[i] at the first maximizing i.
//
// Sentinel convention: when nothing repeats (length < 2, or all substrings unique like "abc")
// the function returns "" - documented so "no repeat" is distinguishable from a one-character
// repeat. Anchors: banana -> "ana", mississippi -> "issi", aaaa -> "aaa", abab -> "ab".
// Cross-checked against a brute-force substring counter on LCG strings.
using System;
using System.Collections.Generic;
using System.Text;

internal class RepeatedSubstring
{
    public static int[] SuffixArrayDoubling(string s)
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
            var current = rank;
            Array.Sort(sa, (a, b) =>
            {
                if (current[a] != current[b]) return current[a].CompareTo(current[b]);
                int ra = a + step < n ? current[a + step] : -1;
                int rb = b + step < n ? current[b + step] : -1;
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
            rank = next;
            if (rank[sa[n - 1]] == n - 1) break;
        }
        return sa;
    }

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

    public static string LongestRepeated(string s)
    {
        if (s.Length < 2) return "";
        var sa = SuffixArrayDoubling(s);
        var lcp = Kasai(s, sa);
        int bestLength = 0;
        int bestStart = -1;
        for (int i = 1; i < s.Length; i++)
        {
            if (lcp[i] > bestLength)
            {
                bestLength = lcp[i];
                bestStart = sa[i];
            }
        }
        return bestLength == 0 ? "" : s.Substring(bestStart, bestLength);
    }

    public static int LongestRepeatedLengthBrute(string s)
    {
        var counts = new Dictionary<string, int>();
        for (int i = 0; i < s.Length; i++)
        {
            for (int j = i + 1; j <= s.Length; j++)
            {
                string t = s.Substring(i, j - i);
                counts.TryGetValue(t, out int seen);
                counts[t] = seen + 1;
            }
        }
        int best = 0;
        foreach (var entry in counts)
        {
            if (entry.Value >= 2 && entry.Key.Length > best) best = entry.Key.Length;
        }
        return best;
    }

    public static int CountOccurrences(string s, string pattern)
    {
        if (pattern.Length == 0) return 0;
        int count = 0;
        for (int i = 0; i + pattern.Length <= s.Length; i++)
        {
            if (string.CompareOrdinal(s, i, pattern, 0, pattern.Length) == 0) count++;
        }
        return count;
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

    public static void Main()
    {
        Check(RepeatedSubstring.LongestRepeated("banana") == "ana", "banana");
        Check(RepeatedSubstring.CountOccurrences("banana", "ana") == 2, "banana count");
        Check(RepeatedSubstring.LongestRepeated("mississippi") == "issi", "mississippi");
        Check(RepeatedSubstring.CountOccurrences("mississippi", "issi") == 2, "mississippi count");
        Check(RepeatedSubstring.LongestRepeated("aaaa") == "aaa", "aaaa");
        Check(RepeatedSubstring.LongestRepeated("abab") == "ab", "abab");
        Check(RepeatedSubstring.LongestRepeated("abcabc") == "abc", "abcabc");
        Check(RepeatedSubstring.LongestRepeated("aabcaa") == "aa", "aabcaa");
        Check(RepeatedSubstring.LongestRepeated("abracadabra") == "abra", "abracadabra");
        Check(RepeatedSubstring.CountOccurrences("abracadabra", "abra") == 2, "abracadabra count");

        Check(RepeatedSubstring.LongestRepeated("") == "", "empty");
        Check(RepeatedSubstring.LongestRepeated("a") == "", "single");
        Check(RepeatedSubstring.LongestRepeated("abc") == "", "abc sentinel");
        Check(RepeatedSubstring.LongestRepeated("abcd") == "", "abcd sentinel");
        Check(RepeatedSubstring.LongestRepeatedLengthBrute("abc") == 0, "abc brute");

        foreach (string alphabet in new[] { "ab", "abc" })
        {
            for (int trial = 0; trial < 40; trial++)
            {
                string s = LcgString(30, alphabet);
                string result = RepeatedSubstring.LongestRepeated(s);
                int best = RepeatedSubstring.LongestRepeatedLengthBrute(s);
                Check(result.Length == best, "lcg length " + s);
                if (best == 0) Check(result.Length == 0, "lcg sentinel " + s);
                else Check(RepeatedSubstring.CountOccurrences(s, result) >= 2, "lcg witness " + s);
            }
        }

        string medium = LcgString(400, "ab");
        string mediumResult = RepeatedSubstring.LongestRepeated(medium);
        Check(mediumResult.Length > 0 && RepeatedSubstring.CountOccurrences(medium, mediumResult) >= 2,
            "medium witness");
        Check(mediumResult.Length == RepeatedSubstring.LongestRepeatedLengthBrute(medium), "medium length");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
