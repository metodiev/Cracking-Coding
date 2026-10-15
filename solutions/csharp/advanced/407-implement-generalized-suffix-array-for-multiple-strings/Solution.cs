// Task 407 - Implement a generalized suffix array for multiple strings. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// The m strings are concatenated with m *distinct* sentinels: string k gets the character
// (char)(1 + k). All sentinels are below every real (lowercase) character, and they are
// distinct and ordered by string index, so a suffix from string i always beats the same text
// extended in string j when i < j. This is the cheapest correct sentinel scheme for the
// generalized suffix array.
//
// Every suffix also carries an owner (which string) and an offset (position inside it), so
// occurrences of a pattern are found with two binary searches over the suffix array: the
// pattern-prefixed suffixes form one contiguous range, and each entry of that range maps back
// to a (string, position) occurrence.
//
// Anchors: ["banana", "ana"] -> occurrences of "ana" = (0,1), (0,3), (1,0). Checks: the
// combined SA validates against a manual comparator, the smallest suffixes are the sentinels
// in index order, suffixes of each string keep their own plain relative order, and LCG-built
// pattern searches agree with a naive scan.
using System;
using System.Collections.Generic;
using System.Text;

internal class GeneralizedSa
{
    public readonly string Text;
    public readonly int[] Sa;
    public readonly int[] Owner;
    public readonly int[] Offset;
    public readonly int[] Sentinels;

    private GeneralizedSa(string text, int[] sa, int[] owner, int[] offset, int[] sentinels)
    {
        Text = text;
        Sa = sa;
        Owner = owner;
        Offset = offset;
        Sentinels = sentinels;
    }

    public static GeneralizedSa Build(IReadOnlyList<string> strings)
    {
        var sb = new StringBuilder();
        var owners = new List<int>();
        var offsets = new List<int>();
        var sentinels = new List<int>();
        for (int k = 0; k < strings.Count; k++)
        {
            string s = strings[k];
            for (int pos = 0; pos < s.Length; pos++)
            {
                sb.Append(s[pos]);
                owners.Add(k);
                offsets.Add(pos);
            }
            sentinels.Add(sb.Length);
            sb.Append((char)(1 + k));
            owners.Add(k);
            offsets.Add(-1);
        }
        string text = sb.ToString();
        return new GeneralizedSa(text, SuffixArrayDoubling(text), owners.ToArray(), offsets.ToArray(),
            sentinels.ToArray());
    }

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

    public static bool IsValid(string s, int[] order)
    {
        if (order.Length != s.Length) return false;
        var seen = new bool[s.Length];
        foreach (int start in order)
        {
            if (start < 0 || start >= s.Length || seen[start]) return false;
            seen[start] = true;
        }
        for (int i = 0; i + 1 < order.Length; i++)
        {
            if (Compare(s, order[i], order[i + 1]) >= 0) return false;
        }
        return true;
    }

    public static int CmpPrefix(string text, int start, string pattern)
    {
        for (int i = 0; i < pattern.Length; i++)
        {
            if (start + i >= text.Length) return -1;
            char ch = text[start + i];
            if (ch != pattern[i]) return ch < pattern[i] ? -1 : 1;
        }
        return 0;
    }

    public static (int Lower, int Upper) PatternRange(string text, int[] sa, string pattern)
    {
        int lo = 0, hi = sa.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (CmpPrefix(text, sa[mid], pattern) < 0) lo = mid + 1;
            else hi = mid;
        }
        int lower = lo;
        hi = sa.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (CmpPrefix(text, sa[mid], pattern) <= 0) lo = mid + 1;
            else hi = mid;
        }
        return (lower, lo);
    }

    public static string FindOccurrences(IReadOnlyList<string> strings, string pattern)
    {
        var built = Build(strings);
        var (lower, upper) = PatternRange(built.Text, built.Sa, pattern);
        var found = new List<(int Owner, int Offset)>();
        for (int i = lower; i < upper; i++)
        {
            int idx = built.Sa[i];
            if (built.Offset[idx] >= 0) found.Add((built.Owner[idx], built.Offset[idx]));
        }
        found.Sort();
        return JoinPairs(found);
    }

    public static string NaiveOccurrences(IReadOnlyList<string> strings, string pattern)
    {
        var found = new List<(int Owner, int Offset)>();
        for (int k = 0; k < strings.Count; k++)
        {
            string s = strings[k];
            for (int i = 0; i + pattern.Length <= s.Length; i++)
            {
                if (string.CompareOrdinal(s, i, pattern, 0, pattern.Length) == 0) found.Add((k, i));
            }
        }
        found.Sort();
        return JoinPairs(found);
    }

    private static string JoinPairs(List<(int Owner, int Offset)> pairs)
    {
        var sb = new StringBuilder();
        foreach (var pair in pairs)
        {
            if (sb.Length > 0) sb.Append(",");
            sb.Append(pair.Owner).Append(":").Append(pair.Offset);
        }
        return sb.ToString();
    }

    public static int[] SuffixArrayNaive(string s)
    {
        int n = s.Length;
        var idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => Compare(s, a, b));
        return idx;
    }

    public static List<string> AllShortSubstrings(IReadOnlyList<string> strings, int maxLen)
    {
        var seen = new SortedSet<string>(StringComparer.Ordinal);
        foreach (string s in strings)
        {
            for (int i = 0; i < s.Length; i++)
            {
                for (int j = i + 1; j <= Math.Min(s.Length, i + maxLen); j++) seen.Add(s.Substring(i, j - i));
            }
        }
        return new List<string>(seen);
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

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static string JoinInts(List<int> values)
    {
        var sb = new StringBuilder();
        foreach (int value in values)
        {
            if (sb.Length > 0) sb.Append(",");
            sb.Append(value);
        }
        return sb.ToString();
    }

    public static void Main()
    {
        var strings = new List<string> { "banana", "ana" };
        var built = GeneralizedSa.Build(strings);
        Check(built.Text.Length == 6 + 1 + 3 + 1, "text length");
        Check(GeneralizedSa.IsValid(built.Text, built.Sa), "combined valid");
        Check(built.Sa[0] == built.Sentinels[0] && built.Sa[1] == built.Sentinels[1], "sentinels first");
        for (int k = 0; k < strings.Count; k++)
        {
            var kept = new List<int>();
            foreach (int idx in built.Sa)
            {
                if (built.Owner[idx] == k && built.Offset[idx] >= 0) kept.Add(built.Offset[idx]);
            }
            Check(JoinInts(kept) == JoinInts(new List<int>(GeneralizedSa.SuffixArrayNaive(strings[k]))),
                "relative order " + k);
        }

        Check(GeneralizedSa.FindOccurrences(strings, "ana") == "0:1,0:3,1:0", "ana");
        Check(GeneralizedSa.FindOccurrences(strings, "a") == "0:1,0:3,0:5,1:0,1:2", "a");
        Check(GeneralizedSa.FindOccurrences(strings, "na") == "0:2,0:4,1:1", "na");
        Check(GeneralizedSa.FindOccurrences(strings, "ban") == "0:0", "ban");
        Check(GeneralizedSa.FindOccurrences(strings, "banana") == "0:0", "banana");
        Check(GeneralizedSa.FindOccurrences(strings, "b") == "0:0", "b");
        Check(GeneralizedSa.FindOccurrences(strings, "xyz") == "", "xyz");
        Check(GeneralizedSa.FindOccurrences(strings, "anan") == "0:1", "anan");
        Check(GeneralizedSa.FindOccurrences(strings, "nanan") == "", "nanan");
        Check(GeneralizedSa.FindOccurrences(strings, "bananas") == "", "bananas");

        Check(GeneralizedSa.FindOccurrences(new[] { "abab" }, "ab") == "0:0,0:2", "single");
        Check(GeneralizedSa.FindOccurrences(new[] { "", "ab" }, "ab") == "1:0", "empty first");
        Check(GeneralizedSa.FindOccurrences(new[] { "", "" }, "a") == "", "both empty");
        Check(GeneralizedSa.FindOccurrences(Array.Empty<string>(), "a") == "", "no strings");

        var quad = new List<string> { "mississippi", "issi", "i", "ppi" };
        var patterns = GeneralizedSa.AllShortSubstrings(quad, 3);
        patterns.AddRange(new[] { "issi", "ississi", "ssippi", "zz", "q", "ippiz" });
        foreach (string pattern in patterns)
        {
            Check(GeneralizedSa.FindOccurrences(quad, pattern) == GeneralizedSa.NaiveOccurrences(quad, pattern),
                "quad " + pattern);
        }
        Check(GeneralizedSa.FindOccurrences(quad, "i") == GeneralizedSa.NaiveOccurrences(quad, "i"), "quad i");

        for (int trial = 0; trial < 30; trial++)
        {
            var random = new List<string>();
            for (int k = 0; k < 3; k++) random.Add(LcgString(12 + seed % 7, "abc"));
            var randomBuilt = GeneralizedSa.Build(random);
            Check(GeneralizedSa.IsValid(randomBuilt.Text, randomBuilt.Sa), "lcg valid");
            for (int k = 0; k < random.Count; k++)
            {
                Check(randomBuilt.Sa[k] == randomBuilt.Sentinels[k], "lcg sentinel " + k);
                var kept = new List<int>();
                foreach (int idx in randomBuilt.Sa)
                {
                    if (randomBuilt.Owner[idx] == k && randomBuilt.Offset[idx] >= 0) kept.Add(randomBuilt.Offset[idx]);
                }
                Check(JoinInts(kept) == JoinInts(new List<int>(GeneralizedSa.SuffixArrayNaive(random[k]))),
                    "lcg relative " + k);
            }
            var randomPatterns = GeneralizedSa.AllShortSubstrings(random, 3);
            randomPatterns.AddRange(new[] { "zzz", "q", "abcabca" });
            foreach (string pattern in randomPatterns)
            {
                Check(GeneralizedSa.FindOccurrences(random, pattern) == GeneralizedSa.NaiveOccurrences(random, pattern),
                    "lcg pattern " + pattern);
            }
        }
        Console.WriteLine("PASS");
    }
}
