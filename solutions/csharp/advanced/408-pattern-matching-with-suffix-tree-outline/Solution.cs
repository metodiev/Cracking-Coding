// Task 408 - Pattern matching with a suffix tree (outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Backing store: the suffix array plus the LCP array, which represent the same object as a
// suffix tree - a suffix-tree node corresponds to an LCP interval, and the locus of a pattern
// is exactly the maximal interval [lo, hi) of suffixes carrying the pattern as prefix. Descent
// from the root consumes the pattern one character at a time: two binary searches over the
// suffix array locate the interval, and every suffix inside it is a match, because in a suffix
// array all pattern-prefixed suffixes are contiguous. The LCP array documents and validates
// the locus: all suffixes in [lo, hi) share at least len(pattern) prefix characters, so
// min(lcp[lo+1 .. hi-1]) >= len(pattern).
//
// Checks: "banana" anchors, then LCG texts where Locate (and its interval width) agree with a
// naive scan for substrings, mutated and absent patterns. The empty pattern is documented to
// match every position.
using System;
using System.Collections.Generic;

internal class SuffixTreeIndex
{
    public readonly string Text;
    public readonly int[] Sa;
    public readonly int[] Lcp;

    public SuffixTreeIndex(string text)
    {
        Text = text;
        Sa = SuffixArrayDoubling(text);
        Lcp = Kasai(text, Sa);
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

    public static int[] Kasai(string s, int[] sa)
    {
        int n = s.Length;
        var result = new int[n];
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
            result[r] = h;
            if (h > 0) h--;
        }
        return result;
    }

    public int Cmp(int start, string pattern)
    {
        for (int i = 0; i < pattern.Length; i++)
        {
            if (start + i >= Text.Length) return -1;
            char ch = Text[start + i];
            if (ch != pattern[i]) return ch < pattern[i] ? -1 : 1;
        }
        return 0;
    }

    public (int Lower, int Upper) Interval(string pattern)
    {
        if (pattern.Length == 0) return (0, Sa.Length);
        int lo = 0, hi = Sa.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (Cmp(Sa[mid], pattern) < 0) lo = mid + 1;
            else hi = mid;
        }
        int lower = lo;
        hi = Sa.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (Cmp(Sa[mid], pattern) <= 0) lo = mid + 1;
            else hi = mid;
        }
        return (lower, lo);
    }

    public int[] Locate(string pattern)
    {
        var (lower, upper) = Interval(pattern);
        var positions = new int[upper - lower];
        Array.Copy(Sa, lower, positions, 0, upper - lower);
        Array.Sort(positions);
        return positions;
    }

    public int Count(string pattern)
    {
        var (lower, upper) = Interval(pattern);
        return upper - lower;
    }

    public int LocusSharedPrefix(string pattern)
    {
        var (lower, upper) = Interval(pattern);
        if (upper - lower <= 1) return 0;
        int best = int.MaxValue;
        for (int i = lower + 1; i < upper; i++) best = Math.Min(best, Lcp[i]);
        return best;
    }

    public static int[] NaiveLocate(string text, string pattern)
    {
        var positions = new List<int>();
        if (pattern.Length == 0)
        {
            for (int i = 0; i < text.Length; i++) positions.Add(i);
            return positions.ToArray();
        }
        for (int i = 0; i + pattern.Length <= text.Length; i++)
        {
            if (string.CompareOrdinal(text, i, pattern, 0, pattern.Length) == 0) positions.Add(i);
        }
        return positions.ToArray();
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
}

public static class Solution
{
    private static int seed = 2718;

    private static string LcgString(int length, string alphabet)
    {
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < length; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            sb.Append(alphabet[seed % alphabet.Length]);
        }
        return sb.ToString();
    }

    private static string Join(int[] values) => string.Join(",", values);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckIndex(string text, IEnumerable<string> extraPatterns)
    {
        var index = new SuffixTreeIndex(text);
        Check(Join(index.Sa) == Join(SuffixTreeIndex.SuffixArrayNaive(text)), "sa " + text);
        var patterns = new List<string>(extraPatterns);
        for (int i = 0; i < text.Length; i++)
        {
            for (int j = i + 1; j <= Math.Min(text.Length, i + 5); j++) patterns.Add(text.Substring(i, j - i));
        }
        patterns.AddRange(new[] { "zz", "qq", "!" });
        foreach (string pattern in patterns)
        {
            Check(Join(index.Locate(pattern)) == Join(SuffixTreeIndex.NaiveLocate(text, pattern)),
                "locate " + text + "/" + pattern);
            Check(index.Count(pattern) == index.Locate(pattern).Length, "count " + pattern);
            var (lower, upper) = index.Interval(pattern);
            for (int k = lower; k < upper; k++)
            {
                int start = index.Sa[k];
                if (pattern.Length > 0)
                {
                    Check(string.CompareOrdinal(text, start, pattern, 0, pattern.Length) == 0, "inside " + pattern);
                }
            }
            if (pattern.Length > 0 && lower > 0)
            {
                int prev = index.Sa[lower - 1];
                Check(!(prev + pattern.Length <= text.Length
                    && string.CompareOrdinal(text, prev, pattern, 0, pattern.Length) == 0), "left " + pattern);
            }
            if (pattern.Length > 0 && upper < index.Sa.Length)
            {
                int next = index.Sa[upper];
                Check(!(next + pattern.Length <= text.Length
                    && string.CompareOrdinal(text, next, pattern, 0, pattern.Length) == 0), "right " + pattern);
            }
            if (pattern.Length > 0 && upper - lower >= 2)
            {
                Check(index.LocusSharedPrefix(pattern) >= pattern.Length, "locus " + pattern);
            }
        }
    }

    public static void Main()
    {
        CheckIndex("banana", new[] { "ana", "na", "a", "banana", "anana", "bananas", "nan" });
        var banana = new SuffixTreeIndex("banana");
        Check(Join(banana.Locate("ana")) == "1,3", "ana");
        Check(Join(banana.Locate("na")) == "2,4", "na");
        Check(Join(banana.Locate("a")) == "1,3,5", "a");
        Check(Join(banana.Locate("banana")) == "0", "banana");
        Check(Join(banana.Locate("anana")) == "1", "anana");
        Check(Join(banana.Locate("")) == "0,1,2,3,4,5", "empty");
        Check(Join(banana.Locate("nana")) == "2", "nana");
        Check(banana.Locate("bananas").Length == 0, "bananas");
        Check(Join(new[] { banana.Interval("ana").Lower, banana.Interval("ana").Upper }) == "1,3", "ana interval");
        Check(banana.LocusSharedPrefix("ana") >= 3, "ana locus");

        CheckIndex("mississippi", new[] { "issi", "iss", "ssi", "mississippi", "ppi", "sipp" });
        CheckIndex("", new[] { "a" });
        CheckIndex("a", new[] { "a", "aa" });
        CheckIndex("aaaa", new[] { "aaa", "aa", "a", "aaaaa" });

        for (int trial = 0; trial < 30; trial++)
        {
            string text = LcgString(30, "abc");
            var mutations = new List<string>();
            for (int k = 0; k < 6; k++) mutations.Add(text.Substring(0, k) + "d" + text.Substring(k + 1));
            CheckIndex(text, mutations);
        }
        for (int trial = 0; trial < 10; trial++)
        {
            string text = LcgString(60, "ab");
            CheckIndex(text, new[] { text, text.Substring(0, text.Length - 1), "z" + text });
        }
        Console.WriteLine("PASS");
    }
}
