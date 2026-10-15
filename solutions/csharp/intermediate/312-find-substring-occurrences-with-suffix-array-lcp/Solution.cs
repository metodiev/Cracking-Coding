// Task 312 - Substring occurrences with a suffix array + LCP (Kasai). (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

public static class Solution
{
    public static int[] SuffixArray(string s)
    {
        var indices = Enumerable.Range(0, s.Length).ToArray();
        Array.Sort(indices, (a, b) => string.CompareOrdinal(s.Substring(a), s.Substring(b)));
        return indices;
    }

    /// <summary>Manual lexicographic comparator; the independent oracle.</summary>
    public static int CompareSuffixes(string s, int i, int j)
    {
        int n = s.Length;
        while (i < n && j < n)
        {
            if (s[i] != s[j]) return s[i] < s[j] ? -1 : 1;
            i++;
            j++;
        }
        if (i == j) return 0;
        return i == n ? -1 : 1;                 // the shorter suffix is smaller
    }

    public static bool IsValidSuffixArray(string s, int[] order)
    {
        var sorted = (int[])order.Clone();
        Array.Sort(sorted);
        for (int i = 0; i < sorted.Length; i++) if (sorted[i] != i) return false;
        for (int k = 0; k + 1 < order.Length; k++) if (CompareSuffixes(s, order[k], order[k + 1]) >= 0) return false;
        return true;
    }

    /// <summary>lcp[i] = LCP(sa[i-1], sa[i]) with lcp[0] = 0; O(n) via the Kasai trick.</summary>
    public static int[] KasaiLcp(string s, int[] sa)
    {
        int n = s.Length;
        var rank = new int[n];
        for (int i = 0; i < n; i++) rank[sa[i]] = i;
        var lcp = new int[n];
        int h = 0;
        for (int i = 0; i < n; i++)
        {
            if (rank[i] > 0)
            {
                int j = sa[rank[i] - 1];
                while (i + h < n && j + h < n && s[i + h] == s[j + h]) h++;
                lcp[rank[i]] = h;
                if (h > 0) h--;
            }
            else
            {
                h = 0;
            }
        }
        return lcp;
    }

    /// <summary>Direct character-by-character LCP of two suffixes; the independent oracle.</summary>
    public static int LcpOfSuffixes(string s, int i, int j)
    {
        int h = 0;
        while (i + h < s.Length && j + h < s.Length && s[i + h] == s[j + h]) h++;
        return h;
    }

    /// <summary>
    /// All start indices of the pattern in s, sorted ascending. The empty pattern matches at
    /// every valid position (n + 1 of them).
    /// </summary>
    public static int[] Occurrences(string s, string pattern)
    {
        int n = s.Length;
        int m = pattern.Length;
        if (m == 0) return Enumerable.Range(0, n + 1).ToArray();
        int[] sa = SuffixArray(s);
        int[] lcp = KasaiLcp(s, sa);
        // first index k with suffix sa[k] >= pattern; matching suffixes are exactly the run at k
        int lo = 0, hi = n;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (string.CompareOrdinal(s.Substring(sa[mid]), pattern) < 0) lo = mid + 1;
            else hi = mid;
        }
        int k = lo;
        if (k == n || string.CompareOrdinal(s.Substring(sa[k], Math.Min(m, n - sa[k])), pattern) != 0) return new int[0];
        var result = new List<int> { sa[k] };
        int t = k + 1;
        while (t < n && lcp[t] >= m)
        {
            result.Add(sa[t]);
            t++;
        }
        result.Sort();
        return result.ToArray();
    }

    /// <summary>Scan every position; the independent oracle (overlapping matches included).</summary>
    public static int[] NaiveOccurrences(string s, string pattern)
    {
        if (pattern.Length == 0) return Enumerable.Range(0, s.Length + 1).ToArray();
        var result = new List<int>();
        for (int i = 0; i + pattern.Length <= s.Length; i++)
        {
            if (string.CompareOrdinal(s.Substring(i, pattern.Length), pattern) == 0) result.Add(i);
        }
        return result.ToArray();
    }

    private static int seed = 3120;

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
        string s = "banana";
        int[] sa = SuffixArray(s);
        Check(string.Join(",", sa) == "5,3,1,0,4,2", "banana suffix array");
        Check(IsValidSuffixArray(s, sa), "banana valid");
        int[] lcp = KasaiLcp(s, sa);
        Check(string.Join(",", lcp) == "0,1,3,0,0,2", "banana lcp");
        for (int i = 1; i < sa.Length; i++) Check(lcp[i] == LcpOfSuffixes(s, sa[i - 1], sa[i]), "lcp check " + i);

        Check(string.Join(",", Occurrences(s, "ana")) == "1,3", "ana");
        Check(string.Join(",", Occurrences(s, "ba")) == "0", "ba");
        Check(string.Join(",", Occurrences(s, "an")) == "1,3", "an");
        Check(string.Join(",", Occurrences(s, "a")) == "1,3,5", "a");
        Check(string.Join(",", Occurrences(s, "n")) == "2,4", "n");
        Check(string.Join(",", Occurrences(s, "banana")) == "0", "banana");
        Check(Occurrences(s, "ab").Length == 0, "ab absent");
        Check(Occurrences(s, "xy").Length == 0, "xy absent");
        Check(Occurrences(s, "bananas").Length == 0, "bananas absent");
        Check(string.Join(",", Occurrences(s, "")) == "0,1,2,3,4,5,6", "empty pattern");
        Check(Occurrences("", "a").Length == 0, "empty text");

        var texts = new List<string> { "a", "aa", "aaaa", "abab", "mississippi", "banana", "" };
        for (int i = 0; i < 30; i++) texts.Add(LcgString(20, "ab"));
        for (int i = 0; i < 10; i++) texts.Add(LcgString(16, "abc"));

        foreach (string text in texts)
        {
            int[] textSa = SuffixArray(text);
            int[] textLcp = KasaiLcp(text, textSa);
            Check(IsValidSuffixArray(text, textSa), "valid " + text);
            for (int i = 1; i < textSa.Length; i++)
                Check(textLcp[i] == LcpOfSuffixes(text, textSa[i - 1], textSa[i]), "lcp " + text + " " + i);
            var patterns = new List<string>();
            for (int i = 0; i < text.Length; i++)
            {
                for (int extra = 1; extra <= 3; extra++)
                {
                    if (i + extra <= text.Length) patterns.Add(text.Substring(i, extra));
                }
            }
            patterns.AddRange(new[] { "zz", "z", "ababab", "", "bbbb", "ccc" });
            foreach (string pattern in patterns)
            {
                Check(string.Join(",", Occurrences(text, pattern)) == string.Join(",", NaiveOccurrences(text, pattern)),
                    "search " + text + " / " + pattern);
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
