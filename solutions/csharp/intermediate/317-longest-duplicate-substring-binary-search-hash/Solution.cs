// Task 317 - Longest duplicate substring via binary search + rolling hash. (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

internal class DoubleRollingHash
{
    public const long Mod1 = 1_000_000_007L;
    public const long Mod2 = 998_244_353L;
    public const long Base = 131L;

    private readonly long[] h1;
    private readonly long[] h2;
    private readonly long[] p1;
    private readonly long[] p2;

    public DoubleRollingHash(string s)
    {
        int n = s.Length;
        h1 = new long[n + 1];
        h2 = new long[n + 1];
        p1 = new long[n + 1];
        p2 = new long[n + 1];
        p1[0] = 1;
        p2[0] = 1;
        for (int i = 0; i < n; i++)
        {
            long code = s[i];
            h1[i + 1] = (h1[i] * Base + code) % Mod1;
            h2[i + 1] = (h2[i] * Base + code) % Mod2;
            p1[i + 1] = p1[i] * Base % Mod1;
            p2[i + 1] = p2[i] * Base % Mod2;
        }
    }

    /// <summary>Hash of s[lo..hi) as a packed long pair.</summary>
    public long HashPair(int lo, int hi)
    {
        long a = (h1[hi] - h1[lo] * p1[hi - lo] % Mod1 + Mod1) % Mod1;
        long b = (h2[hi] - h2[lo] * p2[hi - lo] % Mod2 + Mod2) % Mod2;
        return a * Mod2 + b;
    }

    public bool Equal(int lo1, int hi1, int lo2, int hi2) => HashPair(lo1, hi1) == HashPair(lo2, hi2);
}

public static class Solution
{
    /// <summary>
    /// Simplest duplicate substring of maximal length, found at the smallest starting index; ""
    /// is the documented sentinel meaning "no duplicate substring exists".
    /// </summary>
    public static string LongestDuplicateSubstring(string s)
    {
        int n = s.Length;
        if (n <= 1) return "";
        var rolling = new DoubleRollingHash(s);
        int lo = 1, hi = n - 1, best = 0;
        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            if (HasDuplicate(rolling, n, mid))
            {
                best = mid;
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }
        if (best == 0) return "";
        for (int i = 0; i + best <= n; i++)
        {
            for (int j = i + 1; j + best <= n; j++)
            {
                if (rolling.Equal(i, i + best, j, j + best)) return s.Substring(i, best);
            }
        }
        throw new InvalidOperationException("binary search and extraction disagree");
    }

    private static bool HasDuplicate(DoubleRollingHash rolling, int n, int length)
    {
        var seen = new HashSet<long>();
        for (int i = 0; i + length <= n; i++)
        {
            if (!seen.Add(rolling.HashPair(i, i + length))) return true;
        }
        return false;
    }

    /// <summary>All-pairs LCP scan; the independent oracle for the best length.</summary>
    public static int LongestDuplicateNaiveLength(string s)
    {
        int best = 0;
        for (int i = 0; i < s.Length; i++)
        {
            for (int j = i + 1; j < s.Length; j++)
            {
                int h = 0;
                while (j + h < s.Length && s[i + h] == s[j + h]) h++;
                if (h > best) best = h;
            }
        }
        return best;
    }

    /// <summary>Occurrences counted with overlaps (non-overlapping counting would undercount repeats).</summary>
    public static int OverlappingCount(string s, string word)
    {
        int count = 0;
        for (int i = 0; i + word.Length <= s.Length; i++)
        {
            if (string.CompareOrdinal(s.Substring(i, word.Length), word) == 0) count++;
        }
        return count;
    }

    private static int seed = 3170;

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
        Check(LongestDuplicateSubstring("banana") == "ana", "banana");           // "ana" at 1 and 3
        Check(LongestDuplicateSubstring("aaaa") == "aaa", "aaaa");               // "aaa" at 0 and 1
        Check(LongestDuplicateSubstring("abcd") == "", "abcd");                  // nothing repeats
        Check(LongestDuplicateSubstring("") == "", "empty");
        Check(LongestDuplicateSubstring("a") == "", "single");
        Check(LongestDuplicateSubstring("aa") == "a", "aa");
        Check(LongestDuplicateSubstring("abab") == "ab", "abab");
        Check(LongestDuplicateSubstring("abcabcabc") == "abcabc", "abcabcabc");  // at 0 and 3
        Check(LongestDuplicateSubstring("mississippi") == "issi", "mississippi"); // at 1 and 4

        var samples = new List<string> { "banana", "aaaa", "abcd", "abab", "abcabcabc",
            "mississippi", "aabcaabxaaaz" };
        for (int i = 0; i < 30; i++) samples.Add(LcgString(24, "ab"));
        for (int i = 0; i < 15; i++) samples.Add(LcgString(18, "abc"));

        foreach (string s in samples)
        {
            string result = LongestDuplicateSubstring(s);
            int naive = LongestDuplicateNaiveLength(s);
            Check(result.Length == naive, "length " + s);
            if (naive == 0)
            {
                Check(result.Length == 0, "empty " + s);
            }
            else
            {
                Check(OverlappingCount(s, result) > 1, "duplicate " + s);        // a real duplicate ...
                Check(result == LongestDuplicateSubstring(s), "deterministic " + s);
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
