// Task 318 - Minimum window subsequence (not necessarily contiguous). (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

public static class Solution
{
    /// <summary>Shortest window of s containing t as a subsequence, "" if none exists.</summary>
    public static string MinWindowSubsequence(string s, string t)
    {
        int n = s.Length, m = t.Length;
        if (m == 0 || m > n) return "";
        var bestStart = new int[m];                          // bestStart[j] = latest start covering t[0..j]
        for (int j = 0; j < m; j++) bestStart[j] = -1;
        int bestLength = n + 1, begin = -1, end = -1;
        for (int i = 0; i < n; i++)
        {
            char ch = s[i];
            for (int j = m - 1; j >= 0; j--)                 // descending keeps row i-1 intact
            {
                if (ch == t[j]) bestStart[j] = j == 0 ? i : bestStart[j - 1];
            }
            if (bestStart[m - 1] != -1)
            {
                int start = bestStart[m - 1];
                if (i - start + 1 < bestLength)              // strictly better keeps the leftmost window
                {
                    bestLength = i - start + 1;
                    begin = start;
                    end = i;
                }
            }
        }
        return begin == -1 ? "" : s.Substring(begin, end - begin + 1);
    }

    public static bool IsSubsequence(string needle, string haystack)
    {
        int k = 0;
        for (int i = 0; i < haystack.Length && k < needle.Length; i++)
        {
            if (haystack[i] == needle[k]) k++;
        }
        return k == needle.Length;
    }

    /// <summary>Tries every window; the independent oracle. Returns null when no window exists.</summary>
    public static string BruteMinWindow(string s, string t)
    {
        if (t.Length == 0 || t.Length > s.Length) return null;
        string best = null;
        for (int left = 0; left < s.Length; left++)
        {
            for (int right = left; right < s.Length; right++)
            {
                string window = s.Substring(left, right - left + 1);
                if (IsSubsequence(t, window) && (best == null || window.Length < best.Length)) best = window;
            }
        }
        return best;
    }

    private static int seed = 3180;

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
        Check(MinWindowSubsequence("abcdebdde", "bde") == "bcde", "classic");       // [1..4] beats [5..8]
        Check(MinWindowSubsequence("abc", "abc") == "abc", "identity");
        Check(MinWindowSubsequence("abc", "d") == "", "absent");
        Check(MinWindowSubsequence("bab", "ab") == "ab", "bab");
        Check(MinWindowSubsequence("a", "a") == "a", "singles");
        Check(MinWindowSubsequence("a", "aa") == "", "too long");
        Check(MinWindowSubsequence("", "a") == "", "empty text");
        Check(MinWindowSubsequence("", "") == "", "both empty");
        Check(MinWindowSubsequence("ab", "") == "", "empty pattern");

        // Hand-verified absent case: "ezu" needs e < z < u, but every u (indices 41, 42) precedes
        // every z (indices 43, 44), and no u follows a z.
        string longText = "jmeqksfrsdcmsiwvaovxtkwmamwhvlbajfmbfmecwuuzzaqq";
        Check(MinWindowSubsequence(longText, "ezu") == "", "long absent");
        Check(BruteMinWindow(longText, "ezu") == null, "long absent oracle");

        var texts = new List<string> { "abcdebdde", "bab", "abc", "aab", "zzz" };
        for (int i = 0; i < 20; i++) texts.Add(LcgString(30, "ab"));
        for (int i = 0; i < 10; i++) texts.Add(LcgString(24, "abc"));

        var patterns = new List<string> { "", "ab", "ba", "z", "cc", "bca" };
        for (int i = 0; i < 20; i++) patterns.Add(LcgString(1, "abc"));
        for (int i = 0; i < 20; i++) patterns.Add(LcgString(2, "abc"));
        for (int i = 0; i < 20; i++) patterns.Add(LcgString(3, "abc"));

        foreach (string text in texts)
        {
            foreach (string pattern in patterns)
            {
                string window = MinWindowSubsequence(text, pattern);
                string oracle = BruteMinWindow(text, pattern);
                if (oracle == null)
                {
                    Check(window.Length == 0, "none " + text + " / " + pattern);
                }
                else
                {
                    Check(window.Length > 0 && text.Contains(window), "in text " + text + " / " + pattern);
                    Check(IsSubsequence(pattern, window), "subsequence " + text + " / " + pattern);
                    Check(window.Length == oracle.Length, "length " + text + " / " + pattern);
                }
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
