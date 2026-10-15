// Task 313 - The Z-function for pattern matching. (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

public static class Solution
{
    public const char Sentinel = '#';

    /// <summary>z[i] = longest common prefix of s and s.Substring(i); z[0] = 0 by convention.</summary>
    public static int[] ZFunction(string s)
    {
        int n = s.Length;
        var z = new int[n];
        int left = 0, right = 0;
        for (int i = 1; i < n; i++)
        {
            if (i < right) z[i] = Math.Min(right - i, z[i - left]);
            while (i + z[i] < n && s[z[i]] == s[i + z[i]]) z[i]++;
            if (i + z[i] > right)
            {
                left = i;
                right = i + z[i];
            }
        }
        return z;
    }

    /// <summary>Direct O(n^2) definition; the independent oracle.</summary>
    public static int[] ZFunctionNaive(string s)
    {
        int n = s.Length;
        var z = new int[n];
        for (int i = 1; i < n; i++)
        {
            int h = 0;
            while (i + h < n && s[h] == s[i + h]) h++;
            z[i] = h;
        }
        return z;
    }

    /// <summary>All start indices of pattern in text via the Z-function on pattern + "#" + text.</summary>
    public static int[] ZSearch(string text, string pattern)
    {
        int m = pattern.Length;
        if (m == 0) return Enumerable.Range(0, text.Length + 1).ToArray();
        string combined = pattern + Sentinel + text;
        int[] z = ZFunction(combined);
        var result = new List<int>();
        for (int i = m + 1; i < combined.Length; i++)
        {
            if (z[i] >= m) result.Add(i - m - 1);
        }
        return result.ToArray();
    }

    /// <summary>Independent oracle: try every position.</summary>
    public static int[] NaiveSearch(string text, string pattern)
    {
        if (pattern.Length == 0) return Enumerable.Range(0, text.Length + 1).ToArray();
        var result = new List<int>();
        for (int i = 0; i + pattern.Length <= text.Length; i++)
        {
            if (string.CompareOrdinal(text.Substring(i, pattern.Length), pattern) == 0) result.Add(i);
        }
        return result.ToArray();
    }

    private static int seed = 3130;

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
        // Worked example, values checked by hand
        Check(string.Join(",", ZFunction("aabcaabxaaaz")) == "0,1,0,0,3,1,0,0,2,2,1,0", "worked example");
        Check(string.Join(",", ZFunction("aaaa")) == "0,3,2,1", "aaaa");
        Check(string.Join(",", ZFunction("abab")) == "0,0,2,0", "abab");
        Check(string.Join(",", ZFunction("a")) == "0", "single");
        Check(ZFunction("").Length == 0, "empty");

        string text = "aabcaabxaaaz";
        Check(string.Join(",", ZSearch(text, "aab")) == "0,4", "aab");
        Check(string.Join(",", ZSearch(text, "aa")) == "0,4,8,9", "aa");
        Check(string.Join(",", ZSearch(text, "az")) == "10", "az");
        Check(string.Join(",", ZSearch(text, "x")) == "7", "x");
        Check(ZSearch(text, "zz").Length == 0, "zz absent");
        Check(string.Join(",", ZSearch(text, "bcaab")) == "2", "bcaab");
        Check(string.Join(",", ZSearch(text, text)) == "0", "self");
        Check(string.Join(",", ZSearch("abc", "")) == "0,1,2,3", "empty pattern");
        Check(ZSearch("", "a").Length == 0, "empty text");

        var texts = new List<string> { "a", "aa", "aaaa", "abab", "aabcaabxaaaz", "mississippi" };
        for (int i = 0; i < 30; i++) texts.Add(LcgString(24, "ab"));
        for (int i = 0; i < 12; i++) texts.Add(LcgString(18, "abc"));

        foreach (string s in texts)
        {
            int[] z = ZFunction(s);
            Check(string.Join(",", z) == string.Join(",", ZFunctionNaive(s)), "z mismatch " + s);
            for (int i = 0; i < s.Length; i++) Check(z[i] >= 0 && z[i] <= s.Length - i, "z bounds " + s + " " + i);
            var patterns = new List<string>();
            for (int i = 0; i < s.Length; i++)
            {
                for (int extra = 1; extra <= 3; extra++)
                {
                    if (i + extra <= s.Length) patterns.Add(s.Substring(i, extra));
                }
            }
            patterns.AddRange(new[] { "zz", "ababab", "" });
            foreach (string pattern in patterns)
            {
                Check(string.Join(",", ZSearch(s, pattern)) == string.Join(",", NaiveSearch(s, pattern)),
                    "search " + s + " / " + pattern);
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
