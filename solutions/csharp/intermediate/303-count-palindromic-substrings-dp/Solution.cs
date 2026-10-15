// Task 303 - Count palindromic substrings. (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    public static int CountPalindromicSubstrings(string s)
    {
        int n = s.Length;
        int count = 0;
        for (int center = 0; center < n; center++)
        {
            foreach (var (left0, right0) in new[] { (center, center), (center, center + 1) })
            {
                int left = left0, right = right0;
                while (left >= 0 && right < n && s[left] == s[right])
                {
                    count++;
                    left--;
                    right++;
                }
            }
        }
        return count;
    }

    public static int BruteForce(string s)
    {
        int total = 0;
        for (int i = 0; i < s.Length; i++)
        {
            for (int j = i + 1; j <= s.Length; j++)
            {
                string part = s.Substring(i, j - i);
                if (part == new string(part.Reverse().ToArray())) total++;
            }
        }
        return total;
    }

    private static int seed = 2468;

    private static string LcgString(int length)
    {
        var chars = new char[length];
        for (int i = 0; i < length; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            chars[i] = "abc"[seed % 3];
        }
        return new string(chars);
    }

    public static void Main()
    {
        Check(CountPalindromicSubstrings("abc") == 3, "abc");
        Check(CountPalindromicSubstrings("aaa") == 6, "aaa");
        Check(CountPalindromicSubstrings("abba") == 6, "abba");
        Check(CountPalindromicSubstrings("") == 0, "empty");
        Check(CountPalindromicSubstrings("a") == 1, "single");

        foreach (string s in new[] { "aab", "aba", "abcba", "abccba", "aaaa" })
            Check(CountPalindromicSubstrings(s) == BruteForce(s), "brute " + s);

        for (int trial = 0; trial < 60; trial++)
        {
            string s = LcgString(9);
            Check(CountPalindromicSubstrings(s) == BruteForce(s), "random " + s);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
