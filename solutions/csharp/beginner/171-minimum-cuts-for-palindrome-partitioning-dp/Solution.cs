// Task 171 - Minimum cuts for palindrome partitioning (DP). (Tier: Beginner, CTCI Ch 8)
using System;

public static class Solution
{
    public static int MinPalindromeCuts(string s)
    {
        int n = s.Length;
        if (n <= 1) return 0;
        var isPal = new bool[n, n];
        var cuts = new int[n];
        for (int end = 0; end < n; end++)
        {
            int best = end;
            for (int start = 0; start <= end; start++)
            {
                if (s[start] == s[end] && (end - start <= 1 || isPal[start + 1, end - 1]))
                {
                    isPal[start, end] = true;
                    best = start == 0 ? 0 : Math.Min(best, cuts[start - 1] + 1);
                }
            }
            cuts[end] = best;
        }
        return cuts[n - 1];
    }

    public static void Main()
    {
        Check(MinPalindromeCuts("aab") == 1, "aab");
        Check(MinPalindromeCuts("a") == 0, "single");
        Check(MinPalindromeCuts("") == 0, "empty");
        Check(MinPalindromeCuts("ab") == 1, "ab");
        Check(MinPalindromeCuts("aabb") == 1, "aabb");
        Check(MinPalindromeCuts("abcba") == 0, "already palindrome");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
