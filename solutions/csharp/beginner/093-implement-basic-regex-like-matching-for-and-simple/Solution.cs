// Task 093 - Implement basic regex-like matching for '.' and '*'. (Tier: Beginner, CTCI Ch 8)
using System;

public static class Solution
{
    public static bool IsMatch(string s, string p)
    {
        var memo = new int[s.Length + 1, p.Length + 1];  // 0 unknown, 1 true, 2 false
        return Rec(s, p, 0, 0, memo);
    }

    private static bool Rec(string s, string p, int i, int j, int[,] memo)
    {
        if (memo[i, j] != 0) return memo[i, j] == 1;
        bool res;
        if (j == p.Length)
        {
            res = i == s.Length;
        }
        else
        {
            bool first = i < s.Length && (p[j] == s[i] || p[j] == '.');
            if (j + 1 < p.Length && p[j + 1] == '*')
            {
                res = Rec(s, p, i, j + 2, memo) || (first && Rec(s, p, i + 1, j, memo));
            }
            else
            {
                res = first && Rec(s, p, i + 1, j + 1, memo);
            }
        }
        memo[i, j] = res ? 1 : 2;
        return res;
    }

    public static void Main()
    {
        Check(IsMatch("", ""), "empty/empty");
        Check(!IsMatch("", "a"), "empty/a");
        Check(IsMatch("", "a*"), "empty/a*");
        Check(IsMatch("", ".*"), "empty/.*");
        Check(!IsMatch("a", ""), "a/empty");
        Check(IsMatch("a", "a"), "a/a");
        Check(IsMatch("a", "."), "a/.");
        Check(!IsMatch("a", "b"), "a/b");
        Check(!IsMatch("aa", "a"), "aa/a");
        Check(IsMatch("aa", "a*"), "aa/a*");
        Check(IsMatch("aa", ".*"), "aa/.*");
        Check(IsMatch("ab", ".*"), "ab/.*");
        Check(IsMatch("aab", "c*a*b"), "aab/c*a*b");
        Check(IsMatch("abc", "ab.*c"), "abc/ab.*c");
        Check(IsMatch("abXYZc", "ab.*c"), "abXYZc/ab.*c");
        Check(!IsMatch("ab", "ab.*c"), "ab/ab.*c");
        Check(IsMatch("aaa", "a*a"), "aaa/a*a");
        Check(!IsMatch("aaa", "ab*a"), "aaa/ab*a");
        Check(!IsMatch("mississippi", "mis*is*p*."), "mississippi");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
