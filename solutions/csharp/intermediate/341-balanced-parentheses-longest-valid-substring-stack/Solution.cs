// Task 341 - Longest valid parentheses substring (stack of indices). (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    /// <summary>Stack of indices; the bottom holds the position of the last unmatched ')'.</summary>
    public static int LongestValidSubstringStack(string s)
    {
        var stack = new int[s.Length + 1];
        int top = 0;
        stack[top] = -1;
        int best = 0;
        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == '(')
            {
                stack[++top] = i;
            }
            else
            {
                top--;                                     // pop
                if (top < 0) stack[++top] = i;             // this ')' becomes the new base
                else best = Math.Max(best, i - stack[top]);
            }
        }
        return best;
    }

    /// <summary>dp[i] = length of the longest valid substring ending exactly at i.</summary>
    public static int LongestValidSubstringDp(string s)
    {
        int n = s.Length;
        var dp = new int[n];
        int best = 0;
        for (int i = 0; i < n; i++)
        {
            if (s[i] != ')') continue;
            if (i >= 1 && s[i - 1] == '(')
            {
                dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
            }
            else if (i >= 1 && dp[i - 1] > 0)
            {
                int j = i - dp[i - 1] - 1;
                if (j >= 0 && s[j] == '(') dp[i] = dp[i - 1] + 2 + (j >= 1 ? dp[j - 1] : 0);
            }
            best = Math.Max(best, dp[i]);
        }
        return best;
    }

    /// <summary>Exhaustive oracle: every substring that stays balanced and returns to 0.</summary>
    public static int LongestValidSubstringBrute(string s)
    {
        int best = 0;
        for (int i = 0; i < s.Length; i++)
        {
            int depth = 0;
            for (int j = i; j < s.Length; j++)
            {
                depth += s[j] == '(' ? 1 : -1;
                if (depth < 0) break;
                if (depth == 0) best = Math.Max(best, j - i + 1);
            }
        }
        return best;
    }

    private static int seed = 34142;

    private static string LcgString(int length)
    {
        var chars = new char[length];
        for (int i = 0; i < length; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            chars[i] = "()"[seed % 2];
        }
        return new string(chars);
    }

    public static void Main()
    {
        Check(LongestValidSubstringStack("(()") == 2, "(() -> 2");
        Check(LongestValidSubstringStack(")()())") == 4, ")()()) -> 4");
        Check(LongestValidSubstringStack("") == 0, "empty -> 0");
        Check(LongestValidSubstringStack("()(())") == 6, "()(()) -> 6");
        Check(LongestValidSubstringStack("()(()") == 2, "()(() -> 2");
        Check(LongestValidSubstringStack("((())") == 4, "((()) -> 4");
        Check(LongestValidSubstringStack("())") == 2, "()) -> 2");
        Check(LongestValidSubstringStack(")(") == 0, ")(");
        Check(LongestValidSubstringStack("(((") == 0, "(((");
        Check(LongestValidSubstringStack("()") == 2, "() -> 2");
        Check(LongestValidSubstringStack("()()") == 4, "()() -> 4");
        Check(LongestValidSubstringStack(")()(") == 2, ")()( -> 2");

        string[] hand = { "(()", ")()())", "", "()(())", "()(()", "((())", "())", ")(", "(((", "()", "()()", ")()(" };
        foreach (string s in hand)
        {
            Check(LongestValidSubstringDp(s) == LongestValidSubstringStack(s), "dp vs stack " + s);
            Check(LongestValidSubstringBrute(s) == LongestValidSubstringStack(s), "brute vs stack " + s);
        }

        for (int trial = 0; trial < 150; trial++)
        {
            seed = (seed * 75 + 74) % 65537;
            string s = LcgString(seed % 25);
            int expected = LongestValidSubstringBrute(s);
            Check(LongestValidSubstringStack(s) == expected, "stack random " + s);
            Check(LongestValidSubstringDp(s) == expected, "dp random " + s);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
