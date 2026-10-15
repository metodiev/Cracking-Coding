// Task 297 - Edit distance variants (multi-dimensional DP). (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Text;

public static class Solution
{
    /// <summary>Minimum cost to turn a into b using insert/delete/replace operations.</summary>
    public static int EditDistance(string a, string b, int insertCost = 1, int deleteCost = 1, int replaceCost = 1)
    {
        int n = a.Length;
        int m = b.Length;
        var dp = new int[n + 1, m + 1];
        for (int i = 1; i <= n; i++) dp[i, 0] = i * deleteCost;
        for (int j = 1; j <= m; j++) dp[0, j] = j * insertCost;
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                int substitution = a[i - 1] == b[j - 1] ? 0 : replaceCost;
                dp[i, j] = Math.Min(
                    dp[i - 1, j - 1] + substitution,
                    Math.Min(dp[i - 1, j] + deleteCost, dp[i, j - 1] + insertCost));
            }
        }
        return dp[n, m];
    }

    /// <summary>Independent memoized recursion straight from the operation definition.</summary>
    public static int BruteEdit(string a, string b, int insertCost, int deleteCost, int replaceCost)
    {
        var memo = new int[a.Length + 1, b.Length + 1];
        for (int i = 0; i <= a.Length; i++)
            for (int j = 0; j <= b.Length; j++)
                memo[i, j] = -1;
        return Rec(a, b, a.Length, b.Length, insertCost, deleteCost, replaceCost, memo);
    }

    private static int Rec(string a, string b, int i, int j, int ins, int del, int rep, int[,] memo)
    {
        if (i == 0) return j * ins;
        if (j == 0) return i * del;
        if (memo[i, j] >= 0) return memo[i, j];
        int substitution = a[i - 1] == b[j - 1] ? 0 : rep;
        int best = Math.Min(
            Rec(a, b, i - 1, j - 1, ins, del, rep, memo) + substitution,
            Math.Min(Rec(a, b, i - 1, j, ins, del, rep, memo) + del,
                     Rec(a, b, i, j - 1, ins, del, rep, memo) + ins));
        memo[i, j] = best;
        return best;
    }

    private static int seed = 31337;

    private static string LcgString(int maxLength)
    {
        seed = (seed * 75 + 74) % 65537;
        int length = seed % (maxLength + 1);
        var builder = new StringBuilder();
        for (int i = 0; i < length; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            builder.Append("abc"[seed % 3]);
        }
        return builder.ToString();
    }

    public static void Main()
    {
        Check(EditDistance("kitten", "sitting") == 3, "kitten -> sitting");
        Check(EditDistance("flaw", "lawn") == 2, "flaw -> lawn");
        Check(EditDistance("abc", "abc") == 0, "equal strings");
        Check(EditDistance("", "abcd") == 4, "empty -> non-empty");
        Check(EditDistance("abcd", "") == 4, "non-empty -> empty");
        Check(EditDistance("", "") == 0, "both empty");

        Check(EditDistance("ab", "ba", 1, 1, 2) == 2, "replace 2 still loses to delete+insert");
        Check(EditDistance("abc", "axc", 1, 1, 3) == 2, "two edits beat one replacement");
        Check(EditDistance("abc", "axc", 1, 1, 1) == 1, "one replacement with unit costs");

        Check(EditDistance("abc", "abcd", 1, 5, 1) == 1, "cheap insertion");
        Check(EditDistance("abcd", "abc", 1, 5, 1) == 5, "expensive deletion");
        Check(EditDistance("abc", "abcd", 7, 1, 1) == 7, "expensive insertion");

        for (int trial = 0; trial < 200; trial++)
        {
            string a = LcgString(6);
            string b = LcgString(6);
            seed = (seed * 75 + 74) % 65537;
            int insertCost = 1 + seed % 3;
            seed = (seed * 75 + 74) % 65537;
            int deleteCost = 1 + seed % 3;
            seed = (seed * 75 + 74) % 65537;
            int replaceCost = 1 + seed % 3;
            Check(EditDistance(a, b, insertCost, deleteCost, replaceCost)
                == BruteEdit(a, b, insertCost, deleteCost, replaceCost), "random trial " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
