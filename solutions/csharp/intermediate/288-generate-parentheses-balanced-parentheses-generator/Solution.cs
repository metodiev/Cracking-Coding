// Task 288 - Generate all balanced parentheses strings. (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    /// <summary>
    /// Open-first DFS: at each step add "(" if some pair is still unopened, then ")" if the
    /// closing count is below the opening count. This emits strings in lexicographic order
    /// ("(" < ")"), so results are deterministic.
    /// </summary>
    public static List<string> GenerateParentheses(int n)
    {
        var results = new List<string>();
        if (n < 0) return results;
        Build("", n, 0, 0, results);
        return results;
    }

    private static void Build(string current, int n, int opens, int closes, List<string> results)
    {
        if (current.Length == 2 * n)
        {
            results.Add(current);
            return;
        }
        if (opens < n) Build(current + "(", n, opens + 1, closes, results);
        if (closes < opens) Build(current + ")", n, opens, closes + 1, results);
    }

    /// <summary>Independent checker: only parentheses, no negative prefix, ends at depth 0.</summary>
    public static bool IsBalanced(string text)
    {
        int depth = 0;
        foreach (char ch in text)
        {
            if (ch == '(')
            {
                depth++;
            }
            else if (ch == ')')
            {
                depth--;
                if (depth < 0) return false;
            }
            else
            {
                return false;
            }
        }
        return depth == 0;
    }

    /// <summary>C(0)=1, C(n+1) = C(n) * 2 * (2n + 1) / (n + 2).</summary>
    public static int Catalan(int n)
    {
        int value = 1;
        for (int i = 0; i < n; i++) value = value * 2 * (2 * i + 1) / (i + 2);
        return value;
    }

    /// <summary>Enumerates all 2^(2n) parenthesis strings and keeps the balanced ones.</summary>
    public static List<string> BruteForceBalanced(int n)
    {
        var results = new List<string>();
        for (int mask = 0; mask < 1 << (2 * n); mask++)
        {
            var chars = new char[2 * n];
            for (int bit = 0; bit < 2 * n; bit++) chars[bit] = ((mask >> bit) & 1) == 1 ? '(' : ')';
            string text = new string(chars);
            if (IsBalanced(text)) results.Add(text);
        }
        results.Sort(StringComparer.Ordinal);
        return results;
    }

    public static void Main()
    {
        Check(GenerateParentheses(0).SequenceEqual(new[] { "" }), "n=0");
        Check(GenerateParentheses(1).SequenceEqual(new[] { "()" }), "n=1");
        Check(GenerateParentheses(2).SequenceEqual(new[] { "(())", "()()" }), "n=2");
        Check(GenerateParentheses(3).SequenceEqual(new[] { "((()))", "(()())", "(())()", "()(())", "()()()" }), "n=3");
        Check(GenerateParentheses(-1).Count == 0, "negative n");

        for (int n = 0; n <= 10; n++)
        {
            var strings = GenerateParentheses(n);
            Check(strings.Count == Catalan(n), "catalan " + n);
            for (int i = 1; i < strings.Count; i++)
            {
                Check(string.CompareOrdinal(strings[i - 1], strings[i]) < 0, "lexicographic " + n);
            }
            foreach (string text in strings)
            {
                Check(text.Length == 2 * n, "length " + n);
                Check(IsBalanced(text), "balanced " + n + " " + text);
                int opens = text.Count(c => c == '(');
                Check(opens == n && text.Length - opens == n, "pairs " + n);
            }
            Check(GenerateParentheses(n).SequenceEqual(strings), "deterministic " + n);
        }

        for (int n = 0; n <= 7; n++)
        {
            Check(GenerateParentheses(n).SequenceEqual(BruteForceBalanced(n)), "oracle " + n);
        }

        Check(Catalan(0) == 1 && Catalan(1) == 1 && Catalan(2) == 2 && Catalan(3) == 5, "catalan small");
        Check(Catalan(4) == 14 && Catalan(5) == 42 && Catalan(8) == 1430 && Catalan(10) == 16796, "catalan known");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
