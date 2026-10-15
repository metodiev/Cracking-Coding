// Task 290 - Palindrome partitioning with backtracking. (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static string text = "";
    private static List<string> current = new List<string>();
    private static List<List<string>> results = new List<List<string>>();

    /// <summary>
    /// All ways to cut a string into palindromic parts. Pieces are extended shortest-first,
    /// so partition lists come out in lexicographic order and results are deterministic.
    /// </summary>
    public static List<List<string>> PalindromePartitioning(string s)
    {
        text = s;
        current = new List<string>();
        results = new List<List<string>>();
        Backtrack(0);
        return results;
    }

    private static void Backtrack(int start)
    {
        if (start == text.Length)
        {
            results.Add(new List<string>(current));
            return;
        }
        for (int end = start + 1; end <= text.Length; end++)
        {
            string piece = text.Substring(start, end - start);
            if (!IsPalindrome(piece)) continue;
            current.Add(piece);
            Backtrack(end);
            current.RemoveAt(current.Count - 1);
        }
    }

    /// <summary>Independent checker: two pointers moving towards each other.</summary>
    public static bool IsPalindrome(string piece)
    {
        int left = 0, right = piece.Length - 1;
        while (left < right)
        {
            if (piece[left] != piece[right]) return false;
            left++;
            right--;
        }
        return true;
    }

    /// <summary>Oracle: enumerate all 2^(n-1) cut sets and keep the palindrome-only partitions.</summary>
    public static List<List<string>> BruteForcePartitions(string s)
    {
        int n = s.Length;
        var found = new List<List<string>>();
        if (n == 0)
        {
            found.Add(new List<string>());
            return found;
        }
        for (int mask = 0; mask < 1 << (n - 1); mask++)
        {
            var parts = new List<string>();
            int start = 0;
            for (int i = 0; i < n - 1; i++)
            {
                if (((mask >> i) & 1) == 1)
                {
                    parts.Add(s.Substring(start, i + 1 - start));
                    start = i + 1;
                }
            }
            parts.Add(s.Substring(start));
            if (parts.All(IsPalindrome)) found.Add(parts);
        }
        found.Sort((a, b) => string.CompareOrdinal(KeyOf(a), KeyOf(b)));
        return found;
    }

    private static string KeyOf(List<string> parts) => string.Join(",", parts);

    private static List<string> Canonical(List<List<string>> partitions)
    {
        var keys = partitions.Select(KeyOf).ToList();
        keys.Sort(StringComparer.Ordinal);
        return keys;
    }

    private static List<string> KeyList(List<List<string>> partitions) => partitions.Select(KeyOf).ToList();

    private static int seed = 13579;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void Main()
    {
        Check(SequencesEqual(PalindromePartitioning("aab"), Parts(new[] { "a", "a", "b" }, new[] { "aa", "b" })), "aab");
        Check(SequencesEqual(PalindromePartitioning("a"), Parts(new[] { "a" })), "single char");
        Check(SequencesEqual(PalindromePartitioning("aba"), Parts(new[] { "a", "b", "a" }, new[] { "aba" })), "aba");
        Check(SequencesEqual(PalindromePartitioning("aaa"),
            Parts(new[] { "a", "a", "a" }, new[] { "a", "aa" }, new[] { "aa", "a" }, new[] { "aaa" })), "aaa");
        Check(SequencesEqual(PalindromePartitioning("aaaa"), BruteForcePartitions("aaaa")), "aaaa");
        Check(PalindromePartitioning("aaaa").Count == 8, "aaaa count");
        Check(PalindromePartitioning("abba").Count == 3, "abba count");

        var empty = PalindromePartitioning("");
        Check(empty.Count == 1 && empty[0].Count == 0, "empty input");

        for (int trial = 0; trial < 200; trial++)
        {
            int length = 1 + NextSeed() % 7;
            var builder = new System.Text.StringBuilder();
            for (int i = 0; i < length; i++) builder.Append("abc"[NextSeed() % 3]);
            string s = builder.ToString();
            var partitions = PalindromePartitioning(s);
            Check(Canonical(partitions).SequenceEqual(Canonical(BruteForcePartitions(s))), "oracle " + s);
            Check(KeyList(partitions).SequenceEqual(Canonical(partitions)), "lexicographic " + s);
            var seen = new HashSet<string>();
            foreach (var parts in partitions)
            {
                var joined = new System.Text.StringBuilder();
                foreach (string piece in parts)
                {
                    Check(piece.Length > 0 && IsPalindrome(piece), "palindrome " + s);
                    joined.Append(piece);
                }
                Check(joined.ToString() == s, "concatenation " + s);
                Check(seen.Add(KeyOf(parts)), "duplicate " + s);
            }
            Check(partitions.Count == seen.Count, "count " + s);
        }

        Console.WriteLine("PASS");
    }

    private static List<List<string>> Parts(params string[][] rows)
    {
        var result = new List<List<string>>();
        foreach (var row in rows) result.Add(new List<string>(row));
        return result;
    }

    private static bool SequencesEqual(List<List<string>> actual, List<List<string>> expected)
    {
        if (actual.Count != expected.Count) return false;
        for (int i = 0; i < actual.Count; i++)
        {
            if (actual[i].Count != expected[i].Count) return false;
            for (int j = 0; j < actual[i].Count; j++)
            {
                if (actual[i][j] != expected[i][j]) return false;
            }
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
