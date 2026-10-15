// Task 309 - Longest common prefix among many strings. (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    /// <summary>min/max bound: the common prefix is the prefix of the two extremes.</summary>
    public static string LongestCommonPrefixSorted(string[] strs)
    {
        if (strs.Length == 0) return "";
        string smallest = strs.Min()!, largest = strs.Max()!;
        int i = 0;
        while (i < smallest.Length && i < largest.Length && smallest[i] == largest[i]) i++;
        return smallest.Substring(0, i);
    }

    /// <summary>Column-by-column scan; the independent oracle.</summary>
    public static string LongestCommonPrefixVertical(string[] strs)
    {
        if (strs.Length == 0) return "";
        for (int column = 0; column < strs[0].Length; column++)
        {
            char ch = strs[0][column];
            for (int k = 1; k < strs.Length; k++)
            {
                if (column >= strs[k].Length || strs[k][column] != ch) return strs[0].Substring(0, column);
            }
        }
        return strs[0];
    }

    /// <summary>Trie walk: keep descending while a single child is present and no word ends here.</summary>
    public static string LongestCommonPrefixTrie(string[] strs)
    {
        var root = new Dictionary<char, object>();
        foreach (string word in strs)
        {
            var node = root;
            foreach (char ch in word)
            {
                if (!node.TryGetValue(ch, out var next))
                {
                    var fresh = new Dictionary<char, object>();
                    node[ch] = fresh;
                    next = fresh;
                }
                node = (Dictionary<char, object>)next;
            }
            node['$'] = true;
        }
        var prefix = new System.Text.StringBuilder();
        var current = root;
        while (current.Count == 1 && !current.ContainsKey('$'))
        {
            var entry = current.First();
            prefix.Append(entry.Key);
            current = (Dictionary<char, object>)entry.Value;
        }
        return prefix.ToString();
    }

    private static int seed = 1592;

    private static List<string[]> LcgLists(int count)
    {
        var lists = new List<string[]>();
        for (int k = 0; k < count; k++)
        {
            int size = 1 + seed % 3;
            var words = new string[size];
            for (int w = 0; w < size; w++)
            {
                int length = 1 + seed % 4;
                var chars = new char[length];
                for (int i = 0; i < length; i++)
                {
                    seed = (seed * 75 + 74) % 65537;
                    chars[i] = "abc"[seed % 3];
                }
                words[w] = new string(chars);
            }
            seed = (seed * 75 + 74) % 65537;
            lists.Add(words);
        }
        return lists;
    }

    public static void Main()
    {
        Check(LongestCommonPrefixSorted(new[] { "flower", "flow", "flight" }) == "fl", "flower group");
        Check(LongestCommonPrefixSorted(new[] { "dog", "racecar", "car" }) == "", "no common prefix");
        Check(LongestCommonPrefixSorted(new[] { "abc", "abc" }) == "abc", "identical");
        Check(LongestCommonPrefixSorted(new[] { "a" }) == "a", "single");
        Check(LongestCommonPrefixSorted(Array.Empty<string>()) == "", "empty list");
        Check(LongestCommonPrefixSorted(new[] { "", "b" }) == "", "empty string");
        Check(LongestCommonPrefixSorted(new[] { "abab", "aba", "ababbb" }) == "aba", "nested words");

        var samples = new[]
        {
            new[] { "flower", "flow", "flight" }, new[] { "dog", "racecar", "car" },
            new[] { "abc", "abc" }, new[] { "", "b" },
        };
        foreach (var strs in samples)
        {
            string expected = LongestCommonPrefixSorted(strs);
            Check(LongestCommonPrefixVertical(strs) == expected, "vertical " + string.Join(",", strs));
            Check(LongestCommonPrefixTrie(strs) == expected, "trie " + string.Join(",", strs));
        }

        foreach (var strs in LcgLists(80))
        {
            string expected = LongestCommonPrefixSorted(strs);
            Check(LongestCommonPrefixVertical(strs) == expected, "random vertical " + string.Join(",", strs));
            Check(LongestCommonPrefixTrie(strs) == expected, "random trie " + string.Join(",", strs));
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
