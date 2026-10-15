// Task 115 - Implement a basic trie for lowercase words. (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public class TrieNode
{
    public readonly Dictionary<char, TrieNode> Children = new Dictionary<char, TrieNode>();
    public bool IsWord;
}

public class Trie
{
    public readonly TrieNode Root = new TrieNode();
    public int WordCount;

    public void Insert(string word)
    {
        var node = Root;
        foreach (var c in word)
        {
            if (!node.Children.TryGetValue(c, out var next))
            {
                next = new TrieNode();
                node.Children[c] = next;
            }
            node = next;
        }
        if (!node.IsWord)
        {
            node.IsWord = true;
            WordCount++;
        }
    }

    public bool Contains(string word)
    {
        var node = Walk(word);
        return node != null && node.IsWord;
    }

    public bool StartsWith(string prefix) => Walk(prefix) != null;

    private TrieNode? Walk(string s)
    {
        var node = Root;
        foreach (var c in s)
        {
            if (!node.Children.TryGetValue(c, out node!)) return null;
        }
        return node;
    }
}

public static class Solution
{
    public static void Main()
    {
        var t = new Trie();
        foreach (var w in new[] { "cat", "car", "card", "dog" }) t.Insert(w);
        Check(t.WordCount == 4, "four words");
        t.Insert("cat");
        Check(t.WordCount == 4, "duplicate not double counted");
        Check(t.Contains("cat") && t.Contains("card"), "contains");
        Check(!t.Contains("ca") && !t.Contains("care"), "prefixes are not words");
        Check(t.StartsWith("ca") && t.StartsWith("") && !t.StartsWith("z"), "startsWith");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
