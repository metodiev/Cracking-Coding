// Task 116 - Insert and search in trie. (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public class ArrayNode
{
    public readonly ArrayNode?[] Kids = new ArrayNode?[26];
    public bool End;
}

public class WordTrie
{
    public readonly ArrayNode Root = new ArrayNode();

    public void Insert(string word)
    {
        var node = Root;
        foreach (var c in word)
        {
            int i = c - 'a';
            if (node.Kids[i] == null) node.Kids[i] = new ArrayNode();
            node = node.Kids[i]!;
        }
        node.End = true;
    }

    public bool Search(string word)
    {
        var node = NodeFor(word);
        return node != null && node.End;
    }

    public int CountWithPrefix(string prefix)
    {
        var node = NodeFor(prefix);
        return node == null ? 0 : Count(node);
    }

    public List<string> WordsWithPrefix(string prefix)
    {
        var outList = new List<string>();
        var node = NodeFor(prefix);
        if (node != null) Collect(node, prefix, outList);
        return outList;
    }

    private ArrayNode? NodeFor(string s)
    {
        var node = Root;
        foreach (var c in s)
        {
            node = node.Kids[c - 'a'];
            if (node == null) return null;
        }
        return node;
    }

    private static int Count(ArrayNode node)
    {
        int total = node.End ? 1 : 0;
        foreach (var kid in node.Kids) if (kid != null) total += Count(kid);
        return total;
    }

    private static void Collect(ArrayNode node, string path, List<string> outList)
    {
        if (node.End) outList.Add(path);
        for (int i = 0; i < 26; i++)
            if (node.Kids[i] != null) Collect(node.Kids[i]!, path + (char)('a' + i), outList);
    }
}

public static class Solution
{
    public static void Main()
    {
        var t = new WordTrie();
        foreach (var w in new[] { "app", "apple", "apply", "apt", "banana" }) t.Insert(w);
        Check(t.Search("app") && t.Search("apply"), "search hits");
        Check(!t.Search("ap") && !t.Search("apples"), "search misses");
        Check(t.CountWithPrefix("app") == 3, "count app");
        Check(t.CountWithPrefix("b") == 1 && t.CountWithPrefix("z") == 0, "count b/z");
        Check(string.Join(",", t.WordsWithPrefix("app")) == "app,apple,apply", "collect");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
