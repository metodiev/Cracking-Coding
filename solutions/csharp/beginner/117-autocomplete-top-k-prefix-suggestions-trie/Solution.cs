// Task 117 - Autocomplete top-k prefix suggestions (trie). (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Linq;

public class AcNode
{
    public readonly Dictionary<char, AcNode> Kids = new Dictionary<char, AcNode>();
    public int Weight;
    public int Best;
}

public class Autocomplete
{
    public readonly AcNode Root = new AcNode();

    public void Add(string word, int weight)
    {
        var node = Root;
        foreach (var c in word)
        {
            if (!node.Kids.TryGetValue(c, out var next))
            {
                next = new AcNode();
                node.Kids[c] = next;
            }
            node = next;
        }
        node.Weight += weight;
        Recompute(Root);
    }

    private int Recompute(AcNode node)
    {
        int best = node.Weight;
        foreach (var kid in node.Kids.Values) best = Math.Max(best, Recompute(kid));
        node.Best = best;
        return best;
    }

    public List<string> Suggest(string prefix, int k)
    {
        var node = Root;
        foreach (var c in prefix)
        {
            if (!node.Kids.TryGetValue(c, out node!)) return new List<string>();
        }
        var found = new List<(string word, int weight)>();
        Collect(node, prefix, found);
        return found
            .OrderByDescending(p => p.weight)
            .ThenBy(p => p.word, StringComparer.Ordinal)
            .Take(k)
            .Select(p => p.word)
            .ToList();
    }

    private static void Collect(AcNode node, string path, List<(string, int)> outList)
    {
        if (node.Weight > 0) outList.Add((path, node.Weight));
        foreach (var (ch, kid) in node.Kids) Collect(kid, path + ch, outList);
    }
}

public static class Solution
{
    public static void Main()
    {
        var ac = new Autocomplete();
        ac.Add("car", 5);
        ac.Add("cart", 3);
        ac.Add("carbon", 2);
        ac.Add("care", 1);
        ac.Add("dog", 9);
        Check(string.Join(",", ac.Suggest("car", 2)) == "car,cart", "top 2 by popularity");
        Check(string.Join(",", ac.Suggest("car", 10)) == "car,cart,carbon,care", "all four");
        Check(string.Join(",", ac.Suggest("do", 5)) == "dog", "do -> dog");
        Check(ac.Suggest("z", 3).Count == 0, "unknown prefix");
        Check(string.Join(",", ac.Suggest("", 1)) == "dog", "top overall");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
