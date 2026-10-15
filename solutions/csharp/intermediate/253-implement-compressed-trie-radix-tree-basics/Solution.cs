// Task 253 - Implement compressed trie (radix tree) basics. (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Linq;

public class RadixNode
{
    public string Label;
    public readonly SortedDictionary<char, RadixNode> Children = new SortedDictionary<char, RadixNode>();
    public bool IsWord;

    public RadixNode(string label) { Label = label; }
}

public static class Solution
{
    public static bool Insert(RadixNode root, string word)
    {
        var node = root;
        string rest = word;
        while (true)
        {
            if (rest.Length == 0)
            {
                bool wasNew = !node.IsWord;
                node.IsWord = true;
                return wasNew;
            }
            char first = rest[0];
            if (!node.Children.TryGetValue(first, out var child))
            {
                var fresh = new RadixNode(rest);
                fresh.IsWord = true;
                node.Children[first] = fresh;
                return true;
            }
            int common = 0;
            while (common < rest.Length && common < child.Label.Length
                   && rest[common] == child.Label[common]) common++;
            if (common == child.Label.Length)                  // consumed the whole edge
            {
                node = child;
                rest = rest.Substring(common);
                continue;
            }
            var split = new RadixNode(child.Label.Substring(0, common));   // split edge
            child.Label = child.Label.Substring(common);
            split.Children[child.Label[0]] = child;
            node.Children[first] = split;
            rest = rest.Substring(common);
            if (rest.Length == 0)
            {
                split.IsWord = true;
            }
            else
            {
                var fresh = new RadixNode(rest);
                fresh.IsWord = true;
                split.Children[rest[0]] = fresh;
            }
            return true;
        }
    }

    public static bool Contains(RadixNode root, string word)
    {
        var node = root;
        int i = 0;
        while (i < word.Length)
        {
            if (!node.Children.TryGetValue(word[i], out var child)) return false;
            if (i + child.Label.Length > word.Length) return false;
            if (word.Substring(i, child.Label.Length) != child.Label) return false;
            i += child.Label.Length;
            node = child;
        }
        return node.IsWord;
    }

    public static List<string> WordsWithPrefix(RadixNode root, string prefix)
    {
        var node = root;
        string soFar = "";
        int i = 0;
        while (i < prefix.Length)
        {
            if (!node.Children.TryGetValue(prefix[i], out var child)) return new List<string>();
            int want = prefix.Length - i;
            int take = Math.Min(want, child.Label.Length);
            if (child.Label.Substring(0, take) != prefix.Substring(i, take)) return new List<string>();
            node = child;
            soFar += child.Label;
            if (want <= child.Label.Length) break;             // prefix ends inside this edge
            i += child.Label.Length;
        }
        var outList = new List<string>();
        Collect(node, soFar, outList);
        return outList;
    }

    private static void Collect(RadixNode node, string soFar, List<string> outList)
    {
        if (node.IsWord) outList.Add(soFar);
        foreach (var pair in node.Children) Collect(pair.Value, soFar + pair.Value.Label, outList);
    }

    public static bool StartsWith(RadixNode root, string prefix)
    {
        return WordsWithPrefix(root, prefix).Count > 0;
    }

    // Deterministic compressed rendering: labels glued by commas, '*' = stored word.
    public static string Shape(RadixNode node)
    {
        string text = node.Label + (node.IsWord ? "*" : "");
        if (node.Children.Count == 0) return text;
        var parts = node.Children.Values.Select(Shape).ToList();
        return text + "[" + string.Join(",", parts) + "]";
    }

    // Number of nodes excluding the root, i.e. the number of edges.
    public static int NodeCount(RadixNode root)
    {
        int total = root.Children.Count;
        foreach (var child in root.Children.Values) total += NodeCount(child);
        return total;
    }

    public static int PlainTrieEdges(string[] words)
    {
        var prefixes = new HashSet<string>();
        foreach (string word in words)
        {
            for (int i = 1; i <= word.Length; i++) prefixes.Add(word.Substring(0, i));
        }
        return prefixes.Count;
    }

    public static void Main()
    {
        var root = new RadixNode("");
        Check(Insert(root, "cart"), "cart inserted");
        Check(Shape(root) == "[cart*]", "cart alone");
        Check(Insert(root, "car"), "car splits the edge");
        Check(Shape(root) == "[car*[t*]]", "split: car* + t*");
        Check(Insert(root, "care"), "care joins under car");
        Check(Shape(root) == "[car*[e*,t*]]", "compressed shape car*[e*,t*]");
        Check(!Insert(root, "car"), "duplicate word");

        Check(Contains(root, "car") && Contains(root, "care") && Contains(root, "cart"), "lookups");
        foreach (string miss in new[] { "ca", "c", "carts", "caret", "card", "" })
        {
            Check(!Contains(root, miss), "not a word: " + miss);
        }

        Check(NodeCount(root) == 3, "three edges");
        Check(PlainTrieEdges(new[] { "cart", "car", "care" }) == 5, "plain trie edges");
        Check(NodeCount(root) < PlainTrieEdges(new[] { "cart", "car", "care" }),
              "compressed smaller than plain");

        Check(Insert(root, "do") && Insert(root, "dot") && Insert(root, "dog"), "do family");
        Check(Shape(root) == "[car*[e*,t*],do*[g*,t*]]", "two families");

        Check(Render(WordsWithPrefix(root, "car")) == "[car, care, cart]", "car prefix");
        Check(Render(WordsWithPrefix(root, "ca")) == "[car, care, cart]", "prefix inside edge");
        Check(Render(WordsWithPrefix(root, "care")) == "[care]", "care prefix");
        Check(Render(WordsWithPrefix(root, "do")) == "[do, dog, dot]", "do prefix");
        Check(Render(WordsWithPrefix(root, "d")) == "[do, dog, dot]", "d prefix");
        Check(Render(WordsWithPrefix(root, "")) == "[car, care, cart, do, dog, dot]", "all words");
        Check(WordsWithPrefix(root, "z").Count == 0, "no z words");
        Check(WordsWithPrefix(root, "carts").Count == 0, "carts missing");
        Check(StartsWith(root, "car") && !StartsWith(root, "cat"), "StartsWith");

        Check(NodeCount(root) == 6, "six edges total");
        Check(PlainTrieEdges(new[] { "car", "care", "cart", "do", "dog", "dot" }) == 9, "plain trie has 9");
        foreach (string word in new[] { "car", "care", "cart", "do", "dog", "dot" })
        {
            Check(Contains(root, word), "still stored: " + word);
        }
        foreach (string miss in new[] { "co", "dat", "dodge", "done" })
        {
            Check(!Contains(root, miss), "still missing: " + miss);
        }
        Console.WriteLine("PASS");
    }

    private static string Render(List<string> values)
    {
        return "[" + string.Join(", ", values) + "]";
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
