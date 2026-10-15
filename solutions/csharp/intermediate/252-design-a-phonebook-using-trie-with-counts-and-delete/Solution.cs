// Task 252 - Design a phonebook using trie with counts and delete. (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;

public class TrieNode
{
    public readonly SortedDictionary<char, TrieNode> Children = new SortedDictionary<char, TrieNode>();
    public int Count;                     // duplicates of the name ending here (0 = prefix only)
}

public class Phonebook
{
    public readonly TrieNode Root = new TrieNode();

    private TrieNode Walk(string name)
    {
        var node = Root;
        foreach (char ch in name)
        {
            if (!node.Children.TryGetValue(ch, out var next)) return null;
            node = next;
        }
        return node;
    }

    public int Insert(string name)
    {
        var node = Root;
        foreach (char ch in name)
        {
            if (!node.Children.TryGetValue(ch, out var next))
            {
                next = new TrieNode();
                node.Children[ch] = next;
            }
            node = next;
        }
        return ++node.Count;
    }

    public int CountOf(string name)
    {
        var node = Walk(name);
        return node == null ? 0 : node.Count;
    }

    /** Decrements the count and prunes dead branches; true iff the name existed. */
    public bool Delete(string name)
    {
        if (string.IsNullOrEmpty(name)) return false;
        return DeleteRec(Root, name, 0);
    }

    private static bool DeleteRec(TrieNode node, string name, int i)
    {
        if (i == name.Length)
        {
            if (node.Count == 0) return false;
            node.Count--;
            return true;
        }
        char ch = name[i];
        if (!node.Children.TryGetValue(ch, out var child)) return false;
        if (!DeleteRec(child, name, i + 1)) return false;
        if (child.Count == 0 && child.Children.Count == 0) node.Children.Remove(ch);
        return true;
    }

    /** All stored (name, count) pairs starting with prefix, lexicographically sorted. */
    public List<string> ListPrefix(string prefix)
    {
        var outList = new List<string>();
        var node = Walk(prefix);
        if (node == null) return outList;
        Collect(node, prefix, outList);
        return outList;
    }

    private static void Collect(TrieNode node, string path, List<string> outList)
    {
        if (node.Count > 0) outList.Add(path + "=" + node.Count);
        foreach (var pair in node.Children) Collect(pair.Value, path + pair.Key, outList);
    }

    public List<string> Words() { return ListPrefix(""); }

    public int NodeCount()
    {
        int total = 0;
        var stack = new Stack<TrieNode>();
        stack.Push(Root);
        while (stack.Count > 0)
        {
            var node = stack.Pop();
            total++;
            foreach (var child in node.Children.Values) stack.Push(child);
        }
        return total;
    }
}

public static class Solution
{
    private const string All = "[alan=1, albert=1, alice=2, b=1, barb=1, bob=1]";

    private static Phonebook Load()
    {
        var book = new Phonebook();
        Check(book.Insert("alice") == 1, "first alice");
        Check(book.Insert("alice") == 2, "duplicate increments");
        Check(book.Insert("alan") == 1, "alan");
        Check(book.Insert("albert") == 1, "albert");
        Check(book.Insert("bob") == 1, "bob");
        Check(book.Insert("barb") == 1, "barb");
        Check(book.Insert("b") == 1, "b");
        return book;
    }

    public static void Main()
    {
        var empty = new Phonebook();
        Check(empty.NodeCount() == 1, "empty phonebook is a bare root");

        var book = Load();
        // 1 root + 17 shared-prefix nodes: a,al,ala,alan,alb,albe,alber,albert,ali,alic,
        // alice,b,ba,bar,barb,bo,bob
        Check(book.NodeCount() == 18, "shared prefixes share nodes");

        Check(book.CountOf("alice") == 2, "alice exact count");
        Check(book.CountOf("alan") == 1, "alan exact count");
        Check(book.CountOf("alicia") == 0, "unknown name");
        Check(book.CountOf("al") == 0, "prefix is not a name");
        Check(book.CountOf("alic") == 0, "deep prefix is not a name");
        Check(book.CountOf("") == 0, "empty name");

        Check(Render(book.ListPrefix("al")) == "[alan=1, albert=1, alice=2]", "al prefix");
        Check(Render(book.ListPrefix("ali")) == "[alice=2]", "ali prefix");
        Check(Render(book.ListPrefix("b")) == "[b=1, barb=1, bob=1]", "b prefix");
        Check(book.ListPrefix("z").Count == 0, "no z names");
        Check(Render(book.ListPrefix("")) == All, "all names sorted");
        Check(Render(book.Words()) == All, "Words()");

        Check(book.Delete("alice"), "first alice delete");
        Check(book.CountOf("alice") == 1 && book.NodeCount() == 18, "count decremented");
        Check(Render(book.ListPrefix("al")) == "[alan=1, albert=1, alice=1]", "alice still listed");
        Check(book.Delete("alice"), "second alice delete");
        Check(book.CountOf("alice") == 0 && book.NodeCount() == 15, "alice/alic/ali pruned");
        Check(Render(book.ListPrefix("al")) == "[alan=1, albert=1]", "alice removed");
        Check(book.ListPrefix("ali").Count == 0, "ali branch pruned");
        Check(!book.Delete("alice"), "third alice delete fails");
        Check(!book.Delete("zzz"), "unknown delete fails");
        Check(!book.Delete(""), "empty delete fails");

        Check(book.Delete("b"), "b count hits zero");
        Check(book.CountOf("b") == 0 && book.NodeCount() == 15, "b node kept for bob/barb");
        Check(Render(book.ListPrefix("b")) == "[barb=1, bob=1]", "b no longer a name");

        Check(book.Delete("barb"), "barb delete");
        Check(book.NodeCount() == 12, "barb/bar/ba pruned");
        Check(Render(book.Words()) == "[alan=1, albert=1, bob=1]", "remaining words");

        Check(book.Delete("alan"), "alan delete");
        Check(book.NodeCount() == 10, "alan/ala pruned");
        Check(book.Delete("albert"), "albert delete");
        Check(book.NodeCount() == 4, "root + b + bo + bob remain");
        Check(Render(book.Words()) == "[bob=1]", "only bob left");

        Check(book.Delete("bob"), "bob delete");
        Check(book.NodeCount() == 1, "back to the empty state");
        Check(book.Words().Count == 0 && book.ListPrefix("").Count == 0, "nothing left");
        Check(!book.Delete("bob"), "double delete fails");
        Console.WriteLine("PASS");
    }

    private static string Render(List<string> entries)
    {
        return "[" + string.Join(", ", entries) + "]";
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
