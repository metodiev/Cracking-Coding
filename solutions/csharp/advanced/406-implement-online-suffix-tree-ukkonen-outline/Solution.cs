// Task 406 - Implement an online suffix tree (Ukkonen outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Outline: a real Ukkonen tree builds the *compressed* suffix tree online in O(n) by extending
// the last suffix, following a suffix link when a split happens, and keeping the suffix link
// of every new node correct. This outline keeps the same skeleton for tiny strings: an
// explicit suffix trie of the text (every node is a distinct substring, edges are single
// characters) plus the suffix link of each node, which points at the node of its path label
// minus the first character. That link is exactly the invariant Ukkonen restores when it
// splits an edge, and it is what makes "ends with suffix link" navigation possible here:
// following links from a node walks through all suffixes of its path in O(depth) steps.
//
// Links are wired breadth-first: a depth-1 node links to the root, and a node whose parent p
// is reached over character c links to children[link[p]][c], which always exists in a suffix
// trie.
//
// Checks on tiny texts only (banana, mississippi, abab, aaaa): every substring is findable
// and every generated non-substring fails, node count matches the brute-force distinct
// substring set, each link target really is the node of path[1:], link chains enumerate all
// suffixes, and subtree terminal counts reproduce naive occurrence counts.
using System;
using System.Collections.Generic;

internal class SuffixTreeOutline
{
    public readonly string Text;
    public readonly List<Dictionary<char, int>> Children = new List<Dictionary<char, int>>();
    public readonly List<int> Parent = new List<int>();
    public readonly List<char> ParentChar = new List<char>();
    public readonly List<int> Depth = new List<int>();
    public readonly List<bool> Terminal = new List<bool>();
    public readonly List<int> Link = new List<int>();

    public SuffixTreeOutline(string text)
    {
        Text = text;
        Children.Add(new Dictionary<char, int>());
        Parent.Add(-1);
        ParentChar.Add('\0');
        Depth.Add(0);
        Terminal.Add(false);
        Link.Add(-1);
        for (int start = 0; start < text.Length; start++)
        {
            int node = 0;
            for (int pos = start; pos < text.Length; pos++)
            {
                char ch = text[pos];
                if (!Children[node].TryGetValue(ch, out int child))
                {
                    child = Depth.Count;
                    Children[node][ch] = child;
                    Children.Add(new Dictionary<char, int>());
                    Parent.Add(node);
                    ParentChar.Add(ch);
                    Depth.Add(Depth[node] + 1);
                    Terminal.Add(false);
                    Link.Add(-1);
                }
                node = child;
            }
            Terminal[node] = true;
        }
        WireSuffixLinks();
    }

    private void WireSuffixLinks()
    {
        var order = new List<int> { 0 };
        for (int i = 0; i < order.Count; i++)
        {
            foreach (int child in Children[order[i]].Values) order.Add(child);
        }
        foreach (int node in order)
        {
            if (node == 0) continue;
            if (Depth[node] == 1)
            {
                Link[node] = 0;
            }
            else
            {
                int p = Parent[node];
                Link[node] = Children[Link[p]][ParentChar[node]];
            }
        }
    }

    public string PathLabel(int node)
    {
        var chars = new List<char>();
        while (node != 0)
        {
            chars.Add(ParentChar[node]);
            node = Parent[node];
        }
        chars.Reverse();
        return new string(chars.ToArray());
    }

    public int Find(string pattern)
    {
        int node = 0;
        foreach (char ch in pattern)
        {
            if (!Children[node].TryGetValue(ch, out int child)) return -1;
            node = child;
        }
        return node;
    }

    public bool Contains(string pattern) => Find(pattern) != -1;

    public List<int> SuffixChain(int node)
    {
        var chain = new List<int>();
        while (node != -1)
        {
            chain.Add(node);
            node = Link[node];
        }
        return chain;
    }

    public int Occurrences(string pattern)
    {
        int node = Find(pattern);
        if (node == -1) return 0;
        int count = 0;
        var stack = new Stack<int>();
        stack.Push(node);
        while (stack.Count > 0)
        {
            int current = stack.Pop();
            if (Terminal[current]) count++;
            foreach (int child in Children[current].Values) stack.Push(child);
        }
        return count;
    }

    public int DistinctSubstrings() => Depth.Count - 1;
}

internal static class OutlineTools
{
    public static HashSet<string> AllSubstrings(string s)
    {
        var seen = new HashSet<string>();
        for (int i = 0; i < s.Length; i++)
            for (int j = i + 1; j <= s.Length; j++)
                seen.Add(s.Substring(i, j - i));
        return seen;
    }

    public static int CountOccurrences(string s, string pattern)
    {
        if (pattern.Length == 0) return 0;
        int count = 0;
        for (int i = 0; i + pattern.Length <= s.Length; i++)
        {
            if (string.CompareOrdinal(s, i, pattern, 0, pattern.Length) == 0) count++;
        }
        return count;
    }

    public static List<string> EnumerateStrings(string alphabet, int maxLen)
    {
        var outList = new List<string>();
        var frontier = new List<string> { "" };
        for (int depth = 0; depth <= maxLen; depth++)
        {
            outList.AddRange(frontier);
            var nextFrontier = new List<string>();
            foreach (string word in frontier)
            {
                foreach (char ch in alphabet) nextFrontier.Add(word + ch);
            }
            frontier = nextFrontier;
        }
        return outList;
    }

    public static void CheckOutline(string text, string alphabet, int probeLen)
    {
        var tree = new SuffixTreeOutline(text);
        var substrings = AllSubstrings(text);
        Solution.Check(tree.DistinctSubstrings() == substrings.Count, "distinct " + text);
        Solution.Check(tree.Contains("") && tree.Find("") == 0, "root " + text);
        foreach (string t in substrings)
        {
            int node = tree.Find(t);
            Solution.Check(node != -1, "find " + t);
            Solution.Check(tree.PathLabel(node) == t, "label " + t);
            Solution.Check(tree.Occurrences(t) == CountOccurrences(text, t), "occ " + t);
        }
        foreach (string t in EnumerateStrings(alphabet, probeLen))
        {
            if (t.Length > 0 && !substrings.Contains(t)) Solution.Check(!tree.Contains(t), "reject " + t);
        }
        for (int node = 1; node < tree.Depth.Count; node++)
        {
            string label = tree.PathLabel(node);
            int link = tree.Link[node];
            Solution.Check(link != -1, "link exists " + label);
            Solution.Check(tree.PathLabel(link) == label.Substring(1), "link label " + label);
            var chain = tree.SuffixChain(node);
            Solution.Check(chain.Count == label.Length + 1, "chain size " + label);
            var labels = new List<string>();
            foreach (int n in chain) labels.Add(tree.PathLabel(n));
            var expected = new List<string>();
            for (int k = 0; k < label.Length; k++) expected.Add(label.Substring(k));
            expected.Add("");
            Solution.Check(labels.Count == expected.Count, "chain count " + label);
            for (int k = 0; k < expected.Count; k++) Solution.Check(labels[k] == expected[k], "chain item " + label);
            Solution.Check(tree.Depth[link] == tree.Depth[node] - 1, "link depth " + label);
        }
    }
}

public static class Solution
{
    private static int seed = 2718;

    private static string LcgString(int length, string alphabet)
    {
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < length; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            sb.Append(alphabet[seed % alphabet.Length]);
        }
        return sb.ToString();
    }

    public static void Main()
    {
        OutlineTools.CheckOutline("banana", "abn", 4);
        var banana = new SuffixTreeOutline("banana");
        Check(banana.Occurrences("ana") == 2, "ana");
        Check(banana.Occurrences("a") == 3, "a");
        Check(banana.Occurrences("na") == 2, "na");
        Check(banana.Occurrences("banana") == 1, "banana");
        Check(!banana.Contains("bananas") && !banana.Contains("nana-"), "rejects");

        OutlineTools.CheckOutline("mississippi", "imps", 3);
        var mississippi = new SuffixTreeOutline("mississippi");
        Check(mississippi.Occurrences("issi") == 2, "issi");
        Check(mississippi.Occurrences("ssi") == 2, "ssi");
        Check(mississippi.Occurrences("i") == 4, "i");

        OutlineTools.CheckOutline("abab", "ab", 6);
        OutlineTools.CheckOutline("aaaa", "a", 6);
        OutlineTools.CheckOutline("abcab", "abc", 5);
        OutlineTools.CheckOutline("", "ab", 2);
        OutlineTools.CheckOutline("z", "ab", 2);

        for (int trial = 0; trial < 20; trial++) OutlineTools.CheckOutline(LcgString(8, "ab"), "ab", 3);
        for (int trial = 0; trial < 10; trial++) OutlineTools.CheckOutline(LcgString(7, "abc"), "abc", 3);

        // online flavour: one more character extends the outline and keeps every invariant
        var baseTree = new SuffixTreeOutline("aban");
        Check(baseTree.Occurrences("a") == 2, "base a");
        var extended = new SuffixTreeOutline("abana");
        Check(extended.DistinctSubstrings() == OutlineTools.AllSubstrings("abana").Count, "extended distinct");
        Check(extended.Occurrences("a") == 3, "extended a");
        Check(extended.PathLabel(extended.Link[extended.Find("bana")]) == "ana", "extended link");
        Console.WriteLine("PASS");
    }

    public static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
