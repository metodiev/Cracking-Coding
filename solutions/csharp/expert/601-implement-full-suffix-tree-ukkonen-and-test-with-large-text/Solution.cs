// Task 601 - Implement full suffix tree (Ukkonen) and test with large text. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Ukkonen's online construction. Phase i (i = 0..m-1) turns the implicit tree of text[0..i-1] into
// the implicit tree of text[0..i]. Inside a phase the *active point* (activeNode, activeEdge,
// activeLength) marks where the next extension happens; each extension adds the suffix text[j..i]
// and one of three rules applies: rule 1 - no child with char text[i], so hang a new leaf edge;
// rule 2' - the active point ends inside an edge, so split that edge and hang the leaf below the
// split node; rule 3 - the active point already continues with text[i], so stop the phase (all
// later extensions of the phase would be implicit too). Suffix links turn "walk back to the next
// suffix" into O(1), and stopping on rule 3 keeps the number of extensions O(m).
//
// Simplifications (documented, not behavioural): a unique sentinel '\0' is appended, so no suffix
// is a prefix of another and there is exactly one leaf per suffix; child lookup uses a
// Dictionary<char,int> per node instead of a sigma-sized array (same construction, hashed child
// access, only the characters used are stored). "Large text" is 1500 characters: enough for the
// phase loop and both split cases to run hard, small enough for the brute-force oracles below.
//
// Verified: canonical banana/mississippi structures; leaf count == m; occurrence counts (subtree
// leaf counts) against a brute-force scanner over every substring of ten fixed texts plus random
// probes into random and large texts; shape invariants (open-ended leaf edges, no single-child
// internal nodes, node count <= 2m, leaf string-depth sum == m(m+1)/2); the sorted leaf-label set
// against the sorted suffix set; degenerate inputs (empty, one character, 800 'a', "abab" x150).
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

internal class SuffixTree
{
    private readonly string text;                 // text[^1] == '\0', never appears anywhere else
    private readonly int[] link;
    private readonly int[] parent;
    private readonly int[] edgeStart;
    private readonly int[] edgeEnd;
    private readonly int[] leafCount;
    private readonly int[] depth;
    private readonly List<Dictionary<char, int>> children;
    private int nodes;

    public SuffixTree(string input)
    {
        text = input + '\0';
        int m = text.Length;
        int capacity = 2 * m + 2;
        link = new int[capacity];
        parent = new int[capacity];
        edgeStart = new int[capacity];
        edgeEnd = new int[capacity];
        leafCount = new int[capacity];
        depth = new int[capacity];
        children = new List<Dictionary<char, int>>(capacity);
        for (int i = 0; i < capacity; i++) children.Add(new Dictionary<char, int>());
        parent[0] = -1;
        link[0] = 0;
        nodes = 1;
        int activeNode = 0, activeEdge = -1, activeLength = 0, remaining = 0, lastSplit = -1;
        for (int i = 0; i < m; i++)
        {
            remaining++;
            lastSplit = -1;
            while (remaining > 0)
            {
                if (activeLength == 0) activeEdge = i;
                char head = text[activeEdge];
                if (!children[activeNode].TryGetValue(head, out int child))
                {
                    // Rule 1: new leaf for text[i..]. Here activeLength == 0, so head == text[i].
                    Leaf(activeNode, i);
                    if (lastSplit != -1) { link[lastSplit] = activeNode; lastSplit = -1; }
                }
                else
                {
                    int edgeLength = edgeEnd[child] - edgeStart[child] + 1;
                    if (activeLength >= edgeLength)
                    {
                        // Skip the whole edge: activeEdge slides along the text window that still
                        // spells the path label of the active point (the skip/count trick).
                        activeEdge += edgeLength;
                        activeLength -= edgeLength;
                        activeNode = child;
                        continue;
                    }
                    if (text[edgeStart[child] + activeLength] == text[i])
                    {
                        if (lastSplit != -1) { link[lastSplit] = activeNode; lastSplit = -1; }
                        activeLength++;
                        break;                        // rule 3: implicit tree already correct
                    }
                    int mid = nodes++;                // rule 2': split the text[activeEdge..] edge
                    parent[mid] = activeNode;
                    edgeStart[mid] = edgeStart[child];
                    edgeEnd[mid] = edgeStart[child] + activeLength - 1;
                    link[mid] = 0;
                    children[activeNode][head] = mid;
                    Leaf(mid, i);
                    edgeStart[child] += activeLength;  // the old child keeps the tail of the edge
                    parent[child] = mid;
                    children[mid][text[edgeStart[child]]] = child;
                    if (lastSplit != -1) link[lastSplit] = mid;
                    lastSplit = mid;
                }
                remaining--;
                if (activeNode == 0)
                {
                    if (activeLength > 0)
                    {
                        activeLength--;
                        activeEdge = i - remaining + 1;
                    }
                }
                else
                {
                    activeNode = link[activeNode];
                }
            }
        }

        // Depth and leaf counts. Preorder puts every parent before its children, so the reversed
        // order accumulates the leaf counts bottom-up in one pass; a node with no children is a
        // leaf and contributes 1.
        var order = new List<int>(nodes);
        var stack = new Stack<int>();
        stack.Push(0);
        while (stack.Count > 0)
        {
            int v = stack.Pop();
            order.Add(v);
            foreach (var kv in children[v])
            {
                depth[kv.Value] = depth[v] + edgeEnd[kv.Value] - edgeStart[kv.Value] + 1;
                stack.Push(kv.Value);
            }
        }
        for (int idx = order.Count - 1; idx >= 1; idx--)
        {
            int v = order[idx];
            leafCount[v] = Math.Max(1, leafCount[v]);   // a leaf has no children, so it counts 1
            leafCount[parent[v]] += leafCount[v];
        }
    }

    // A new leaf hanging off p whose edge label is text[start..m-1], keyed by its first character.
    private int Leaf(int p, int start)
    {
        int leaf = nodes++;
        parent[leaf] = p;
        edgeStart[leaf] = start;
        edgeEnd[leaf] = text.Length - 1;
        children[p][text[start]] = leaf;
        return leaf;
    }

    public int NodeCount => nodes;
    public int Leaves => leafCount[0];

    // Walk the pattern down the tree. If it ends inside an edge the answer is the leaf count below
    // that edge, because the pattern is then a prefix of the edge label and of every suffix that
    // descends from it. A missing child or a mismatching label character means "not a substring".
    public int Occurrences(string pattern)
    {
        int node = 0, usedInEdge = 0, edgeHead = -1;
        for (int i = 0; i < pattern.Length; i++)
        {
            if (usedInEdge == 0 && !children[node].TryGetValue(pattern[i], out edgeHead)) return 0;
            if (text[edgeStart[edgeHead] + usedInEdge] != pattern[i]) return 0;
            usedInEdge++;
            if (usedInEdge == edgeEnd[edgeHead] - edgeStart[edgeHead] + 1)
            {
                node = edgeHead;
                usedInEdge = 0;
            }
        }
        return usedInEdge == 0 ? leafCount[node] : leafCount[edgeHead];
    }

    public bool IsSubstring(string pattern) => Occurrences(pattern) > 0;

    // Leaves are the suffixes, so the sum of the leaf string depths is the total length of all
    // suffixes, m(m+1)/2: true for any correct suffix tree, and it pins shape and edge labels.
    public long LeafDepthSum()
    {
        long total = 0;
        for (int v = 1; v < nodes; v++) if (children[v].Count == 0) total += depth[v];
        return total;
    }

    // Two shape invariants of a compressed suffix tree over a sentinel-terminated text: no internal
    // node has a single child (Ukkonen never builds one), and every leaf edge is open ended.
    public bool StructureIsCompressed()
    {
        for (int v = 1; v < nodes; v++)
        {
            int childCount = children[v].Count;
            if (childCount == 1) return false;
            if (childCount == 0 && edgeEnd[v] != text.Length - 1) return false;
        }
        return true;
    }

    public List<string> LeafLabels()
    {
        var labels = new List<string>();
        void Walk(int v, StringBuilder sb)
        {
            if (children[v].Count == 0)
            {
                labels.Add(sb.ToString());
                return;
            }
            foreach (var kv in children[v])
            {
                int len = edgeEnd[kv.Value] - edgeStart[kv.Value] + 1;
                sb.Append(text, edgeStart[kv.Value], len);
                Walk(kv.Value, sb);
                sb.Length -= len;
            }
        }
        Walk(0, new StringBuilder());
        labels.Sort(StringComparer.Ordinal);
        return labels;
    }

    public static int CountOccurrences(string haystack, string needle)
    {
        int count = 0;
        for (int i = 0; i + needle.Length <= haystack.Length; i++)
            if (string.CompareOrdinal(haystack, i, needle, 0, needle.Length) == 0) count++;
        return count;
    }
}

public static class Solution
{
    private static ulong state = 0x9E3779B97F4A7C15UL;

    // splitmix64 with a fixed seed: every generated input is reproducible.
    private static ulong NextRandom()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static string RandomText(int length, string alphabet)
    {
        var sb = new StringBuilder(length);
        for (int i = 0; i < length; i++) sb.Append(alphabet[(int)(NextRandom() % (ulong)alphabet.Length)]);
        return sb.ToString();
    }

    private static string Repeat(string unit, int times) => string.Concat(Enumerable.Repeat(unit, times));

    public static void Main()
    {
        Check(new SuffixTree("a").Leaves == 2 && new SuffixTree("").Leaves == 1, "boundary leaf counts");

        var banana = new SuffixTree("banana");
        Check(banana.Leaves == 7, "banana leaves");
        Check(banana.Occurrences("an") == 2 && banana.Occurrences("ana") == 2 && banana.Occurrences("na") == 2,
            "banana an/ana/na");
        Check(banana.Occurrences("nana") == 1 && banana.Occurrences("banana") == 1 && banana.Occurrences("") == 7,
            "banana nana/whole/empty");
        Check(banana.Occurrences("bn") == 0 && banana.Occurrences("bananas") == 0 && !banana.IsSubstring("z"),
            "banana absent");
        Check(banana.IsSubstring("nan"), "banana substring");

        var mississippi = new SuffixTree("mississippi");
        Check(mississippi.Occurrences("issi") == 2 && mississippi.Occurrences("ss") == 2, "mississippi issi/ss");
        Check(mississippi.Occurrences("ppi") == 1 && mississippi.Occurrences("miss") == 1 &&
            mississippi.Occurrences("pii") == 0, "mississippi ppi/miss/absent");

        // The full sorted leaf-label set equals the sorted suffix set: a complete structural oracle.
        var small = new[] { "", "a", "ab", "aa", "aaa", "abab", "banana", "mississippi", "abracadabra", "abcabcab" };
        foreach (string s in small)
        {
            var tree = new SuffixTree(s);
            var labels = tree.LeafLabels();
            var expected = Enumerable.Range(0, s.Length + 1).Select(i => s.Substring(i)).OrderBy(x => x, StringComparer.Ordinal).ToList();
            Check(labels.Count == s.Length + 1, "leaf count " + s);
            for (int i = 0; i < expected.Count; i++) Check(labels[i] == expected[i] + "\0", "leaf label " + s + " #" + i);
            for (int i = 0; i < s.Length; i++)
            {
                for (int j = i + 1; j <= s.Length; j++)
                {
                    string p = s.Substring(i, j - i);
                    Check(tree.Occurrences(p) == SuffixTree.CountOccurrences(s, p), "occ " + s + " " + p);
                }
            }
            Check(tree.Occurrences("") == s.Length + 1, "empty pattern " + s);
            Check(!tree.IsSubstring("~"), "absent " + s);
        }

        foreach (string alphabet in new[] { "ab", "abc", "abcde" })
        {
            for (int trial = 0; trial < 12; trial++)
            {
                string s = RandomText(60, alphabet);
                var tree = new SuffixTree(s);
                Check(tree.Leaves == s.Length + 1, "random leaves");
                Check(tree.LeafDepthSum() == (long)(s.Length + 1) * (s.Length + 2) / 2, "random depth sum");
                Check(tree.StructureIsCompressed() && tree.NodeCount <= 2 * (s.Length + 1), "random invariants");
                for (int i = 0; i < s.Length; i += 3) Check(tree.IsSubstring(s.Substring(i)), "random suffix " + i);
                for (int probe = 0; probe < 40; probe++)
                {
                    int start = (int)(NextRandom() % (ulong)s.Length);
                    int len = 1 + (int)(NextRandom() % (ulong)(s.Length - start));
                    string p = s.Substring(start, len);
                    Check(tree.Occurrences(p) == SuffixTree.CountOccurrences(s, p), "random occ " + p);
                }
            }
        }

        // Degenerate shapes: a single distinct character and a highly periodic text.
        var repeated = new SuffixTree(new string('a', 800));
        Check(repeated.Leaves == 801 && repeated.LeafDepthSum() == 801L * 802 / 2, "repeated shape");
        for (int len = 1; len <= 40; len++) Check(repeated.Occurrences(Repeat("a", len)) == 801 - len, "repeated " + len);
        Check(repeated.Occurrences("b") == 0, "repeated absent");

        var periodic = new SuffixTree(Repeat("abab", 150));
        Check(periodic.Leaves == 601 && periodic.LeafDepthSum() == 601L * 602 / 2, "periodic shape");
        Check(periodic.Occurrences("ababab") == 298 && periodic.StructureIsCompressed(), "periodic count/branching");

        // Large text: 1501 phases over a 1500 character string.
        string large = RandomText(1500, "abcde");
        var largeTree = new SuffixTree(large);
        Check(largeTree.Leaves == 1501, "large leaves");
        Check(largeTree.LeafDepthSum() == 1501L * 1502 / 2, "large depth sum");
        Check(largeTree.StructureIsCompressed(), "large invariants");
        for (int i = 0; i < large.Length; i += 37) Check(largeTree.Occurrences(large.Substring(i)) >= 1, "large " + i);
        for (int probe = 0; probe < 300; probe++)
        {
            int start = (int)(NextRandom() % (ulong)large.Length);
            int len = Math.Min(1 + (int)(NextRandom() % 25), large.Length - start);
            string p = large.Substring(start, len);
            Check(largeTree.Occurrences(p) == SuffixTree.CountOccurrences(large, p), "large occ " + p);
        }
        Check(!largeTree.IsSubstring(new string('z', 30)), "large absent");

        Console.WriteLine("large 1500 chars -> " + largeTree.NodeCount + " nodes, " + largeTree.Leaves + " leaves, depth sum " +
            largeTree.LeafDepthSum() + "; periodic 600 chars -> " + periodic.NodeCount + " nodes");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
