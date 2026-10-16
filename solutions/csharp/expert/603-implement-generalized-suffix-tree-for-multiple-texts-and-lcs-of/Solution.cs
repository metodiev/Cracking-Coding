// Task 603 - Implement generalized suffix tree for multiple texts and LCS of many strings. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// The generalized suffix tree of texts t0..t{k-1} is the compressed trie of all their suffixes. The
// texts are concatenated as t0 + S0 + t1 + S1 + ... with one *unique* sentinel char per text (chars
// 1..k, below every ordinary character), so no suffix of one text can be mistaken for a suffix of
// another, no suffix is a prefix of another, and every suffix ends in a leaf. Every leaf remembers
// its owner text, so a node's subtree knows which texts contain the node's label:
//   DocumentsOf / DocumentCount - which / how many texts contain p (one bit per text);
//   OccurrencesIn(p, t)         - how often p occurs inside t (owner-filtered leaf count);
//   LongestCommonSubstringOfAll / ...Of(subset) - deepest label covering all / a chosen subset.
// A label is concat[labelPos[node] .. +len]: while inserting, the label of the node being walked
// always occurs at pos - depth[node], so an edge split records its start in O(1). Labels are cut at
// the first sentinel, and a node label prefixes every leaf label below it, so a sentinel-free label
// covers exactly the texts in that node's mask.
//
// Simplifications (documented, not behavioural): suffixes are inserted one by one with skip-down
// navigation instead of Ukkonen's online linearisation, so construction costs O(total length *
// depth) rather than O(total length); the tree itself is a complete generalized suffix tree. Sizes:
// 1..6 texts of at most 60 characters in the checks, at most 64 texts overall (one bit per text);
// texts must not contain the reserved control characters 1..k or '\0'; an empty text has no
// suffixes, so it is reported as containing nothing, not even the empty pattern.
//
// Verified: document masks, document counts and per-text occurrence counts against a brute-force
// scanner; the LCS of all texts, of every pair and of every single text against a brute-force
// enumeration of the shortest selected text, with the returned string checked to occur in every
// selected text; mask monotonicity and depth sanity; the empty pattern convention; degenerate sets
// (single text, identical texts, disjoint alphabets, an empty member) and random text sets.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

internal class GeneralizedSuffixTree
{
    private readonly string concat;
    private readonly int[] owner;             // owner text of each concat position, -1 for sentinels
    private readonly int[] edgeStart, edgeEnd, parent, depth, labelPos, leafOwner;
    private readonly ulong[] mask;            // distinct owner texts in a subtree, one bit per text
    private readonly List<Dictionary<char, int>> children;
    private int nodes;

    public GeneralizedSuffixTree(IReadOnlyList<string> texts)
    {
        TextCount = texts.Count;
        var sb = new StringBuilder();
        owner = new int[texts.Sum(t => t.Length) + texts.Count];
        for (int t = 0; t < texts.Count; t++)
        {
            foreach (char ch in texts[t])
            {
                owner[sb.Length] = t;
                sb.Append(ch);
            }
            owner[sb.Length] = -1;
            sb.Append((char)(t + 1));                 // unique sentinel of this text
        }
        concat = sb.ToString();
        int m = concat.Length;
        int capacity = 2 * m + 2;
        edgeStart = new int[capacity]; edgeEnd = new int[capacity]; parent = new int[capacity];
        depth = new int[capacity]; labelPos = new int[capacity]; leafOwner = new int[capacity];
        mask = new ulong[capacity];
        children = new List<Dictionary<char, int>>(capacity);
        for (int i = 0; i < capacity; i++) children.Add(new Dictionary<char, int>());
        nodes = 1;
        parent[0] = -1;
        leafOwner[0] = -1;

        // One insertion per suffix start, keeping the invariant "the label of node occurs at
        // pos - depth[node]" in the concatenation.
        for (int start = 0; start < m; start++)
        {
            if (owner[start] < 0) continue;           // a sentinel position starts no text suffix
            int node = 0, pos = start;
            while (true)
            {
                if (!children[node].TryGetValue(concat[pos], out int child))
                {
                    // Divergence at a node boundary: hang the rest of this suffix as a leaf. The
                    // owner is always the text of the suffix start - pos may already lie on a
                    // sentinel or inside the next text of the concatenation.
                    NewLeaf(node, pos, owner[start]);
                    break;
                }
                int edgeLen = edgeEnd[child] - edgeStart[child] + 1;
                int matched = 0;
                while (matched < edgeLen && pos + matched < m &&
                       concat[edgeStart[child] + matched] == concat[pos + matched]) matched++;
                if (matched == edgeLen)
                {
                    node = child;
                    pos += matched;
                    if (pos == m) throw new Exception("FAILED: a suffix is a prefix of another");
                    continue;
                }
                if (matched == m - pos) throw new Exception("FAILED: suffix exhausted inside an edge");
                int mid = nodes++;                    // divergence: split the edge after `matched`
                parent[mid] = node;
                depth[mid] = depth[node] + matched;
                labelPos[mid] = pos - depth[node];
                leafOwner[mid] = -1;
                edgeStart[mid] = edgeStart[child];
                edgeEnd[mid] = edgeStart[child] + matched - 1;
                children[node][concat[pos]] = mid;
                edgeStart[child] += matched;          // the old child keeps the tail of the edge
                parent[child] = mid;
                children[mid][concat[edgeStart[child]]] = child;
                // The new leaf belongs to the text that owns the suffix *start*: the split point
                // itself may already sit on that text's sentinel, where owner[] is -1.
                NewLeaf(mid, pos + matched, owner[start]);
                break;
            }
        }

        // Mask accumulation: every leaf pushes its owner bit up the parent chain, so a node's mask is
        // exactly the set of texts that contain its path label.
        for (int leaf = 1; leaf < nodes; leaf++)
        {
            if (leafOwner[leaf] < 0) continue;
            ulong bit = 1UL << leafOwner[leaf];
            for (int v = leaf; v != -1; v = parent[v]) mask[v] |= bit;   // parent[0] == -1
        }
    }

    public int TextCount { get; }
    public int NodeCount => nodes;
    private bool IsSentinel(char c) => c >= (char)1 && c <= (char)TextCount;

    // A leaf's path label is parent label plus edge label concat[pos..]: depth[p] + (m - pos) long,
    // occurring where the parent label started.
    private void NewLeaf(int p, int pos, int textIndex)
    {
        int leaf = nodes++;
        parent[leaf] = p;
        depth[leaf] = depth[p] + concat.Length - pos;
        labelPos[leaf] = pos - depth[p];
        edgeStart[leaf] = pos;
        edgeEnd[leaf] = concat.Length - 1;
        leafOwner[leaf] = textIndex;
        children[p][concat[pos]] = leaf;
    }

    // Pattern walk; the pattern ends either on a node boundary or inside an edge, and in the second
    // case the whole edge subtree is the occurrence set.
    private int Walk(string pattern)
    {
        int node = 0, offset = 0, edge = -1;
        foreach (char c in pattern)
        {
            if (offset == 0 && !children[node].TryGetValue(c, out edge)) return -1;
            if (concat[edgeStart[edge] + offset] != c) return -1;
            if (++offset == edgeEnd[edge] - edgeStart[edge] + 1)
            {
                node = edge;
                offset = 0;
            }
        }
        return offset == 0 ? node : edge;
    }

    public ulong DocumentsOf(string pattern)
    {
        int node = pattern.Length == 0 ? 0 : Walk(pattern);
        return node == -1 ? 0UL : mask[node];
    }

    public int DocumentCount(string pattern) => System.Numerics.BitOperations.PopCount(DocumentsOf(pattern));

    public int OccurrencesIn(string pattern, int textIndex)
    {
        int node = pattern.Length == 0 ? 0 : Walk(pattern);
        if (node == -1) return 0;
        int count = 0;
        void Visit(int v)
        {
            if (leafOwner[v] == textIndex) count++;
            foreach (var kv in children[v]) Visit(kv.Value);
        }
        Visit(node);
        return count;
    }

    // Longest sentinel-free label whose subtree covers every requested text. A node's path label is
    // a prefix of every leaf label below it, so cutting it at the first sentinel yields a common
    // substring of exactly the texts in mask[node]; for a leaf of one text that prefix is the
    // remaining suffix of that text, which is how a single-text query returns the whole text.
    private string LongestCoveringLabel(ulong wanted)
    {
        string best = "";
        for (int v = 1; v < nodes; v++)
        {
            if ((mask[v] & wanted) != wanted) continue;
            string label = concat.Substring(labelPos[v], depth[v]);
            int cut = label.Length;
            for (int i = 0; i < label.Length && cut == label.Length; i++) if (IsSentinel(label[i])) cut = i;
            if (cut > best.Length) best = label.Substring(0, cut);
        }
        return best;
    }

    public string LongestCommonSubstringOfAll()
    {
        ulong wanted = TextCount >= 64 ? ulong.MaxValue : (1UL << TextCount) - 1;
        return LongestCoveringLabel(wanted);
    }

    public string LongestCommonSubstringOf(IEnumerable<int> textIndexes)
    {
        ulong wanted = 0;
        foreach (int t in textIndexes) wanted |= 1UL << t;
        return LongestCoveringLabel(wanted);
    }

    public bool MasksAreMonotone()
    {
        for (int v = 1; v < nodes; v++) if ((mask[v] & ~mask[parent[v]]) != 0) return false;
        return true;
    }}

public static class Solution
{
    private static ulong seed = 0x6A09E667F3BCC909UL;

    private static ulong NextRandom()
    {
        seed += 0x9E3779B97F4A7C15UL;
        ulong z = seed;
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

    private static int BruteOccurrencesIn(string text, string pattern)
    {
        int count = 0;
        for (int i = 0; i + pattern.Length <= text.Length; i++)
        {
            if (string.CompareOrdinal(text, i, pattern, 0, pattern.Length) == 0) count++;
        }
        return count;
    }

    // Longest substring of the shortest selected text that occurs in every selected text. Ties do
    // not matter because only the length is compared with the tree's answer.
    private static string BruteLcs(IReadOnlyList<string> texts, IReadOnlyList<int> chosen)
    {
        string probe = chosen.Select(t => texts[t]).OrderBy(t => t.Length).First();
        for (int len = probe.Length; len > 0; len--)
            for (int start = 0; start + len <= probe.Length; start++)
                if (chosen.All(t => BruteOccurrencesIn(texts[t], probe.Substring(start, len)) > 0))
                    return probe.Substring(start, len);
        return "";
    }

    // The tree's LCS must be as long as the brute-force one, and the string itself must occur in
    // every selected text - which also rules out labels carrying a sentinel.
    private static void CheckLcs(GeneralizedSuffixTree tree, string[] texts, int[] chosen, string label)
    {
        string expected = BruteLcs(texts, chosen);
        string actual = tree.LongestCommonSubstringOf(chosen);
        Check(actual.Length == expected.Length, label + ": got \"" + actual + "\", brute force length " + expected.Length);
        foreach (int t in chosen) Check(Occurs(texts[t], actual), label + " membership of \"" + actual + "\" in " + t);
    }

    private static bool Occurs(string text, string pattern) => pattern.Length == 0 || BruteOccurrencesIn(text, pattern) > 0;

    // Full battery for one text set: LCS of all texts, every pair and every single text, plus
    // document masks, document counts and per-text occurrence counts for short substrings.
    private static void Battery(string[] texts)
    {
        var tree = new GeneralizedSuffixTree(texts);
        CheckLcs(tree, texts, Enumerable.Range(0, texts.Length).ToArray(), "lcs all " + string.Join("|", texts));
        Check(tree.MasksAreMonotone(), "masks monotone " + string.Join("|", texts));
        for (int t = 0; t < texts.Length; t++)
        {
            if (texts[t].Length > 0) Check(tree.OccurrencesIn(texts[t], t) == 1, "whole text once");
            for (int start = 0; start < texts[t].Length; start++)
            {
                for (int end = start + 1; end <= Math.Min(start + 5, texts[t].Length); end++)
                {
                    string p = texts[t].Substring(start, end - start);
                    ulong expected = 0;
                    for (int other = 0; other < texts.Length; other++) if (Occurs(texts[other], p)) expected |= 1UL << other;
                    Check(tree.OccurrencesIn(p, t) == BruteOccurrencesIn(texts[t], p), "occ " + p + " in " + t);
                    Check(tree.DocumentsOf(p) == expected && tree.DocumentCount(p) == System.Numerics.BitOperations.PopCount(expected),
                        "docs " + p);
                }
            }
        }
        for (int a = 0; a < texts.Length; a++)
        {
            CheckLcs(tree, texts, new[] { a }, "lcs " + a);
            for (int b = a + 1; b < texts.Length; b++) CheckLcs(tree, texts, new[] { a, b }, "lcs " + a + "," + b);
        }
    }

    public static void Main()
    {
        var canonical = new[] { "banana", "ananas", "bandana" };
        var tree = new GeneralizedSuffixTree(canonical);
        Check(tree.DocumentsOf("ana") == 0b111UL && tree.DocumentCount("ana") == 3, "ana in all three");
        Check(tree.DocumentCount("nan") == 2 && tree.DocumentCount("band") == 1, "nan/band documents");
        Check(tree.OccurrencesIn("an", 2) == 2 && tree.OccurrencesIn("zz", 0) == 0 && tree.DocumentsOf("zz") == 0UL,
            "an counts, absent pattern");
        Check(tree.LongestCommonSubstringOfAll() == "ana", "canonical lcs");
        Check(tree.OccurrencesIn("", 1) == 6 && tree.DocumentsOf("") == 0b111UL, "empty pattern convention");
        Check(tree.MasksAreMonotone() && tree.NodeCount <= 2 * (6 + 6 + 7 + 3), "canonical structure");

        var fixedSets = new[]
        {
            new[] { "single" }, new[] { "aaaa" }, new[] { "abc", "abc" }, new[] { "aaaa", "aaaa", "aaaa" },
            new[] { "aaa", "bbb" }, new[] { "abc", "", "abd" }, new[] { "a", "ab", "abc", "abcd" },
            new[] { "abracadabra", "cadabra", "abra", "dabra" }, new[] { "mississippi", "sippis", "pizza" },
            new[] { "abab", "baba" }, new[] { "xyzxyz", "yzxyzx", "zxyzxy" },
        };
        foreach (string[] texts in fixedSets) Battery(texts);

        var withEmpty = new GeneralizedSuffixTree(new[] { "abc", "", "abd" });
        var disjoint = new GeneralizedSuffixTree(new[] { "aaa", "bbb" });
        Check(withEmpty.LongestCommonSubstringOfAll() == "" && withEmpty.DocumentCount("ab") == 2 &&
            withEmpty.DocumentsOf("") == 0b101UL && withEmpty.OccurrencesIn("", 1) == 0, "empty member");
        Check(withEmpty.LongestCommonSubstringOf(new[] { 0, 2 }) == "ab" &&
            disjoint.LongestCommonSubstringOfAll() == "" && disjoint.DocumentsOf("a") == 1UL, "empty pair, disjoint");

        foreach (string alphabet in new[] { "ab", "abc", "abcde" })
        {
            for (int trial = 0; trial < 6; trial++) Battery(Enumerable.Range(0, 2 + (int)(NextRandom() % 4))
                .Select(_ => RandomText(10 + (int)(NextRandom() % 40), alphabet)).ToArray());
        }

        var many = new GeneralizedSuffixTree(Enumerable.Range(0, 6).Select(i => "abcabc" + (char)('a' + i)).ToArray());
        Check(many.DocumentCount("abcabc") == 6 && many.DocumentsOf("bcab").Equals(63UL), "six documents");

        Console.WriteLine("3-text tree: " + tree.NodeCount + " nodes, LCS \"" + tree.LongestCommonSubstringOfAll() +
            "\"; 6-text tree: " + many.NodeCount + " nodes, mask " + many.DocumentsOf("abcabc"));
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
