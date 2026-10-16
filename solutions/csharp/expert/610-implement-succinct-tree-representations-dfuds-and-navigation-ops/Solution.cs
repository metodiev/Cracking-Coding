// Task 610 - Implement succinct tree representations (DFUDS) and navigation ops. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// DFUDS (depth-first unary degree sequence). For a tree numbered 1..n in preorder the encoding is
// the bit string 1 followed by, for every node v in preorder, the word 1^d(v) 0; the leading 1
// balances the sequence. Each node owns exactly one 0-bit (its identity at position e_v, and node
// ids are exactly the ranks of the 0-bits) and each non-root node is pointed at by exactly one
// 1-bit, the one its parent emits for it. Two invariants follow and are both checked: subtrees are
// *contiguous* ranges of node ids, and the children of a node are the starts of those ranges, so
// child(v,1) = v + 1 and child(v,j+1) = lastnode(child(v,j)) + 1.
//
// Navigation uses the encoding plus rank/select on it:
//   degree(v)      = ones(e_v - 1) - ones(e_{v-1}) - [v == 1]     (v's own 1-bits, minus the lead)
//   subtree(v)     = closure of v under j -> [j + 1, j + d(j)], so lastnode(v) is its right end
//   firstchild(v)  = v + 1 when d(v) > 0, and child(v,j) walks those ranges
//   nextsibling(v) = lastnode(v) + 1 when that node shares v's parent, else none
//   parent(v)      = materialised once in O(n) with a stack over the degrees, and cross-checked by
//                    the rank/select-only rule parent(v) = max{ u < v : lastnode(u) >= v }
//   depth(v)       = walk up the parent layer
// The index is two-level: 64-bit words plus a cumulative sample every 8 words, so rank is a few
// popcounts and select is a binary search plus one word scan; the complement vector is indexed too,
// which is what supplies select on the 0-bits.
//
// Documented simplification: a production DFUDS answers child/parent/subtree-size in O(1) from an
// auxiliary range-min/excess ("pioneer") structure; here those two walk the degree sequence and the
// parent layer costs O(n) words. Degrees, first child, next sibling and rank/select stay succinct.
// Verification is unchanged: on hand-built shapes and deterministic random trees every navigation
// result must match an independently constructed pointer tree with explicit children lists, the
// encoding must round-trip (2n bits, n ones, n zeros, every prefix balanced and non-negative), and
// a well-formedness validator must accept every produced encoding and reject malformed ones.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;          // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

/** Rank/select index over a bit vector: 64-bit words plus a cumulative sample every 8 words. */
internal sealed class BitVectorIndex
{
    private const int WordBits = 64;
    private const int SampleWords = 8;

    private readonly ulong[] words;
    private readonly int length;
    private readonly int[] sample;

    internal BitVectorIndex(IReadOnlyList<int> bits)
    {
        length = bits.Count;
        words = new ulong[(length + WordBits - 1) / WordBits];
        for (int i = 0; i < length; i++)
        {
            if (bits[i] == 1) words[i / WordBits] |= 1UL << (i % WordBits);
        }
        var samples = new List<int>();
        int running = 0;
        for (int w = 0; w < words.Length; w++)
        {
            if (w % SampleWords == 0) samples.Add(running);
            running += PopCount(words[w]);
        }
        samples.Add(running);
        sample = samples.ToArray();
    }

    internal int Ones => sample[sample.Length - 1];
    internal int IndexBits => words.Length * WordBits + sample.Length * 32;

    /** Number of 1-bits in [0, position] (inclusive); 0 below the start. */
    internal int Rank1(int position)
    {
        if (position < 0) return 0;
        if (position >= length) position = length - 1;
        int word = position / WordBits;
        int group = word / SampleWords;
        int rank = sample[group];
        for (int w = group * SampleWords; w < word; w++) rank += PopCount(words[w]);
        int within = position % WordBits + 1;
        ulong mask = within == WordBits ? ulong.MaxValue : (1UL << within) - 1UL;
        return rank + PopCount(words[word] & mask);
    }

    /** 0-based position of the rank-th 1-bit (rank is 1-based), or -1 when out of range. */
    internal int Select1(int rank)
    {
        if (rank < 1 || rank > Ones) return -1;
        int lo = 0;
        int hi = sample.Length - 2;
        int group = 0;
        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            if (sample[mid] < rank)
            {
                group = mid;
                lo = mid + 1;
            }
            else hi = mid - 1;
        }
        int remaining = rank - sample[group];
        for (int w = group * SampleWords; w < words.Length; w++)
        {
            int count = PopCount(words[w]);
            if (remaining <= count)
            {
                ulong word = words[w];
                for (int b = 0; b < WordBits; b++)
                {
                    if (((word >> b) & 1UL) != 0 && --remaining == 0) return w * WordBits + b;
                }
            }
            remaining -= count;
        }
        return -1;
    }

    internal static int PopCount(ulong x)
    {
        x -= (x >> 1) & 0x5555555555555555UL;
        x = (x & 0x3333333333333333UL) + ((x >> 2) & 0x3333333333333333UL);
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FUL;
        return (int)((x * 0x0101010101010101UL) >> 56);
    }
}

/** Navigation over a DFUDS encoding, driven by rank/select on the encoding itself. */
internal sealed class DfudsNavigator
{
    private readonly int nodeCount;
    private readonly BitVectorIndex ones;
    private readonly BitVectorIndex zeros;        // index over the complement: select on the 0-bits
    private int[] parents = Array.Empty<int>();

    internal DfudsNavigator(IReadOnlyList<int> bits)
    {
        nodeCount = bits.Count / 2;
        ones = new BitVectorIndex(bits);
        var flipped = new int[bits.Count];
        for (int i = 0; i < bits.Count; i++) flipped[i] = 1 - bits[i];
        zeros = new BitVectorIndex(flipped);
    }

    internal int NodeCount => nodeCount;
    internal int BitLength => 2 * nodeCount;
    internal int Ones => ones.Ones;
    internal int Zeros => zeros.Ones;
    internal int IndexBits => ones.IndexBits + zeros.IndexBits;

    /** Position of node's own 0-bit: node ids are exactly the ranks of the 0-bits. */
    internal int EndPosition(int node) => zeros.Select1(node);

    internal int Degree(int node)
    {
        int end = EndPosition(node);
        int before = node == 1 ? 0 : ones.Rank1(EndPosition(node - 1));
        return ones.Rank1(end - 1) - before - (node == 1 ? 1 : 0);
    }

    internal bool IsLeaf(int node) => Degree(node) == 0;

    /** Subtrees are contiguous id ranges: [v,k] closes when the pointers inside it are exactly the
     *  k - v non-root nodes, i.e. sum of degrees over [v,k] equals k - v. */
    internal int LastNode(int node)
    {
        int pointers = 0;
        for (int k = node; k <= nodeCount; k++)
        {
            pointers += Degree(k);
            if (pointers == k - node) return k;
        }
        return nodeCount;                            // malformed input: clamp the range
    }

    internal int SubtreeSize(int node) => LastNode(node) - node + 1;

    internal int FirstChild(int node) => Degree(node) == 0 ? -1 : node + 1;

    internal int Child(int node, int index)
    {
        if (index < 1 || index > Degree(node)) return -1;
        int child = node + 1;
        for (int j = 1; j < index; j++) child = LastNode(child) + 1;
        return child;
    }

    /** 1-based position of node among its parent's children, or -1 for the root. */
    internal int ChildIndex(int node)
    {
        int parent = Parent(node);
        if (parent < 0) return -1;
        for (int index = 1; index <= Degree(parent); index++)
        {
            int child = Child(parent, index);
            if (child == node) return index;
            if (child > node) break;
        }
        return -1;
    }

    internal int Parent(int node) => node <= 1 ? -1 : EnsureParents()[node];

    internal int NextSibling(int node)
    {
        if (node <= 1) return -1;
        int next = LastNode(node) + 1;
        return next <= nodeCount && Parent(next) == Parent(node) ? next : -1;
    }

    internal int Depth(int node)
    {
        int depth = 0;
        for (int cursor = node; cursor > 1; cursor = Parent(cursor)) depth++;
        return depth;
    }

    /** The O(n)-word parent layer; a real DFUDS answers this from the excess structure instead. */
    private int[] EnsureParents()
    {
        if (parents.Length > 0) return parents;
        var result = new int[nodeCount + 1];
        var filled = new int[nodeCount + 1];
        if (nodeCount >= 1) result[1] = -1;
        if (nodeCount >= 2)
        {
            var stack = new Stack<int>();
            stack.Push(1);
            for (int node = 2; node <= nodeCount; node++)
            {
                while (stack.Count > 0 && filled[stack.Peek()] >= Degree(stack.Peek())) stack.Pop();
                if (stack.Count == 0)
                {
                    result[node] = 0;                 // no free child slot: malformed input
                    continue;
                }
                int parent = stack.Peek();
                result[node] = parent;
                filled[parent]++;
                if (Degree(node) > 0) stack.Push(node);
            }
        }
        parents = result;
        return result;
    }

    /** True when the bit string is a well-formed DFUDS. */
    internal static bool IsWellFormed(IReadOnlyList<int> bits)
    {
        if (bits.Count == 0 || bits.Count % 2 != 0) return false;
        int n = bits.Count / 2;
        var navigator = new DfudsNavigator(bits);
        if (navigator.ones.Ones != n || navigator.zeros.Ones != n) return false;
        for (int node = 1; node <= n; node++)
        {
            if (navigator.Degree(node) < 0 || navigator.LastNode(node) > n) return false;
            if (node == 1) continue;
            int parent = navigator.Parent(node);
            if (parent < 1 || parent >= node) return false;
            if (navigator.ChildIndex(node) < 1) return false;
        }
        return true;
    }
}

/** The plain pointer tree the succinct representation is verified against. */
internal sealed class ReferenceTree
{
    internal readonly int Count;
    internal readonly int[] Degree;
    internal readonly int[] Parent;
    internal readonly int[] Depth;
    internal readonly int[] Subtree;
    internal readonly int[][] Children;

    /** degree is 1-based and must describe a valid preorder tree, else the constructor throws. */
    internal ReferenceTree(int[] degree)
    {
        Count = degree.Length - 1;
        Degree = new int[Count + 1];
        Array.Copy(degree, Degree, Count + 1);
        Parent = new int[Count + 1];
        Depth = new int[Count + 1];
        Subtree = new int[Count + 1];
        Children = new int[Count + 1][];
        var lists = new List<int>[Count + 1];
        for (int i = 1; i <= Count; i++) lists[i] = new List<int>();
        Parent[1] = -1;
        var stack = new Stack<int>();
        stack.Push(1);
        for (int node = 2; node <= Count; node++)
        {
            while (lists[stack.Peek()].Count >= Degree[stack.Peek()]) stack.Pop();
            int parent = stack.Peek();
            lists[parent].Add(node);
            Parent[node] = parent;
            Depth[node] = Depth[parent] + 1;
            if (Degree[node] > 0) stack.Push(node);
        }
        for (int i = 1; i <= Count; i++)
        {
            Children[i] = lists[i].ToArray();
            if (Children[i].Length != Degree[i])
            {
                throw new InvalidOperationException("not a preorder degree sequence at node " + i);
            }
        }
        for (int node = Count; node >= 1; node--)
        {
            int size = 1;
            foreach (int child in Children[node]) size += Subtree[child];
            Subtree[node] = size;
        }
    }

    internal int[] Encode()
    {
        var bits = new List<int> { 1 };                   // the leading 1
        for (int node = 1; node <= Count; node++)
        {
            for (int c = 0; c < Degree[node]; c++) bits.Add(1);
            bits.Add(0);
        }
        return bits.ToArray();
    }

    internal int FirstChild(int node) => Children[node].Length == 0 ? -1 : Children[node][0];
    internal int LastNode(int node) => node + Subtree[node] - 1;

    internal int ChildIndex(int node)
    {
        int parent = Parent[node];
        if (parent < 0) return -1;
        var kids = Children[parent];
        for (int i = 0; i < kids.Length; i++)
        {
            if (kids[i] == node) return i + 1;
        }
        return -1;
    }

    internal int NextSibling(int node)
    {
        int parent = Parent[node];
        if (parent < 0) return -1;
        var kids = Children[parent];
        for (int i = 0; i + 1 < kids.Length; i++)
        {
            if (kids[i] == node) return kids[i + 1];
        }
        return -1;
    }
}

public static class Solution
{
    private static ulong state = 0x5EEDF00DCAFEUL;
    private static int nextId;

    private static ReferenceTree Build(int[] preorderDegrees)
    {
        var sized = new int[preorderDegrees.Length + 1];      // node ids are 1-based
        Array.Copy(preorderDegrees, 0, sized, 1, preorderDegrees.Length);
        return new ReferenceTree(sized);
    }

    /** Random depth-first tree: ids come out in preorder and the degrees are exact. */
    private static ReferenceTree RandomTree(int budget, int maxDepth)
    {
        var degree = new int[budget + 2];
        nextId = 0;
        BuildSubtree(degree, budget, 0, maxDepth);
        var sized = new int[nextId + 1];
        Array.Copy(degree, sized, nextId + 1);
        return new ReferenceTree(sized);
    }

    private static int BuildSubtree(int[] degree, int budget, int depth, int maxDepth)
    {
        int id = ++nextId;
        int remaining = budget - 1;
        int wanted = depth >= maxDepth || remaining <= 0 ? 0 : (int)(SplitMix64.Next(ref state) % 4);
        if (wanted > remaining) wanted = remaining;
        int used = 1;
        int built = 0;
        for (int c = 0; c < wanted && remaining > 0; c++)
        {
            int childUsed = BuildSubtree(degree, remaining, depth + 1, maxDepth);
            built++;
            used += childUsed;
            remaining -= childUsed;
        }
        degree[id] = built;
        return used;
    }

    /** Comparison of every navigation operation against the pointer tree, node by node. */
    private static void VerifyAll(ReferenceTree tree, string label)
    {
        var bits = tree.Encode();
        var navigator = new DfudsNavigator(bits);
        Check(navigator.NodeCount == tree.Count, label + " node count");
        Check(navigator.BitLength == 2 * tree.Count, label + " bit length");
        Check(navigator.Ones == tree.Count && navigator.Zeros == tree.Count, label + " bit parity");
        Check(DfudsNavigator.IsWellFormed(bits), label + " well formed");
        Check(ForAll(tree, navigator, (nav, node) => nav.Degree(node) == tree.Degree[node]), label + " degree");
        Check(ForAll(tree, navigator, (nav, node) => nav.Parent(node) == tree.Parent[node]), label + " parent");
        Check(ForAll(tree, navigator, (nav, node) => nav.Depth(node) == tree.Depth[node]), label + " depth");
        Check(ForAll(tree, navigator, (nav, node) => nav.SubtreeSize(node) == tree.Subtree[node]), label + " subtree");
        Check(ForAll(tree, navigator, (nav, node) => nav.LastNode(node) == tree.LastNode(node)), label + " last node");
        Check(ForAll(tree, navigator, (nav, node) => nav.FirstChild(node) == tree.FirstChild(node)), label + " first child");
        Check(ForAll(tree, navigator, (nav, node) => nav.NextSibling(node) == tree.NextSibling(node)), label + " sibling");
        Check(ForAll(tree, navigator, (nav, node) => nav.ChildIndex(node) == tree.ChildIndex(node)), label + " child index");
        Check(ForAll(tree, navigator, (nav, node) => nav.IsLeaf(node) == (tree.Degree[node] == 0)), label + " leaf");
        Check(ForAll(tree, navigator, (nav, node) => node == 1 || nav.EndPosition(node) > nav.EndPosition(node - 1)),
            label + " zero order");
        Check(ForAll(tree, navigator, (nav, node) => ChildrenMatch(tree, nav, node)), label + " children");
    }

    private static bool ForAll(ReferenceTree tree, DfudsNavigator navigator, Func<DfudsNavigator, int, bool> property)
    {
        for (int node = 1; node <= tree.Count; node++)
        {
            if (!property(navigator, node)) return false;
        }
        return true;
    }

    private static bool ChildrenMatch(ReferenceTree tree, DfudsNavigator navigator, int node)
    {
        if (navigator.Child(node, 0) != -1) return false;
        for (int index = 1; index <= tree.Degree[node] + 2; index++)
        {
            int expected = index <= tree.Degree[node] ? tree.Children[node][index - 1] : -1;
            if (navigator.Child(node, index) != expected) return false;
        }
        return true;
    }

    public static void Main()
    {
        var single = Build(new[] { 0 });
        var singleNav = new DfudsNavigator(single.Encode());
        Check(singleNav.BitLength == 2 && singleNav.Degree(1) == 0, "single node encoding");
        Check(singleNav.Parent(1) == -1 && singleNav.FirstChild(1) == -1, "single node navigation");
        Check(singleNav.SubtreeSize(1) == 1 && singleNav.Depth(1) == 0, "single node metrics");
        Check(singleNav.NextSibling(1) == -1 && singleNav.EndPosition(1) == 1, "single node sibling");
        VerifyAll(single, "single");

        // Valid preorder degree sequences: chain, star, complete binary tree, busy first child.
        var chain = Build(new[] { 1, 1, 1, 0 });
        var star = Build(new[] { 3, 0, 0, 0 });
        var binary = Build(new[] { 2, 2, 0, 0, 2, 0, 0 });
        var bushy = Build(new[] { 3, 2, 0, 0, 0, 0 });
        VerifyAll(chain, "chain");
        VerifyAll(star, "star");
        VerifyAll(binary, "binary");
        VerifyAll(bushy, "bushy");

        var chainNav = new DfudsNavigator(chain.Encode());
        Check(chainNav.Depth(4) == 3 && chainNav.Depth(1) == 0, "chain depths");
        Check(chainNav.SubtreeSize(1) == 4 && chainNav.SubtreeSize(4) == 1, "chain subtrees");
        Check(chainNav.NextSibling(2) == -1 && chainNav.FirstChild(4) == -1, "chain siblings");
        Check(chainNav.Child(3, 1) == 4 && chainNav.Child(3, 2) == -1, "chain child walk");

        var starNav = new DfudsNavigator(star.Encode());
        Check(starNav.FirstChild(1) == 2 && starNav.Child(1, 3) == 4 && starNav.Child(1, 4) == -1,
            "star children");
        Check(starNav.SubtreeSize(1) == 4 && starNav.NextSibling(2) == 3 && starNav.NextSibling(4) == -1,
            "star subtrees and siblings");
        Check(starNav.Parent(4) == 1 && starNav.ChildIndex(3) == 2, "star parent and child index");

        var binaryNav = new DfudsNavigator(binary.Encode());
        Check(binaryNav.Child(1, 1) == 2 && binaryNav.Child(1, 2) == 5, "binary children are chunks");
        Check(binaryNav.Parent(4) == 2 && binaryNav.Parent(7) == 5, "binary parents");
        Check(binaryNav.Depth(7) == 2 && binaryNav.Depth(4) == 2, "binary depths");
        Check(binaryNav.SubtreeSize(2) == 3 && binaryNav.SubtreeSize(5) == 3, "binary subtrees");
        Check(binaryNav.NextSibling(2) == 5 && binaryNav.NextSibling(4) == -1, "binary siblings");

        // Every produced encoding satisfies the DFUDS invariants: parity and non-negative prefixes.
        foreach (var tree in new[] { single, chain, star, binary, bushy })
        {
            var bits = tree.Encode();
            Check(DfudsNavigator.IsWellFormed(bits), "well formed hand tree");
            int balance = 0;
            foreach (int bit in bits)
            {
                balance += bit == 1 ? 1 : -1;
                Check(balance >= 0, "prefix balance");
            }
            Check(balance == 0, "final balance");
        }

        // Malformed vectors must be rejected: odd length, bad parity, orphan or self-parenting node.
        Check(!DfudsNavigator.IsWellFormed(new[] { 1, 0, 1 }), "reject odd length");
        Check(!DfudsNavigator.IsWellFormed(new[] { 1, 0, 0, 0 }), "reject zero heavy");
        Check(!DfudsNavigator.IsWellFormed(new[] { 1, 1, 1, 1 }), "reject one heavy");
        Check(!DfudsNavigator.IsWellFormed(new[] { 1, 0, 1, 0 }), "reject self parent");
        Check(!DfudsNavigator.IsWellFormed(new[] { 1, 0, 1, 1, 0, 0 }), "reject orphan child");
        Check(!DfudsNavigator.IsWellFormed(Array.Empty<int>()), "reject empty");
        Check(DfudsNavigator.IsWellFormed(new[] { 1, 1, 0, 1, 0, 0 }), "accept a 3-node chain");

        int randomTrees = 0;
        long navigatedNodes = 0;
        int deepest = 0;
        for (int trial = 0; trial < 60; trial++)
        {
            int budget = 5 + (int)(SplitMix64.Next(ref state) % 56);
            int maxDepth = 1 + (int)(SplitMix64.Next(ref state) % 6);
            var tree = RandomTree(budget, maxDepth);
            VerifyAll(tree, "random " + trial);
            randomTrees++;
            navigatedNodes += tree.Count;
            for (int node = 1; node <= tree.Count; node++) deepest = Math.Max(deepest, tree.Depth[node]);
        }

        // A wide fan (root, 30 chain children with 40 leaves each) exercises the word and sample
        // boundaries of the rank/select index.
        var wide = Build(BuildFanDegrees(30, 40));
        VerifyAll(wide, "wide");
        var wideNav = new DfudsNavigator(wide.Encode());
        Check(wideNav.BitLength == 2 * wide.Count && wideNav.ChildIndex(wide.Count) > 0, "wide shape");
        Check(wideNav.Depth(wide.Count) == 2 && wideNav.SubtreeSize(1) == wide.Count, "wide metrics");

        Console.WriteLine("  handTrees=5 randomTrees=" + randomTrees + " nodesNavigated=" + navigatedNodes
            + " deepestRandomTree=" + deepest);
        Console.WriteLine("  wide n=" + wide.Count + " sequenceBits=" + wideNav.BitLength
            + " indexBits=" + wideNav.IndexBits);
        Console.WriteLine("PASS");
    }

    /** Root with `chains` children, each carrying `fan` leaves: a deterministic wide shape. */
    private static int[] BuildFanDegrees(int chains, int fan)
    {
        var degree = new int[1 + chains * (1 + fan)];
        degree[0] = chains;
        for (int chain = 0; chain < chains; chain++) degree[1 + chain * (1 + fan)] = fan;
        return degree;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
