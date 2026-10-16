// Task 611 - Implement dynamic succinct trees (conceptual). (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A *dynamic* balanced-parentheses (BP) tree. The tree is the bit string BP in which each node
// contributes "()" (here 1 = '(', 0 = ')') in depth-first order, so the pairs of BP are exactly the
// nodes in DFS preorder and every structural query is rank/select on that string:
//   subtreeSize(i) = (matchClose(i) - openPos(i) + 1) / 2      depth(i) = ones(openPos) - zeros(openPos)
//   firstChild(i)  = pair after openPos(i)                     nextSibling(i) = pair after closePos(i)
//   parent(i)      = enclosing pair of openPos(i)              child(i,k)     = sibling walk
// Mutations preserve the invariant by editing the string in place: link(parent) inserts one "()"
// pair immediately before the parent's closing paren (the new node becomes the parent's last child)
// and cut(v) deletes the whole BP substring of v's subtree. The dynamic bit vector is a list of
// 64-bit chunks that split above 63 bits and merge below 16, giving rank/select in O(#chunks).
//
// Documented simplifications (this is the "conceptual" variant): the block list is a plain list
// instead of a weight-balanced/AVL tree of blocks, so updates are O(#chunks) and not O(log n);
// parenthesis matching and the parent walk scan the string instead of using a range-min/excess
// structure; and a node handle is its DFS index, because a real dynamic BP tree also carries a
// node-to-position map (here the reference model re-derives the map after every mutation).
//
// Invariants verified after *every* mutation, against an independent pointer tree with stable ids:
// BP stays balanced (n pairs, non-negative prefix excess, zero total), the pair count equals the
// reference node count and the DFS order, and for every reference node the DFS index, parent,
// depth, subtree size, degree, first child, next sibling and full child list read from BP agree
// with the reference tree. The chunked bit vector is checked separately against a plain list
// oracle for insert, delete, access, rank and select.

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

/** Chunked dynamic bit vector: 64-bit blocks, split above 63 bits, merged below 16. */
internal sealed class DynamicBitVector
{
    private const int MaxChunk = 63;
    private const int MinChunk = 16;

    private readonly List<ulong> blocks = new List<ulong>();
    private readonly List<int> counts = new List<int>();

    internal DynamicBitVector(IReadOnlyList<int> bits)
    {
        for (int start = 0; start < bits.Count; start += MaxChunk)
        {
            ulong word = 0;
            int used = 0;
            for (int b = start; b < Math.Min(bits.Count, start + MaxChunk); b++)
            {
                if (bits[b] == 1) word |= 1UL << used;
                used++;
            }
            blocks.Add(word);
            counts.Add(used);
        }
    }

    internal int Length
    {
        get
        {
            int sum = 0;
            foreach (int used in counts) sum += used;
            return sum;
        }
    }

    internal int Ones
    {
        get
        {
            int sum = 0;
            foreach (ulong word in blocks) sum += DynamicBitVector.PopCount(word);
            return sum;
        }
    }

    internal int BlockCount => blocks.Count;

    internal int BitAt(int index)
    {
        int block = Locate(index, out int offset);
        return (int)((blocks[block] >> offset) & 1UL);
    }

    internal void Insert(int index, int bit)
    {
        if (blocks.Count == 0)
        {
            if (index != 0) throw new ArgumentOutOfRangeException(nameof(index));
            blocks.Add(bit == 1 ? 1UL : 0UL);
            counts.Add(1);
            return;
        }
        int block = Locate(index, out int offset);
        ulong word = blocks[block];
        ulong low = offset == 0 ? 0UL : word & ((1UL << offset) - 1UL);
        ulong high = word >> offset;
        blocks[block] = low | ((bit == 1 ? 1UL : 0UL) << offset) | (high << (offset + 1));
        counts[block]++;
        Split(block);
    }

    internal void Delete(int index)
    {
        int block = Locate(index, out int offset);
        ulong word = blocks[block];
        ulong low = offset == 0 ? 0UL : word & ((1UL << offset) - 1UL);
        blocks[block] = low | ((word >> (offset + 1)) << offset);
        counts[block]--;
        if (counts[block] < MinChunk && blocks.Count > 1) Merge(block);
        if (counts[block] == 0 && blocks.Count > 1)
        {
            blocks.RemoveAt(block);
            counts.RemoveAt(block);
        }
    }

    /** Number of 1-bits in [0, index). */
    internal int Rank1(int index)
    {
        int seen = 0;
        int position = 0;
        int block = 0;
        while (block < blocks.Count && position + counts[block] <= index)
        {
            seen += DynamicBitVector.PopCount(blocks[block]);
            position += counts[block];
            block++;
        }
        if (block < blocks.Count && position < index)
        {
            seen += DynamicBitVector.PopCount(blocks[block] & ((1UL << (index - position)) - 1UL));
        }
        return seen;
    }

    internal int Rank0(int index) => index - Rank1(index);

    internal int Select1(int rank) => Select(rank, true);

    internal int Select0(int rank) => Select(rank, false);

    private int Select(int rank, bool one)
    {
        if (rank < 1) return -1;
        int remaining = rank;
        int position = 0;
        for (int block = 0; block < blocks.Count; block++)
        {
            int used = counts[block];
            int count = one ? DynamicBitVector.PopCount(blocks[block]) : used - DynamicBitVector.PopCount(blocks[block]);
            if (remaining <= count)
            {
                ulong word = one ? blocks[block] : ~blocks[block] & ((1UL << used) - 1UL);
                for (int i = 0; i < used; i++)
                {
                    if (((word >> i) & 1UL) != 0 && --remaining == 0) return position + i;
                }
            }
            remaining -= count;
            position += used;
        }
        return -1;
    }

    /** Block holding `index` plus the offset inside it. */
    private int Locate(int index, out int offset)
    {
        if (index < 0 || index > Length) throw new ArgumentOutOfRangeException(nameof(index));
        if (blocks.Count == 0) throw new InvalidOperationException("the vector has no blocks");
        int position = 0;
        for (int block = 0; block < blocks.Count; block++)
        {
            int end = position + counts[block];
            if (index < end || block + 1 == blocks.Count)
            {
                offset = index - position;
                return block;
            }
            position = end;
        }
        offset = 0;
        return 0;                                        // unreachable
    }

    private void Split(int block)
    {
        if (counts[block] <= MaxChunk) return;
        int used = counts[block];
        int half = used / 2;
        ulong word = blocks[block];
        ulong lower = 0;
        ulong upper = 0;
        for (int i = 0; i < half; i++)
        {
            if (((word >> i) & 1UL) != 0) lower |= 1UL << i;
        }
        for (int i = half; i < used; i++)
        {
            if (((word >> i) & 1UL) != 0) upper |= 1UL << (i - half);
        }
        blocks[block] = lower;
        counts[block] = half;
        blocks.Insert(block + 1, upper);
        counts.Insert(block + 1, used - half);
    }

    private void Merge(int block)
    {
        int other = block + 1 < blocks.Count ? block + 1 : block - 1;
        int left = Math.Min(block, other);
        int right = Math.Max(block, other);
        if (counts[left] + counts[right] > MaxChunk) return;
        blocks[left] |= blocks[right] << counts[left];
        counts[left] += counts[right];
        blocks.RemoveAt(right);
        counts.RemoveAt(right);
    }

    internal static int PopCount(ulong x)
    {
        x -= (x >> 1) & 0x5555555555555555UL;
        x = (x & 0x3333333333333333UL) + ((x >> 2) & 0x3333333333333333UL);
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FUL;
        return (int)((x * 0x0101010101010101UL) >> 56);
    }
}

/** Dynamic BP tree: the tree *is* the bit vector; node handles are DFS indices. */
internal sealed class DynamicBpTree
{
    private readonly DynamicBitVector bits;

    internal DynamicBpTree()
    {
        bits = new DynamicBitVector(new[] { 1, 0 });          // a single root node
    }

    internal int Count => bits.Length / 2;
    internal int BitLength => bits.Length;
    internal int BlockCount => bits.BlockCount;
    internal bool Balanced()
    {
        int excess = 0;
        for (int i = 0; i < bits.Length; i++)
        {
            excess += bits.BitAt(i) == 1 ? 1 : -1;
            if (excess < 0) return false;
        }
        return excess == 0 && bits.Ones == Count && bits.Rank0(bits.Length) == Count;
    }

    internal int OpenPosition(int index) => bits.Select1(index);
    internal int ClosePosition(int index) => MatchClose(OpenPosition(index));

    /** DFS index of the pair that opens at `position`: opens appear in preorder. */
    private int IndexAt(int position) => bits.Rank1(position) + 1;

    /** Matching ')' of the '(' at `open`: scan while the excess returns to zero. */
    internal int MatchClose(int open)
    {
        int depth = 0;
        for (int position = open; position < bits.Length; position++)
        {
            depth += bits.BitAt(position) == 1 ? 1 : -1;
            if (depth == 0) return position;
        }
        return -1;
    }

    /** Matching '(' of the ')' at `close`. */
    internal int MatchOpen(int close)
    {
        int depth = 0;
        for (int position = close; position >= 0; position--)
        {
            depth += bits.BitAt(position) == 0 ? 1 : -1;
            if (depth == 0) return position;
        }
        return -1;
    }

    /** The enclosing pair of a node is the parent, so it is the pair closing just after it. */
    internal int MatchingOpenOf(int index) => MatchOpen(ClosePosition(index));

    internal int SubtreeSize(int index) => (ClosePosition(index) - OpenPosition(index) + 1) / 2;

    internal int Depth(int index)
    {
        int open = OpenPosition(index);
        return bits.Rank1(open) - bits.Rank0(open);
    }

    internal int Parent(int index)
    {
        if (index <= 1) return -1;
        int open = OpenPosition(index);
        int target = bits.Rank1(open) - bits.Rank0(open) - 1;   // depth of the enclosing pair's open
        for (int position = open - 1; position >= 0; position--)
        {
            if (bits.BitAt(position) == 1 && bits.Rank1(position) - bits.Rank0(position) == target)
            {
                return IndexAt(position);
            }
        }
        return -1;
    }

    internal int FirstChild(int index)
    {
        int open = OpenPosition(index);
        int close = MatchClose(open);
        return close == open + 1 ? -1 : IndexAt(open + 1);
    }

    internal int NextSibling(int index)
    {
        int close = ClosePosition(index);
        return close + 1 < bits.Length && bits.BitAt(close + 1) == 1 ? IndexAt(close + 1) : -1;
    }

    internal int Degree(int index)
    {
        int count = 0;
        for (int child = FirstChild(index); child != -1; child = NextSibling(child)) count++;
        return count;
    }

    internal int Child(int index, int position)
    {
        if (position < 1) return -1;
        int child = FirstChild(index);
        for (int k = 1; k < position && child != -1; k++) child = NextSibling(child);
        return child;
    }

    /** Append one node as the last child of the parent identified by its DFS index. */
    internal int Link(int parentIndex)
    {
        if (parentIndex < 1 || parentIndex > Count) return -1;
        int close = ClosePosition(parentIndex);
        bits.Insert(close, 1);
        bits.Insert(close + 1, 0);
        return IndexAt(close);
    }

    /** Delete the whole subtree rooted at the node with this DFS index (the root is protected). */
    internal bool Cut(int index)
    {
        if (index <= 1 || index > Count) return false;
        int open = OpenPosition(index);
        int size = ClosePosition(index) - open + 1;
        for (int i = 0; i < size; i++) bits.Delete(open);
        return true;
    }
}

/** Independent pointer model with stable ids; DFS indices are re-derived on demand. */
internal sealed class ReferenceTree
{
    private readonly List<int>[] children;
    private readonly int[] parent;
    private readonly int[] depth;
    private readonly bool[] alive;
    private int nextId = 2;
    private int count = 1;

    internal ReferenceTree(int capacity)
    {
        children = new List<int>[capacity];
        parent = new int[capacity];
        depth = new int[capacity];
        alive = new bool[capacity];
        for (int i = 0; i < capacity; i++) children[i] = new List<int>();
        alive[1] = true;
        parent[1] = 0;
    }

    internal int Count => count;
    internal int Capacity => children.Length;

    internal List<int> Kids(int id) => children[id];
    internal int ParentId(int id) => parent[id];
    internal int Depth(int id) => depth[id];
    internal bool Alive(int id) => id >= 1 && id < Capacity && alive[id];

    internal int AddChild(int parentId)
    {
        if (nextId >= Capacity) return -1;
        int id = nextId++;
        alive[id] = true;
        parent[id] = parentId;
        depth[id] = depth[parentId] + 1;
        children[parentId].Add(id);
        count++;
        return id;
    }

    internal void CutSubtree(int id)
    {
        children[parent[id]].Remove(id);
        Disable(id);
    }

    private void Disable(int id)
    {
        alive[id] = false;
        count--;
        foreach (int child in children[id]) Disable(child);
    }

    internal int SubtreeSize(int id)
    {
        int size = 1;
        foreach (int child in children[id]) size += SubtreeSize(child);
        return size;
    }

    internal int SiblingOf(int id)
    {
        int p = parent[id];
        if (p == 0) return 0;
        var kids = children[p];
        int at = kids.IndexOf(id);
        return at + 1 < kids.Count ? kids[at + 1] : 0;
    }

    /** Preorder list of live ids plus the 1-based DFS index of each id (0 for dead ids). */
    internal List<int> DfsOrder(int[] indexMap)
    {
        var order = new List<int>();
        Walk(1, order, indexMap);
        return order;
    }

    private void Walk(int id, List<int> order, int[] indexMap)
    {
        indexMap[id] = order.Count + 1;
        order.Add(id);
        foreach (int child in children[id]) Walk(child, order, indexMap);
    }
}

public static class Solution
{
    private static ulong state = 0xBEEF5EEDUL;

    /** Every operation in the model must be reproduced by the BP representation. */
    private static void Verify(ReferenceTree tree, DynamicBpTree bp, string label)
    {
        var indexMap = new int[tree.Capacity];
        var order = tree.DfsOrder(indexMap);
        Check(bp.Count == tree.Count, label + " count");
        Check(bp.Count == order.Count, label + " dfs order length");
        Check(bp.Balanced(), label + " balanced");
        Check(bp.BitLength == 2 * bp.Count, label + " bit length");
        for (int at = 0; at < order.Count; at++)
        {
            int node = order[at];
            int index = indexMap[node];
            var kids = tree.Kids(node);
            Check(index == at + 1, label + " dfs index");
            Check(bp.MatchOpen(bp.ClosePosition(index)) == bp.OpenPosition(index),
                label + " bracket match " + node);
            Check(bp.Depth(index) == tree.Depth(node), label + " depth " + node);
            Check(bp.SubtreeSize(index) == tree.SubtreeSize(node), label + " subtree " + node);
            Check(bp.Degree(index) == kids.Count, label + " degree " + node);
            int expectedParent = tree.ParentId(node) == 0 ? -1 : indexMap[tree.ParentId(node)];
            Check(bp.Parent(index) == expectedParent, label + " parent " + node);
            Check(bp.FirstChild(index) == (kids.Count == 0 ? -1 : indexMap[kids[0]]),
                label + " first child " + node);
            int sibling = tree.SiblingOf(node);
            Check(bp.NextSibling(index) == (sibling == 0 ? -1 : indexMap[sibling]), label + " sibling " + node);
            for (int k = 0; k < kids.Count; k++)
            {
                Check(bp.Child(index, k + 1) == indexMap[kids[k]], label + " child " + node + "/" + (k + 1));
            }
            Check(bp.Child(index, kids.Count + 1) == -1, label + " child overflow " + node);
        }
    }

    /** Cross-check the chunked bit vector against a plain list after every edit. */
    private static void VerifyBits(DynamicBitVector vector, List<int> oracle, string label)
    {
        Check(vector.Length == oracle.Count, label + " length");
        int ones = 0;
        int zeros = 0;
        for (int i = 0; i < oracle.Count; i++)
        {
            if (vector.BitAt(i) != oracle[i]) throw new Exception("FAILED: " + label + " access " + i);
            ones += oracle[i];
            zeros += 1 - oracle[i];
            Check(vector.Rank1(i + 1) == ones && vector.Rank0(i + 1) == zeros, label + " rank " + i);
            if (oracle[i] == 1) Check(vector.Select1(ones) == i, label + " select1 " + i);
            else Check(vector.Select0(zeros) == i, label + " select0 " + i);
        }
        Check(vector.Ones == ones, label + " ones");
        Check(vector.Select1(ones + 1) == -1 && vector.Select0(zeros + 1) == -1, label + " select bound");
    }

    public static void Main()
    {
        // Chunked bit vector: edge cases, then random edits against a list oracle.
        var small = new DynamicBitVector(new[] { 1, 0, 1, 1 });
        Check(small.Length == 4 && small.Ones == 3 && small.BlockCount == 1, "bit vector basics");
        Check(small.BitAt(0) == 1 && small.BitAt(1) == 0 && small.Rank1(4) == 3 && small.Select1(2) == 2,
            "bit vector queries");
        small.Insert(2, 1);
        small.Delete(0);
        Check(small.Length == 4 && small.BitAt(0) == 0 && small.BitAt(1) == 1, "bit vector edit");
        var grows = new DynamicBitVector(Array.Empty<int>());
        Check(grows.Length == 0 && grows.Select1(1) == -1 && grows.Rank1(0) == 0, "empty bit vector");
        for (int i = 0; i < 200; i++) grows.Insert(grows.Length, i % 3 == 0 ? 1 : 0);
        Check(grows.Length == 200 && grows.Ones == 67 && grows.BlockCount >= 3, "bit vector grows");

        var oracle = new List<int>();
        var vector = new DynamicBitVector(Array.Empty<int>());
        for (int step = 0; step < 240; step++)
        {
            if (oracle.Count == 0 || SplitMix64.Next(ref state) % 4 != 0)
            {
                int at = (int)(SplitMix64.Next(ref state) % (ulong)(oracle.Count + 1));
                int bit = (int)(SplitMix64.Next(ref state) % 2);
                vector.Insert(at, bit);
                oracle.Insert(at, bit);
            }
            else
            {
                int at = (int)(SplitMix64.Next(ref state) % (ulong)oracle.Count);
                vector.Delete(at);
                oracle.RemoveAt(at);
            }
            VerifyBits(vector, oracle, "edit " + step);
        }

        // BP tree: seed a shape, then run a deterministic link/cut schedule.
        var tree = new ReferenceTree(600);
        var bp = new DynamicBpTree();
        Verify(tree, bp, "initial");

        var parent = tree.AddChild(1);
        Check(bp.Link(1) == 2 && parent == 2, "first link");
        Verify(tree, bp, "first link");
        var second = tree.AddChild(1);
        Check(bp.Link(1) == 3 && second == 3, "second link");
        Verify(tree, bp, "second link");
        var grand = tree.AddChild(second);
        Check(bp.Link(3) == 4 && grand == 4, "grandchild");
        Verify(tree, bp, "grandchild");
        Check(bp.Depth(4) == 2 && bp.SubtreeSize(1) == 4 && bp.Parent(4) == 3, "depth/size/parent");
        Check(bp.Parent(2) == 1 && bp.NextSibling(2) == 3 && bp.NextSibling(3) == -1, "sibling walk");
        Check(bp.Child(3, 1) == 4 && bp.Child(3, 2) == -1 && bp.FirstChild(4) == -1, "child walk");

        // Rejected operations: link under a missing parent, cut of the root or a missing node.
        Check(bp.Link(0) == -1 && bp.Link(99) == -1, "reject bad link");
        Check(!bp.Cut(1) && !bp.Cut(0) && !bp.Cut(99), "reject bad cut");

        Check(bp.Cut(2), "cut child");
        tree.CutSubtree(2);
        Verify(tree, bp, "cut child");
        Check(bp.Count == 3 && bp.Parent(2) == 1 && bp.Child(2, 1) == 3 && bp.SubtreeSize(2) == 2,
            "after cut indexes shift");
        Check(bp.Cut(2), "cut the middle child");
        tree.CutSubtree(3);
        Verify(tree, bp, "cut middle");
        Check(bp.Count == 1 && bp.SubtreeSize(1) == 1 && bp.Depth(1) == 0, "root only");

        // A deep chain, then a long deterministic schedule of links and cuts.
        int chainTail = tree.AddChild(1);
        Check(bp.Link(1) == 2, "chain first link");
        var chainIds = new List<int> { chainTail };
        for (int i = 0; i < 8; i++)
        {
            int parentIndex = 1 + chainIds.Count;         // the chain tail sits last in DFS order
            chainTail = tree.AddChild(chainTail);
            chainIds.Add(chainTail);
            Check(bp.Link(parentIndex) == parentIndex + 1, "chain link " + i);
        }
        Verify(tree, bp, "deep chain");
        Check(chainIds.Count == 9 && bp.Depth(10) == 9 && bp.SubtreeSize(1) == 10, "chain metrics");
        Check(bp.Parent(5) == 4 && bp.Degree(4) == 1, "chain links");

        int links = 0;
        int cuts = 0;
        for (int step = 0; step < 60; step++)
        {
            var indexMap = new int[tree.Capacity];
            var order = tree.DfsOrder(indexMap);
            ulong roll = SplitMix64.Next(ref state);
            if (roll % 3 == 2 && order.Count > 2)
            {
                int pick = 2 + (int)(roll % (ulong)(order.Count - 1));
                int node = order[pick - 1];
                Check(bp.Cut(pick), "scheduled cut " + step);
                tree.CutSubtree(node);
                cuts++;
            }
            else
            {
                int pick = 1 + (int)(roll % (ulong)order.Count);
                int parentId = order[pick - 1];
                // The new node is appended as the last child, so it lands right after the parent's
                // current subtree.
                int expectedIndex = pick + tree.SubtreeSize(parentId);
                int child = tree.AddChild(parentId);
                Check(child > 0, "model capacity " + step);
                int actual = bp.Link(pick);
                Check(actual == expectedIndex, "scheduled link " + step + " parent=" + pick
                    + " expected=" + expectedIndex + " actual=" + actual + " nodes=" + tree.Count);
                links++;
            }
            Verify(tree, bp, "schedule " + step);
        }

        // Cutting everything except the root leaves the minimal tree.
        for (int guard = 0; guard < 200 && bp.Count > 1; guard++)
        {
            var indexMap = new int[tree.Capacity];
            tree.DfsOrder(indexMap);
            Check(bp.Cut(2), "dismantle");
            tree.CutSubtree(KidByIndex(tree, indexMap, 2));
        }
        Verify(tree, bp, "dismantled");
        Check(bp.Count == 1 && bp.BitLength == 2 && bp.Degree(1) == 0, "only the root remains");

        Check(bp.Link(1) == 2, "regrow after dismantle");
        tree.AddChild(1);
        Verify(tree, bp, "regrown");

        Console.WriteLine("  bitVectorEdits=240 finalNodes=" + bp.Count + " links=" + links + " cuts=" + cuts
            + " bpBlocks=" + bp.BlockCount + " bpBits=" + bp.BitLength);
        Console.WriteLine("PASS");
    }

    private static int KidByIndex(ReferenceTree tree, int[] indexMap, int index)
    {
        for (int id = 1; id < tree.Capacity; id++)
        {
            if (tree.Alive(id) && indexMap[id] == index) return id;
        }
        return 0;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
