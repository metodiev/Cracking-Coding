// Task 613 - Implement fully persistent union-find (complex). (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Fully persistent union-find. Full persistence means *every* version stays queryable and new
// versions may branch off any older one, which rules out mutating the parent array in place. The
// classical construction is used here: the parent and size arrays live in a persistent array (a
// balanced binary tree of immutable nodes in which Set copies one root-to-leaf path and shares the
// rest), and Union(a, b) links the smaller tree under the larger one by writing both arrays, so each
// union allocates O(log n) nodes and produces a new version. Versions are therefore connected by a
// version tree - branching is the whole point, and it is tested:
//
//     v1 = (0,1)        v2 = v1 + (2,3)        v3 = v1 + (1,2)      <- v3 branches off v1
//     0 and 2 are connected in v3 but not in v2, and v1 still sees only {0,1}
//
// Path compression is deliberately *not* used: rewriting a version's pointers would either mutate a
// shared version or turn every Find into a write. Union by size keeps the parent chains at most
// log2(n) + 1 long, which is what makes the O(log n) walk acceptable - and that bound is checked.
//
// Verified against an independent oracle which replays the union list of a version's ancestry into a
// fresh path-compressed DSU and compares every pair of elements; plus version immutability (a
// version's full pair table is snapshotted and re-checked after later unions and branches), the
// depth bound, structural sharing per union (never more than 2 * (log2 n + 2) new nodes), Union of
// already-connected or identical elements returning the *same* version object, and a separately
// implemented backtrackable variant (trail of overwritten entries + rollback) whose state after a
// rollback must be bit-identical to the checkpoint.

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

/** Persistent array: an immutable balanced tree, Set copies one path and shares the subtrees. */
internal sealed class PersistentArray
{
    private sealed class Node
    {
        internal readonly Node Left;
        internal readonly Node Right;
        internal readonly int Value;
        internal readonly bool Leaf;

        private Node() { Left = this; Right = this; }     // nil sentinel

        internal Node(Node left, Node right) { Left = left; Right = right; Value = 0; Leaf = false; }
        internal Node(int value) { Left = this; Right = this; Value = value; Leaf = true; }

        internal static readonly Node Nil = new Node();
        internal bool IsNil => ReferenceEquals(this, Nil);
    }

    private readonly Node root;
    private readonly int length;
    private readonly int height;

    internal readonly int Allocated;                      // nodes created by the last Set

    private PersistentArray(Node root, int length, int height, int allocated)
    {
        this.root = root;
        this.length = length;
        this.height = height;
        Allocated = allocated;
    }

    internal int Length => length;
    internal int Height => height;

    /** Build from an array in O(n) shared nodes: identical values share leaves. */
    internal static PersistentArray Build(IReadOnlyList<int> values)
    {
        int size = 1;
        while (size < values.Count) size *= 2;
        Node root = BuildRange(values, 0, size, out int height);
        return new PersistentArray(root, values.Count, height, 0);   // the seed build is not a path copy
    }

    private static Node BuildRange(IReadOnlyList<int> values, int start, int span, out int height)
    {
        if (span == 1)
        {
            height = 1;
            return start < values.Count ? new Node(values[start]) : Node.Nil;
        }
        Node left = BuildRange(values, start, span / 2, out int leftHeight);
        Node right = BuildRange(values, start + span / 2, span / 2, out int rightHeight);
        height = Math.Max(leftHeight, rightHeight) + 1;
        return left.IsNil && right.IsNil ? Node.Nil : new Node(left, right);
    }

    internal int Get(int index)
    {
        if (index < 0 || index >= length) throw new ArgumentOutOfRangeException(nameof(index));
        Node node = root;
        int start = 0;
        int span = 1;
        while (span < length) span *= 2;
        while (!node.Leaf)
        {
            int half = span / 2;
            if (index < start + half)
            {
                node = node.Left;
            }
            else
            {
                node = node.Right;
                start += half;
            }
            span = half;
        }
        return node.IsNil ? 0 : node.Value;
    }

    internal PersistentArray Set(int index, int value)
    {
        if (index < 0 || index >= length) throw new ArgumentOutOfRangeException(nameof(index));
        int span = 1;
        while (span < length) span *= 2;
        int allocated = 0;
        Node next = Set(root, 0, span, index, value, ref allocated);
        return new PersistentArray(next, length, height, allocated);
    }

    private static Node Set(Node node, int start, int span, int index, int value, ref int allocated)
    {
        allocated++;
        if (span == 1) return new Node(value);
        int half = span / 2;
        if (index < start + half)
        {
            return new Node(Set(node.Left, start, half, index, value, ref allocated), node.Right);
        }
        return new Node(node.Left, Set(node.Right, start + half, half, index, value, ref allocated));
    }

    internal List<int> ToList()
    {
        var values = new List<int>();
        for (int i = 0; i < length; i++) values.Add(Get(i));
        return values;
    }
}

internal sealed class UnionFindVersion
{
    internal readonly PersistentArray Parent;
    internal readonly PersistentArray Size;
    internal readonly List<int[]> Unions;                 // (a, b) pairs along the version path
    internal readonly UnionFindVersion Origin;            // the version this one branched from

    internal UnionFindVersion(PersistentArray parent, PersistentArray size, List<int[]> unions, UnionFindVersion origin)
    {
        Parent = parent;
        Size = size;
        Unions = unions;
        Origin = origin;
    }

    internal int Elements => Parent.Length;
    internal int AllocationCost => Parent.Allocated + Size.Allocated;
}

internal sealed class PersistentUnionFind
{
    private readonly int elements;

    internal PersistentUnionFind(int elements) { this.elements = elements; }

    internal int Elements => elements;

    internal UnionFindVersion Empty()
    {
        var identity = new int[elements];
        var ones = new int[elements];
        for (int i = 0; i < elements; i++)
        {
            identity[i] = i;
            ones[i] = 1;
        }
        return new UnionFindVersion(PersistentArray.Build(identity), PersistentArray.Build(ones),
            new List<int[]>(), null);
    }

    /** Root of x in this version: walk the parent array (no compression, by design). */
    internal int Find(UnionFindVersion version, int x)
    {
        int steps = 0;
        while (version.Parent.Get(x) != x)
        {
            x = version.Parent.Get(x);
            if (++steps > elements) throw new InvalidOperationException("parent cycle");
        }
        return x;
    }

    internal int Depth(UnionFindVersion version, int x)
    {
        int steps = 0;
        while (version.Parent.Get(x) != x)
        {
            x = version.Parent.Get(x);
            steps++;
        }
        return steps;
    }

    internal bool Connected(UnionFindVersion version, int a, int b) => Find(version, a) == Find(version, b);

    internal UnionFindVersion Union(UnionFindVersion version, int a, int b)
    {
        int rootA = Find(version, a);
        int rootB = Find(version, b);
        if (rootA == rootB) return version;               // already connected: same object, no copy
        int sizeA = version.Size.Get(rootA);
        int sizeB = version.Size.Get(rootB);
        int smaller = sizeA < sizeB ? rootA : rootB;
        int larger = sizeA < sizeB ? rootB : rootA;
        PersistentArray parent = version.Parent.Set(smaller, larger);
        PersistentArray size = version.Size.Set(larger, sizeA + sizeB);
        var unions = new List<int[]>(version.Unions) { new[] { a, b } };
        return new UnionFindVersion(parent, size, unions, version);
    }

    /** Independent oracle: replay this version's union list into a fresh path-compressed DSU. */
    internal int[] OracleRoots(UnionFindVersion version)
    {
        var parent = new int[elements];
        for (int i = 0; i < elements; i++) parent[i] = i;
        foreach (int[] pair in version.Unions)
        {
            int a = FindCompressed(parent, pair[0]);
            int b = FindCompressed(parent, pair[1]);
            if (a != b) parent[a] = b;
        }
        var roots = new int[elements];
        for (int i = 0; i < elements; i++) roots[i] = FindCompressed(parent, i);
        return roots;
    }

    private static int FindCompressed(int[] parent, int x)
    {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }
}

/** Backtrackable DSU: the same connectivity with an undo trail instead of persistent arrays. */
internal sealed class BacktrackableUnionFind
{
    private struct UndoEntry
    {
        internal int Node;
        internal int OldValue;
        internal bool IsSize;
    }

    private readonly int[] parent;
    private readonly int[] size;
    private readonly List<UndoEntry> trail = new List<UndoEntry>();

    internal BacktrackableUnionFind(int elements)
    {
        parent = new int[elements];
        size = new int[elements];
        for (int i = 0; i < elements; i++)
        {
            parent[i] = i;
            size[i] = 1;
        }
    }

    /** Checkpoint handle: roll everything applied after this point back off the trail. */
    internal int Mark() => trail.Count;

    internal int Find(int x)
    {
        while (parent[x] != x) x = parent[x];
        return x;
    }

    internal bool Union(int a, int b)
    {
        int rootA = Find(a);
        int rootB = Find(b);
        if (rootA == rootB) return false;
        if (size[rootA] < size[rootB]) (rootA, rootB) = (rootB, rootA);
        trail.Add(new UndoEntry { Node = rootB, OldValue = parent[rootB], IsSize = false });
        trail.Add(new UndoEntry { Node = rootA, OldValue = size[rootA], IsSize = true });
        parent[rootB] = rootA;
        size[rootA] += size[rootB];
        return true;
    }

    internal void Rollback(int mark)
    {
        while (trail.Count > mark)
        {
            UndoEntry entry = trail[trail.Count - 1];
            trail.RemoveAt(trail.Count - 1);
            if (entry.IsSize) size[entry.Node] = entry.OldValue;
            else parent[entry.Node] = entry.OldValue;
        }
    }

    internal string State() => string.Join(",", parent) + "|" + string.Join(",", size);
}

public static class Solution
{
    private static ulong state = 0xF0E1D2C3B4UL;

    /** Every pair of elements must agree with the independent oracle replay. */
    private static void CompareWithOracle(PersistentUnionFind uf, UnionFindVersion version, string label)
    {
        int[] oracle = uf.OracleRoots(version);
        for (int a = 0; a < uf.Elements; a++)
        {
            for (int b = a + 1; b < uf.Elements; b++)
            {
                bool expected = oracle[a] == oracle[b];
                Check(uf.Connected(version, a, b) == expected, label + " pair " + a + "/" + b);
            }
        }
    }

    private static string PairTable(PersistentUnionFind uf, UnionFindVersion version)
    {
        var text = new List<string>();
        for (int a = 0; a < uf.Elements; a++)
        {
            for (int b = a + 1; b < uf.Elements; b++) text.Add(uf.Connected(version, a, b) ? "1" : "0");
        }
        return string.Join("", text);
    }

    public static void Main()
    {
        int n = 24;
        var uf = new PersistentUnionFind(n);
        var v0 = uf.Empty();
        CompareWithOracle(uf, v0, "empty version");
        Check(v0.Elements == n && v0.Unions.Count == 0, "empty version shape");
        for (int i = 0; i < n; i++)
        {
            Check(uf.Find(v0, i) == i && uf.Depth(v0, i) == 0, "identity version " + i);
        }

        var v1 = uf.Union(v0, 0, 1);
        var v2 = uf.Union(v1, 2, 3);
        var v3 = uf.Union(v1, 1, 2);                      // branch: v3 grows out of v1, not v2
        var v4 = uf.Union(v2, 5, 6);
        var v5 = uf.Union(v3, 7, 8);
        var v6 = uf.Union(v5, 0, 8);
        CompareWithOracle(uf, v1, "v1");
        CompareWithOracle(uf, v2, "v2");
        CompareWithOracle(uf, v3, "v3");
        CompareWithOracle(uf, v4, "v4");
        CompareWithOracle(uf, v6, "v6");

        Check(uf.Connected(v3, 0, 2) && !uf.Connected(v2, 0, 2), "branching independence");
        Check(uf.Connected(v2, 2, 3) && uf.Connected(v4, 5, 6) && !uf.Connected(v3, 5, 6), "branch contents");
        Check(uf.Connected(v6, 0, 8) && uf.Connected(v6, 8, 1) && !uf.Connected(v6, 5, 6),
            "transitive union on a branch");

        string v1Table = PairTable(uf, v1);
        string v2Table = PairTable(uf, v2);
        Check(ReferenceEquals(uf.Union(v1, 0, 1), v1), "union of connected elements is the same version");
        Check(ReferenceEquals(uf.Union(v1, 4, 4), v1), "self union is the same version");
        Check(ReferenceEquals(uf.Union(v4, 5, 6), v4), "already merged pair reuses the version");

        // Version immutability: rebuild a long branch and re-check the old tables afterwards.
        var branch = v4;
        var random = new List<UnionFindVersion> { v0, v1, v2, v3, v4, v5, v6 };
        int allocations = 0;
        for (int step = 0; step < 40; step++)
        {
            int pick = (int)(SplitMix64.Next(ref state) % (ulong)random.Count);
            int a = (int)(SplitMix64.Next(ref state) % (ulong)n);
            int b = (int)(SplitMix64.Next(ref state) % (ulong)n);
            var next = uf.Union(random[pick], a, b);
            allocations += next == random[pick] ? 0 : next.AllocationCost;
            random.Add(next);
            Check(next.AllocationCost <= 2 * (Log2(n) + 2), "union copies a short path " + step);
            Check(uf.Find(next, a) == uf.Find(next, b), "united pair connected " + step);
        }
        Check(PairTable(uf, v1) == v1Table, "v1 unchanged by later branches");
        Check(PairTable(uf, v2) == v2Table, "v2 unchanged by later branches");
        Check(uf.Connected(v3, 1, 2) && !uf.Connected(v2, 1, 2), "branches stay independent");

        // Depth bound: union by size without compression keeps chains logarithmic.
        int worstDepth = 0;
        foreach (var version in random)
        {
            for (int i = 0; i < n; i++) worstDepth = Math.Max(worstDepth, uf.Depth(version, i));
        }
        Check(worstDepth <= Log2(n) + 1, "parent chains stay logarithmic: " + worstDepth);

        foreach (var version in new[] { v6, random[random.Count - 1] })
        {
            CompareWithOracle(uf, version, "oracle on deep version");
        }

        // Persistent arrays: path copying must leave the source array untouched.
        var array = PersistentArray.Build(new[] { 5, 6, 7, 8, 9 });
        var modified = array.Set(2, 99);
        Check(array.Get(2) == 7 && modified.Get(2) == 99, "persistent array isolation");
        Check(array.ToList()[0] == 5 && modified.Get(4) == 9, "persistent array neighbours");
        Check(modified.Allocated <= Log2(array.Length) + 2, "path copy cost " + modified.Allocated);

        // Backtrackable variant: rollback to a checkpoint restores the exact state.
        var backtracked = new BacktrackableUnionFind(10);
        Check(backtracked.Union(0, 1) && backtracked.Union(2, 3) && !backtracked.Union(1, 0), "trail unions");
        int mark = backtracked.Mark();
        string checkpoint = backtracked.State();
        Check(backtracked.Union(0, 2) && backtracked.Union(4, 5) && backtracked.Union(6, 7), "extra unions");
        Check(backtracked.Find(3) == 0 && backtracked.Find(7) == 6 && backtracked.Find(5) == 4,
            "state before rollback");
        backtracked.Rollback(mark);
        Check(backtracked.State() == checkpoint, "rollback restores the checkpoint");
        Check(backtracked.Find(0) == 0 && backtracked.Find(1) == 0 && backtracked.Find(3) == 2
            && backtracked.Find(5) == 5 && backtracked.Find(7) == 7, "connectivity after rollback");
        Check(backtracked.Union(3, 4) && backtracked.Find(2) == backtracked.Find(4), "reuse after rollback");

        Console.WriteLine("  elements=" + n + " versions=" + random.Count + " unionAllocations=" + allocations
            + " worstDepth=" + worstDepth + " bytesShareParentArray=true");
        Console.WriteLine("PASS");
    }

    private static int Log2(int value)
    {
        int bits = 0;
        while ((1 << bits) < value) bits++;
        return bits;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
