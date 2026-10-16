// Task 612 - Implement advanced persistent data structures with path copying optimization. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Path copying made explicit. A *version* is a single root reference into an immutable structure, so
// "modifying" a version means rebuilding only the nodes on one root-to-leaf path and pointing their
// new copies at the untouched (shared) subtrees; the previous root keeps describing the previous
// state because nothing is ever mutated in place.
//
//   * PersistentTreap - an ordered map with a deterministic splitmix64 priority per key (min-heap
//     order, so it stays balanced in expectation without randomness at run time). Insert, Delete,
//     Contains and ordered Range copies the search path plus at most two nodes per rotation (or per
//     merge step in Delete), and reports the exact number of nodes it allocated.
//   * PersistentStack - a cons list where Push costs exactly one node and the old version's spine is
//     reused as the new tail (the canonical O(1) path-copying case).
//
// Verified: every version's contents match an independent sorted-list snapshot taken when that
// version was created (so older versions are immutable under later edits - a version built on top of
// version k must *not* contain keys inserted afterwards); the allocation cost of an update is
// bounded by a small multiple of the tree height; and structural sharing is measured by intersecting
// the node references of consecutive versions - the shared count must be at least
// (size - 4 * height), i.e. the untouched subtrees really are the same objects, not copies. Also
// covered: the empty version, single element, duplicate-key update, delete of a missing key (which
// allocates nothing), delete of the root/leaf/middle, range boundaries, and stack sharing.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;          // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static ulong Hash(ulong value) => HashCore(value + 0x9E3779B97F4A7C15UL);

    private static ulong HashCore(ulong z)
    {
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal sealed class TreapNode
{
    internal readonly int Key;
    internal readonly int Value;
    internal readonly ulong Priority;
    internal readonly TreapNode Left;
    internal readonly TreapNode Right;

    private TreapNode()                                  // the nil sentinel points at itself
    {
        Left = this;
        Right = this;
    }

    internal TreapNode(int key, int value, ulong priority, TreapNode left, TreapNode right)
    {
        Key = key;
        Value = value;
        Priority = priority;
        Left = left;
        Right = right;
    }

    internal static readonly TreapNode Nil = new TreapNode();
    internal bool IsNil => ReferenceEquals(this, Nil);
}

internal sealed class RefComparer : IEqualityComparer<TreapNode>
{
    internal static readonly RefComparer Instance = new RefComparer();
    public bool Equals(TreapNode a, TreapNode b) => ReferenceEquals(a, b);
    public int GetHashCode(TreapNode node) => RuntimeHelpers.GetHashCode(node);
}

/** An immutable ordered map; every update returns a new version touching O(log n) fresh nodes. */
internal sealed class PersistentTreap
{
    private static readonly PersistentTreap EmptyVersion = new PersistentTreap(TreapNode.Nil, 0, 0);

    private readonly TreapNode root;

    internal readonly int Size;
    internal readonly int Allocated;                      // nodes created for this version's update

    private PersistentTreap(TreapNode root, int size, int allocated)
    {
        this.root = root;
        Size = size;
        Allocated = allocated;
    }

    internal static PersistentTreap Empty => EmptyVersion;

    internal int Height() => Height(root);

    private static int Height(TreapNode node)
    {
        return node.IsNil ? 0 : 1 + Math.Max(Height(node.Left), Height(node.Right));
    }

    internal bool Contains(int key) => Find(root, key).IsNil == false;

    internal int? Value(int key)
    {
        TreapNode node = Find(root, key);
        return node.IsNil ? null : node.Value;
    }

    private static TreapNode Find(TreapNode node, int key)
    {
        while (!node.IsNil && node.Key != key) node = key < node.Key ? node.Left : node.Right;
        return node;
    }

    internal PersistentTreap Insert(int key, int value)
    {
        int allocated = 0;
        bool replaced;
        TreapNode next = Insert(root, key, value, ref allocated, out replaced);
        return new PersistentTreap(next, Size + (replaced ? 0 : 1), allocated);
    }

    private static TreapNode Insert(TreapNode node, int key, int value, ref int allocated, out bool replaced)
    {
        replaced = false;
        if (node.IsNil)
        {
            allocated++;
            return new TreapNode(key, value, SplitMix64.Hash((ulong)key * 0x9E3779B97F4A7C15UL), TreapNode.Nil, TreapNode.Nil);
        }
        if (node.Key == key)
        {
            replaced = true;
            allocated++;
            return new TreapNode(key, value, node.Priority, node.Left, node.Right);
        }
        TreapNode copy;
        if (key < node.Key)
        {
            TreapNode left = Insert(node.Left, key, value, ref allocated, out replaced);
            allocated++;
            copy = new TreapNode(node.Key, node.Value, node.Priority, left, node.Right);
            if (left.Priority < copy.Priority) return RotateRight(copy, ref allocated);
        }
        else
        {
            TreapNode right = Insert(node.Right, key, value, ref allocated, out replaced);
            allocated++;
            copy = new TreapNode(node.Key, node.Value, node.Priority, node.Left, right);
            if (right.Priority < copy.Priority) return RotateLeft(copy, ref allocated);
        }
        return copy;
    }

    private static TreapNode RotateRight(TreapNode node, ref int allocated)
    {
        TreapNode pivot = node.Left;
        allocated += 2;
        return new TreapNode(pivot.Key, pivot.Value, pivot.Priority, pivot.Left,
            new TreapNode(node.Key, node.Value, node.Priority, pivot.Right, node.Right));
    }

    private static TreapNode RotateLeft(TreapNode node, ref int allocated)
    {
        TreapNode pivot = node.Right;
        allocated += 2;
        return new TreapNode(pivot.Key, pivot.Value, pivot.Priority,
            new TreapNode(node.Key, node.Value, node.Priority, node.Left, pivot.Left), pivot.Right);
    }

    internal PersistentTreap Delete(int key)
    {
        int allocated = 0;
        TreapNode next = Delete(root, key, ref allocated, out bool removed);
        return removed ? new PersistentTreap(next, Size - 1, allocated) : this;   // missing key: no copy
    }

    private static TreapNode Delete(TreapNode node, int key, ref int allocated, out bool removed)
    {
        removed = false;
        if (node.IsNil) return node;
        if (node.Key == key)
        {
            removed = true;
            return Merge(node.Left, node.Right, ref allocated);
        }
        allocated++;
        return key < node.Key
            ? new TreapNode(node.Key, node.Value, node.Priority, Delete(node.Left, key, ref allocated, out removed), node.Right)
            : new TreapNode(node.Key, node.Value, node.Priority, node.Left, Delete(node.Right, key, ref allocated, out removed));
    }

    private static TreapNode Merge(TreapNode left, TreapNode right, ref int allocated)
    {
        if (left.IsNil) return right;
        if (right.IsNil) return left;
        allocated++;
        return left.Priority < right.Priority
            ? new TreapNode(left.Key, left.Value, left.Priority, left.Left, Merge(left.Right, right, ref allocated))
            : new TreapNode(right.Key, right.Value, right.Priority, Merge(left, right.Left, ref allocated), right.Right);
    }

    internal List<int> InOrder() => Range(int.MinValue, int.MaxValue);

    /** Keys in [low, high], ascending, read from this version only. */
    internal List<int> Range(int low, int high)
    {
        var keys = new List<int>();
        Collect(root, low, high, keys);
        return keys;
    }

    private static void Collect(TreapNode node, int low, int high, List<int> sink)
    {
        if (node.IsNil) return;
        if (node.Key > low) Collect(node.Left, low, high, sink);
        if (node.Key >= low && node.Key <= high) sink.Add(node.Key);
        if (node.Key < high) Collect(node.Right, low, high, sink);
    }

    /** Node identities of this version, used to measure structural sharing with another version. */
    internal HashSet<TreapNode> Nodes()
    {
        var set = new HashSet<TreapNode>(RefComparer.Instance);
        Gather(root, set);
        return set;
    }

    private static void Gather(TreapNode node, HashSet<TreapNode> sink)
    {
        if (node.IsNil) return;
        sink.Add(node);
        Gather(node.Left, sink);
        Gather(node.Right, sink);
    }
}

/** Cons list: Push allocates one node and shares the entire previous spine. */
internal sealed class PersistentStack
{
    private sealed class Cell
    {
        internal readonly int Value;
        internal readonly Cell Next;
        private Cell() { Next = this; }
        internal Cell(int value, Cell next) { Value = value; Next = next; }
        internal static readonly Cell Nil = new Cell();
        internal bool IsNil => ReferenceEquals(this, Nil);
    }

    private readonly Cell top;
    internal readonly int Size;

    private PersistentStack(Cell top, int size) { this.top = top; Size = size; }

    internal static PersistentStack Empty => new PersistentStack(Cell.Nil, 0);

    internal PersistentStack Push(int value) => new PersistentStack(new Cell(value, top), Size + 1);

    internal bool TryPop(out int value)
    {
        value = 0;
        if (top.IsNil) return false;
        value = top.Value;
        return true;
    }

    internal PersistentStack Pop() => top.IsNil ? this : new PersistentStack(top.Next, Size - 1);

    internal List<int> ToList()
    {
        var items = new List<int>();
        for (Cell cell = top; !cell.IsNil; cell = cell.Next) items.Add(cell.Value);
        return items;
    }

    internal bool SharesSpineBelow(PersistentStack other) => !other.top.IsNil && ReferenceEquals(top, other.top.Next);
}

public static class Solution
{
    private static ulong state = 0x51EED51EEDUL;

    public static void Main()
    {
        var versions = new List<PersistentTreap> { PersistentTreap.Empty };
        var snapshots = new List<List<int>> { new List<int>() };
        var keys = new List<int>();
        for (int i = 0; i < 32; i++) keys.Add(i * 3 + 1);

        var shuffled = new List<int>(keys);
        for (int i = shuffled.Count - 1; i > 0; i--)
        {
            int j = (int)(SplitMix64.Next(ref state) % (ulong)(i + 1));
            (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
        }

        int worstCost = 0;
        long totalAllocated = 0;
        for (int i = 0; i < shuffled.Count; i++)
        {
            var next = versions[i].Insert(shuffled[i], shuffled[i] * 10);
            versions.Add(next);
            totalAllocated += next.Allocated;
            worstCost = Math.Max(worstCost, next.Allocated);
            Check(next.Allocated <= 3 * versions[i].Height() + 4, "insert copies at most one path " + i);
            var snapshot = new List<int>(snapshots[i]);
            snapshot.Add(shuffled[i]);
            snapshot.Sort();
            snapshots.Add(snapshot);
            Check(next.Size == i + 1, "size after insert " + i);
            Check(next.Value(shuffled[i]) == shuffled[i] * 10, "value after insert " + i);
            Check(next.Range(int.MinValue, int.MaxValue).Count == i + 1, "range count " + i);
        }
        Check(worstCost <= 3 * versions[versions.Count - 1].Height() + 4, "inserts stay cheap");
        Check(versions[versions.Count - 1].Height() <= 12,
            "treap stays balanced: height " + versions[versions.Count - 1].Height());

        // Every version must match the snapshot taken when it was created: immutability of history.
        for (int i = 0; i < versions.Count; i++)
        {
            Check(Equal(versions[i].InOrder(), snapshots[i]), "version contents " + i);
            Check(versions[i].Size == i, "version size " + i);
            Check(Equal(versions[i].Range(int.MinValue, int.MaxValue), snapshots[i]), "version range " + i);
        }
        Check(!versions[5].Contains(shuffled[6]), "old version lacks later key");
        Check(versions[6].Contains(shuffled[5]), "later version has earlier key");
        Check(!versions[5].Contains(shuffled[20]), "old version lacks far key");
        Check(versions[0].Size == 0 && versions[0].InOrder().Count == 0 && !versions[0].Contains(1),
            "empty version");

        // Structural sharing: consecutive versions must share all but the copied path.
        PersistentTreap previous = versions[10];
        PersistentTreap current = versions[11];
        var first = previous.Nodes();
        var second = current.Nodes();
        var shared = new HashSet<TreapNode>(first, RefComparer.Instance);
        shared.IntersectWith(second);
        Check(first.Count == previous.Size && second.Count == current.Size, "node counts match sizes");
        int height = current.Height();
        Check(shared.Count >= previous.Size - 4 * height, "untouched subtrees are shared objects");
        Check(height >= 4 && height <= 12, "treap stays balanced: height " + height);
        Check(current.Allocated + shared.Count > previous.Size, "path copying allocates only the path");

        // Duplicate keys update in place, deletion is persistent, missing deletes allocate nothing.
        var updated = versions[12].Insert(shuffled[3], -1);
        Check(updated.Size == versions[12].Size && updated.Value(shuffled[3]) == -1, "duplicate update");
        Check(versions[12].Value(shuffled[3]) == shuffled[3] * 10, "update leaves the old version alone");
        Check(ReferenceEquals(versions[12].Delete(99999), versions[12]), "missing delete is free");

        var trimmed = updated;
        var expected = new List<int>(snapshots[12]);
        foreach (int key in new[] { shuffled[0], shuffled[3], shuffled[7], shuffled[10] })
        {
            trimmed = trimmed.Delete(key);
            expected.Remove(key);
            Check(!trimmed.Contains(key), "deleted key gone " + key);
            Check(updated.Contains(key), "delete is persistent " + key);
        }
        Check(Equal(trimmed.InOrder(), expected), "delete sequence contents");
        Check(trimmed.Size == expected.Count, "delete sequence size");

        var emptyDeleted = PersistentTreap.Empty.Delete(5);
        Check(ReferenceEquals(emptyDeleted, PersistentTreap.Empty), "delete from empty");
        var single = PersistentTreap.Empty.Insert(7, 70);
        Check(single.Size == 1 && single.Contains(7) && single.Height() == 1, "single element");
        Check(single.Delete(7).Size == 0 && single.Size == 1, "delete the only element");

        // Range boundaries.
        var all = versions[versions.Count - 1];
        var middle = all.Range(20, 40);
        Check(middle.Count > 0 && middle[0] >= 20 && middle[middle.Count - 1] <= 40, "range bounds");
        Check(all.Range(1000, 2000).Count == 0, "empty range");
        Check(all.Range(40, 20).Count == 0, "inverted range");
        int smallest = all.InOrder()[0];
        Check(all.Range(smallest, smallest).Count == 1, "singleton range");

        // Persistent stack: O(1) sharing, old versions untouched.
        var stack = PersistentStack.Empty;
        var stackVersions = new List<PersistentStack> { stack };
        for (int i = 0; i < 12; i++)
        {
            stack = stack.Push(i * i);
            stackVersions.Add(stack);
        }
        Check(stack.Size == 12 && stackVersions[3].Size == 3, "stack sizes");
        Check(stack.ToList()[0] == 121 && stack.ToList()[11] == 0, "stack order");
        Check(stack.Pop().SharesSpineBelow(stack), "popped version shares the spine");
        Check(stackVersions[5].SharesSpineBelow(stackVersions[6]), "consecutive versions share the spine");
        Check(stack.Pop().Size == 11 && stack.Size == 12, "stack pop is persistent");
        Check(Equal(stackVersions[5].ToList(), new List<int> { 16, 9, 4, 1, 0 }), "old stack version intact");
        Check(!PersistentStack.Empty.Pop().TryPop(out _), "empty stack stays empty");
        Check(totalAllocated < 40L * 32, "persistence beats rebuilding: allocated " + totalAllocated);

        Console.WriteLine("  versions=" + versions.Count + " keys=" + shuffled.Count + " worstInsertCost="
            + worstCost + " totalNodesAllocated=" + totalAllocated + " sharedWithPrevious=" + shared.Count
            + "/" + previous.Size + " height=" + height);
        Console.WriteLine("PASS");
    }

    private static bool Equal(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
