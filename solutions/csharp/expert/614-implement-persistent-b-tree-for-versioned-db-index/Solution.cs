// Task 614 - Implement persistent B-tree for versioned DB index. (Tier: Expert, CTCI Ch 14 — Databases)
//
// A versioned B+tree index. Pages are immutable: a write copies the pages on one root-to-leaf path
// (read-copy-update) and shares every untouched page with the previous root, so a *version* is just
// a root reference and an old root keeps answering old snapshots - the classic MVCC index shape.
// Leaves hold the key/value pairs, internal pages hold separator keys, and a node that overflows its
// capacity splits: a leaf split promotes a *copy* of its median key (the key stays in the right
// leaf, as B+trees require) while an internal split promotes the median key and removes it. A root
// split adds one level, which is the only way the tree grows.
//
// Lookup descends by binary search. RangeScan is an in-order descent pruned by the separators, so it
// touches only the pages it needs (a production index would also follow sibling links to avoid the
// descent). Delete is deliberately a prototype path: this index is insert-oriented, so Delete
// rebuilds the tree from its entries without the key - correct, and documented as O(n log n) rather
// than the merge/borrow rebalancing a full B-tree delete would do.
//
// Verified: after every insert a validator checks the B-tree invariants over the whole tree (keys
// sorted and inside their bounds, internal pages have one more child than keys, all leaves at the
// same depth, occupancy within capacity), every version is compared against an independent sorted
// entry list for point lookups and for boundary ranges, older versions must *not* see keys inserted
// later, an update of an existing key must keep the size and leave the old version's value alone,
// the copy cost of an insert is bounded by a small multiple of the tree height, and shared pages
// between consecutive versions are counted by reference to show the path-only copying.

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
}

internal sealed class BTreePage
{
    internal readonly bool Leaf;
    internal readonly int Count;
    internal readonly int[] Keys;
    internal readonly int[] Values;                        // leaves only
    internal readonly BTreePage[] Children;                // internal pages only

    internal BTreePage(int[] keys, int[] values, int count)
    {
        Leaf = true;
        Count = count;
        Keys = keys;
        Values = values;
        Children = Array.Empty<BTreePage>();
    }

    internal BTreePage(int[] keys, BTreePage[] children, int count)
    {
        Leaf = false;
        Count = count;
        Keys = keys;
        Children = children;
        Values = Array.Empty<int>();
    }
}

internal sealed class RefComparer : IEqualityComparer<BTreePage>
{
    internal static readonly RefComparer Instance = new RefComparer();
    public bool Equals(BTreePage a, BTreePage b) => ReferenceEquals(a, b);
    public int GetHashCode(BTreePage page) => RuntimeHelpers.GetHashCode(page);
}

internal sealed class PersistentBTree
{
    internal const int Capacity = 4;                       // max keys per page, so max 5 children

    private readonly BTreePage root;

    internal readonly int Size;
    internal readonly int Allocated;                       // pages created by the last insert

    private PersistentBTree(BTreePage root, int size, int allocated)
    {
        this.root = root;
        Size = size;
        Allocated = allocated;
    }

    internal static PersistentBTree Empty => new PersistentBTree(null, 0, 0);

    internal int Height()
    {
        int height = 0;
        for (BTreePage page = root; page != null; page = page.Leaf ? null : page.Children[0]) height++;
        return height;
    }

    internal int? Lookup(int key)
    {
        BTreePage page = root;
        while (page != null)
        {
            int at = page.Leaf ? LowerBound(page, key) : UpperBound(page, key);
            if (page.Leaf) return at < page.Count && page.Keys[at] == key ? page.Values[at] : (int?)null;
            page = page.Children[at];
        }
        return null;
    }

    internal PersistentBTree Insert(int key, int value)
    {
        int allocated = 0;
        if (root == null)
        {
            allocated++;
            return new PersistentBTree(NewLeaf(new[] { key }, new[] { value }, 1, 0), 1, allocated);
        }
        InsertResult result = Insert(root, key, value, ref allocated, out bool added);
        if (!result.Split)
        {
            return new PersistentBTree(result.Page, Size + (added ? 1 : 0), allocated);
        }
        allocated++;
        var promoted = new int[] { result.Separator };
        var children = new[] { result.Page, result.Right };
        return new PersistentBTree(new BTreePage(promoted, children, 1), Size + (added ? 1 : 0), allocated);
    }

    private struct InsertResult
    {
        internal BTreePage Page;
        internal bool Split;
        internal BTreePage Right;
        internal int Separator;
    }

    private static InsertResult Insert(BTreePage page, int key, int value, ref int allocated, out bool added)
    {
        // A page's child i covers [Keys[i-1], Keys[i]), and leaf separators are copies of the first
        // key of the right page, so the internal descent is the upper bound of the key.
        int at = page.Leaf ? LowerBound(page, key) : UpperBound(page, key);
        if (page.Leaf && at < page.Count && page.Keys[at] == key)
        {
            added = false;                                 // value update: copy just this leaf
            allocated++;
            var updatedValues = Copy(page.Values, page.Count);
            updatedValues[at] = value;
            return new InsertResult { Page = NewLeaf(Copy(page.Keys, page.Count), updatedValues, page.Count, 0) };
        }
        if (page.Leaf)
        {
            added = true;
            int count = page.Count + 1;
            allocated++;
            return InsertLeaf(page, at, key, value, count, ref allocated);
        }
        InsertResult child = Insert(page.Children[at], key, value, ref allocated, out added);
        if (!child.Split)
        {
            allocated++;
            var children = CopyChildren(page, page.Count + 1);
            children[at] = child.Page;
            return new InsertResult { Page = new BTreePage(Copy(page.Keys, page.Count), children, page.Count) };
        }
        int grown = page.Count + 1;
        allocated++;
        var keys = new int[grown];
        var kids = new BTreePage[grown + 1];
        for (int i = 0; i < at; i++)
        {
            keys[i] = page.Keys[i];
            kids[i] = page.Children[i];
        }
        keys[at] = child.Separator;
        kids[at] = child.Page;
        kids[at + 1] = child.Right;
        for (int i = at + 1; i < grown; i++)
        {
            keys[i] = page.Keys[i - 1];
            kids[i + 1] = page.Children[i];
        }
        if (grown <= Capacity)
        {
            return new InsertResult { Page = new BTreePage(keys, kids, grown) };
        }
        int leftKeys = grown / 2;
        int rightKeys = grown - leftKeys - 1;
        allocated += 2;
        var leftKeyArray = new int[leftKeys];
        var leftChildren = new BTreePage[leftKeys + 1];
        for (int i = 0; i < leftKeys; i++) leftKeyArray[i] = keys[i];
        for (int i = 0; i <= leftKeys; i++) leftChildren[i] = kids[i];
        var rightKeyArray = new int[rightKeys];
        var rightChildren = new BTreePage[rightKeys + 1];
        for (int i = 0; i < rightKeys; i++) rightKeyArray[i] = keys[leftKeys + 1 + i];
        for (int i = 0; i <= rightKeys; i++) rightChildren[i] = kids[leftKeys + 1 + i];
        return new InsertResult
        {
            Page = new BTreePage(leftKeyArray, leftChildren, leftKeys),
            Split = true,
            Separator = keys[leftKeys],
            Right = new BTreePage(rightKeyArray, rightChildren, rightKeys),
        };
    }

    private static InsertResult InsertLeaf(BTreePage page, int at, int key, int value, int count, ref int allocated)
    {
        var keys = new int[count];
        var values = new int[count];
        for (int i = 0; i < at; i++)
        {
            keys[i] = page.Keys[i];
            values[i] = page.Values[i];
        }
        keys[at] = key;
        values[at] = value;
        for (int i = at + 1; i < count; i++)
        {
            keys[i] = page.Keys[i - 1];
            values[i] = page.Values[i - 1];
        }
        if (count <= Capacity) return new InsertResult { Page = NewLeaf(keys, values, count, 0) };
        int leftCount = count / 2;
        int rightCount = count - leftCount;
        allocated += 2;
        return new InsertResult
        {
            Page = NewLeaf(keys, values, leftCount, 0),
            Split = true,
            Separator = keys[leftCount],
            Right = NewLeaf(keys, values, rightCount, leftCount),
        };
    }

    /** A leaf that reuses the tail of a shared array instead of copying it twice. */
    private static BTreePage NewLeaf(int[] keys, int[] values, int count, int offset)
    {
        if (offset == 0) return new BTreePage(keys, values, count);
        var trimmedKeys = new int[count];
        var trimmedValues = new int[count];
        Array.Copy(keys, offset, trimmedKeys, 0, count);
        Array.Copy(values, offset, trimmedValues, 0, count);
        return new BTreePage(trimmedKeys, trimmedValues, count);
    }

    private static int[] Copy(int[] source, int count)
    {
        var copy = new int[count];
        Array.Copy(source, copy, count);
        return copy;
    }

    private static BTreePage[] CopyChildren(BTreePage page, int count)
    {
        var copy = new BTreePage[count];
        Array.Copy(page.Children, copy, count);
        return copy;
    }

    internal List<int[]> RangeScan(int low, int high)
    {
        var entries = new List<int[]>();
        if (low <= high) Collect(root, low, high, entries);
        return entries;
    }

    private static void Collect(BTreePage page, int low, int high, List<int[]> sink)
    {
        if (page == null) return;
        if (page.Leaf)
        {
            for (int i = 0; i < page.Count; i++)
            {
                if (page.Keys[i] >= low && page.Keys[i] <= high) sink.Add(new[] { page.Keys[i], page.Values[i] });
            }
            return;
        }
        for (int i = 0; i <= page.Count; i++)
        {
            bool needs = (i == 0 || page.Keys[i - 1] <= high) && (i == page.Count || page.Keys[i] > low);
            if (needs) Collect(page.Children[i], low, high, sink);
        }
    }

    internal List<int[]> Entries()
    {
        var entries = new List<int[]>();
        Collect(root, int.MinValue, int.MaxValue, entries);
        return entries;
    }

    /** Prototype delete: rebuild the entries without the key (documented in the header). */
    internal PersistentBTree Delete(int key)
    {
        if (Lookup(key) == null) return this;
        PersistentBTree fresh = Empty;
        foreach (int[] entry in Entries())
        {
            if (entry[0] != key) fresh = fresh.Insert(entry[0], entry[1]);
        }
        return fresh;
    }

    internal HashSet<BTreePage> Pages()
    {
        var set = new HashSet<BTreePage>(RefComparer.Instance);
        Gather(root, set);
        return set;
    }

    private static void Gather(BTreePage page, HashSet<BTreePage> sink)
    {
        if (page == null) return;
        sink.Add(page);
        foreach (BTreePage child in page.Children) Gather(child, sink);
    }

    /** B-tree invariant check over [low, high); returns the subtree height, or -1 when invalid.
     *  Bounds are inclusive-low / exclusive-high because a B+tree leaf split copies its separator up,
     *  so a separator may also appear as the smallest key of the right subtree. */
    internal static int ValidatePage(BTreePage page, bool isRoot, int low, int high)
    {
        if (page == null) return 0;
        if (page.Count > Capacity) return -1;
        if (!isRoot && page.Count < 1) return -1;
        for (int i = 0; i < page.Count; i++)
        {
            if (i > 0 && page.Keys[i - 1] >= page.Keys[i]) return -1;
            if (page.Keys[i] < low || page.Keys[i] >= high) return -1;
        }
        if (page.Leaf) return 1;
        if (page.Children.Length != page.Count + 1) return -1;
        int depth = -1;
        for (int i = 0; i <= page.Count; i++)
        {
            int childLow = i == 0 ? low : page.Keys[i - 1];
            int childHigh = i == page.Count ? high : page.Keys[i];
            int childDepth = ValidatePage(page.Children[i], false, childLow, childHigh);
            if (childDepth < 0) return -1;
            if (depth < 0) depth = childDepth;
            else if (depth != childDepth) return -1;
        }
        return depth + 1;
    }

    internal int Validate()
    {
        if (root == null) return Size == 0 ? 0 : -1;
        return ValidatePage(root, true, int.MinValue, int.MaxValue);
    }

    internal int PageCount() => Pages().Count;

    private static int LowerBound(BTreePage page, int key)
    {
        int lo = 0;
        int hi = page.Count;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (page.Keys[mid] < key) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    private static int UpperBound(BTreePage page, int key)
    {
        int lo = 0;
        int hi = page.Count;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (page.Keys[mid] <= key) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
}

public static class Solution
{
    private static ulong state = 0xDB14DB14UL;

    private static bool SameEntries(List<int[]> a, List<int[]> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++)
        {
            if (a[i][0] != b[i][0] || a[i][1] != b[i][1]) return false;
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    public static void Main()
    {
        var versions = new List<PersistentBTree> { PersistentBTree.Empty };
        var oracles = new List<List<int[]>> { new List<int[]>() };
        var keys = new List<int>();
        for (int i = 1; i <= 60; i++) keys.Add(i * 2);
        var order = new List<int>(keys);
        for (int i = order.Count - 1; i > 0; i--)
        {
            int j = (int)(SplitMix64.Next(ref state) % (ulong)(i + 1));
            (order[i], order[j]) = (order[j], order[i]);
        }

        int worstCost = 0;
        for (int i = 0; i < order.Count; i++)
        {
            PersistentBTree next = versions[i].Insert(order[i], order[i] + 1000);
            versions.Add(next);
            worstCost = Math.Max(worstCost, next.Allocated);
            var oracle = new List<int[]>(oracles[i]) { new int[] { order[i], order[i] + 1000 } };
            oracle.Sort((x, y) => x[0].CompareTo(y[0]));
            oracles.Add(oracle);
            Check(next.Size == i + 1, "size after insert " + i);
            Check(next.Validate() >= 1, "btree invariants after insert " + i);
            Check(next.Height() == next.Validate(), "height matches validation " + i);
            Check(next.Allocated <= 2 * (next.Height() + 1) + 2, "insert copies one path " + i);
            Check(next.Lookup(order[i]) == order[i] + 1000, "lookup of the new key " + i);
        }
        // Every version answers exactly like its own snapshot, and never like a later one.
        for (int i = 0; i < versions.Count; i++)
        {
            Check(versions[i].Size == i, "version size " + i);
            Check(SameEntries(versions[i].Entries(), oracles[i]), "version entries " + i);
            Check(versions[i].Validate() >= 0, "version invariants " + i);
        }
        Check(versions[17].Lookup(order[20]) == null, "old version lacks later key");
        Check(versions[21].Lookup(order[20]) == order[20] + 1000, "later version has that key");
        Check(versions[0].Lookup(2) == null && versions[0].Size == 0 && versions[0].Height() == 0,
            "empty version");

        // Deep version: three levels of pages, all leaves still at the same depth.
        PersistentBTree deep = versions[versions.Count - 1];
        Check(deep.Height() >= 3 && deep.Size == order.Count, "deep tree height " + deep.Height());
        // Range scans: full, boundary, empty, inverted, and a window that crosses pages.
        Check(SameEntries(deep.RangeScan(int.MinValue, int.MaxValue), oracles[oracles.Count - 1]),
            "full range");
        var lastOracle = oracles[oracles.Count - 1];
        int first = lastOracle[0][0];
        int last = lastOracle[lastOracle.Count - 1][0];
        Check(deep.RangeScan(first, first).Count == 1 && deep.RangeScan(first, first)[0][0] == first,
            "singleton range");
        Check(deep.RangeScan(last, last).Count == 1, "last key range");
        Check(deep.RangeScan(last + 1, last + 50).Count == 0, "range above all keys");
        Check(deep.RangeScan(int.MinValue, first - 1).Count == 0, "range below all keys");
        Check(deep.RangeScan(30, 20).Count == 0, "inverted range");
        var window = deep.RangeScan(20, 40);
        int expectedWindow = 0;
        foreach (int key in keys)
        {
            if (key >= 20 && key <= 40) expectedWindow++;
        }
        Check(window.Count == expectedWindow && window[0][0] == 20 && window[window.Count - 1][0] == 40,
            "window across pages");
        // Value updates keep the size and leave the earlier version untouched.
        PersistentBTree baseVersion = versions[30];
        PersistentBTree updated = baseVersion.Insert(order[5], -7);
        Check(updated.Size == baseVersion.Size && updated.Lookup(order[5]) == -7, "in place update");
        Check(baseVersion.Lookup(order[5]) == order[5] + 1000, "update leaves the old version alone");
        Check(updated.Validate() == baseVersion.Validate(), "update keeps the shape");
        Check(updated.Allocated <= baseVersion.Height() + 1, "update copies one path");

        // Structural sharing: the untouched pages really are the same objects.
        var before = versions[29].Pages();
        var after = versions[30].Pages();
        var shared = new HashSet<BTreePage>(before, RefComparer.Instance);
        shared.IntersectWith(after);
        Check(before.Count > 0 && after.Count > 0, "pages reachable");
        Check(shared.Count >= before.Count - versions[30].Height() - 1, "shared pages exceed the copied path");
        Check(versions[30].Allocated + shared.Count >= before.Count, "path copying touches few pages");

        // Prototype delete: rebuild without the key, keep the old version intact.
        PersistentBTree deleted = versions[40].Delete(order[3]);
        Check(deleted.Size == versions[40].Size - 1 && deleted.Lookup(order[3]) == null, "delete removes");
        Check(versions[40].Lookup(order[3]) == order[3] + 1000, "delete is versioned");
        Check(deleted.Validate() >= 1 && deleted.Size == versions[40].Size - 1,
            "delete keeps the index valid");
        Check(versions[40].Delete(999999).Size == versions[40].Size, "delete of a missing key");
        var afterDelete = new List<int[]>(oracles[40]);
        afterDelete.RemoveAll(entry => entry[0] == order[3]);
        Check(SameEntries(deleted.Entries(), afterDelete), "entries after delete");
        // Small shapes: one key, an exact leaf fill, and the first overflow.
        var single = PersistentBTree.Empty.Insert(5, 50);
        Check(single.Size == 1 && single.Height() == 1 && single.Lookup(5) == 50, "single key");
        var filled = PersistentBTree.Empty;
        for (int i = 1; i <= PersistentBTree.Capacity; i++) filled = filled.Insert(i, i);
        Check(filled.Height() == 1 && filled.Validate() == 1, "leaf exactly full");
        filled = filled.Insert(PersistentBTree.Capacity + 1, 99);
        Check(filled.Height() == 2 && filled.Validate() == 2 && filled.Size == PersistentBTree.Capacity + 1,
            "first leaf split");
        Check(filled.RangeScan(1, 99).Count == PersistentBTree.Capacity + 1, "split leaf scan");

        Console.WriteLine("  versions=" + versions.Count + " keys=" + keys.Count + " height=" + deep.Height()
            + " pages=" + deep.PageCount() + " worstInsertCost=" + worstCost
            + " sharedPages=" + shared.Count + "/" + before.Count);
        Console.WriteLine("PASS");
    }
}
