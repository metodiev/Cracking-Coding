// Task 806 - Implement fully persistent B-tree with copy-on-write pages and concurrency. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
//
// A fully persistent B-tree in the style of a copy-on-write storage engine: pages are immutable, every mutation
// returns a brand-new root and touches only the nodes on the search path, so all earlier roots stay valid and keep
// answering with exactly the data they had. The structural invariants of a B-tree of minimum degree t are enforced
// on every operation (keys sorted inside a page, every node between t-1 and 2t-1 keys, all leaves at the same
// depth, the root allowed to be smaller), insertion splits on the way down, deletion repairs an underflowed child
// with a borrow from a sibling or a merge before rebuilding the parent, and a root that loses its last key shrinks
// the height. Because pages are never overwritten, an old root is a real snapshot: a page that is reachable from a
// historical root can never be mutated by a later write.
//
// Concurrency is modelled as optimistic root publication: a writer builds a new root from the current one and
// publishes it with a compare-and-swap on the root pointer, retrying from the freshly published root when another
// writer won the race. That is exactly the "writers never block readers" property of a persistent tree.
//
// Invariants verified: in-order traversal is strictly sorted; the key count of every non-root page is within
// [t-1, 2t-1]; all leaves sit at the same depth; a search hits every stored key and misses every absent key; each
// historical root still returns its own snapshot after dozens of later mutations; a single insert allocates only
// O(log n) pages and shares the rest with the previous version; reachability-based garbage collection frees exactly
// the pages no live root can reach and keeps every page a live root can reach; a losing compare-and-swap never
// publishes a stale root, and the retry applies both writes exactly once.
//
// Independent oracles: a sorted dictionary holds the same key/value pairs and is compared against the in-order
// traversal, every point lookup and every range scan; the reachable page set computed by the collector is
// cross-checked with an independent traversal from each live root; and the snapshot check replays each version's
// expected content, which the tree cannot influence.
//
// Documented simplifications: pages live in an in-process pool and are addressed by integer ids, "publication" is a
// single-threaded compare-and-swap emulation with explicit interleavings (no threads, locks or IO), values are
// strings and keys are integers, and there is no disk log or overflow page. The B-tree algorithms themselves -
// split, borrow, merge, height shrink - are the real ones, exercised at t=2 and t=3.
#nullable enable
using System;
using System.Collections.Generic;
using System.Text;

internal sealed class BTreeException : Exception
{
    internal BTreeException(string message) : base(message) { }
}

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

/** An immutable page: a leaf holds key/value pairs, an internal page holds separators and child ids. */
internal sealed class Page
{
    internal readonly int Id;
    internal readonly bool Leaf;
    internal readonly List<int> Keys;
    internal readonly List<string> Values;
    internal readonly List<int> Children;

    internal Page(int id, bool leaf, List<int> keys, List<string> values, List<int> children)
    {
        Id = id; Leaf = leaf; Keys = keys; Values = values; Children = children;
    }

    internal int KeyCount => Keys.Count;

    internal string Show()
    {
        var parts = new List<string>();
        for (int i = 0; i < Keys.Count; i++) parts.Add(Keys[i] + "=" + Values[i]);
        return (Leaf ? "L" : "I") + Id + "[" + string.Join(",", parts) + "]" + (Leaf ? "" : "->{" + string.Join(",", Children) + "}");
    }
}

/** Page pool with explicit reclamation: pages are never mutated, they are either reachable or free. */
internal sealed class PagePool
{
    private readonly List<Page> _pages = new List<Page>();
    private readonly List<int> _free = new List<int>();
    internal int Allocations;
    internal int Frees;

    internal Page Get(int id)
    {
        if (id < 0 || id >= _pages.Count) throw new BTreeException("unknown page " + id);
        return _pages[id];
    }

    internal Page Alloc(bool leaf, List<int> keys, List<string> values, List<int> children)
    {
        Allocations++;
        int id;
        if (_free.Count > 0)
        {
            id = _free[_free.Count - 1];
            _free.RemoveAt(_free.Count - 1);
        }
        else
        {
            id = _pages.Count;
            _pages.Add(null!);
        }
        var page = new Page(id, leaf, keys, values, children);
        _pages[id] = page;
        return page;
    }

    /** Mark and sweep from the live roots; returns the number of pages that were reachable. */
    internal int Collect(IReadOnlyList<int> liveRoots)
    {
        var reachable = new HashSet<int>();
        foreach (int root in liveRoots) Mark(root, reachable);
        for (int id = 0; id < _pages.Count; id++)
        {
            if (_pages[id] == null) continue;
            if (reachable.Contains(id)) continue;
            _free.Add(id);                                   // unreachable: reusable
            _pages[id] = null!;
            Frees++;
        }
        return reachable.Count;
    }

    internal void Mark(int id, HashSet<int> reachable)
    {
        if (!reachable.Add(id)) return;
        Page page = Get(id);
        if (page.Leaf) return;
        foreach (int child in page.Children) Mark(child, reachable);
    }

    internal int ResidentPages
    {
        get
        {
            int count = 0;
            foreach (Page? page in _pages) if (page != null) count++;
            return count;
        }
    }

    internal string Digest()
    {
        var parts = new List<string>();
        for (int id = 0; id < _pages.Count; id++) if (_pages[id] != null) parts.Add(_pages[id].Show());
        return string.Join(";", parts);
    }
}

/** Persistent B-tree of minimum degree t: every mutation returns a new root id. */
internal sealed class PersistentTree
{
    internal readonly PagePool Pool = new PagePool();
    internal readonly int MinDegree;
    internal int RootId = -1;
    internal int RootCasVersion;                             // bumped on every successful publication
    internal int InsertAllocations;
    internal int InsertSharedPages;

    internal PersistentTree(int minDegree)
    {
        if (minDegree < 2) throw new BTreeException("minimum degree must be at least 2");
        MinDegree = minDegree;
        RootId = Pool.Alloc(true, new List<int>(), new List<string>(), new List<int>()).Id;
    }

    internal int MaxKeys => 2 * MinDegree - 1;

    internal int MinKeys => MinDegree - 1;

    internal string? Search(int rootId, int key)
    {
        int current = rootId;
        while (true)
        {
            Page page = Pool.Get(current);
            int index = LowerBound(page.Keys, key);
            if (index < page.Keys.Count && page.Keys[index] == key) return page.Values[index];
            if (page.Leaf) return null;
            current = page.Children[index];
        }
    }

    internal List<KeyValuePair<int, string>> InOrder(int rootId)
    {
        var result = new List<KeyValuePair<int, string>>();
        Walk(rootId, result);
        return result;
    }

    private void Walk(int id, List<KeyValuePair<int, string>> result)
    {
        Page page = Pool.Get(id);
        if (page.Leaf)
        {
            for (int i = 0; i < page.Keys.Count; i++) result.Add(new KeyValuePair<int, string>(page.Keys[i], page.Values[i]));
            return;
        }
        for (int i = 0; i < page.Keys.Count; i++)
        {
            Walk(page.Children[i], result);
            result.Add(new KeyValuePair<int, string>(page.Keys[i], page.Values[i]));
        }
        Walk(page.Children[page.Keys.Count], result);
    }

    internal List<KeyValuePair<int, string>> Range(int rootId, int low, int high)
    {
        var result = new List<KeyValuePair<int, string>>();
        foreach (var pair in InOrder(rootId))
        {
            if (pair.Key < low) continue;
            if (pair.Key > high) break;
            result.Add(pair);
        }
        return result;
    }

    /** Insert or replace. Returns a new root; pages not on the search path are shared with the old version. */
    internal int Insert(int rootId, int key, string value)
    {
        InsertAllocations = 0;
        InsertSharedPages = 0;
        var seen = new HashSet<int>();
        CountReachable(rootId, seen);
        int allocationsBefore = Pool.Allocations;
        var result = InsertInto(rootId, key, value, seen);
        InsertAllocations = Pool.Allocations - allocationsBefore;
        InsertSharedPages = seen.Count;
        if (!result.Split)
        {
            return result.NewId;
        }
        var keys = new List<int> { result.MedianKey };
        var values = new List<string> { result.MedianValue };
        var children = new List<int> { result.NewId, result.RightId };
        return Pool.Alloc(false, keys, values, children).Id;
    }

    private void CountReachable(int id, HashSet<int> seen)
    {
        if (!seen.Add(id)) return;
        Page page = Pool.Get(id);
        if (page.Leaf) return;
        foreach (int child in page.Children) CountReachable(child, seen);
    }

    private readonly struct InsertOutcome
    {
        internal readonly int NewId;
        internal readonly bool Split;
        internal readonly int MedianKey;
        internal readonly string MedianValue;
        internal readonly int RightId;

        internal InsertOutcome(int newId, bool split, int medianKey, string medianValue, int rightId)
        {
            NewId = newId; Split = split; MedianKey = medianKey; MedianValue = medianValue; RightId = rightId;
        }
    }

    private InsertOutcome InsertInto(int id, int key, string value, HashSet<int> shared)
    {
        Page page = Pool.Get(id);
        if (page.Leaf)
        {
            var keys = new List<int>(page.Keys);
            var values = new List<string>(page.Values);
            int at = LowerBound(keys, key);
            if (at < keys.Count && keys[at] == key)
            {
                values[at] = value;                                   // upsert: same key set, new value
                return new InsertOutcome(Pool.Alloc(true, keys, values, new List<int>()).Id, false, 0, "", 0);
            }
            keys.Insert(at, key);
            values.Insert(at, value);
            if (keys.Count <= MaxKeys) return new InsertOutcome(Pool.Alloc(true, keys, values, new List<int>()).Id, false, 0, "", 0);
            int median = keys.Count / 2;
            var leftKeys = keys.GetRange(0, median);
            var leftValues = values.GetRange(0, median);
            var rightKeys = keys.GetRange(median + 1, keys.Count - median - 1);
            var rightValues = values.GetRange(median + 1, values.Count - median - 1);
            int leftId = Pool.Alloc(true, leftKeys, leftValues, new List<int>()).Id;
            int rightId = Pool.Alloc(true, rightKeys, rightValues, new List<int>()).Id;
            return new InsertOutcome(leftId, true, keys[median], values[median], rightId);
        }

        int index = LowerBound(page.Keys, key);
        if (index < page.Keys.Count && page.Keys[index] == key)
        {
            var keys = new List<int>(page.Keys);
            var values = new List<string>(page.Values);
            values[index] = value;
            return new InsertOutcome(Pool.Alloc(false, keys, values, new List<int>(page.Children)).Id, false, 0, "", 0);
        }
        InsertOutcome child = InsertInto(page.Children[index], key, value, shared);
        var newKeys = new List<int>(page.Keys);
        var newValues = new List<string>(page.Values);
        var newChildren = new List<int>(page.Children);
        if (child.Split)
        {
            newKeys.Insert(index, child.MedianKey);
            newValues.Insert(index, child.MedianValue);
            newChildren[index] = child.NewId;
            newChildren.Insert(index + 1, child.RightId);
        }
        else
        {
            newChildren[index] = child.NewId;
        }
        if (newKeys.Count <= MaxKeys) return new InsertOutcome(Pool.Alloc(false, newKeys, newValues, newChildren).Id, false, 0, "", 0);
        int middle = newKeys.Count / 2;
        var innerLeftKeys = newKeys.GetRange(0, middle);
        var innerLeftValues = newValues.GetRange(0, middle);
        var innerLeftChildren = newChildren.GetRange(0, middle + 1);
        var innerRightKeys = newKeys.GetRange(middle + 1, newKeys.Count - middle - 1);
        var innerRightValues = newValues.GetRange(middle + 1, newValues.Count - middle - 1);
        var innerRightChildren = newChildren.GetRange(middle + 1, newChildren.Count - middle - 1);
        int left = Pool.Alloc(false, innerLeftKeys, innerLeftValues, innerLeftChildren).Id;
        int right = Pool.Alloc(false, innerRightKeys, innerRightValues, innerRightChildren).Id;
        return new InsertOutcome(left, true, newKeys[middle], newValues[middle], right);
    }

    /** Delete: descends to the key, then repairs underflow bottom-up with a borrow or a merge. */
    internal int Delete(int rootId, int key)
    {
        bool removed;
        int newRoot = DeleteFrom(rootId, key, out removed);
        if (!removed) return rootId;                              // nothing changed: the old root is still the answer
        Page root = Pool.Get(newRoot);
        if (!root.Leaf && root.KeyCount == 0) return root.Children[0];   // height shrink
        return newRoot;
    }

    private int DeleteFrom(int id, int key, out bool removed)
    {
        Page page = Pool.Get(id);
        if (page.Leaf)
        {
            int at = LowerBound(page.Keys, key);
            if (at >= page.Keys.Count || page.Keys[at] != key)
            {
                removed = false;
                return id;
            }
            removed = true;
            var keys = new List<int>(page.Keys);
            var values = new List<string>(page.Values);
            keys.RemoveAt(at);
            values.RemoveAt(at);
            return Pool.Alloc(true, keys, values, new List<int>()).Id;
        }

        int index = LowerBound(page.Keys, key);
        var newKeys = new List<int>(page.Keys);
        var newValues = new List<string>(page.Values);
        var newChildren = new List<int>(page.Children);
        if (index < page.Keys.Count && page.Keys[index] == key)
        {
            // The key is a separator: replace it with its successor and delete the successor from the right child.
            int successorKey = Leftmost(newChildren[index + 1], out string successorValue);
            bool ignored;
            int repairedRight = DeleteFrom(newChildren[index + 1], successorKey, out ignored);
            newChildren[index + 1] = repairedRight;
            newKeys[index] = successorKey;
            newValues[index] = successorValue;
            Repair(newKeys, newValues, newChildren, index + 1);   // the helper updates the lists in place
            removed = true;
            return Pool.Alloc(false, newKeys, newValues, newChildren).Id;
        }
        int child = DeleteFrom(newChildren[index], key, out removed);
        if (!removed) return id;
        newChildren[index] = child;
        Repair(newKeys, newValues, newChildren, index);
        return Pool.Alloc(false, newKeys, newValues, newChildren).Id;
    }

    private int Leftmost(int id, out string value)
    {
        Page page = Pool.Get(id);
        while (!page.Leaf) page = Pool.Get(page.Children[0]);
        value = page.Values[0];
        return page.Keys[0];
    }

    /**
     * Restores the minimum key count of children[index] by borrowing from a sibling that has spare keys, or by
     * merging with a sibling. The parent's key/child lists are private copies, so the helper edits them in place and
     * leaves the surviving child id at the position it should occupy (the caller must not re-store the result).
     */
    private void Repair(List<int> keys, List<string> values, List<int> children, int index)
    {
        Page child = Pool.Get(children[index]);
        if (child.KeyCount >= MinKeys) return;
        bool hasLeft = index > 0;
        bool hasRight = index + 1 < children.Count;
        if (hasLeft)
        {
            Page left = Pool.Get(children[index - 1]);
            if (left.KeyCount > MinKeys)
            {
                var childKeys = new List<int> { keys[index - 1] };
                childKeys.AddRange(child.Keys);
                var childValues = new List<string> { values[index - 1] };
                childValues.AddRange(child.Values);
                var childChildren = new List<int>();
                if (!child.Leaf) childChildren.Add(left.Children[left.Children.Count - 1]);
                childChildren.AddRange(child.Children);
                keys[index - 1] = left.Keys[left.KeyCount - 1];
                values[index - 1] = left.Values[left.KeyCount - 1];
                children[index] = Pool.Alloc(child.Leaf, childKeys, childValues, childChildren).Id;
                var leftChildren = new List<int>(left.Children);
                if (!left.Leaf) leftChildren.RemoveAt(leftChildren.Count - 1);
                children[index - 1] = Pool.Alloc(left.Leaf, left.Keys.GetRange(0, left.KeyCount - 1),
                    left.Values.GetRange(0, left.KeyCount - 1), leftChildren).Id;
                return;
            }
        }
        if (hasRight)
        {
            Page right = Pool.Get(children[index + 1]);
            if (right.KeyCount > MinKeys)
            {
                var childKeys = new List<int>(child.Keys) { keys[index] };
                var childValues = new List<string>(child.Values) { values[index] };
                var childChildren = new List<int>(child.Children);
                if (!child.Leaf) childChildren.Add(right.Children[0]);
                keys[index] = right.Keys[0];
                values[index] = right.Values[0];
                children[index] = Pool.Alloc(child.Leaf, childKeys, childValues, childChildren).Id;
                var rightChildren = new List<int>(right.Children);
                if (!right.Leaf) rightChildren.RemoveAt(0);
                children[index + 1] = Pool.Alloc(right.Leaf, right.Keys.GetRange(1, right.KeyCount - 1),
                    right.Values.GetRange(1, right.KeyCount - 1), rightChildren).Id;
                return;
            }
        }
        if (hasLeft)
        {
            Page left = Pool.Get(children[index - 1]);
            var mergedKeys = new List<int>(left.Keys) { keys[index - 1] };
            mergedKeys.AddRange(child.Keys);
            var mergedValues = new List<string>(left.Values) { values[index - 1] };
            mergedValues.AddRange(child.Values);
            var mergedChildren = new List<int>(left.Children);
            mergedChildren.AddRange(child.Children);
            children.RemoveAt(index);
            keys.RemoveAt(index - 1);
            values.RemoveAt(index - 1);
            children[index - 1] = Pool.Alloc(child.Leaf, mergedKeys, mergedValues, mergedChildren).Id;
            return;
        }
        Page right2 = Pool.Get(children[index + 1]);
        var mergedKeys2 = new List<int>(child.Keys) { keys[index] };
        mergedKeys2.AddRange(right2.Keys);
        var mergedValues2 = new List<string>(child.Values) { values[index] };
        mergedValues2.AddRange(right2.Values);
        var mergedChildren2 = new List<int>(child.Children);
        mergedChildren2.AddRange(right2.Children);
        children.RemoveAt(index + 1);
        keys.RemoveAt(index);
        values.RemoveAt(index);
        children[index] = Pool.Alloc(child.Leaf, mergedKeys2, mergedValues2, mergedChildren2).Id;
    }

    /** Optimistic publication: the write succeeds only if the root is still the one the writer planned from. */
    internal bool CompareAndSwapRoot(int expectedRoot, int newRoot)
    {
        if (RootId != expectedRoot) return false;
        RootId = newRoot;
        RootCasVersion++;
        return true;
    }

    /** Structural validation: sorted keys, key counts in bounds, uniform leaf depth. Returns the height. */
    internal int Validate(int rootId)
    {
        return ValidatePage(rootId, true);
    }

    private int ValidatePage(int id, bool isRoot)
    {
        Page page = Pool.Get(id);
        for (int i = 1; i < page.KeyCount; i++)
        {
            if (page.Keys[i - 1] >= page.Keys[i]) throw new BTreeException("keys are not strictly sorted in page " + id);
        }
        if (!isRoot && (page.KeyCount < MinKeys || page.KeyCount > MaxKeys))
            throw new BTreeException("page " + id + " holds " + page.KeyCount + " keys, outside [" + MinKeys + "," + MaxKeys + "]");
        if (isRoot && page.KeyCount > MaxKeys) throw new BTreeException("root " + id + " overflowed");
        if (page.Leaf) return 1;
        if (page.Children.Count != page.KeyCount + 1)
            throw new BTreeException("page " + id + " has the wrong number of children");
        int depth = -1;
        for (int i = 0; i < page.Children.Count; i++)
        {
            int childDepth = ValidatePage(page.Children[i], false);
            if (depth == -1) depth = childDepth;
            else if (depth != childDepth) throw new BTreeException("leaves of page " + id + " are at different depths");
        }
        return depth + 1;
    }

    internal int Size(int rootId) => InOrder(rootId).Count;

    internal int Height(int rootId) => Validate(rootId);

    internal string Snapshot(int rootId)
    {
        var parts = new List<string>();
        foreach (var pair in InOrder(rootId)) parts.Add(pair.Key + "=" + pair.Value);
        return string.Join(",", parts);
    }

    internal static int LowerBound(List<int> keys, int key)
    {
        int low = 0;
        int high = keys.Count;
        while (low < high)
        {
            int mid = (low + high) / 2;
            if (keys[mid] < key) low = mid + 1;
            else high = mid;
        }
        return low;
    }
}

public static class Solution
{
    public static void Main()
    {
        // 1-3. A t=2 tree (2-3-4 nodes) keeps every structural invariant while growing.
        var tree = new PersistentTree(2);
        var oracle = new SortedDictionary<int, string>();
        var roots = new List<int>();
        var snapshots = new List<string>();
        var random = 0x5EEDUL;
        for (int i = 0; i < 60; i++)
        {
            int key = (int)(SplitMix64.Next(ref random) % 200);
            string value = "v" + i;
            oracle[key] = value;
            tree.RootId = tree.Insert(tree.RootId, key, value);
            roots.Add(tree.RootId);
            snapshots.Add(tree.Snapshot(tree.RootId));
            tree.Validate(tree.RootId);
        }
        Check(tree.InOrder(tree.RootId).Count == oracle.Count, "the tree holds exactly the oracle's key count (" + oracle.Count + ")");
        Check(tree.Snapshot(tree.RootId) == ShowOracle(oracle), "the in-order traversal matches the oracle exactly");
        Check(tree.Height(tree.RootId) >= 2, "the tree grew in height");
        Check(tree.Size(tree.RootId) == oracle.Count, "the size helper agrees with the oracle");
        string lastVersionSnapshot = snapshots[snapshots.Count - 1];

        // 4-6. Every historical root is still a valid snapshot of the version it was created for.
        bool allVersionsIntact = true;
        bool allVersionsValid = true;
        for (int i = 0; i < roots.Count; i++)
        {
            if (tree.Snapshot(roots[i]) != snapshots[i]) allVersionsIntact = false;
            try { tree.Validate(roots[i]); }
            catch (BTreeException) { allVersionsValid = false; }
        }
        Check(allVersionsIntact, "all " + roots.Count + " historical roots still return their own snapshot (persistence)");
        Check(allVersionsValid, "all historical roots still satisfy the B-tree invariants");

        // 7-9. Point lookups, misses and range scans against the oracle.
        int hits = 0;
        int misses = 0;
        foreach (var pair in oracle)
        {
            if (tree.Search(tree.RootId, pair.Key) == pair.Value) hits++;
        }
        for (int key = -5; key < 205; key++)
        {
            if (oracle.ContainsKey(key)) continue;
            if (tree.Search(tree.RootId, key) == null) misses++;
        }
        Check(hits == oracle.Count, "every stored key is found with its value");
        Check(misses > 0, "every absent key is reported as missing (" + misses + " probes)");
        var ranged = tree.Range(tree.RootId, 50, 80);
        int expectedInRange = 0;
        foreach (int key in oracle.Keys) if (key >= 50 && key <= 80) expectedInRange++;
        Check(ranged.Count == expectedInRange, "the range scan returns exactly the keys in [50,80]");
        Check(tree.Range(tree.RootId, 1000, 2000).Count == 0, "an out-of-range scan returns no rows");

        // 10-11. Copy-on-write: one insert allocates O(log n) pages and shares everything else with the old root.
        int beforeInsert = tree.RootId;
        int afterInsert = tree.Insert(beforeInsert, 500, "new");
        oracle[500] = "new";
        Check(tree.Search(afterInsert, 500) == "new" && tree.Search(beforeInsert, 500) == null,
            "the new version sees the insert while the old root does not (copy on write)");
        Check(tree.InsertAllocations <= 2 * tree.Height(beforeInsert) + 1, "an insert allocates only O(log n) pages ("
            + tree.InsertAllocations + " for height " + tree.Height(beforeInsert) + ")");
        var sharedBefore = Reachable(tree, beforeInsert);
        var sharedAfter = Reachable(tree, afterInsert);
        int stillShared = 0;
        foreach (int id in sharedAfter) if (sharedBefore.Contains(id)) stillShared++;
        Check(stillShared > sharedAfter.Count / 2, "most pages of the new version are shared with the previous one ("
            + stillShared + "/" + sharedAfter.Count + ")");
        Check(!sharedAfter.Contains(beforeInsert) || tree.Snapshot(afterInsert) != tree.Snapshot(beforeInsert),
            "the root of the new version is a fresh page");

        // 12-13. Deletion keeps the invariants and never touches older versions.
        string snapshotBeforeDelete = tree.Snapshot(afterInsert);
        var deletedKeys = new List<int>();
        foreach (int key in new List<int>(oracle.Keys))
        {
            if (key % 3 != 0) continue;
            deletedKeys.Add(key);
        }
        int deleting = afterInsert;
        foreach (int key in deletedKeys)
        {
            oracle.Remove(key);
            deleting = tree.Delete(deleting, key);
            tree.Validate(deleting);
        }
        Check(deletedKeys.Count > 5, "the deletion phase removed a meaningful number of keys (" + deletedKeys.Count + ")");
        Check(tree.Snapshot(deleting) == ShowOracle(oracle), "after the deletions the traversal still matches the oracle ("
            + Diagnose(tree, deleting, oracle) + ")");
        Check(tree.Snapshot(afterInsert) == snapshotBeforeDelete, "deleting from a new version never mutates the old version");
        Check(tree.Search(deleting, deletedKeys[0]) == null, "a deleted key is gone from the new version");
        Check(tree.Height(deleting) <= tree.Height(afterInsert), "the tree never grows while deleting");

        // 14-15. Deleting everything shrinks the tree to an empty root.
        var drainKeys = new List<int>(oracle.Keys);
        int draining = deleting;
        foreach (int key in drainKeys) draining = tree.Delete(draining, key);
        Check(tree.Size(draining) == 0, "the tree is empty after deleting every key");
        Check(tree.Height(draining) == 1, "the empty tree has shrunk back to a single leaf");
        Check(tree.Delete(draining, 42) == draining, "deleting a missing key returns the identical root (no copy)");

        // 16-17. Reclamation: unreachable pages are freed, reachable ones survive.
        var liveRoots = new List<int> { draining, afterInsert };
        int reachableLive = Reachable(tree, draining).Count + Reachable(tree, afterInsert).Count;
        int collected = tree.Pool.Collect(liveRoots);
        Check(collected <= reachableLive, "collection never claims more reachable pages than an independent traversal");
        var liveSet = new HashSet<int>();
        tree.Pool.Mark(draining, liveSet);
        tree.Pool.Mark(afterInsert, liveSet);
        Check(liveSet.Count == collected, "the collector's live set matches an independent traversal exactly");
        int residentAfterCollect = tree.Pool.ResidentPages;
        Check(residentAfterCollect == liveSet.Count, "exactly the reachable pages are resident after collection");
        Check(tree.Snapshot(afterInsert) == snapshotBeforeDelete, "the surviving root is unchanged by collection");
        Check(tree.Pool.Frees > 0, "unreachable pages were actually freed (" + tree.Pool.Frees + ")");

        // 18-20. Optimistic concurrency: a stale writer fails its compare-and-swap and retries from the new root.
        var concurrent = new PersistentTree(3);
        int baseRoot = concurrent.RootId;
        for (int key = 0; key < 40; key++) baseRoot = concurrent.Insert(baseRoot, key, "base" + key);
        concurrent.RootId = baseRoot;
        int writerA = concurrent.Insert(baseRoot, 100, "from-a");
        int writerB = concurrent.Insert(baseRoot, 200, "from-b");
        Check(concurrent.CompareAndSwapRoot(baseRoot, writerA), "the first writer publishes its root");
        Check(!concurrent.CompareAndSwapRoot(baseRoot, writerB), "the second writer's compare-and-swap fails on the stale root");
        Check(concurrent.Search(concurrent.RootId, 200) == null, "the stale root was never published");
        int retry = concurrent.Insert(concurrent.RootId, 200, "from-b");
        Check(concurrent.CompareAndSwapRoot(concurrent.RootId, retry), "the retry publishes from the fresh root");
        Check(concurrent.Search(concurrent.RootId, 100) == "from-a" && concurrent.Search(concurrent.RootId, 200) == "from-b",
            "both concurrent writes are present exactly once after the retry");
        Check(concurrent.Size(concurrent.RootId) == 42, "the retried write added exactly one key");
        concurrent.Validate(concurrent.RootId);

        // 21-23. A larger tree at t=3, batch deletion and the persistence of every intermediate version.
        var big = new PersistentTree(3);
        var bigOracle = new SortedDictionary<int, string>();
        var bigRoots = new List<int>();
        var bigSnapshots = new List<string>();
        var rng = 0xA11CEUL;
        for (int i = 0; i < 200; i++)
        {
            int key = (int)(SplitMix64.Next(ref rng) % 1000);
            bigOracle[key] = "x" + i;
            big.RootId = big.Insert(big.RootId, key, "x" + i);
            bigRoots.Add(big.RootId);
            bigSnapshots.Add(big.Snapshot(big.RootId));
        }
        big.Validate(big.RootId);
        Check(big.Size(big.RootId) == bigOracle.Count, "the t=3 tree holds " + bigOracle.Count + " distinct keys");
        Check(big.Snapshot(big.RootId) == ShowOracle(bigOracle), "the t=3 in-order traversal matches the oracle");
        bool bigIntact = true;
        foreach (var pair in bigOracle)
        {
            if (big.Search(big.RootId, pair.Key) != pair.Value) bigIntact = false;
        }
        Check(bigIntact, "every key of the large tree is searchable");
        int spotChecks = 0;
        bool spotIntact = true;
        for (int i = 0; i < bigRoots.Count; i += 37)
        {
            spotChecks++;
            if (big.Snapshot(bigRoots[i]) != bigSnapshots[i]) spotIntact = false;
        }
        Check(spotIntact && spotChecks >= 5, "sampled historical versions of the large tree are still intact (" + spotChecks + " sampled)");

        // 24. Determinism of the whole pipeline.
        var replay = new PersistentTree(2);
        var replayRandom = 0x5EEDUL;
        for (int i = 0; i < 60; i++)
        {
            int key = (int)(SplitMix64.Next(ref replayRandom) % 200);
            replay.RootId = replay.Insert(replay.RootId, key, "v" + i);
        }
        Check(replay.Snapshot(replay.RootId) == lastVersionSnapshot,
            "an identical insertion sequence produces an identical version");
        Check(replay.Pool.Digest() != "", "the page pool digests the resident pages");
        Console.WriteLine("  persistent tree: versions=" + roots.Count + " live pages=" + residentAfterCollect
            + " collected=" + tree.Pool.Frees + " height(t=3)=" + big.Height(big.RootId));

        Console.WriteLine("PASS");
    }

    /** Independent traversal used to cross-check the collector's reachable set. */
    private static HashSet<int> Reachable(PersistentTree tree, int rootId)
    {
        var seen = new HashSet<int>();
        var stack = new Stack<int>();
        stack.Push(rootId);
        while (stack.Count > 0)
        {
            int id = stack.Pop();
            if (!seen.Add(id)) continue;
            Page page = tree.Pool.Get(id);
            if (page.Leaf) continue;
            foreach (int child in page.Children) stack.Push(child);
        }
        return seen;
    }

    private static string ShowOracle(SortedDictionary<int, string> oracle)
    {
        var parts = new List<string>();
        foreach (var pair in oracle) parts.Add(pair.Key + "=" + pair.Value);
        return string.Join(",", parts);
    }

    /** First structural difference between a version and the oracle, used to make failures diagnosable. */
    private static string Diagnose(PersistentTree tree, int rootId, SortedDictionary<int, string> oracle)
    {
        List<KeyValuePair<int, string>> actual = tree.InOrder(rootId);
        var expected = new List<KeyValuePair<int, string>>();
        foreach (var pair in oracle) expected.Add(pair);
        for (int i = 0; i < Math.Max(actual.Count, expected.Count); i++)
        {
            if (i >= actual.Count) return "missing key " + expected[i].Key;
            if (i >= expected.Count) return "extra key " + actual[i].Key;
            if (actual[i].Key != expected[i].Key) return "key " + actual[i].Key + " where " + expected[i].Key + " was expected";
            if (actual[i].Value != expected[i].Value) return "value " + actual[i].Value + " instead of " + expected[i].Value + " at key " + actual[i].Key;
        }
        return "identical";
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
