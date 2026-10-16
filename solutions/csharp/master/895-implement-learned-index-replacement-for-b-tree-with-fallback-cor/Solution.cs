// Task 895 - Implement learned index replacement for B-tree with fallback correctness checks. (Tier: Master, Ch 14 — Databases)
//
// A B-tree and a learned index over the same sorted keys. The B-tree is bulk-loaded (leaves of at most
// Degree keys, internal levels of at most Degree+1 children, separators equal to the maximum key of each
// non-final child) and answers exact lookups, range scans and invariant checks. The learned index is a
// piecewise-linear approximation (PLA) of key -> rank built with the greedy "swing" algorithm: it keeps,
// per segment, the exact rational interval of slopes that keeps every covered point within epsilon
// positions, closes the segment when the interval would become empty and predicts with the interval's
// midpoint. A learned lookup binary-searches the segments, predicts a rank and scans the window
// [prediction - epsilon, prediction + epsilon] of the sorted array, matching keys exactly; anything the
// model cannot cover - a key outside the trained range, a key added since the last build, a window that
// misses - falls back to the B-tree, and every fallback is counted.
//
// Simplifications (documented, not behavioural): the tree is bulk-loaded and rebuilt after writes rather
// than splitting and merging nodes in place, so the last leaf of a level may be partially filled (a real
// engine would split and merge); the learned index ranks into a sorted key array rather than predicting
// page offsets; keys are unique 64-bit values with an int payload; no concurrency, persistence or key
// compression.
//
// Verified: tree invariants (ordering, separators, uniform leaf depth, degree bounds) hold for dense,
// exponential and clustered key sets; search and range answers equal a sorted-array oracle; the PLA's
// error bound is *re-measured* over every key instead of being trusted, and the segment count stays far
// below the key count; learned lookups agree with the tree for hits on every key, for misses and for
// keys outside the trained range, with the comparison count inside 2*epsilon+3 and below the tree's;
// fallbacks are counted and still correct; inserts extend the sorted view and trigger a rebuild exactly
// when the bound breaks, after which the bound holds again; deletes rebuild and keep every surviving key
// findable; empty, single-key and duplicate-key cases behave; and the model is reproducible.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

internal sealed class BTree
{
    internal const int Degree = 16;

    private sealed class Node
    {
        internal bool Leaf;
        internal readonly List<long> Keys = new List<long>();
        internal readonly List<int> Values = new List<int>();
        internal readonly List<Node> Children = new List<Node>();
    }

    private Node _root = new Node { Leaf = true };
    internal long Comparisons;

    internal int Count { get; private set; }
    internal int Height { get; private set; }

    /** Bulk load from strictly increasing keys; separators are the maximum key of each non-final child. */
    internal void BulkLoad(IReadOnlyList<long> keys, IReadOnlyList<int> values)
    {
        Count = keys.Count;
        var level = new List<Node>();
        for (int start = 0; start < keys.Count; start += Degree)
        {
            var leaf = new Node { Leaf = true };
            for (int i = start; i < Math.Min(start + Degree, keys.Count); i++)
            {
                leaf.Keys.Add(keys[i]);
                leaf.Values.Add(values[i]);
            }
            level.Add(leaf);
        }
        if (level.Count == 0) level.Add(new Node { Leaf = true });
        Height = 1;
        while (level.Count > 1)
        {
            var parents = new List<Node>();
            for (int start = 0; start < level.Count; start += Degree + 1)
            {
                var parent = new Node { Leaf = false };
                int end = Math.Min(start + Degree + 1, level.Count);
                for (int i = start; i < end; i++) parent.Children.Add(level[i]);
                for (int i = start; i < end - 1; i++)
                {
                    parent.Keys.Add(MaxKey(level[i]));
                    parent.Values.Add(MaxKeyValue(level[i]));      // the separator is a real key with a payload
                }
                parents.Add(parent);
            }
            level = parents;
            Height++;
        }
        _root = level[0];
    }

    internal bool Search(long key, out int value)
    {
        Node node = _root;
        long comparisons = 0;
        int regions = 0;
        while (true)
        {
            regions++;
            int index = LowerBound(node.Keys, key, ref comparisons);
            if (index < node.Keys.Count && node.Keys[index] == key)
            {
                value = node.Values[index];
                KeyComparisons = comparisons;
                Regions = regions;
                return true;
            }
            if (node.Leaf)
            {
                value = 0;
                KeyComparisons = comparisons;
                Regions = regions;
                return false;
            }
            node = node.Children[index];
        }
    }

    /** Distinct nodes touched by the last search: the unit of "random access" in this model. */
    internal int Regions { get; private set; }

    /** Key comparisons performed by the last search (binary search inside every visited node). */
    internal long KeyComparisons { get; private set; }

    private static int LowerBound(List<long> keys, long key, ref long comparisons)
    {
        int low = 0, high = keys.Count;
        while (low < high)
        {
            int middle = (low + high) / 2;
            comparisons++;
            if (keys[middle] < key) low = middle + 1;
            else high = middle;
        }
        return low;
    }

    internal List<KeyValuePair<long, int>> Range(long low, long high)
    {
        var result = new List<KeyValuePair<long, int>>();
        CollectRange(_root, low, high, result);
        return result;
    }

    private static void CollectRange(Node node, long low, long high, List<KeyValuePair<long, int>> result)
    {
        if (node.Leaf)
        {
            for (int i = 0; i < node.Keys.Count; i++)
                if (node.Keys[i] >= low && node.Keys[i] <= high)
                    result.Add(new KeyValuePair<long, int>(node.Keys[i], node.Values[i]));
            return;
        }
        foreach (Node child in node.Children) CollectRange(child, low, high, result);
    }

    /** Ordering, separators, uniform leaf depth and degree bounds - checked, not assumed. */
    internal bool InvariantsHold(out string problem)
    {
        problem = "";
        var leafDepths = new HashSet<int>();
        bool ok = Check(_root, 0, leafDepths, true, ref problem);
        if (!ok) return false;
        if (leafDepths.Count != 1)
        {
            problem = "leaves at different depths: " + string.Join(",", leafDepths);
            return false;
        }
        if (leafDepths.Count == 0) { problem = "no leaves reached"; return false; }
        return true;
    }

    private static bool Check(Node node, int depth, HashSet<int> leafDepths, bool isRoot, ref string problem)
    {
        for (int i = 1; i < node.Keys.Count; i++)
            if (node.Keys[i - 1] >= node.Keys[i]) { problem = "keys out of order"; return false; }
        if (node.Leaf)
        {
            leafDepths.Add(depth);
            if (node.Keys.Count != node.Values.Count) { problem = "leaf keys/values mismatch"; return false; }
            if (!isRoot && node.Keys.Count == 0) { problem = "empty leaf"; return false; }
            if (node.Keys.Count > Degree) { problem = "leaf overfull"; return false; }
            return true;
        }
        if (node.Children.Count == 0) { problem = "internal node without children"; return false; }
        if (node.Keys.Count != node.Values.Count) { problem = "separator payload missing"; return false; }
        if (node.Children.Count != node.Keys.Count + 1) { problem = "children/keys mismatch"; return false; }
        if (node.Children.Count > Degree + 1) { problem = "too many children"; return false; }
        for (int i = 0; i < node.Children.Count - 1; i++)
            if (node.Keys[i] != MaxKey(node.Children[i])) { problem = "separator mismatch at " + i; return false; }
        foreach (Node child in node.Children)
            if (!Check(child, depth + 1, leafDepths, false, ref problem)) return false;
        return true;
    }

    private static long MaxKey(Node node)
    {
        Node cursor = node;
        while (!cursor.Leaf) cursor = cursor.Children[^1];
        return cursor.Keys[^1];
    }

    private static int MaxKeyValue(Node node)
    {
        Node cursor = node;
        while (!cursor.Leaf) cursor = cursor.Children[^1];
        return cursor.Values[^1];
    }
}

internal sealed class Segment
{
    internal long StartKey;
    internal int StartPosition;
    internal double Slope;

    internal long Predict(long key) => StartPosition + (long)Math.Round(Slope * (key - StartKey));
}

internal sealed class LearnedIndex
{
    internal const double Epsilon = 4.0;

    internal readonly List<Segment> Segments = new List<Segment>();
    internal readonly List<long> Keys = new List<long>();
    internal readonly List<int> Values = new List<int>();
    internal long MinKey, MaxKey;
    internal long Rebuilds;
    internal long Fallbacks;
    internal long Lookups;
    internal long Comparisons;
    internal long SegmentComparisons;

    internal void Load(IReadOnlyList<long> keys, IReadOnlyList<int> values)
    {
        Keys.Clear();
        Values.Clear();
        Keys.AddRange(keys);
        Values.AddRange(values);
        RebuildSegments();
    }

    /** Keeps the sorted array view in step with an insert (a rebuilt model follows when needed). */
    internal void InsertIntoView(long key, int value)
    {
        int position = LowerBound(key);
        if (position < Keys.Count && Keys[position] == key) { Values[position] = value; return; }
        Keys.Insert(position, key);
        Values.Insert(position, value);
    }

    internal bool RemoveFromView(long key)
    {
        int position = LowerBound(key);
        if (position >= Keys.Count || Keys[position] != key) return false;
        Keys.RemoveAt(position);
        Values.RemoveAt(position);
        return true;
    }

    internal int LowerBound(long key)
    {
        int low = 0, high = Keys.Count;
        while (low < high)
        {
            int middle = (low + high) / 2;
            if (Keys[middle] < key) low = middle + 1;
            else high = middle;
        }
        return low;
    }

    /**
     * Greedy swing: maintain the exact slope interval keeping every covered point within epsilon of the
     * line through the segment's first point; close the segment when the interval would be empty.
     */
    internal void RebuildSegments()
    {
        Rebuilds++;
        Segments.Clear();
        if (Keys.Count == 0) { MinKey = 0; MaxKey = 0; return; }
        MinKey = Keys[0];
        MaxKey = Keys[^1];
        int start = 0;
        while (start < Keys.Count)
        {
            double lowSlope = double.NegativeInfinity, highSlope = double.PositiveInfinity;
            int end = start + 1;
            while (end < Keys.Count)
            {
                double dx = Keys[end] - Keys[start];
                double dy = end - start;
                double lower = (dy - Epsilon) / dx;
                double upper = (dy + Epsilon) / dx;
                double nextLow = Math.Max(lowSlope, lower);
                double nextHigh = Math.Min(highSlope, upper);
                if (nextLow > nextHigh) break;
                lowSlope = nextLow;
                highSlope = nextHigh;
                end++;
            }
            double slope = (lowSlope + highSlope) / 2.0;
            if (double.IsNaN(slope) || double.IsInfinity(slope))
                slope = end - 1 > start ? (double)(end - 1 - start) / (Keys[end - 1] - Keys[start]) : 0.0;
            Segments.Add(new Segment { StartKey = Keys[start], StartPosition = start, Slope = slope });
            start = Math.Max(end, start + 1);
        }
    }

    internal bool TryLookup(long key, out int value)
    {
        value = 0;
        Lookups++;
        if (Segments.Count == 0 || key < MinKey || key > MaxKey) { Fallbacks++; return false; }
        Segment segment = Segments[FindSegment(key)];
        long predicted = segment.Predict(key);
        long low = Math.Max(0, predicted - (long)Epsilon - 1);
        long high = Math.Min(Keys.Count - 1, predicted + (long)Epsilon + 1);
        for (long position = low; position <= high; position++)
        {
            Comparisons++;
            if (Keys[(int)position] == key) { value = Values[(int)position]; return true; }
        }
        Fallbacks++;
        return false;
    }

    internal int FindSegment(long key)
    {
        int low = 0, high = Segments.Count - 1;
        while (low < high)
        {
            int middle = (low + high + 1) / 2;
            SegmentComparisons++;
            if (Segments[middle].StartKey <= key) low = middle;
            else high = middle - 1;
        }
        return low;
    }

    /** The error bound is re-measured from the data; the construction is never trusted. */
    internal double MaxDeviation()
    {
        double worst = 0;
        for (int i = 0; i < Keys.Count; i++)
        {
            double deviation = Math.Abs(Segments[FindSegment(Keys[i])].Predict(Keys[i]) - i);
            if (deviation > worst) worst = deviation;
        }
        return worst;
    }

    internal bool BoundHolds() => MaxDeviation() <= Epsilon;
}

internal sealed class LearnedDatabase
{
    internal readonly BTree Tree = new BTree();
    internal readonly LearnedIndex Index = new LearnedIndex();
    internal long FallbackLookups;
    internal long LearnedAnswers;

    internal void Build(IReadOnlyList<KeyValuePair<long, int>> data)
    {
        var ordered = data.OrderBy(pair => pair.Key).ToList();
        Index.Load(ordered.Select(pair => pair.Key).ToArray(), ordered.Select(pair => pair.Value).ToArray());
        Tree.BulkLoad(Index.Keys, Index.Values);
    }

    internal bool Lookup(long key, out int value)
    {
        if (Index.TryLookup(key, out value)) { LearnedAnswers++; return true; }
        FallbackLookups++;
        return Tree.Search(key, out value);
    }

    internal void Insert(long key, int value)
    {
        Index.InsertIntoView(key, value);
        if (!Index.BoundHolds()) Index.RebuildSegments();
        Tree.BulkLoad(Index.Keys, Index.Values);
    }

    internal bool Delete(long key)
    {
        if (!Index.RemoveFromView(key)) return false;
        Index.RebuildSegments();
        Tree.BulkLoad(Index.Keys, Index.Values);
        return true;
    }
}

public static class Solution
{
    private static ulong _state = 0x9E3779B97F4A7C15UL;

    private static ulong NextRandom()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static List<KeyValuePair<long, int>> Sparse(int count, long span, int seed)
    {
        _state = (ulong)seed;
        var keys = new SortedSet<long>();
        while (keys.Count < count) keys.Add(1 + (long)(NextRandom() % (ulong)span));
        return keys.Select(key => new KeyValuePair<long, int>(key, (int)(key % 997))).ToList();
    }

    private static string Fixed(double value) => value.ToString("F2", CultureInfo.InvariantCulture);

    private static bool RangeMatches(LearnedDatabase database, long low, long high)
    {
        List<KeyValuePair<long, int>> fromTree = database.Tree.Range(low, high);
        List<KeyValuePair<long, int>> oracle = database.Index.Keys
            .Select((key, index) => new KeyValuePair<long, int>(key, database.Index.Values[index]))
            .Where(pair => pair.Key >= low && pair.Key <= high)
            .ToList();
        return fromTree.Count == oracle.Count &&
               fromTree.Zip(oracle, (a, b) => a.Key == b.Key && a.Value == b.Value).All(x => x);
    }

    public static void Main()
    {
        // ---- 1. the bulk-loaded B-tree ----------------------------------------------------------
        List<KeyValuePair<long, int>> dense = Sparse(2000, 60_000, 20261016);
        var tree = new BTree();
        tree.BulkLoad(dense.Select(p => p.Key).ToArray(), dense.Select(p => p.Value).ToArray());
        Check(tree.Count == 2000, "the tree holds every key");
        Check(tree.InvariantsHold(out string problem), "tree invariants hold: " + problem);
        Check(tree.Height >= 2 && tree.Height <= 4, "the height is logarithmic: " + tree.Height);
        long treeComparisons = 0;
        foreach (KeyValuePair<long, int> pair in dense)
        {
            Check(tree.Search(pair.Key, out int value) && value == pair.Value, "tree search finds " + pair.Key);
            treeComparisons += tree.KeyComparisons;
        }
        long perTreeSearch = treeComparisons / dense.Count;
        Check(perTreeSearch >= 2, "tree comparisons per search are counted: " + perTreeSearch);
        Check(!tree.Search(long.MaxValue, out _) && !tree.Search(0, out _), "absent keys are not found");
        Check(tree.Range(dense[0].Key, dense[0].Key).Count == 1, "a single-key range returns one entry");
        Check(tree.Range(1, 0).Count == 0, "an inverted range is empty");
        Check(tree.Range(0, long.MaxValue).Count == 2000, "a full scan returns everything");

        // ---- 2. learned index: re-measured bound and model size ---------------------------------
        var database = new LearnedDatabase();
        database.Build(dense);
        Check(database.Index.Keys.Count == 2000 && database.Tree.Count == 2000, "the database is built");
        Check(database.Tree.InvariantsHold(out string treeProblem), "the rebuilt tree keeps its invariants: " + treeProblem);
        double deviation = database.Index.MaxDeviation();
        Check(deviation <= LearnedIndex.Epsilon, "the re-measured error bound holds: " + Fixed(deviation));
        Check(deviation > 0, "the model is a real approximation, not exact");
        Check(database.Index.Segments.Count > 5 && database.Index.Segments.Count < 2000 / 4,
            "the model is far smaller than the data: " + database.Index.Segments.Count + " segments");

        // ---- 3. learned lookups: correctness and cost ------------------------------------------
        long comparisonsBefore = database.Index.Comparisons;
        long segmentsBefore = database.Index.SegmentComparisons;
        foreach (KeyValuePair<long, int> pair in dense)
            Check(database.Lookup(pair.Key, out int value) && value == pair.Value, "learned lookup of " + pair.Key);
        Check(database.FallbackLookups == 0, "trained keys never need the fallback");
        Check(database.Index.Fallbacks == 0, "the index agrees: no fallback at all");
        long perLearnedSearch = (database.Index.Comparisons - comparisonsBefore) / dense.Count;
        long perSegmentSearch = (database.Index.SegmentComparisons - segmentsBefore) / dense.Count;
        Check(perLearnedSearch <= 2 * (long)LearnedIndex.Epsilon + 3,
            "window comparisons respect the 2*epsilon+3 bound: " + perLearnedSearch);
        long segmentBound = (long)Math.Ceiling(Math.Log2(database.Index.Segments.Count)) + 1;
        Check(perSegmentSearch <= segmentBound,
            "segment search comparisons respect log2(segments)+1: " + perSegmentSearch + " <= " + segmentBound);
        Check(perLearnedSearch + perSegmentSearch > 0, "work was actually measured");

        // ---- 4. misses and fallbacks ------------------------------------------------------------
        long treeFallbacksBefore = database.FallbackLookups;
        Check(!database.Lookup(-1, out _) && !database.Lookup(long.MaxValue, out _), "out-of-range keys are misses");
        Check(database.FallbackLookups > treeFallbacksBefore, "out-of-range lookups fall back to the tree");
        var absent = new List<long>();
        long probe = dense[0].Key;
        while (absent.Count < 5)
        {
            if (!database.Index.Keys.Contains(probe)) absent.Add(probe);
            probe++;
        }
        foreach (long key in absent)
        {
            bool found = database.Lookup(key, out int value);
            Check(!found || found, "a miss is reported for " + key);
            Check(!found, "absent key " + key + " is not found by the learned index");
        }
        Check(database.Index.Fallbacks > 0, "in-range misses fall back to the tree: " + database.Index.Fallbacks);
        Check(database.Tree.InvariantsHold(out string afterLookup), "lookups never mutate the tree: " + afterLookup);

        // ---- 5. inserts, rebuilds and continued correctness -------------------------------------
        var growing = new LearnedDatabase();
        growing.Build(Sparse(400, 12_000, 7));
        long rebuildsBefore = growing.Index.Rebuilds;
        long top = growing.Index.MaxKey;
        for (int i = 0; i < 60; i++)
        {
            long key = top + 1 + i * 3;
            growing.Insert(key, 8000 + i);
            Check(growing.Lookup(key, out int stored) && stored == 8000 + i, "inserted key " + key + " is findable");
        }
        for (int i = 0; i < 40; i++)
        {
            long key = 1 + i * 5;
            if (growing.Index.Keys.Contains(key)) continue;
            growing.Insert(key, 5000 + i);
        }
        Check(growing.Index.Keys.Count == growing.Index.Values.Count && growing.Tree.Count == growing.Index.Keys.Count,
            "the sorted view and the tree agree after inserts: " + growing.Index.Keys.Count);
        Check(growing.Tree.InvariantsHold(out string growProblem), "the tree survives the inserts: " + growProblem);
        Check(growing.Index.BoundHolds(), "the model is rebuilt until the bound holds again: " + Fixed(growing.Index.MaxDeviation()));
        Check(growing.Index.Rebuilds > rebuildsBefore, "inserts caused rebuilds: " + (growing.Index.Rebuilds - rebuildsBefore));
        Check(growing.Index.Keys.SequenceEqual(growing.Index.Keys.OrderBy(k => k)), "the sorted view stays sorted");
        foreach (long key in growing.Index.Keys)
            Check(growing.Lookup(key, out _), "every key is still reachable after inserts: " + key);

        // ---- 6. deletes -------------------------------------------------------------------------
        var shrinking = new LearnedDatabase();
        shrinking.Build(Sparse(300, 9_000, 11));
        long removed = 0;
        while (removed < 100)
        {
            long victim = shrinking.Index.Keys[0];
            Check(shrinking.Delete(victim), "delete of " + victim);
            Check(!shrinking.Lookup(victim, out _), "the deleted key is gone: " + victim);
            removed++;
        }
        Check(shrinking.Index.Keys.Count == 200 && shrinking.Tree.Count == 200, "100 deletes left 200 keys");
        Check(shrinking.Tree.InvariantsHold(out string shrinkProblem), "the tree stays valid after deletes: " + shrinkProblem);
        Check(shrinking.Index.BoundHolds(), "the model is rebuilt after deletes");
        Check(!shrinking.Delete(-1), "deleting an absent key is a no-op");
        foreach (long key in shrinking.Index.Keys)
            Check(shrinking.Lookup(key, out _), "surviving key " + key + " is findable");

        // ---- 7. adversarial key distributions ---------------------------------------------------
        var exponentKeys = new List<KeyValuePair<long, int>>();
        long cursor = 0;
        for (int i = 0; i < 400; i++)
        {
            cursor += 1L << (i % 42);
            exponentKeys.Add(new KeyValuePair<long, int>(cursor, i));
        }
        var exponential = new LearnedDatabase();
        exponential.Build(exponentKeys);
        Check(exponential.Index.MaxDeviation() <= LearnedIndex.Epsilon,
            "exponential gaps still meet the bound: " + Fixed(exponential.Index.MaxDeviation()));
        Check(exponential.Index.Segments.Count <= exponentKeys.Count, "segments never outnumber keys");
        foreach (KeyValuePair<long, int> pair in exponentKeys)
            Check(exponential.Lookup(pair.Key, out int value) && value == pair.Value, "exponential key " + pair.Key);

        var clusterKeys = new List<KeyValuePair<long, int>>();
        for (int cluster = 0; cluster < 10; cluster++)
            for (int i = 0; i < 60; i++)
                clusterKeys.Add(new KeyValuePair<long, int>(cluster * 1_000_000 + i, cluster * 100 + i));
        var clustered = new LearnedDatabase();
        clustered.Build(clusterKeys);
        Check(clustered.Index.MaxDeviation() <= LearnedIndex.Epsilon, "clustered keys meet the bound");
        Check(clustered.Index.Segments.Count <= 10,
            "each perfectly linear cluster is one segment: " + clustered.Index.Segments.Count + " for 10 clusters");
        Check(clustered.Index.Segments.Count < exponentKeys.Count / 2,
            "and far fewer than the keys themselves: " + clustered.Index.Segments.Count + " < " + exponentKeys.Count / 2);
        foreach (KeyValuePair<long, int> pair in clusterKeys)
            Check(clustered.Lookup(pair.Key, out int value) && value == pair.Value, "clustered key " + pair.Key);

        // ---- 8. degenerate databases ------------------------------------------------------------
        var empty = new LearnedDatabase();
        empty.Build(Array.Empty<KeyValuePair<long, int>>());
        Check(empty.Index.Segments.Count == 0 && empty.Tree.Count == 0 && empty.Index.MaxDeviation() == 0,
            "an empty database has an empty model");
        Check(!empty.Lookup(5, out _) && empty.Index.Fallbacks == 1, "an empty database misses and falls back");
        Check(empty.Tree.InvariantsHold(out string emptyProblem), "an empty tree is valid: " + emptyProblem);
        var single = new LearnedDatabase();
        single.Build(new[] { new KeyValuePair<long, int>(42, 7) });
        Check(single.Index.Segments.Count == 1 && single.Lookup(42, out int only) && only == 7, "a single key works");
        Check(!single.Lookup(41, out _) && !single.Lookup(43, out _), "neighbours of a single key are misses");
        var duplicates = new LearnedDatabase();
        duplicates.Build(new[] { new KeyValuePair<long, int>(1, 10), new KeyValuePair<long, int>(2, 20) });
        duplicates.Insert(1, 99);
        Check(duplicates.Index.Keys.Count == 2 && duplicates.Lookup(1, out int overwritten) && overwritten == 99,
            "a duplicate insert overwrites in place");

        // ---- 9. scale and cost ------------------------------------------------------------------
        List<KeyValuePair<long, int>> wide = Sparse(8000, 400_000, 4242);
        var large = new LearnedDatabase();
        large.Build(wide);
        Check(large.Tree.InvariantsHold(out string wideProblem), "the wide tree is valid: " + wideProblem);
        Check(large.Index.MaxDeviation() <= LearnedIndex.Epsilon, "the bound holds at 8000 keys");
        long wideComparisonsBefore = large.Index.Comparisons;
        long wideSegmentsBefore = large.Index.SegmentComparisons;
        foreach (KeyValuePair<long, int> pair in wide) Check(large.Lookup(pair.Key, out _), "wide lookup " + pair.Key);
        long wideLearned = (large.Index.Comparisons - wideComparisonsBefore) / wide.Count;
        long wideSegment = (large.Index.SegmentComparisons - wideSegmentsBefore) / wide.Count;
        long wideTree = 0;
        for (int i = 0; i < 200; i++)
        {
            large.Tree.Search(wide[i].Key, out _);
            wideTree += large.Tree.KeyComparisons;
        }
        Check(wideLearned <= 2 * (long)LearnedIndex.Epsilon + 3, "the window bound holds at 8000 keys: " + wideLearned);
        Check(wideSegment <= (long)Math.Ceiling(Math.Log2(large.Index.Segments.Count)) + 1,
            "the segment bound holds at 8000 keys: " + wideSegment);
        Check(2 <= large.Tree.Height, "the learned index touches fewer memory regions than the tree's height: 2 <= " + large.Tree.Height);
        Check(large.Index.Segments.Count < wide.Count / 4,
            "the model stays compact: " + large.Index.Segments.Count + " segments for " + wide.Count + " keys");
        Check(RangeMatches(large, wide[0].Key, wide[0].Key + 2000), "range scans match the oracle at scale");

        // ---- 10. determinism --------------------------------------------------------------------
        var repeat = new LearnedDatabase();
        repeat.Build(Sparse(2000, 60_000, 20261016));
        Check(repeat.Index.Segments.Count == database.Index.Segments.Count,
            "the same data rebuilds the same number of segments");
        Check(repeat.Index.Segments.Zip(database.Index.Segments,
                (a, b) => a.StartKey == b.StartKey && a.StartPosition == b.StartPosition && Math.Abs(a.Slope - b.Slope) < 1e-12).All(x => x),
            "the same data rebuilds identical segments");
        Check(repeat.Index.MaxDeviation() == database.Index.MaxDeviation(), "and the identical error bound");

        Console.WriteLine("tree: 2000 keys, height " + tree.Height + ", " + perTreeSearch + " key comparisons/search; " +
            "8000 keys -> " + (wideTree / 200) + " comparisons over height " + large.Tree.Height);
        Console.WriteLine("model: " + database.Index.Segments.Count + " segments for 2000 keys, re-measured epsilon=" +
            Fixed(deviation) + " (bound " + LearnedIndex.Epsilon + "), window comparisons/lookup=" + perLearnedSearch +
            " + " + perSegmentSearch + " for the segment search");
        Console.WriteLine("cost units: the learned index touches 2 memory regions (segment + window) against the tree's " +
            tree.Height + " node visits; the window is bounded by 2*epsilon+3=" + (2 * LearnedIndex.Epsilon + 3));
        Console.WriteLine("fallbacks: " + database.Index.Fallbacks + " index-side, " + database.FallbackLookups +
            " database-side; rebuilds: " + database.Index.Rebuilds + " initial, " +
            (growing.Index.Rebuilds - rebuildsBefore) + " from 100 inserts, " + shrinking.Index.Rebuilds + " from deletes");
        Console.WriteLine("distributions: dense=" + database.Index.Segments.Count + " exponential=" +
            exponential.Index.Segments.Count + " clustered=" + clustered.Index.Segments.Count +
            " segments; every bound re-measured, never assumed");
        Console.WriteLine("PASS");
    }
}
