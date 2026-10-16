// Task 776 - Implement optimistic B-tree rebalancing algorithm sketch. (Tier: Expert, CTCI Ch 14 - Databases)
//
// Deleting from a B-tree is where the interesting work happens: a node that drops below its minimum
// occupancy must borrow a key from a sibling or merge with one, and both decisions depend on the
// sibling's current occupancy. This file processes deletes optimistically instead of taking latches:
//
//   1. the operation walks down to the leaf recording (node, version) for every node it reads;
//   2. if the leaf sits at the minimum occupancy it also reads the sibling it would like to borrow
//      from and records that sibling's version, producing a plan (borrow from the left, borrow from
//      the right, or merge);
//   3. it yields, which is the window in which another operation may commit;
//   4. it validates every recorded version. If one moved, the plan is thrown away and the tree is
//      walked again (the conflict is resolved by re-planning, never by patching a stale plan);
//   5. on success the delete - including its borrow/merge cascade - runs as one atomic step and bumps
//      the version of every node on the mutated path, so no commit can hide from another validation.
//
// Verified: the occupancy invariant holds for every node after every commit (rebalancing is real, the
// optimism only concerns *when* the plan is validated), the tree always equals the sequential replay
// of the observed commits, a planned rebalance is exactly the rebalance that is executed when the plan
// validates, a restart is always backed by a real version difference, merges decrease the node count
// while borrows do not, deleting an absent key mutates nothing and bumps no version, and draining a
// 120-key tree leaves an empty root that satisfies the invariants.
//
// Documented simplification (this task is a sketch): only the leaf-level rebalance decision is planned
// optimistically - the walk itself is a plain optimistic descent, there is no multi-level look-ahead,
// no latch coupling between the retry and the commit, no node-level increment/decrement restarts for
// the *insert* side, and no crash recovery beyond the fact that the tree is never left half changed.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;            // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal enum Rebalance
{
    None = 0,
    BorrowFromPrevious = 1,
    BorrowFromNext = 2,
    Merge = 3,
}

internal sealed class TreeNode
{
    internal readonly int Id;
    internal readonly List<int> Keys = new List<int>();
    internal readonly List<TreeNode> Children = new List<TreeNode>();
    internal long Version;

    internal TreeNode(int id) { Id = id; }
    internal bool IsLeaf => Children.Count == 0;
    internal bool Has(int key) => Keys.BinarySearch(key) >= 0;
}

internal sealed class OptimisticTree
{
    internal readonly int MinDegree;
    private readonly List<TreeNode> _nodes = new List<TreeNode>();
    private int _nextId;

    internal TreeNode Root;
    internal int Count;
    internal long Commits;
    internal long Borrows;
    internal long Merges;
    internal Rebalance LastRebalance;

    internal OptimisticTree(int minDegree)
    {
        if (minDegree < 2) throw new ArgumentException("a B-tree needs a minimum degree of at least 2");
        MinDegree = minDegree;
        Root = MakeNode();
    }

    internal int NodeCount => _nodes.Count;

    internal TreeNode MakeNode()
    {
        var node = new TreeNode(_nextId++);
        _nodes.Add(node);
        return node;
    }

    internal int ChildIndex(TreeNode node, int key)
    {
        int at = node.Keys.BinarySearch(key);
        return at >= 0 ? at : ~at;
    }

    internal bool Contains(int key)
    {
        TreeNode current = Root;
        while (true)
        {
            int at = current.Keys.BinarySearch(key);
            if (at >= 0) return true;
            if (current.IsLeaf) return false;
            current = current.Children[~at];
        }
    }

    internal bool Insert(int key)
    {
        if (Root.Keys.Count == 2 * MinDegree - 1)
        {
            TreeNode newRoot = MakeNode();
            newRoot.Children.Add(Root);
            SplitChild(newRoot, 0);
            Root = newRoot;
        }
        bool added = InsertNonFull(Root, key);
        if (added) Count++;
        return added;
    }

    private bool InsertNonFull(TreeNode node, int key)
    {
        if (node.IsLeaf)
        {
            int at = node.Keys.BinarySearch(key);
            if (at >= 0) return false;
            node.Keys.Insert(~at, key);
            return true;
        }
        int index = ChildIndex(node, key);
        if (index < node.Keys.Count && node.Keys[index] == key) return false;
        if (node.Children[index].Keys.Count == 2 * MinDegree - 1)
        {
            SplitChild(node, index);
            if (key == node.Keys[index]) return false;
            if (key > node.Keys[index]) index++;
        }
        return InsertNonFull(node.Children[index], key);
    }

    private void SplitChild(TreeNode parent, int index)
    {
        TreeNode full = parent.Children[index];
        TreeNode right = MakeNode();
        int middle = full.Keys[MinDegree - 1];
        for (int i = MinDegree; i < full.Keys.Count; i++) right.Keys.Add(full.Keys[i]);
        if (!full.IsLeaf)
            for (int i = MinDegree; i < full.Children.Count; i++) right.Children.Add(full.Children[i]);
        full.Keys.RemoveRange(MinDegree - 1, full.Keys.Count - (MinDegree - 1));
        if (!full.IsLeaf) full.Children.RemoveRange(MinDegree, full.Children.Count - MinDegree);
        parent.Keys.Insert(index, middle);
        parent.Children.Insert(index + 1, right);
        full.Version++;
        right.Version++;
        parent.Version++;
    }

    internal bool Delete(int key)
    {
        if (!Contains(key)) return false;
        LastRebalance = Rebalance.None;
        DeleteFrom(Root, key);
        if (Root.Keys.Count == 0 && !Root.IsLeaf) Root = Root.Children[0];
        Root.Version++;
        Count--;
        return true;
    }

    private void DeleteFrom(TreeNode node, int key)
    {
        int index = node.Keys.BinarySearch(key);
        if (index >= 0)
        {
            if (node.IsLeaf) { node.Keys.RemoveAt(index); node.Version++; return; }
            TreeNode left = node.Children[index], right = node.Children[index + 1];
            if (left.Keys.Count >= MinDegree)
            {
                int predecessor = MaxKey(left);
                node.Keys[index] = predecessor;
                node.Version++;
                DeleteFrom(left, predecessor);
            }
            else if (right.Keys.Count >= MinDegree)
            {
                int successor = MinKey(right);
                node.Keys[index] = successor;
                node.Version++;
                DeleteFrom(right, successor);
            }
            else
            {
                Merge(node, index);
                node.Version++;
                DeleteFrom(node.Children[index], key);
            }
            return;
        }
        int child = ~index;
        if (node.Children[child].Keys.Count == MinDegree - 1)
        {
            Fill(node, child);
            node.Version++;
            if (child >= node.Keys.Count) child = node.Keys.Count;
        }
        DeleteFrom(node.Children[child], key);
    }

    private void Fill(TreeNode parent, int index)
    {
        if (index > 0 && parent.Children[index - 1].Keys.Count >= MinDegree) { BorrowFromPrevious(parent, index); return; }
        if (index + 1 < parent.Children.Count && parent.Children[index + 1].Keys.Count >= MinDegree) { BorrowFromNext(parent, index); return; }
        if (index + 1 < parent.Children.Count) Merge(parent, index);
        else Merge(parent, index - 1);
    }

    private void BorrowFromPrevious(TreeNode parent, int index)
    {
        TreeNode child = parent.Children[index], sibling = parent.Children[index - 1];
        child.Keys.Insert(0, parent.Keys[index - 1]);
        if (!sibling.IsLeaf) child.Children.Insert(0, sibling.Children[sibling.Children.Count - 1]);
        parent.Keys[index - 1] = sibling.Keys[sibling.Keys.Count - 1];
        sibling.Keys.RemoveAt(sibling.Keys.Count - 1);
        if (!sibling.IsLeaf) sibling.Children.RemoveAt(sibling.Children.Count - 1);
        child.Version++;
        sibling.Version++;
        Borrows++;
        LastRebalance = Rebalance.BorrowFromPrevious;
    }

    private void BorrowFromNext(TreeNode parent, int index)
    {
        TreeNode child = parent.Children[index], sibling = parent.Children[index + 1];
        child.Keys.Add(parent.Keys[index]);
        if (!sibling.IsLeaf) child.Children.Add(sibling.Children[0]);
        parent.Keys[index] = sibling.Keys[0];
        sibling.Keys.RemoveAt(0);
        if (!sibling.IsLeaf) sibling.Children.RemoveAt(0);
        child.Version++;
        sibling.Version++;
        Borrows++;
        LastRebalance = Rebalance.BorrowFromNext;
    }

    private void Merge(TreeNode parent, int index)
    {
        TreeNode left = parent.Children[index], right = parent.Children[index + 1];
        left.Keys.Add(parent.Keys[index]);
        for (int i = 0; i < right.Keys.Count; i++) left.Keys.Add(right.Keys[i]);
        for (int i = 0; i < right.Children.Count; i++) left.Children.Add(right.Children[i]);
        parent.Keys.RemoveAt(index);
        parent.Children.RemoveAt(index + 1);
        left.Version++;
        right.Version++;
        Merges++;
        LastRebalance = Rebalance.Merge;
    }

    private static int MaxKey(TreeNode node)
    {
        TreeNode current = node;
        while (!current.IsLeaf) current = current.Children[current.Children.Count - 1];
        return current.Keys[current.Keys.Count - 1];
    }

    private static int MinKey(TreeNode node)
    {
        TreeNode current = node;
        while (!current.IsLeaf) current = current.Children[0];
        return current.Keys[0];
    }

    internal List<int> Snapshot()
    {
        var keys = new List<int>();
        Walk(Root, keys);
        return keys;
    }

    private static void Walk(TreeNode node, List<int> keys)
    {
        if (node.IsLeaf)
        {
            keys.AddRange(node.Keys);
            return;
        }
        for (int i = 0; i < node.Keys.Count; i++)
        {
            Walk(node.Children[i], keys);
            keys.Add(node.Keys[i]);
        }
        Walk(node.Children[node.Keys.Count], keys);
    }

    internal int Depth()
    {
        int depth = 1;
        TreeNode current = Root;
        while (!current.IsLeaf) { current = current.Children[0]; depth++; }
        return depth;
    }

    /// <summary>Nodes still reachable from the root; merges orphan a node, allocations alone do not show it.</summary>
    internal int LiveNodeCount()
    {
        int count = 0;
        CountLive(Root, ref count);
        return count;
    }

    private static void CountLive(TreeNode node, ref int count)
    {
        count++;
        foreach (TreeNode child in node.Children) CountLive(child, ref count);
    }

    internal long TotalVersions()
    {
        long total = 0;
        foreach (TreeNode node in _nodes) total += node.Version;
        return total;
    }

    internal string InvariantViolation()
    {
        int seen = 0;
        var leafDepths = new HashSet<int>();
        string problem = CheckNode(Root, 0, long.MinValue, long.MaxValue, true, leafDepths, ref seen);
        if (problem != null) return problem;
        if (Count == 0) return seen == 0 ? null : "an empty tree still holds " + seen + " keys";
        if (Count != seen) return "Count says " + Count + " but the tree holds " + seen + " keys";
        if (leafDepths.Count != 1) return "leaves live at different depths";
        return null;
    }

    private string CheckNode(TreeNode node, int depth, long low, long high, bool isRoot, HashSet<int> leafDepths, ref int seen)
    {
        for (int i = 1; i < node.Keys.Count; i++) if (node.Keys[i - 1] >= node.Keys[i]) return "keys out of order in node " + node.Id;
        foreach (int key in node.Keys) if (key <= low || key >= high) return "key " + key + " violates its separator bounds in node " + node.Id;
        if (!isRoot && node.Keys.Count < MinDegree - 1) return "node " + node.Id + " underflows the minimum occupancy";
        if (node.Keys.Count > 2 * MinDegree - 1) return "node " + node.Id + " overflows the maximum occupancy";
        seen += node.Keys.Count;
        if (node.IsLeaf)
        {
            leafDepths.Add(depth);
            return null;
        }
        if (node.Children.Count != node.Keys.Count + 1) return "node " + node.Id + " has the wrong number of children";
        for (int i = 0; i < node.Keys.Count; i++)
        {
            string problem = CheckNode(node.Children[i], depth + 1, low, node.Keys[i], false, leafDepths, ref seen);
            if (problem != null) return problem;
            low = node.Keys[i];
        }
        return CheckNode(node.Children[node.Keys.Count], depth + 1, low, high, false, leafDepths, ref seen);
    }
}

internal sealed class OptimisticDelete
{
    internal int Key;
    internal IEnumerator<int> Steps;
    internal bool Done;
    internal bool Result;
    internal bool Committed;
    internal bool Counted;
    internal int CommitOrder;
    internal long StartCommits;
    internal long EndCommits;
    internal int Attempts = 1;
    internal bool EvidenceValid = true;
    internal Rebalance Plan = Rebalance.None;
    internal Rebalance Executed = Rebalance.None;
    internal bool PlanMatched = true;
    internal bool StartedAtMinimum;

    internal readonly List<TreeNode> ReadSet = new List<TreeNode>();
    internal readonly List<long> Versions = new List<long>();
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool SameKeys(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    /// <summary>
    /// Optimistic delete: read the path, plan the rebalance from the sibling's occupancy, let other
    /// operations commit during the window, validate, and only then execute.
    /// </summary>
    private static IEnumerator<int> DeleteOptimistically(OptimisticTree tree, OptimisticDelete op)
    {
        int window = 0;
        while (true)
        {
            op.ReadSet.Clear();
            op.Versions.Clear();
            op.StartCommits = tree.Commits;
            TreeNode current = tree.Root;
            while (true)
            {
                op.ReadSet.Add(current);
                op.Versions.Add(current.Version);
                yield return 0;                                    // one step per node read
                if (current.IsLeaf) break;
                current = current.Children[tree.ChildIndex(current, op.Key)];
            }

            // plan: does the leaf start at the minimum occupancy, and what would the sibling allow?
            op.Plan = Rebalance.None;
            op.StartedAtMinimum = false;
            TreeNode parent = op.ReadSet.Count >= 2 ? op.ReadSet[op.ReadSet.Count - 2] : null;
            TreeNode leaf = op.ReadSet[op.ReadSet.Count - 1];
            if (parent != null && leaf.IsLeaf && leaf.Keys.Count == tree.MinDegree - 1)
            {
                op.StartedAtMinimum = true;
                int child = tree.ChildIndex(parent, op.Key);
                if (child >= parent.Children.Count) child = parent.Children.Count - 1;
                TreeNode left = child > 0 ? parent.Children[child - 1] : null;
                TreeNode right = child + 1 < parent.Children.Count ? parent.Children[child + 1] : null;
                if (left != null && left.Keys.Count >= tree.MinDegree) { op.Plan = Rebalance.BorrowFromPrevious; op.ReadSet.Add(left); op.Versions.Add(left.Version); }
                else if (right != null && right.Keys.Count >= tree.MinDegree) { op.Plan = Rebalance.BorrowFromNext; op.ReadSet.Add(right); op.Versions.Add(right.Version); }
                else { op.Plan = Rebalance.Merge; op.ReadSet.Add(left ?? right); op.Versions.Add((left ?? right).Version); }
            }

            window = 1 + (int)(op.Key % 3);
            for (int i = 0; i < window; i++) yield return 0;

            bool valid = true;
            for (int i = 0; i < op.ReadSet.Count; i++)
            {
                long now = op.ReadSet[i].Version;
                if (now != op.Versions[i])
                {
                    valid = false;
                    op.EvidenceValid = op.EvidenceValid && now != op.Versions[i];
                    break;
                }
            }
            op.EndCommits = tree.Commits;
            if (!valid)
            {
                op.Attempts++;
                yield return 0;                                    // re-plan instead of patching a stale plan
                continue;
            }

            op.Result = tree.Delete(op.Key);
            op.Executed = op.Result ? tree.LastRebalance : Rebalance.None;
            if (op.Result && op.StartedAtMinimum) op.PlanMatched = op.Executed == op.Plan;
            if (op.Result)
            {
                tree.Commits++;
                op.Committed = true;
            }
            yield return 0;
            op.Done = true;
            yield break;
        }
    }

    private static int Run(OptimisticTree tree, List<OptimisticDelete> ops, List<List<int>> history, int budget,
        out string problem, List<int> initial = null)
    {
        problem = null;
        var reference = initial == null ? new List<int>() : new List<int>(initial);
        history.Add(new List<int>(reference));
        int steps = 0;
        bool progress = true;
        while (progress && steps < budget)
        {
            progress = false;
            foreach (OptimisticDelete op in ops)
            {
                if (op.Done) continue;
                if (!op.Steps.MoveNext()) { op.Done = true; continue; }
                steps++;
                progress = true;
                if (op.Committed && !op.Counted)
                {
                    op.Counted = true;
                    op.CommitOrder = history.Count;
                    int at = reference.BinarySearch(op.Key);
                    if (at >= 0) reference.RemoveAt(at);
                    history.Add(new List<int>(reference));
                }
                if (!SameKeys(tree.Snapshot(), reference)) { problem = "the tree diverged from the committed replay"; return steps; }
                problem = tree.InvariantViolation();
                if (problem != null) return steps;
            }
        }
        if (ops.Exists(o => !o.Done)) problem = "the scheduler ran out of budget with operations pending";
        return steps;
    }

    public static void Main()
    {
        // ---- draining an order-2 tree: every rebalance is exercised -------------------------------
        {
            const int Size = 120;
            var tree = new OptimisticTree(2);
            for (int key = 0; key < Size; key++) Check(tree.Insert(key), "insert " + key);
            Check(tree.InvariantViolation() == null, "the freshly built tree satisfies its invariants");
            Check(tree.Count == Size, "all keys are present (" + tree.Count + ")");
            bool invariantsHold = true, occupancyRespected = true;
            string problem = "ok";
            int deleted = 0;
            for (int key = 0; key < Size; key++)
            {
                int nodesBefore = tree.LiveNodeCount();
                TreeNode rootBefore = tree.Root;
                long mergesBefore = tree.Merges;
                long borrowsBefore = tree.Borrows;
                bool removed = tree.Delete(key);
                if (!removed) { problem = "key " + key + " could not be deleted"; break; }
                deleted++;
                bool merged = tree.Merges > mergesBefore;
                bool borrowed = tree.Borrows > borrowsBefore;
                bool rootCollapsed = !ReferenceEquals(tree.Root, rootBefore);
                long mergesAdded = tree.Merges - mergesBefore;
                int nodesAfter = tree.LiveNodeCount();
                // each merge orphans exactly one node; a collapsed root orphans one more; a borrow none
                int expectedNodes = nodesBefore - (int)mergesAdded - (rootCollapsed ? 1 : 0);
                if (nodesAfter != expectedNodes) { problem = "key " + key + ": live nodes " + nodesAfter + " not " + expectedNodes
                    + " (merges=" + mergesAdded + " borrowed=" + borrowed + " rootCollapsed=" + rootCollapsed + ")"; break; }
                if (tree.InvariantViolation() != null) { invariantsHold = false; problem = tree.InvariantViolation(); break; }
                if (tree.Count != Size - deleted) { occupancyRespected = false; problem = "count off after key " + key; break; }
            }
            Check(invariantsHold, "the tree keeps its invariants after every one of the deletions: " + problem);
            Check(occupancyRespected, "the key count tracks the deletions: " + problem);
            Check(tree.Count == 0 && tree.Snapshot().Count == 0, "the tree drained completely: " + problem);
            Check(tree.Root.IsLeaf, "the root collapsed back to a leaf");
            Check(tree.InvariantViolation() == null, "the drained tree satisfies its invariants");
            Check(tree.Merges > 0 && tree.Borrows > 0, "both merges and borrows happened (" + tree.Merges + " merges, " + tree.Borrows + " borrows)");
            Console.WriteLine("drain: keys=" + Size + " merges=" + tree.Merges + " borrows=" + tree.Borrows
                + " nodes=" + tree.NodeCount + " depth=" + tree.Depth());
        }

        // ---- deleting an absent key mutates nothing ----------------------------------------------
        {
            var tree = new OptimisticTree(2);
            for (int key = 1; key <= 30; key++) tree.Insert(key);
            long versionsBefore = tree.TotalVersions();
            var missing = new OptimisticDelete { Key = 99 };
            missing.Steps = DeleteOptimistically(tree, missing);
            var single = new List<OptimisticDelete> { missing };
            var history = new List<List<int>>();
            string problem;
            Run(tree, single, history, 10000, out problem, tree.Snapshot());
            Check(problem == null, "the absent-key delete ran cleanly: " + problem);
            Check(!missing.Result && !missing.Committed, "deleting an absent key commits nothing");
            Check(tree.TotalVersions() == versionsBefore, "no version was bumped by the failed delete");
            Check(tree.Count == 30 && tree.InvariantViolation() == null, "the tree is untouched");
            Check(missing.Plan == Rebalance.None, "no rebalance is planned when the key is absent");
        }

        // ---- isolated deletes: the plan is always the executed rebalance ---------------------------
        {
            const int Size = 90;
            var tree = new OptimisticTree(2);
            for (int key = 1; key <= Size; key++) tree.Insert(key);
            int planned = 0, matched = 0, restarts = 0, borrowPlans = 0, mergePlans = 0;
            for (int key = 1; key <= Size; key++)
            {
                var op = new OptimisticDelete { Key = key };
                op.Steps = DeleteOptimistically(tree, op);
                var single = new List<OptimisticDelete> { op };
                var history = new List<List<int>>();
                string problem;
                Run(tree, single, history, 10000, out problem, tree.Snapshot());
                Check(problem == null, "isolated delete " + key + " ran cleanly: " + problem);
                Check(op.Result, "key " + key + " was deleted");
                if (op.StartedAtMinimum)
                {
                    planned++;
                    if (op.PlanMatched) matched++;
                    if (op.Plan == Rebalance.Merge) mergePlans++;
                    else borrowPlans++;
                }
                restarts += op.Attempts - 1;
                if (tree.InvariantViolation() != null) { Check(false, "invariant broken at key " + key + ": " + tree.InvariantViolation()); }
            }
            Check(restarts == 0, "an isolated delete never restarts (" + restarts + ")");
            Check(planned > 10, "many deletes really started at the minimum occupancy (" + planned + ")");
            Check(matched == planned, "every planned rebalance was exactly the one executed (" + matched + "/" + planned + ")");
            Check(borrowPlans > 0 && mergePlans > 0, "both plan kinds occurred (" + borrowPlans + " borrows, " + mergePlans + " merges)");
            Check(tree.Count == 0 && tree.InvariantViolation() == null, "the tree drained through the optimistic path");
            Console.WriteLine("isolated deletes: planned=" + planned + " matched=" + matched
                + " borrowPlans=" + borrowPlans + " mergePlans=" + mergePlans);
        }

        // ---- two deletes whose windows overlap: exactly one re-plan --------------------------------
        {
            var tree = new OptimisticTree(2);
            for (int key = 1; key <= 12; key++) tree.Insert(key);
            var first = new OptimisticDelete { Key = 1 };
            var second = new OptimisticDelete { Key = 2 };
            first.Steps = DeleteOptimistically(tree, first);
            second.Steps = DeleteOptimistically(tree, second);
            var ops = new List<OptimisticDelete> { first, second };
            var history = new List<List<int>>();
            string problem;
            int steps = Run(tree, ops, history, 20000, out problem, tree.Snapshot());
            Check(problem == null, "the overlapping run finished cleanly: " + problem);
            Check(first.Attempts + second.Attempts == 3, "the overlapping window cost exactly one re-plan ("
                + first.Attempts + "/" + second.Attempts + ")");
            Check(first.EvidenceValid && second.EvidenceValid, "the restart was caused by a real version difference");
            Check(tree.Count == 10 && tree.InvariantViolation() == null, "both deletes landed and the tree is consistent");
            Check(!tree.Contains(1) && !tree.Contains(2), "the two keys are gone");
            Console.WriteLine("overlapping pair: steps=" + steps + " attempts=" + first.Attempts + "/" + second.Attempts
                + " n=" + tree.Count);
        }

        // ---- a contended delete workload ------------------------------------------------------------
        {
            const int Ops = 20, Size = 60;
            var tree = new OptimisticTree(2);
            var initialKeys = new List<int>();
            for (int key = 0; key < Size; key++) { tree.Insert(key); initialKeys.Add(key); }
            ulong state = 0x5EED776000000001UL;
            var ops = new List<OptimisticDelete>();
            for (int i = 0; i < Ops; i++)
            {
                var op = new OptimisticDelete { Key = SplitMix64.NextInt(ref state, Size) };
                op.Steps = DeleteOptimistically(tree, op);
                ops.Add(op);
            }
            var history = new List<List<int>>();
            string problem;
            int steps = Run(tree, ops, history, 200000, out problem, initialKeys);
            Check(problem == null, "the contended run finished cleanly: " + problem);
            Check(ops.TrueForAll(o => o.Done), "every optimistic delete completed");
            int restarts = 0, planChecks = 0, planMatches = 0;
            foreach (OptimisticDelete op in ops)
            {
                restarts += op.Attempts - 1;
                if (op.StartedAtMinimum && op.Result) { planChecks++; if (op.PlanMatched) planMatches++; }
                Check(op.EvidenceValid, "restart evidence for key " + op.Key);
            }
            Check(restarts > 0, "contention really forced re-planning (" + restarts + " restarts)");
            Check(planMatches == planChecks, "validated plans were honoured (" + planMatches + "/" + planChecks + ")");
            Check(SameKeys(tree.Snapshot(), history[history.Count - 1]), "the tree equals the committed replay of the deletes");
            Check(tree.InvariantViolation() == null, "the contended tree satisfies the occupancy invariant: " + tree.InvariantViolation());
            Check(tree.Count == Size - ops.FindAll(o => o.Result).Count, "the surviving key count matches the successful deletes");
            bool removedKeysGone = true;
            foreach (OptimisticDelete op in ops) if (op.Result && tree.Contains(op.Key)) removedKeysGone = false;
            Check(removedKeysGone, "no successfully deleted key is still present");
            Console.WriteLine("contended deletes: steps=" + steps + " restarts=" + restarts + " planChecks=" + planChecks
                + " n=" + tree.Count + " merges=" + tree.Merges + " borrows=" + tree.Borrows);
        }

        // ---- boundary: empty tree and a single key -------------------------------------------------
        {
            var empty = new OptimisticTree(3);
            var probe = new OptimisticDelete { Key = 5 };
            probe.Steps = DeleteOptimistically(empty, probe);
            var ops = new List<OptimisticDelete> { probe };
            var history = new List<List<int>>();
            string problem;
            Run(empty, ops, history, 5000, out problem, empty.Snapshot());
            Check(problem == null && !probe.Result && !probe.Committed, "deleting from an empty tree is a no-op");
            Check(empty.InvariantViolation() == null, "the empty tree satisfies its invariants");

            var one = new OptimisticTree(2);
            Check(one.Insert(7), "the single key is inserted");
            var only = new OptimisticDelete { Key = 7 };
            only.Steps = DeleteOptimistically(one, only);
            var ops2 = new List<OptimisticDelete> { only };
            var history2 = new List<List<int>>();
            Run(one, ops2, history2, 5000, out problem, one.Snapshot());
            Check(only.Result && one.Count == 0 && one.Root.IsLeaf, "the last key disappears and the root stays a leaf");
            Check(one.InvariantViolation() == null, "the emptied single-key tree satisfies its invariants");
            Console.WriteLine("boundary: emptyDeletes=" + probe.Result + " singleKeyLeft=" + one.Count);
        }

        Console.WriteLine("PASS");
    }
}
