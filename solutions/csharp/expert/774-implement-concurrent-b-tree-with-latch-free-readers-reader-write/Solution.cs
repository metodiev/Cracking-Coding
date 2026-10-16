// Task 774 - Implement concurrent B-tree with latch-free readers (reader-writer locks). (Tier: Expert, CTCI Ch 14 - Databases)
//
// The file implements a real B-tree (CLRS order-t tree: preemptive splitting on insert, borrow/merge
// cascade on delete) and runs it under an explicit latch protocol:
//
//   * readers take a shared (S) latch on the node they inspect and use latch coupling - the child
//     latch is acquired before the parent latch is released - so a reader holds at most two latches
//     and can never inspect a node that a writer is changing;
//   * writers take an exclusive (X) latch on the whole root-to-leaf path, mutate the tree in one
//     atomic step and only then release the latches bottom-up. Writes are therefore serialised and
//     commit in an order that can be replayed sequentially.
//
// Concurrency is simulated with a deterministic cooperative scheduler: every operation is an iterator
// that yields once per atomic action (one node visit, one latch attempt, one mutation), and the
// scheduler advances the operations round-robin. Nothing depends on timing, yet the interleavings are
// real: readers are routinely stalled by a writer holding the path and writers are stalled by readers
// holding S latches.
//
// Verified after every scheduler step: the tree equals the sequential replay of the committed prefix
// (no torn state is ever visible), the occupancy/order/equal-leaf-depth invariants hold, latch
// invariants hold (no node carries readers and a writer at once, a reader never holds more than two
// latches, a writer never more than depth+1) and every search answer equals the presence of its key in
// the state at the commit index recorded when the leaf was inspected (the linearizability check).
// Contention really occurred, and every operation finished inside the step budget, so the protocol is
// deadlock free. The minimum order t = 2, duplicate inserts, deletes of absent keys and draining the
// tree to empty are covered as well.
//
// Documented simplification: the scheduler is single threaded and one scheduler step is atomic, so the
// code verifies the *protocol* (which latch is required when, and what a reader may observe) rather
// than memory-level races; writers latch the whole path instead of taking a B-link style right-link
// descent; no page fix, no redo/undo logging and no crash recovery are modelled.
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

internal sealed class BNode
{
    internal readonly int Id;
    internal readonly List<int> Keys = new List<int>();
    internal readonly List<BNode> Children = new List<BNode>();

    internal BNode(int id) { Id = id; }
    internal bool IsLeaf => Children.Count == 0;
    internal bool Has(int key) => Keys.BinarySearch(key) >= 0;
    internal string Describe() => "node" + Id + "[" + string.Join(",", Keys) + "]";
}

internal sealed class BTree
{
    internal readonly int MinDegree;
    private readonly List<BNode> _nodes = new List<BNode>();
    private int _nextId;

    internal BNode Root;
    internal int Count;
    internal long Commits;

    internal BTree(int minDegree)
    {
        if (minDegree < 2) throw new ArgumentException("a B-tree needs a minimum degree of at least 2");
        MinDegree = minDegree;
        Root = MakeNode();
    }

    internal int NodeCount => _nodes.Count;

    internal BNode MakeNode()
    {
        var node = new BNode(_nextId++);
        _nodes.Add(node);
        return node;
    }

    internal bool Contains(int key)
    {
        BNode current = Root;
        while (true)
        {
            int at = current.Keys.BinarySearch(key);
            if (at >= 0) return true;
            if (current.IsLeaf) return false;
            current = current.Children[~at];
        }
    }

    /// <summary>Index of the child subtree that must contain the key (internal nodes only).</summary>
    internal int ChildIndex(BNode node, int key)
    {
        int at = node.Keys.BinarySearch(key);
        return at >= 0 ? at : ~at;
    }

    internal bool Insert(int key)
    {
        if (Root.Keys.Count == 2 * MinDegree - 1)
        {
            BNode newRoot = MakeNode();
            newRoot.Children.Add(Root);
            SplitChild(newRoot, 0);
            Root = newRoot;
        }
        bool added = InsertNonFull(Root, key);
        if (added) Count++;
        return added;
    }

    private bool InsertNonFull(BNode node, int key)
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

    internal void SplitChild(BNode parent, int index)
    {
        BNode full = parent.Children[index];
        BNode right = MakeNode();
        int middle = full.Keys[MinDegree - 1];
        for (int i = MinDegree; i < full.Keys.Count; i++) right.Keys.Add(full.Keys[i]);
        if (!full.IsLeaf)
            for (int i = MinDegree; i < full.Children.Count; i++) right.Children.Add(full.Children[i]);
        full.Keys.RemoveRange(MinDegree - 1, full.Keys.Count - (MinDegree - 1));
        if (!full.IsLeaf) full.Children.RemoveRange(MinDegree, full.Children.Count - MinDegree);
        parent.Keys.Insert(index, middle);
        parent.Children.Insert(index + 1, right);
    }

    internal bool Delete(int key)
    {
        if (!Contains(key)) return false;
        DeleteFrom(Root, key);
        if (Root.Keys.Count == 0 && !Root.IsLeaf) Root = Root.Children[0];
        Count--;
        return true;
    }

    private void DeleteFrom(BNode node, int key)
    {
        int index = node.Keys.BinarySearch(key);
        if (index >= 0)
        {
            if (node.IsLeaf) { node.Keys.RemoveAt(index); return; }
            BNode left = node.Children[index], right = node.Children[index + 1];
            if (left.Keys.Count >= MinDegree)
            {
                int predecessor = MaxKey(left);
                node.Keys[index] = predecessor;
                DeleteFrom(left, predecessor);
            }
            else if (right.Keys.Count >= MinDegree)
            {
                int successor = MinKey(right);
                node.Keys[index] = successor;
                DeleteFrom(right, successor);
            }
            else
            {
                Merge(node, index);
                DeleteFrom(node.Children[index], key);
            }
            return;
        }

        int child = ~index;
        if (node.Children[child].Keys.Count == MinDegree - 1)
        {
            Fill(node, child);
            if (child >= node.Keys.Count) child = node.Keys.Count;
        }
        DeleteFrom(node.Children[child], key);
    }

    private void Fill(BNode parent, int index)
    {
        if (index > 0 && parent.Children[index - 1].Keys.Count >= MinDegree) { BorrowFromPrevious(parent, index); return; }
        if (index + 1 < parent.Children.Count && parent.Children[index + 1].Keys.Count >= MinDegree) { BorrowFromNext(parent, index); return; }
        if (index + 1 < parent.Children.Count) Merge(parent, index);
        else Merge(parent, index - 1);
    }

    private void BorrowFromPrevious(BNode parent, int index)
    {
        BNode child = parent.Children[index], sibling = parent.Children[index - 1];
        child.Keys.Insert(0, parent.Keys[index - 1]);
        if (!sibling.IsLeaf) child.Children.Insert(0, sibling.Children[sibling.Children.Count - 1]);
        parent.Keys[index - 1] = sibling.Keys[sibling.Keys.Count - 1];
        sibling.Keys.RemoveAt(sibling.Keys.Count - 1);
        if (!sibling.IsLeaf) sibling.Children.RemoveAt(sibling.Children.Count - 1);
    }

    private void BorrowFromNext(BNode parent, int index)
    {
        BNode child = parent.Children[index], sibling = parent.Children[index + 1];
        child.Keys.Add(parent.Keys[index]);
        if (!sibling.IsLeaf) child.Children.Add(sibling.Children[0]);
        parent.Keys[index] = sibling.Keys[0];
        sibling.Keys.RemoveAt(0);
        if (!sibling.IsLeaf) sibling.Children.RemoveAt(0);
    }

    private void Merge(BNode parent, int index)
    {
        BNode left = parent.Children[index], right = parent.Children[index + 1];
        left.Keys.Add(parent.Keys[index]);
        for (int i = 0; i < right.Keys.Count; i++) left.Keys.Add(right.Keys[i]);
        for (int i = 0; i < right.Children.Count; i++) left.Children.Add(right.Children[i]);
        parent.Keys.RemoveAt(index);
        parent.Children.RemoveAt(index + 1);
    }

    private static int MaxKey(BNode node)
    {
        BNode current = node;
        while (!current.IsLeaf) current = current.Children[current.Children.Count - 1];
        return current.Keys[current.Keys.Count - 1];
    }

    private static int MinKey(BNode node)
    {
        BNode current = node;
        while (!current.IsLeaf) current = current.Children[0];
        return current.Keys[0];
    }

    internal List<int> Snapshot()
    {
        var keys = new List<int>();
        Walk(Root, keys);
        return keys;
    }

    private static void Walk(BNode node, List<int> keys)
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
        BNode current = Root;
        while (!current.IsLeaf) { current = current.Children[0]; depth++; }
        return depth;
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

    private string CheckNode(BNode node, int depth, long low, long high, bool isRoot, HashSet<int> leafDepths, ref int seen)
    {
        for (int i = 1; i < node.Keys.Count; i++) if (node.Keys[i - 1] >= node.Keys[i]) return "keys out of order in " + node.Describe();
        foreach (int key in node.Keys) if (key <= low || key >= high) return "key " + key + " violates its separator bounds in " + node.Describe();
        if (!isRoot && node.Keys.Count < MinDegree - 1) return node.Describe() + " underflows the minimum occupancy";
        if (node.Keys.Count > 2 * MinDegree - 1) return node.Describe() + " overflows the maximum occupancy";
        seen += node.Keys.Count;                              // CLRS stores real keys in internal nodes too
        if (node.IsLeaf)
        {
            leafDepths.Add(depth);
            return null;
        }
        if (node.Children.Count != node.Keys.Count + 1) return node.Describe() + " has the wrong number of children";
        for (int i = 0; i < node.Keys.Count; i++)
        {
            string problem = CheckNode(node.Children[i], depth + 1, low, node.Keys[i], false, leafDepths, ref seen);
            if (problem != null) return problem;
            low = node.Keys[i];
        }
        return CheckNode(node.Children[node.Keys.Count], depth + 1, low, high, false, leafDepths, ref seen);
    }
}

internal sealed class LatchTable
{
    private readonly int[] _readers;
    private readonly int[] _writers;

    internal long ReadAcquires, WriteAcquires, ReadWaits, WriteWaits;
    internal int MaxSimultaneousLatches;
    internal int ReaderOverlaps;                 // a reader entering a node another reader already holds
    internal int HeldLatches;

    internal LatchTable(int nodes)
    {
        _readers = new int[nodes];
        _writers = new int[nodes];
    }

    internal bool TryReadEnter(int node)
    {
        if (_writers[node] > 0) { ReadWaits++; return false; }
        if (_readers[node] > 0) ReaderOverlaps++;
        _readers[node]++;
        ReadAcquires++;
        HeldLatches++;
        if (HeldLatches > MaxSimultaneousLatches) MaxSimultaneousLatches = HeldLatches;
        return true;
    }

    internal void ReadExit(int node) { _readers[node]--; HeldLatches--; }

    internal bool TryWriteEnter(int node)
    {
        if (_writers[node] > 0 || _readers[node] > 0) { WriteWaits++; return false; }
        _writers[node]++;
        WriteAcquires++;
        HeldLatches++;
        if (HeldLatches > MaxSimultaneousLatches) MaxSimultaneousLatches = HeldLatches;
        return true;
    }

    internal void WriteExit(int node) { _writers[node]--; HeldLatches--; }

    internal string Violation()
    {
        for (int i = 0; i < _readers.Length; i++)
        {
            if (_readers[i] > 0 && _writers[i] > 0) return "node " + i + " carries readers and a writer at once";
            if (_readers[i] < 0 || _writers[i] < 0) return "node " + i + " has a negative latch count";
        }
        return null;
    }
}

internal sealed class Operation
{
    internal int Kind;                       // 0 search, 1 insert, 2 delete
    internal int Key;
    internal IEnumerator<int> Steps;
    internal bool Done;
    internal bool Result;
    internal bool Committed;                 // the atomic mutation step has run
    internal bool Counted;                   // the scheduler already replayed that commit
    internal long CommitAtRead = -1;         // commits completed when the leaf was inspected
    internal int CommitOrder;                // 1-based position in the commit replay
    internal int HeldLatches;
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void ApplyToReference(List<int> reference, int kind, int key)
    {
        if (kind == 1)
        {
            int at = reference.BinarySearch(key);
            if (at < 0) reference.Insert(~at, key);
        }
        else if (kind == 2)
        {
            int at = reference.BinarySearch(key);
            if (at >= 0) reference.RemoveAt(at);
        }
    }

    private static bool SameKeys(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    // ---- the simulated operations: exactly one atomic action per yield ---------------------------

    private static IEnumerator<int> SearchSteps(BTree tree, LatchTable latches, Operation op)
    {
        while (!latches.TryReadEnter(tree.Root.Id)) yield return 0;
        op.HeldLatches++;
        BNode current = tree.Root;
        yield return 0;
        while (!current.IsLeaf)
        {
            BNode child = current.Children[tree.ChildIndex(current, op.Key)];
            while (!latches.TryReadEnter(child.Id)) yield return 0;
            op.HeldLatches++;
            latches.ReadExit(current.Id);                     // coupling: child latched before parent released
            op.HeldLatches--;
            yield return 0;
            current = child;
        }
        op.CommitAtRead = tree.Commits;
        op.Result = current.Has(op.Key);
        latches.ReadExit(current.Id);
        op.HeldLatches--;
        yield return 0;
        op.Done = true;
    }

    private static IEnumerator<int> WriteSteps(BTree tree, LatchTable latches, Operation op)
    {
        var path = new List<int>();
        while (!latches.TryWriteEnter(tree.Root.Id)) yield return 0;
        path.Add(tree.Root.Id);
        op.HeldLatches++;
        yield return 0;
        BNode current = tree.Root;
        while (!current.IsLeaf)
        {
            BNode child = current.Children[tree.ChildIndex(current, op.Key)];
            while (!latches.TryWriteEnter(child.Id)) yield return 0;
            path.Add(child.Id);
            op.HeldLatches++;
            yield return 0;
            current = child;
        }
        op.Result = op.Kind == 1 ? tree.Insert(op.Key) : tree.Delete(op.Key);
        tree.Commits++;
        op.Committed = true;
        yield return 0;                                        // the mutation really is one atomic step
        for (int i = path.Count - 1; i >= 0; i--) { latches.WriteExit(path[i]); op.HeldLatches--; }
        yield return 0;
        op.Done = true;
    }

    /// <summary>
    /// Round-robin scheduler. Every step is atomic, so the tree between steps must equal the replay of
    /// the commits the scheduler has already seen, and the commits are seen in their real order.
    /// </summary>
    private static int RunCooperative(BTree tree, LatchTable latches, List<Operation> ops,
        List<List<int>> history, int stepBudget, out string problem)
    {
        problem = null;
        var reference = new List<int>();
        history.Add(new List<int>());                          // history[k] = state after k commits
        int steps = 0;
        bool progress = true;
        while (progress && steps < stepBudget)
        {
            progress = false;
            foreach (Operation op in ops)
            {
                if (op.Done) continue;
                if (!op.Steps.MoveNext()) { op.Done = true; continue; }
                steps++;
                progress = true;
                if (op.Committed && !op.Counted)
                {
                    op.Counted = true;
                    op.CommitOrder = history.Count;
                    ApplyToReference(reference, op.Kind, op.Key);
                    history.Add(new List<int>(reference));
                }
                problem = latches.Violation();
                if (problem != null) return steps;
                if (op.Kind == 0 && op.HeldLatches > 2) { problem = "a reader held " + op.HeldLatches + " latches"; return steps; }
                if (op.Kind != 0 && op.HeldLatches > tree.Depth() + 1) { problem = "a writer held " + op.HeldLatches + " latches"; return steps; }
                problem = tree.InvariantViolation();
                if (problem != null) return steps;
                if (!SameKeys(tree.Snapshot(), reference)) { problem = "the tree diverged from the committed replay"; return steps; }
            }
        }
        if (ops.Exists(o => !o.Done)) problem = "the scheduler ran out of budget with operations still pending";
        return steps;
    }

    private static bool AllReadsLinear(List<Operation> ops, List<List<int>> history, out int checkedReads)
    {
        checkedReads = 0;
        bool allLinear = true;
        foreach (Operation op in ops)
        {
            if (op.Kind != 0 || !op.Done) continue;
            int index = (int)Math.Min(op.CommitAtRead, history.Count - 1);
            if (op.Result != (history[index].BinarySearch(op.Key) >= 0)) allLinear = false;
            checkedReads++;
        }
        return allLinear;
    }

    public static void Main()
    {
        // ---- sequential sanity: the structure itself --------------------------------------------
        {
            var tree = new BTree(3);
            var reference = new List<int>();
            ulong state = 0x5EED774000000001UL;
            bool agree = true;
            string problem = "ok";
            for (int i = 0; i < 400; i++)
            {
                int key = SplitMix64.NextInt(ref state, 1000);
                bool added = tree.Insert(key);
                ApplyToReference(reference, 1, key);
                if (!SameKeys(tree.Snapshot(), reference)) { agree = false; problem = "insert " + key + " diverged"; break; }
            }
            Check(agree, "400 inserts match the sorted reference: " + problem);
            Check(tree.InvariantViolation() == null, "the tree satisfies its invariants after the inserts: " + tree.InvariantViolation());
            Check(tree.Count == reference.Count && tree.Count > 200, "the tree holds exactly the inserted keys (" + tree.Count + ")");

            int removed = 0;
            for (int i = 0; i < 300; i++)
            {
                int key = SplitMix64.NextInt(ref state, 1000);
                bool deleted = tree.Delete(key);
                if (deleted) removed++;
                ApplyToReference(reference, 2, key);
                if (!SameKeys(tree.Snapshot(), reference)) { agree = false; problem = "delete " + key + " diverged"; break; }
            }
            Check(agree, "300 deletions match the sorted reference: " + problem);
            Check(removed > 50, "the deletions really removed keys (" + removed + ")");
            Check(tree.InvariantViolation() == null, "the tree still satisfies its invariants: " + tree.InvariantViolation());
            Check(tree.Depth() <= 6, "a tree over 1000 keys stays shallow (depth " + tree.Depth() + ")");
            Console.WriteLine("sequential: n=" + tree.Count + " nodes=" + tree.NodeCount + " depth=" + tree.Depth() + " removed=" + removed);
        }

        // ---- boundary: the minimum order and draining the tree -----------------------------------
        {
            var tree = new BTree(2);
            for (int key = 1; key <= 60; key++) Check(tree.Insert(key), "insert " + key + " into the order-2 tree");
            Check(!tree.Insert(30), "a duplicate insert is refused");
            Check(tree.InvariantViolation() == null, "the order-2 tree satisfies its invariants: " + tree.InvariantViolation());
            Check(tree.Count == 60 && tree.Depth() >= 3, "60 keys in an order-2 tree need several levels (depth " + tree.Depth() + ")");
            var snapshot = tree.Snapshot();
            for (int i = 0; i < snapshot.Count; i++) Check(snapshot[i] == i + 1, "the snapshot is sorted and complete");
            for (int key = 1; key <= 60; key++) Check(tree.Delete(key), "delete " + key + " from the order-2 tree");
            Check(tree.Count == 0 && tree.Snapshot().Count == 0, "draining the tree leaves it empty");
            Check(tree.InvariantViolation() == null, "the drained tree still satisfies its invariants");
            Check(!tree.Delete(1) && !tree.Contains(1), "deleting from the drained tree is a no-op");
            Check(tree.Insert(5) && tree.Count == 1 && tree.Contains(5), "the drained tree accepts new keys again");
            Check(tree.InvariantViolation() == null, "the revived tree satisfies its invariants");

            var empty = new BTree(4);
            Check(!empty.Contains(1) && empty.Snapshot().Count == 0, "an empty tree contains nothing");
            Check(empty.InvariantViolation() == null, "the empty tree satisfies its invariants");
            bool threw = false;
            try { new BTree(1); } catch (ArgumentException) { threw = true; }
            Check(threw, "a minimum degree below 2 is rejected");
        }

        // ---- the concurrent simulation ------------------------------------------------------------
        {
            const int Writers = 10, Readers = 12, Capacity = 5000;
            var tree = new BTree(3);
            var latches = new LatchTable(Capacity);
            ulong state = 0xC0FFEE7740000002UL;
            var operations = new List<Operation>();
            for (int i = 0; i < Writers; i++)
            {
                var op = new Operation { Kind = i % 3 == 2 ? 2 : 1, Key = 1 + i * 7 + SplitMix64.NextInt(ref state, 5) };
                op.Steps = WriteSteps(tree, latches, op);
                operations.Add(op);
            }
            for (int i = 0; i < Readers; i++)
            {
                var op = new Operation { Kind = 0, Key = SplitMix64.NextInt(ref state, 100) };
                op.Steps = SearchSteps(tree, latches, op);
                operations.Add(op);
            }

            var history = new List<List<int>>();
            string problem;
            int steps = RunCooperative(tree, latches, operations, history, 200000, out problem);
            Check(problem == null, "the cooperative run finished cleanly: " + problem);
            Check(operations.TrueForAll(o => o.Done), "every operation completed, so the protocol is deadlock free");
            Check(steps < 200000, "the run needed only " + steps + " scheduler steps");
            Check(latches.WriteWaits > 0, "writers really had to wait for readers (" + latches.WriteWaits + " waits)");
            Check(latches.ReaderOverlaps > 0, "two readers really shared a node (" + latches.ReaderOverlaps + " overlaps)");
            Check(latches.MaxSimultaneousLatches >= 2, "several latches were held at once (" + latches.MaxSimultaneousLatches + ")");
            Check(tree.NodeCount < Capacity, "the latch table covered every node (" + tree.NodeCount + ")");
            Check(tree.InvariantViolation() == null, "the concurrently updated tree satisfies its invariants: " + tree.InvariantViolation());
            Check(SameKeys(tree.Snapshot(), history[history.Count - 1]), "the final tree matches the committed replay");
            Check(history.Count == tree.Commits + 1, "the replay history has one state per commit");
            int readCount;
            Check(AllReadsLinear(operations, history, out readCount), "every read answer matches a committed state");
            Check(readCount == Readers, "all " + Readers + " readers produced a linearizable answer");
            Console.WriteLine("concurrent run: steps=" + steps + " commits=" + tree.Commits + " readAcquires=" + latches.ReadAcquires
                + " writeAcquires=" + latches.WriteAcquires + " readWaits=" + latches.ReadWaits + " writeWaits=" + latches.WriteWaits
                + " maxLatches=" + latches.MaxSimultaneousLatches + " readerOverlaps=" + latches.ReaderOverlaps);
        }

        // ---- a reader-heavy run: three distinct writers against many readers ---------------------
        {
            const int Writers = 3, Readers = 25, Capacity = 5000;
            var tree = new BTree(4);
            var latches = new LatchTable(Capacity);
            var operations = new List<Operation>();
            for (int i = 0; i < Writers; i++)
            {
                var op = new Operation { Kind = 1, Key = i * 11 + 3 };
                op.Steps = WriteSteps(tree, latches, op);
                operations.Add(op);
            }
            for (int i = 0; i < Readers; i++)
            {
                var op = new Operation { Kind = 0, Key = i * 3 % 40 };
                op.Steps = SearchSteps(tree, latches, op);
                operations.Add(op);
            }
            var history = new List<List<int>>();
            string problem;
            int steps = RunCooperative(tree, latches, operations, history, 200000, out problem);
            Check(problem == null, "the reader-heavy run finished cleanly: " + problem);
            Check(operations.TrueForAll(o => o.Done), "every reader-heavy operation completed");
            Check(latches.ReaderOverlaps > 0, "readers overlapped on one node (" + latches.ReaderOverlaps + " times)");
            Check(tree.Count == Writers, "the three distinct keys are the whole content (" + tree.Count + ")");
            Check(tree.InvariantViolation() == null, "the reader-heavy tree satisfies its invariants");
            int readCount;
            Check(AllReadsLinear(operations, history, out readCount), "the reader-heavy reads are linearizable too");
            Check(readCount == Readers, "all " + Readers + " readers answered");
            Console.WriteLine("reader-heavy run: steps=" + steps + " overlaps=" + latches.ReaderOverlaps
                + " readWaits=" + latches.ReadWaits + " writeWaits=" + latches.WriteWaits + " n=" + tree.Count);
        }

        // ---- one reader against one writer, fully deterministic -----------------------------------
        {
            var tree = new BTree(2);
            var latches = new LatchTable(200);
            var writer = new Operation { Kind = 1, Key = 7 };
            var reader = new Operation { Kind = 0, Key = 7 };
            writer.Steps = WriteSteps(tree, latches, writer);
            reader.Steps = SearchSteps(tree, latches, reader);
            var operations = new List<Operation> { reader, writer };
            var history = new List<List<int>>();
            string problem;
            RunCooperative(tree, latches, operations, history, 10000, out problem);
            Check(problem == null, "the two-op run finished cleanly: " + problem);
            Check(writer.Result && tree.Contains(7), "the single writer inserted its key");
            int index = (int)Math.Min(reader.CommitAtRead, history.Count - 1);
            Check(reader.Result == (history[index].BinarySearch(7) >= 0), "the lone reader answered from a committed state");
            Check(tree.InvariantViolation() == null, "the two-op tree satisfies its invariants");
            Console.WriteLine("two-op run: readerSaw=" + reader.Result + " readerWaits=" + latches.ReadWaits
                + " writerWaits=" + latches.WriteWaits + " steps=" + history.Count);
        }

        Console.WriteLine("PASS");
    }
}
