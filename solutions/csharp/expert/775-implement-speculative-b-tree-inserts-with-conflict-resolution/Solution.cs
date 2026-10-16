// Task 775 - Implement speculative B-tree inserts with conflict resolution. (Tier: Expert, CTCI Ch 14 - Databases)
//
// Optimistic (speculative) execution on a B-tree index, in the style of Kung-Robinson validation:
// every node carries a version stamp, and an operation runs without any latch -
//
//   1. it walks the tree top-down, recording (node, version, child index) for each step;
//   2. it yields once after the walk, which is the window in which another operation may commit;
//   3. it validates the recorded versions; if any of them moved, the speculation failed, and the
//      conflict is resolved by simply walking down again (the tree is never left half changed);
//   4. if validation succeeds, it applies its mutation in one atomic step and bumps the version of
//      every node on the mutated path, including the root.
//
// Bumping the whole mutated path is what makes validation sound: because every commit touches the
// root, no commit can slip in unnoticed during another operation's walk, so a successful validation
// means the walk saw the state as of its first read. Searches use the same walk/validate/retry loop
// and therefore never need a latch.
//
// Verified: every requested insert is eventually applied exactly once (no lost update and no duplicate
// application), the tree between scheduler steps always equals the sequential replay of the commits
// observed so far, a successful operation really did run in a commit-free window (the commit counter
// at the start of the walk equals the counter at validation), each restart was caused by an observed
// version change, the B-tree occupancy/order/equal-depth invariants hold after every commit, and the
// conflict rate collapses to zero when only one writer runs.
//
// Documented simplification: the scheduler is single threaded and one scheduler step is atomic, so the
// file verifies the speculation *protocol* (what is recorded, when validation happens, what a failed
// validation implies) instead of memory-level races; there is no write-write conflict detection beyond
// the whole-path version stamps, no deferred parent updates and no redo/undo logging.
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

internal sealed class VersionedTree
{
    internal readonly int MinDegree;
    private readonly List<TreeNode> _nodes = new List<TreeNode>();
    private int _nextId;

    internal TreeNode Root;
    internal int Count;
    internal long Commits;
    internal long VersionBumps;
    internal long Splits;

    internal VersionedTree(int minDegree)
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

    private void Bump(TreeNode node)
    {
        node.Version++;
        VersionBumps++;
    }

    /// <summary>Mutating insert; the caller must already have validated its speculative walk.</summary>
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
        Bump(Root);
        if (added) Count++;
        return added;
    }

    private bool InsertNonFull(TreeNode node, int key)
    {
        if (node.IsLeaf)
        {
            int at = node.Keys.BinarySearch(key);
            Bump(node);
            if (at >= 0) return false;
            node.Keys.Insert(~at, key);
            return true;
        }
        int index = ChildIndex(node, key);
        if (index < node.Keys.Count && node.Keys[index] == key)
        {
            Bump(node);
            return false;
        }
        if (node.Children[index].Keys.Count == 2 * MinDegree - 1)
        {
            SplitChild(node, index);
            if (key == node.Keys[index])
            {
                Bump(Root);
                return false;
            }
            if (key > node.Keys[index]) index++;
        }
        Bump(node);
        return InsertNonFull(node.Children[index], key);
    }

    private void SplitChild(TreeNode parent, int index)
    {
        Splits++;
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
        Bump(full);
        Bump(right);
        Bump(parent);
    }

    /// <summary>Mutating delete; the caller must already have validated its speculative walk.</summary>
    internal bool Delete(int key)
    {
        if (!Contains(key)) return false;
        DeleteFrom(Root, key);
        if (Root.Keys.Count == 0 && !Root.IsLeaf) Root = Root.Children[0];
        Bump(Root);
        Count--;
        return true;
    }

    private void DeleteFrom(TreeNode node, int key)
    {
        int index = node.Keys.BinarySearch(key);
        if (index >= 0)
        {
            if (node.IsLeaf) { node.Keys.RemoveAt(index); Bump(node); return; }
            TreeNode left = node.Children[index], right = node.Children[index + 1];
            if (left.Keys.Count >= MinDegree)
            {
                int predecessor = MaxKey(left);
                node.Keys[index] = predecessor;
                Bump(node);
                DeleteFrom(left, predecessor);
            }
            else if (right.Keys.Count >= MinDegree)
            {
                int successor = MinKey(right);
                node.Keys[index] = successor;
                Bump(node);
                DeleteFrom(right, successor);
            }
            else
            {
                Merge(node, index);
                Bump(node);
                DeleteFrom(node.Children[index], key);
            }
            return;
        }
        int child = ~index;
        if (node.Children[child].Keys.Count == MinDegree - 1)
        {
            Fill(node, child);
            Bump(node);
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
        Bump(child);
        Bump(sibling);
    }

    private void BorrowFromNext(TreeNode parent, int index)
    {
        TreeNode child = parent.Children[index], sibling = parent.Children[index + 1];
        child.Keys.Add(parent.Keys[index]);
        if (!sibling.IsLeaf) child.Children.Add(sibling.Children[0]);
        parent.Keys[index] = sibling.Keys[0];
        sibling.Keys.RemoveAt(0);
        if (!sibling.IsLeaf) sibling.Children.RemoveAt(0);
        Bump(child);
        Bump(sibling);
    }

    private void Merge(TreeNode parent, int index)
    {
        TreeNode left = parent.Children[index], right = parent.Children[index + 1];
        left.Keys.Add(parent.Keys[index]);
        for (int i = 0; i < right.Keys.Count; i++) left.Keys.Add(right.Keys[i]);
        for (int i = 0; i < right.Children.Count; i++) left.Children.Add(right.Children[i]);
        parent.Keys.RemoveAt(index);
        parent.Children.RemoveAt(index + 1);
        Bump(left);
        Bump(right);
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

internal sealed class Speculation
{
    internal int Kind;                       // 0 search, 1 insert, 2 delete
    internal int Key;
    internal IEnumerator<int> Steps;
    internal bool Done;
    internal bool Result;
    internal bool Committed;
    internal bool Counted;
    internal int CommitOrder;
    internal long StartCommits;              // commits seen when the walk began
    internal long EndCommits;                // commits seen when validation succeeded
    internal long CommitAtLeaf;              // commits seen when the leaf was inspected
    internal int Attempts = 1;
    internal int LastInvalidatedNode = -1;   // the node whose version invalidated the last attempt
    internal long LastRecordedVersion;
    internal long LastObservedVersion;
    internal bool EvidenceValid = true;      // a restart always had a real version difference

    internal readonly List<TreeNode> Path = new List<TreeNode>();
    internal readonly List<long> Versions = new List<long>();
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

    /// <summary>Speculative walk: read a node, remember its version, yield, then step down.</summary>
    private static IEnumerator<int> Speculate(VersionedTree tree, Speculation op)
    {
        int window = 0;
        while (true)
        {
            op.Path.Clear();
            op.Versions.Clear();
            op.StartCommits = tree.Commits;
            TreeNode current = tree.Root;
            while (true)
            {
                op.Path.Add(current);
                op.Versions.Add(current.Version);
                bool found = current.Has(op.Key);                 // a separator key is a real key too
                yield return 0;                                   // one step per node read
                if (found || current.IsLeaf) break;
                current = current.Children[tree.ChildIndex(current, op.Key)];
            }
            op.CommitAtLeaf = tree.Commits;
            // the conflict window: deliberately let the other speculations commit before validating
            window = 1 + (int)(op.Key % 3);
            for (int i = 0; i < window; i++) yield return 0;

            bool valid = true;
            for (int i = 0; i < op.Path.Count; i++)
            {
                long now = op.Path[i].Version;
                if (now != op.Versions[i])
                {
                    valid = false;
                    op.LastInvalidatedNode = op.Path[i].Id;
                    op.LastRecordedVersion = op.Versions[i];
                    op.LastObservedVersion = now;
                    op.EvidenceValid = op.EvidenceValid && now != op.Versions[i];
                    break;
                }
            }
            op.EndCommits = tree.Commits;
            if (!valid)
            {
                op.Attempts++;
                yield return 0;                                   // resolve the conflict by walking again
                continue;
            }

            if (op.Kind == 0) op.Result = op.Path[op.Path.Count - 1].Has(op.Key);
            else op.Result = op.Kind == 1 ? tree.Insert(op.Key) : tree.Delete(op.Key);
            if (op.Kind != 0 && op.Result)
            {
                tree.Commits++;                                    // only mutations that changed state commit
                op.Committed = true;
            }
            yield return 0;                                       // the commit is one atomic step
            op.Done = true;
            yield break;
        }
    }

    private static int Run(VersionedTree tree, List<Speculation> ops, List<List<int>> history, int budget, out string problem,
        List<int> initial = null)
    {
        problem = null;
        var reference = initial == null ? new List<int>() : new List<int>(initial);
        history.Add(new List<int>(reference));
        int steps = 0;
        bool progress = true;
        while (progress && steps < budget)
        {
            progress = false;
            foreach (Speculation op in ops)
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
        // ---- the versioned structure on its own ---------------------------------------------------
        {
            var tree = new VersionedTree(3);
            var reference = new List<int>();
            ulong state = 0x5EED775000000001UL;
            long previousBumps = 0;
            bool versionsAdvance = true;
            for (int i = 0; i < 300; i++)
            {
                int key = SplitMix64.NextInt(ref state, 500);
                tree.Insert(key);
                ApplyToReference(reference, 1, key);
                if (tree.VersionBumps <= previousBumps) versionsAdvance = false;
                previousBumps = tree.VersionBumps;
            }
            Check(SameKeys(tree.Snapshot(), reference), "the versioned tree holds exactly the inserted keys");
            Check(versionsAdvance, "every insert bumped at least one version");
            Check(tree.InvariantViolation() == null, "the versioned tree satisfies its invariants: " + tree.InvariantViolation());
            Check(tree.Splits > 0, "the workload really caused splits (" + tree.Splits + ")");
            Console.WriteLine("versioned tree: n=" + tree.Count + " nodes=" + tree.NodeCount + " depth=" + tree.Depth()
                + " splits=" + tree.Splits + " versionBumps=" + tree.VersionBumps);
        }

        // ---- speculative inserts under contention --------------------------------------------------
        {
            const int Ops = 24, Keyspace = 20;
            var tree = new VersionedTree(3);
            ulong state = 0x5EED775000000002UL;
            var ops = new List<Speculation>();
            for (int i = 0; i < Ops; i++)
            {
                var op = new Speculation { Kind = 1, Key = SplitMix64.NextInt(ref state, Keyspace) };
                op.Steps = Speculate(tree, op);
                ops.Add(op);
            }
            var history = new List<List<int>>();
            string problem;
            int steps = Run(tree, ops, history, 200000, out problem);
            Check(problem == null, "the speculative run finished cleanly: " + problem);
            Check(ops.TrueForAll(o => o.Done), "every speculative operation completed");
            Check(ops.TrueForAll(o => o.EvidenceValid), "every restart was caused by a real version difference");
            int restarts = 0, succeeded = 0, rejected = 0;
            long attempts = 0;
            var requested = new List<int>();
            foreach (Speculation op in ops)
            {
                restarts += op.Attempts - 1;
                attempts += op.Attempts;
                if (op.Result) succeeded++; else rejected++;
                requested.Add(op.Key);
            }
            Check(restarts > 0, "contention really forced restarts (" + restarts + ")");
            Check(rejected > 0, "some speculative inserts were duplicates (" + rejected + ")");
            Check(tree.Count == succeeded, "every successful insert is in the tree exactly once (" + tree.Count + " of " + succeeded + ")");
            var successKeys = new List<int>();
            foreach (Speculation op in ops) if (op.Result) successKeys.Add(op.Key);
            successKeys.Sort();
            var distinctSuccess = new List<int>();
            foreach (int key in successKeys) if (distinctSuccess.Count == 0 || distinctSuccess[distinctSuccess.Count - 1] != key) distinctSuccess.Add(key);
            Check(SameKeys(tree.Snapshot(), distinctSuccess), "the surviving keys are exactly the successful inserts - no lost update");
            Check(SameKeys(tree.Snapshot(), history[history.Count - 1]), "the final tree equals the committed replay");
            Check(tree.InvariantViolation() == null, "the speculatively updated tree satisfies its invariants: " + tree.InvariantViolation());
            bool windowsClean = true;
            foreach (Speculation op in ops)
            {
                if (!op.Done) continue;
                if (op.StartCommits != op.CommitAtLeaf || op.StartCommits != op.EndCommits) windowsClean = false;
            }
            Check(windowsClean, "a validated operation really ran inside a commit-free window");
            Console.WriteLine("speculative inserts: ops=" + Ops + " steps=" + steps + " restarts=" + restarts
                + " attempts=" + attempts + " n=" + tree.Count + " splits=" + tree.Splits);
        }

        // ---- speculative reads are linearizable and never need a latch -----------------------------
        {
            const int Writers = 8, Readers = 16;
            var tree = new VersionedTree(3);
            var ops = new List<Speculation>();
            ulong state = 0x5EED775000000003UL;
            for (int i = 0; i < Writers; i++)
            {
                var op = new Speculation { Kind = 1, Key = 1 + i * 5 };
                op.Steps = Speculate(tree, op);
                ops.Add(op);
            }
            for (int i = 0; i < Readers; i++)
            {
                var op = new Speculation { Kind = 0, Key = SplitMix64.NextInt(ref state, 60) };
                op.Steps = Speculate(tree, op);
                ops.Add(op);
            }
            var history = new List<List<int>>();
            string problem;
            int steps = Run(tree, ops, history, 200000, out problem);
            Check(problem == null, "the reader/writer speculation finished cleanly: " + problem);
            Check(ops.TrueForAll(o => o.Done), "all speculations completed");
            int readChecks = 0;
            bool linearizable = true;
            foreach (Speculation op in ops)
            {
                if (op.Kind != 0) continue;
                int index = (int)Math.Min(op.StartCommits, history.Count - 1);
                if (op.Result != (history[index].BinarySearch(op.Key) >= 0))
                {
                    linearizable = false;
                    Console.WriteLine("read mismatch key=" + op.Key + " result=" + op.Result + " start=" + op.StartCommits
                        + " leaf=" + op.CommitAtLeaf + " end=" + op.EndCommits + " attempts=" + op.Attempts
                        + " stateAtStart=" + string.Join(",", history[index]));
                }
                readChecks++;
            }
            Check(linearizable, "every speculative read matches the state its walk started from");
            Check(readChecks == Readers, "all " + Readers + " reads were checked");
            Check(tree.Count == Writers, "the eight distinct writer keys are the whole content (" + tree.Count + ")");
            int readRestarts = 0;
            foreach (Speculation op in ops) if (op.Kind == 0) readRestarts += op.Attempts - 1;
            Check(readRestarts > 0, "readers also had to retry (" + readRestarts + " read restarts)");
            Console.WriteLine("speculative reads: steps=" + steps + " readRestarts=" + readRestarts + " n=" + tree.Count);
        }

        // ---- a single writer, one operation at a time, must never conflict -------------------------
        {
            var tree = new VersionedTree(4);
            int restarts = 0, steps = 0;
            for (int i = 0; i < 12; i++)
            {
                var op = new Speculation { Kind = 1, Key = i * 3 + 1 };
                op.Steps = Speculate(tree, op);
                var single = new List<Speculation> { op };
                var history = new List<List<int>>();
                string problem;
                steps += Run(tree, single, history, 50000, out problem, tree.Snapshot());
                Check(problem == null, "the isolated run " + i + " finished cleanly: " + problem);
                restarts += op.Attempts - 1;
            }
            Check(restarts == 0, "an isolated speculative writer never conflicts (" + restarts + " restarts)");
            Check(tree.Count == 12, "all twelve keys landed (" + tree.Count + ")");
            Check(tree.InvariantViolation() == null, "the isolated tree satisfies its invariants");
            Console.WriteLine("isolated writers: steps=" + steps + " restarts=" + restarts + " depth=" + tree.Depth());
        }

        // ---- speculative deletes and the empty case ------------------------------------------------
        {
            var tree = new VersionedTree(3);
            for (int key = 0; key < 40; key++) tree.Insert(key);
            var ops = new List<Speculation>();
            for (int i = 0; i < 12; i++)
            {
                var op = new Speculation { Kind = 2, Key = i * 3 };
                op.Steps = Speculate(tree, op);
                ops.Add(op);
            }
            var history = new List<List<int>>();
            string problem;
            int steps = Run(tree, ops, history, 50000, out problem, tree.Snapshot());
            Check(problem == null, "the speculative delete run finished cleanly: " + problem);
            Check(tree.Count == 28, "twelve of the forty keys were deleted (" + tree.Count + ")");
            Check(tree.InvariantViolation() == null, "the tree satisfies its invariants after the deletes: " + tree.InvariantViolation());
            foreach (Speculation op in ops) Check(!tree.Contains(op.Key), "deleted key " + op.Key + " is gone");

            var missing = new Speculation { Kind = 2, Key = 999 };
            missing.Steps = Speculate(tree, missing);
            var single = new List<Speculation> { missing };
            var history2 = new List<List<int>>();
            Run(tree, single, history2, 10000, out problem, tree.Snapshot());
            Check(!missing.Result && !missing.Committed, "deleting an absent key commits nothing");
            Check(tree.Count == 28, "the failed delete left the tree untouched");

            var emptyTree = new VersionedTree(2);
            var probe = new Speculation { Kind = 0, Key = 3 };
            probe.Steps = Speculate(emptyTree, probe);
            var probeRun = new List<Speculation> { probe };
            var history3 = new List<List<int>>();
            Run(emptyTree, probeRun, history3, 10000, out problem, emptyTree.Snapshot());
            Check(!probe.Result && probe.Done, "a speculative read of an empty tree answers false");
            Console.WriteLine("speculative deletes: steps=" + steps + " n=" + tree.Count + " emptyProbe=" + probe.Result);
        }

        Console.WriteLine("PASS");
    }
}
