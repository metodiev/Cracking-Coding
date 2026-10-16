// Task 623 - Implement hazard pointers memory reclamation in concurrent structures. (Tier: Expert, CTCI Ch 15 - Threads and Locks)
//
// A Treiber stack that frees a popped node immediately is broken: another thread may be about
// to dereference exactly that node. Hazard pointers make reclamation safe with two rules.
//   (1) A reader publishes the pointer it is about to use into one of its K per-thread hazard
//       slots and only then re-reads the shared root. The re-read is mandatory: when the root
//       moved, the node must not be touched at all.
//   (2) A popped node is RETIRED, never freed directly. As soon as the retire list reaches R
//       the retirer scans every thread's hazard slots and physically frees only the nodes that
//       no slot references.
// Safety sketch: a node enters the scanner's retire list only when it is popped, i.e. after
// some reader's root re-read observed it; that re-read is program-ordered after the hazard
// store, and the scan's slot reads happen after the retire, so the scan must observe the slot.
// Hence a node a reader has published is never freed underneath it. With R = K * threads + 1
// the number of unreclaimed retired nodes is bounded by R + K * threads.
//
// Concurrency is *simulated deterministically* - no threads, no sleeps, no timers. Every node
// operation accepts a callback invoked in the hazardous window (right after the shared root is
// read), so a test can splice a retire + scan of another simulated thread exactly there. The
// C# Interlocked/Volatile primitives used here are the real ones; only the scheduler is fake.
//
// Simplification (documented per the authoring contract): the stack is the classic Treiber
// stack and the reclaim machinery is the real hazard-pointer algorithm; only the thread
// interleaving is simulated. Verified invariants: no freed node is ever dereferenced, popped
// values form a permutation of the pushed values, and the retire list stays bounded.
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

#nullable enable

internal sealed class UseAfterFreeException : Exception
{
    public UseAfterFreeException(string message) : base(message) { }
}

internal sealed class HpNode
{
    public readonly int Id;
    public readonly int Value;
    public HpNode? Next;
    public bool Freed;

    public HpNode(int id, int value) { Id = id; Value = value; }
}

internal sealed class HpStack
{
    public readonly int K;
    private readonly HpNode?[][] hazards;
    private readonly List<HpNode> retired = new List<HpNode>();
    private readonly List<int> freedIds = new List<int>();
    private HpNode? head;
    private int nextId = 1;

    public int Threshold;
    public int ScansRun;
    public int UseAfterFrees;
    public int PopRetries;

    public HpStack(int threads, int slotsPerThread, int threshold)
    {
        K = slotsPerThread;
        Threshold = threshold;
        hazards = new HpNode?[threads][];
        for (int t = 0; t < threads; t++) hazards[t] = new HpNode?[slotsPerThread];
    }

    public HpNode? Head => Volatile.Read(ref head);
    public int RetiredCount => retired.Count;
    public IReadOnlyList<int> FreedIds => freedIds;

    public void Push(int value)
    {
        var node = new HpNode(nextId++, value);
        while (true)
        {
            HpNode? cur = Volatile.Read(ref head);
            node.Next = cur;
            if (ReferenceEquals(Interlocked.CompareExchange(ref head, node, cur), cur)) return;
        }
    }

    /// <summary>Safe pop: publish a hazard, re-validate the root, then CAS while protected.</summary>
    public bool TryPop(int thread, int slot, out int value, Action? afterReadHead = null)
    {
        value = 0;
        bool paused = false;
        while (true)
        {
            HpNode? cur = Volatile.Read(ref head);
            if (cur == null) return false;
            if (!paused) { paused = true; afterReadHead?.Invoke(); }
            hazards[thread][slot] = cur;                       // publish
            if (!ReferenceEquals(Volatile.Read(ref head), cur)) // re-validate
            {
                PopRetries++;
                hazards[thread][slot] = null;
                continue;
            }
            if (ReferenceEquals(Interlocked.CompareExchange(ref head, DerefNext(cur), cur), cur))
            {
                value = cur.Value;
                hazards[thread][slot] = null;
                Retire(cur);
                return true;
            }
            PopRetries++;
            hazards[thread][slot] = null;
        }
    }

    /// <summary>The unsafe idiom, kept only so a test can prove the use-after-free detector fires.</summary>
    public bool TryPopUnprotected(int thread, Action? afterReadHead, out int value)
    {
        value = 0;
        HpNode? cur = Volatile.Read(ref head);
        if (cur == null) return false;
        afterReadHead?.Invoke();
        HpNode? next = DerefNext(cur);   // no hazard, no re-validation: dangles on purpose
        if (ReferenceEquals(Interlocked.CompareExchange(ref head, next, cur), cur))
        {
            value = cur.Value;
            Retire(cur);
            return true;
        }
        return false;
    }

    private HpNode? DerefNext(HpNode node)
    {
        if (node.Freed)
        {
            UseAfterFrees++;
            throw new UseAfterFreeException("node " + node.Id + " was reclaimed while in use");
        }
        return node.Next;
    }

    private void Retire(HpNode node)
    {
        retired.Add(node);
        if (retired.Count >= Threshold) Scan();
    }

    /// <summary>Free every retired node that no thread's hazard slot references.</summary>
    public int Scan()
    {
        ScansRun++;
        var protectedIds = new HashSet<int>();
        foreach (HpNode?[] slots in hazards)
        {
            for (int i = 0; i < slots.Length; i++)
            {
                HpNode? h = Volatile.Read(ref slots[i]);
                if (h != null) protectedIds.Add(h.Id);
            }
        }
        int freed = 0;
        var keep = new List<HpNode>();
        foreach (HpNode node in retired)
        {
            if (protectedIds.Contains(node.Id)) { keep.Add(node); continue; }
            node.Freed = true;
            freedIds.Add(node.Id);
            freed++;
        }
        retired.Clear();
        retired.AddRange(keep);
        return freed;
    }

    /// <summary>Ids still reachable from the head (dereferences through the chain).</summary>
    public List<int> LiveIds()
    {
        var ids = new List<int>();
        HpNode? cur = Volatile.Read(ref head);
        while (cur != null)
        {
            if (cur.Freed) throw new UseAfterFreeException("live chain contains reclaimed node " + cur.Id);
            ids.Add(cur.Id);
            cur = cur.Next;
        }
        return ids;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int NextSeed(int seed) => (seed * 1103515245 + 12345) & 0x7FFFFFFF;

    private static void Baseline()
    {
        var stack = new HpStack(1, 2, 1);
        Check(!stack.TryPop(0, 0, out _), "pop on empty stack");
        for (int i = 1; i <= 6; i++) stack.Push(i);
        var got = new List<int>();
        while (stack.TryPop(0, 0, out int v)) got.Add(v);
        Check(string.Join(",", got) == "6,5,4,3,2,1", "LIFO order, got " + string.Join(",", got));
        Check(stack.LiveIds().Count == 0, "empty after draining");
        Check(stack.FreedIds.Count == 6 && stack.RetiredCount == 0, "all six retired nodes reclaimed");
        Check(stack.UseAfterFrees == 0, "no use-after-free in baseline");
    }

    private static void ProtectionWindow()
    {
        var stack = new HpStack(2, 2, 1);
        stack.Push(10);
        stack.Push(20);
        bool helperDone = false;
        Action pause = () =>
        {
            if (helperDone) return;
            helperDone = true;
            // Thread 1 pops the node thread 0 was reading; its retire crosses the threshold,
            // so the scan runs before thread 0 publishes its hazard.
            Check(stack.TryPop(1, 0, out int stolen), "helper pop succeeds");
            Check(stolen == 20, "helper stole the head");
        };
        Check(stack.TryPop(0, 0, out int value, pause), "protected pop still succeeds");
        Check(value == 10, "protected pop re-validated and took the next node, got " + value);
        Check(stack.PopRetries > 0, "the interleaving really forced a retry");
        Check(stack.UseAfterFrees == 0, "no use-after-free while protected");
        Check(stack.FreedIds.Count == 2 && new HashSet<int>(stack.FreedIds).Contains(2),
            "the stolen node and the node popped after the retry were both reclaimed");
        // The freed nodes are never handed out again: both nodes were popped exactly once above.
        Check(stack.LiveIds().Count == 0, "stack drained by the two pops");
    }

    private static void UnprotectedIsCaught()
    {
        var stack = new HpStack(2, 2, 1);
        stack.Push(10);
        stack.Push(20);
        bool helperDone = false;
        Action pause = () =>
        {
            if (helperDone) return;
            helperDone = true;
            Check(stack.TryPop(1, 0, out int stolen) && stolen == 20, "helper stole the head");
        };
        bool caught = false;
        try
        {
            stack.TryPopUnprotected(0, pause, out _);
        }
        catch (UseAfterFreeException)
        {
            caught = true;
        }
        Check(caught, "unprotected read after the reclaim is detected");
        Check(stack.UseAfterFrees == 1, "detector fired exactly once");
    }

    private static void BoundedRetireList()
    {
        int threads = 3, k = 2;
        var stack = new HpStack(threads, k, k * threads + 1);
        int seed = 999;
        var pushed = new List<int>();
        for (int i = 1; i <= 60; i++) { stack.Push(i); pushed.Add(i); }
        int pops = 0;
        while (stack.TryPop(0, 0, out _)) pops++;
        while (stack.TryPop(1, 0, out _)) pops++;
        while (stack.TryPop(2, 0, out _)) pops++;
        Check(pops == pushed.Count, "every pushed value was popped: " + pops);
        Check(stack.RetiredCount + stack.FreedIds.Count == pops, "each retired node is either freed or still retired");
        Check(stack.RetiredCount <= k * threads, "retire list never exceeds K * threads inside a scan");
        Check(stack.ScansRun >= 1, "at least one scan ran");
        Check(stack.UseAfterFrees == 0, "no use-after-free while draining");

        // Re-check the bound with fine-grained arithmetic on a fresh stack.
        var probe = new HpStack(threads, k, k * threads + 1);
        for (int i = 0; i < 25; i++)
        {
            probe.Push(i);
            Check(probe.TryPop(seed % threads, 0, out _), "probe pop");
            seed = NextSeed(seed);
            Check(probe.RetiredCount <= probe.Threshold, "retire list bounded by R after every retire");
        }
        Check(probe.FreedIds.Count + probe.RetiredCount == 25, "retire accounting is exact");
    }

    private static void Stress()
    {
        int threads = 3, k = 2;
        var stack = new HpStack(threads, k, k * threads + 1);
        var expected = new List<int>();
        var popped = new List<int>();
        var pushedValues = new List<int>();
        int seed = 4242;

        void Record(int v)
        {
            Check(expected.Contains(v), "popped value was pushed and not yet popped: " + v);
            expected.Remove(v);
            popped.Add(v);
        }

        for (int step = 0; step < 600; step++)
        {
            seed = NextSeed(seed);
            int thread = seed % threads;
            int choice = (seed / 7) % 5;
            if (choice < 3 || expected.Count == 0)
            {
                int value = 1000 + step;
                stack.Push(value);
                expected.Add(value);
                pushedValues.Add(value);
                continue;
            }
            int helper = (thread + 1) % threads;
            bool helperDone = false;
            Action pause = () =>
            {
                if (helperDone) return;
                helperDone = true;
                if (expected.Count == 0) return;
                if (stack.TryPop(helper, 0, out int hv)) Record(hv);
            };
            if (stack.TryPop(thread, 0, out int v, pause)) Record(v);
        }

        for (int t = 0; t < threads; t++)
        {
            while (stack.TryPop(t, 0, out int v)) Record(v);
        }
        Check(expected.Count == 0, "nothing left unaccounted for");
        Check(popped.Count == pushedValues.Count, "every pushed value was popped exactly once");
        Check(stack.LiveIds().Count == 0, "stack drained to empty");
        Check(stack.RetiredCount <= stack.K * threads, "bounded retired leftover, got " + stack.RetiredCount);
        Check(stack.FreedIds.Count + stack.RetiredCount == popped.Count, "freed + retired equals pops");
        var freed = new HashSet<int>(stack.FreedIds);
        foreach (int id in stack.LiveIds()) Check(!freed.Contains(id), "live node is not reclaimed");
        Check(stack.UseAfterFrees == 0, "no use-after-free under interleaving");
        Check(stack.PopRetries > 0, "the stress schedule produced real CAS contention");
        Console.WriteLine("stress: pops=" + popped.Count + " retries=" + stack.PopRetries
            + " scans=" + stack.ScansRun + " reclaimed=" + stack.FreedIds.Count);
    }

    public static void Main()
    {
        Baseline();
        ProtectionWindow();
        UnprotectedIsCaught();
        BoundedRetireList();
        Stress();
        Console.WriteLine("PASS");
    }
}
