// Task 356 - Implement lock-free stack outline (CAS). (Tier: Intermediate, CTCI Ch 15)
using System;
using System.Collections.Generic;
using System.Threading;

public static class Solution
{
    private const int Threads = 4;
    private const int PushesPerThread = 250;

    private class Node
    {
        public int Value { get; }
        public Node? Next { get; set; }
        public Node(int value) { Value = value; }
    }

    /// <summary>
    /// Michael-Scott style lock-free stack: push/pop are Interlocked.CompareExchange retry
    /// loops on the single head pointer. Pop() returns null, the documented sentinel, on an
    /// empty stack; pushed values are ints and can never be null. The ABA hazard of pointer
    /// CAS is documented but cannot occur here: nodes are never recycled during a run, and
    /// any lost CAS retries with a freshly read head.
    /// </summary>
    private class LockFreeStack
    {
        private Node? head;
        private long casAttempts;
        private long casFailures;

        public long CasAttempts() { return Interlocked.Read(ref casAttempts); }
        public long CasFailures() { return Interlocked.Read(ref casFailures); }

        public int Push(int value)
        {
            int retries = 0;
            while (true)
            {
                Node? current = Volatile.Read(ref head);
                var node = new Node(value) { Next = current };
                Interlocked.Increment(ref casAttempts);
                if (Interlocked.CompareExchange(ref head, node, current) == current) return retries;
                Interlocked.Increment(ref casFailures);
                retries++;
            }
        }

        /// <summary>Pops the top value; returns null (the documented sentinel) when empty.</summary>
        public int? Pop()
        {
            while (true)
            {
                Node? current = Volatile.Read(ref head);
                if (current == null) return null;
                Interlocked.Increment(ref casAttempts);
                if (Interlocked.CompareExchange(ref head, current.Next, current) == current)
                {
                    return current.Value;
                }
                Interlocked.Increment(ref casFailures);
            }
        }

        public bool IsEmpty() { return Volatile.Read(ref head) == null; }

        public List<int> SnapshotValues()
        {
            var values = new List<int>();
            for (Node? node = Volatile.Read(ref head); node != null; node = node.Next) values.Add(node.Value);
            return values;
        }
    }

    public static void Main()
    {
        // Single-threaded script: LIFO order and the empty sentinel.
        var stack = new LockFreeStack();
        Check(stack.Pop() == null, "pop on an empty stack returns the sentinel");
        for (int value = 1; value <= 5; value++) stack.Push(value);
        var drained = new List<int>();
        for (int i = 0; i < 5; i++) drained.Add(stack.Pop()!.Value);
        Check(ListsEqual(drained, new List<int> { 5, 4, 3, 2, 1 }), "LIFO order");
        Check(stack.Pop() == null, "empty again after draining");
        Check(stack.CasFailures() == 0, "single-threaded script never loses a CAS");

        // Concurrent pushes: no node is lost.
        var shared = new LockFreeStack();
        int total = Threads * PushesPerThread;
        var pushers = new Thread[Threads];
        for (int t = 0; t < Threads; t++)
        {
            int threadIndex = t;
            pushers[t] = new Thread(() =>
            {
                for (int offset = 0; offset < PushesPerThread; offset++)
                {
                    shared.Push(threadIndex * PushesPerThread + offset);
                }
            });
            pushers[t].Start();
        }
        foreach (var pusher in pushers)
        {
            pusher.Join(10000);                    // timeout sanity: must terminate
            Check(!pusher.IsAlive, "pusher did not finish in time");
        }

        var values = shared.SnapshotValues();
        Check(values.Count == total, "no node lost under concurrent pushes");
        var sorted = new List<int>(values);
        sorted.Sort();
        Check(ListsEqual(sorted, Range(total)), "every pushed value present exactly once");

        // Concurrent pops: no node is lost or duplicated.
        var poppedPerThread = new List<int>[Threads];
        for (int i = 0; i < Threads; i++) poppedPerThread[i] = new List<int>();
        var poppers = new Thread[Threads];
        for (int t = 0; t < Threads; t++)
        {
            int threadIndex = t;
            poppers[t] = new Thread(() =>
            {
                var local = new List<int>();
                while (true)
                {
                    int? value = shared.Pop();
                    if (value == null) break;
                    local.Add(value.Value);
                }
                poppedPerThread[threadIndex] = local;
            });
            poppers[t].Start();
        }
        foreach (var popper in poppers)
        {
            popper.Join(10000);
            Check(!popper.IsAlive, "popper did not finish in time");
        }

        var allPopped = new List<int>();
        foreach (var chunk in poppedPerThread) allPopped.AddRange(chunk);
        Check(allPopped.Count == total, "no node lost under concurrent pops");
        allPopped.Sort();
        Check(ListsEqual(allPopped, Range(total)), "every pushed value popped exactly once");
        Check(shared.Pop() == null, "stack is empty again");
        Check(shared.IsEmpty(), "empty state after the concurrent phase");

        // Exact CAS accounting: every successful push/pop is one attempt, every retry one
        // failure. The poppers that stopped with the sentinel made no CAS attempt at all.
        Check(shared.CasFailures() >= 0, "CAS failures are non-negative (they may be 0)");
        Check(shared.CasAttempts() == 2L * total + shared.CasFailures(),
              "attempts == successful operations + failures");

        Console.WriteLine("PASS");
    }

    private static List<int> Range(int total)
    {
        var result = new List<int>();
        for (int i = 0; i < total; i++) result.Add(i);
        return result;
    }

    private static bool ListsEqual(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
