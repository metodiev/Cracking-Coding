// Task 347 - Read-write lock with writer preference (no writer starvation). (Tier: Intermediate, CTCI Ch 15)
using System;
using System.Collections.Generic;
using System.Threading;

public class ReadWriteLock
{
    private readonly object gate = new object();
    private int readers;
    private bool writer;
    private int writersWaiting;

    public void AcquireRead() => AcquireRead(null);

    public void AcquireRead(Action? onWaiting)
    {
        lock (gate)
        {
            bool notified = false;
            while (writer || writersWaiting > 0)
            {
                if (!notified && onWaiting != null)
                {
                    onWaiting();                       // fired while holding the monitor
                    notified = true;
                }
                Monitor.Wait(gate);
            }
            readers++;
        }
    }

    public void ReleaseRead()
    {
        lock (gate)
        {
            readers--;
            if (readers == 0) Monitor.PulseAll(gate);
        }
    }

    public void AcquireWrite() => AcquireWrite(null);

    public void AcquireWrite(Action? onWaiting)
    {
        lock (gate)
        {
            writersWaiting++;
            bool notified = false;
            while (writer || readers > 0)
            {
                if (!notified && onWaiting != null)
                {
                    onWaiting();
                    notified = true;
                }
                Monitor.Wait(gate);
            }
            writersWaiting--;
            writer = true;
        }
    }

    public void ReleaseWrite()
    {
        lock (gate)
        {
            writer = false;
            Monitor.PulseAll(gate);
        }
    }

    public int[] State()
    {
        lock (gate) return new[] { readers, writer ? 1 : 0, writersWaiting };
    }
}

/// <summary>Invariant tracker: readers and writers must never overlap.</summary>
public class LockAuditor
{
    private readonly object gate = new object();
    private int activeReaders;
    private int activeWriters;
    private int maxReaders;
    private int violations;
    private readonly List<string> grants = new List<string>();

    public void EnterRead(string tag)
    {
        lock (gate)
        {
            if (activeWriters > 0) violations++;
            activeReaders++;
            if (activeReaders > maxReaders) maxReaders = activeReaders;
            grants.Add("R:" + tag);
        }
    }

    public void ExitRead()
    {
        lock (gate) activeReaders--;
    }

    public void EnterWrite(string tag)
    {
        lock (gate)
        {
            if (activeWriters > 0 || activeReaders > 0) violations++;
            activeWriters++;
            grants.Add("W:" + tag);
        }
    }

    public void ExitWrite()
    {
        lock (gate) activeWriters--;
    }

    public int[] Counters()
    {
        lock (gate) return new[] { activeReaders, activeWriters, maxReaders, violations };
    }

    public List<string> Grants()
    {
        lock (gate) return new List<string>(grants);
    }
}

public class Gate
{
    public readonly ManualResetEventSlim Started = new ManualResetEventSlim(false);
    public readonly ManualResetEventSlim Entered = new ManualResetEventSlim(false);
    public readonly ManualResetEventSlim Registered = new ManualResetEventSlim(false);
    public readonly ManualResetEventSlim Release = new ManualResetEventSlim(false);
    public readonly ManualResetEventSlim Done = new ManualResetEventSlim(false);
}

public static class Solution
{
    /// <summary>
    /// Policy (documented): once a writer waits, newly arriving readers block; waiting writers are
    /// served when the current readers/writer drain, so readers cannot starve writers.  The scripted
    /// demo uses real threads, but every ordering decision passes through ManualResetEventSlim gates;
    /// acquire() takes an optional callback that fires once the caller has registered as waiting
    /// while still holding the monitor, so a release cannot slip between registration and Wait (no
    /// lost wake-up).  Bounded waits are liveness checks only; threads are background threads so a
    /// stuck run fails fast.
    /// </summary>
    public static void Main()
    {
        var rw = new ReadWriteLock();
        var auditor = new LockAuditor();
        var gates = new Dictionary<string, Gate>();
        foreach (string tag in new[] { "R1", "R2", "R3", "R4", "W1", "W2" }) gates[tag] = new Gate();
        var threads = new List<Thread>();

        Start(gates, threads, rw, auditor, "R1", false);
        Start(gates, threads, rw, auditor, "R2", false);
        Check(Await(gates["R1"].Entered) && Await(gates["R2"].Entered), "both readers hold the lock");
        Check(string.Join(",", auditor.Counters()) == "2,0,2,0", "two readers active, max 2");
        Check(string.Join(",", rw.State()) == "2,0,0", "rw state: 2 readers");

        Start(gates, threads, rw, auditor, "W1", true);
        Check(Await(gates["W1"].Registered), "W1 registered as waiting");
        Check(!gates["W1"].Entered.IsSet, "W1 not granted yet");
        Check(string.Join(",", rw.State()) == "2,0,1", "rw state: one writer waiting");

        Start(gates, threads, rw, auditor, "R3", true);
        Check(Await(gates["R3"].Registered), "R3 registered as waiting");
        Check(!gates["R3"].Entered.IsSet, "R3 blocked by the waiting writer (writer preference)");

        gates["R1"].Release.Set();
        gates["R2"].Release.Set();
        Check(Await(gates["R1"].Done) && Await(gates["R2"].Done), "readers drained");
        Check(Await(gates["W1"].Entered), "W1 acquired after readers drained");
        Check(!gates["R3"].Entered.IsSet, "R3 still blocked while the writer holds");
        Check(string.Join(",", auditor.Counters()) == "0,1,2,0", "writer active, no readers");
        Check(string.Join(",", rw.State()) == "0,1,0", "rw state: writer holds");

        gates["W1"].Release.Set();
        Check(Await(gates["W1"].Done), "W1 done");
        Check(Await(gates["R3"].Entered), "R3 finally granted after the writer");
        Check(string.Join(",", auditor.Counters()) == "1,0,2,0", "reader active again");
        gates["R3"].Release.Set();
        Check(Await(gates["R3"].Done), "R3 done");

        Start(gates, threads, rw, auditor, "W2", false);
        Check(Await(gates["W2"].Entered), "W2 acquired immediately");
        Start(gates, threads, rw, auditor, "R4", true);
        Check(Await(gates["R4"].Registered), "R4 registered as waiting");
        Check(!gates["R4"].Entered.IsSet, "R4 blocked behind the writer");
        gates["W2"].Release.Set();
        Check(Await(gates["W2"].Done), "W2 done");
        Check(Await(gates["R4"].Entered), "R4 granted");
        gates["R4"].Release.Set();
        Check(Await(gates["R4"].Done), "R4 done");

        foreach (Thread thread in threads) Check(thread.Join(TimeSpan.FromSeconds(15)), "thread terminated");

        int[] counters = auditor.Counters();
        Check(counters[3] == 0, "readers and writers never overlapped");
        Check(counters[2] == 2, "two readers shared the lock at peak");
        Check(string.Join(",", auditor.Grants()) == "R:R1,R:R2,W:W1,R:R3,W:W2,R:R4", "grant order");
        Check(counters[0] == 0 && counters[1] == 0, "all locks released");
        Check(string.Join(",", rw.State()) == "0,0,0", "final rw state empty");
        Console.WriteLine("PASS");
    }

    private static bool Await(ManualResetEventSlim signal) => signal.Wait(TimeSpan.FromSeconds(15));

    private static void Start(Dictionary<string, Gate> gates, List<Thread> threads, ReadWriteLock rw,
                              LockAuditor auditor, string tag, bool registerFirst)
    {
        Gate gate = gates[tag];
        Thread thread = new Thread(() =>
        {
            gate.Started.Set();
            Action? onWaiting = registerFirst ? gate.Registered.Set : (Action?)null;
            if (tag.StartsWith("R"))
            {
                rw.AcquireRead(onWaiting);
                auditor.EnterRead(tag);
            }
            else
            {
                rw.AcquireWrite(onWaiting);
                auditor.EnterWrite(tag);
            }
            gate.Entered.Set();
            bool released = gate.Release.Wait(TimeSpan.FromSeconds(15));
            if (tag.StartsWith("R"))
            {
                auditor.ExitRead();
                rw.ReleaseRead();
            }
            else
            {
                auditor.ExitWrite();
                rw.ReleaseWrite();
            }
            if (released) gate.Done.Set();
        })
        { IsBackground = true, Name = tag };
        threads.Add(thread);
        thread.Start();
        Check(Await(gate.Started), tag + " never started");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
