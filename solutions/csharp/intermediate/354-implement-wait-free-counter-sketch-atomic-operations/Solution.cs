// Task 354 - Implement wait-free counter sketch (atomic operations). (Tier: Intermediate, CTCI Ch 15)
using System;
using System.Threading;

public static class Solution
{
    private const int Threads = 8;
    private const int Increments = 20000;

    /// <summary>
    /// CAS counter sketch: Increment() is an Interlocked.CompareExchange retry loop
    /// instead of +=. Strictly this is lock-free, not wait-free (a thread can be
    /// overtaken indefinitely); the wait-free primitive behind "wait-free counters" is
    /// fetch-and-add, which .NET exposes as Interlocked.Add. The retry counters are
    /// asserted >= 0 and may legitimately be 0 when the scheduler never interleaves a
    /// read with its CAS; the afterRead hook forces one lost race so the retry path is
    /// exercised deterministically.
    /// </summary>
    private class CasCounter
    {
        private long value;
        private long casAttempts;
        private long casFailures;

        public long Get() { return Interlocked.Read(ref value); }
        public long CasAttempts() { return Interlocked.Read(ref casAttempts); }
        public long CasFailures() { return Interlocked.Read(ref casFailures); }

        public bool CompareAndSwap(long expected, long newValue)
        {
            Interlocked.Increment(ref casAttempts);
            if (Interlocked.CompareExchange(ref value, newValue, expected) == expected) return true;
            Interlocked.Increment(ref casFailures);
            return false;
        }

        public long Increment() { return Increment(null); }

        /// <summary>afterRead runs once between the read and the CAS of the first attempt.</summary>
        public long Increment(Action<long>? afterRead)
        {
            long retries = 0;
            Action<long>? hook = afterRead;
            while (true)
            {
                long current = Interlocked.Read(ref value);
                if (hook != null)
                {
                    hook(current);
                    hook = null;
                }
                if (CompareAndSwap(current, current + 1)) return retries;
                retries++;
            }
        }
    }

    public static void Main()
    {
        // Single-threaded baseline: no contention, no retries.
        var counter = new CasCounter();
        for (int i = 0; i < 100; i++) Check(counter.Increment() == 0, "uncontended CAS never retries");
        Check(counter.Get() == 100, "baseline total");
        Check(counter.CasAttempts() == 100 && counter.CasFailures() == 0, "baseline CAS accounting");

        // Deterministic retry-path demonstration: a concurrent writer bumps the value after
        // our read, so the first CAS fails (retries == 1) and the loop rereads and succeeds.
        var forced = new CasCounter();
        long retries = forced.Increment(current => forced.CompareAndSwap(current, current + 1));
        Check(retries == 1, "stale read forces exactly one retry");
        Check(forced.Get() == 2, "both the stealing write and our retry are counted");

        // Concurrent increments: exactly Threads * Increments, no lost updates.
        var shared = new CasCounter();
        var retriesPerThread = new long[Threads];
        var workers = new Thread[Threads];
        for (int t = 0; t < Threads; t++)
        {
            int threadIndex = t;
            workers[t] = new Thread(() =>
            {
                long local = 0;
                for (int i = 0; i < Increments; i++) local += shared.Increment();
                retriesPerThread[threadIndex] = local;
            });
            workers[t].Start();
        }
        foreach (var worker in workers)
        {
            worker.Join(20000);                    // timeout sanity: must terminate
            Check(!worker.IsAlive, "increment worker did not finish in time");
        }

        Check(shared.Get() == (long)Threads * Increments, "no increment lost");
        long totalRetries = 0;
        foreach (long retryCount in retriesPerThread)
        {
            Check(retryCount >= 0, "retry counters are non-negative");
            totalRetries += retryCount;
        }
        // Exact CAS accounting: every retry corresponds to exactly one failed CAS.
        Check(totalRetries == shared.CasFailures(), "retries equal recorded CAS failures");
        Check(shared.CasAttempts() == (long)Threads * Increments + shared.CasFailures(),
              "attempts == increments + failures");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
