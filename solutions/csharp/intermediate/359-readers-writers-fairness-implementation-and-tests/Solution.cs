// Task 359 - Readers-writers fairness implementation and tests. (Tier: Intermediate, CTCI Ch 15)
using System;
using System.Collections.Generic;
using System.Threading;

public static class Solution
{
    private const int Readers = 4;
    private const int Writers = 3;
    private const int Iterations = 50;

    /// <summary>
    /// Readers-writers lock with the documented WRITER-PREFERENCE policy: a reader may
    /// enter only when no writer is active and no writer is waiting (waitingWriters > 0),
    /// so a stream of arriving readers can never starve a writer; a writer enters when no
    /// writer is active and all active readers have left. Mutual exclusion is checked on
    /// every entry/exit, so a writer releasing while readers are active (or a reader next
    /// to a writer) would be counted as a violation.
    /// </summary>
    private class WriterPreferenceRW
    {
        private readonly object gate = new object();
        private int activeReaders;
        private bool activeWriter;
        private int waitingWriters;
        private int readerEntries;
        private int writerEntries;
        private int maxActiveReaders;
        private int violations;

        public void AcquireRead()
        {
            lock (gate)
            {
                while (activeWriter || waitingWriters > 0) Monitor.Wait(gate);
                if (activeWriter) violations++;           // unreachable: writer would overlap
                activeReaders++;
                if (activeReaders > maxActiveReaders) maxActiveReaders = activeReaders;
                readerEntries++;
            }
        }

        public void ReleaseRead()
        {
            lock (gate)
            {
                if (activeReaders <= 0 || activeWriter) violations++;
                activeReaders--;
                if (activeReaders == 0) Monitor.PulseAll(gate);   // last reader releases writers
            }
        }

        public void AcquireWrite(Action? beforeWait)
        {
            lock (gate)
            {
                waitingWriters++;
                beforeWait?.Invoke();                     // hook runs with the intent registered
                while (activeWriter || activeReaders > 0) Monitor.Wait(gate);
                if (activeWriter || activeReaders > 0) violations++;
                waitingWriters--;
                activeWriter = true;
                writerEntries++;
            }
        }

        public void AcquireWrite() { AcquireWrite(null); }

        public void ReleaseWrite()
        {
            lock (gate)
            {
                if (!activeWriter || activeReaders > 0) violations++;
                activeWriter = false;
                Monitor.PulseAll(gate);
            }
        }

        public int ReaderEntries { get { lock (gate) { return readerEntries; } } }
        public int WriterEntries { get { lock (gate) { return writerEntries; } } }
        public int MaxActiveReaders { get { lock (gate) { return maxActiveReaders; } } }
        public int Violations { get { lock (gate) { return violations; } } }
        public bool IsClean { get { lock (gate) { return activeReaders == 0 && !activeWriter; } } }
    }

    public static void Main()
    {
        // Part 1: scripted writer-preference ordering test.
        var rw = new WriterPreferenceRW();
        var log = new List<string>();
        var logGate = new object();
        var readerAAcquired = new ManualResetEventSlim();
        var releaseA = new ManualResetEventSlim();
        var writerRegistered = new ManualResetEventSlim();
        var writerEntered = new ManualResetEventSlim();
        var releaseW = new ManualResetEventSlim();
        var readerBEntered = new ManualResetEventSlim();

        var readerA = new Thread(() =>
        {
            rw.AcquireRead();
            lock (logGate) { log.Add("readA"); }
            readerAAcquired.Set();
            Check(releaseA.Wait(10000), "reader A released in time");
            rw.ReleaseRead();
        });
        var writerW = new Thread(() =>
        {
            rw.AcquireWrite(writerRegistered.Set);
            lock (logGate) { log.Add("writeW"); }
            writerEntered.Set();
            Check(releaseW.Wait(10000), "writer W released in time");
            rw.ReleaseWrite();
        });
        var readerB = new Thread(() =>
        {
            rw.AcquireRead();
            lock (logGate) { log.Add("readB"); }
            readerBEntered.Set();
            rw.ReleaseRead();
        });

        readerA.Start();
        Check(readerAAcquired.Wait(10000), "reader A got in first");
        writerW.Start();
        Check(writerRegistered.Wait(10000), "writer W registered its request");
        readerB.Start();                              // reader B arrives while W waits

        releaseA.Set();
        Check(writerEntered.Wait(10000), "writer W entered after reader A left");
        Check(!readerBEntered.IsSet, "new reader waits while a writer is pending");
        releaseW.Set();
        Check(readerBEntered.Wait(10000), "reader B entered once the writer was done");

        readerA.Join(10000);
        writerW.Join(10000);
        readerB.Join(10000);
        Check(!readerA.IsAlive && !writerW.IsAlive && !readerB.IsAlive, "scripted threads finished");
        lock (logGate)
        {
            Check(log.Count == 3 && log[0] == "readA" && log[1] == "writeW" && log[2] == "readB",
                  "writer-preference order");
        }
        Check(rw.WriterEntries == 1 && rw.ReaderEntries == 2, "entries counted");
        Check(rw.Violations == 0, "scripted scenario kept mutual exclusion");

        // Part 2 + 3: threaded workload with the fairness accounting.
        var workload = new WriterPreferenceRW();
        int[] state = { 0 };                          // guarded by the write lock
        var observations = new List<int>();
        var observationGate = new object();
        var writerCompletions = new int[Writers];
        var readerCompletions = new int[Readers];

        var workers = new Thread[Readers + Writers];
        for (int i = 0; i < Readers; i++)
        {
            int readerId = i;
            workers[i] = new Thread(() =>
            {
                for (int iteration = 0; iteration < Iterations; iteration++)
                {
                    workload.AcquireRead();
                    lock (observationGate) { observations.Add(state[0]); }
                    workload.ReleaseRead();
                    readerCompletions[readerId]++;
                }
            });
        }
        for (int i = 0; i < Writers; i++)
        {
            int writerId = i;
            workers[Readers + i] = new Thread(() =>
            {
                for (int iteration = 0; iteration < Iterations; iteration++)
                {
                    workload.AcquireWrite();
                    state[0]++;
                    workload.ReleaseWrite();
                    writerCompletions[writerId]++;
                }
            });
        }
        foreach (var worker in workers) worker.Start();
        foreach (var worker in workers)
        {
            worker.Join(10000);                       // timeout sanity: must terminate
            Check(!worker.IsAlive, "workload thread did not finish in time");
        }

        int totalWriterCompletions = 0;
        foreach (int count in writerCompletions) totalWriterCompletions += count;
        Check(totalWriterCompletions == Writers * Iterations, "every writer request was served");
        Check(workload.WriterEntries == Writers * Iterations && workload.WriterEntries == 150,
              "writer entries == writer requests (no starvation)");
        Check(workload.ReaderEntries == Readers * Iterations && workload.ReaderEntries == 200,
              "reader entries == reader requests");
        Check(state[0] == 150, "all writer increments applied exactly once");
        Check(workload.Violations == 0, "mutual exclusion held throughout the workload");
        Check(observations.Count == 200, "every read produced an observation");
        foreach (int value in observations) Check(value >= 0 && value <= 150, "readers only saw committed values");
        Check(workload.MaxActiveReaders >= 1, "readers actually ran");
        Check(workload.IsClean, "lock left clean");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
