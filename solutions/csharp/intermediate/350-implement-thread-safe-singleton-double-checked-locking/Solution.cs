// Task 350 - Thread-safe singleton via double-checked locking. (Tier: Intermediate, CTCI Ch 15)
using System;
using System.Collections.Generic;
using System.Threading;

public class Singleton
{
    private static volatile Singleton? instance;
    private static readonly object gate = new object();
    private static int instantiationCount;

    private Singleton()
    {
        Interlocked.Increment(ref instantiationCount);   // only reachable while holding the lock
    }

    public static Singleton GetInstance()
    {
        Singleton? local = instance;          // fast path: one volatile read, no lock
        if (local != null) return local;
        lock (gate)
        {
            if (instance == null)             // re-check under the lock
            {
                instance = new Singleton();   // volatile write publishes the constructed object
            }
            return instance;
        }
    }

    public static int InstantiationCount() => Volatile.Read(ref instantiationCount);
}

public static class Solution
{
    public static void Main()
    {
        Check(Singleton.InstantiationCount() == 0, "lazy: nothing constructed yet");

        const int callers = 32;
        var gate = new ManualResetEventSlim(false);
        var results = new Singleton?[callers];
        var threads = new List<Thread>();
        for (int index = 0; index < callers; index++)
        {
            int i = index;
            threads.Add(new Thread(() =>
            {
                gate.Wait();
                results[i] = Singleton.GetInstance();
            })
            { IsBackground = true });
        }
        foreach (Thread thread in threads) thread.Start();
        gate.Set();                           // release all callers at once
        foreach (Thread thread in threads) Check(thread.Join(TimeSpan.FromSeconds(15)), "caller terminated");

        Check(results[0] != null, "first caller received an instance");
        for (int i = 1; i < callers; i++)
        {
            Check(ReferenceEquals(results[i], results[0]), "caller " + i + " received the same instance");
        }
        Check(Singleton.InstantiationCount() == 1, "the constructor ran exactly once");
        Check(ReferenceEquals(Singleton.GetInstance(), results[0]), "later callers reuse the instance");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
