// Task 358 - Dining philosophers solution using ordered locks. (Tier: Intermediate, CTCI Ch 15)
using System;
using System.Collections.Generic;
using System.Threading;

public static class Solution
{
    private const int Philosophers = 5;
    private const int Meals = 40;

    /// <summary>
    /// Dining philosophers solved by resource ordering: philosopher i uses forks i and
    /// (i+1)%5 but always locks the lower-numbered fork first (philosopher 4 therefore
    /// takes fork 0 before fork 4). Because every philosopher acquires its two forks in
    /// the same global order, no circular wait can form, and each fork monitor is held by
    /// at most one philosopher at any moment. Ownership accounting is recorded while the
    /// fork is still held, so the per-fork single-holder invariant can be checked race-free.
    /// </summary>
    private class ForkTable
    {
        private readonly object[] forks = new object[Philosophers];
        private readonly object accounting = new object();
        private readonly int[] owners = new int[Philosophers];      // -1 when free
        private readonly int[] holders = new int[Philosophers];
        private readonly int[] maxHolders = new int[Philosophers];
        private readonly int[] acquisitions = new int[Philosophers];
        private readonly int[] meals = new int[Philosophers];
        private int violations;

        public ForkTable()
        {
            for (int i = 0; i < Philosophers; i++)
            {
                forks[i] = new object();
                owners[i] = -1;
            }
        }

        /// <summary>Locks both forks in increasing index order and eats exactly one meal.</summary>
        public void Eat(int philosopher, int first, int second)
        {
            lock (forks[first])
            {
                RecordAcquire(philosopher, first);
                lock (forks[second])
                {
                    RecordAcquire(philosopher, second);
                    lock (accounting) { meals[philosopher]++; }
                    RecordRelease(philosopher, second);
                }
                RecordRelease(philosopher, first);
            }
        }

        private void RecordAcquire(int philosopher, int fork)
        {
            lock (accounting)
            {
                if (owners[fork] != -1) violations++;               // would mean the lock failed
                owners[fork] = philosopher;
                holders[fork]++;
                if (holders[fork] > maxHolders[fork]) maxHolders[fork] = holders[fork];
                acquisitions[fork]++;
            }
        }

        private void RecordRelease(int philosopher, int fork)
        {
            lock (accounting)
            {
                if (owners[fork] != philosopher) violations++;
                owners[fork] = -1;
                holders[fork]--;
            }
        }

        public int Violations { get { lock (accounting) { return violations; } } }
        public int[] Meals() { return meals; }
        public int[] Owners() { return owners; }
        public int[] Holders() { return holders; }
        public int[] MaxHolders() { return maxHolders; }
        public int[] Acquisitions() { return acquisitions; }
    }

    /// <summary>Documented lock ordering: the lower-numbered of the two forks always comes first.</summary>
    private static int[] ForkOrder(int philosopher)
    {
        int left = philosopher;
        int right = (philosopher + 1) % Philosophers;
        return new[] { Math.Min(left, right), Math.Max(left, right) };
    }

    public static void Main()
    {
        // Hand-checked ordering table: only philosopher 4 wraps around, taking fork 0 first.
        int[][] expectedOrder = { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 0, 4 } };
        for (int i = 0; i < Philosophers; i++)
        {
            Check(ListsEqual(new List<int>(ForkOrder(i)), new List<int>(expectedOrder[i])),
                  "documented increasing fork order " + i);
        }

        var table = new ForkTable();
        var diners = new Thread[Philosophers];
        for (int i = 0; i < Philosophers; i++)
        {
            int philosopher = i;
            diners[i] = new Thread(() =>
            {
                int[] order = ForkOrder(philosopher);
                for (int meal = 0; meal < Meals; meal++) table.Eat(philosopher, order[0], order[1]);
            });
            diners[i].Start();
        }
        foreach (var diner in diners)
        {
            diner.Join(10000);                    // timeout sanity: must terminate
            Check(!diner.IsAlive, "philosopher did not finish in time (deadlock?)");
        }

        int totalMeals = 0;
        foreach (int count in table.Meals())
        {
            Check(count == Meals, "every philosopher ate Meals times");
            totalMeals += count;
        }
        Check(totalMeals == Philosophers * Meals, "total meals == 5 * Meals");
        Check(table.Violations == 0, "per-fork single-holder invariant never violated");
        foreach (int owner in table.Owners()) Check(owner == -1, "every fork released at the end");
        foreach (int count in table.Holders()) Check(count == 0, "no residual holders");
        foreach (int count in table.MaxHolders()) Check(count == 1, "one holder at a time per fork");
        foreach (int count in table.Acquisitions())
        {
            Check(count == 2 * Meals, "each fork acquired by exactly its two neighbours");
        }

        Console.WriteLine("PASS");
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
