// Task 510 - Implement task DAG executor with parallel workers (respect dependencies). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
//
// A non-preemptive list-scheduling executor for a DAG of unit-time tasks. Two independent
// deterministic engines implement the same policy - "whenever a worker is free, take the ready task
// with the smallest (priority, id) key" - so their schedules can be compared:
//   * EventExecutor jumps from completion to completion (clock = min finish of the running set,
//     then all tasks finishing at that instant release their successors);
//   * TickExecutor walks the integer clock one tick at a time, assigning at the start of a tick and
//     retiring tasks whose remaining time reaches zero.
// The workload carries real payloads: value(task) = id + sum(value(dependencies)), computed after
// the dependencies complete, and cross-checked against a recursive brute-force evaluation, so the
// executor is exercised as a computation engine and not just as a timeline.
// Documented checks on a hand-simulated four-task example (W = 2): task 1 (priority 0) and task 0
// (priority 1) start together at tick 0, the freed worker picks task 3 at tick 2, and task 2 waits
// until tick 3, giving makespan 5; with W = 1 the makespan is the full 8 ticks of work, and with
// W = 4 it collapses to the critical path 5. On a 20-task seeded DAG (W = 2, 3, 4) the self-test
// asserts: every task starts at or after the finish of all dependencies, no more than W tasks
// overlap, the two engines produce identical start/finish vectors, the makespan respects the
// Graham bound makespan <= 2 * max(critical path, ceil(total work / W)), the makespan is at least
// both lower bounds, worker utilisation stays within (0, 1], payloads match the brute-force
// evaluation, tasks starting in the same batch are ordered by (priority, id), and the whole
// simulation is reproducible. Finally a real multi-threaded executor (lock + Monitor pulse, no
// sleeps) runs the same DAG on four threads and is asserted only on stable facts: every task runs
// exactly once, dependencies are never claimed early, in-flight tasks never exceed the number of
// threads, and the payloads are identical to the deterministic run.
using System;
using System.Collections.Generic;
using System.Threading;

internal class Seeded
{
    private ulong state;

    internal Seeded(ulong seed) => state = seed;

    internal int Next(int bound)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong h = state;
        h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9UL;
        h = (h ^ (h >> 27)) * 0x94D049BB133111EBUL;
        h ^= h >> 31;
        return (int)(h % (ulong)bound);
    }
}

internal class Task
{
    internal long Id;
    internal long Duration;
    internal long Priority;
    internal long[] Dependencies = Array.Empty<long>();
}

internal class Workload
{
    internal long Count;
    internal long[] Duration = Array.Empty<long>();
    internal long[] Priority = Array.Empty<long>();
    internal List<int>[] Dependencies = Array.Empty<List<int>>();
    internal List<int>[] Dependents = Array.Empty<List<int>>();

    internal static Workload FromTasks(List<Task> tasks)
    {
        int n = tasks.Count;
        var workload = new Workload
        {
            Count = n,
            Duration = new long[n],
            Priority = new long[n],
            Dependencies = new List<int>[n],
            Dependents = new List<int>[n],
        };
        for (int i = 0; i < n; i++)
        {
            workload.Dependencies[i] = new List<int>();
            workload.Dependents[i] = new List<int>();
        }
        var index = new Dictionary<long, int>();
        for (int i = 0; i < n; i++) index[tasks[i].Id] = i;
        for (int i = 0; i < n; i++)
        {
            workload.Duration[i] = tasks[i].Duration;
            workload.Priority[i] = tasks[i].Priority;
            foreach (long dependency in tasks[i].Dependencies)
            {
                if (!index.TryGetValue(dependency, out int parent)) continue;
                if (workload.Dependencies[i].Contains(parent)) continue;
                workload.Dependencies[i].Add(parent);
                workload.Dependents[parent].Add(i);
            }
        }
        return workload;
    }

    internal long TotalWork()
    {
        long total = 0;
        foreach (long duration in Duration) total += duration;
        return total;
    }

    /** Longest weighted path, by DP over a Kahn order. */
    internal long CriticalPath()
    {
        var indegree = new int[Count];
        for (int i = 0; i < Count; i++) indegree[i] = Dependencies[i].Count;
        var queue = new Queue<int>();
        for (int i = 0; i < Count; i++)
        {
            if (indegree[i] == 0) queue.Enqueue(i);
        }
        var earliest = new long[Count];
        var tail = new long[Count];
        var order = new List<int>();
        while (queue.Count > 0)
        {
            int node = queue.Dequeue();
            order.Add(node);
            foreach (int next in Dependents[node])
            {
                indegree[next] -= 1;
                if (indegree[next] == 0) queue.Enqueue(next);
            }
        }
        for (int i = order.Count - 1; i >= 0; i--)
        {
            int node = order[i];
            long best = 0;
            foreach (int next in Dependents[node]) best = Math.Max(best, tail[next]);
            tail[node] = Duration[node] + best;
        }
        long critical = 0;
        for (int i = 0; i < Count; i++)
        {
            if (Dependencies[i].Count == 0) critical = Math.Max(critical, tail[i]);
        }
        return critical;
    }

    /** Recursive brute-force payload evaluation (independent of the executors). */
    internal long BruteForceValue(int task, Dictionary<int, long> memo)
    {
        if (memo.TryGetValue(task, out long cached)) return cached;
        long value = task;
        foreach (int parent in Dependencies[task]) value += BruteForceValue(parent, memo);
        memo[task] = value;
        return value;
    }

    internal long[] BruteForceValues()
    {
        var memo = new Dictionary<int, long>();
        var values = new long[Count];
        for (int i = 0; i < Count; i++) values[i] = BruteForceValue(i, memo);
        return values;
    }

    internal int PickReady(bool[] done, bool[] started)
    {
        int best = -1;
        for (int i = 0; i < Count; i++)
        {
            if (done[i] || started[i]) continue;
            bool ready = true;
            foreach (int parent in Dependencies[i])
            {
                if (!done[parent])
                {
                    ready = false;
                    break;
                }
            }
            if (!ready) continue;
            if (best < 0 || Priority[i] < Priority[best]
                    || (Priority[i] == Priority[best] && i < best)) best = i;
        }
        return best;
    }
}

internal class Schedule
{
    internal long[] Start = Array.Empty<long>();
    internal long[] Finish = Array.Empty<long>();
    internal long[] Values = Array.Empty<long>();
    internal long Makespan;

    internal int MaxConcurrent()
    {
        var events = new List<(long Tick, int Delta)>();
        for (int i = 0; i < Start.Length; i++)
        {
            events.Add((Start[i], 1));
            events.Add((Finish[i], -1));
        }
        events.Sort((a, b) => a.Tick != b.Tick ? a.Tick.CompareTo(b.Tick) : a.Delta.CompareTo(b.Delta));
        int current = 0;
        int worst = 0;
        foreach ((long _, int delta) in events)
        {
            current += delta;
            worst = Math.Max(worst, current);
        }
        return worst;
    }
}

internal static class EventExecutor
{
    internal static Schedule Run(Workload workload, int workers)
    {
        int n = (int)workload.Count;
        var schedule = new Schedule
        {
            Start = new long[n],
            Finish = new long[n],
            Values = new long[n],
        };
        var done = new bool[n];
        var started = new bool[n];
        var running = new List<(long Finish, int Task)>();
        int free = workers;
        long clock = 0;
        int remaining = n;
        while (remaining > 0)
        {
            while (free > 0)
            {
                int task = workload.PickReady(done, started);
                if (task < 0) break;
                started[task] = true;
                schedule.Start[task] = clock;
                schedule.Finish[task] = clock + workload.Duration[task];
                running.Add((schedule.Finish[task], task));
                free -= 1;
            }
            if (running.Count == 0) break;
            long next = long.MaxValue;
            foreach ((long finish, int _) in running) next = Math.Min(next, finish);
            clock = next;
            var completed = new List<int>();
            for (int i = running.Count - 1; i >= 0; i--)
            {
                if (running[i].Finish == clock)
                {
                    completed.Add(running[i].Task);
                    running.RemoveAt(i);
                }
            }
            foreach (int task in completed)
            {
                long value = task;
                foreach (int parent in workload.Dependencies[task]) value += schedule.Values[parent];
                schedule.Values[task] = value;
                done[task] = true;
                free += 1;
                remaining -= 1;
            }
            schedule.Makespan = Math.Max(schedule.Makespan, clock);
        }
        return schedule;
    }
}

internal static class TickExecutor
{
    internal static Schedule Run(Workload workload, int workers)
    {
        int n = (int)workload.Count;
        var schedule = new Schedule
        {
            Start = new long[n],
            Finish = new long[n],
            Values = new long[n],
        };
        var done = new bool[n];
        var started = new bool[n];
        var remainingTime = new long[n];
        var running = new List<int>();
        int free = workers;
        int remaining = n;
        long clock = 0;
        while (remaining > 0)
        {
            while (free > 0)
            {
                int task = workload.PickReady(done, started);
                if (task < 0) break;
                started[task] = true;
                schedule.Start[task] = clock;
                remainingTime[task] = workload.Duration[task];
                running.Add(task);
                free -= 1;
            }
            if (running.Count == 0) break;
            for (int i = running.Count - 1; i >= 0; i--)
            {
                int task = running[i];
                remainingTime[task] -= 1;
                if (remainingTime[task] != 0) continue;
                schedule.Finish[task] = clock + 1;
                long value = task;
                foreach (int parent in workload.Dependencies[task]) value += schedule.Values[parent];
                schedule.Values[task] = value;
                done[task] = true;
                free += 1;
                remaining -= 1;
                running.RemoveAt(i);
            }
            clock += 1;
            schedule.Makespan = Math.Max(schedule.Makespan, clock);
        }
        return schedule;
    }
}

/** Real threads: workers claim ready tasks under a lock and wait on a monitor when none is ready. */
internal class ThreadedExecutor
{
    private readonly object gate = new object();
    private readonly Workload workload;
    private readonly bool[] done;
    private readonly bool[] started;
    private readonly long[] values;
    private int tasksLeft;
    private int inFlight;
    private int maxInFlight;
    private int executedCount;
    private bool dependencyViolation;
    private bool joinedSuccessfully = true;

    internal ThreadedExecutor(Workload workload)
    {
        this.workload = workload;
        int n = (int)workload.Count;
        done = new bool[n];
        started = new bool[n];
        values = new long[n];
        tasksLeft = n;
    }

    internal int MaxInFlight => maxInFlight;

    internal int ExecutedCount => executedCount;

    internal bool DependencyViolation => dependencyViolation;

    internal bool Joined => joinedSuccessfully;

    internal long[] Values => values;

    internal void Run(int workers)
    {
        var threads = new Thread[workers];
        for (int i = 0; i < workers; i++) threads[i] = new Thread(Worker);
        foreach (Thread thread in threads) thread.Start();
        foreach (Thread thread in threads)
        {
            if (!thread.Join(10000)) joinedSuccessfully = false;
        }
    }

    private void Worker()
    {
        while (true)
        {
            int task = -1;
            long value = 0;
            lock (gate)
            {
                while (true)
                {
                    if (tasksLeft == 0) return;
                    task = Pick();
                    if (task >= 0) break;
                    Monitor.Wait(gate, 1000);
                    if (tasksLeft == 0) return;
                }
                started[task] = true;
                inFlight += 1;
                maxInFlight = Math.Max(maxInFlight, inFlight);
                executedCount += 1;
                value = task;
                foreach (int parent in workload.Dependencies[task])
                {
                    if (!done[parent]) dependencyViolation = true;
                    value += values[parent];
                }
            }
            lock (gate)
            {
                values[task] = value;
                done[task] = true;
                inFlight -= 1;
                tasksLeft -= 1;
                Monitor.PulseAll(gate);
            }
        }
    }

    private int Pick()
    {
        return workload.PickReady(done, started);
    }
}

public static class Solution
{
    public static void Main()
    {
        // --- hand-simulated four-task example -----------------------------------------------------------
        var crafted = Workload.FromTasks(new List<Task>
        {
            new Task { Id = 0, Duration = 3, Priority = 1, Dependencies = Array.Empty<long>() },
            new Task { Id = 1, Duration = 2, Priority = 0, Dependencies = Array.Empty<long>() },
            new Task { Id = 2, Duration = 2, Priority = 0, Dependencies = new[] { 0L } },
            new Task { Id = 3, Duration = 1, Priority = 2, Dependencies = Array.Empty<long>() },
        });
        Schedule craftedTwo = EventExecutor.Run(crafted, 2);
        Check(craftedTwo.Makespan == 5, "W = 2 makespan is 5 (got " + craftedTwo.Makespan + ")");
        Check(craftedTwo.Start[0] == 0 && craftedTwo.Start[1] == 0 && craftedTwo.Start[3] == 2
                && craftedTwo.Start[2] == 3, "the hand-simulated start times");
        Check(craftedTwo.Finish[1] == 2 && craftedTwo.Finish[0] == 3 && craftedTwo.Finish[3] == 3
                && craftedTwo.Finish[2] == 5, "the hand-simulated finish times");
        Check(craftedTwo.MaxConcurrent() == 2, "exactly two tasks overlap with W = 2");
        Schedule craftedOne = EventExecutor.Run(crafted, 1);
        Check(craftedOne.Makespan == 8, "W = 1 serialises everything onto the total work");
        Check(craftedOne.Makespan == crafted.TotalWork(), "W = 1 makespan == total work");
        Schedule craftedFour = EventExecutor.Run(crafted, 4);
        Check(craftedFour.Makespan == 5, "W = 4 collapses to the critical path");
        Check(craftedFour.Makespan == crafted.CriticalPath(), "W = 4 makespan == critical path");
        Check(craftedTwo.Values[2] == 2 + 0 && craftedFour.Values[3] == 3, "payloads flow along the edges");

        // --- the seeded 20-task workload -----------------------------------------------------------------
        var rng = new Seeded(510510UL);
        var tasks = new List<Task>();
        for (int i = 0; i < 20; i++)
        {
            var deps = new List<long>();
            if (i > 0)
            {
                int candidates = rng.Next(3);
                for (int c = 0; c < candidates; c++)
                {
                    int parent = rng.Next(i);
                    if (!deps.Contains(parent)) deps.Add(parent);
                }
            }
            tasks.Add(new Task
            {
                Id = i,
                Duration = rng.Next(4) + 1,
                Priority = rng.Next(3),
                Dependencies = deps.ToArray(),
            });
        }
        var workload = Workload.FromTasks(tasks);
        Check(workload.Count == 20, "the seeded workload has 20 tasks");
        long[] expectedValues = workload.BruteForceValues();
        long totalWork = workload.TotalWork();
        long criticalPath = workload.CriticalPath();
        Check(criticalPath >= 1 && criticalPath <= totalWork, "the critical path is a valid lower bound");

        foreach (int workers in new[] { 2, 3, 4 })
        {
            Schedule schedule = EventExecutor.Run(workload, workers);
            Schedule ticks = TickExecutor.Run(workload, workers);
            Check(schedule.Makespan >= criticalPath, "makespan >= critical path at W = " + workers);
            long balancedLowerBound = (totalWork + workers - 1) / workers;
            Check(schedule.Makespan >= balancedLowerBound, "makespan >= ceil(work / W) at W = " + workers);
            Check(schedule.Makespan <= criticalPath + balancedLowerBound,
                    "Graham bound makespan <= CP + ceil(work / W) at W = " + workers);
            Check(schedule.Makespan <= 2 * Math.Max(criticalPath, balancedLowerBound),
                    "Graham corollary at W = " + workers + " (makespan " + schedule.Makespan + ")");
            Check(schedule.MaxConcurrent() <= workers, "no more than W tasks overlap at W = " + workers);
            Check(schedule.Makespan == ticks.Makespan, "engines agree on the makespan at W = " + workers);
            for (int i = 0; i < workload.Count; i++)
            {
                Check(schedule.Start[i] == ticks.Start[i] && schedule.Finish[i] == ticks.Finish[i],
                        "engines agree on task " + i + " at W = " + workers);
                Check(schedule.Values[i] == expectedValues[i], "payload of task " + i + " at W = " + workers);
                Check(schedule.Finish[i] == schedule.Start[i] + workload.Duration[i],
                        "duration honoured for task " + i);
                foreach (int parent in workload.Dependencies[i])
                {
                    Check(schedule.Start[i] >= schedule.Finish[parent],
                            "task " + i + " waits for its dependency " + parent);
                }
            }
            long idle = (long)workers * schedule.Makespan - totalWork;
            Check(idle >= 0, "the workers cannot be over-committed");
            double utilisation = totalWork / (double)((long)workers * schedule.Makespan);
            Check(utilisation > 0.25 && utilisation <= 1.0, "utilisation at W = " + workers + " is " + utilisation);
            CheckListScheduling(workload, schedule, workers);
            Schedule repeat = EventExecutor.Run(workload, workers);
            for (int i = 0; i < workload.Count; i++)
            {
                Check(repeat.Start[i] == schedule.Start[i] && repeat.Values[i] == schedule.Values[i],
                        "deterministic schedule at W = " + workers + " for task " + i);
            }
        }

        // --- degenerate workloads ------------------------------------------------------------------------------
        Schedule emptyEvent = EventExecutor.Run(Workload.FromTasks(new List<Task>()), 3);
        Schedule emptyTicks = TickExecutor.Run(Workload.FromTasks(new List<Task>()), 3);
        Check(emptyEvent.Makespan == 0 && emptyTicks.Makespan == 0, "an empty DAG takes no time");
        var single = Workload.FromTasks(new List<Task>
        {
            new Task { Id = 7, Duration = 4, Priority = 0, Dependencies = Array.Empty<long>() },
        });
        Check(EventExecutor.Run(single, 2).Makespan == 4, "a single task takes its duration");
        Check(EventExecutor.Run(single, 2).Values[0] == 0, "a single task's payload is its index");
        var chain = new List<Task>();
        for (int i = 0; i < 5; i++)
        {
            chain.Add(new Task
            {
                Id = i,
                Duration = i + 1,
                Priority = 0,
                Dependencies = i == 0 ? Array.Empty<long>() : new[] { (long)(i - 1) },
            });
        }
        var chainWorkload = Workload.FromTasks(chain);
        Schedule chainSerial = EventExecutor.Run(chainWorkload, 1);
        Schedule chainParallel = TickExecutor.Run(chainWorkload, 3);
        Check(chainSerial.Makespan == 15, "a chain with W = 1 takes the sum of durations");
        Check(chainParallel.Makespan == 15, "a chain cannot be parallelised");
        Check(chainParallel.MaxConcurrent() == 1, "a chain never runs two tasks at once");
        var wide = new List<Task>();
        for (int i = 0; i < 9; i++)
        {
            wide.Add(new Task { Id = i, Duration = 2, Priority = i % 3, Dependencies = Array.Empty<long>() });
        }
        var wideWorkload = Workload.FromTasks(wide);
        Schedule wideThree = EventExecutor.Run(wideWorkload, 3);
        Check(wideThree.Makespan == 6, "nine identical tasks on three workers take 3 * 2 ticks");
        Check(wideThree.MaxConcurrent() == 3, "all three workers are busy");
        var duplicateDeps = new List<Task>
        {
            new Task { Id = 0, Duration = 1, Priority = 0, Dependencies = Array.Empty<long>() },
            new Task { Id = 1, Duration = 1, Priority = 0, Dependencies = new[] { 0L, 0L, 0L } },
        };
        var duplicateWorkload = Workload.FromTasks(duplicateDeps);
        Check(duplicateWorkload.Dependencies[1].Count == 1, "duplicate dependencies are collapsed");
        Check(EventExecutor.Run(duplicateWorkload, 2).Makespan == 2, "the deduplicated chain takes two ticks");

        // --- real threads ----------------------------------------------------------------------------------------
        var threaded = new ThreadedExecutor(workload);
        threaded.Run(4);
        Check(threaded.Joined, "all worker threads joined within ten seconds");
        Check(threaded.ExecutedCount == workload.Count, "every task ran exactly once on real threads");
        Check(!threaded.DependencyViolation, "no task was claimed before its dependencies finished");
        Check(threaded.MaxInFlight <= 4 && threaded.MaxInFlight >= 1,
                "in-flight tasks stayed within the thread count (saw " + threaded.MaxInFlight + ")");
        for (int i = 0; i < workload.Count; i++)
        {
            Check(threaded.Values[i] == expectedValues[i], "threaded payload of task " + i);
        }
        var threadedAgain = new ThreadedExecutor(workload);
        threadedAgain.Run(3);
        Check(threadedAgain.Joined && threadedAgain.ExecutedCount == workload.Count,
                "a three-thread run completes as well");
        Check(!threadedAgain.DependencyViolation, "the three-thread run kept the dependencies");
        for (int i = 0; i < workload.Count; i++)
        {
            Check(threadedAgain.Values[i] == expectedValues[i], "three-thread payload of task " + i);
        }

        Console.WriteLine("PASS");
    }

    /** Replays the schedule to verify the greedy list-scheduling policy directly. */
    private static void CheckListScheduling(Workload workload, Schedule schedule, int workers)
    {
        for (long t = 0; t <= schedule.Makespan; t++)
        {
            var batch = new List<int>();
            int inFlight = 0;
            for (int i = 0; i < workload.Count; i++)
            {
                if (schedule.Start[i] == t) batch.Add(i);
                if (schedule.Start[i] <= t && t < schedule.Finish[i]) inFlight += 1;
            }
            bool anyEligibleUnstarted = false;
            for (int z = 0; z < workload.Count; z++)
            {
                if (schedule.Start[z] <= t) continue;
                bool eligible = true;
                foreach (int parent in workload.Dependencies[z])
                {
                    if (schedule.Finish[parent] > t)
                    {
                        eligible = false;
                        break;
                    }
                }
                if (!eligible) continue;
                anyEligibleUnstarted = true;
                foreach (int x in batch)
                {
                    Check(workload.Priority[x] < workload.Priority[z]
                            || (workload.Priority[x] == workload.Priority[z] && x < z),
                            "greedy (priority, id) order at t = " + t + ": " + x + " before eligible " + z);
                }
            }
            if (anyEligibleUnstarted)
            {
                Check(inFlight == workers, "every worker is busy while work is eligible at t = " + t);
            }
        }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
