// Task 351 - Implement thread pool basic with worker threads and task queue. (Tier: Intermediate, CTCI Ch 15)
using System;
using System.Collections.Generic;
using System.Threading;

public static class Solution
{
    private const int NumWorkers = 4;
    private const int NumTasks = 200;

    private static long TaskFunction(int taskId)
    {
        long seed = (taskId * 75L + 74) % 65537;
        return (long)taskId * taskId + seed;
    }

    /// <summary>FIFO work queue; idle workers block on Monitor.Wait instead of spinning.</summary>
    private class TaskQueue
    {
        private readonly Queue<int> items = new Queue<int>();
        private bool closed;

        public void Enqueue(int taskId)
        {
            lock (items)
            {
                items.Enqueue(taskId);
                Monitor.PulseAll(items);
            }
        }

        public bool TryDequeue(out int taskId)
        {
            lock (items)
            {
                while (items.Count == 0 && !closed) Monitor.Wait(items);
                if (items.Count == 0) { taskId = -1; return false; }   // drained and closed
                taskId = items.Dequeue();
                return true;
            }
        }

        public void Close()
        {
            lock (items)
            {
                closed = true;
                Monitor.PulseAll(items);
            }
        }

        public int Count { get { lock (items) { return items.Count; } } }
    }

    private class ThreadPool
    {
        private readonly TaskQueue queue = new TaskQueue();
        private readonly object accounting = new object();
        private readonly Dictionary<int, int> runCounts = new Dictionary<int, int>();
        private readonly Dictionary<int, long> results = new Dictionary<int, long>();
        private readonly Thread[] workers;

        public ThreadPool(int numWorkers, Func<int, long> taskFunction)
        {
            if (numWorkers <= 0) throw new ArgumentException("numWorkers must be positive");
            TaskFunction = taskFunction;
            workers = new Thread[numWorkers];
            for (int i = 0; i < numWorkers; i++) workers[i] = new Thread(WorkerLoop);
        }

        private Func<int, long> TaskFunction { get; }

        public void Start()
        {
            foreach (var worker in workers) worker.Start();
        }

        public void Submit(int taskId) { queue.Enqueue(taskId); }

        private void WorkerLoop()
        {
            while (queue.TryDequeue(out int taskId))
            {
                long value = TaskFunction(taskId);
                lock (accounting)
                {
                    runCounts[taskId] = runCounts.TryGetValue(taskId, out int count) ? count + 1 : 1;
                    results[taskId] = value;
                }
            }                                     // TryDequeue false when drained and closed
        }

        public bool ShutdownAndWait(int timeoutMillis)
        {
            queue.Close();
            foreach (var worker in workers) worker.Join(timeoutMillis);
            foreach (var worker in workers) if (worker.IsAlive) return false;
            return true;
        }

        public int RunCount(int taskId)
        {
            lock (accounting) { return runCounts.TryGetValue(taskId, out int count) ? count : 0; }
        }

        public long Result(int taskId)
        {
            lock (accounting) { return results[taskId]; }
        }

        public int ResultCount { get { lock (accounting) { return results.Count; } } }

        public int TotalRuns
        {
            get
            {
                lock (accounting)
                {
                    int total = 0;
                    foreach (var pair in runCounts) total += pair.Value;
                    return total;
                }
            }
        }

        public int Pending { get { return queue.Count; } }
    }

    public static void Main()
    {
        var pool = new ThreadPool(NumWorkers, TaskFunction);
        pool.Start();
        for (int taskId = 0; taskId < NumTasks; taskId++) pool.Submit(taskId);

        Check(pool.ShutdownAndWait(10000), "every worker stopped within the timeout");
        Check(pool.Pending == 0, "queue fully drained");
        Check(pool.ResultCount == NumTasks, "one result per task");
        Check(pool.TotalRuns == NumTasks, "total completions exact");
        for (int taskId = 0; taskId < NumTasks; taskId++)
        {
            Check(pool.RunCount(taskId) == 1, "task ran exactly once: " + taskId);
            Check(pool.Result(taskId) == TaskFunction(taskId), "task result: " + taskId);
        }

        var pool2 = new ThreadPool(2, TaskFunction);
        pool2.Start();
        pool2.Submit(5);
        pool2.Submit(5);
        pool2.Submit(7);
        Check(pool2.ShutdownAndWait(10000), "second pool stopped cleanly");
        Check(pool2.Pending == 0, "second queue drained");
        Check(pool2.RunCount(5) == 2, "duplicate tasks are both counted");
        Check(pool2.RunCount(7) == 1, "single submission counted once");
        Check(pool2.Result(5) == TaskFunction(5) && pool2.Result(7) == TaskFunction(7), "duplicate results");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
