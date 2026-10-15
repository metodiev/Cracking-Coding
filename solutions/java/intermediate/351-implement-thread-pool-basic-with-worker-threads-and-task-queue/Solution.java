/** Task 351 - Implement thread pool basic with worker threads and task queue. (Tier: Intermediate, CTCI Ch 15) */
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Function;

public class Solution {
    static final int NUM_WORKERS = 4;
    static final int NUM_TASKS = 200;
    static final int SHUTDOWN = -1;                   // graceful shutdown sentinel

    static long taskFunction(int taskId) {
        long seed = (taskId * 75L + 74) % 65537;
        return (long) taskId * taskId + seed;
    }

    /** FIFO work queue (LinkedBlockingQueue): idle workers block, never spin. */
    static class ThreadPool {
        private final BlockingQueue<Integer> queue = new LinkedBlockingQueue<>();
        private final Function<Integer, Long> taskFunction;
        private final Object accounting = new Object();
        private final Map<Integer, Integer> runCounts = new HashMap<>();
        private final Map<Integer, Long> results = new HashMap<>();
        private final Thread[] workers;

        ThreadPool(int numWorkers, Function<Integer, Long> taskFunction) {
            if (numWorkers <= 0) throw new IllegalArgumentException("numWorkers must be positive");
            this.taskFunction = taskFunction;
            workers = new Thread[numWorkers];
            for (int i = 0; i < numWorkers; i++) workers[i] = new Thread(this::workerLoop);
        }

        void start() {
            for (Thread worker : workers) worker.start();
        }

        void submit(int taskId) {
            try {
                queue.put(taskId);
            } catch (InterruptedException exc) {
                Thread.currentThread().interrupt();
            }
        }

        private void workerLoop() {
            try {
                while (true) {
                    int taskId = queue.take();
                    if (taskId == SHUTDOWN) return;             // sentinel: stop this worker
                    long value = taskFunction.apply(taskId);
                    synchronized (accounting) {
                        runCounts.merge(taskId, 1, Integer::sum);
                        results.put(taskId, value);
                    }
                }
            } catch (InterruptedException exc) {
                Thread.currentThread().interrupt();
            }
        }

        boolean shutdownAndWait(long timeoutMillis) {
            try {
                for (int i = 0; i < workers.length; i++) queue.put(SHUTDOWN);
                for (Thread worker : workers) worker.join(timeoutMillis);
            } catch (InterruptedException exc) {
                Thread.currentThread().interrupt();
            }
            for (Thread worker : workers) if (worker.isAlive()) return false;
            return true;
        }

        int runCount(int taskId) {
            synchronized (accounting) { return runCounts.getOrDefault(taskId, 0); }
        }

        int resultCount() {
            synchronized (accounting) { return results.size(); }
        }

        int totalRuns() {
            synchronized (accounting) {
                int total = 0;
                for (int count : runCounts.values()) total += count;
                return total;
            }
        }

        long result(int taskId) {
            synchronized (accounting) { return results.get(taskId); }
        }

        boolean isEmpty() { return queue.isEmpty(); }
    }

    public static void main(String[] args) {
        ThreadPool pool = new ThreadPool(NUM_WORKERS, Solution::taskFunction);
        pool.start();
        for (int taskId = 0; taskId < NUM_TASKS; taskId++) pool.submit(taskId);

        check(pool.shutdownAndWait(10000), "every worker stopped within the timeout");
        check(pool.isEmpty(), "queue fully drained");
        check(pool.resultCount() == NUM_TASKS, "one result per task");
        check(pool.totalRuns() == NUM_TASKS, "total completions exact");
        for (int taskId = 0; taskId < NUM_TASKS; taskId++) {
            check(pool.runCount(taskId) == 1, "task ran exactly once: " + taskId);
            check(pool.result(taskId) == taskFunction(taskId), "task result: " + taskId);
        }

        ThreadPool pool2 = new ThreadPool(2, Solution::taskFunction);
        pool2.start();
        pool2.submit(5);
        pool2.submit(5);
        pool2.submit(7);
        check(pool2.shutdownAndWait(10000), "second pool stopped cleanly");
        check(pool2.isEmpty(), "second queue drained");
        check(pool2.runCount(5) == 2, "duplicate tasks are both counted");
        check(pool2.runCount(7) == 1, "single submission counted once");
        check(pool2.result(5) == taskFunction(5) && pool2.result(7) == taskFunction(7), "duplicate results");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
