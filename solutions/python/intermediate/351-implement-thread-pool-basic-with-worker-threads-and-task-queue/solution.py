"""Task 351 - Implement thread pool basic with worker threads and task queue. (Tier: Intermediate, CTCI Ch 15)

Classic thread pool: NUM_WORKERS persistent worker threads consume task ids from a
shared FIFO queue (queue.Queue - a condition variable underneath, so idle workers
block instead of spinning). Each worker runs task_function(id) and records its
outcome in accounting slots, so the self-test proves every submitted task ran
exactly once. Graceful shutdown pushes one sentinel (None) per worker; workers
drain the queue and exit, and every join carries a timeout so a broken shutdown
fails an assertion instead of hanging the process.
"""
import queue
import threading

NUM_WORKERS = 4
NUM_TASKS = 200


def task_function(task_id):
    """Deterministic workload: square plus an LCG-derived offset."""
    seed = (task_id * 75 + 74) % 65537
    return task_id * task_id + seed


class ThreadPool:
    def __init__(self, num_workers, task_function):
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        self.task_function = task_function
        self._queue = queue.Queue()
        self._accounting = threading.Lock()
        self.run_counts = {}
        self.results = {}
        self.workers = [threading.Thread(target=self._worker_loop, daemon=True)
                        for _ in range(num_workers)]

    def start(self):
        for worker in self.workers:
            worker.start()

    def submit(self, task_id):
        self._queue.put(task_id)

    def _worker_loop(self):
        while True:
            task_id = self._queue.get()
            if task_id is None:                    # sentinel: graceful shutdown
                return
            value = self.task_function(task_id)
            with self._accounting:
                self.run_counts[task_id] = self.run_counts.get(task_id, 0) + 1
                self.results[task_id] = value

    def shutdown_and_wait(self, timeout=10.0):
        """Sentinel per worker; True when every worker exited within the timeout."""
        for _ in self.workers:
            self._queue.put(None)
        for worker in self.workers:
            worker.join(timeout)
        return all(not worker.is_alive() for worker in self.workers)

    def pending(self):
        return self._queue.qsize()


def main():
    pool = ThreadPool(NUM_WORKERS, task_function)
    pool.start()
    for task_id in range(NUM_TASKS):
        pool.submit(task_id)

    assert pool.shutdown_and_wait(), "every worker stopped within the timeout"
    assert pool.pending() == 0, "queue fully drained"

    expected = [task_function(task_id) for task_id in range(NUM_TASKS)]
    assert len(pool.results) == NUM_TASKS, "one result per task"
    assert sum(pool.run_counts.values()) == NUM_TASKS, "total completions exact"
    for task_id in range(NUM_TASKS):
        assert pool.run_counts[task_id] == 1, "task {} ran exactly once".format(task_id)
        assert pool.results[task_id] == expected[task_id], "task {} result".format(task_id)

    # Duplicate submission is not lost either: 5 runs twice, 7 once.
    pool2 = ThreadPool(2, task_function)
    pool2.start()
    pool2.submit(5)
    pool2.submit(5)
    pool2.submit(7)
    assert pool2.shutdown_and_wait(), "second pool stopped cleanly"
    assert pool2.pending() == 0, "second queue drained"
    assert pool2.run_counts[5] == 2, "duplicate tasks are both counted"
    assert pool2.run_counts[7] == 1, "single submission counted once"
    assert pool2.results[5] == expected[5] and pool2.results[7] == expected[7]


if __name__ == "__main__":
    main()
    print("PASS")
