/** Task 354 - Implement wait-free counter sketch (atomic operations). (Tier: Intermediate, CTCI Ch 15) */
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongConsumer;

public class Solution {
    static final int THREADS = 8;
    static final int INCREMENTS = 20000;

    /**
     * CAS counter sketch: increment() is a compare-and-swap retry loop over an AtomicLong
     * instead of +=. Strictly this is lock-free, not wait-free (a thread can be overtaken
     * indefinitely); the wait-free primitive behind "wait-free counters" is fetch-and-add,
     * which AtomicLong exposes as getAndAdd. The retry counters are asserted >= 0 and may
     * legitimately be 0 when the scheduler never interleaves a read and its CAS; the
     * afterRead hook forces one lost race so the retry path is exercised deterministically.
     */
    static class CasCounter {
        private final AtomicLong value = new AtomicLong();
        private final AtomicLong casAttempts = new AtomicLong();
        private final AtomicLong casFailures = new AtomicLong();

        long get() { return value.get(); }
        long casAttempts() { return casAttempts.get(); }
        long casFailures() { return casFailures.get(); }

        boolean compareAndSwap(long expected, long newValue) {
            casAttempts.incrementAndGet();
            if (value.compareAndSet(expected, newValue)) return true;
            casFailures.incrementAndGet();
            return false;
        }

        long increment() { return increment(null); }

        /** afterRead runs once between the read and the CAS of the first attempt. */
        long increment(LongConsumer afterRead) {
            long retries = 0;
            LongConsumer hook = afterRead;
            while (true) {
                long current = value.get();
                if (hook != null) {
                    hook.accept(current);
                    hook = null;
                }
                if (compareAndSwap(current, current + 1)) return retries;
                retries++;
            }
        }
    }

    public static void main(String[] args) {
        // Single-threaded baseline: no contention, no retries.
        CasCounter counter = new CasCounter();
        for (int i = 0; i < 100; i++) check(counter.increment() == 0, "uncontended CAS never retries");
        check(counter.get() == 100, "baseline total");
        check(counter.casAttempts() == 100 && counter.casFailures() == 0, "baseline CAS accounting");

        // Deterministic retry-path demonstration: a concurrent writer bumps the value after
        // our read, so the first CAS fails (retries == 1) and the loop rereads and succeeds.
        CasCounter forced = new CasCounter();
        long retries = forced.increment(current -> forced.compareAndSwap(current, current + 1));
        check(retries == 1, "stale read forces exactly one retry");
        check(forced.get() == 2, "both the stealing write and our retry are counted");

        // Concurrent increments: exactly THREADS * INCREMENTS, no lost updates.
        CasCounter shared = new CasCounter();
        long[] retriesPerThread = new long[THREADS];
        Thread[] workers = new Thread[THREADS];
        for (int t = 0; t < THREADS; t++) {
            final int threadIndex = t;
            workers[t] = new Thread(() -> {
                long local = 0;
                for (int i = 0; i < INCREMENTS; i++) local += shared.increment();
                retriesPerThread[threadIndex] = local;
            });
            workers[t].start();
        }
        for (Thread worker : workers) {
            join(worker, 20000);
            check(!worker.isAlive(), "increment worker did not finish in time");
        }

        check(shared.get() == (long) THREADS * INCREMENTS, "no increment lost");
        long totalRetries = 0;
        for (long retryCount : retriesPerThread) {
            check(retryCount >= 0, "retry counters are non-negative");
            totalRetries += retryCount;
        }
        // Exact CAS accounting: every retry corresponds to exactly one failed CAS.
        check(totalRetries == shared.casFailures(), "retries equal recorded CAS failures");
        check(shared.casAttempts() == (long) THREADS * INCREMENTS + shared.casFailures(),
              "attempts == increments + failures");

        System.out.println("PASS");
    }

    static void join(Thread thread, long timeoutMillis) {
        try {
            thread.join(timeoutMillis);
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
        }
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
