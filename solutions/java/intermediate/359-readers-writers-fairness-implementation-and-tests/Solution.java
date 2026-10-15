/** Task 359 - Readers-writers fairness implementation and tests. (Tier: Intermediate, CTCI Ch 15) */
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public class Solution {
    static final int READERS = 4;
    static final int WRITERS = 3;
    static final int ITERATIONS = 50;

    /**
     * Readers-writers lock with the documented WRITER-PREFERENCE policy: a reader may
     * enter only when no writer is active and no writer is waiting (waitingWriters > 0),
     * so a stream of arriving readers can never starve a writer; a writer enters when no
     * writer is active and all active readers have left. Mutual exclusion is checked on
     * every entry/exit, so a writer releasing while readers are active (or a reader next
     * to a writer) would be counted as a violation.
     */
    static class WriterPreferenceRW {
        private int activeReaders;
        private boolean activeWriter;
        private int waitingWriters;
        private int readerEntries;
        private int writerEntries;
        private int maxActiveReaders;
        private int violations;

        synchronized void acquireRead() throws InterruptedException {
            while (activeWriter || waitingWriters > 0) wait();
            if (activeWriter) violations++;               // unreachable: writer would overlap
            activeReaders++;
            maxActiveReaders = Math.max(maxActiveReaders, activeReaders);
            readerEntries++;
        }

        synchronized void releaseRead() {
            if (activeReaders <= 0 || activeWriter) violations++;
            activeReaders--;
            if (activeReaders == 0) notifyAll();          // last reader lets writers proceed
        }

        synchronized void acquireWrite(Runnable beforeWait) throws InterruptedException {
            waitingWriters++;
            if (beforeWait != null) beforeWait.run();     // hook runs with the intent registered
            while (activeWriter || activeReaders > 0) wait();
            if (activeWriter || activeReaders > 0) violations++;
            waitingWriters--;
            activeWriter = true;
            writerEntries++;
        }

        synchronized void acquireWrite() throws InterruptedException {
            acquireWrite(null);
        }

        synchronized void releaseWrite() {
            if (!activeWriter || activeReaders > 0) violations++;
            activeWriter = false;
            notifyAll();
        }

        synchronized int readerEntries() { return readerEntries; }
        synchronized int writerEntries() { return writerEntries; }
        synchronized int maxActiveReaders() { return maxActiveReaders; }
        synchronized int violations() { return violations; }
        synchronized boolean isClean() { return activeReaders == 0 && !activeWriter; }
    }

    public static void main(String[] args) {
        // Part 1: scripted writer-preference ordering test.
        WriterPreferenceRW rw = new WriterPreferenceRW();
        List<String> log = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch readerAAcquired = new CountDownLatch(1);
        CountDownLatch releaseA = new CountDownLatch(1);
        CountDownLatch writerRegistered = new CountDownLatch(1);
        CountDownLatch writerEntered = new CountDownLatch(1);
        CountDownLatch releaseW = new CountDownLatch(1);
        CountDownLatch readerBEntered = new CountDownLatch(1);

        Thread readerA = new Thread(() -> {
            try {
                rw.acquireRead();
                log.add("readA");
                readerAAcquired.countDown();
                if (!releaseA.await(10, TimeUnit.SECONDS)) throw new AssertionError("reader A released in time");
                rw.releaseRead();
            } catch (InterruptedException exc) {
                Thread.currentThread().interrupt();
            }
        });
        Thread writerW = new Thread(() -> {
            try {
                rw.acquireWrite(writerRegistered::countDown);
                log.add("writeW");
                writerEntered.countDown();
                if (!releaseW.await(10, TimeUnit.SECONDS)) throw new AssertionError("writer W released in time");
                rw.releaseWrite();
            } catch (InterruptedException exc) {
                Thread.currentThread().interrupt();
            }
        });
        Thread readerB = new Thread(() -> {
            try {
                rw.acquireRead();
                log.add("readB");
                readerBEntered.countDown();
                rw.releaseRead();
            } catch (InterruptedException exc) {
                Thread.currentThread().interrupt();
            }
        });

        readerA.start();
        check(await(readerAAcquired), "reader A got in first");
        writerW.start();
        check(await(writerRegistered), "writer W registered its request");
        readerB.start();                                  // reader B arrives while W waits

        releaseA.countDown();
        check(await(writerEntered), "writer W entered after reader A left");
        check(readerBEntered.getCount() == 1, "new reader waits while a writer is pending");
        releaseW.countDown();
        check(await(readerBEntered), "reader B entered once the writer was done");

        join(readerA);
        join(writerW);
        join(readerB);
        check(log.equals(List.of("readA", "writeW", "readB")), "writer-preference order");
        check(rw.writerEntries() == 1 && rw.readerEntries() == 2, "entries counted");
        check(rw.violations() == 0, "scripted scenario kept mutual exclusion");

        // Part 2 + 3: threaded workload with the fairness accounting.
        WriterPreferenceRW workload = new WriterPreferenceRW();
        int[] state = {0};                                // guarded by the write lock
        List<Integer> observations = Collections.synchronizedList(new ArrayList<>());
        int[] writerCompletions = new int[WRITERS];
        int[] readerCompletions = new int[READERS];

        Thread[] workers = new Thread[READERS + WRITERS];
        for (int i = 0; i < READERS; i++) {
            final int readerId = i;
            workers[i] = new Thread(() -> {
                try {
                    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
                        workload.acquireRead();
                        observations.add(state[0]);
                        workload.releaseRead();
                        readerCompletions[readerId]++;
                    }
                } catch (InterruptedException exc) {
                    Thread.currentThread().interrupt();
                }
            });
        }
        for (int i = 0; i < WRITERS; i++) {
            final int writerId = i;
            workers[READERS + i] = new Thread(() -> {
                try {
                    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
                        workload.acquireWrite();
                        state[0]++;
                        workload.releaseWrite();
                        writerCompletions[writerId]++;
                    }
                } catch (InterruptedException exc) {
                    Thread.currentThread().interrupt();
                }
            });
        }
        for (Thread worker : workers) worker.start();
        for (Thread worker : workers) {
            join(worker);
            check(!worker.isAlive(), "workload thread did not finish in time");
        }

        int totalWriterCompletions = 0;
        for (int count : writerCompletions) totalWriterCompletions += count;
        check(totalWriterCompletions == WRITERS * ITERATIONS, "every writer request was served");
        check(workload.writerEntries() == WRITERS * ITERATIONS && workload.writerEntries() == 150,
              "writer entries == writer requests (no starvation)");
        check(workload.readerEntries() == READERS * ITERATIONS && workload.readerEntries() == 200,
              "reader entries == reader requests");
        check(state[0] == 150, "all writer increments applied exactly once");
        check(workload.violations() == 0, "mutual exclusion held throughout the workload");
        check(observations.size() == 200, "every read produced an observation");
        for (int value : observations) check(0 <= value && value <= 150, "readers only saw committed values");
        check(workload.maxActiveReaders() >= 1, "readers actually ran");
        check(workload.isClean(), "lock left clean");

        System.out.println("PASS");
    }

    static boolean await(CountDownLatch latch) {
        try {
            return latch.await(10, TimeUnit.SECONDS);
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    static void join(Thread thread) {
        try {
            thread.join(10000);
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
        }
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
