/**
 * Task 347 - Read-write lock with writer preference (no writer starvation). (Tier: Intermediate, CTCI Ch 15)
 *
 * Policy: once a writer waits, newly arriving readers block; waiting writers are served when the
 * current readers/writer drain, so readers cannot starve writers.  The scripted demo uses real
 * threads, but every ordering decision is passed through CountDownLatch gates; acquire() takes an
 * optional Runnable that fires once the caller has registered as waiting while still holding the
 * monitor, so a release cannot slip between registration and wait() (no lost wake-up).  Bounded
 * waits are liveness checks only and threads are daemons, so a stuck run fails fast.
 */
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

class ReadWriteLock {
    private int readers = 0;
    private boolean writer = false;
    private int writersWaiting = 0;

    void acquireRead() {
        acquireRead(null);
    }

    synchronized void acquireRead(Runnable onWaiting) {
        boolean notified = false;
        while (writer || writersWaiting > 0) {
            if (!notified && onWaiting != null) {
                onWaiting.run();                       // fired while holding the monitor
                notified = true;
            }
            waitFor();
        }
        readers++;
    }

    synchronized void releaseRead() {
        readers--;
        if (readers == 0) notifyAll();
    }

    void acquireWrite() {
        acquireWrite(null);
    }

    synchronized void acquireWrite(Runnable onWaiting) {
        writersWaiting++;
        boolean notified = false;
        while (writer || readers > 0) {
            if (!notified && onWaiting != null) {
                onWaiting.run();
                notified = true;
            }
            waitFor();
        }
        writersWaiting--;
        writer = true;
    }

    synchronized void releaseWrite() {
        writer = false;
        notifyAll();
    }

    synchronized int[] state() {
        return new int[]{readers, writer ? 1 : 0, writersWaiting};
    }

    private void waitFor() {
        try {
            wait();
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(exc);
        }
    }
}

/** Invariant tracker: readers and writers must never overlap. */
class LockAuditor {
    private int activeReaders = 0;
    private int activeWriters = 0;
    private int maxReaders = 0;
    private int violations = 0;
    private final List<String> grants = new ArrayList<>();

    synchronized void enterRead(String tag) {
        if (activeWriters > 0) violations++;
        activeReaders++;
        maxReaders = Math.max(maxReaders, activeReaders);
        grants.add("R:" + tag);
    }

    synchronized void exitRead() {
        activeReaders--;
    }

    synchronized void enterWrite(String tag) {
        if (activeWriters > 0 || activeReaders > 0) violations++;
        activeWriters++;
        grants.add("W:" + tag);
    }

    synchronized void exitWrite() {
        activeWriters--;
    }

    synchronized int[] counters() {
        return new int[]{activeReaders, activeWriters, maxReaders, violations};
    }

    synchronized List<String> grants() {
        return new ArrayList<>(grants);
    }
}

class Gate {
    final CountDownLatch started = new CountDownLatch(1);
    final CountDownLatch entered = new CountDownLatch(1);
    final CountDownLatch registered = new CountDownLatch(1);
    final CountDownLatch release = new CountDownLatch(1);
    final CountDownLatch done = new CountDownLatch(1);
}

public class Solution {
    static boolean await(CountDownLatch latch) {
        try {
            return latch.await(15, TimeUnit.SECONDS);
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    static boolean join(Thread thread) {
        try {
            thread.join(15000);
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
        }
        return !thread.isAlive();
    }

    static Thread readerThread(ReadWriteLock rw, LockAuditor auditor, String tag, Gate gate, boolean registerFirst) {
        Thread thread = new Thread(() -> {
            gate.started.countDown();
            rw.acquireRead(registerFirst ? gate.registered::countDown : null);
            auditor.enterRead(tag);
            gate.entered.countDown();
            boolean released = await(gate.release);
            auditor.exitRead();
            rw.releaseRead();
            if (released) gate.done.countDown();
        }, tag);
        thread.setDaemon(true);
        return thread;
    }

    static Thread writerThread(ReadWriteLock rw, LockAuditor auditor, String tag, Gate gate, boolean registerFirst) {
        Thread thread = new Thread(() -> {
            gate.started.countDown();
            rw.acquireWrite(registerFirst ? gate.registered::countDown : null);
            auditor.enterWrite(tag);
            gate.entered.countDown();
            boolean released = await(gate.release);
            auditor.exitWrite();
            rw.releaseWrite();
            if (released) gate.done.countDown();
        }, tag);
        thread.setDaemon(true);
        return thread;
    }

    static void start(Map<String, Gate> gates, List<Thread> threads, ReadWriteLock rw, LockAuditor auditor,
                      String tag, boolean registerFirst) {
        Gate gate = gates.get(tag);
        Thread thread = tag.startsWith("R")
                ? readerThread(rw, auditor, tag, gate, registerFirst)
                : writerThread(rw, auditor, tag, gate, registerFirst);
        threads.add(thread);
        thread.start();
        check(await(gate.started), tag + " never started");
    }

    public static void main(String[] args) {
        ReadWriteLock rw = new ReadWriteLock();
        LockAuditor auditor = new LockAuditor();
        Map<String, Gate> gates = new HashMap<>();
        for (String tag : new String[]{"R1", "R2", "R3", "R4", "W1", "W2"}) gates.put(tag, new Gate());
        List<Thread> threads = new ArrayList<>();

        // two readers hold the lock
        start(gates, threads, rw, auditor, "R1", false);
        start(gates, threads, rw, auditor, "R2", false);
        check(await(gates.get("R1").entered) && await(gates.get("R2").entered), "both readers hold the lock");
        check(Arrays.equals(auditor.counters(), new int[]{2, 0, 2, 0}), "two readers active, max 2");
        check(Arrays.equals(rw.state(), new int[]{2, 0, 0}), "rw state: 2 readers");

        // a writer arrives and must register as waiting
        start(gates, threads, rw, auditor, "W1", true);
        check(await(gates.get("W1").registered), "W1 registered as waiting");
        check(gates.get("W1").entered.getCount() == 1, "W1 not granted yet");
        check(Arrays.equals(rw.state(), new int[]{2, 0, 1}), "rw state: one writer waiting");

        // a reader arriving after the writer blocks too: writer preference
        start(gates, threads, rw, auditor, "R3", true);
        check(await(gates.get("R3").registered), "R3 registered as waiting");
        check(gates.get("R3").entered.getCount() == 1, "R3 blocked by the waiting writer");

        // drain the readers; W1 inherits the lock even though R3 is waiting
        gates.get("R1").release.countDown();
        gates.get("R2").release.countDown();
        check(await(gates.get("R1").done) && await(gates.get("R2").done), "readers drained");
        check(await(gates.get("W1").entered), "W1 acquired after readers drained");
        check(gates.get("R3").entered.getCount() == 1, "R3 still blocked while the writer holds");
        check(Arrays.equals(auditor.counters(), new int[]{0, 1, 2, 0}), "writer active, no readers");
        check(Arrays.equals(rw.state(), new int[]{0, 1, 0}), "rw state: writer holds");

        gates.get("W1").release.countDown();
        check(await(gates.get("W1").done), "W1 done");
        check(await(gates.get("R3").entered), "R3 finally granted after the writer");
        check(Arrays.equals(auditor.counters(), new int[]{1, 0, 2, 0}), "reader active again");
        gates.get("R3").release.countDown();
        check(await(gates.get("R3").done), "R3 done");

        // phase 2: writer with no readers acquires immediately; a reader behind it waits
        start(gates, threads, rw, auditor, "W2", false);
        check(await(gates.get("W2").entered), "W2 acquired immediately");
        start(gates, threads, rw, auditor, "R4", true);
        check(await(gates.get("R4").registered), "R4 registered as waiting");
        check(gates.get("R4").entered.getCount() == 1, "R4 blocked behind the writer");
        gates.get("W2").release.countDown();
        check(await(gates.get("W2").done), "W2 done");
        check(await(gates.get("R4").entered), "R4 granted");
        gates.get("R4").release.countDown();
        check(await(gates.get("R4").done), "R4 done");

        for (Thread thread : threads) check(join(thread), "thread terminated");

        int[] counters = auditor.counters();
        check(counters[3] == 0, "readers and writers never overlapped");
        check(counters[2] == 2, "two readers shared the lock at peak");
        check(auditor.grants().equals(Arrays.asList("R:R1", "R:R2", "W:W1", "R:R3", "W:W2", "R:R4")), "grant order");
        check(counters[0] == 0 && counters[1] == 0, "all locks released");
        check(Arrays.equals(rw.state(), new int[]{0, 0, 0}), "final rw state empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
