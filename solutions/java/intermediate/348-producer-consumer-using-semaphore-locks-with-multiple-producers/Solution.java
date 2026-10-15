/**
 * Task 348 - Multi-producer/multi-consumer with counting semaphores. (Tier: Intermediate, CTCI Ch 15)
 *
 * Bounded buffer of capacity 7: a Semaphore(1) mutex plus the counting semaphores `empty` (free
 * slots) and `full` (queued items).  Two producers fill disjoint halves of 0..999, two consumers
 * drain.  After the producers finish, the main thread reserves slot permits and appends one -1
 * sentinel per consumer, so consumers stop deterministically with no timing guess.  Assertions:
 * every item consumed exactly once, per-producer order preserved, capacity never exceeded, and
 * all threads terminate (daemons + bounded joins).
 */
import java.util.*;
import java.util.concurrent.Semaphore;

public class Solution {
    static void produce(Semaphore empty, Semaphore full, Semaphore mutex, Deque<Integer> buffer,
                        int[] maxSeen, int from, int to) {
        try {
            for (int item = from; item < to; item++) {
                empty.acquire();
                mutex.acquire();
                try {
                    buffer.addLast(item);
                    maxSeen[0] = Math.max(maxSeen[0], buffer.size());
                } finally {
                    mutex.release();
                }
                full.release();
            }
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
        }
    }

    static void consume(Semaphore empty, Semaphore full, Semaphore mutex, Deque<Integer> buffer,
                        List<int[]> consumed, int sentinel) {
        try {
            while (true) {
                full.acquire();
                mutex.acquire();
                int item = buffer.removeFirst();
                if (item == sentinel) {
                    mutex.release();
                    break;                    // the sentinel's slot permit stays consumed
                }
                consumed.add(new int[]{consumed.size(), item});   // global order recorded under the mutex
                mutex.release();
                empty.release();
            }
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
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

    public static void main(String[] args) throws InterruptedException {
        final int capacity = 7;
        final int items = 1000;
        final int sentinel = -1;
        Semaphore empty = new Semaphore(capacity);
        Semaphore full = new Semaphore(0);
        Semaphore mutex = new Semaphore(1);
        Deque<Integer> buffer = new ArrayDeque<>();
        List<int[]> consumed = new ArrayList<>();
        int[] maxSeen = {0};

        Thread producerA = new Thread(() -> produce(empty, full, mutex, buffer, maxSeen, 0, items / 2), "PA");
        Thread producerB = new Thread(() -> produce(empty, full, mutex, buffer, maxSeen, items / 2, items), "PB");
        Thread consumerA = new Thread(() -> consume(empty, full, mutex, buffer, consumed, sentinel), "CA");
        Thread consumerB = new Thread(() -> consume(empty, full, mutex, buffer, consumed, sentinel), "CB");
        for (Thread thread : new Thread[]{producerA, producerB, consumerA, consumerB}) thread.setDaemon(true);
        for (Thread thread : new Thread[]{consumerA, consumerB, producerA, producerB}) thread.start();

        check(join(producerA) && join(producerB), "producers finished");

        for (int i = 0; i < 2; i++) {         // reserve a slot and append one sentinel per consumer
            empty.acquire();
            mutex.acquire();
            buffer.addLast(sentinel);
            mutex.release();
            full.release();
        }

        check(join(consumerA) && join(consumerB), "consumers finished");

        check(consumed.size() == items, "no item lost or duplicated");
        int[] ordered = new int[consumed.size()];
        for (int i = 0; i < ordered.length; i++) {
            check(consumed.get(i)[0] == i, "consumption index " + i);
            ordered[i] = consumed.get(i)[1];
        }
        int[] sorted = ordered.clone();
        Arrays.sort(sorted);
        for (int i = 0; i < items; i++) check(sorted[i] == i, "every item exactly once: " + i);
        check(maxSeen[0] <= capacity, "capacity never exceeded");
        check(maxSeen[0] >= 1, "buffer was actually used");

        int expected = 0;
        for (int item : ordered) {
            if (item < items / 2) check(item == expected++, "producer A order preserved");
        }
        check(expected == items / 2, "all producer A items consumed");
        expected = items / 2;
        for (int item : ordered) {
            if (item >= items / 2) check(item == expected++, "producer B order preserved");
        }
        check(expected == items, "all producer B items consumed");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
