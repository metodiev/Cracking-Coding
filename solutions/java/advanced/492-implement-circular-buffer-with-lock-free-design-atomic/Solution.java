/** Task 492 - Implement circular buffer with lock-free design (atomic). (Tier: Advanced, CTCI Ch 15)
 *
 * Bounded MPMC ring buffer in the Vyukov style: capacity slots each carry a sequence number
 * (slot i starts at i), head and tail are AtomicLongs advanced only by compare-and-set, and
 * every operation retries when its CAS loses the race. A slot is publishable when seq == pos
 * and consumable when seq == pos + 1; the claimer overwrites the slot and then stores
 * seq = pos + 1 (enqueue) or seq = pos + capacity (dequeue), which re-opens the slot one lap
 * later. One extra occupancy check (tail - head >= capacity => full) is required because for
 * capacity 1 the final lap's free and published sequence values coincide. tryEnqueue never
 * blocks: it reports false (full); tryDequeue reports null (empty), so callers spin outside.
 * The self-test asserts: single-threaded wraparound/full/empty semantics, exact global FIFO for
 * one producer / one consumer across REAL threads (bounded capacity forces both to spin),
 * exactly-once delivery plus per-producer FIFO for many producers / consumers - the position
 * claimed by the head CAS acts as the linearization witness - and a deterministic hook that
 * lets a competing producer claim the position just before the outer CAS, so the retry path is
 * exercised and must fail exactly once. All worker threads join with a timeout and are checked
 * to have terminated.
 */
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicLongArray;

class LockFreeCircularBuffer {
    interface CasHook {
        void run(LockFreeCircularBuffer buffer, long pos);
    }

    private final int capacity;
    private final Object[] slots;
    private final AtomicLongArray seq;
    final AtomicLong head = new AtomicLong();
    final AtomicLong tail = new AtomicLong();
    final AtomicLong casAttempts = new AtomicLong();
    final AtomicLong casFailures = new AtomicLong();
    final AtomicLong enqueueRetries = new AtomicLong();
    final AtomicLong dequeueRetries = new AtomicLong();
    private volatile CasHook casHook;

    LockFreeCircularBuffer(int capacity, CasHook hook) {
        if (capacity <= 0) throw new IllegalArgumentException("capacity must be positive");
        this.capacity = capacity;
        this.slots = new Object[capacity];
        long[] initial = new long[capacity];
        for (int i = 0; i < capacity; i++) initial[i] = i;      // slot i is free at position i
        this.seq = new AtomicLongArray(initial);
        this.casHook = hook;
    }

    /** True when value was enqueued, false when the queue is full. */
    boolean tryEnqueue(Object value) {
        while (true) {
            long pos = tail.get();
            if (pos - head.get() >= capacity) {
                return false;                       // occupancy guard (needed for capacity 1)
            }
            int slot = (int) (pos % capacity);
            long slotSeq = seq.get(slot);
            if (slotSeq == pos) {
                CasHook hook = casHook;
                if (hook != null) {
                    casHook = null;                 // fires once
                    hook.run(this, pos);
                }
                casAttempts.incrementAndGet();
                if (tail.compareAndSet(pos, pos + 1)) {
                    slots[slot] = value;
                    seq.set(slot, pos + 1);         // publish (release store)
                    return true;
                }
                casFailures.incrementAndGet();      // lost the race: reread and retry
            } else if (slotSeq < pos) {
                return false;                       // the previous lap has not caught up
            }
            enqueueRetries.incrementAndGet();
        }
    }

    /** Oldest value, or null when empty. When claimSink is given, claimSink[0] receives the
     * position claimed by the head CAS (the linearization witness used by the self-test). */
    Object tryDequeue(long[] claimSink) {
        while (true) {
            long pos = head.get();
            int slot = (int) (pos % capacity);
            long slotSeq = seq.get(slot);
            if (slotSeq == pos + 1) {
                casAttempts.incrementAndGet();
                if (head.compareAndSet(pos, pos + 1)) {
                    if (claimSink != null) claimSink[0] = pos;
                    Object value = slots[slot];
                    seq.set(slot, pos + capacity);  // reopen one lap later
                    return value;
                }
                casFailures.incrementAndGet();
            } else if (slotSeq < pos + 1) {
                return null;                        // nothing published yet: empty
            }
            dequeueRetries.incrementAndGet();
        }
    }
}

public class Solution {
    private static void join(Thread thread, long timeoutMillis) {
        try {
            thread.join(timeoutMillis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("interrupted while joining", e);
        }
    }

    public static void main(String[] args) {
        // Single-threaded semantics: wraparound, full, empty, capacity 1.
        LockFreeCircularBuffer buf = new LockFreeCircularBuffer(4, null);
        for (int value = 1; value <= 4; value++) check(buf.tryEnqueue(value), "enqueue " + value);
        check(!buf.tryEnqueue(5), "full");
        for (int value = 1; value <= 4; value++) {
            check(Objects.equals(buf.tryDequeue(null), value), "dequeue " + value);
        }
        check(buf.tryDequeue(null) == null, "empty");
        check(buf.tryEnqueue(5) && buf.tryEnqueue(6), "after wrap");
        check(Objects.equals(buf.tryDequeue(null), 5) && Objects.equals(buf.tryDequeue(null), 6),
                "wrapped twice");
        check(buf.tryDequeue(null) == null, "empty again");

        LockFreeCircularBuffer single = new LockFreeCircularBuffer(1, null);
        check(single.tryEnqueue("a"), "cap1 push");
        check(!single.tryEnqueue("b"), "cap1 full");
        check(Objects.equals(single.tryDequeue(null), "a"), "cap1 pop");
        check(single.tryEnqueue("b"), "cap1 push again");
        check(Objects.equals(single.tryDequeue(null), "b"), "cap1 pop again");

        // Deterministic retry-path demonstration: the hook, running just before the outer CAS,
        // lets a "concurrent" producer claim the same position; the outer CAS must fail exactly
        // once, reread the tail and succeed on the next lap, hook's item first in FIFO order.
        LockFreeCircularBuffer forced = new LockFreeCircularBuffer(8, (buffer, pos) -> {
            check(buffer.tryEnqueue("stealer"), "hook enqueue");
            check(buffer.head.get() == 0 && buffer.tail.get() == pos + 1, "hook claim");
        });
        check(forced.tryEnqueue("mine"), "outer enqueue");
        check(forced.casFailures.get() == 1, "stale CAS fails exactly once");
        check(forced.enqueueRetries.get() == 1, "exactly one retry lap");
        check(Objects.equals(forced.tryDequeue(null), "stealer"), "hook item first");
        check(Objects.equals(forced.tryDequeue(null), "mine"), "retried item second");
        check(forced.tryDequeue(null) == null, "forced empty");

        // SPSC across real threads: bounded capacity forces the producer to spin on FULL and
        // the consumer to spin on EMPTY; the dequeued stream must be exactly 0, 1, ..., total-1.
        int total = 20000;
        LockFreeCircularBuffer spsc = new LockFreeCircularBuffer(8, null);
        List<Integer> received = new ArrayList<>();
        Thread producer = new Thread(() -> {
            for (int value = 0; value < total; value++) {
                while (!spsc.tryEnqueue(value)) Thread.yield();
            }
        });
        Thread consumer = new Thread(() -> {
            while (received.size() < total) {
                Object value = spsc.tryDequeue(null);
                if (value == null) {
                    Thread.yield();
                } else {
                    received.add((Integer) value);
                }
            }
        });
        producer.start();
        consumer.start();
        join(producer, 30000);
        join(consumer, 30000);
        check(!producer.isAlive() && !consumer.isAlive(), "SPSC threads terminate");
        check(received.size() == total, "SPSC count");
        for (int i = 0; i < total; i++) {
            check(received.get(i) == i, "SPSC FIFO at " + i);
        }

        // MPMC accounting: 4 producers x 300 tagged items, 3 consumers, capacity 16. Consumers
        // record the claimed position, so the merged record reproduces the dequeue order:
        // positions 0..total-1 exactly once, and each producer's items in its enqueue order.
        int producers = 4;
        int perProducer = 300;
        int consumers = 3;
        int grand = producers * perProducer;
        LockFreeCircularBuffer mpmc = new LockFreeCircularBuffer(16, null);
        List<List<long[]>> records = new ArrayList<>();
        for (int c = 0; c < consumers; c++) records.add(new ArrayList<>());
        AtomicLong tally = new AtomicLong();

        Thread[] workers = new Thread[producers + consumers];
        for (int p = 0; p < producers; p++) {
            final int producerIndex = p;
            workers[p] = new Thread(() -> {
                for (int i = 0; i < perProducer; i++) {
                    int[] item = {producerIndex, i};
                    while (!mpmc.tryEnqueue(item)) Thread.yield();
                }
            });
        }
        for (int c = 0; c < consumers; c++) {
            final int consumerIndex = c;
            workers[producers + c] = new Thread(() -> {
                List<long[]> local = records.get(consumerIndex);
                while (tally.get() < grand) {
                    long[] claim = new long[1];
                    Object value = mpmc.tryDequeue(claim);
                    if (value == null) {
                        Thread.yield();
                        continue;
                    }
                    int[] item = (int[]) value;
                    local.add(new long[]{claim[0], item[0], item[1]});
                    tally.incrementAndGet();
                }
            });
        }
        for (Thread worker : workers) worker.start();
        for (Thread worker : workers) join(worker, 30000);
        for (Thread worker : workers) check(!worker.isAlive(), "MPMC worker terminates");

        check(tally.get() == grand, "exactly-once delivery");
        List<long[]> merged = new ArrayList<>();
        for (List<long[]> local : records) merged.addAll(local);
        merged.sort(Comparator.comparingLong(record -> record[0]));
        check(merged.size() == grand, "merged size");
        for (int i = 0; i < grand; i++) {
            check(merged.get(i)[0] == i, "claimed positions are a permutation at " + i);
        }
        List<List<Long>> order = new ArrayList<>();
        for (int p = 0; p < producers; p++) order.add(new ArrayList<>());
        for (long[] record : merged) order.get((int) record[1]).add(record[2]);
        for (int p = 0; p < producers; p++) {
            for (int i = 0; i < perProducer; i++) {
                check(order.get(p).get(i) == i, "per-producer FIFO producer " + p + " item " + i);
            }
        }
        check(mpmc.casAttempts.get() >= grand, "every operation used the CAS path");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
