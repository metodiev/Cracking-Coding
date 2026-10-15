/** Task 195 - Implement basic producer-consumer using threads (mutex/condition). (Tier: Beginner, CTCI Ch 15) */
import java.util.*;

public class Solution {
    static class BoundedBuffer {
        private final int capacity;
        private final Deque<Integer> items = new ArrayDeque<>();
        private boolean closed = false;
        private int maxSeen = 0;

        BoundedBuffer(int capacity) {
            if (capacity <= 0) throw new IllegalArgumentException("capacity must be positive");
            this.capacity = capacity;
        }

        synchronized void put(int item) throws InterruptedException {
            while (!closed && items.size() >= capacity) wait();   // block while full
            if (closed) return;
            items.addLast(item);
            maxSeen = Math.max(maxSeen, items.size());
            notifyAll();
        }

        synchronized Integer take() throws InterruptedException {
            while (!closed && items.isEmpty()) wait();            // block while empty
            if (items.isEmpty()) return null;
            int item = items.removeFirst();
            notifyAll();
            return item;
        }

        synchronized void close() {
            closed = true;
            notifyAll();
        }

        synchronized int maxSeen() {
            return maxSeen;
        }
    }

    public static void main(String[] args) {
        final int items = 1000;
        final int capacity = 7;
        BoundedBuffer buffer = new BoundedBuffer(capacity);
        List<Integer> consumedA = new ArrayList<>();
        List<Integer> consumedB = new ArrayList<>();

        Thread consumerA = new Thread(() -> consumeAll(buffer, consumedA));
        Thread consumerB = new Thread(() -> consumeAll(buffer, consumedB));
        Thread producerA = new Thread(() -> produce(buffer, 0, items / 2));
        Thread producerB = new Thread(() -> produce(buffer, items / 2, items));

        consumerA.start();
        consumerB.start();
        producerA.start();
        producerB.start();

        check(join(producerA), "producer A terminated");
        check(join(producerB), "producer B terminated");
        buffer.close();
        check(join(consumerA), "consumer A terminated");
        check(join(consumerB), "consumer B terminated");

        List<Integer> consumed = new ArrayList<>(consumedA);
        consumed.addAll(consumedB);
        check(consumed.size() == items, "no items lost or duplicated");
        Collections.sort(consumed);
        List<Integer> expected = new ArrayList<>();
        for (int i = 0; i < items; i++) expected.add(i);
        check(consumed.equals(expected), "every item consumed exactly once");
        check(buffer.maxSeen() <= capacity, "capacity never exceeded");
        check(buffer.maxSeen() >= 1, "buffer was actually used");
        System.out.println("PASS");
    }

    static void produce(BoundedBuffer buffer, int from, int to) {
        try {
            for (int item = from; item < to; item++) buffer.put(item);
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
        }
    }

    static void consumeAll(BoundedBuffer buffer, List<Integer> out) {
        try {
            while (true) {
                Integer item = buffer.take();
                if (item == null) break;
                out.add(item);
            }
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
        }
    }

    static boolean join(Thread thread) {
        try {
            thread.join(10000);                                   // timeout sanity: must terminate
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
        }
        return !thread.isAlive();
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
