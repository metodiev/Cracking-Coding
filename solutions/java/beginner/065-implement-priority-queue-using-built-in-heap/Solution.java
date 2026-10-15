/** Task 065 - Implement priority queue using built-in heap. (Tier: Beginner, CTCI Ch 3) */
import java.util.*;

public class Solution {
    static class PriorityQueueRunner<T> {
        private final java.util.PriorityQueue<Object[]> heap;
        private int counter = 0;

        PriorityQueueRunner() {
            heap = new java.util.PriorityQueue<>(
                (a, b) -> {
                    int cmp = Integer.compare((int) a[0], (int) b[0]);
                    if (cmp != 0) return cmp;
                    return Integer.compare((int) a[1], (int) b[1]);
                });
        }

        void push(T item, int priority) {
            heap.add(new Object[]{priority, counter++, item});
        }

        @SuppressWarnings("unchecked")
        T pop() {
            if (heap.isEmpty()) throw new NoSuchElementException("queue empty");
            return (T) heap.poll()[2];
        }

        @SuppressWarnings("unchecked")
        T peek() {
            if (heap.isEmpty()) throw new NoSuchElementException("queue empty");
            return (T) heap.peek()[2];
        }

        int size() { return heap.size(); }
    }

    public static void main(String[] args) {
        PriorityQueueRunner<String> pq = new PriorityQueueRunner<>();
        pq.push("low", 5);
        pq.push("high", 1);
        pq.push("mid", 3);
        check(pq.peek().equals("high"), "peek min priority");
        check(pq.pop().equals("high"), "pop 1");
        check(pq.pop().equals("mid"), "pop 2");
        check(pq.pop().equals("low"), "pop 3");
        check(pq.size() == 0, "emptied");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
