/** Task 173 - Implement queue that supports getMin in O(1). (Tier: Beginner, CTCI Ch 3) */
import java.util.*;

public class Solution {
    static class MinStackInner {
        final Deque<Integer> data = new ArrayDeque<>();
        final Deque<Integer> mins = new ArrayDeque<>();

        void push(int value) {
            data.push(value);
            mins.push(mins.isEmpty() ? value : Math.min(value, mins.peek()));
        }

        int pop() { mins.pop(); return data.pop(); }
        int peek() { return data.peek(); }
        int min() { return mins.peek(); }
        int size() { return data.size(); }
    }

    static class MinQueue {
        final MinStackInner in = new MinStackInner();
        final MinStackInner out = new MinStackInner();

        void enqueue(int value) { in.push(value); }

        int dequeue() {
            if (isEmpty()) throw new NoSuchElementException("queue empty");
            if (out.size() == 0) while (in.size() > 0) out.push(in.pop());
            return out.pop();
        }

        int min() {
            if (isEmpty()) throw new NoSuchElementException("queue empty");
            if (in.size() == 0) return out.min();
            if (out.size() == 0) return in.min();
            return Math.min(in.min(), out.min());
        }

        boolean isEmpty() { return in.size() == 0 && out.size() == 0; }
    }

    public static void main(String[] args) {
        MinQueue q = new MinQueue();
        for (int v : new int[]{3, 1, 4, 1, 5}) q.enqueue(v);
        check(q.min() == 1, "min after enqueues");
        check(q.dequeue() == 3 && q.min() == 1, "dequeue 3");
        check(q.dequeue() == 1 && q.min() == 1, "duplicate min kept");
        check(q.dequeue() == 4 && q.min() == 1, "dequeue 4");
        check(q.dequeue() == 1 && q.min() == 5, "min moves to 5");
        q.enqueue(0);
        check(q.min() == 0, "new min");
        check(q.dequeue() == 5 && q.dequeue() == 0 && q.isEmpty(), "drained");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
