/** Task 064 - Implement circular queue (ring buffer). (Tier: Beginner, CTCI Ch 3) */
class CircularQueue {
    private final Integer[] data;
    private int head = 0, size = 0;

    CircularQueue(int capacity) {
        if (capacity <= 0) throw new IllegalArgumentException("capacity must be positive");
        data = new Integer[capacity];
    }

    void enqueue(int value) {
        if (isFull()) throw new IllegalStateException("queue full");
        data[(head + size) % data.length] = value;
        size++;
    }

    int dequeue() {
        if (isEmpty()) throw new IllegalStateException("queue empty");
        int value = data[head];
        data[head] = null;
        head = (head + 1) % data.length;
        size--;
        return value;
    }

    int front() {
        if (isEmpty()) throw new IllegalStateException("queue empty");
        return data[head];
    }

    int rear() {
        if (isEmpty()) throw new IllegalStateException("queue empty");
        return data[(head + size - 1) % data.length];
    }

    boolean isEmpty() { return size == 0; }
    boolean isFull() { return size == data.length; }
    int size() { return size; }
}

public class Solution {
    public static void main(String[] args) {
        CircularQueue q = new CircularQueue(3);
        check(q.isEmpty() && !q.isFull(), "starts empty");
        q.enqueue(1); q.enqueue(2); q.enqueue(3);
        check(q.isFull() && q.size() == 3, "fills up");
        check(q.front() == 1 && q.rear() == 3, "front/rear");
        check(q.dequeue() == 1, "dequeue");
        q.enqueue(4);
        check(q.front() == 2 && q.rear() == 4, "wrap-around");
        check(q.dequeue() == 2 && q.dequeue() == 3 && q.dequeue() == 4, "drain in order");
        check(q.isEmpty(), "empty at end");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
