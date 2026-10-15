/** Task 050 - Implement queue with array (enqueue/dequeue). (Tier: Beginner, CTCI Ch 3) */
class QueueArray {
    private final int[] data;
    private int head = 0, size = 0;

    QueueArray(int capacity) { data = new int[capacity]; }

    void enqueue(int value) {
        if (size == data.length) throw new IllegalStateException("queue full");
        data[(head + size) % data.length] = value;
        size++;
    }

    int dequeue() {
        if (isEmpty()) throw new IllegalStateException("queue empty");
        int value = data[head];
        head = (head + 1) % data.length;
        size--;
        return value;
    }

    int front() {
        if (isEmpty()) throw new IllegalStateException("queue empty");
        return data[head];
    }

    boolean isEmpty() { return size == 0; }
    int size() { return size; }
}

public class Solution {
    public static void main(String[] args) {
        QueueArray q = new QueueArray(3);
        q.enqueue(1); q.enqueue(2); q.enqueue(3);
        check(q.front() == 1, "front");
        check(q.dequeue() == 1, "dequeue 1");
        q.enqueue(4);
        check(q.dequeue() == 2 && q.dequeue() == 3 && q.dequeue() == 4, "wrap-around FIFA order");
        check(q.isEmpty(), "empty at end");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
