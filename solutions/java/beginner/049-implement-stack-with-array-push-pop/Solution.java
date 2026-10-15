/** Task 049 - Implement stack with array (push/pop). (Tier: Beginner, CTCI Ch 3) */
class StackArray {
    private final int[] data;
    private int top = -1;

    StackArray(int capacity) { data = new int[capacity]; }

    void push(int value) {
        if (top + 1 == data.length) throw new IllegalStateException("stack full");
        data[++top] = value;
    }

    int pop() {
        if (isEmpty()) throw new IllegalStateException("stack empty");
        return data[top--];
    }

    int peek() {
        if (isEmpty()) throw new IllegalStateException("stack empty");
        return data[top];
    }

    boolean isEmpty() { return top < 0; }
    int size() { return top + 1; }
}

public class Solution {
    public static void main(String[] args) {
        StackArray s = new StackArray(3);
        check(s.isEmpty(), "starts empty");
        s.push(1); s.push(2); s.push(3);
        check(s.peek() == 3 && s.size() == 3, "after pushes");
        check(s.pop() == 3 && s.pop() == 2 && s.pop() == 1, "LIFO order");
        check(s.isEmpty(), "ends empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
