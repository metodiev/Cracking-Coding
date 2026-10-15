/** Task 172 - Implement stack that supports getMin in O(1). (Tier: Beginner, CTCI Ch 3) */
import java.util.*;

public class Solution {
    static class MinStack {
        private final Deque<Integer> main = new ArrayDeque<>();
        private final Deque<Integer> mins = new ArrayDeque<>();

        void push(int value) {
            main.push(value);
            mins.push(mins.isEmpty() ? value : Math.min(value, mins.peek()));
        }

        int pop() {
            if (main.isEmpty()) throw new NoSuchElementException("stack empty");
            mins.pop();
            return main.pop();
        }

        int peek() {
            if (main.isEmpty()) throw new NoSuchElementException("stack empty");
            return main.peek();
        }

        int min() {
            if (mins.isEmpty()) throw new NoSuchElementException("stack empty");
            return mins.peek();
        }

        boolean isEmpty() { return main.isEmpty(); }
        int size() { return main.size(); }
    }

    public static void main(String[] args) {
        MinStack s = new MinStack();
        for (int v : new int[]{5, 3, 7, 3, 9}) s.push(v);
        check(s.min() == 3 && s.peek() == 9 && s.size() == 5, "state after pushes");
        check(s.pop() == 9 && s.min() == 3, "pop 9");
        s.push(1);
        check(s.min() == 1, "new minimum");
        check(s.pop() == 1 && s.min() == 3, "pop 1");
        check(s.pop() == 3 && s.min() == 3, "duplicate minimum preserved");
        check(s.pop() == 7 && s.min() == 3, "pop 7");
        check(s.pop() == 3 && s.min() == 5, "back to 5");
        check(s.pop() == 5 && s.isEmpty(), "drained");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
