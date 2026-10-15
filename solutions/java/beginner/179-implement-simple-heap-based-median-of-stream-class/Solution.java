/** Task 179 - Implement simple heap-based median-of-stream class. (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

public class Solution {
    /** Hand-rolled binary heap; maxFirst=true gives a max-heap on top of ints. */
    static class IntHeap {
        private int[] data = new int[4];
        private int size = 0;
        private final boolean maxFirst;

        IntHeap(boolean maxFirst) { this.maxFirst = maxFirst; }

        private boolean better(int a, int b) { return maxFirst ? a > b : a < b; }

        void push(int value) {
            if (size == data.length) data = Arrays.copyOf(data, data.length * 2);
            data[size] = value;
            int i = size++;
            while (i > 0) {
                int parent = (i - 1) / 2;
                if (!better(data[i], data[parent])) break;
                int t = data[i];
                data[i] = data[parent];
                data[parent] = t;
                i = parent;
            }
        }

        int pop() {
            if (size == 0) throw new NoSuchElementException("heap empty");
            int top = data[0];
            data[0] = data[--size];
            int i = 0;
            while (true) {
                int best = i, left = 2 * i + 1, right = 2 * i + 2;
                if (left < size && better(data[left], data[best])) best = left;
                if (right < size && better(data[right], data[best])) best = right;
                if (best == i) break;
                int t = data[i];
                data[i] = data[best];
                data[best] = t;
                i = best;
            }
            return top;
        }

        int peek() {
            if (size == 0) throw new NoSuchElementException("heap empty");
            return data[0];
        }

        int size() { return size; }
    }

    static class MedianOfStream {
        private final IntHeap low = new IntHeap(true), high = new IntHeap(false);
        private int count = 0;

        void add(int value) {
            count++;
            if (low.size() == 0 || value <= low.peek()) low.push(value);
            else high.push(value);
            if (low.size() > high.size() + 1) high.push(low.pop());
            else if (high.size() > low.size()) low.push(high.pop());
        }

        double median() {
            if (count == 0) throw new IllegalStateException("no values yet");
            if (low.size() == high.size()) return (low.peek() + high.peek()) / 2.0;
            return low.peek();
        }
    }

    public static void main(String[] args) {
        MedianOfStream m = new MedianOfStream();
        int[] values = {1, 2, 3, 4, 5};
        double[] expected = {1.0, 1.5, 2.0, 2.5, 3.0};
        for (int i = 0; i < values.length; i++) {
            m.add(values[i]);
            check(m.median() == expected[i], "stream " + values[i] + " -> " + m.median());
        }

        MedianOfStream m2 = new MedianOfStream();
        m2.add(10);
        check(m2.median() == 10.0, "single");
        m2.add(-5);
        check(m2.median() == 2.5, "two");
        m2.add(-5);
        check(m2.median() == -5.0, "three with duplicate");

        // randomized cross-check against sorting
        Random rng = new Random(5);
        MedianOfStream m3 = new MedianOfStream();
        List<Integer> seen = new ArrayList<>();
        for (int t = 0; t < 200; t++) {
            int x = rng.nextInt(101) - 50;
            seen.add(x);
            m3.add(x);
            Collections.sort(seen);
            int n = seen.size();
            double exp = n % 2 == 1 ? seen.get(n / 2) : (seen.get(n / 2 - 1) + seen.get(n / 2)) / 2.0;
            check(m3.median() == exp, "random " + t);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
