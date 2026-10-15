/** Task 164 - Compute running median using two heaps. (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static class RunningMedian {
        private final PriorityQueue<Integer> low = new PriorityQueue<>(Comparator.reverseOrder());
        private final PriorityQueue<Integer> high = new PriorityQueue<>();

        void add(int value) {
            if (low.isEmpty() || value <= low.peek()) low.add(value);
            else high.add(value);
            if (low.size() > high.size() + 1) high.add(low.poll());
            else if (high.size() > low.size()) low.add(high.poll());
        }

        double median() {
            if (low.isEmpty()) throw new IllegalStateException("no values yet");
            if (low.size() == high.size()) return (low.peek() + high.peek()) / 2.0;
            return low.peek();
        }
    }

    static double[] runningMedians(int[] stream) {
        RunningMedian rm = new RunningMedian();
        double[] out = new double[stream.length];
        for (int i = 0; i < stream.length; i++) {
            rm.add(stream[i]);
            out[i] = rm.median();
        }
        return out;
    }

    static void checkSeq(int[] stream, double[] expected, String label) {
        double[] got = runningMedians(stream);
        check(got.length == expected.length, label + " length");
        for (int i = 0; i < expected.length; i++) check(got[i] == expected[i], label + " at " + i + " -> " + got[i]);
    }

    public static void main(String[] args) {
        checkSeq(new int[]{1, 2, 3}, new double[]{1.0, 1.5, 2.0}, "1,2,3");
        checkSeq(new int[]{5, 15, 1, 3}, new double[]{5.0, 10.0, 5.0, 4.0}, "5,15,1,3");
        checkSeq(new int[]{2, 2, 2}, new double[]{2.0, 2.0, 2.0}, "duplicates");
        checkSeq(new int[]{-1, -2, -3}, new double[]{-1.0, -1.5, -2.0}, "negatives");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
