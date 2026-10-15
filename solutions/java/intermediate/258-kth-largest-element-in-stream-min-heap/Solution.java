/** Task 258 - Kth largest element in stream (min-heap). (Tier: Intermediate, CTCI Ch 10) */
import java.util.*;

class KthLargest {
    private final int k;
    private final PriorityQueue<Integer> heap = new PriorityQueue<>();

    /**
     * Keeps only the k largest values seen, so heap.peek() is the k-th largest.
     * add() returns null until at least k values have been seen.
     */
    KthLargest(int k, int[] nums) {
        if (k < 1) throw new IllegalArgumentException("k must be >= 1");
        this.k = k;
        for (int v : nums) heap.add(v);
        while (heap.size() > k) heap.poll();
    }

    Integer add(int val) {
        heap.add(val);
        if (heap.size() > k) heap.poll();
        if (heap.size() < k) return null;
        return heap.peek();
    }
}

public class Solution {
    public static void main(String[] args) {
        KthLargest kl = new KthLargest(3, new int[]{4, 5, 8, 2});
        check(kl.add(3) == 4, "add 3 -> 4");
        check(kl.add(5) == 5, "add 5 -> 5");
        check(kl.add(10) == 5, "add 10 -> 5");
        check(kl.add(9) == 8, "add 9 -> 8");
        check(kl.add(4) == 8, "add 4 -> 8");

        KthLargest grow = new KthLargest(5, new int[]{3, 1});   // k > initial count
        check(grow.add(2) == null, "2 values seen: still waiting");
        check(grow.add(7) == null, "4 values seen: still waiting");
        check(grow.add(9) == 1, "5th largest of {1,2,3,7,9}");
        check(grow.add(4) == 2, "5th largest of {2,3,4,7,9}");

        KthLargest top = new KthLargest(1, new int[]{4, 5, 8, 2});   // k = 1: running max
        check(top.add(3) == 8, "max ignores 3");
        check(top.add(10) == 10, "new max");
        check(top.add(4) == 10, "max stays");

        KthLargest dupes = new KthLargest(2, new int[]{5, 5, 5});
        check(dupes.add(5) == 5, "duplicates all");
        check(dupes.add(4) == 5, "4 is too small");
        check(dupes.add(6) == 5, "{5,5,6}");
        check(dupes.add(7) == 6, "{6,7} keeps 2nd largest");

        KthLargest empty = new KthLargest(2, new int[]{});
        check(empty.add(5) == null, "empty stream: waiting");
        check(empty.add(3) == 3, "{3,5} -> 3");
        check(empty.add(10) == 5, "{5,10} -> 5");
        check(empty.add(8) == 8, "{8,10} -> 8");

        KthLargest start = new KthLargest(2, new int[]{1, 2, 3});   // initial pile trimmed
        check(start.add(0) == 2, "top 2 of {0,2,3} -> 2");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
