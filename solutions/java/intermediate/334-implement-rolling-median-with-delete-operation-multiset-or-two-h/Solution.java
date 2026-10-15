/** Task 334 - Implement rolling median with delete operation (two heaps + lazy deletion).
 * (Tier: Intermediate, CTCI Ch 7) */
import java.util.*;

class MedianMultiset {
    // `low` keeps the smaller half as a max-heap, `high` the larger half as a min-heap, so
    // len(low) == len(high) or len(low) == len(high) + 1. delete() marks one occurrence as
    // pending in whichever heap holds it; the mark is consumed when the value reaches a top.
    private final PriorityQueue<Integer> low = new PriorityQueue<>(Comparator.reverseOrder());
    private final PriorityQueue<Integer> high = new PriorityQueue<>();
    private final Map<Integer, Integer> pendingLow = new HashMap<>();
    private final Map<Integer, Integer> pendingHigh = new HashMap<>();
    private int pendingLowTotal = 0;
    private int pendingHighTotal = 0;
    private final Map<Integer, Integer> active = new HashMap<>();
    private int count = 0;

    int size() {
        return count;
    }

    void add(int value) {
        cleanLow();
        cleanHigh();
        if (!low.isEmpty() && value <= low.peek()) low.add(value);
        else high.add(value);
        count++;
        active.merge(value, 1, Integer::sum);
        rebalance();
    }

    /** Removes one occurrence of `value`; throws when it is absent. */
    void delete(int value) {
        if (active.getOrDefault(value, 0) < 1) {
            throw new IllegalArgumentException("value not present: " + value);
        }
        active.merge(value, -1, Integer::sum);
        cleanLow();
        if (!low.isEmpty() && value <= low.peek()) {
            pendingLow.merge(value, 1, Integer::sum);
            pendingLowTotal++;
        } else {
            pendingHigh.merge(value, 1, Integer::sum);
            pendingHighTotal++;
        }
        count--;
        cleanLow();
        cleanHigh();
        rebalance();
    }

    /** Median as a Double, or null when the multiset is empty. */
    Double median() {
        cleanLow();
        cleanHigh();
        if (count == 0) return null;
        if (count % 2 == 1) return (double) low.peek();
        return (low.peek() + high.peek()) / 2.0;
    }

    private void cleanLow() {
        while (!low.isEmpty() && pendingLow.getOrDefault(low.peek(), 0) > 0) {
            int value = low.poll();
            pendingLow.merge(value, -1, Integer::sum);
            pendingLowTotal--;
        }
    }

    private void cleanHigh() {
        while (!high.isEmpty() && pendingHigh.getOrDefault(high.peek(), 0) > 0) {
            int value = high.poll();
            pendingHigh.merge(value, -1, Integer::sum);
            pendingHighTotal--;
        }
    }

    private void rebalance() {
        while (true) {
            int lowSize = low.size() - pendingLowTotal;
            int highSize = high.size() - pendingHighTotal;
            if (lowSize > highSize + 1) {
                cleanLow();
                high.add(low.poll());
            } else if (highSize > lowSize) {
                cleanHigh();
                low.add(high.poll());
            } else {
                return;
            }
        }
    }
}

class SortedListOracle {
    final List<Integer> data = new ArrayList<>();

    void add(int value) {
        int index = Collections.binarySearch(data, value);
        if (index < 0) index = -index - 1;
        data.add(index, value);
    }

    void delete(int value) {
        int index = Collections.binarySearch(data, value);
        if (index < 0 || data.get(index) != value) throw new AssertionError("oracle delete");
        data.remove(index);
    }

    Double median() {
        int n = data.size();
        if (n == 0) return null;
        if (n % 2 == 1) return (double) data.get(n / 2);
        return (data.get(n / 2 - 1) + data.get(n / 2)) / 2.0;
    }
}

public class Solution {
    private static int seed = 4242;

    public static void main(String[] args) {
        MedianMultiset m = new MedianMultiset();
        check(m.size() == 0 && m.median() == null, "empty");
        m.add(5);
        check(m.median() == 5.0, "median 5");
        m.add(3);
        check(m.size() == 2 && m.median() == 4.0, "median 4");
        m.add(8);
        check(m.median() == 5.0, "median 5 again");
        m.add(1);
        check(m.median() == 4.0, "median of {1,3,5,8}");
        m.delete(5);
        check(m.size() == 3 && m.median() == 3.0, "median of {1,3,8}");
        m.delete(1);
        check(m.median() == 5.5, "median of {3,8}");
        m.delete(3);
        check(m.median() == 8.0, "median of {8}");
        m.delete(8);
        check(m.size() == 0 && m.median() == null, "back to empty");
        boolean threw = false;
        try {
            m.delete(7);
        } catch (IllegalArgumentException expected) {
            threw = true;
        }
        check(threw, "delete absent raises");

        MedianMultiset dupes = new MedianMultiset();
        for (int i = 0; i < 3; i++) dupes.add(4);
        check(dupes.median() == 4.0, "dupes median");
        dupes.delete(4);
        check(dupes.median() == 4.0, "dupes median after one delete");
        dupes.delete(4);
        check(dupes.median() == 4.0 && dupes.size() == 1, "dupes median after two deletes");
        dupes.delete(4);
        check(dupes.median() == null, "dupes empty");

        MedianMultiset seq = new MedianMultiset();
        for (int value = 1; value <= 10; value++) seq.add(value);
        check(seq.median() == 5.5, "seq median");
        int[][] steps = {{5, 6}, {6, 5}, {10, 4}, {1, 5}, {7, 4}, {2, 6}};
        double[] expected = {6.0, 5.5, 4.0, 5.5, 4.0, 6.0};
        for (int i = 0; i < steps.length; i++) {
            seq.delete(steps[i][0]);
            check(seq.median() == expected[i], "seq step " + i);
        }
        check(seq.size() == 4, "seq size");

        MedianMultiset fast = new MedianMultiset();
        SortedListOracle oracle = new SortedListOracle();
        for (int step = 0; step < 2000; step++) {
            seed = (seed * 75 + 74) % 65537;
            if (!oracle.data.isEmpty() && seed % 10 >= 6) {
                seed = (seed * 75 + 74) % 65537;
                int value = oracle.data.get(seed % oracle.data.size());
                fast.delete(value);
                oracle.delete(value);
            } else {
                seed = (seed * 75 + 74) % 65537;
                int value = 1 + seed % 50;
                fast.add(value);
                oracle.add(value);
            }
            check(fast.size() == oracle.data.size(), "random size " + step);
            seed = (seed * 75 + 74) % 65537;
            if (seed % 5 == 0 || step % 7 == 0) {
                check(fast.median().equals(oracle.median()), "random median " + step);
            }
        }
        while (!oracle.data.isEmpty()) {
            int value = oracle.data.get(0);
            fast.delete(value);
            oracle.delete(value);
            check(fast.median().equals(oracle.median()), "drain median");
        }
        check(fast.median() == null, "drained empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
