/** Task 432 - Implement data structure for dynamic median (order-statistics tree). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Approach: two heaps with lazy deletion. Values up to the median live in a max-heap (low), the
 * rest in a min-heap (high); every live low value is <= every live high value. Deletions do not
 * search the heaps: a per-value pending count marks the removal and is consumed when a marked
 * value surfaces at a heap top (prune), so remove/insert are O(log n) amortised. median() returns
 * the lower median - the live element at 0-based index (size-1)//2 - which is exactly the top of
 * low, so even sizes answer with a stored value instead of a float; the empty multiset answers
 * null. Cross-checked against a sorted-list oracle (binary-search insertion) on LCG operation
 * scripts that insert, delete and re-query, including deletions until the structure empties.
 */
import java.util.*;

class DynamicMedian {
    private final PriorityQueue<Integer> low = new PriorityQueue<>(Collections.reverseOrder());
    private final PriorityQueue<Integer> high = new PriorityQueue<>();
    private final Map<Integer, Integer> pendingLow = new HashMap<>();
    private final Map<Integer, Integer> pendingHigh = new HashMap<>();
    private final Map<Integer, Integer> counts = new HashMap<>();
    private int total = 0;
    private int lowCount = 0;

    int size() {
        return total;
    }

    Integer median() {
        if (total == 0) return null;
        prune();
        return low.peek();
    }

    void insert(int value) {
        prune();
        counts.merge(value, 1, Integer::sum);
        if (!low.isEmpty() && value > low.peek()) {
            high.add(value);
        } else {
            low.add(value);
            lowCount++;
        }
        total++;
        rebalance();
    }

    boolean remove(int value) {
        Integer live = counts.get(value);
        if (live == null) return false;
        if (live == 1) counts.remove(value);
        else counts.put(value, live - 1);
        prune();
        if (!low.isEmpty() && value <= low.peek()) {
            pendingLow.merge(value, 1, Integer::sum);
            lowCount--;
        } else {
            pendingHigh.merge(value, 1, Integer::sum);
        }
        total--;
        rebalance();
        return true;
    }

    private void prune() {
        while (!low.isEmpty() && pendingLow.getOrDefault(low.peek(), 0) > 0) {
            int top = low.poll();
            pendingLow.merge(top, -1, Integer::sum);
        }
        while (!high.isEmpty() && pendingHigh.getOrDefault(high.peek(), 0) > 0) {
            int top = high.poll();
            pendingHigh.merge(top, -1, Integer::sum);
        }
    }

    private void rebalance() {
        int target = (total + 1) / 2;
        while (true) {
            prune();
            if (lowCount > target) {
                int top = low.poll();
                lowCount--;
                high.add(top);
                continue;
            }
            if (lowCount < target) {
                int top = high.poll();
                lowCount++;
                low.add(top);
                continue;
            }
            break;
        }
    }
}

public class Solution {
    static Integer oracleMedian(List<Integer> sortedValues) {
        if (sortedValues.isEmpty()) return null;
        return sortedValues.get((sortedValues.size() - 1) / 2);
    }

    static void sortedInsert(List<Integer> sortedValues, int value) {
        int index = Collections.binarySearch(sortedValues, value);
        if (index < 0) index = -(index + 1);
        sortedValues.add(index, value);
    }

    public static void main(String[] args) {
        DynamicMedian tracker = new DynamicMedian();
        check(tracker.median() == null, "empty median");
        check(!tracker.remove(0), "remove from empty");

        int[] inserts = {5, 3, 8, 9};
        int[] expected = {5, 3, 5, 5};
        for (int i = 0; i < inserts.length; i++) {
            tracker.insert(inserts[i]);
            check(tracker.median() == expected[i], "hand median");
        }
        check(tracker.size() == 4, "hand size");
        check(tracker.remove(5), "remove 5");
        check(tracker.median() == 8, "median after removing 5");
        check(tracker.remove(3), "remove 3");
        check(tracker.median() == 8, "median after removing 3");
        check(tracker.remove(9), "remove 9");
        check(tracker.median() == 8, "median after removing 9");
        check(tracker.remove(8), "remove 8");
        check(tracker.median() == null, "empty again");
        check(tracker.size() == 0, "size back to 0");
        check(!tracker.remove(8), "remove missing");
        tracker.insert(7);
        check(tracker.median() == 7, "reinsert after empty");
        check(tracker.remove(7), "remove last");
        check(tracker.median() == null, "empty after last");

        for (int i = 0; i < 3; i++) {
            tracker.insert(3);
            check(tracker.median() == 3, "duplicates median");
        }
        check(tracker.remove(3) && tracker.remove(3), "duplicate removal");
        check(tracker.median() == 3, "one duplicate left");
        check(tracker.remove(3), "remove last duplicate");
        check(tracker.median() == null, "duplicates drained");

        tracker = new DynamicMedian();
        List<Integer> oracle = new ArrayList<>();
        for (int value = 1; value <= 10; value++) {
            tracker.insert(value);
            sortedInsert(oracle, value);
        }
        int[] removalOrder = {10, 1, 9, 2, 8, 3, 7, 4, 6, 5};
        for (int value : removalOrder) {
            check(tracker.remove(value), "scripted removal");
            oracle.remove(Integer.valueOf(value));
            check(Objects.equals(tracker.median(), oracleMedian(oracle)), "median while draining");
        }
        check(tracker.median() == null, "drained to empty");

        tracker = new DynamicMedian();
        oracle = new ArrayList<>();
        int seed = 4320;
        for (int step = 0; step < 400; step++) {
            seed = (seed * 75 + 74) % 65537;
            int op = seed % 4;
            if (op <= 1 || oracle.isEmpty()) {
                seed = (seed * 75 + 74) % 65537;
                int value = seed % 40 - 20;
                tracker.insert(value);
                sortedInsert(oracle, value);
            } else if (op == 2) {
                seed = (seed * 75 + 74) % 65537;
                int value = oracle.remove(seed % oracle.size());
                check(tracker.remove(value), "random removal");
            } else {
                seed = (seed * 75 + 74) % 65537;
                int missing = 100 + seed % 10;
                check(!tracker.remove(missing), "random absent removal");
            }
            check(Objects.equals(tracker.median(), oracleMedian(oracle)), "random median step " + step);
            check(tracker.size() == oracle.size(), "random size step " + step);
        }
        check(tracker.size() > 0, "workload left values");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
