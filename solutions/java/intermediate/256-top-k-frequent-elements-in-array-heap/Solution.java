/** Task 256 - Top K frequent elements in array (heap). (Tier: Intermediate, CTCI Ch 10) */
import java.util.*;

public class Solution {
    /**
     * Counts occurrences, keeps the k best in a min-heap and reports them
     * frequency-descending with value-ascending tie-breaks. An entry is "worse" when its
     * count is smaller, or - on equal counts - when its value is larger; the heap root is
     * the worst kept candidate. O(n + d log k) time for d distinct values.
     */
    static List<Integer> topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> freq = new HashMap<>();
        for (int x : nums) freq.merge(x, 1, Integer::sum);
        List<Integer> out = new ArrayList<>();
        if (k <= 0) return out;
        PriorityQueue<int[]> heap =
                new PriorityQueue<>((a, b) -> a[0] != b[0] ? a[0] - b[0] : b[1] - a[1]);
        for (Map.Entry<Integer, Integer> e : freq.entrySet()) {
            int[] entry = {e.getValue(), e.getKey()};
            if (heap.size() < k) {
                heap.add(entry);
            } else if (isBetter(entry, heap.peek())) {
                heap.poll();
                heap.add(entry);
            }
        }
        for (int[] entry : heap) out.add(entry[1]);
        out.sort((a, b) -> freq.get(a).equals(freq.get(b))
                ? a - b : freq.get(b) - freq.get(a));
        return out;
    }

    static boolean isBetter(int[] candidate, int[] worst) {
        if (candidate[0] != worst[0]) return candidate[0] > worst[0];
        return candidate[1] < worst[1];
    }

    public static void main(String[] args) {
        check(topKFrequent(new int[]{1, 1, 1, 2, 2, 3}, 2).toString().equals("[1, 2]"), "classic k=2");
        check(topKFrequent(new int[]{1, 1, 1, 2, 2, 3}, 3).toString().equals("[1, 2, 3]"), "classic k=3");
        check(topKFrequent(new int[]{1, 1, 1, 2, 2, 3}, 6).toString().equals("[1, 2, 3]"), "k >= distinct");
        check(topKFrequent(new int[]{1, 1, 1, 2, 2, 3}, 99).toString().equals("[1, 2, 3]"), "huge k");
        check(topKFrequent(new int[]{1, 1, 1, 2, 2, 3}, 0).isEmpty(), "k=0");
        check(topKFrequent(new int[]{1, 1, 1, 2, 2, 3}, 1).toString().equals("[1]"), "k=1");

        check(topKFrequent(new int[]{4, 4, 5, 5, 6}, 2).toString().equals("[4, 5]"), "count tie");
        check(topKFrequent(new int[]{3, 1, 2}, 2).toString().equals("[1, 2]"), "all count 1");
        check(topKFrequent(new int[]{2, 2, 3, 3, 1}, 1).toString().equals("[2]"), "tie at the edge");
        check(topKFrequent(new int[]{-1, -1, 2, 2}, 2).toString().equals("[-1, 2]"), "negatives");
        check(topKFrequent(new int[]{5}, 1).toString().equals("[5]"), "single element");
        check(topKFrequent(new int[]{}, 3).isEmpty(), "empty array");
        check(topKFrequent(new int[]{7, 7, 7}, 3).toString().equals("[7]"), "one distinct value");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
