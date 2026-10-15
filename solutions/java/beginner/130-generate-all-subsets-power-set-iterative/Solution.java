/** Task 130 - Generate all subsets (power set) iterative. (Tier: Beginner, CTCI Ch 8) */
import java.util.*;

public class Solution {
    static List<List<Integer>> subsetsBitmask(int[] items) {
        int n = items.length;
        List<List<Integer>> out = new ArrayList<>();
        for (int mask = 0; mask < (1 << n); mask++) {
            List<Integer> subset = new ArrayList<>();
            for (int i = 0; i < n; i++)
                if ((mask & (1 << i)) != 0) subset.add(items[i]);
            out.add(subset);
        }
        return out;
    }

    static List<List<Integer>> subsetsIncremental(int[] items) {
        List<List<Integer>> out = new ArrayList<>();
        out.add(new ArrayList<>());
        for (int x : items) {
            int size = out.size();
            for (int i = 0; i < size; i++) {
                List<Integer> extended = new ArrayList<>(out.get(i));
                extended.add(x);
                out.add(extended);
            }
        }
        return out;
    }

    public static void main(String[] args) {
        List<List<Integer>> mask = subsetsBitmask(new int[]{1, 2, 3});
        check(mask.size() == 8, "eight subsets");
        check(mask.get(0).isEmpty(), "empty first");
        check(mask.get(1).toString().equals("[1]"), "second is [1]");
        check(mask.get(7).toString().equals("[1, 2, 3]"), "full set last");
        check(subsetsBitmask(new int[]{}).size() == 1, "empty input -> one subset");
        check(subsetsIncremental(new int[]{1, 2}).toString().equals("[[], [1], [2], [1, 2]]"), "incremental order");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
