/** Task 131 - Generate all subsets recursive. (Tier: Beginner, CTCI Ch 8) */
import java.util.*;

public class Solution {
    static List<List<Integer>> subsetsRecursive(int[] items) {
        List<List<Integer>> out = new ArrayList<>();
        backtrack(items, 0, new ArrayList<>(), out);
        return out;
    }

    private static void backtrack(int[] items, int i, List<Integer> current, List<List<Integer>> out) {
        if (i == items.length) {
            out.add(new ArrayList<>(current));
            return;
        }
        backtrack(items, i + 1, current, out);          // skip
        current.add(items[i]);
        backtrack(items, i + 1, current, out);          // take
        current.remove(current.size() - 1);
    }

    public static void main(String[] args) {
        List<List<Integer>> result = subsetsRecursive(new int[]{1, 2, 3});
        check(result.size() == 8, "eight subsets");
        check(result.toString().equals("[[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]"),
                "deterministic skip-first order: " + result);
        check(subsetsRecursive(new int[]{}).size() == 1, "empty input");
        check(subsetsRecursive(new int[]{1, 2, 3, 4}).size() == 16, "2^4");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
