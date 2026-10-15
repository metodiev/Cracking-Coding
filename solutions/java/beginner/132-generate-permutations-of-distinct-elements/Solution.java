/** Task 132 - Generate permutations of distinct elements. (Tier: Beginner, CTCI Ch 8) */
import java.util.*;

public class Solution {
    static List<List<Integer>> permutations(int[] items) {
        List<List<Integer>> out = new ArrayList<>();
        backtrack(items, new boolean[items.length], new ArrayList<>(), out);
        return out;
    }

    private static void backtrack(int[] items, boolean[] used, List<Integer> current, List<List<Integer>> out) {
        if (current.size() == items.length) {
            out.add(new ArrayList<>(current));
            return;
        }
        for (int i = 0; i < items.length; i++) {
            if (used[i]) continue;
            used[i] = true;
            current.add(items[i]);
            backtrack(items, used, current, out);
            current.remove(current.size() - 1);
            used[i] = false;
        }
    }

    public static void main(String[] args) {
        List<List<Integer>> perms = permutations(new int[]{1, 2, 3});
        check(perms.toString().equals("[[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]"),
                "deterministic order: " + perms);
        check(permutations(new int[]{}).size() == 1, "empty");
        check(permutations(new int[]{7}).size() == 1, "single");
        check(permutations(new int[]{1, 2, 3, 4}).size() == 24, "4! = 24");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
