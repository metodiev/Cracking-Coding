/** Task 133 - Generate permutations with duplicates (unique). (Tier: Beginner, CTCI Ch 8) */
import java.util.*;

public class Solution {
    static List<List<Integer>> uniquePermutations(int[] items) {
        int[] sorted = items.clone();
        Arrays.sort(sorted);
        List<List<Integer>> out = new ArrayList<>();
        backtrack(sorted, new boolean[sorted.length], new ArrayList<>(), out);
        return out;
    }

    private static void backtrack(int[] items, boolean[] used, List<Integer> current, List<List<Integer>> out) {
        if (current.size() == items.length) {
            out.add(new ArrayList<>(current));
            return;
        }
        for (int i = 0; i < items.length; i++) {
            if (used[i]) continue;
            if (i > 0 && items[i] == items[i - 1] && !used[i - 1]) continue;
            used[i] = true;
            current.add(items[i]);
            backtrack(items, used, current, out);
            current.remove(current.size() - 1);
            used[i] = false;
        }
    }

    public static void main(String[] args) {
        check(uniquePermutations(new int[]{1, 1, 2}).toString().equals("[[1, 1, 2], [1, 2, 1], [2, 1, 1]]"),
                "unique perms of [1,1,2]");
        check(uniquePermutations(new int[]{1, 1, 1}).size() == 1, "all same");
        check(uniquePermutations(new int[]{1, 2, 3}).size() == 6, "distinct case");
        check(uniquePermutations(new int[]{}).size() == 1, "empty");
        check(uniquePermutations(new int[]{1, 2, 2, 3}).size() == 12, "4!/2! = 12");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
