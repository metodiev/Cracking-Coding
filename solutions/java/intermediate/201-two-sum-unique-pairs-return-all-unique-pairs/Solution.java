/** Task 201 - Two-sum unique pairs (return all unique pairs). (Tier: Intermediate, CTCI Ch 16) */
import java.util.*;

public class Solution {
    static List<int[]> twoSumUniquePairs(int[] nums, int target) {
        Set<Integer> seen = new HashSet<>();
        Set<List<Integer>> pairs = new TreeSet<>((a, b) -> {
            int c = Integer.compare(a.get(0), b.get(0));
            return c != 0 ? c : Integer.compare(a.get(1), b.get(1));
        });
        for (int x : nums) {
            int y = target - x;
            if (seen.contains(y)) pairs.add(List.of(Math.min(x, y), Math.max(x, y)));
            seen.add(x);
        }
        List<int[]> out = new ArrayList<>();
        for (List<Integer> pair : pairs) out.add(new int[]{pair.get(0), pair.get(1)});
        return out;
    }

    private static String render(List<int[]> pairs) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < pairs.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append("(").append(pairs.get(i)[0]).append(", ").append(pairs.get(i)[1]).append(")");
        }
        return sb.append("]").toString();
    }

    public static void main(String[] args) {
        check(render(twoSumUniquePairs(new int[]{1, 1, 2, 3, 2}, 4)).equals("[(1, 3), (2, 2)]"), "dup-friendly");
        check(render(twoSumUniquePairs(new int[]{1, 5, 3, 3, 5}, 6)).equals("[(1, 5), (3, 3)]"), "second case");
        check(twoSumUniquePairs(new int[]{1, 2, 3}, 100).isEmpty(), "no pairs");
        check(twoSumUniquePairs(new int[0], 5).isEmpty(), "empty");
        check(render(twoSumUniquePairs(new int[]{0, 0, 0}, 0)).equals("[(0, 0)]"), "zeros");
        check(render(twoSumUniquePairs(new int[]{-1, 1, 0}, 0)).equals("[(-1, 1)]"), "negatives");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
