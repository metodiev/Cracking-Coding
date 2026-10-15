/** Task 203 - 4-sum problem (k-sum generalization). (Tier: Intermediate, CTCI Ch 16) */
import java.util.*;

public class Solution {
    static List<List<Integer>> kSum(int[] nums, int target, int k) {
        int[] a = nums.clone();
        Arrays.sort(a);
        List<List<Integer>> results = new ArrayList<>();
        search(a, 0, target, k, new ArrayList<>(), results);
        return results;
    }

    private static void search(int[] a, int start, int remaining, int kLeft,
                               List<Integer> current, List<List<Integer>> results) {
        if (kLeft == 2) {
            int lo = start, hi = a.length - 1;
            while (lo < hi) {
                int total = a[lo] + a[hi];
                if (total == remaining) {
                    List<Integer> found = new ArrayList<>(current);
                    found.add(a[lo]);
                    found.add(a[hi]);
                    results.add(found);
                    while (lo < hi && a[lo] == a[lo + 1]) lo++;
                    while (lo < hi && a[hi] == a[hi - 1]) hi--;
                    lo++;
                    hi--;
                } else if (total < remaining) lo++;
                else hi--;
            }
            return;
        }
        for (int i = start; i + kLeft <= a.length; i++) {
            if (i > start && a[i] == a[i - 1]) continue;
            if ((long) a[i] * kLeft > remaining) break;
            current.add(a[i]);
            search(a, i + 1, remaining - a[i], kLeft - 1, current, results);
            current.remove(current.size() - 1);
        }
    }

    public static void main(String[] args) {
        check(kSum(new int[]{1, 0, -1, 0, -2, 2}, 0, 4).toString()
                .equals("[[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]"), "classic four-sum");
        check(kSum(new int[]{2, 2, 2, 2, 2}, 8, 4).toString().equals("[[2, 2, 2, 2]]"), "all twos");
        check(kSum(new int[]{1, 2, 3, 4}, 100, 4).isEmpty(), "no match");
        check(kSum(new int[]{}, 0, 4).isEmpty(), "empty");
        check(kSum(new int[]{1, 2, 3, 4}, 5, 2).toString().equals("[[1, 4], [2, 3]]"), "k=2");
        check(kSum(new int[]{-1, 0, 1, 2, -1, -4}, 0, 3).toString().equals("[[-1, -1, 2], [-1, 0, 1]]"), "k=3");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
