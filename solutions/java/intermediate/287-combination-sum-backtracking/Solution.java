/** Task 287 - Combination sum with repetition via backtracking. (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    private static int[] numbers;
    private static List<List<Integer>> results;
    private static List<Integer> current;

    /**
     * All unique combinations of positive candidates that sum to target, each candidate
     * reusable. Candidates are walked in increasing order and the DFS continues from the
     * same index after taking a value, so every multiset appears exactly once.
     */
    static List<List<Integer>> combinationSum(int[] candidates, int target) {
        numbers = candidates.clone();
        Arrays.sort(numbers);
        results = new ArrayList<>();
        current = new ArrayList<>();
        backtrack(0, target);
        results.sort((a, b) -> {
            int size = Math.min(a.size(), b.size());
            for (int i = 0; i < size; i++) {
                if (!a.get(i).equals(b.get(i))) return Integer.compare(a.get(i), b.get(i));
            }
            return Integer.compare(a.size(), b.size());
        });
        return results;
    }

    private static void backtrack(int start, int remaining) {
        if (remaining == 0) {
            results.add(new ArrayList<>(current));
            return;
        }
        for (int i = start; i < numbers.length; i++) {
            if (numbers[i] > remaining) break;
            current.add(numbers[i]);
            backtrack(i, remaining - numbers[i]);   // same index: repetition allowed
            current.remove(current.size() - 1);
        }
    }

    /** Oracle: choose how many copies of each candidate to take (counts per candidate). */
    static List<List<Integer>> bruteForceCombinations(int[] candidates, int target) {
        int[] sorted = candidates.clone();
        Arrays.sort(sorted);
        List<List<Integer>> found = new ArrayList<>();
        build(sorted, 0, target, new ArrayList<>(), found);
        found.sort(Solution::compareCombinations);
        return found;
    }

    private static void build(int[] sorted, int index, int remaining, List<Integer> chosen,
                              List<List<Integer>> found) {
        if (index == sorted.length) {
            if (remaining == 0) found.add(new ArrayList<>(chosen));
            return;
        }
        int value = sorted[index];
        int copies = 0;
        while (value * copies <= remaining) {
            build(sorted, index + 1, remaining - value * copies, chosen, found);
            copies++;
            if (value * copies <= remaining) chosen.add(value);
        }
        for (int i = 0; i < copies - 1; i++) chosen.remove(chosen.size() - 1);
    }

    private static int compareCombinations(List<Integer> a, List<Integer> b) {
        int size = Math.min(a.size(), b.size());
        for (int i = 0; i < size; i++) {
            if (!a.get(i).equals(b.get(i))) return Integer.compare(a.get(i), b.get(i));
        }
        return Integer.compare(a.size(), b.size());
    }

    private static int seed = 13579;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    static String key(List<List<Integer>> combinations) {
        List<String> parts = new ArrayList<>();
        for (List<Integer> combination : combinations) parts.add(combination.toString());
        Collections.sort(parts);
        return String.join("|", parts);
    }

    public static void main(String[] args) {
        check(combinationSum(new int[]{2, 3, 6, 7}, 7).equals(List.of(List.of(2, 2, 3), List.of(7))), "classic seven");
        check(combinationSum(new int[]{2, 3, 5}, 8)
                .equals(List.of(List.of(2, 2, 2, 2), List.of(2, 3, 3), List.of(3, 5))), "classic eight");
        check(combinationSum(new int[]{7, 2, 3, 6}, 7).equals(List.of(List.of(2, 2, 3), List.of(7))), "unsorted input");

        check(combinationSum(new int[]{2, 3}, 0).equals(List.of(List.of())), "target zero");
        check(combinationSum(new int[]{2, 3}, -1).isEmpty(), "negative target");
        check(combinationSum(new int[]{2, 4}, 7).isEmpty(), "odd target even values");
        check(combinationSum(new int[]{2}, 1).isEmpty(), "below every candidate");
        check(combinationSum(new int[]{5}, 10).equals(List.of(List.of(5, 5))), "repetition");
        check(combinationSum(new int[]{8}, 4).isEmpty(), "single too large");

        for (int trial = 0; trial < 200; trial++) {
            int size = 1 + nextSeed() % 4;
            List<Integer> picked = new ArrayList<>();
            while (picked.size() < size) {
                int value = 1 + nextSeed() % 8;
                if (!picked.contains(value)) picked.add(value);
            }
            int[] candidates = new int[picked.size()];
            for (int i = 0; i < candidates.length; i++) candidates[i] = picked.get(i);
            int target = nextSeed() % 13;

            List<List<Integer>> found = combinationSum(candidates, target);
            check(key(found).equals(key(bruteForceCombinations(candidates, target))), "oracle " + trial);
            Set<String> unique = new HashSet<>();
            for (List<Integer> combination : found) {
                int sum = 0;
                int previous = Integer.MIN_VALUE;
                for (int value : combination) {
                    sum += value;
                    check(picked.contains(value), "value in candidates " + trial);
                    check(value >= previous, "sorted combination " + trial);
                    previous = value;
                }
                check(sum == target, "sum " + trial);
                check(unique.add(combination.toString()), "duplicate " + trial);
            }
        }

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
