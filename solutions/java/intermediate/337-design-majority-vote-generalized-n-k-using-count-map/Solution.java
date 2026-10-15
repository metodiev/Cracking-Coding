/** Task 337 - Design majority vote generalized (> n/k) using count map. (Tier: Intermediate, CTCI Ch 7) */
import java.util.*;

public class Solution {
    /**
     * Generalised Boyer-Moore: keep at most k - 1 candidate counters; a new value with no
     * free slot cancels one vote from every counter. A verification pass then keeps exactly
     * the values with count * k > n.
     */
    static int[] majorityElements(int[] values, int k) {
        if (k < 2) throw new IllegalArgumentException("k must be >= 2");
        int n = values.length;
        if (n == 0) return new int[0];

        Map<Integer, Integer> slots = new HashMap<>();
        for (int value : values) {
            if (slots.containsKey(value)) {
                slots.merge(value, 1, Integer::sum);
            } else if (slots.size() < k - 1) {
                slots.put(value, 1);
            } else {
                for (Integer candidate : new ArrayList<>(slots.keySet())) {
                    slots.merge(candidate, -1, Integer::sum);
                    if (slots.get(candidate) == 0) slots.remove(candidate);
                }
            }
        }

        Map<Integer, Integer> counts = new HashMap<>();
        for (int value : values) counts.merge(value, 1, Integer::sum);
        List<Integer> found = new ArrayList<>();
        for (int value : slots.keySet()) {
            if (counts.get(value) * k > n) found.add(value);
        }
        Collections.sort(found);
        int[] result = new int[found.size()];
        for (int i = 0; i < found.size(); i++) result[i] = found.get(i);
        return result;
    }

    static int[] bruteMajority(int[] values, int k) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (int value : values) counts.merge(value, 1, Integer::sum);
        List<Integer> found = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
            if (entry.getValue() * k > values.length) found.add(entry.getKey());
        }
        Collections.sort(found);
        int[] result = new int[found.size()];
        for (int i = 0; i < found.size(); i++) result[i] = found.get(i);
        return result;
    }

    private static int seed = 6060;

    private static String join(int[] values) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) builder.append(',');
            builder.append(values[i]);
        }
        return builder.toString();
    }

    public static void main(String[] args) {
        // Hand-checked with n = 6, k = 4: the bar is count * 4 > 6, so 1 (3 times) and
        // 2 (2 times) qualify while 3 (once) does not.
        check(join(majorityElements(new int[]{1, 1, 1, 2, 2, 3}, 4)).equals("1,2"), "hand n=6 k=4");
        check(join(bruteMajority(new int[]{1, 1, 1, 2, 2, 3}, 4)).equals("1,2"), "hand brute");

        check(join(majorityElements(new int[]{3, 2, 3}, 2)).equals("3"), "classic majority");
        check(join(majorityElements(new int[]{2, 2, 1, 1, 1, 2, 2}, 2)).equals("2"), "classic seven");

        check(majorityElements(new int[]{1, 2, 3, 4}, 4).length == 0, "no majority k=4");
        check(majorityElements(new int[]{1, 2, 3, 4}, 2).length == 0, "no majority k=2");
        check(majorityElements(new int[]{1, 2, 2, 3, 3, 4}, 3).length == 0, "no majority n=6 k=3");
        check(majorityElements(new int[]{}, 3).length == 0, "empty");

        check(majorityElements(new int[]{5, 5, 6, 6}, 2).length == 0, "boundary count*k == n");
        check(join(majorityElements(new int[]{5, 5, 6, 6}, 3)).equals("5,6"), "boundary with k=3");
        check(join(majorityElements(new int[]{7, 8, 9, 7, 8, 9}, 4)).equals("7,8,9"), "all pass");

        for (int trial = 0; trial < 400; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int n = seed % 60;
            seed = (seed * 75 + 74) % 65537;
            int k = 2 + seed % 5;
            seed = (seed * 75 + 74) % 65537;
            int alphabet = 1 + seed % 9;
            int[] values = new int[n];
            for (int i = 0; i < n; i++) {
                seed = (seed * 75 + 74) % 65537;
                values[i] = seed % alphabet;
            }
            int[] found = majorityElements(values, k);
            check(join(found).equals(join(bruteMajority(values, k))), "random set " + trial);
            check(found.length <= k - 1, "bounded candidates " + trial);
            Map<Integer, Integer> counts = new HashMap<>();
            for (int value : values) counts.merge(value, 1, Integer::sum);
            for (int value : found) {
                check(counts.get(value) * k > n, "every report exceeds n/k " + trial);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
