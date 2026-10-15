/** Task 187 - Implement basic rolling hash for integer arrays. (Tier: Beginner, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

public class Solution {
    static final long DEFAULT_BASE = 257;
    static final long DEFAULT_MOD = 1_000_000_007L;

    static long powMod(long base, long exponent, long mod) {
        long result = 1;
        for (long i = 0; i < exponent; i++) result = result * base % mod;
        return result;
    }

    static long normalize(long value, long mod) {
        return ((value % mod) + mod) % mod;
    }

    /** Comma-joined window contents; used as an exact key when verifying hash matches. */
    static String windowKey(int[] values, int start, int k) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < k; i++) sb.append(values[start + i]).append(',');
        return sb.toString();
    }

    static Map<Integer, List<Integer>> findDuplicateWindows(int[] values, int k) {
        return findDuplicateWindows(values, k, DEFAULT_BASE, DEFAULT_MOD);
    }

    /**
     * Map each repeated window's first start index to its later occurrences. Each length-k
     * window hash slides in O(1); equal hashes are only candidates, so every bucket is
     * confirmed by comparing the windows themselves (a collision cannot fake a duplicate).
     */
    static Map<Integer, List<Integer>> findDuplicateWindows(int[] values, int k, long base, long mod) {
        if (k <= 0) throw new IllegalArgumentException("window length must be positive");
        Map<Integer, List<Integer>> duplicates = new TreeMap<>();
        int n = values.length;
        if (k > n) return duplicates;

        long highest = powMod(base, k - 1, mod);
        long windowHash = 0;
        for (int i = 0; i < k; i++) windowHash = (windowHash * base + normalize(values[i], mod)) % mod;

        Map<Long, List<Integer>> buckets = new HashMap<>();
        buckets.computeIfAbsent(windowHash, h -> new ArrayList<>()).add(0);
        for (int start = 1; start + k <= n; start++) {
            windowHash = (windowHash - normalize(values[start - 1], mod) * highest) % mod;
            if (windowHash < 0) windowHash += mod;
            windowHash = (windowHash * base + normalize(values[start + k - 1], mod)) % mod;
            buckets.computeIfAbsent(windowHash, h -> new ArrayList<>()).add(start);
        }

        for (List<Integer> indices : buckets.values()) {
            if (indices.size() < 2) continue;
            Map<String, List<Integer>> groups = new LinkedHashMap<>();
            for (int index : indices) {
                groups.computeIfAbsent(windowKey(values, index, k), key -> new ArrayList<>()).add(index);
            }
            for (List<Integer> positions : groups.values()) {
                if (positions.size() > 1) {
                    duplicates.put(positions.get(0), new ArrayList<>(positions.subList(1, positions.size())));
                }
            }
        }
        return duplicates;
    }

    /** Reference implementation: group windows directly, no hashing. */
    static Map<Integer, List<Integer>> findDuplicateWindowsNaive(int[] values, int k) {
        Map<Integer, List<Integer>> duplicates = new TreeMap<>();
        if (k <= 0 || k > values.length) return duplicates;
        Map<String, List<Integer>> groups = new LinkedHashMap<>();
        for (int start = 0; start + k <= values.length; start++) {
            groups.computeIfAbsent(windowKey(values, start, k), key -> new ArrayList<>()).add(start);
        }
        for (List<Integer> positions : groups.values()) {
            if (positions.size() > 1) {
                duplicates.put(positions.get(0), new ArrayList<>(positions.subList(1, positions.size())));
            }
        }
        return duplicates;
    }

    /** Rendering such as "0:[3, 6];1:[4];2:[5]" so tests read the whole answer. */
    static String describe(Map<Integer, List<Integer>> duplicates) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<Integer, List<Integer>> entry : duplicates.entrySet()) {
            if (sb.length() > 0) sb.append(';');
            sb.append(entry.getKey()).append(':').append(entry.getValue());
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check(describe(findDuplicateWindows(new int[] {1, 2, 3, 1, 2, 3, 1, 2, 3}, 3)).equals("0:[3, 6];1:[4];2:[5]"),
                "triple repeat");
        check(describe(findDuplicateWindows(new int[] {2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}, 3)).equals("0:[3, 6];1:[4, 7];2:[5, 8]"),
                "three interleaved windows");
        check(describe(findDuplicateWindows(new int[] {5, 5, 5, 5}, 2)).equals("0:[1, 2]"), "all equal");
        check(describe(findDuplicateWindows(new int[] {4, 7, 4, 7, 4}, 2)).equals("0:[2];1:[3]"), "overlaps count");
        check(describe(findDuplicateWindows(new int[] {7, 7, 8, 7, 7}, 1)).equals("0:[1, 3, 4]"), "k = 1");
        check(describe(findDuplicateWindows(new int[] {1, 2, 3, 1, 2, 4}, 3)).isEmpty(), "near-duplicate only");
        check(describe(findDuplicateWindows(new int[] {1, 2, 3, 4}, 4)).isEmpty(), "single window");
        check(describe(findDuplicateWindows(new int[] {1, 2, 3, 4}, 5)).isEmpty(), "k larger than array");
        check(describe(findDuplicateWindows(new int[] {}, 2)).isEmpty(), "empty array");
        check(describe(findDuplicateWindows(new int[] {-5, -5, 3, -5, -5}, 2)).equals("0:[3]"), "negative values");
        boolean threw = false;
        try { findDuplicateWindows(new int[] {1, 2}, 0); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "k = 0 rejected");

        // agreement with the direct grouping on pseudo-random arrays
        Random rng = new Random(23);
        for (int t = 0; t < 200; t++) {
            int[] values = new int[rng.nextInt(31)];
            for (int i = 0; i < values.length; i++) values[i] = rng.nextInt(4);
            int k = 1 + rng.nextInt(4);
            check(describe(findDuplicateWindows(values, k)).equals(describe(findDuplicateWindowsNaive(values, k))),
                    "cross-check k = " + k);
        }

        // a tiny modulus makes collisions common; results must still be exact
        for (int t = 0; t < 200; t++) {
            int[] values = new int[rng.nextInt(31)];
            for (int i = 0; i < values.length; i++) values[i] = rng.nextInt(200);
            int k = 1 + rng.nextInt(3);
            check(describe(findDuplicateWindows(values, k, 257, 97)).equals(describe(findDuplicateWindowsNaive(values, k))),
                    "tiny modulus cross-check k = " + k);
        }

        // concrete collision: the windows (5, 63) and (6, 0) share one hash under mod 97
        check((5L * 257 + 63) % 97 == (6L * 257 + 0) % 97, "collision premise");
        check(describe(findDuplicateWindows(new int[] {5, 63, 4, 6, 0}, 2, 257, 97)).isEmpty(), "collision rejected");
        check(describe(findDuplicateWindows(new int[] {5, 63, 4, 6, 0, 5, 63}, 2, 257, 97)).equals("0:[5]"),
                "collision plus real duplicate");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
