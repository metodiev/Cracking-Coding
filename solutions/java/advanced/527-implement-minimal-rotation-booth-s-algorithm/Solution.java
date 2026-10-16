/** Task 527 - Implement minimal rotation (Booth’s algorithm). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Booth's algorithm finds the lexicographically smallest rotation of a string in O(n) time and O(1)
 * extra space: the two candidate starts i and j are advanced by the length k of the common prefix
 * of their rotations, whichever start loses the comparison at offset k jumps past k, and the
 * smaller of the two surviving starts is returned. When several starts achieve the same minimal
 * rotation the algorithm returns the SMALLEST one, which is documented and verified (periodic
 * inputs such as "abab" have two identical minimal rotations). Two independent reference
 * implementations are used by the checks: a brute-force minimum over all n rotations that also
 * keeps the smallest index (O(n^2) string comparisons) and a cyclic doubling rank implementation
 * that sorts rotations by (rank[i], rank[(i + p) mod n]) pairs until all ranks are distinct
 * (O(n log^2 n)), so the O(n) result is cross-validated twice on different input sizes.
 * minimalRotation(index, s) returns the rotated string itself, and the documented structural
 * property of that string is checked as well: the minimal rotation of any word is a power u^k of a
 * single Lyndon word u (verified with an in-file Duval factorization plus the Lyndon test "smaller
 * than all of its proper suffixes").
 *
 * Checks: exhaustive agreement with the brute force over ALL binary strings of length 1..10 and ALL
 * ternary strings of length 1..7, and over 40 LCG strings of length up to 60 (index and rotation
 * both compared); hand cases "" -> 0, "a" -> 0, "aaaa" -> 0, "aab" -> 0, "baab" -> 1 giving "aabb",
 * "abab" -> 0, "baba" -> 1 giving "abab", "cba" -> 2 giving "acb", "banana" -> 5 giving
 * "abanan" and "mississippi" -> 10 giving "imississipp"; the returned index is the
 * smallest index whose rotation is minimal; a 2000-character LCG string must agree with the
 * doubling reference (and the returned rotation must beat 200 sampled other rotations); the
 * minimal rotation must be a power of a Lyndon word, must equal the minimal rotation of itself
 * (idempotence), and for a Lyndon input the answer must be index 0; repeated runs are identical.
 */
import java.util.*;

public class Solution {
    /** Booth's O(n) minimal-rotation index; ties resolve to the smallest index. */
    static int booth(String s) {
        int n = s.length();
        if (n == 0) return 0;
        String doubled = s + s;
        int i = 0;
        int j = 1;
        int k = 0;
        while (i < n && j < n && k < n) {
            char a = doubled.charAt(i + k);
            char b = doubled.charAt(j + k);
            if (a == b) {
                k++;
                continue;
            }
            if (a > b) {
                i = i + k + 1;
                if (i <= j) i = j + 1;
            } else {
                j = j + k + 1;
                if (j <= i) j = i + 1;
            }
            k = 0;
        }
        return Math.min(i, j);
    }

    static String minimalRotation(String s) {
        int index = booth(s);
        if (s.isEmpty()) return "";
        return s.substring(index) + s.substring(0, index);
    }

    static String rotate(String s, int index) {
        int n = s.length();
        if (n == 0) return "";
        int start = ((index % n) + n) % n;
        return s.substring(start) + s.substring(0, start);
    }

    /** Brute force: smallest index whose rotation is minimal. */
    static int brute(String s) {
        int n = s.length();
        if (n == 0) return 0;
        String best = null;
        int bestIndex = 0;
        for (int start = 0; start < n; start++) {
            String candidate = rotate(s, start);
            if (best == null || candidate.compareTo(best) < 0) {
                best = candidate;
                bestIndex = start;
            }
        }
        return bestIndex;
    }

    /** Independent O(n log^2 n) cyclic doubling reference; ties resolve to the smallest index. */
    static int doubling(String s) {
        int n = s.length();
        if (n == 0) return 0;
        int[] initial = new int[n];
        for (int i = 0; i < n; i++) initial[i] = s.charAt(i);
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        for (int p = 1; p < n; p *= 2) {
            final int step = p;
            final int[] rank = initial;
            Arrays.sort(order, (a, b) -> {
                if (rank[a] != rank[b]) return Integer.compare(rank[a], rank[b]);
                int secondA = rank[(a + step) % n];
                int secondB = rank[(b + step) % n];
                if (secondA != secondB) return Integer.compare(secondA, secondB);
                return Integer.compare(a, b);
            });
            int[] next = new int[n];
            int current = 0;
            next[order[0]] = 0;
            for (int i = 1; i < n; i++) {
                int previous = order[i - 1];
                int index = order[i];
                boolean same = rank[previous] == rank[index]
                        && rank[(previous + step) % n] == rank[(index + step) % n];
                if (!same) current++;
                next[index] = current;
            }
            initial = next;
            if (current == n - 1) break;
        }
        int best = 0;
        for (int i = 1; i < n; i++) {
            if (initial[i] < initial[best]) best = i;
        }
        return best;
    }

    /** Duval factorization into non-increasing Lyndon words. */
    static List<String> duval(String s) {
        List<String> factors = new ArrayList<>();
        int n = s.length();
        int i = 0;
        while (i < n) {
            int j = i + 1;
            int k = i;
            while (j < n && s.charAt(k) <= s.charAt(j)) {
                if (s.charAt(k) < s.charAt(j)) k = i;
                else k++;
                j++;
            }
            while (i <= k) {
                factors.add(s.substring(i, i + j - k));
                i += j - k;
            }
        }
        return factors;
    }

    static boolean isLyndon(String w) {
        int n = w.length();
        if (n == 0) return false;
        for (int i = 1; i < n; i++) {
            if (w.compareTo(w.substring(i)) >= 0) return false;
        }
        return true;
    }

    static String lcgString(int n, int seed, int alphabet) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            builder.append((char) ('a' + seed % alphabet));
        }
        return builder.toString();
    }

    static void checkAllBinary() {
        for (int length = 1; length <= 10; length++) {
            for (int mask = 0; mask < (1 << length); mask++) {
                StringBuilder builder = new StringBuilder();
                for (int bit = 0; bit < length; bit++) {
                    builder.append(((mask >> bit) & 1) == 0 ? 'a' : 'b');
                }
                String s = builder.toString();
                compareWithBrute(s);
            }
        }
    }

    static void checkAllTernary() {
        int[] digits = new int[7];
        for (int length = 1; length <= 7; length++) {
            int total = 1;
            for (int i = 0; i < length; i++) total *= 3;
            for (int code = 0; code < total; code++) {
                int value = code;
                StringBuilder builder = new StringBuilder();
                for (int i = 0; i < length; i++) {
                    digits[i] = value % 3;
                    value /= 3;
                    builder.append((char) ('a' + digits[i]));
                }
                compareWithBrute(builder.toString());
            }
        }
    }

    static void compareWithBrute(String s) {
        int index = booth(s);
        int expected = brute(s);
        check(index == expected, "booth index " + index + " vs brute " + expected + " for " + s);
        check(rotate(s, index).equals(rotate(s, expected)), "rotation for " + s);
        check(doubling(s) == expected, "doubling index for " + s);
    }

    public static void main(String[] args) {
        checkAllBinary();
        checkAllTernary();

        check(booth("") == 0 && minimalRotation("").isEmpty(), "empty string");
        check(booth("a") == 0 && minimalRotation("a").equals("a"), "single character");
        check(booth("aaaa") == 0 && minimalRotation("aaaa").equals("aaaa"), "all equal");
        check(booth("aab") == 0, "already minimal");
        check(booth("baab") == 1 && minimalRotation("baab").equals("aabb"), "baab -> aabb at index 1");
        check(booth("abab") == 0, "periodic input keeps the smallest index");
        check(booth("baba") == 1 && minimalRotation("baba").equals("abab"), "baba -> abab");
        check(booth("cba") == 2 && minimalRotation("cba").equals("acb"), "cba -> acb");
        check(booth("banana") == 5 && minimalRotation("banana").equals("abanan"),
                "banana -> abanan at index 5");
        check(booth("mississippi") == 10 && minimalRotation("mississippi").equals("imississipp"),
                "mississippi rotates to imississipp");

        check(booth("abab") == 0 && rotate("abab", 2).equals(minimalRotation("abab")),
                "both minimal starts give the same rotation");
        List<Integer> minimalStarts = new ArrayList<>();
        for (int start = 0; start < 4; start++) {
            if (rotate("abab", start).equals(minimalRotation("abab"))) minimalStarts.add(start);
        }
        check(minimalStarts.equals(Arrays.asList(0, 2)), "minimal starts are 0 and 2");
        check(Collections.min(minimalStarts) == booth("abab"), "booth returns the smallest start");

        int seed = 3;
        for (int trial = 0; trial < 40; trial++) {
            int n = 1 + trial * 3 % 60;
            int alphabet = 2 + trial % 3;
            String s = lcgString(n, 2000 + trial * 61, alphabet);
            compareWithBrute(s);
            String minimal = minimalRotation(s);
            check(minimal.length() == s.length(), "rotation preserves the length");
            check(minimalRotation(minimal).equals(minimal), "idempotent minimal rotation");
            List<String> factors = duval(minimal);
            check(!factors.isEmpty(), "the minimal rotation factorizes");
            boolean allEqual = true;
            for (String factor : factors) {
                if (!factor.equals(factors.get(0))) allEqual = false;
            }
            check(allEqual, "the minimal rotation is a power of one word for " + s);
            check(isLyndon(factors.get(0)), "that word is Lyndon for " + s);
            check(rotate(s, booth(s)).equals(minimal), "rotate agrees with minimalRotation");
        }

        for (String lyndon : new String[]{"a", "ab", "abb", "aab", "abc", "abac", "aabab"}) {
            check(isLyndon(lyndon), lyndon + " should be Lyndon");
            check(booth(lyndon) == 0, "a Lyndon word is its own minimal rotation: " + lyndon);
            check(minimalRotation(lyndon).equals(lyndon), "minimal rotation of " + lyndon);
            check(duval(lyndon).size() == 1, lyndon + " has a single Duval factor");
        }

        String big = lcgString(2000, 987654, 4);
        int bigIndex = booth(big);
        check(bigIndex == doubling(big), "2000-character string matches the doubling reference");
        String bigMinimal = minimalRotation(big);
        for (int sample = 0; sample < 200; sample++) {
            seed = (seed * 75 + 74) % 65537;
            int other = seed % big.length();
            check(bigMinimal.compareTo(rotate(big, other)) <= 0,
                    "sampled rotation " + other + " is not smaller");
        }
        check(rotate(big, bigIndex).equals(bigMinimal), "big rotation reproduced");
        check(booth(big) == bigIndex && minimalRotation(big).equals(bigMinimal),
                "deterministic re-run on the 2000-character string");
        check(booth("abcdefghij") == 0 && doubling("abcdefghij") == 0, "sorted string is minimal");
        check(booth("jihgfedcba") == 9, "reverse-sorted string picks the last start");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
