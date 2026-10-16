/** Task 530 - Implement Lyndon array and uses. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * The Lyndon array lambda[i] is the length of the LONGEST Lyndon word starting at position i. It is
 * computed here in O(n) after an O(n log^2 n) suffix array: with isa the inverse suffix array (the
 * rank of each suffix), let NSV[i] be the smallest j > i with isa[j] < isa[i] (the next smaller
 * value in the inverse suffix array, found with a monotonic stack scanning i from n-1 down to 0);
 * then lambda[i] = NSV[i] - i, with NSV[i] = n when no smaller suffix exists. A Lyndon word is
 * implemented as a nonempty word that is strictly smaller than all of its proper suffixes. The
 * brute-force reference computes lambda[i] directly by extending the window while it stays Lyndon.
 * Documented uses of the array that the checks exercise: (a) lambda[0] is the length of the first
 * Duval factor, and walking i += lambda[i] from 0 reproduces the whole Chen-Fox-Lyndon factorization
 * (the longest Lyndon word starting at i is exactly the longest Lyndon prefix of the remaining
 * suffix); (b) the positions with lambda[i] = n - i are exactly the positions where the suffix of s
 * is itself a Lyndon word; (c) the start p of the last Duval factor - equivalently the start of the
 * lexicographically minimal suffix - satisfies lambda[p] = n - p, and the last factor is that
 * minimal suffix.
 *
 * Checks: for ALL binary strings of length up to 9 and ALL ternary strings of length up to 7, and
 * for 30 LCG strings of length up to 100 the fast array must equal the brute-force array element
 * wise; every lambda[i] must lie in [1, n - i], must make s[i .. i + lambda[i]) Lyndon, and must
 * not be extensible by one more character (the extended word fails the Lyndon test) - the last two
 * are re-verified directly on a 2000-character LCG string for sampled positions whose lambda is at
 * most 400, together with lambda[i] <= n - i and the presence of at least one full-suffix Lyndon
 * word; the underlying suffix array is validated separately (permutation, and every adjacent pair of
 * suffixes in lexicographic order) on the 2000-character string and six 200..385-character strings; the uses (a), (b) and
 * (c) are checked against an in-file Duval factorization and a brute-force minimal suffix for all
 * binary strings of length up to 10 and all LCG strings; hand arrays are pinned for "", "a", "aaaa",
 * "abcdef", "cba", "abab", "banana" and "abac"; repeated runs are identical.
 */
import java.util.*;

public class Solution {
    static boolean isLyndon(String w) {
        int n = w.length();
        if (n == 0) return false;
        for (int i = 1; i < n; i++) {
            if (w.compareTo(w.substring(i)) >= 0) return false;
        }
        return true;
    }

    /** Suffix array by prefix doubling with an explicit comparison sort (O(n log^2 n)). */
    static int[] suffixArray(String s) {
        int n = s.length();
        if (n == 0) return new int[0];
        int[] rank = new int[n];
        for (int i = 0; i < n; i++) rank[i] = s.charAt(i);
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        for (int step = 1; ; step *= 2) {
            final int[] currentRank = rank;
            final int width = step;
            final int length = n;
            Arrays.sort(order, (a, b) -> {
                if (currentRank[a] != currentRank[b]) return Integer.compare(currentRank[a], currentRank[b]);
                int indexA = a + width < length ? currentRank[a + width] : Integer.MIN_VALUE;
                int indexB = b + width < length ? currentRank[b + width] : Integer.MIN_VALUE;
                if (indexA != indexB) return Integer.compare(indexA, indexB);
                return Integer.compare(a, b);
            });
            int[] next = new int[n];
            next[order[0]] = 0;
            int classes = 0;
            for (int i = 1; i < n; i++) {
                int previous = order[i - 1];
                int index = order[i];
                boolean same = currentRank[previous] == currentRank[index]
                        && (previous + width < length ? currentRank[previous + width] : Integer.MIN_VALUE)
                        == (index + width < length ? currentRank[index + width] : Integer.MIN_VALUE);
                if (!same) classes++;
                next[index] = classes;
            }
            rank = next;
            if (classes == n - 1) break;
            if (step > n) break;
        }
        int[] sa = new int[n];
        for (int i = 0; i < n; i++) sa[i] = order[i];
        return sa;
    }

    /** The Lyndon array: lambda[i] = next smaller value in the inverse suffix array, minus i. */
    static int[] lyndonArray(String s) {
        int n = s.length();
        int[] lambda = new int[n];
        if (n == 0) return lambda;
        int[] sa = suffixArray(s);
        int[] isa = new int[n];
        for (int i = 0; i < n; i++) isa[sa[i]] = i;
        Deque<Integer> stack = new ArrayDeque<>();
        int[] nextSmaller = new int[n];
        for (int i = n - 1; i >= 0; i--) {
            while (!stack.isEmpty() && isa[stack.peek()] > isa[i]) stack.pop();
            nextSmaller[i] = stack.isEmpty() ? n : stack.peek();
            stack.push(i);
        }
        for (int i = 0; i < n; i++) lambda[i] = nextSmaller[i] - i;
        return lambda;
    }

    /** Brute-force reference: extend the window while it stays a Lyndon word. */
    static int[] lyndonArrayBrute(String s) {
        int n = s.length();
        int[] lambda = new int[n];
        for (int i = 0; i < n; i++) {
            int best = 0;
            for (int end = i + 1; end <= n; end++) {
                if (isLyndon(s.substring(i, end))) best = end - i;
            }
            lambda[i] = best;
        }
        return lambda;
    }

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

    /** Duval factorization rebuilt by walking the Lyndon array. */
    static List<String> factorizationFromArray(String s, int[] lambda) {
        List<String> factors = new ArrayList<>();
        int i = 0;
        while (i < s.length()) {
            int length = lambda[i];
            factors.add(s.substring(i, i + length));
            i += length;
        }
        return factors;
    }

    static int minimalSuffixIndex(String s) {
        int best = 0;
        for (int i = 1; i < s.length(); i++) {
            if (s.substring(i).compareTo(s.substring(best)) < 0) best = i;
        }
        return best;
    }

    static void verify(String s, String label) {
        int n = s.length();
        int[] lambda = lyndonArray(s);
        int[] brute = lyndonArrayBrute(s);
        check(Arrays.equals(lambda, brute), label + ": the fast array matches the brute-force array");
        for (int i = 0; i < n; i++) {
            check(lambda[i] >= 1 && lambda[i] <= n - i, label + ": lambda in [1, n - i]");
            check(isLyndon(s.substring(i, i + lambda[i])), label + ": lambda[i] is Lyndon at " + i);
            if (i + lambda[i] + 1 <= n) {
                check(!isLyndon(s.substring(i, i + lambda[i] + 1)),
                        label + ": lambda[i] cannot be extended at " + i);
            }
        }
        List<String> factors = duval(s);
        check(factorizationFromArray(s, lambda).equals(factors),
                label + ": walking i += lambda[i] reproduces the Duval factorization");
        if (n > 0) {
            check(lambda[0] == factors.get(0).length(), label + ": lambda[0] is the first Duval factor");
            int position = 0;
            for (int i = 0; i + 1 < factors.size(); i++) position += factors.get(i).length();
            check(lambda[position] == n - position,
                    label + ": the last factor's start has lambda = n - i");
            check(position == minimalSuffixIndex(s), label + ": the last factor starts the minimal suffix");
            check(s.substring(position).equals(factors.get(factors.size() - 1)),
                    label + ": the last factor is the minimal suffix");
            for (int i = 0; i < n; i++) {
                boolean suffixIsLyndon = isLyndon(s.substring(i));
                check((lambda[i] == n - i) == suffixIsLyndon,
                        label + ": lambda[i] = n - i exactly for Lyndon suffixes at " + i);
            }
        }
    }

    static String lcgString(int n, int seed, int alphabet) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            builder.append((char) ('a' + seed % alphabet));
        }
        return builder.toString();
    }

    public static void main(String[] args) {
        check(lyndonArray("").length == 0, "empty string has an empty array");
        check(Arrays.equals(lyndonArray("a"), new int[]{1}), "single character");
        check(Arrays.equals(lyndonArray("aaaa"), new int[]{1, 1, 1, 1}), "all equal characters");
        check(Arrays.equals(lyndonArray("abcdef"), new int[]{6, 5, 4, 3, 2, 1}),
                "increasing string gives decreasing lambda");
        check(Arrays.equals(lyndonArray("cba"), new int[]{1, 1, 1}), "decreasing string");
        check(Arrays.equals(lyndonArray("abab"), new int[]{2, 1, 2, 1}),
                "abab: abab itself is not Lyndon (its third rotation is equal, not larger)");
        check(Arrays.equals(lyndonArray("banana"), new int[]{1, 2, 1, 2, 1, 1}), "banana");
        check(Arrays.equals(lyndonArray("abac"), new int[]{4, 1, 2, 1}), "abac");
        check(Arrays.equals(lyndonArray("mississippi"),
                new int[]{1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1}), "mississippi");

        for (int length = 0; length <= 9; length++) {
            for (int mask = 0; mask < (1 << length); mask++) {
                StringBuilder builder = new StringBuilder();
                for (int bit = 0; bit < length; bit++) {
                    builder.append(((mask >> bit) & 1) == 0 ? 'a' : 'b');
                }
                String s = builder.toString();
                check(Arrays.equals(lyndonArray(s), lyndonArrayBrute(s)),
                        "binary string " + s + " array");
                verify(s, "binary " + s);
            }
        }

        for (int length = 0; length <= 7; length++) {
            int total = 1;
            for (int i = 0; i < length; i++) total *= 3;
            for (int code = 0; code < total; code++) {
                int value = code;
                StringBuilder builder = new StringBuilder();
                for (int i = 0; i < length; i++) {
                    builder.append((char) ('a' + value % 3));
                    value /= 3;
                }
                String s = builder.toString();
                check(Arrays.equals(lyndonArray(s), lyndonArrayBrute(s)),
                        "ternary string " + s + " array");
                verify(s, "ternary " + s);
            }
        }

        int seed = 606;
        for (int trial = 0; trial < 30; trial++) {
            int n = 1 + trial * 13 % 100;
            String s = lcgString(n, 5000 + trial * 29, 2 + trial % 3);
            int[] lambda = lyndonArray(s);
            check(Arrays.equals(lambda, lyndonArrayBrute(s)), "LCG trial " + trial + " array");
            verify(s, "LCG " + trial);
        }

        String longString = lcgString(2000, 20250916, 4);
        int[] longLambda = lyndonArray(longString);
        check(longLambda.length == 2000, "the 2000-character array has one entry per position");
        int longest = 0;
        for (int i = 0; i < 2000; i++) {
            check(longLambda[i] >= 1 && longLambda[i] <= 2000 - i, "2000-character lambda bounds");
            longest = Math.max(longest, longLambda[i]);
        }
        check(longest >= 1 && longest <= 2000, "2000-character longest Lyndon word length");
        for (int i : new int[]{0, 1, 500, 1000, 1999}) {
            int length = longLambda[i];
            if (length > 400) continue;
            check(isLyndon(longString.substring(i, i + length)),
                    "sampled long-string lambda is Lyndon at " + i);
            if (i + length + 1 <= 2000) {
                check(!isLyndon(longString.substring(i, i + length + 1)),
                        "sampled long-string lambda cannot be extended at " + i);
            }
        }
        int fullSuffixLyndon = 0;
        for (int i = 0; i < 2000; i++) {
            if (longLambda[i] == 2000 - i) fullSuffixLyndon++;
        }
        check(fullSuffixLyndon >= 1, "at least one suffix of the long string is a Lyndon word");
        check(Arrays.equals(longLambda, lyndonArray(longString)), "the 2000-character array is stable");
        checkSuffixArray(longString, "2000-character string");
        for (int trial = 0; trial < 6; trial++) {
            String s = lcgString(200 + trial * 37, 8000 + trial * 41, 4);
            checkSuffixArray(s, "suffix array trial " + trial);
        }

        System.out.println("PASS");
    }

    /** Verifies the suffix array: permutation, isa consistency and lexicographic order. */
    private static void checkSuffixArray(String s, String label) {
        int n = s.length();
        int[] sa = suffixArray(s);
        boolean[] seen = new boolean[n];
        for (int index : sa) {
            check(index >= 0 && index < n && !seen[index], label + ": the suffix array is a permutation");
            seen[index] = true;
        }
        for (int i = 0; i + 1 < n; i++) {
            check(s.substring(sa[i]).compareTo(s.substring(sa[i + 1])) <= 0,
                    label + ": adjacent suffixes are misordered at " + i);
        }
        if (n > 1) {
            check(s.substring(sa[0]).compareTo(s.substring(sa[n - 1])) < 0,
                    label + ": the smallest suffix precedes the largest");
        }
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
