/** Task 529 - Implement advanced string factoring (Lyndon factorization Duval). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Duval's algorithm computes the Chen-Fox-Lyndon factorization of a string into non-increasing
 * Lyndon words in O(n) time and O(1) extra space: the three indices i, j, k walk the string keeping
 * the current factor as a prefix of a repetition of the shortest Lyndon word found so far, and each
 * round emits one factor of length j - k. A Lyndon word is defined and tested here as a nonempty
 * word that is strictly smaller than all of its proper suffixes (equivalently, smaller than all of
 * its proper rotations). The file also carries two independent reference implementations and two
 * classical uses of the factorization: a slow O(n^2) greedy that repeatedly peels off the longest
 * Lyndon prefix, and (a) the fact that the LAST factor of the factorization begins at the
 * lexicographically minimal suffix of the string, and (b) the necklace counting formula for the
 * number of Lyndon words of length L over a sigma-letter alphabet, (1/L) * sum_{d | L} mu(d) *
 * sigma^(L/d), with the Mobius function computed here by trial division and the formula checked
 * against exhaustive enumeration. Documented conventions: the empty string factorizes into no
 * factors at all, factors are compared with the natural lexicographic order on lower-case letters,
 * and the emitted factors are non-increasing (w_1 >= w_2 >= ... >= w_k).
 *
 * Checks: hand factorizations "banana" -> b / an / an / a, "ababab" -> ab / ab / ab, "aaaa" -> a x 4,
 * "abcabc" -> abc / abc, "mississippi" -> m / iss / iss / ipp / i, "cba" -> c / b / a, "aabb" ->
 * aabb, "bananas" -> b / ananas, "abacabad" -> abacabad; for ALL binary strings of length up to 12
 * and ALL ternary strings of length up to 8 (and 40 LCG strings) the concatenation must rebuild the
 * string, every factor must pass the Lyndon test, the factors must be non-increasing, the slow
 * greedy must produce the identical factorization, and an exhaustive enumeration of every
 * non-increasing Lyndon factorization must find exactly ONE - the uniqueness theorem; the first
 * factor length must equal the brute-force longest Lyndon prefix length; the last factor must start
 * at the brute-force minimal suffix position; the necklace formula must match exhaustive counting of
 * Lyndon words for sigma = 2 and lengths 1..10 and sigma = 3 and lengths 1..7; repeated runs are
 * identical.
 */
import java.util.*;

public class Solution {
    /** Duval's O(n) Chen-Fox-Lyndon factorization. */
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

    /** Slow reference: peel off the longest Lyndon prefix each round. */
    static List<String> greedy(String s) {
        List<String> factors = new ArrayList<>();
        int position = 0;
        while (position < s.length()) {
            int best = 0;
            for (int end = position + 1; end <= s.length(); end++) {
                if (isLyndon(s.substring(position, end))) best = end - position;
            }
            factors.add(s.substring(position, position + best));
            position += best;
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

    static int longestLyndonPrefix(String s) {
        int best = 0;
        for (int end = 1; end <= s.length(); end++) {
            if (isLyndon(s.substring(0, end))) best = end;
        }
        return best;
    }

    static int minimalSuffixIndex(String s) {
        int best = 0;
        for (int i = 1; i < s.length(); i++) {
            if (s.substring(i).compareTo(s.substring(best)) < 0) best = i;
        }
        return best;
    }

    /** Counts every non-increasing Lyndon factorization by exhaustive backtracking. */
    static long countFactorizations(String s) {
        return countFrom(s, 0, null);
    }

    private static long countFrom(String s, int position, String previous) {
        if (position == s.length()) return 1;
        long total = 0;
        for (int end = position + 1; end <= s.length(); end++) {
            String factor = s.substring(position, end);
            if (!isLyndon(factor)) continue;
            if (previous != null && factor.compareTo(previous) > 0) continue;
            total += countFrom(s, end, factor);
        }
        return total;
    }

    static int mobius(int n) {
        int result = 1;
        int value = n;
        int divisor = 2;
        while (divisor * divisor <= value) {
            if (value % divisor == 0) {
                value /= divisor;
                if (value % divisor == 0) return 0;
                result = -result;
            }
            divisor++;
        }
        if (value > 1) result = -result;
        return result;
    }

    /** Necklace / Lyndon word count: (1/L) sum_{d | L} mu(d) sigma^(L/d). */
    static long lyndonWordCount(int length, int alphabet) {
        long total = 0;
        for (int divisor = 1; divisor <= length; divisor++) {
            if (length % divisor != 0) continue;
            total += mobius(divisor) * power(alphabet, length / divisor);
        }
        return total / length;
    }

    static long power(long base, int exponent) {
        long result = 1;
        for (int i = 0; i < exponent; i++) result *= base;
        return result;
    }

    static long bruteLyndonWordCount(int length, int alphabet) {
        int total = 1;
        for (int i = 0; i < length; i++) total *= alphabet;
        long count = 0;
        char[] word = new char[length];
        for (int code = 0; code < total; code++) {
            int value = code;
            for (int i = 0; i < length; i++) {
                word[i] = (char) ('a' + value % alphabet);
                value /= alphabet;
            }
            if (isLyndon(new String(word))) count++;
        }
        return count;
    }

    static String lcgString(int n, int seed, int alphabet) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            builder.append((char) ('a' + seed % alphabet));
        }
        return builder.toString();
    }

    static void verifyFactorization(String s, String label) {
        List<String> factors = duval(s);
        StringBuilder rebuilt = new StringBuilder();
        for (String factor : factors) {
            rebuilt.append(factor);
            check(isLyndon(factor), label + ": factor " + factor + " must be Lyndon");
        }
        check(rebuilt.toString().equals(s), label + ": factors rebuild the string");
        for (int i = 0; i + 1 < factors.size(); i++) {
            check(factors.get(i).compareTo(factors.get(i + 1)) >= 0,
                    label + ": factors must be non-increasing");
        }
        check(factors.equals(greedy(s)), label + ": slow greedy must agree");
        if (s.length() <= 16) {
            check(countFactorizations(s) == 1, label + ": the factorization must be unique");
        }
        int expectedPrefix = longestLyndonPrefix(s);
        int actualPrefix = factors.isEmpty() ? 0 : factors.get(0).length();
        check(actualPrefix == expectedPrefix, label + ": the first factor is the longest Lyndon prefix");
        if (!s.isEmpty()) {
            int start = 0;
            for (int i = 0; i + 1 < factors.size(); i++) start += factors.get(i).length();
            check(start == minimalSuffixIndex(s),
                    label + ": the last factor starts at the minimal suffix");
            check(factors.get(factors.size() - 1).equals(s.substring(start)),
                    label + ": the last factor is the minimal suffix");
        }
    }

    public static void main(String[] args) {
        verifyFactorization("", "empty");
        check(duval("").isEmpty(), "the empty string has no factors");

        String[][] hand = {
                {"banana", "b|an|an|a"},
                {"ababab", "ab|ab|ab"},
                {"aaaa", "a|a|a|a"},
                {"abcabc", "abc|abc"},
                {"mississippi", "m|iss|iss|ipp|i"},
                {"cba", "c|b|a"},
                {"aabb", "aabb"},
                {"bananas", "b|ananas"},
                {"abacabad", "abacabad"},
                {"zyxw", "z|y|x|w"},
                {"a", "a"},
                {"abac", "abac"},
                {"abaab", "ab|aab"},
        };
        for (String[] entry : hand) {
            String joined = String.join("|", duval(entry[0]));
            check(joined.equals(entry[1]), "hand factorization of " + entry[0] + " was " + joined);
            verifyFactorization(entry[0], entry[0]);
        }

        for (int length = 0; length <= 12; length++) {
            int total = 1 << length;
            for (int mask = 0; mask < total; mask++) {
                StringBuilder builder = new StringBuilder();
                for (int bit = 0; bit < length; bit++) {
                    builder.append(((mask >> bit) & 1) == 0 ? 'a' : 'b');
                }
                verifyFactorization(builder.toString(), "binary " + builder);
            }
        }

        for (int length = 1; length <= 8; length++) {
            int total = 1;
            for (int i = 0; i < length; i++) total *= 3;
            for (int code = 0; code < total; code++) {
                int value = code;
                StringBuilder builder = new StringBuilder();
                for (int i = 0; i < length; i++) {
                    builder.append((char) ('a' + value % 3));
                    value /= 3;
                }
                verifyFactorization(builder.toString(), "ternary " + builder);
            }
        }

        int seed = 4242;
        for (int trial = 0; trial < 40; trial++) {
            int n = 1 + trial * 5 % 80;
            String s = lcgString(n, 3000 + trial * 83, 2 + trial % 2);
            verifyFactorization(s, "LCG " + trial);
            List<String> factors = duval(s);
            int total = 0;
            for (String factor : factors) total += factor.length();
            check(total == s.length(), "factor lengths sum to n");
            check(factors.size() <= s.length(), "at most n factors");
            check(factors.equals(duval(s)), "deterministic re-run");
        }

        long[][] expected = {{2, 1, 2}, {2, 2, 1}, {2, 3, 2}, {2, 4, 3}, {2, 5, 6}, {2, 6, 9},
                {2, 7, 18}, {2, 8, 30}, {3, 1, 3}, {3, 2, 3}, {3, 3, 8}, {3, 4, 18}, {3, 5, 48},
                {3, 6, 116}};
        for (long[] entry : expected) {
            int alphabet = (int) entry[0];
            int length = (int) entry[1];
            check(lyndonWordCount(length, alphabet) == entry[2],
                    "necklace count for sigma = " + alphabet + ", L = " + length);
        }
        for (int length = 1; length <= 10; length++) {
            check(lyndonWordCount(length, 2) == bruteLyndonWordCount(length, 2),
                    "necklace count matches enumeration for sigma = 2, L = " + length);
        }
        for (int length = 1; length <= 7; length++) {
            check(lyndonWordCount(length, 3) == bruteLyndonWordCount(length, 3),
                    "necklace count matches enumeration for sigma = 3, L = " + length);
        }
        check(lyndonWordCount(4, 2) == 3 && lyndonWordCount(6, 2) == 9, "documented Lyndon counts");
        check(mobius(1) == 1 && mobius(2) == -1 && mobius(4) == 0 && mobius(30) == -1,
                "Mobius values");

        String longString = lcgString(400, 20250916, 3);
        List<String> longFactors = duval(longString);
        StringBuilder rebuilt = new StringBuilder();
        for (String factor : longFactors) rebuilt.append(factor);
        check(rebuilt.toString().equals(longString), "400-character string rebuilds");
        check(longFactors.equals(greedy(longString)), "400-character slow greedy agrees");
        check(longestLyndonPrefix(longString) == longFactors.get(0).length(),
                "400-character first factor is the longest Lyndon prefix");
        int longStart = 0;
        for (int i = 0; i + 1 < longFactors.size(); i++) longStart += longFactors.get(i).length();
        check(longStart == minimalSuffixIndex(longString),
                "400-character last factor starts at the minimal suffix");
        check(longString.substring(longStart).equals(longFactors.get(longFactors.size() - 1)),
                "400-character last factor is the minimal suffix");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
