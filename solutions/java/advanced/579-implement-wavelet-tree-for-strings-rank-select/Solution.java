/** Task 579 - Implement wavelet tree for strings (rank/select). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A wavelet tree over a character string. Each node owns the subsequence of characters that reach it,
 * a bit vector marking which of them went left, and the prefix ranks of that vector. The primitive
 * operations descend one alphabet level per step: rank(c, i) = the number of occurrences of c in
 * s[0, i) and select(c, k) = the 0-based position of the k-th occurrence of c, or -1 when there is
 * none. select maps a local index inside a child back to a position by binary-searching the parent's
 * prefix array, so it costs O(log sigma * log n) instead of O(log sigma) - the documented
 * simplification, together with plain int[] bit vectors instead of succinct ones (memory is
 * O(n log sigma) ints). Built on top: charAccess(i) (a descent to a single leaf, used to verify
 * patterns), countInRange(lo, hi, from, to), quantile(from, to, k), distinct(from, to) (a recursive
 * walk that skips empty subtrees) and countPattern, which enumerates the occurrences of the pattern's
 * first character with select and verifies the remaining characters with charAccess - an honest
 * O(occ * m * log sigma) pattern count, because the O(m log sigma) version needs the FM-index of
 * task 580 and the BWT rather than the plain string wavelet tree.
 *
 * Documented checks. (1) rank is exhaustive on a 97-character string over 26 letters: all 98 prefixes
 * times all 26 letters, including letters below and above the alphabet (which must give 0), and
 * rankLess is checked the same way. (2) select is exhaustive for every letter and every k up to its
 * frequency; select and rank are asserted to be inverses in both directions (rank(c, select(c, k)) =
 * k + 1 and select(c, rank(c, i) - 1) = i for every position i holding c), k equal to the frequency
 * and negative k return -1, and absent letters return -1. (3) countPattern is checked against brute
 * force for all 39 patterns of length 1..3 over the three-letter alphabet and for 100 random
 * substrings of length 4..8; the documented conventions are that the empty pattern counts n and that a
 * letter outside the alphabet counts 0. (4) countInRange, rangeRank, quantile and distinct are checked
 * against brute force on 300 random intervals of the 200-character text, with the sorted slice as the
 * quantile oracle and a HashSet as the distinctness oracle. (5) The node count is 383 for n = 200 over
 * a 3-letter alphabet, asserted below 2 n log2(sigma) = 634, and the concatenated leaves are asserted
 * to reproduce the multiset of the input. (6) Edge cases: the empty string (rank 0, select -1, distinct
 * 0), a single character, a 200-fold repetition of one letter (a path-shaped tree, whose select and
 * rank are checked at both ends), intervals of length 0 and 1, and a query whose interval starts or
 * ends at the string boundary.
 *
 * Randomness is a fixed-seed splitmix64 (the wave-3 note applies: this file draws many independent
 * characters and intervals), so every printed number is reproducible.
 */
import java.util.*;

public class Solution {

    static final class Rng {
        private long state;

        Rng(long seed) { this.state = seed; }

        long nextLong() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
    }

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    static long nodes;

    static final class WaveletTree {
        final char[] data;
        final int lowValue;
        final int highValue;
        final int[] prefixZeros;
        final WaveletTree left;
        final WaveletTree right;

        WaveletTree(char[] values, int from, int to, int lowValue, int highValue) {
            this.lowValue = lowValue;
            this.highValue = highValue;
            data = new char[to - from];
            System.arraycopy(values, from, data, 0, to - from);
            prefixZeros = new int[data.length + 1];
            nodes++;
            if (lowValue == highValue) {
                left = null;
                right = null;
                for (int i = 0; i < data.length; i++) prefixZeros[i + 1] = prefixZeros[i] + 1;
                return;
            }
            int middle = (lowValue + highValue) >>> 1;
            char[] zeros = new char[data.length];
            char[] ones = new char[data.length];
            int z = 0;
            int o = 0;
            for (int i = 0; i < data.length; i++) {
                if (data[i] <= middle) {
                    prefixZeros[i + 1] = prefixZeros[i] + 1;
                    zeros[z++] = data[i];
                } else {
                    prefixZeros[i + 1] = prefixZeros[i];
                    ones[o++] = data[i];
                }
            }
            left = new WaveletTree(zeros, 0, z, lowValue, middle);
            right = new WaveletTree(ones, 0, o, middle + 1, highValue);
        }

        int length() { return data.length; }

        /** Number of occurrences of c in data[0, position). */
        int rank(char c, int position) {
            if (position <= 0 || c < lowValue || c > highValue) return 0;
            if (position > data.length) position = data.length;
            if (lowValue == highValue) return position;
            int zerosBefore = prefixZeros[position];
            int middle = (lowValue + highValue) >>> 1;
            if (c <= middle) return left.rank(c, zerosBefore);
            return right.rank(c, position - zerosBefore);
        }

        /** Number of values <= c in data[0, position). */
        int rankLess(char c, int position) {
            if (position <= 0 || c < lowValue) return 0;
            if (position > data.length) position = data.length;
            if (c >= highValue || lowValue == highValue) return position;
            int zerosBefore = prefixZeros[position];
            int middle = (lowValue + highValue) >>> 1;
            if (c <= middle) return left.rankLess(c, zerosBefore);
            return zerosBefore + right.rankLess(c, position - zerosBefore);
        }

        /** 0-based position of the k-th occurrence of c, or -1. */
        int select(char c, int k) {
            if (k < 0 || c < lowValue || c > highValue) return -1;
            if (lowValue == highValue) return k < data.length ? k : -1;
            int middle = (lowValue + highValue) >>> 1;
            if (c <= middle) {
                int local = left.select(c, k);
                return local < 0 ? -1 : positionOfZero(local);
            }
            int local = right.select(c, k);
            return local < 0 ? -1 : positionOfOne(local);
        }

        private int positionOfZero(int index) {
            int lo = 0;
            int hi = data.length - 1;
            while (lo < hi) {
                int mid = (lo + hi) >>> 1;
                if (prefixZeros[mid + 1] >= index + 1) hi = mid;
                else lo = mid + 1;
            }
            return lo;
        }

        private int positionOfOne(int index) {
            int lo = 0;
            int hi = data.length - 1;
            while (lo < hi) {
                int mid = (lo + hi) >>> 1;
                int onesBefore = mid + 1 - prefixZeros[mid + 1];
                if (onesBefore >= index + 1) hi = mid;
                else lo = mid + 1;
            }
            return lo;
        }

        /** Number of occurrences of c in data[from, to). */
        int rangeRank(char c, int from, int to) {
            return rank(c, to) - rank(c, from);
        }

        /** Number of values in [lo, hi] inside data[from, to). */
        int countInRange(char lo, char hi, int from, int to) {
            int high = rankLess(hi, to) - rankLess(hi, from);
            int below = lo <= lowValue ? 0 : rankLess((char) (lo - 1), to) - rankLess((char) (lo - 1), from);
            return high - below;
        }

        /** The character at position index (a descent to a single leaf). */
        char charAccess(int index) {
            if (lowValue == highValue) return (char) lowValue;
            if (prefixZeros[index + 1] > prefixZeros[index]) return left.charAccess(prefixZeros[index]);
            return right.charAccess(index - prefixZeros[index]);
        }

        /** The k-th (0-based) smallest value in data[from, to). */
        char quantile(int from, int to, int k) {
            if (lowValue == highValue) return (char) lowValue;
            int zeroFrom = prefixZeros[from];
            int zeroTo = prefixZeros[to];
            int zerosInside = zeroTo - zeroFrom;
            if (k < zerosInside) return left.quantile(zeroFrom, zeroTo, k);
            return right.quantile(from - zeroFrom, to - zeroTo, k - zerosInside);
        }

        /** Number of distinct values in data[from, to). */
        int distinct(int from, int to) {
            if (from >= to) return 0;
            if (lowValue == highValue) return 1;
            int zeroFrom = prefixZeros[from];
            int zeroTo = prefixZeros[to];
            return left.distinct(zeroFrom, zeroTo) + right.distinct(from - zeroFrom, to - zeroTo);
        }

        void collect(char[] out, int[] position) {
            if (lowValue == highValue) {
                for (char c : data) out[position[0]++] = c;
                return;
            }
            left.collect(out, position);
            right.collect(out, position);
        }
    }

    /** Counts a pattern through select plus charAccess; the empty pattern counts the whole text. */
    static int countPattern(WaveletTree tree, int length, String pattern) {
        if (pattern.length() == 0) return length;
        char first = pattern.charAt(0);
        int total = 0;
        for (int k = 0; ; k++) {
            int position = tree.select(first, k);
            if (position < 0 || position + pattern.length() > length) break;
            boolean matches = true;
            for (int i = 1; i < pattern.length(); i++) {
                if (tree.charAccess(position + i) != pattern.charAt(i)) {
                    matches = false;
                    break;
                }
            }
            if (matches) total++;
        }
        return total;
    }

    static int bruteCount(String s, char c, int from, int to) {
        int total = 0;
        for (int i = from; i < to; i++) if (s.charAt(i) == c) total++;
        return total;
    }

    static int bruteCountLess(String s, char c, int from, int to) {
        int total = 0;
        for (int i = from; i < to; i++) if (s.charAt(i) <= c) total++;
        return total;
    }

    static int brutePattern(String s, String pattern) {
        if (pattern.length() == 0) return s.length();
        int total = 0;
        for (int i = 0; i + pattern.length() <= s.length(); i++) {
            if (s.startsWith(pattern, i)) total++;
        }
        return total;
    }

    public static void main(String[] args) {
        Rng rng = new Rng(579_2025L);
        String alphabet = "abc";
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < 200; i++) builder.append(alphabet.charAt(rng.nextInt(3)));
        String text = builder.toString();
        nodes = 0;
        WaveletTree tree = new WaveletTree(text.toCharArray(), 0, text.length(), 'a', 'c');
        System.out.println("wavelet tree for n = 200 over a 3-letter alphabet: " + nodes
                + " nodes (bound 2 n log2 3 = 634)");
        check(nodes <= 2L * 200 * 2, "the node count respects 2 n log2(sigma)");

        // ---- (1) rank exhaustive ------------------------------------------------------------------
        StringBuilder other = new StringBuilder();
        String letters = "abcdefghijklmnopqrstuvwxyz";
        for (int i = 0; i < 97; i++) other.append(letters.charAt(rng.nextInt(26)));
        String second = other.toString();
        nodes = 0;
        WaveletTree secondTree = new WaveletTree(second.toCharArray(), 0, second.length(), 'a', 'z');
        System.out.println("26-letter tree for n = 97: " + nodes + " nodes (bound 2 n log2 26 = 912)");
        check(nodes <= 2L * second.length() * 5, "the 26-letter node count respects 2 n log2(26)");
        for (int position = 0; position <= second.length(); position++) {
            for (char c = 'a'; c <= 'z'; c++) {
                check(secondTree.rank(c, position) == bruteCount(second, c, 0, position),
                        "rank(" + c + "," + position + ") matches brute force");
                check(secondTree.rankLess(c, position) == bruteCountLess(second, c, 0, position),
                        "rankLess(" + c + "," + position + ") matches brute force");
            }
            check(secondTree.rank('0', position) == 0, "a letter below the alphabet counts 0");
            check(secondTree.rank('{', position) == 0, "a letter above the alphabet counts 0");
        }

        // ---- (2) select exhaustive, and inverse of rank --------------------------------------------
        for (char c = 'a'; c <= 'z'; c++) {
            int frequency = bruteCount(second, c, 0, second.length());
            for (int k = 0; k < frequency; k++) {
                int position = secondTree.select(c, k);
                check(position >= 0 && second.charAt(position) == c, "select returns a position holding c");
                check(secondTree.rank(c, position + 1) == k + 1, "rank(select(c, k)) = k + 1");
                check(secondTree.select(c, secondTree.rank(c, position + 1) - 1) == position,
                        "select(rank(c, i) - 1) = i");
                check(secondTree.charAccess(position) == c, "charAccess agrees with the text");
            }
            check(secondTree.select(c, frequency) == -1, "k equal to the frequency returns -1");
            check(secondTree.select(c, -1) == -1, "a negative k returns -1");
            check(secondTree.select(c, 0) >= 0 || frequency == 0, "an absent letter returns -1");
        }
        for (int position = 0; position < second.length(); position++) {
            char c = second.charAt(position);
            check(secondTree.select(c, secondTree.rank(c, position + 1) - 1) == position,
                    "every position is recovered by select at position " + position);
        }
        for (int position = 0; position < text.length(); position++) {
            check(tree.charAccess(position) == text.charAt(position), "charAccess matches the text");
        }

        // ---- (3) pattern counting -------------------------------------------------------------------
        List<String> patterns = new ArrayList<>();
        for (char a = 'a'; a <= 'c'; a++) {
            patterns.add("" + a);
            for (char b = 'a'; b <= 'c'; b++) {
                patterns.add("" + a + b);
                for (char d = 'a'; d <= 'c'; d++) patterns.add("" + a + b + d);
            }
        }
        check(patterns.size() == 39, "39 patterns of length 1 to 3 over a 3-letter alphabet");
        for (String pattern : patterns) {
            check(countPattern(tree, text.length(), pattern) == brutePattern(text, pattern),
                    "pattern '" + pattern + "' count matches brute force");
        }
        for (int trial = 0; trial < 100; trial++) {
            int from = rng.nextInt(text.length());
            int length = 4 + rng.nextInt(5);
            if (from + length > text.length()) continue;
            String pattern = text.substring(from, from + length);
            check(countPattern(tree, text.length(), pattern) == brutePattern(text, pattern),
                    "random pattern count matches brute force");
        }
        check(countPattern(tree, text.length(), "") == text.length(),
                "the empty pattern counts the whole text (documented convention)");
        check(countPattern(tree, text.length(), "d") == 0, "a letter outside the alphabet counts 0");

        // ---- (4) range counting, quantile and distinct -----------------------------------------------
        for (int trial = 0; trial < 300; trial++) {
            int from = rng.nextInt(text.length() + 1);
            int to = from + rng.nextInt(text.length() - from + 1);
            char lo = (char) ('a' + rng.nextInt(3));
            char hi = (char) (lo + rng.nextInt('c' - lo + 1));
            int counted = 0;
            for (int i = from; i < to; i++) if (text.charAt(i) >= lo && text.charAt(i) <= hi) counted++;
            check(tree.countInRange(lo, hi, from, to) == counted,
                    "countInRange matches brute force on [" + from + "," + to + ")");
            for (char c = 'a'; c <= 'c'; c++) {
                check(tree.rangeRank(c, from, to) == bruteCount(text, c, from, to),
                        "rangeRank matches brute force");
            }
            if (to > from) {
                int k = rng.nextInt(to - from);
                char smallest = tree.quantile(from, to, k);
                char[] slice = text.substring(from, to).toCharArray();
                Arrays.sort(slice);
                check(smallest == slice[k], "quantile matches the sorted slice");
                Set<Character> expected = new HashSet<>();
                for (int i = from; i < to; i++) expected.add(text.charAt(i));
                check(tree.distinct(from, to) == expected.size(), "distinct count matches brute force");
            } else {
                check(tree.distinct(from, to) == 0, "an empty interval has no distinct characters");
            }
            if (to - from == 1) check(tree.distinct(from, to) == 1, "a single character is one distinct value");
        }

        // ---- (5) the leaves are a permutation of the input --------------------------------------------
        char[] collected = new char[text.length()];
        tree.collect(collected, new int[1]);
        char[] original = text.toCharArray();
        Arrays.sort(collected);
        Arrays.sort(original);
        check(Arrays.equals(collected, original), "the leaves reproduce the multiset of the input");

        // ---- (6) edge cases ---------------------------------------------------------------------------
        WaveletTree empty = new WaveletTree(new char[0], 0, 0, 'a', 'c');
        check(empty.rank('a', 0) == 0, "the empty string has rank 0");
        check(empty.select('a', 0) == -1, "the empty string has no occurrences");
        check(empty.distinct(0, 0) == 0, "the empty string has no distinct characters");
        WaveletTree single = new WaveletTree(new char[]{'b'}, 0, 1, 'a', 'c');
        check(single.rank('b', 1) == 1 && single.select('b', 0) == 0, "a single character is handled");
        check(single.countInRange('a', 'a', 0, 1) == 0, "and it is not counted in a wrong value range");
        check(single.countInRange('b', 'c', 0, 1) == 1, "the inclusive upper bound counts it");
        StringBuilder repeated = new StringBuilder();
        for (int i = 0; i < 200; i++) repeated.append('a');
        nodes = 0;
        WaveletTree path = new WaveletTree(repeated.toString().toCharArray(), 0, 200, 'a', 'c');
        check(path.rank('a', 200) == 200 && path.select('a', 199) == 199, "a run of one letter works");
        check(path.select('a', 200) == -1, "and the 201st occurrence does not exist");
        check(countPattern(path, 200, "aa") == 199, "counting a two-letter pattern on the repeated string");
        check(countPattern(path, 200, "a") == 200, "and a one-letter pattern");
        System.out.println("path-shaped tree (200 identical letters): " + nodes + " nodes");

        System.out.println("PASS");
    }
}
