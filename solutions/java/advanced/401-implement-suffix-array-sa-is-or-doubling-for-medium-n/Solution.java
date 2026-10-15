/** Task 401 - Implement suffix array (SA-IS or doubling) for medium N. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * `SuffixArrays.doubling` sorts the suffixes by their first 2^k characters, doubling k until
 * every rank is unique: O(n log^2 n) with comparison sorting. No sentinel is appended - a
 * position past the end of the string ranks below every character, which resolves "one suffix
 * is a prefix of the other" correctly because the shorter suffix must win.
 *
 * Correctness is pinned by two independent oracles: a naive suffix sort and the hand-written
 * character-by-character comparator. The property tests include medium N (a few hundred
 * characters), so the doubling loop really runs several rounds.
 */
import java.util.*;

class SuffixArrays {
    static int[] doubling(String s) {
        int n = s.length();
        if (n == 0) return new int[0];
        Integer[] sa = new Integer[n];
        for (int i = 0; i < n; i++) sa[i] = i;
        final int[] rank = new int[n];
        for (int i = 0; i < n; i++) rank[i] = s.charAt(i);
        for (int k = 1; ; k <<= 1) {
            final int step = k;
            Arrays.sort(sa, (a, b) -> {
                if (rank[a] != rank[b]) return Integer.compare(rank[a], rank[b]);
                int ra = a + step < n ? rank[a + step] : -1;
                int rb = b + step < n ? rank[b + step] : -1;
                return Integer.compare(ra, rb);
            });
            int[] next = new int[n];
            for (int i = 1; i < n; i++) {
                int prev = sa[i - 1], cur = sa[i];
                int pa = prev + step < n ? rank[prev + step] : -1;
                int pb = cur + step < n ? rank[cur + step] : -1;
                next[cur] = next[prev] + (rank[prev] != rank[cur] || pa != pb ? 1 : 0);
            }
            System.arraycopy(next, 0, rank, 0, n);
            if (rank[sa[n - 1]] == n - 1) break;
        }
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = sa[i];
        return out;
    }

    static int compare(String s, int i, int j) {
        int n = s.length();
        while (i < n && j < n) {
            if (s.charAt(i) != s.charAt(j)) return s.charAt(i) < s.charAt(j) ? -1 : 1;
            i++;
            j++;
        }
        if (i == j) return 0;
        return i == n ? -1 : 1;
    }

    static int[] naive(String s) {
        int n = s.length();
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> compare(s, a, b));
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = idx[i];
        return out;
    }

    static boolean isValid(String s, int[] order) {
        int n = s.length();
        if (order.length != n) return false;
        boolean[] seen = new boolean[n];
        for (int start : order) {
            if (start < 0 || start >= n || seen[start]) return false;
            seen[start] = true;
        }
        for (int i = 0; i + 1 < n; i++) {
            if (compare(s, order[i], order[i + 1]) >= 0) return false;
        }
        return true;
    }

    static String join(int[] values) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(values[i]);
        }
        return sb.toString();
    }
}

public class Solution {
    private static int seed = 2718;

    static String lcgString(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append(alphabet.charAt(seed % alphabet.length()));
        }
        return sb.toString();
    }

    static String repeat(String unit, int times) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < times; i++) sb.append(unit);
        return sb.toString();
    }

    public static void main(String[] args) {
        check(SuffixArrays.join(SuffixArrays.doubling("banana")).equals("5,3,1,0,4,2"), "banana");
        check(SuffixArrays.join(SuffixArrays.doubling("mississippi"))
                .equals("10,7,4,1,0,9,8,6,3,5,2"), "mississippi");

        check(SuffixArrays.doubling("").length == 0, "empty");
        check(SuffixArrays.join(SuffixArrays.doubling("a")).equals("0"), "single");
        check(SuffixArrays.join(SuffixArrays.doubling("aaaa")).equals("3,2,1,0"), "aaaa");
        check(SuffixArrays.join(SuffixArrays.doubling("abab")).equals("2,0,3,1"), "abab");
        String abracadabra = "abracadabra";
        check(SuffixArrays.join(SuffixArrays.doubling(abracadabra))
                .equals(SuffixArrays.join(SuffixArrays.naive(abracadabra))), "abracadabra");

        String[] small = {"", "a", "b", "aa", "ab", "ba", "aaaa", "abab", "banana", "mississippi", "abracadabra"};
        for (String s : small) {
            int[] order = SuffixArrays.doubling(s);
            check(SuffixArrays.join(order).equals(SuffixArrays.join(SuffixArrays.naive(s))), "small naive " + s);
            check(SuffixArrays.isValid(s, order), "small valid " + s);
        }

        for (String alphabet : new String[]{"ab", "abc", "abcdefghijklmnopqrstuvwxyz"}) {
            for (int trial = 0; trial < 40; trial++) {
                String s = lcgString(28, alphabet);
                int[] order = SuffixArrays.doubling(s);
                check(SuffixArrays.join(order).equals(SuffixArrays.join(SuffixArrays.naive(s))), "lcg naive " + s);
                check(SuffixArrays.isValid(s, order), "lcg valid " + s);
            }
        }

        String[] medium = {
            lcgString(500, "ab"),
            lcgString(500, "abcdefghijklmnopqrstuvwxyz"),
            repeat("a", 400),
            repeat("abcab", 100),
        };
        for (String s : medium) {
            int[] order = SuffixArrays.doubling(s);
            check(SuffixArrays.join(order).equals(SuffixArrays.join(SuffixArrays.naive(s))), "medium naive");
            check(SuffixArrays.isValid(s, order), "medium comparator");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
