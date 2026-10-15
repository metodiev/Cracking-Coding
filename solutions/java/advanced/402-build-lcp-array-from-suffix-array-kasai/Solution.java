/** Task 402 - Build the LCP array from the suffix array (Kasai). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Kasai's algorithm walks the suffixes in text order, reusing the previous longest common
 * prefix: if suffix i shares h characters with its suffix-array predecessor, then suffix i+1
 * shares at least h-1 with its own predecessor, so each position is advanced amortized once
 * and the whole scan is O(n). lcp[i] is the LCP of suffixes sa[i-1] and sa[i], lcp[0] = 0.
 *
 * Hand-checked anchors: banana -> [0, 1, 3, 0, 0, 2] and mississippi -> [0, 1, 1, 4, 0, 0, 1,
 * 0, 2, 1, 3]. A pairwise char-by-char oracle and the identity "distinct substrings =
 * n(n+1)/2 - sum(lcp)" cross-check it on LCG strings.
 */
import java.util.*;

class LcpArrays {
    static int[] kasai(String s, int[] sa) {
        int n = s.length();
        int[] lcp = new int[n];
        int[] rank = new int[n];
        for (int i = 0; i < n; i++) rank[sa[i]] = i;
        int h = 0;
        for (int i = 0; i < n; i++) {
            int r = rank[i];
            if (r == 0) {
                h = 0;
                continue;
            }
            int j = sa[r - 1];
            while (i + h < n && j + h < n && s.charAt(i + h) == s.charAt(j + h)) h++;
            lcp[r] = h;
            if (h > 0) h--;
        }
        return lcp;
    }

    static int pairLcp(String s, int i, int j) {
        int n = s.length();
        int h = 0;
        while (i + h < n && j + h < n && s.charAt(i + h) == s.charAt(j + h)) h++;
        return h;
    }

    static int[] naive(String s, int[] sa) {
        int n = sa.length;
        if (n == 0) return new int[0];
        int[] lcp = new int[n];
        for (int i = 1; i < n; i++) lcp[i] = pairLcp(s, sa[i - 1], sa[i]);
        return lcp;
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

    static int[] suffixArrayNaive(String s) {
        int n = s.length();
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> compare(s, a, b));
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = idx[i];
        return out;
    }

    static int distinctSubstringsBrute(String s) {
        Set<String> seen = new HashSet<>();
        for (int i = 0; i < s.length(); i++)
            for (int j = i + 1; j <= s.length(); j++)
                seen.add(s.substring(i, j));
        return seen.size();
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

    static int sum(int[] values) {
        int total = 0;
        for (int value : values) total += value;
        return total;
    }

    public static void main(String[] args) {
        int[] banana = LcpArrays.kasai("banana", new int[]{5, 3, 1, 0, 4, 2});
        check(LcpArrays.join(banana).equals("0,1,3,0,0,2"), "banana");
        check(LcpArrays.join(banana).equals(LcpArrays.join(LcpArrays.naive("banana", new int[]{5, 3, 1, 0, 4, 2}))),
                "banana naive");

        int[] mississippi = LcpArrays.kasai("mississippi", new int[]{10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2});
        check(LcpArrays.join(mississippi).equals("0,1,1,4,0,0,1,0,2,1,3"), "mississippi");

        check(LcpArrays.kasai("", new int[0]).length == 0, "empty");
        check(LcpArrays.join(LcpArrays.kasai("a", new int[]{0})).equals("0"), "single");
        check(LcpArrays.join(LcpArrays.kasai("aaaa", new int[]{3, 2, 1, 0})).equals("0,1,2,3"), "aaaa");
        check(LcpArrays.join(LcpArrays.kasai("abab", new int[]{2, 0, 3, 1})).equals("0,2,0,1"), "abab");

        String[] small = {"", "a", "aaa", "abab", "banana", "mississippi", "abracadabra", "aaaaabaaaab"};
        for (String s : small) {
            int[] sa = LcpArrays.suffixArrayNaive(s);
            int[] lcp = LcpArrays.kasai(s, sa);
            check(LcpArrays.join(lcp).equals(LcpArrays.join(LcpArrays.naive(s, sa))), "small " + s);
            check(lcp.length == s.length(), "length " + s);
            check(lcp.length == 0 || lcp[0] == 0, "first " + s);
        }

        check(LcpArrays.distinctSubstringsBrute("banana") == 15, "banana distinct");
        check(6 * 7 / 2 - sum(banana) == 15, "banana identity");

        for (String alphabet : new String[]{"ab", "abc"}) {
            for (int trial = 0; trial < 40; trial++) {
                String s = lcgString(24, alphabet);
                int[] sa = LcpArrays.suffixArrayNaive(s);
                int[] lcp = LcpArrays.kasai(s, sa);
                check(LcpArrays.join(lcp).equals(LcpArrays.join(LcpArrays.naive(s, sa))), "lcg " + s);
                check(s.length() * (s.length() + 1) / 2 - sum(lcp) == LcpArrays.distinctSubstringsBrute(s),
                        "identity " + s);
            }
        }

        String medium = lcgString(400, "ab");
        int[] mediumSa = LcpArrays.suffixArrayNaive(medium);
        int[] mediumLcp = LcpArrays.kasai(medium, mediumSa);
        check(LcpArrays.join(mediumLcp).equals(LcpArrays.join(LcpArrays.naive(medium, mediumSa))), "medium");
        check(mediumLcp[0] == 0, "medium first");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
