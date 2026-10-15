/** Task 310 - Suffix array by naively sorting suffixes. (Tier: Intermediate, CTCI Ch 1) */
import java.util.*;

public class Solution {
    static int[] suffixArray(String s) {
        Integer[] indices = new Integer[s.length()];
        for (int i = 0; i < s.length(); i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> s.substring(a).compareTo(s.substring(b)));
        int[] out = new int[indices.length];
        for (int i = 0; i < indices.length; i++) out[i] = indices[i];
        return out;
    }

    /** Manual lexicographic comparator; the independent oracle. */
    static int compareSuffixes(String s, int i, int j) {
        int n = s.length();
        while (i < n && j < n) {
            if (s.charAt(i) != s.charAt(j)) return s.charAt(i) < s.charAt(j) ? -1 : 1;
            i++;
            j++;
        }
        if (i == j) return 0;
        return i == n ? -1 : 1;                 // the shorter suffix is smaller
    }

    static boolean isValidSuffixArray(String s, int[] order) {
        int[] sorted = order.clone();
        Arrays.sort(sorted);
        for (int i = 0; i < sorted.length; i++) if (sorted[i] != i) return false;
        for (int k = 0; k + 1 < order.length; k++) if (compareSuffixes(s, order[k], order[k + 1]) >= 0) return false;
        return true;
    }

    private static int seed = 2718;

    private static String lcgString(int length) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append("ab".charAt(seed % 2));
        }
        return sb.toString();
    }

    static String join(int[] values) {
        StringBuilder sb = new StringBuilder();
        for (int v : values) {
            if (sb.length() > 0) sb.append(',');
            sb.append(v);
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check(join(suffixArray("banana")).equals("5,3,1,0,4,2"), "banana");
        int[] mississippi = suffixArray("mississippi");
        check(join(mississippi).equals("10,7,4,1,0,9,8,6,3,5,2"), "mississippi");
        check(isValidSuffixArray("mississippi", mississippi), "mississippi valid");

        check(suffixArray("").length == 0, "empty");
        check(join(suffixArray("a")).equals("0"), "single");
        check(join(suffixArray("aaaa")).equals("3,2,1,0"), "aaaa");
        check(join(suffixArray("abab")).equals("2,0,3,1"), "abab");

        String[] samples = new String[60];
        for (int i = 0; i < samples.length; i++) samples[i] = lcgString(12);
        for (String s : samples) check(isValidSuffixArray(s, suffixArray(s)), "random " + s);
        check(isValidSuffixArray("banana", suffixArray("banana")), "banana valid");
        check(isValidSuffixArray("aaaa", suffixArray("aaaa")), "aaaa valid");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
