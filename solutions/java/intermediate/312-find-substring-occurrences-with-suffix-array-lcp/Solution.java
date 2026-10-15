/** Task 312 - Substring occurrences with a suffix array + LCP (Kasai). (Tier: Intermediate, CTCI Ch 1) */
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

    /** lcp[i] = LCP(sa[i-1], sa[i]) with lcp[0] = 0; O(n) via the Kasai trick. */
    static int[] kasaiLcp(String s, int[] sa) {
        int n = s.length();
        int[] rank = new int[n];
        for (int i = 0; i < n; i++) rank[sa[i]] = i;
        int[] lcp = new int[n];
        int h = 0;
        for (int i = 0; i < n; i++) {
            if (rank[i] > 0) {
                int j = sa[rank[i] - 1];
                while (i + h < n && j + h < n && s.charAt(i + h) == s.charAt(j + h)) h++;
                lcp[rank[i]] = h;
                if (h > 0) h--;
            } else {
                h = 0;
            }
        }
        return lcp;
    }

    /** Direct character-by-character LCP of two suffixes; the independent oracle. */
    static int lcpOfSuffixes(String s, int i, int j) {
        int h = 0;
        while (i + h < s.length() && j + h < s.length() && s.charAt(i + h) == s.charAt(j + h)) h++;
        return h;
    }

    /**
     * All start indices of the pattern in s, sorted ascending. The empty pattern matches at
     * every valid position (n + 1 of them).
     */
    static int[] occurrences(String s, String pattern) {
        int n = s.length();
        int m = pattern.length();
        if (m == 0) {
            int[] all = new int[n + 1];
            for (int i = 0; i <= n; i++) all[i] = i;
            return all;
        }
        int[] sa = suffixArray(s);
        int[] lcp = kasaiLcp(s, sa);
        // first index k with suffix sa[k] >= pattern; matching suffixes are exactly the run at k
        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (s.substring(sa[mid]).compareTo(pattern) < 0) lo = mid + 1;
            else hi = mid;
        }
        int k = lo;
        if (k == n || !s.startsWith(pattern, sa[k])) return new int[0];
        List<Integer> result = new ArrayList<>();
        result.add(sa[k]);
        int t = k + 1;
        while (t < n && lcp[t] >= m) {
            result.add(sa[t]);
            t++;
        }
        Collections.sort(result);
        int[] out = new int[result.size()];
        for (int i = 0; i < out.length; i++) out[i] = result.get(i);
        return out;
    }

    /** Scan every position; the independent oracle (overlapping matches included). */
    static int[] naiveOccurrences(String s, String pattern) {
        if (pattern.isEmpty()) {
            int[] all = new int[s.length() + 1];
            for (int i = 0; i <= s.length(); i++) all[i] = i;
            return all;
        }
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i + pattern.length() <= s.length(); i++) {
            if (s.startsWith(pattern, i)) result.add(i);
        }
        int[] out = new int[result.size()];
        for (int i = 0; i < out.length; i++) out[i] = result.get(i);
        return out;
    }

    private static int seed = 3120;

    private static String lcgString(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append(alphabet.charAt(seed % alphabet.length()));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        String s = "banana";
        int[] sa = suffixArray(s);
        check(Arrays.equals(sa, new int[]{5, 3, 1, 0, 4, 2}), "banana suffix array");
        check(isValidSuffixArray(s, sa), "banana valid");
        int[] lcp = kasaiLcp(s, sa);
        check(Arrays.equals(lcp, new int[]{0, 1, 3, 0, 0, 2}), "banana lcp");
        for (int i = 1; i < sa.length; i++) check(lcp[i] == lcpOfSuffixes(s, sa[i - 1], sa[i]), "lcp check " + i);

        check(Arrays.equals(occurrences(s, "ana"), new int[]{1, 3}), "ana");
        check(Arrays.equals(occurrences(s, "ba"), new int[]{0}), "ba");
        check(Arrays.equals(occurrences(s, "an"), new int[]{1, 3}), "an");
        check(Arrays.equals(occurrences(s, "a"), new int[]{1, 3, 5}), "a");
        check(Arrays.equals(occurrences(s, "n"), new int[]{2, 4}), "n");
        check(Arrays.equals(occurrences(s, "banana"), new int[]{0}), "banana");
        check(occurrences(s, "ab").length == 0, "ab absent");
        check(occurrences(s, "xy").length == 0, "xy absent");
        check(occurrences(s, "bananas").length == 0, "bananas absent");
        check(Arrays.equals(occurrences(s, ""), new int[]{0, 1, 2, 3, 4, 5, 6}), "empty pattern");
        check(occurrences("", "a").length == 0, "empty text");

        String[] fixed = {"a", "aa", "aaaa", "abab", "mississippi", "banana", ""};
        List<String> texts = new ArrayList<>(Arrays.asList(fixed));
        for (int i = 0; i < 30; i++) texts.add(lcgString(20, "ab"));
        for (int i = 0; i < 10; i++) texts.add(lcgString(16, "abc"));

        for (String text : texts) {
            int[] textSa = suffixArray(text);
            int[] textLcp = kasaiLcp(text, textSa);
            check(isValidSuffixArray(text, textSa), "valid " + text);
            for (int i = 1; i < textSa.length; i++)
                check(textLcp[i] == lcpOfSuffixes(text, textSa[i - 1], textSa[i]), "lcp " + text + " " + i);
            List<String> patterns = new ArrayList<>();
            for (int i = 0; i < text.length(); i++) {
                for (int extra = 1; extra <= 3; extra++) {
                    if (i + extra <= text.length()) patterns.add(text.substring(i, i + extra));
                }
            }
            patterns.addAll(Arrays.asList("zz", "z", "ababab", "", "bbbb", "ccc"));
            for (String pattern : patterns) {
                check(Arrays.equals(occurrences(text, pattern), naiveOccurrences(text, pattern)),
                        "search " + text + " / " + pattern);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
