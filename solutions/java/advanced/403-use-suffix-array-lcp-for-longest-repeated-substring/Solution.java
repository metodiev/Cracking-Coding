/** Task 403 - Use the suffix array + LCP array for the longest repeated substring. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A substring occurring at least twice is a common prefix of two different suffixes, and those
 * two suffixes are adjacent somewhere in the suffix array, so the answer length is max(lcp)
 * and one witness is the prefix of that length of sa[i] at the first maximizing i.
 *
 * Sentinel convention: when nothing repeats (length < 2 or all substrings unique, e.g. "abc")
 * the function returns "" - documented so "no repeat" is distinguishable from a one-character
 * repeat. Anchors: banana -> "ana", mississippi -> "issi", aaaa -> "aaa", abab -> "ab".
 * Cross-checked against a brute-force substring counter on LCG strings (same maximum length,
 * witness re-verified to occur at least twice).
 */
import java.util.*;

class RepeatedSubstring {
    static int[] suffixArrayDoubling(String s) {
        int n = s.length();
        if (n == 0) return new int[0];
        Integer[] sa = new Integer[n];
        for (int i = 0; i < n; i++) sa[i] = i;
        int[] rank = new int[n];
        for (int i = 0; i < n; i++) rank[i] = s.charAt(i);
        for (int k = 1; ; k <<= 1) {
            final int step = k;
            final int[] current = rank;
            Arrays.sort(sa, (a, b) -> {
                if (current[a] != current[b]) return Integer.compare(current[a], current[b]);
                int ra = a + step < n ? current[a + step] : -1;
                int rb = b + step < n ? current[b + step] : -1;
                return Integer.compare(ra, rb);
            });
            int[] next = new int[n];
            for (int i = 1; i < n; i++) {
                int prev = sa[i - 1], cur = sa[i];
                int pa = prev + step < n ? rank[prev + step] : -1;
                int pb = cur + step < n ? rank[cur + step] : -1;
                next[cur] = next[prev] + (rank[prev] != rank[cur] || pa != pb ? 1 : 0);
            }
            rank = next;
            if (rank[sa[n - 1]] == n - 1) break;
        }
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = sa[i];
        return out;
    }

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

    static String longestRepeated(String s) {
        if (s.length() < 2) return "";
        int[] sa = suffixArrayDoubling(s);
        int[] lcp = kasai(s, sa);
        int bestLength = 0;
        int bestStart = -1;
        for (int i = 1; i < s.length(); i++) {
            if (lcp[i] > bestLength) {
                bestLength = lcp[i];
                bestStart = sa[i];
            }
        }
        return bestLength == 0 ? "" : s.substring(bestStart, bestStart + bestLength);
    }

    static int longestRepeatedLengthBrute(String s) {
        Map<String, Integer> counts = new HashMap<>();
        for (int i = 0; i < s.length(); i++)
            for (int j = i + 1; j <= s.length(); j++)
                counts.merge(s.substring(i, j), 1, Integer::sum);
        int best = 0;
        for (Map.Entry<String, Integer> entry : counts.entrySet())
            if (entry.getValue() >= 2 && entry.getKey().length() > best) best = entry.getKey().length();
        return best;
    }

    static int countOccurrences(String s, String pattern) {
        if (pattern.isEmpty()) return 0;
        int count = 0;
        for (int i = 0; i + pattern.length() <= s.length(); i++)
            if (s.startsWith(pattern, i)) count++;
        return count;
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

    public static void main(String[] args) {
        check(RepeatedSubstring.longestRepeated("banana").equals("ana"), "banana");
        check(RepeatedSubstring.countOccurrences("banana", "ana") == 2, "banana count");
        check(RepeatedSubstring.longestRepeated("mississippi").equals("issi"), "mississippi");
        check(RepeatedSubstring.countOccurrences("mississippi", "issi") == 2, "mississippi count");
        check(RepeatedSubstring.longestRepeated("aaaa").equals("aaa"), "aaaa");
        check(RepeatedSubstring.longestRepeated("abab").equals("ab"), "abab");
        check(RepeatedSubstring.longestRepeated("abcabc").equals("abc"), "abcabc");
        check(RepeatedSubstring.longestRepeated("aabcaa").equals("aa"), "aabcaa");
        check(RepeatedSubstring.longestRepeated("abracadabra").equals("abra"), "abracadabra");
        check(RepeatedSubstring.countOccurrences("abracadabra", "abra") == 2, "abracadabra count");

        check(RepeatedSubstring.longestRepeated("").equals(""), "empty");
        check(RepeatedSubstring.longestRepeated("a").equals(""), "single");
        check(RepeatedSubstring.longestRepeated("abc").equals(""), "abc sentinel");
        check(RepeatedSubstring.longestRepeated("abcd").equals(""), "abcd sentinel");
        check(RepeatedSubstring.longestRepeatedLengthBrute("abc") == 0, "abc brute");

        for (String alphabet : new String[]{"ab", "abc"}) {
            for (int trial = 0; trial < 40; trial++) {
                String s = lcgString(30, alphabet);
                String result = RepeatedSubstring.longestRepeated(s);
                int best = RepeatedSubstring.longestRepeatedLengthBrute(s);
                check(result.length() == best, "lcg length " + s);
                if (best == 0) check(result.isEmpty(), "lcg sentinel " + s);
                else check(RepeatedSubstring.countOccurrences(s, result) >= 2, "lcg witness " + s);
            }
        }

        String medium = lcgString(400, "ab");
        String mediumResult = RepeatedSubstring.longestRepeated(medium);
        check(!mediumResult.isEmpty() && RepeatedSubstring.countOccurrences(medium, mediumResult) >= 2, "medium witness");
        check(mediumResult.length() == RepeatedSubstring.longestRepeatedLengthBrute(medium), "medium length");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
