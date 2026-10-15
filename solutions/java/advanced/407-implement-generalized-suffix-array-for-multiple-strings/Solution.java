/** Task 407 - Implement a generalized suffix array for multiple strings. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * The m strings are concatenated with m *distinct* sentinels: string k gets the character
 * (char)(1 + k). All sentinels are below every real (lowercase) character, and they are
 * distinct and ordered by string index, so a suffix from string i always beats the same text
 * extended in string j when i < j. This is the cheapest correct sentinel scheme for the
 * generalized suffix array.
 *
 * Every suffix also carries an owner (which string) and an offset (position inside it), so
 * occurrences of a pattern are found with two binary searches over the suffix array: the
 * pattern-prefixed suffixes form one contiguous range, and each entry of that range maps back
 * to a (string, position) occurrence.
 *
 * Anchors: ["banana", "ana"] -> occurrences of "ana" = (0,1), (0,3), (1,0). Checks: the
 * combined SA validates against a manual comparator, the smallest suffixes are the sentinels
 * in index order, suffixes of each string keep their own plain relative order, and LCG-built
 * pattern searches agree with a naive scan.
 */
import java.util.*;

class GeneralizedSa {
    final String text;
    final int[] sa;
    final int[] owner;
    final int[] offset;
    final int[] sentinels;

    GeneralizedSa(String text, int[] sa, int[] owner, int[] offset, int[] sentinels) {
        this.text = text;
        this.sa = sa;
        this.owner = owner;
        this.offset = offset;
        this.sentinels = sentinels;
    }

    static GeneralizedSa build(List<String> strings) {
        StringBuilder sb = new StringBuilder();
        List<Integer> owners = new ArrayList<>();
        List<Integer> offsets = new ArrayList<>();
        List<Integer> sentinels = new ArrayList<>();
        for (int k = 0; k < strings.size(); k++) {
            String s = strings.get(k);
            for (int pos = 0; pos < s.length(); pos++) {
                sb.append(s.charAt(pos));
                owners.add(k);
                offsets.add(pos);
            }
            sentinels.add(sb.length());
            sb.append((char) (1 + k));
            owners.add(k);
            offsets.add(-1);
        }
        String text = sb.toString();
        int[] sa = suffixArrayDoubling(text);
        int[] owner = new int[owners.size()];
        int[] offset = new int[offsets.size()];
        int[] sentinel = new int[sentinels.size()];
        for (int i = 0; i < owner.length; i++) {
            owner[i] = owners.get(i);
            offset[i] = offsets.get(i);
        }
        for (int i = 0; i < sentinel.length; i++) sentinel[i] = sentinels.get(i);
        return new GeneralizedSa(text, sa, owner, offset, sentinel);
    }

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

    static boolean isValid(String s, int[] order) {
        if (order.length != s.length()) return false;
        boolean[] seen = new boolean[s.length()];
        for (int start : order) {
            if (start < 0 || start >= s.length() || seen[start]) return false;
            seen[start] = true;
        }
        for (int i = 0; i + 1 < order.length; i++)
            if (compare(s, order[i], order[i + 1]) >= 0) return false;
        return true;
    }

    static int cmpPrefix(String text, int start, String pattern) {
        for (int i = 0; i < pattern.length(); i++) {
            if (start + i >= text.length()) return -1;
            char ch = text.charAt(start + i);
            if (ch != pattern.charAt(i)) return ch < pattern.charAt(i) ? -1 : 1;
        }
        return 0;
    }

    static int[] patternRange(String text, int[] sa, String pattern) {
        int lo = 0, hi = sa.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (cmpPrefix(text, sa[mid], pattern) < 0) lo = mid + 1;
            else hi = mid;
        }
        int lower = lo;
        hi = sa.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (cmpPrefix(text, sa[mid], pattern) <= 0) lo = mid + 1;
            else hi = mid;
        }
        return new int[]{lower, lo};
    }

    static String findOccurrences(List<String> strings, String pattern) {
        GeneralizedSa built = build(strings);
        int[] range = patternRange(built.text, built.sa, pattern);
        List<int[]> found = new ArrayList<>();
        for (int i = range[0]; i < range[1]; i++) {
            int idx = built.sa[i];
            if (built.offset[idx] >= 0) found.add(new int[]{built.owner[idx], built.offset[idx]});
        }
        found.sort((a, b) -> a[0] != b[0] ? Integer.compare(a[0], b[0]) : Integer.compare(a[1], b[1]));
        return join(found);
    }

    static String naiveOccurrences(List<String> strings, String pattern) {
        List<int[]> found = new ArrayList<>();
        for (int k = 0; k < strings.size(); k++) {
            String s = strings.get(k);
            for (int i = 0; i + pattern.length() <= s.length(); i++) {
                if (s.startsWith(pattern, i)) found.add(new int[]{k, i});
            }
        }
        found.sort((a, b) -> a[0] != b[0] ? Integer.compare(a[0], b[0]) : Integer.compare(a[1], b[1]));
        return join(found);
    }

    private static String join(List<int[]> pairs) {
        StringBuilder sb = new StringBuilder();
        for (int[] pair : pairs) {
            if (sb.length() > 0) sb.append(",");
            sb.append(pair[0]).append(":").append(pair[1]);
        }
        return sb.toString();
    }

    static String joinInts(List<Integer> values) {
        StringBuilder sb = new StringBuilder();
        for (int value : values) {
            if (sb.length() > 0) sb.append(",");
            sb.append(value);
        }
        return sb.toString();
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

    static List<String> allShortSubstrings(List<String> strings, int maxLen) {
        TreeSet<String> seen = new TreeSet<>();
        for (String s : strings)
            for (int i = 0; i < s.length(); i++)
                for (int j = i + 1; j <= Math.min(s.length(), i + maxLen); j++)
                    seen.add(s.substring(i, j));
        return new ArrayList<>(seen);
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

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    public static void main(String[] args) {
        List<String> strings = Arrays.asList("banana", "ana");
        GeneralizedSa built = GeneralizedSa.build(strings);
        check(built.text.length() == 6 + 1 + 3 + 1, "text length");
        check(GeneralizedSa.isValid(built.text, built.sa), "combined valid");
        check(built.sa[0] == built.sentinels[0] && built.sa[1] == built.sentinels[1], "sentinels first");
        for (int k = 0; k < strings.size(); k++) {
            List<Integer> kept = new ArrayList<>();
            for (int idx : built.sa)
                if (built.owner[idx] == k && built.offset[idx] >= 0) kept.add(built.offset[idx]);
            StringBuilder expected = new StringBuilder();
            for (int pos : GeneralizedSa.suffixArrayNaive(strings.get(k))) {
                if (expected.length() > 0) expected.append(",");
                expected.append(pos);
            }
            check(GeneralizedSa.joinInts(kept).equals(expected.toString()), "relative order " + k);
        }

        check(GeneralizedSa.findOccurrences(strings, "ana").equals("0:1,0:3,1:0"), "ana");
        check(GeneralizedSa.findOccurrences(strings, "a").equals("0:1,0:3,0:5,1:0,1:2"), "a");
        check(GeneralizedSa.findOccurrences(strings, "na").equals("0:2,0:4,1:1"), "na");
        check(GeneralizedSa.findOccurrences(strings, "ban").equals("0:0"), "ban");
        check(GeneralizedSa.findOccurrences(strings, "banana").equals("0:0"), "banana");
        check(GeneralizedSa.findOccurrences(strings, "b").equals("0:0"), "b");
        check(GeneralizedSa.findOccurrences(strings, "xyz").isEmpty(), "xyz");
        check(GeneralizedSa.findOccurrences(strings, "anan").equals("0:1"), "anan");
        check(GeneralizedSa.findOccurrences(strings, "nanan").isEmpty(), "nanan");
        check(GeneralizedSa.findOccurrences(strings, "bananas").isEmpty(), "bananas");

        check(GeneralizedSa.findOccurrences(Arrays.asList("abab"), "ab").equals("0:0,0:2"), "single");
        check(GeneralizedSa.findOccurrences(Arrays.asList("", "ab"), "ab").equals("1:0"), "empty first");
        check(GeneralizedSa.findOccurrences(Arrays.asList("", ""), "a").isEmpty(), "both empty");
        check(GeneralizedSa.findOccurrences(new ArrayList<>(), "a").isEmpty(), "no strings");

        List<String> quad = Arrays.asList("mississippi", "issi", "i", "ppi");
        List<String> patterns = GeneralizedSa.allShortSubstrings(quad, 3);
        patterns.addAll(Arrays.asList("issi", "ississi", "ssippi", "zz", "q", "ippiz"));
        for (String pattern : patterns) {
            check(GeneralizedSa.findOccurrences(quad, pattern).equals(GeneralizedSa.naiveOccurrences(quad, pattern)),
                    "quad " + pattern);
        }
        check(GeneralizedSa.findOccurrences(quad, "i").equals(GeneralizedSa.naiveOccurrences(quad, "i")), "quad i");

        for (int trial = 0; trial < 30; trial++) {
            List<String> random = new ArrayList<>();
            for (int k = 0; k < 3; k++) random.add(lcgString(12 + seed % 7, "abc"));
            GeneralizedSa randomBuilt = GeneralizedSa.build(random);
            check(GeneralizedSa.isValid(randomBuilt.text, randomBuilt.sa), "lcg valid");
            for (int k = 0; k < random.size(); k++) {
                check(randomBuilt.sa[k] == randomBuilt.sentinels[k], "lcg sentinel " + k);
                List<Integer> kept = new ArrayList<>();
                for (int idx : randomBuilt.sa)
                    if (randomBuilt.owner[idx] == k && randomBuilt.offset[idx] >= 0) kept.add(randomBuilt.offset[idx]);
                StringBuilder expected = new StringBuilder();
                for (int pos : GeneralizedSa.suffixArrayNaive(random.get(k))) {
                    if (expected.length() > 0) expected.append(",");
                    expected.append(pos);
                }
                check(GeneralizedSa.joinInts(kept).equals(expected.toString()), "lcg relative " + k);
            }
            List<String> randomPatterns = GeneralizedSa.allShortSubstrings(random, 3);
            randomPatterns.addAll(Arrays.asList("zzz", "q", "abcabca"));
            for (String pattern : randomPatterns) {
                check(GeneralizedSa.findOccurrences(random, pattern).equals(GeneralizedSa.naiveOccurrences(random, pattern)),
                        "lcg pattern " + pattern);
            }
        }
        System.out.println("PASS");
    }
}
