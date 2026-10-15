/** Task 408 - Pattern matching with a suffix tree (outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Backing store: the suffix array plus the LCP array, which represent the same object as a
 * suffix tree - a suffix-tree node corresponds to an LCP interval, and the locus of a pattern
 * is exactly the maximal interval [lo, hi) of suffixes carrying the pattern as prefix. Descent
 * from the root consumes the pattern one character at a time: two binary searches over the
 * suffix array locate the interval (compare the pattern against the suffix at the midpoint),
 * and every suffix inside the interval is a match, because in a suffix array all
 * pattern-prefixed suffixes are contiguous. The LCP array documents and validates the locus:
 * all suffixes in [lo, hi) share at least len(pattern) prefix characters, so
 * min(lcp[lo+1 .. hi-1]) >= len(pattern).
 *
 * Checks: "banana" anchors, then LCG texts where Locate (and its interval width) agree with a
 * naive scan for substrings, mutated and absent patterns. The empty pattern is documented to
 * match every position.
 */
import java.util.*;

class SuffixTreeIndex {
    final String text;
    final int[] sa;
    final int[] lcp;

    SuffixTreeIndex(String text) {
        this.text = text;
        this.sa = suffixArrayDoubling(text);
        this.lcp = kasai(text, sa);
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

    static int[] kasai(String s, int[] sa) {
        int n = s.length();
        int[] result = new int[n];
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
            result[r] = h;
            if (h > 0) h--;
        }
        return result;
    }

    int cmp(int start, String pattern) {
        for (int i = 0; i < pattern.length(); i++) {
            if (start + i >= text.length()) return -1;
            char ch = text.charAt(start + i);
            if (ch != pattern.charAt(i)) return ch < pattern.charAt(i) ? -1 : 1;
        }
        return 0;
    }

    int[] interval(String pattern) {
        if (pattern.isEmpty()) return new int[]{0, sa.length};
        int lo = 0, hi = sa.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (cmp(sa[mid], pattern) < 0) lo = mid + 1;
            else hi = mid;
        }
        int lower = lo;
        hi = sa.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (cmp(sa[mid], pattern) <= 0) lo = mid + 1;
            else hi = mid;
        }
        return new int[]{lower, lo};
    }

    int[] locate(String pattern) {
        int[] range = interval(pattern);
        int[] positions = Arrays.copyOfRange(sa, range[0], range[1]);
        Arrays.sort(positions);
        return positions;
    }

    int count(String pattern) {
        int[] range = interval(pattern);
        return range[1] - range[0];
    }

    int locusSharedPrefix(String pattern) {
        int[] range = interval(pattern);
        if (range[1] - range[0] <= 1) return 0;
        int best = Integer.MAX_VALUE;
        for (int i = range[0] + 1; i < range[1]; i++) best = Math.min(best, lcp[i]);
        return best;
    }

    static int[] naiveLocate(String text, String pattern) {
        List<Integer> positions = new ArrayList<>();
        if (pattern.isEmpty()) {
            for (int i = 0; i < text.length(); i++) positions.add(i);
            return toArray(positions);
        }
        for (int i = 0; i + pattern.length() <= text.length(); i++) {
            if (text.startsWith(pattern, i)) positions.add(i);
        }
        return toArray(positions);
    }

    private static int[] toArray(List<Integer> values) {
        int[] out = new int[values.size()];
        for (int i = 0; i < out.length; i++) out[i] = values.get(i);
        return out;
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

    static String join(int[] values) {
        StringBuilder sb = new StringBuilder();
        for (int value : values) {
            if (sb.length() > 0) sb.append(",");
            sb.append(value);
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

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static void checkIndex(String text, List<String> extraPatterns) {
        SuffixTreeIndex index = new SuffixTreeIndex(text);
        check(SuffixTreeIndex.join(index.sa).equals(SuffixTreeIndex.join(SuffixTreeIndex.suffixArrayNaive(text))),
                "sa " + text);
        List<String> patterns = new ArrayList<>(extraPatterns);
        for (int i = 0; i < text.length(); i++)
            for (int j = i + 1; j <= Math.min(text.length(), i + 5); j++)
                patterns.add(text.substring(i, j));
        patterns.addAll(Arrays.asList("zz", "qq", "!"));
        for (String pattern : patterns) {
            int[] positions = index.locate(pattern);
            check(SuffixTreeIndex.join(positions).equals(SuffixTreeIndex.join(SuffixTreeIndex.naiveLocate(text, pattern))),
                    "locate " + text + "/" + pattern);
            check(index.count(pattern) == positions.length, "count " + pattern);
            int[] range = index.interval(pattern);
            for (int k = range[0]; k < range[1]; k++) {
                int start = index.sa[k];
                if (!pattern.isEmpty()) check(text.startsWith(pattern, start), "inside " + pattern);
            }
            if (!pattern.isEmpty() && range[0] > 0) {
                int prev = index.sa[range[0] - 1];
                check(!(prev + pattern.length() <= text.length() && text.startsWith(pattern, prev)), "left " + pattern);
            }
            if (!pattern.isEmpty() && range[1] < index.sa.length) {
                int next = index.sa[range[1]];
                check(!(next + pattern.length() <= text.length() && text.startsWith(pattern, next)), "right " + pattern);
            }
            if (!pattern.isEmpty() && range[1] - range[0] >= 2)
                check(index.locusSharedPrefix(pattern) >= pattern.length(), "locus " + pattern);
        }
    }

    public static void main(String[] args) {
        checkIndex("banana", Arrays.asList("ana", "na", "a", "banana", "anana", "bananas", "nan"));
        SuffixTreeIndex banana = new SuffixTreeIndex("banana");
        check(SuffixTreeIndex.join(banana.locate("ana")).equals("1,3"), "ana");
        check(SuffixTreeIndex.join(banana.locate("na")).equals("2,4"), "na");
        check(SuffixTreeIndex.join(banana.locate("a")).equals("1,3,5"), "a");
        check(SuffixTreeIndex.join(banana.locate("banana")).equals("0"), "banana");
        check(SuffixTreeIndex.join(banana.locate("anana")).equals("1"), "anana");
        check(SuffixTreeIndex.join(banana.locate("")).equals("0,1,2,3,4,5"), "empty");
        check(SuffixTreeIndex.join(banana.locate("nana")).equals("2"), "nana");
        check(banana.locate("bananas").length == 0, "bananas");
        check(SuffixTreeIndex.join(banana.interval("ana")).equals("1,3"), "ana interval");
        check(banana.locusSharedPrefix("ana") >= 3, "ana locus");

        checkIndex("mississippi", Arrays.asList("issi", "iss", "ssi", "mississippi", "ppi", "sipp"));
        checkIndex("", Arrays.asList("a"));
        checkIndex("a", Arrays.asList("a", "aa"));
        checkIndex("aaaa", Arrays.asList("aaa", "aa", "a", "aaaaa"));

        for (int trial = 0; trial < 30; trial++) {
            String text = lcgString(30, "abc");
            List<String> mutations = new ArrayList<>();
            for (int k = 0; k < 6; k++) mutations.add(text.substring(0, k) + "d" + text.substring(k + 1));
            checkIndex(text, mutations);
        }
        for (int trial = 0; trial < 10; trial++) {
            String text = lcgString(60, "ab");
            checkIndex(text, Arrays.asList(text, text.substring(0, text.length() - 1), "z" + text));
        }
        System.out.println("PASS");
    }
}
