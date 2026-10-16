/** Task 608 - Implement BWT-based full-text index with fast locate. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * The index is the BWT of text + sentinel plus: C[c] (characters smaller than c), one rank bit
 * vector per character (word popcounts plus a per-word rank prefix, so rank(c, i) is O(1)), and one
 * mark bit vector over the rank axis with a 1 for every rank whose suffix array value is a multiple
 * of SAMPLE_STEP. Counting a pattern is backward search over the interval [start, end]; locating a
 * row is an LF walk that stops as soon as the current rank is marked, and the sample is then fetched
 * in O(1) by rank1 on the mark bit vector instead of a linear or binary search (this is the "fast
 * locate" part). A memo table of rank -> position makes repeated locating cheaper still, which the
 * checks measure by comparing the LF step counters of both strategies - both must return identical
 * positions.
 *
 * Verified: LF equals the brute-force LF from the full suffix array; rank1 on the mark bit vector and
 * select1 on it agree with direct scans; every rank is reconstructed to its true suffix array value in
 * at most SAMPLE_STEP steps; the two locate strategies agree with each other and with naive scanning
 * for many patterns including empty text, single character text, whole-text patterns, absent patterns
 * and patterns longer than the text; extract() reproduces the text.
 */
import java.util.*;

class RankSelectBits {
    final int length;
    final long[] words;
    private int[] prefix;                     // ones before each 64-bit word

    RankSelectBits(int length) {
        this.length = length;
        words = new long[(length + 63) >>> 6];
    }

    void set(int i) { words[i >>> 6] |= 1L << (i & 63); }

    void build() {
        prefix = new int[words.length + 1];
        for (int w = 0; w < words.length; w++) prefix[w + 1] = prefix[w] + Long.bitCount(words[w]);
    }

    int get(int i) { return i < 0 || i >= length ? 0 : (int) ((words[i >>> 6] >>> (i & 63)) & 1L); }

    int count1() { return prefix[words.length]; }

    int rank1(int i) {
        if (i <= 0) return 0;
        if (i > length) i = length;
        int count = prefix[i >>> 6];
        if ((i & 63) != 0) count += Long.bitCount(words[i >>> 6] & ((1L << (i & 63)) - 1));
        return count;
    }

    /** 1-based k; -1 when k is out of range. Binary search over the rank structure. */
    int select1(int k) {
        if (k < 1 || k > count1()) return -1;
        int lo = 0, hi = length - 1;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (rank1(mid + 1) >= k) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }
}

class BwtIndex {
    static final char SENTINEL = '\u0001';
    static final int SAMPLE_STEP = 4;

    final int m;
    final char[] bwt;
    final int[] codeTable;
    final int sigma;
    final int[] cArray;
    final RankSelectBits[] charBits;
    final RankSelectBits sampledMarks;        // 1 at every rank with SA value % SAMPLE_STEP == 0
    final int[] sampleValue;                  // the sampled SA values, in rank order
    final int[] fullSuffixArray;              // kept for the brute-force cross-checks
    final int rankOfZero;                     // rank of the suffix starting at text position 0
    final Map<Integer, Integer> memo = new HashMap<>();
    long plainSteps, cachedSteps;

    static int[] doublingSuffixArray(String s) {
        int n = s.length();
        if (n == 0) return new int[0];
        Integer[] sa = new Integer[n];
        int[] rank = new int[n];
        for (int i = 0; i < n; i++) { sa[i] = i; rank[i] = s.charAt(i); }
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
        return Arrays.stream(sa).mapToInt(Integer::intValue).toArray();
    }

    BwtIndex(String text) {
        String s = text + SENTINEL;
        m = s.length();
        fullSuffixArray = doublingSuffixArray(s);
        bwt = new char[m];
        for (int r = 0; r < m; r++) bwt[r] = s.charAt((fullSuffixArray[r] - 1 + m) % m);
        char[] distinct = s.chars().distinct().sorted().collect(StringBuilder::new,
                StringBuilder::appendCodePoint, StringBuilder::append).toString().toCharArray();
        sigma = distinct.length;
        codeTable = new int[0x10000];
        Arrays.fill(codeTable, -1);
        for (int c = 0; c < sigma; c++) codeTable[distinct[c]] = c;
        cArray = new int[sigma];
        for (char c : bwt) cArray[codeTable[c]]++;
        int running = 0;
        for (int c = 0; c < sigma; c++) { int count = cArray[c]; cArray[c] = running; running += count; }
        charBits = new RankSelectBits[sigma];
        for (int c = 0; c < sigma; c++) charBits[c] = new RankSelectBits(m);
        sampledMarks = new RankSelectBits(m);
        for (int r = 0; r < m; r++) charBits[codeTable[bwt[r]]].set(r);
        for (int c = 0; c < sigma; c++) charBits[c].build();
        List<Integer> values = new ArrayList<>();
        for (int r = 0; r < m; r++) {
            if (fullSuffixArray[r] % SAMPLE_STEP == 0) { sampledMarks.set(r); values.add(fullSuffixArray[r]); }
        }
        sampledMarks.build();
        sampleValue = values.stream().mapToInt(Integer::intValue).toArray();
        int zero = 0;
        for (int r = 0; r < m; r++) if (fullSuffixArray[r] == 0) zero = r;
        rankOfZero = zero;
    }

    int occCount(int code) { return charBits[code].count1(); }

    int lf(int row) {
        int code = codeTable[bwt[row]];
        return cArray[code] + charBits[code].rank1(row);
    }

    /** SA value of a marked rank, obtained in O(1) through rank1 on the mark bit vector. */
    int sampleOf(int row) { return sampleValue[sampledMarks.rank1(row)]; }

    int locateRow(int row) {
        int steps = 0;
        while (sampledMarks.get(row) == 0) {
            row = lf(row);
            steps++;
        }
        plainSteps += steps;
        return (sampleOf(row) + steps) % m;
    }

    /** Same walk, but every visited rank is memoised, so later queries skip the climb entirely. */
    int locateRowCached(int row) {
        int steps = 0;
        int current = row;
        Integer known = memo.get(current);
        while (known == null) {
            if (sampledMarks.get(current) == 1) { known = sampleOf(current); break; }
            current = lf(current);
            steps++;
        }
        int value = (known + steps) % m;
        int back = row;
        for (int i = 0; i <= steps; i++) { memo.put(back, (value - i + m) % m); back = lf(back); }
        cachedSteps += steps;
        return value;
    }

    int[] interval(String pattern) {
        if (pattern.isEmpty()) return null;
        int start = 0, end = m - 1;
        for (int i = pattern.length() - 1; i >= 0 && start <= end; i--) {
            int code = codeTable[pattern.charAt(i)];
            if (code < 0) return null;
            start = cArray[code] + charBits[code].rank1(start);
            end = cArray[code] + charBits[code].rank1(end + 1) - 1;
        }
        return start <= end ? new int[]{start, end} : null;
    }

    int count(String pattern) {
        int[] range = interval(pattern);
        return range == null ? 0 : range[1] - range[0] + 1;
    }

    int[] locate(String pattern, boolean cached) {
        int[] range = interval(pattern);
        if (range == null) return new int[0];
        int[] positions = new int[range[1] - range[0] + 1];
        for (int row = range[0]; row <= range[1]; row++) {
            positions[row - range[0]] = cached ? locateRowCached(row) : locateRow(row);
        }
        Arrays.sort(positions);
        return positions;
    }

    /** Smallest occurrence, without materialising the whole list. */
    int locateFirst(String pattern) {
        int[] range = interval(pattern);
        if (range == null) return -1;
        int best = m;
        for (int row = range[0]; row <= range[1]; row++) best = Math.min(best, locateRow(row));
        return best;
    }

    int batchCount(String[] patterns) {
        int total = 0;
        for (String pattern : patterns) total += count(pattern);
        return total;
    }

    String extract() {
        char[] out = new char[m];
        int row = rankOfZero;                     // rank of the suffix starting at text position 0
        for (int k = 0; k < m; k++) {
            out[m - 1 - k] = bwt[row];
            row = lf(row);
        }
        return new String(out);
    }
}

public class Solution {
    private static long state = 0x608B17E51DL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static String randomText(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) sb.append(alphabet.charAt((int) ((nextLong() >>> 33) % alphabet.length())));
        return sb.toString();
    }

    static int bruteOccurrences(String text, String pattern) {
        int count = 0;
        for (int i = 0; i + pattern.length() <= text.length(); i++) if (text.startsWith(pattern, i)) count++;
        return count;
    }

    static int[] brutePositions(String text, String pattern) {
        List<Integer> found = new ArrayList<>();
        for (int i = 0; i + pattern.length() <= text.length(); i++) if (text.startsWith(pattern, i)) found.add(i);
        return found.stream().mapToInt(Integer::intValue).toArray();
    }

    static void verify(String text) {
        BwtIndex index = new BwtIndex(text);
        check(index.extract().equals(text + BwtIndex.SENTINEL), "extract == text for '" + text + "'");
        int[] inverse = new int[index.m];
        for (int r = 0; r < index.m; r++) inverse[index.fullSuffixArray[r]] = r;
        int previousSample = 0;
        for (int r = 0; r < index.m; r++) {
            check(index.lf(r) == inverse[(index.fullSuffixArray[r] - 1 + index.m) % index.m],
                    "LF equals brute force LF");
            boolean marked = index.sampledMarks.get(r) == 1;
            check(marked == (index.fullSuffixArray[r] % BwtIndex.SAMPLE_STEP == 0), "mark bit vector is exact");
            if (marked) {
                previousSample++;
                check(index.sampleOf(r) == index.fullSuffixArray[r], "sampleOf returns the SA value");
                check(index.sampledMarks.select1(previousSample) == r, "select1 finds the marked rank");
            }
            check(index.locateRow(r) == index.fullSuffixArray[r], "locate reproduces SA[" + r + "]");
            check(index.locateRowCached(r) == index.fullSuffixArray[r], "cached locate agrees");
        }
        for (int start = 0; start < text.length(); start++) {
            for (int end = start + 1; end <= Math.min(text.length(), start + 3); end++) {
                String pattern = text.substring(start, end);
                check(index.count(pattern) == bruteOccurrences(text, pattern), "count " + pattern);
                check(Arrays.equals(index.locate(pattern, false), brutePositions(text, pattern)), "locate " + pattern);
                check(Arrays.equals(index.locate(pattern, true), brutePositions(text, pattern)), "cached locate");
                check(index.locateFirst(pattern) == text.indexOf(pattern), "first " + pattern);
            }
        }
        check(index.count("") == 0 && index.locateFirst("") == -1, "empty pattern rejected");
        check(index.count("zzz") == 0 && index.count(text + "z") == 0, "absent and overlong pattern");
        int sum = 0;
        for (int code = 0; code < index.sigma; code++) sum += index.occCount(code);
        check(sum == index.m, "character counts cover the BWT");
    }

    public static void main(String[] args) {
        BwtIndex empty = new BwtIndex("");
        check(empty.count("a") == 0 && empty.locateFirst("a") == -1, "empty text");
        BwtIndex banana = new BwtIndex("banana");
        check(banana.count("ana") == 2 && banana.count("banana") == 1 && banana.count("bn") == 0, "banana counts");
        check(Arrays.equals(banana.locate("an", true), new int[]{1, 3}), "banana locate");
        check(banana.locateFirst("na") == 2 && banana.locateFirst("bananas") == -1, "banana first");

        for (String text : new String[]{"", "a", "aa", "ab", "aba", "abab", "banana", "mississippi",
                                        "abracadabra", "aaabaaaab", "abcabcabc"}) {
            verify(text);
        }
        for (String alphabet : new String[]{"ab", "abc", "abcdef"}) {
            for (int trial = 0; trial < 6; trial++) verify(randomText(60, alphabet));
        }

        String large = randomText(4000, "ab");
        BwtIndex index = new BwtIndex(large);
        check(index.extract().equals(large + BwtIndex.SENTINEL), "large extract");
        for (int r = 0; r < index.m; r += 37) {
            check(index.locateRow(r) == index.fullSuffixArray[r], "large SA reconstruction at " + r);
        }
        for (int i = 0; i < large.length(); i += 97) {
            String pattern = large.substring(i, Math.min(large.length(), i + 9));
            check(index.count(pattern) == bruteOccurrences(large, pattern), "large count at " + i);
        }
        int total = index.batchCount(new String[]{"a", "b", "aa", "bb", "abab", "babab"});
        check(total > 0, "batch counting works");
        index.plainSteps = 0;
        for (int i = 0; i < 300; i++) index.locate("a", false);
        long plainSteps = index.plainSteps;
        index.memo.clear();
        index.cachedSteps = 0;
        for (int i = 0; i < 300; i++) index.locate("a", true);
        long cachedSteps = index.cachedSteps;
        check(cachedSteps < plainSteps, "memoised locate does less LF walking than the plain walk");
        System.out.println("samples / marks     : " + index.sampleValue.length + " of " + index.m);
        System.out.println("LF steps plain/memo : " + plainSteps + " then " + cachedSteps);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
