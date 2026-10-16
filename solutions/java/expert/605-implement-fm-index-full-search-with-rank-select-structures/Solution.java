/** Task 605 - Implement FM-index full search with rank/select structures. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * BitVector - 512-bit superblock rank checkpoints plus popcounts of the remaining whole words and one
 * masked partial word, so rank1(i) is O(1) after the checkpoints; select1(k) binary searches the
 * checkpoints and then walks words, dropping the lowest set bit with word &= word - 1.
 * FMIndex - the BWT of text + sentinel with one BitVector per alphabet character: LF(i) =
 * C[c] + rank(c, i), backward search over [start, end] counts a pattern in O(m log sigma), and
 * locating enumerates the interval then walks LF to the nearest sampled suffix array entry.
 *
 * Verified: rank1/select1 against brute force on structured and pseudo-random bit vectors (empty,
 * all-zero, all-one, single bit) plus out-of-range k; LF against the brute-force LF derived from the
 * full suffix array; select1(c, rank1(c, i) + 1) == i for every BWT row; and count / locateAll /
 * firstOccurrence / kthOccurrence against naive scanning including empty text, patterns longer than
 * the text and absent patterns.
 */
import java.util.*;

class BitVector {
    final int length;
    final long[] words;
    private int[] checkpoints;            // ones before each 512-bit superblock, plus the total

    BitVector(int length) {
        this.length = length;
        words = new long[(length + 63) >>> 6];
    }

    void set(int i) { words[i >>> 6] |= 1L << (i & 63); }

    /** Build the superblock checkpoints; must be called once after all set() calls. */
    void build() {
        int blocks = (length + 511) >>> 9;
        checkpoints = new int[blocks + 1];
        for (int b = 0; b < blocks; b++) {
            int end = Math.min((b + 1) << 9, length);
            int count = 0;
            for (int i = b << 9; i < end; i++) if ((words[i >>> 6] & (1L << (i & 63))) != 0) count++;
            checkpoints[b + 1] = checkpoints[b] + count;
        }
    }

    int count1() { return checkpoints[checkpoints.length - 1]; }

    /** Number of set bits in [0, i). */
    int rank1(int i) {
        if (i <= 0) return 0;
        if (i > length) i = length;
        int count = checkpoints[i >>> 9];
        int wordStart = (i >>> 9) << 3;
        for (int w = wordStart; w < (i >>> 6); w++) count += Long.bitCount(words[w]);
        if ((i & 63) != 0) count += Long.bitCount(words[i >>> 6] & ((1L << (i & 63)) - 1));
        return count;
    }

    /** Position of the k-th set bit (k is 1-based), or -1 when k is outside [1, count1]. */
    int select1(int k) {
        if (k < 1 || k > count1()) return -1;
        int lo = 0, hi = checkpoints.length - 1, block = 0;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            if (checkpoints[mid] < k) { block = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        int remaining = k - checkpoints[block];
        for (int w = block << 3; w < words.length; w++) {
            int bits = Long.bitCount(words[w]);
            if (remaining > bits) { remaining -= bits; continue; }
            long word = words[w];
            while (remaining > 1) { word &= word - 1; remaining--; }
            return (w << 6) + Long.numberOfTrailingZeros(word);
        }
        return -1;
    }
}

class FMIndex {
    static final char SENTINEL = '\u0001';
    static final int SAMPLE_STEP = 8;

    final int m;
    final char[] bwt;
    final int[] codeTable;
    final int sigma;
    final int[] cArray;
    final int[] sampleValue;
    final boolean[] sampledRank;
    final int[] fullSuffixArray;
    long locateSteps;

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

    FMIndex(String text) {
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
        for (int c = 0; c < sigma; c++) {
            int count = cArray[c];
            cArray[c] = running;
            running += count;
        }
        charBit = new BitVector[sigma];
        for (int c = 0; c < sigma; c++) charBit[c] = new BitVector(m);
        for (int r = 0; r < m; r++) charBit[codeTable[bwt[r]]].set(r);
        for (int c = 0; c < sigma; c++) charBit[c].build();
        List<Integer> values = new ArrayList<>();
        sampledRank = new boolean[m];
        for (int r = 0; r < m; r++) {
            if (fullSuffixArray[r] % SAMPLE_STEP == 0) { sampledRank[r] = true; values.add(fullSuffixArray[r]); }
        }
        sampleValue = values.stream().mapToInt(Integer::intValue).toArray();
    }

    private final BitVector[] charBit;

    int rankOfCode(int code, int i) { return charBit[code].rank1(i); }

    int occCount(int code) { return charBit[code].count1(); }

    /** select1 on the BWT character bit vector, 1-based. */
    int selectCode(int code, int k) { return charBit[code].select1(k); }

    int lf(int row) {
        int code = codeTable[bwt[row]];
        return cArray[code] + rankOfCode(code, row);
    }

    /** Sample index of a sampled rank, found by counting sampled ranks before it. */
    int locateRow(int row) {
        int steps = 0;
        while (!sampledRank[row]) {
            row = lf(row);
            steps++;
        }
        int index = 0;
        for (int r = 0; r < row; r++) if (sampledRank[r]) index++;
        locateSteps += steps;
        return (sampleValue[index] + steps) % m;
    }

    /** [start, end] of suffixes starting with pattern (inclusive), or null. */
    int[] interval(String pattern) {
        if (pattern.isEmpty()) return null;
        int start = 0, end = m - 1;
        for (int i = pattern.length() - 1; i >= 0 && start <= end; i--) {
            int code = codeTable[pattern.charAt(i)];
            if (code < 0) return null;
            start = cArray[code] + rankOfCode(code, start);
            end = cArray[code] + rankOfCode(code, end + 1) - 1;
        }
        return start <= end ? new int[]{start, end} : null;
    }

    int count(String pattern) {
        int[] range = interval(pattern);
        return range == null ? 0 : range[1] - range[0] + 1;
    }

    /** All occurrences in ascending text order. */
    int[] locateAll(String pattern) {
        int[] range = interval(pattern);
        if (range == null) return new int[0];
        int[] positions = new int[range[1] - range[0] + 1];
        for (int row = range[0]; row <= range[1]; row++) positions[row - range[0]] = locateRow(row);
        Arrays.sort(positions);
        return positions;
    }

    int firstOccurrence(String pattern) {
        int[] all = locateAll(pattern);
        return all.length == 0 ? -1 : all[0];
    }

    int kthOccurrence(String pattern, int k) {
        int[] all = locateAll(pattern);
        return k <= 0 || k > all.length ? -1 : all[k - 1];
    }
}

public class Solution {
    private static long state = 0x605FACEB00CL;

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

    static BitVector fromBits(boolean[] bits) {
        BitVector vector = new BitVector(bits.length);
        for (int i = 0; i < bits.length; i++) if (bits[i]) vector.set(i);
        vector.build();
        return vector;
    }

    /** rank1/select1 must invert each other and match a direct scan on every position. */
    static void verifyBitVector(BitVector vector, boolean[] bits) {
        int total = 0;
        for (boolean bit : bits) if (bit) total++;
        check(vector.count1() == total, "count1");
        for (int i = 0; i <= bits.length; i++) {
            int expected = 0;
            for (int k = 0; k < i && k < bits.length; k++) if (bits[k]) expected++;
            check(vector.rank1(i) == expected, "rank1(" + i + ")");
        }
        for (int k = 1; k <= total; k++) {
            int position = vector.select1(k);
            check(position >= 0 && position < bits.length && bits[position]
                    && vector.rank1(position) == k - 1, "select1/rank1 inverse for k=" + k);
        }
        check(vector.select1(0) == -1 && vector.select1(total + 1) == -1, "select1 range checks");
    }

    static void verifyIndex(String text) {
        FMIndex fm = new FMIndex(text);
        int m = fm.m;
        int[] inverse = new int[m];
        for (int r = 0; r < m; r++) inverse[fm.fullSuffixArray[r]] = r;
        for (int r = 0; r < m; r++) {
            check(fm.lf(r) == inverse[(fm.fullSuffixArray[r] - 1 + m) % m], "LF vs brute force");
            int code = fm.codeTable[fm.bwt[r]];
            check(fm.selectCode(code, fm.rankOfCode(code, r) + 1) == r, "select1 inverts rank1 on the BWT");
        }
        for (int start = 0; start < text.length(); start++) {
            for (int end = start + 1; end <= Math.min(text.length(), start + 3); end++) {
                String pattern = text.substring(start, end);
                check(fm.count(pattern) == bruteOccurrences(text, pattern), "count " + pattern + " in " + text);
                check(Arrays.equals(fm.locateAll(pattern), brutePositions(text, pattern)),
                        "locate " + pattern + " in " + text);
                check(fm.firstOccurrence(pattern) == text.indexOf(pattern), "first " + pattern);
            }
        }
        check(fm.count("") == 0 && fm.firstOccurrence("") == -1, "empty pattern rejected");
        check(fm.count("zzz") == 0 && fm.count(text + "z") == 0, "absent and overlong pattern");
        int sum = 0;
        for (int code = 0; code < fm.sigma; code++) sum += fm.occCount(code);
        check(sum == m, "character counts cover the BWT");
    }

    public static void main(String[] args) {
        boolean[][] structured = {new boolean[0], new boolean[]{true}, new boolean[]{false}, new boolean[64],
                                  new boolean[65], new boolean[]{true, false, true, true, false, false, false, true}};
        for (boolean[] bits : structured) {
            if (bits.length == 64 || bits.length == 65) Arrays.fill(bits, true);
            verifyBitVector(fromBits(bits), bits);
        }
        for (int trial = 0; trial < 12; trial++) {
            boolean[] bits = new boolean[1 + (int) ((nextLong() >>> 40) % 1500)];
            for (int i = 0; i < bits.length; i++) bits[i] = (nextLong() >>> 33) % 3 == 0;
            verifyBitVector(fromBits(bits), bits);
        }

        FMIndex empty = new FMIndex("");
        check(empty.count("a") == 0 && empty.locateAll("a").length == 0 && new FMIndex("a").count("a") == 1,
                "empty text and single char");
        FMIndex banana = new FMIndex("banana");
        check(banana.count("ana") == 2 && banana.count("banana") == 1 && banana.count("bn") == 0, "banana counts");
        check(Arrays.equals(banana.locateAll("ana"), new int[]{1, 3}), "banana locate");
        check(Arrays.equals(banana.locateAll("a"), new int[]{1, 3, 5}), "banana all a");
        check(banana.kthOccurrence("a", 2) == 3 && banana.kthOccurrence("a", 4) == -1, "banana kth");
        check(banana.firstOccurrence("na") == 2 && banana.firstOccurrence("zz") == -1, "banana first");
        check(banana.count("bananaz") == 0, "banana overlong miss");

        for (String text : new String[]{"", "a", "aa", "ab", "aba", "abab", "banana", "mississippi",
                                        "abracadabra", "aaabaaaab", "abcabcabc", "mississippiissippi"}) {
            verifyIndex(text);
        }
        for (String alphabet : new String[]{"ab", "abc", "abcdef"}) {
            for (int trial = 0; trial < 8; trial++) verifyIndex(randomText(70, alphabet));
        }

        String large = randomText(3000, "ab");
        FMIndex largeIndex = new FMIndex(large);
        for (int i = 0; i < 40; i++) {
            int from = (int) ((nextLong() >>> 33) % (large.length() - 20));
            String pattern = large.substring(from, from + 12);
            check(largeIndex.count(pattern) == bruteOccurrences(large, pattern), "large count at " + from);
            check(largeIndex.firstOccurrence(pattern) == large.indexOf(pattern), "large first at " + from);
        }
        int seen = 0;
        for (char c : new char[]{'a', 'b'}) seen += largeIndex.count(String.valueOf(c));
        check(seen == large.length(), "large character counts sum to the text length");
        for (int r = 0; r < largeIndex.m; r += 371) {
            int code = largeIndex.codeTable[largeIndex.bwt[r]];
            check(largeIndex.selectCode(code, largeIndex.rankOfCode(code, r) + 1) == r, "large select/rank");
        }
        System.out.println("bit vector bits     : " + (largeIndex.sigma * largeIndex.m) + " bits, "
                + largeIndex.sampleValue.length + " SA samples, " + largeIndex.locateSteps + " LF steps");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
