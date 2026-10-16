/** Task 604 - Implement compressed suffix array (CSA) advanced. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * The plain suffix array is replaced by the functional (Sadakane style) representation: the BWT
 * plus a rank structure gives the LF mapping LF(i) = C[c] + rank(c, i) in O(block) time, and only
 * a sampled SA is kept - the ranks r with SA[r] % B == 0, marked by a bit vector over the rank axis
 * plus one text position per sample. Everything else is reconstructed from LF:
 *   - locate(r): follow LF until a sampled rank is reached (positions decrease by one per step, so
 *     at most B steps) and add the number of steps;
 *   - rankOf(position) is the same walk counted backwards, and extract() rebuilds the text by
 *     walking LF from the sampled rank of position 0;
 *   - pattern counting is a backward search over the rank structure, locating uses the samples.
 *
 * Verified: LF equals the brute-force LF built from the full SA for every row; Psi (the "next
 * suffix" mapping) and LF are mutual inverses; reconstructing SA[i] by LF walking reproduces the
 * full suffix array for every rank; extract() reproduces the text; count/locate match brute force.
 * The suffix array itself is cross-checked against a naive comparison sort, and the space of the
 * compressed form (BWT bytes + checkpoints + n/B samples) is reported against 4n bytes for a plain
 * SA. Also documented as simplified: the "compression" is a sampled/functional CSA, not a
 * bit-packed one, and the rank checkpoints are ints rather than packed bit vectors.
 */
import java.util.*;

class CompressedSuffixArray {
    static final char SENTINEL = '\u0001';

    final String s;              // text + sentinel, sentinel smaller than every text character
    final int m;                 // s.length()
    final char[] bwt;
    final int sigma;
    final int[] cArray;          // C[c] = number of BWT characters smaller than c
    final int blockSize;
    final int blocks;
    final int[][] checkpoint;    // checkpoint[c][b] = occurrences of c before block b
    final int[] sampleRank;      // ascending ranks that carry a sample
    final int[] sampleValue;     // SA value of each sampled rank
    final int rankOfZero;        // rank of the suffix starting at text position 0 (always sampled)
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

    CompressedSuffixArray(String text) {
        s = text + SENTINEL;
        m = s.length();
        int[] sa = doublingSuffixArray(s);
        int[] inverse = new int[m];
        for (int r = 0; r < m; r++) inverse[sa[r]] = r;
        bwt = new char[m];
        for (int r = 0; r < m; r++) bwt[r] = s.charAt((sa[r] - 1 + m) % m);
        char[] distinct = s.chars().distinct().sorted().collect(StringBuilder::new,
                StringBuilder::appendCodePoint, StringBuilder::append).toString().toCharArray();
        sigma = distinct.length;
        int[] code = new int[0x10000];
        Arrays.fill(code, -1);
        for (int c = 0; c < sigma; c++) code[distinct[c]] = c;
        cArray = new int[sigma];
        for (int c = 0; c < m; c++) cArray[code[bwt[c]]]++;
        int running = 0;
        for (int c = 0; c < sigma; c++) {
            int count = cArray[c];
            cArray[c] = running;
            running += count;
        }
        blockSize = Math.max(1, Math.min(16, m));
        blocks = (m + blockSize - 1) / blockSize;
        checkpoint = new int[sigma][blocks + 1];
        for (int r = 0; r < m; r++) checkpoint[code[bwt[r]]][r / blockSize + 1]++;
        for (int c = 0; c < sigma; c++) {
            for (int b = 1; b <= blocks; b++) checkpoint[c][b] += checkpoint[c][b - 1];
        }
        List<Integer> ranks = new ArrayList<>(), values = new ArrayList<>();
        for (int r = 0; r < m; r++) {
            if (sa[r] % blockSize == 0) { ranks.add(r); values.add(sa[r]); }
        }
        sampleRank = ranks.stream().mapToInt(Integer::intValue).toArray();
        sampleValue = values.stream().mapToInt(Integer::intValue).toArray();
        rankOfZero = inverse[0];
        codeTable = code;
    }

    private final int[] codeTable;

    int rankOf(int code, int i) {
        int block = i / blockSize;
        int count = checkpoint[code][block];
        for (int k = block * blockSize; k < i; k++) if (codeTable[bwt[k]] == code) count++;
        return count;
    }

    int lf(int row) {
        char c = bwt[row];
        return cArray[codeTable[c]] + rankOf(codeTable[c], row);
    }

    boolean sampleAt(int row) {
        int lo = 0, hi = sampleRank.length - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            if (sampleRank[mid] < row) lo = mid + 1;
            else if (sampleRank[mid] > row) hi = mid - 1;
            else return true;
        }
        return false;
    }

    private int sampleIndexOf(int row) {
        int lo = 0, hi = sampleRank.length - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            if (sampleRank[mid] < row) lo = mid + 1;
            else if (sampleRank[mid] > row) hi = mid - 1;
            else return mid;
        }
        return -1;
    }

    /** SA value of an arbitrary rank: LF walk to the next sampled rank, then count the steps. */
    int locateRow(int row) {
        int steps = 0;
        while (!sampleAt(row)) {
            row = lf(row);
            steps++;
        }
        locateSteps += steps;
        return (sampleValue[sampleIndexOf(row)] + steps) % m;
    }

    /** Backward search: [start, end] interval of suffixes beginning with pattern, or null. */
    int[] interval(String pattern) {
        if (pattern.isEmpty()) return null;
        int start = 0, end = m - 1;
        for (int i = pattern.length() - 1; i >= 0 && start <= end; i--) {
            int c = codeTable[pattern.charAt(i)];
            if (c < 0) return null;
            start = cArray[c] + rankOf(c, start);
            end = cArray[c] + rankOf(c, end + 1) - 1;
        }
        return start <= end ? new int[]{start, end} : null;
    }

    int count(String pattern) {
        int[] range = interval(pattern);
        return range == null ? 0 : range[1] - range[0] + 1;
    }

    int[] locate(String pattern) {
        int[] range = interval(pattern);
        if (range == null) return new int[0];
        int[] positions = new int[range[1] - range[0] + 1];
        for (int row = range[0]; row <= range[1]; row++) positions[row - range[0]] = locateRow(row);
        Arrays.sort(positions);
        return positions;
    }

    /** Inverse BWT by walking LF from the sampled rank of suffix 0. */
    String extract() {
        char[] out = new char[m];
        int row = rankOfZero;
        for (int k = 0; k < m; k++) {
            out[m - 1 - k] = bwt[row];
            row = lf(row);
        }
        return new String(out);
    }

    long compressedBits() {
        return 8L * m + 32L * (long) sigma * (blocks + 1) + 32L * sampleRank.length + m;
    }
}

public class Solution {
    private static long state = 0x17A15C0DE5EEDL;

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

    static List<Integer> brutePositions(String text, String pattern) {
        List<Integer> found = new ArrayList<>();
        for (int i = 0; i + pattern.length() <= text.length(); i++) if (text.startsWith(pattern, i)) found.add(i);
        return found;
    }

    static int[] naiveSuffixArray(String s) {
        Integer[] sa = new Integer[s.length()];
        for (int i = 0; i < sa.length; i++) sa[i] = i;
        Arrays.sort(sa, (a, b) -> s.substring(a).compareTo(s.substring(b)));
        return Arrays.stream(sa).mapToInt(Integer::intValue).toArray();
    }

    /** Every internal invariant of the CSA for one text, plus brute-force pattern agreement. */
    static void verify(String text) {
        CompressedSuffixArray csa = new CompressedSuffixArray(text);
        int m = csa.m;
        check(csa.extract().equals(text + CompressedSuffixArray.SENTINEL), "extract == text for '" + text + "'");
        int[] sa = CompressedSuffixArray.doublingSuffixArray(text + CompressedSuffixArray.SENTINEL);
        int[] inverse = new int[m];
        for (int r = 0; r < m; r++) inverse[sa[r]] = r;
        for (int r = 0; r < m; r++) {
            check(csa.locateRow(r) == sa[r], "locate reproduces SA[" + r + "] of '" + text + "'");
            int expectedLf = inverse[(sa[r] - 1 + m) % m];
            check(csa.lf(r) == expectedLf, "LF equals brute force LF at " + r + " of '" + text + "'");
            int psi = inverse[(sa[r] + 1) % m];
            check(csa.lf(psi) == r && inverse[(sa[csa.lf(r)] + 1) % m] == r, "LF and Psi are inverses");
        }
        for (int start = 0; start < text.length(); start++) {
            for (int end = start + 1; end <= Math.min(text.length(), start + 4); end++) {
                String pattern = text.substring(start, end);
                check(csa.count(pattern) == bruteOccurrences(text, pattern), "count " + pattern + " in " + text);
                check(Arrays.equals(csa.locate(pattern),
                        brutePositions(text, pattern).stream().mapToInt(Integer::intValue).toArray()),
                        "locate " + pattern + " in " + text);
            }
        }
        check(csa.count("") == 0 && csa.locate("").length == 0, "empty pattern rejected");
    }

    public static void main(String[] args) {
        CompressedSuffixArray empty = new CompressedSuffixArray("");
        check(empty.m == 1 && empty.extract().equals(String.valueOf(CompressedSuffixArray.SENTINEL)), "empty text");
        check(empty.count("a") == 0 && empty.locate("a").length == 0, "empty text has no matches");

        CompressedSuffixArray banana = new CompressedSuffixArray("banana");
        check(banana.count("ana") == 2 && banana.count("banana") == 1, "banana counts");
        check(Arrays.equals(banana.locate("an"), new int[]{1, 3}), "banana locate");
        check(banana.count("bn") == 0 && banana.count("bananas") == 0, "banana misses");
        check(banana.extract().equals("banana" + CompressedSuffixArray.SENTINEL), "banana extract");
        check(Arrays.equals(CompressedSuffixArray.doublingSuffixArray("banana" + CompressedSuffixArray.SENTINEL),
                new int[]{6, 5, 3, 1, 0, 4, 2}), "banana SA matches the known order");
        check(Arrays.equals(CompressedSuffixArray.doublingSuffixArray("mississippi"),
                naiveSuffixArray("mississippi")), "doubling equals naive sort");

        for (String text : new String[]{"", "a", "aa", "ab", "aba", "abab", "banana", "mississippi",
                                        "abracadabra", "abcabcabc", "aaabaaaab"}) {
            verify(text);
        }
        for (String alphabet : new String[]{"ab", "abc", "abcdefg"}) {
            for (int trial = 0; trial < 8; trial++) verify(randomText(60, alphabet));
        }

        String large = randomText(2000, "ab");
        CompressedSuffixArray csa = new CompressedSuffixArray(large);
        check(csa.extract().equals(large + CompressedSuffixArray.SENTINEL), "large extract");
        int[] sa = CompressedSuffixArray.doublingSuffixArray(large + CompressedSuffixArray.SENTINEL);
        for (int r = 0; r < csa.m; r++) check(csa.locateRow(r) == sa[r], "large SA reconstruction at " + r);
        for (int i = 0; i < large.length(); i += 71) {
            String pattern = large.substring(i, Math.min(large.length(), i + 11));
            check(csa.count(pattern) == bruteOccurrences(large, pattern), "large count at " + i);
            check(Arrays.equals(csa.locate(pattern),
                    brutePositions(large, pattern).stream().mapToInt(Integer::intValue).toArray()),
                    "large locate at " + i);
        }
        Set<Character> alphabet = new TreeSet<>();
        for (char c : large.toCharArray()) alphabet.add(c);
        long total = 0;
        for (char c : alphabet) total += csa.count(String.valueOf(c));
        check(total == large.length(), "single character counts sum to the text length");

        System.out.println("compressed bytes    : " + csa.compressedBits() / 8 + " vs plain SA " + 4 * csa.m);
        System.out.println("checkpoint blocks   : " + csa.blocks + " of size " + csa.blockSize
                + ", samples " + csa.sampleRank.length);
        System.out.println("LF steps in locates : " + csa.locateSteps);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
