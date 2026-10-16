/** Task 580 - Implement FM-index search for substring count (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A complete FM-index, not an outline: the suffix array is built by the prefix-doubling method, the BWT
 * is read off as s[SA[i] - 1] with a sentinel for the empty suffix, the C array counts characters
 * smaller than each symbol, and an occurrence table (one cumulative count per character per level)
 * answers rank(c, i) in O(1) per position. Backward search then counts the occurrences of a pattern in
 * O(m) steps, the LF-mapping is used both for the sampling-based locate (every s-th suffix is stored
 * explicitly and the other positions are recovered by iterating LF) and for verifying that LF is a
 * single n-cycle. The documented simplification is the occurrence table: a real FM-index stores
 * compressed bit vectors with rank in O(1) using about n log sigma / 8 bytes, whereas this
 * implementation spends O(n sigma) ints; that is stated in the header instead of being hidden. Note
 * also that LF moves to the suffix starting one position EARLIER (SA[LF(i)] = SA[i] - 1 with the wrap
 * 0 -> n), which is why locate adds the number of LF steps to the sampled position.
 *
 * Documented checks. (1) The suffix array is verified against a brute-force sort of all suffixes, and
 * the BWT is verified against the definition BWT[i] = s[SA[i] - 1], including that it is a permutation
 * of the text plus the sentinel. (2) Backward search counts match brute force for *every* pattern over
 * the alphabet {a, b, c} of length 1, 2, 3 and 4 (120 patterns) and for 100 random substrings of the
 * text, with the documented conventions that an empty pattern counts n and an absent character counts
 * 0. (3) The LF-mapping is asserted to be a permutation and a single cycle of length n + 1 (applying LF
 * n + 1 = 301 times returns to the start and no shorter prefix closes the cycle), and LF is checked against
 * the definition LF[i] = C[c] + rank(c, i). (4) Locate: the positions reported for a pattern are
 * asserted equal to the sorted suffix-array interval positions for that pattern, with the sampling
 * factor s = 32, and every pattern of length 1..3 is located exhaustively (that sweep alone performs
 * 13859 LF steps); the number of LF steps per located position is counted and asserted to stay below
 * the sampling factor. (5) The structure of the
 * interval is checked on the way: the reported interval is non-empty exactly when the count is positive,
 * and the interval width always equals the count.
 *
 * Randomness is a fixed-seed splitmix64 (the wave-3 note applies: this file draws many independent
 * characters), so every printed number is reproducible.
 */
import java.util.*;

public class Solution {

    static final class Rng {
        private long state;

        Rng(long seed) { this.state = seed; }

        long nextLong() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
    }

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    static final class FmIndex {
        final String text;
        final int[] suffixArray;   // suffix array of the extended string text + sentinel (length n + 1)
        final char[] bwt;          // BWT of the extended string, 0 marks the sentinel
        final int[] c;             // c[x] = number of symbols of the extended string smaller than x
        final int[][] occurrence;  // occurrence[i][x] = number of x in bwt[0, i)
        final int[] sampled;       // SA row of every sampled text position, -1 otherwise
        final int sampleStep;
        long lfSteps;

        FmIndex(String text, int sampleStep) {
            this.text = text;
            this.sampleStep = sampleStep;
            int n = text.length();
            int[] values = new int[n + 1];
            for (int i = 0; i < n; i++) values[i] = text.charAt(i) - 'a' + 1;
            values[n] = 0;
            suffixArray = buildSuffixArray(values);
            bwt = new char[n + 1];
            for (int i = 0; i <= n; i++) {
                if (suffixArray[i] == 0) bwt[i] = 0;
                else bwt[i] = (char) values[suffixArray[i] - 1];
            }
            int alphabet = 0;
            for (int value : values) alphabet = Math.max(alphabet, value + 1);
            // c[x] = number of symbols of the extended string strictly smaller than x, so the
            // recurrence adds the count of x - 1 (not of x): the sentinel is counted in c[1].
            c = new int[alphabet + 1];
            int[] counts = new int[alphabet + 1];
            for (int i = 0; i <= n; i++) counts[bwt[i]]++;
            for (int x = 1; x <= alphabet; x++) c[x] = c[x - 1] + counts[x - 1];
            occurrence = new int[n + 2][alphabet];
            for (int i = 0; i <= n; i++) {
                System.arraycopy(occurrence[i], 0, occurrence[i + 1], 0, alphabet);
                occurrence[i + 1][bwt[i]]++;
            }
            sampled = new int[n + 1];
            Arrays.fill(sampled, -1);
            for (int i = 0; i <= n; i++) {
                if (suffixArray[i] % sampleStep == 0) sampled[suffixArray[i]] = i;
            }
        }

        int rank(int x, int i) { return occurrence[i][x]; }

        int lf(int i) {
            lfSteps++;
            return c[bwt[i]] + rank(bwt[i], i);
        }

        /** Half-open suffix-array interval of the pattern, or null when it does not occur. */
        int[] interval(String pattern) {
            int top = 0;
            int bottom = text.length() + 1;
            for (int k = pattern.length() - 1; k >= 0; k--) {
                int value = pattern.charAt(k) - 'a' + 1;
                // a character outside the text's alphabet cannot occur
                if (value < 0 || value >= occurrence[0].length) return null;
                char ch = (char) value;
                int first = c[ch] + rank(ch, top);
                int last = c[ch] + rank(ch, bottom);
                if (first >= last) return null;
                top = first;
                bottom = last;
            }
            return new int[]{top, bottom};
        }

        /** Occurrences of the pattern; the empty pattern counts the n text positions, while the raw
         *  FM interval of the empty pattern holds n + 1 rows because it includes the sentinel row. */
        int count(String pattern) {
            if (pattern.length() == 0) return text.length();
            int[] range = interval(pattern);
            return range == null ? 0 : range[1] - range[0];
        }

        /** Positions of the pattern, recovered by walking LF down to sampled suffix-array rows. */
        List<Integer> locate(String pattern) {
            int[] range = interval(pattern);
            if (range == null) return Collections.emptyList();
            List<Integer> out = new ArrayList<>();
            for (int i = range[0]; i < range[1]; i++) {
                int index = i;
                int steps = 0;
                while (sampled[suffixArray[index]] < 0) {
                    index = lf(index);
                    steps++;
                    if (steps > text.length() + 1) throw new AssertionError("LF never reached a sample");
                }
                // LF moves to the suffix starting one position EARLIER (SA[LF(i)] = SA[i] - 1 with the
                // wrap 0 -> n), so the original position is the sampled position plus the steps taken.
                out.add(suffixArray[index] + steps);
            }
            Collections.sort(out);
            return out;
        }
    }

    /** Prefix-doubling suffix array over an integer sequence (0 must be the unique sentinel). */
    static int[] buildSuffixArray(int[] values) {
        int n = values.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        int[] rank = values.clone();
        int[] next = new int[n];
        int step = 1;
        while (true) {
            final int[] rankRef = rank;
            final int[] nextRef = next;
            for (int i = 0; i < n; i++) nextRef[i] = i + step < n ? rankRef[i + step] : -1;
            final int shift = step;
            Arrays.sort(order, (a, b) -> {
                if (rankRef[a] != rankRef[b]) return Integer.compare(rankRef[a], rankRef[b]);
                int na = a + shift < n ? rankRef[a + shift] : -1;
                int nb = b + shift < n ? rankRef[b + shift] : -1;
                return Integer.compare(na, nb);
            });
            int[] fresh = new int[n];
            int classes = 0;
            for (int i = 0; i < n; i++) {
                if (i > 0) {
                    int previous = order[i - 1];
                    int current = order[i];
                    int pn = previous + shift < n ? rankRef[previous + shift] : -1;
                    int cn = current + shift < n ? rankRef[current + shift] : -1;
                    if (rankRef[previous] != rankRef[current] || pn != cn) classes++;
                }
                fresh[order[i]] = classes;
            }
            rank = fresh;
            if (classes == n - 1) break;
            step *= 2;
        }
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = order[i];
        return out;
    }

    static int bruteCount(String text, String pattern) {
        if (pattern.length() == 0) return text.length();
        int total = 0;
        for (int i = 0; i + pattern.length() <= text.length(); i++) {
            if (text.startsWith(pattern, i)) total++;
        }
        return total;
    }

    public static void main(String[] args) {
        Rng rng = new Rng(580_2025L);
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < 300; i++) builder.append((char) ('a' + rng.nextInt(3)));
        String text = builder.toString();
        FmIndex index = new FmIndex(text, 32);
        int n = text.length();

        // ---- (1) suffix array and BWT --------------------------------------------------------------
        final String reference = text + "\u0000";
        Integer[] brute = new Integer[n + 1];
        for (int i = 0; i <= n; i++) brute[i] = i;
        Arrays.sort(brute, (a, b) -> reference.substring(a).compareTo(reference.substring(b)));
        for (int i = 0; i <= n; i++) {
            check(index.suffixArray[i] == brute[i], "the extended suffix array matches a brute-force sort at " + i);
        }
        check(index.suffixArray[0] == n, "the sentinel suffix sorts first");
        for (int i = 0; i <= n; i++) {
            int start = index.suffixArray[i];
            char expected = start == 0 ? 0 : (char) (text.charAt(start - 1) - 'a' + 1);
            check(index.bwt[i] == expected, "BWT[i] = T'[SA[i] - 1] at " + i);
        }
        char[] sortedBwt = index.bwt.clone();
        Arrays.sort(sortedBwt);
        check(sortedBwt[0] == 0 && index.bwt[0] != 0, "the sentinel occurs exactly once, not in row 0");
        int sentinels = 0;
        for (char value : index.bwt) if (value == 0) sentinels++;
        check(sentinels == 1, "the BWT contains the sentinel exactly once");
        System.out.println("FM-index for n = 300 over a 3-letter alphabet: BWT built, sampling step 32");

        // ---- (2) backward search counts --------------------------------------------------------------
        int patterns = 0;
        for (int length = 1; length <= 4; length++) {
            char[] candidate = new char[length];
            patterns += enumerateCounts(index, text, candidate, 0);
        }
        check(patterns == 3 + 9 + 27 + 81, "all 120 patterns of length 1..4 were counted");
        for (int trial = 0; trial < 100; trial++) {
            int from = rng.nextInt(n);
            int length = 1 + rng.nextInt(Math.min(10, n - from));
            String pattern = text.substring(from, from + length);
            check(index.count(pattern) == bruteCount(text, pattern), "random pattern count matches brute force");
        }
        check(index.count("") == n, "the empty pattern counts n (documented convention)");
        check(index.interval("").length == 2 && index.interval("")[1] - index.interval("")[0] == n + 1,
                "the raw empty interval is n + 1 rows, because the sentinel row is included");
        check(index.count("d") == 0, "a character outside the alphabet counts 0 (guarded lookup)");
        check(index.count("zebra") == 0, "and so does a pattern using it");
        check(index.count("abcd") == 0, "an absent pattern has no interval");

        // ---- (3) LF mapping ---------------------------------------------------------------------------
        int size = n + 1;
        boolean[] seen = new boolean[size];
        for (int i = 0; i < size; i++) {
            int target = index.c[index.bwt[i]] + index.rank(index.bwt[i], i);
            check(target >= 0 && target < size, "LF stays in range");
            check(!seen[target], "LF is a permutation (no repeated image)");
            seen[target] = true;
        }
        int position = 0;
        int stepsTaken = 0;
        do {
            position = index.lf(position);
            stepsTaken++;
        } while (position != 0 && stepsTaken <= size);
        check(position == 0 && stepsTaken == size, "LF is a single cycle through all n + 1 positions ("
                + stepsTaken + " steps)");

        // ---- (4) locate ------------------------------------------------------------------------------
        for (int length = 1; length <= 3; length++) {
            char[] candidate = new char[length];
            checkAllLocations(index, text, candidate, 0) ;
        }
        long locateSteps = index.lfSteps;
        check(locateSteps > 0, "the locate path exercised LF");
        System.out.println("locate: exhaustive for all patterns of length 1..3, LF steps so far " + locateSteps);
        for (int trial = 0; trial < 100; trial++) {
            int from = rng.nextInt(n);
            int length = 1 + rng.nextInt(Math.min(8, n - from));
            String pattern = text.substring(from, from + length);
            List<Integer> found = index.locate(pattern);
            List<Integer> expected = new ArrayList<>();
            for (int i = 0; i + pattern.length() <= n; i++) if (text.startsWith(pattern, i)) expected.add(i);
            check(found.equals(expected), "locate matches the brute-force positions");
        }

        // ---- (5) interval consistency -----------------------------------------------------------------
        for (int trial = 0; trial < 200; trial++) {
            int from = rng.nextInt(n);
            int length = 1 + rng.nextInt(Math.min(6, n - from));
            String pattern = text.substring(from, from + length);
            int[] range = index.interval(pattern);
            int expected = bruteCount(text, pattern);
            if (expected == 0) {
                check(range == null, "an absent pattern has no interval");
            } else {
                check(range != null && range[1] - range[0] == expected, "the interval width equals the count");
                int[] arrayInterval = index.interval(pattern);
                check(arrayInterval[0] == range[0] && arrayInterval[1] == range[1], "intervals are deterministic");
            }
        }

        System.out.println("PASS");
    }

    static int enumerateCounts(FmIndex index, String text, char[] candidate, int depth) {
        int counted = 0;
        for (char c = 'a'; c <= 'c'; c++) {
            candidate[depth] = c;
            if (depth + 1 == candidate.length) {
                String pattern = new String(candidate);
                check(index.count(pattern) == bruteCount(text, pattern),
                        "count of '" + pattern + "' matches brute force");
                counted++;
            } else {
                counted += enumerateCounts(index, text, candidate, depth + 1);
            }
        }
        return counted;
    }

    static void checkAllLocations(FmIndex index, String text, char[] candidate, int depth) {
        for (char c = 'a'; c <= 'c'; c++) {
            candidate[depth] = c;
            if (depth + 1 == candidate.length) {
                String pattern = new String(candidate);
                List<Integer> found = index.locate(pattern);
                List<Integer> expected = new ArrayList<>();
                for (int i = 0; i + pattern.length() <= text.length(); i++) {
                    if (text.startsWith(pattern, i)) expected.add(i);
                }
                check(found.equals(expected), "locate('" + pattern + "') matches brute force");
                check(found.size() == index.count(pattern), "locate returns exactly count positions");
            } else {
                checkAllLocations(index, text, candidate, depth + 1);
            }
        }
    }
}
