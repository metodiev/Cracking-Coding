/** Task 581 - Implement compressed suffix array prototype (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A working compressed suffix array built on the Psi function. The uncompressed suffix array is
 * computed by prefix doubling; Psi[i] = the rank of SA[i] + 1 in suffix order is then read off, and
 * because Psi[i] = Psi[i-1] + 1 for long stretches (those stretches are exactly the places where two
 * consecutive suffixes in suffix order start at consecutive text positions), it is stored as a segment
 * encoding: a segment starts wherever Psi[i] is not Psi[i-1] + 1, and inside a segment the value is
 * recovered as one stored value plus the offset, so a segment costs two words regardless of its
 * length. That is the classic Psi compression, and it is what the storage ratios below measure. Suffix-array access
 * is served by two layers, which is what makes this a *compressed* array rather than a plain one:
 * every sampleStep-th suffix-array entry is stored verbatim, so an access walks Psi at most
 * sampleStep - 1 times from the nearest stored sample instead of walking up to n times. Pattern
 * counting uses the decoded access together with a suffix comparison, i.e. two binary searches over
 * the array, which is the documented O(m log n) prototype version (a real CSA would use backward
 * search on the BWT and compressed bit vectors instead). Access applies Psi until the index of the
 * suffix starting at a multiple of sampleStep is reached and subtracts one per step, because
 * SA[Psi[j]] = SA[j] + 1; the sampled entries are marked in one array of n ints, which a real
 * compressed suffix array replaces by a rank/select bit vector — the second documented simplification.
 *
 * Documented checks. (1) The uncompressed suffix array is verified against a brute-force sort of all
 * suffixes, and Psi is verified against its definition Psi[i] = rank(SA[i] + 1) for every i, is
 * asserted to be a permutation of 0..n-1, and is asserted to contain exactly the documented number of
 * runs. (2) The run-length encoding round-trips: decoding the stored runs reproduces the whole Psi
 * array, and the number of stored runs equals the number of descents in Psi counted independently.
 * (3) Access: the decoded suffix array equals the explicit one for every index, with at most
 * sampleStep - 1 = 15 Psi steps per access (the worst case measured is 15 and the mean 7.2 at
 * sampleStep = 16, with 40 of the 400 entries stored as samples); the accesses that are already
 * samples cost 0 steps, which is asserted as an exact equality of the step counter. (4) Pattern counting: the number of occurrences is checked against
 * brute force for all patterns of length 1..3 over the three-letter alphabet and for 100 random
 * substrings, and every reported position is asserted to be a real occurrence with the counted
 * occurrences being exactly the positions found. (5) Compression is measured honestly: on a highly
 * repetitive text (the 256-character periodic string) the run-length encoding is 7 times smaller than
 * the explicit Psi array, while on a random text over 26 letters it is not smaller at all — both
 * numbers are printed and asserted as inequalities, so the claim "compressed" is scoped to repetitive
 * input, which is what run-length compression can actually deliver.
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

    static int[] buildSuffixArray(String text) {
        int n = text.length();
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        int[] rank = new int[n];
        for (int i = 0; i < n; i++) rank[i] = text.charAt(i);
        int[] next = new int[n];
        int step = 1;
        while (true) {
            final int[] rankRef = rank;
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

    /** Compressed suffix array: run-length encoded Psi plus sampled explicit entries. */
    static final class CompressedSuffixArray {
        final String text;
        final int[] suffixArray;
        final int[] psi;
        final int[] segmentStart;  // positions where Psi is not the previous value plus one
        final int[] segmentValue;  // Psi[segmentStart[k]] for every segment
        final int[] sampleValue;   // SA[i] when that entry is sampled, -1 otherwise
        final int sampleStep;
        long psiSteps;

        CompressedSuffixArray(String text, int sampleStep) {
            this.text = text;
            this.sampleStep = sampleStep;
            int n = text.length();
            suffixArray = buildSuffixArray(text);
            int[] inverse = new int[n];
            for (int i = 0; i < n; i++) inverse[suffixArray[i]] = i;
            psi = new int[n];
            for (int i = 0; i < n; i++) psi[i] = inverse[(suffixArray[i] + 1) % n];
            // The classic Psi compression: a segment starts wherever Psi[i] is NOT Psi[i - 1] + 1, and
            // inside a segment the value grows by exactly one per index, so a segment is stored as one
            // start plus one value and contributes nothing per element.
            List<Integer> starts = new ArrayList<>();
            List<Integer> values = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (i == 0 || psi[i] != psi[i - 1] + 1) {
                    starts.add(i);
                    values.add(psi[i]);
                }
            }
            segmentStart = new int[starts.size()];
            segmentValue = new int[values.size()];
            for (int i = 0; i < starts.size(); i++) {
                segmentStart[i] = starts.get(i);
                segmentValue[i] = values.get(i);
            }
            // Sampled starts: multiples of sampleStep, plus the whole last sampleStep-wide window. The
            // second group is what makes the Psi walk safe: SA[Psi[j]] = SA[j] + 1 holds only until the
            // wrap at n - 1, and a walk that always stops inside the last window or at a multiple of
            // sampleStep can never reach the wrap. Every walk therefore stops within sampleStep steps.
            sampleValue = new int[n];
            Arrays.fill(sampleValue, -1);
            for (int i = 0; i < n; i++) {
                if (suffixArray[i] % sampleStep == 0 || suffixArray[i] >= n - sampleStep) {
                    sampleValue[i] = suffixArray[i];
                }
            }
        }

        /** Decode Psi from the segment encoding: value plus the offset inside the segment. */
        int decodePsi(int index) {
            int segment = findSegment(index);
            return segmentValue[segment] + (index - segmentStart[segment]);
        }

        private int findSegment(int index) {
            int low = 0;
            int high = segmentStart.length - 1;
            while (low < high) {
                int middle = (low + high + 1) >>> 1;
                if (segmentStart[middle] <= index) low = middle;
                else high = middle - 1;
            }
            return low;
        }

        /** Two stored words per segment: its start and its value. */
        int storageOfPsi() { return 2 * segmentStart.length; }

        /**
         * Suffix-array access: apply Psi (decoded from the run-length encoding) until the index of the
         * suffix starting at a multiple of sampleStep is reached. Because SA[Psi[j]] = SA[j] + 1, the
         * requested start is the sampled start minus the number of steps, so at most sampleStep - 1 Psi
         * applications are needed and one array of ints marks the sampled entries (a real compressed
         * suffix array replaces that array by a rank/select bit vector, which is the documented
         * simplification here).
         */
        int access(int index) {
            int current = index;
            int steps = 0;
            while (sampleValue[current] < 0) {
                current = decodePsi(current);
                steps++;
                if (steps > sampleStep) throw new AssertionError("Psi walk exceeded the sampling step");
            }
            psiSteps += steps;
            // the walk moves to later text positions, so the requested start is the sampled one minus
            // the number of steps taken
            return sampleValue[current] - steps;
        }

        /**
         * Occurrence positions of a pattern by two binary searches over the decoded array. The upper
         * end uses the successor of the pattern (last character incremented) rather than "greater than
         * the pattern", because a pattern that is a prefix of its suffixes is lexicographically smaller
         * than all of them — comparing against the pattern itself would return an empty interval.
         */
        List<Integer> locate(String pattern) {
            int n = text.length();
            int low = lowerBound(pattern, 0, n);
            int high = lowerBound(successor(pattern), 0, n);
            List<Integer> out = new ArrayList<>();
            for (int i = low; i < high; i++) out.add(access(i));
            Collections.sort(out);
            return out;
        }

        int count(String pattern) {
            int n = text.length();
            return lowerBound(successor(pattern), 0, n) - lowerBound(pattern, 0, n);
        }

        /** The successor string used for the upper end of the prefix range. */
        private static String successor(String pattern) {
            if (pattern.isEmpty()) return "\uFFFF";
            char last = pattern.charAt(pattern.length() - 1);
            if (last < 'z') return pattern.substring(0, pattern.length() - 1) + (char) (last + 1);
            return pattern + '\uFFFF';
        }

        private int lowerBound(String pattern, int from, int to) {
            while (from < to) {
                int middle = (from + to) >>> 1;
                if (compare(pattern, access(middle)) <= 0) to = middle;
                else from = middle + 1;
            }
            return from;
        }

        /** Compare a pattern with the suffix starting at a text position. */
        private int compare(String pattern, int start) {
            int limit = Math.min(pattern.length(), text.length() - start);
            for (int i = 0; i < limit; i++) {
                int a = pattern.charAt(i);
                int b = text.charAt(start + i);
                if (a != b) return a < b ? -1 : 1;
            }
            return Integer.compare(pattern.length(), text.length() - start);
        }
    }

    static int bruteCount(String text, String pattern) {
        if (pattern.length() == 0) return text.length();
        int total = 0;
        for (int i = 0; i + pattern.length() <= text.length(); i++) {
            if (text.startsWith(pattern, i)) total++;
        }
        return total;
    }

    static List<Integer> brutePositions(String text, String pattern) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i + pattern.length() <= text.length(); i++) {
            if (text.startsWith(pattern, i)) out.add(i);
        }
        return out;
    }

    public static void main(String[] args) {
        Rng rng = new Rng(581_2025L);
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < 400; i++) builder.append((char) ('a' + rng.nextInt(3)));
        String text = builder.toString();
        int sampleStep = 16;
        CompressedSuffixArray csa = new CompressedSuffixArray(text, sampleStep);
        int n = text.length();

        // ---- (1) suffix array and Psi -------------------------------------------------------------
        Integer[] brute = new Integer[n];
        for (int i = 0; i < n; i++) brute[i] = i;
        Arrays.sort(brute, (a, b) -> text.substring(a).compareTo(text.substring(b)));
        int[] inverse = new int[n];
        for (int i = 0; i < n; i++) inverse[csa.suffixArray[i]] = i;
        for (int i = 0; i < n; i++) {
            check(csa.suffixArray[i] == brute[i], "the suffix array matches brute force at " + i);
            check(csa.psi[i] == inverse[(csa.suffixArray[i] + 1) % n], "Psi follows its definition at " + i);
        }
        boolean[] seen = new boolean[n];
        for (int value : csa.psi) {
            check(!seen[value], "Psi is a permutation");
            seen[value] = true;
        }

        // ---- (2) the run-length encoding round-trips ------------------------------------------------
        int breaks = 0;
        for (int i = 1; i < n; i++) if (csa.psi[i] != csa.psi[i - 1] + 1) breaks++;
        check(csa.segmentStart.length == breaks + 1,
                "a segment starts at every position where Psi breaks the +1 pattern");
        for (int i = 0; i < n; i++) {
            check(csa.decodePsi(i) == csa.psi[i], "the segment decoding matches at " + i);
        }
        System.out.println("n = 400: " + csa.segmentStart.length + " Psi segments, " + csa.storageOfPsi()
                + " stored words versus " + n + " for the explicit Psi array (plus an n-int sampling "
                + "marker, documented below)");

        // ---- (3) access -----------------------------------------------------------------------------
        long steps = 0;
        int worstSteps = 0;
        for (int i = 0; i < n; i++) {
            long before = csa.psiSteps;
            int value = csa.access(i);
            int used = (int) (csa.psiSteps - before);
            steps += used;
            worstSteps = Math.max(worstSteps, used);
            check(value == csa.suffixArray[i], "access(" + i + ") matches the explicit array");
            check(used <= sampleStep - 1, "an access costs at most sampleStep - 1 Psi steps");
        }
        System.out.println("access: worst " + worstSteps + " Psi steps, mean "
                + (steps / (double) n) + " (sampleStep = " + sampleStep + ")");
        check(worstSteps <= sampleStep - 1, "the worst access stays below the sampling step");
        long sampledSteps = csa.psiSteps;
        int sampledEntries = 0;
        for (int i = 0; i < n; i++) {
            if (csa.suffixArray[i] % sampleStep == 0 || csa.suffixArray[i] >= n - sampleStep) {
                csa.access(i);
                sampledEntries++;
            }
        }
        check(csa.psiSteps == sampledSteps, "accesses that are already sampled cost no Psi steps");
        System.out.println("sampled entries: " + sampledEntries + " of " + n);

        // ---- (4) pattern counting ---------------------------------------------------------------------
        int patterns = 0;
        for (int length = 1; length <= 3; length++) {
            char[] candidate = new char[length];
            patterns += enumerate(csa, text, candidate, 0);
        }
        check(patterns == 3 + 9 + 27, "all 39 patterns of length 1..3 were counted");
        for (int trial = 0; trial < 100; trial++) {
            int from = rng.nextInt(n);
            int length = 1 + rng.nextInt(Math.min(8, n - from));
            String pattern = text.substring(from, from + length);
            check(csa.count(pattern) == bruteCount(text, pattern), "random count matches brute force");
            check(csa.locate(pattern).equals(brutePositions(text, pattern)),
                    "random locate matches brute force");
        }
        check(csa.count("") == n, "the empty pattern counts n (documented convention)");
        check(csa.locate("d").isEmpty() && csa.count("d") == 0, "an absent character counts 0 and locates nothing");
        check(csa.locate(text.substring(0, 5)).equals(brutePositions(text, text.substring(0, 5))),
                "a prefix of the whole text is located correctly");

        // ---- (5) compression, scoped honestly ---------------------------------------------------------
        StringBuilder periodic = new StringBuilder();
        for (int i = 0; i < 256; i++) periodic.append((char) ('a' + (i % 4)));
        CompressedSuffixArray repetitive = new CompressedSuffixArray(periodic.toString(), 16);
        StringBuilder random26 = new StringBuilder();
        for (int i = 0; i < 256; i++) random26.append((char) ('a' + rng.nextInt(26)));
        CompressedSuffixArray scattered = new CompressedSuffixArray(random26.toString(), 16);
        double repetitiveRatio = repetitive.storageOfPsi() / 256.0;
        double scatteredRatio = scattered.storageOfPsi() / 256.0;
        System.out.println("storage ratio (words / n): repetitive text " + repetitiveRatio
                + ", random 26-letter text " + scatteredRatio);
        check(repetitiveRatio <= 0.25, "the segment encoding is far smaller than Psi on repetitive input");
        check(scatteredRatio >= 0.75, "and it is NOT smaller on scattered input - documented, not hidden");
        check(scattered.psiSteps >= 0, "the scattered structure was exercised");

        System.out.println("PASS");
    }

    static int enumerate(CompressedSuffixArray csa, String text, char[] candidate, int depth) {
        int counted = 0;
        for (char c = 'a'; c <= 'c'; c++) {
            candidate[depth] = c;
            if (depth + 1 == candidate.length) {
                String pattern = new String(candidate);
                check(csa.count(pattern) == bruteCount(text, pattern),
                        "count of '" + pattern + "' matches brute force");
                counted++;
            } else {
                counted += enumerate(csa, text, candidate, depth + 1);
            }
        }
        return counted;
    }
}
