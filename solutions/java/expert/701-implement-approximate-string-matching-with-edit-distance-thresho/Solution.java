/** Task 701 - Implement approximate string matching with edit distance threshold using Ukkonen.
 * (Tier: Expert, CTCI Beyond -- Advanced Data Structures & Algorithms)
 *
 * Ukkonen's algorithm prunes the Levenshtein DP matrix D[i][j] = edit(P[0,j), text prefix of length
 * i) in two ways that are provably safe: a cell with |i - j| > k can never be <= k (an edit distance
 * is at least the difference of the prefix lengths), and a cell whose value exceeds k can only feed
 * successors that also exceed k, so every value above k is saturated to k + 1. On top of the band the
 * classic row cutoff applies: D is non-decreasing in i for fixed j, so once every cell of a row
 * exceeds k, no later row can hold a value <= k and the scan stops.
 *
 * The application here is *approximate substring* matching, where the occurrence may start anywhere,
 * so the recurrence is run once per candidate start s on a window of min(n - s, m + k) characters --
 * a match with at most k edits covers at most m + k text characters from its start. Running the
 * bound-anchored recurrence per start is exactly what keeps the band and the cutoff valid: for the
 * free-start (Sellers) recurrence the same two prunes are unsound, which is why the band is expressed
 * relative to the window start instead of the text origin. unsoundFreeStartCutoff() below is that
 * broken variant, kept so the check battery can exhibit the occurrence it misses (pattern "abc" in
 * text "xabcx" with k = 0: row 1 already saturates because of the leading 'x', although an exact
 * occurrence follows).
 *
 * Cost is O(n * k * (m + k)) worst case instead of O(n * m) for the unbanded DP, which is the usual
 * price of the k-band pruning; every row really does touch at most 2k + 1 cells.
 *
 * Verified: the reported occurrence set equals the brute-force set (textbook Levenshtein on every
 * text interval), every reported distance is re-confirmed by an independent Levenshtein call on the
 * reported interval, per-end minima and the set of ends within the threshold match brute force, the
 * unsound variant demonstrably misses a real occurrence, and empty/degenerate/invalid inputs behave.
 */
import java.util.*;

class UkkonenMatcher {
    /** One approximate occurrence: text[start, end) matches the pattern with `distance` edits. */
    static final class Match {
        final int start;
        final int end;
        final int distance;

        Match(int start, int end, int distance) {
            this.start = start;
            this.end = end;
            this.distance = distance;
        }

        @Override
        public String toString() {
            return "[" + start + "," + end + ")d" + distance;
        }
    }

    private final String text;
    private final int k;
    private int windowsScanned;
    private long cellsVisited;

    UkkonenMatcher(String text, int k) {
        if (k < 0) throw new IllegalArgumentException("threshold must be non-negative: " + k);
        this.text = Objects.requireNonNull(text, "text");
        this.k = k;
    }

    int threshold() {
        return k;
    }

    /** Telemetry: how many per-start windows were scanned and how many band cells were filled. */
    long cellsVisited() {
        return cellsVisited;
    }

    int windowsScanned() {
        return windowsScanned;
    }

    /** Every (start, end) pair whose edit distance to the pattern is at most the threshold. */
    List<Match> occurrences(String pattern) {
        Objects.requireNonNull(pattern, "pattern");
        List<Match> found = new ArrayList<>();
        int m = pattern.length();
        if (m == 0 || text.isEmpty()) return found;
        for (int start = 0; start < text.length(); start++) {
            scanFrom(pattern, start, found);
        }
        found.sort(Comparator.<Match>comparingInt(match -> match.end)
                .thenComparingInt(match -> match.start));
        return found;
    }

    /** For every end index, the smallest distance reachable from any start (smallest start on ties). */
    Map<Integer, Match> bestPerEnd(String pattern) {
        Map<Integer, Match> best = new TreeMap<>();
        for (Match match : occurrences(pattern)) {
            Match current = best.get(match.end);
            if (current == null || match.distance < current.distance
                    || (match.distance == current.distance && match.start < current.start)) {
                best.put(match.end, match);
            }
        }
        return best;
    }

    private void scanFrom(String pattern, int start, List<Match> out) {
        int m = pattern.length();
        int rows = Math.min(text.length() - start, m + k);
        int saturated = k + 1;
        int[] prev = new int[m + 1];
        int[] cur = new int[m + 1];
        for (int j = 0; j <= m; j++) prev[j] = Math.min(j, saturated);
        windowsScanned++;
        for (int i = 1; i <= rows; i++) {
            int lo = Math.max(1, i - k);
            int hi = Math.min(m, i + k);
            cur[0] = Math.min(i, saturated);
            int rowMin = cur[0];
            for (int j = 1; j <= m; j++) cur[j] = saturated;
            char textChar = text.charAt(start + i - 1);
            for (int j = lo; j <= hi; j++) {
                int cost = textChar == pattern.charAt(j - 1) ? 0 : 1;
                int best = Math.min(prev[j - 1] + cost, Math.min(prev[j] + 1, cur[j - 1] + 1));
                cur[j] = Math.min(best, saturated);
                if (cur[j] < rowMin) rowMin = cur[j];
                cellsVisited++;
            }
            if (cur[m] <= k) out.add(new Match(start, start + i, cur[m]));
            if (rowMin > k) break;                 // Ukkonen cutoff: every later row saturates too
            int[] swap = prev;
            prev = cur;
            cur = swap;
        }
    }

    /**
     * Deliberately unsound reference: the bound-anchored DP with a band and row cutoff taken relative
     * to the text origin, applied to the free-start recurrence. Kept only to show what the correct
     * windowing avoids; the check battery asserts that it misses a real occurrence.
     */
    static List<Match> unsoundFreeStartCutoff(String text, String pattern, int k) {
        List<Match> found = new ArrayList<>();
        int m = pattern.length();
        int n = text.length();
        int saturated = k + 1;
        int[] prev = new int[m + 1];
        int[] cur = new int[m + 1];
        for (int j = 0; j <= m; j++) prev[j] = Math.min(j, saturated);
        for (int i = 1; i <= n; i++) {
            cur[0] = 0;                            // free start
            int rowMin = Integer.MAX_VALUE;
            for (int j = 1; j <= m; j++) cur[j] = saturated;
            for (int j = Math.max(1, i - k); j <= Math.min(m, i + k); j++) {
                int cost = text.charAt(i - 1) == pattern.charAt(j - 1) ? 0 : 1;
                cur[j] = Math.min(saturated,
                        Math.min(prev[j - 1] + cost, Math.min(prev[j] + 1, cur[j - 1] + 1)));
                if (cur[j] < rowMin) rowMin = cur[j];
            }
            if (cur[m] <= k) found.add(new Match(0, i, cur[m]));
            if (rowMin > k) return found;
            int[] swap = prev;
            prev = cur;
            cur = swap;
        }
        return found;
    }

    /** Textbook Levenshtein distance between two strings, used as the independent oracle. */
    static int levenshtein(String a, String b) {
        int[] prev = new int[b.length() + 1];
        for (int j = 0; j <= b.length(); j++) prev[j] = j;
        for (int i = 1; i <= a.length(); i++) {
            int[] cur = new int[b.length() + 1];
            cur[0] = i;
            for (int j = 1; j <= b.length(); j++) {
                int cost = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1;
                cur[j] = Math.min(Math.min(prev[j - 1] + cost, prev[j] + 1), cur[j - 1] + 1);
            }
            prev = cur;
        }
        return prev[b.length()];
    }

    /** Brute force: every non-empty text interval whose distance to the pattern is at most k. */
    static List<String> bruteForceOccurrences(String text, String pattern, int k) {
        List<String> found = new ArrayList<>();
        for (int end = 1; end <= text.length(); end++) {
            for (int start = 0; start < end; start++) {
                if (levenshtein(pattern, text.substring(start, end)) <= k) {
                    found.add(start + ":" + end);
                }
            }
        }
        return found;
    }
}

public class Solution {
    private static long state = 0x123456789ABCDEFL;

    /** splitmix64 - deterministic PRNG, no Math.random. */
    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), bound);
    }

    static String randomString(int length, int alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) sb.append((char) ('a' + nextInt(alphabet)));
        return sb.toString();
    }

    static List<String> asPairs(List<UkkonenMatcher.Match> matches) {
        List<String> pairs = new ArrayList<>();
        for (UkkonenMatcher.Match match : matches) pairs.add(match.start + ":" + match.end);
        return pairs;
    }

    public static void main(String[] args) {
        // --- exact occurrences (k = 0) ----------------------------------------
        UkkonenMatcher exact = new UkkonenMatcher("abcdef", 0);
        List<UkkonenMatcher.Match> one = exact.occurrences("cde");
        check(one.size() == 1, "exact match count, got " + one.size());
        check(one.get(0).start == 2 && one.get(0).end == 5 && one.get(0).distance == 0,
                "exact match span " + one.get(0));
        check(exact.occurrences("xyz").isEmpty(), "k = 0 must not admit a distant pattern");
        check(exact.occurrences("cdec").isEmpty(), "k = 0 longer pattern");

        // --- the free-start cutoff counterexample ------------------------------
        UkkonenMatcher fuzzy = new UkkonenMatcher("xabcx", 0);
        List<UkkonenMatcher.Match> counter = fuzzy.occurrences("abc");
        check(counter.size() == 1 && counter.get(0).start == 1 && counter.get(0).end == 4,
                "windowing must find the occurrence after a saturating row: " + counter);
        check(UkkonenMatcher.unsoundFreeStartCutoff("xabcx", "abc", 0).isEmpty(),
                "origin-anchored free-start cutoff is unsound and misses [1,4)");

        // --- degenerate and rejected inputs ------------------------------------
        check(new UkkonenMatcher("", 2).occurrences("ab").isEmpty(), "empty text has no occurrence");
        check(new UkkonenMatcher("abc", 3).occurrences("").isEmpty(), "empty pattern has no occurrence");
        check(new UkkonenMatcher("ab", 1).occurrences("abc").size() == 1, "one insertion away fits");
        check(new UkkonenMatcher("ab", 0).occurrences("abc").isEmpty(), "one insertion away, k = 0");
        check(new UkkonenMatcher("a", 0).occurrences("a").size() == 1, "single character text");
        List<UkkonenMatcher.Match> deletion = new UkkonenMatcher("a", 1).occurrences("ab");
        check(deletion.size() == 1 && deletion.get(0).start == 0 && deletion.get(0).end == 1
                        && deletion.get(0).distance == 1,
                "single character text, one deletion away: " + deletion);
        check(new UkkonenMatcher("a", 1).occurrences("abc").isEmpty(),
                "single character text, pattern two edits away");

        boolean rejected = false;
        try {
            new UkkonenMatcher("abc", -1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "negative threshold must be rejected");
        rejected = false;
        try {
            new UkkonenMatcher("abc", 1).occurrences(null);
        } catch (NullPointerException expected) {
            rejected = true;
        }
        check(rejected, "null pattern must be rejected");

        // --- brute-force cross-verification on small random instances ---------
        int adversarialHits = 0;
        for (int trial = 0; trial < 40; trial++) {
            int alphabet = 2 + nextInt(3);
            String text = randomString(3 + nextInt(34), alphabet);
            String pattern = randomString(1 + nextInt(6), alphabet);
            int k = nextInt(4);
            UkkonenMatcher matcher = new UkkonenMatcher(text, k);
            List<UkkonenMatcher.Match> matches = matcher.occurrences(pattern);
            List<String> expected = UkkonenMatcher.bruteForceOccurrences(text, pattern, k);
            check(asPairs(matches).equals(expected),
                    "trial " + trial + " occurrence set, text=" + text + " pattern=" + pattern
                            + " k=" + k + "\n  got      " + asPairs(matches) + "\n  expected " + expected);
            for (UkkonenMatcher.Match match : matches) {
                check(match.distance <= k, "trial " + trial + " distance above threshold");
                check(match.start >= 0 && match.start < match.end && match.end <= text.length(),
                        "trial " + trial + " span " + match);
                int recheck = UkkonenMatcher.levenshtein(pattern, text.substring(match.start, match.end));
                check(recheck == match.distance,
                        "trial " + trial + " claimed " + match.distance + " but is " + recheck);
            }
            Map<Integer, UkkonenMatcher.Match> best = matcher.bestPerEnd(pattern);
            for (Map.Entry<Integer, UkkonenMatcher.Match> entry : best.entrySet()) {
                int end = entry.getKey();
                int bruteMin = k + 1;
                for (int start = 0; start < end; start++) {
                    bruteMin = Math.min(bruteMin,
                            UkkonenMatcher.levenshtein(pattern, text.substring(start, end)));
                }
                check(entry.getValue().distance == bruteMin,
                        "trial " + trial + " end " + end + " best " + entry.getValue().distance
                                + " vs brute " + bruteMin);
            }
            if (!matches.isEmpty() && k > 0) adversarialHits++;
            List<UkkonenMatcher.Match> again = matcher.occurrences(pattern);
            check(asPairs(again).equals(asPairs(matches)), "trial " + trial + " determinism");
        }
        check(adversarialHits > 20, "roughly half the trials should be fuzzy matches: " + adversarialHits);

        // --- k >= pattern length: everything within reach ---------------------
        UkkonenMatcher loose = new UkkonenMatcher("qrs", 3);
        check(loose.occurrences("abc").size() == UkkonenMatcher.bruteForceOccurrences("qrs", "abc", 3).size(),
                "loose threshold must agree with brute force");

        // --- larger text, cross-checked with String.indexOf for k = 0 ---------
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < 60; i++) builder.append(randomString(8, 4));
        String haystack = builder.toString();
        UkkonenMatcher large = new UkkonenMatcher(haystack, 0);
        for (int trial = 0; trial < 5; trial++) {
            String needle = randomString(3, 4);
            List<UkkonenMatcher.Match> found = large.occurrences(needle);
            List<String> indexOfPairs = new ArrayList<>();
            for (int at = haystack.indexOf(needle); at >= 0; at = haystack.indexOf(needle, at + 1)) {
                indexOfPairs.add(at + ":" + (at + needle.length()));
            }
            check(asPairs(found).equals(indexOfPairs),
                    "indexOf agreement for needle " + needle + ": " + asPairs(found));
        }
        check(large.cellsVisited() > 0, "the band really was traversed");
        check(large.windowsScanned() > 0, "one window per start position");

        // --- boundary: pattern exactly the text ---------------------------------
        String whole = "deterministic";
        UkkonenMatcher self = new UkkonenMatcher(whole, 0);
        List<UkkonenMatcher.Match> wholeMatch = self.occurrences(whole);
        check(wholeMatch.size() == 1 && wholeMatch.get(0).start == 0 && wholeMatch.get(0).end == whole.length(),
                "whole-text match " + wholeMatch);

        // --- boundary: threshold equal to pattern length finds the degenerates ---
        UkkonenMatcher everything = new UkkonenMatcher("ab", 2);
        check(everything.occurrences("zz").size() == UkkonenMatcher.bruteForceOccurrences("ab", "zz", 2).size(),
                "threshold == pattern length agrees with brute force");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
