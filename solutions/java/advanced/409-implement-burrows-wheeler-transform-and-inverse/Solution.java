/** Task 409 - Implement the Burrows-Wheeler transform and its inverse. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Cyclic BWT: sort all n cyclic rotations of s, take the last column, and record the *primary
 * index* - the row holding the rotation that starts at position 0. Hand anchors: banana ->
 * ("nnbaaa", 3), abab -> ("bbaa", 0), aaaa -> ("aaaa", 0). Reversal is only possible with the
 * primary index: "ab" and "ba" both transform to "ba" but use primary 0 and 1.
 *
 * Inverse: with F = sorted(L) (the first column) the LF-mapping row_of(L[i]) -> position in F
 * (counting equal characters in order) maps a rotation to the rotation starting one position
 * earlier; walking n steps from the primary row and reversing the collected characters
 * rebuilds the string. Ties between equal rotations are resolved by row index (stable sort).
 *
 * Sentinel variant: BWT(s + "$") with a unique smallest character needs no primary index - the
 * row whose last character is "$" is the rotation starting at 0, so the same LF walk recovers
 * s exactly ("banana" -> "annb$aa").
 *
 * Checks: hand values, LF-mapping is a permutation, multiset preserved, both inverses
 * round-trip on LCG strings including the empty string, and the medium case (300 characters).
 */
import java.util.*;

class BwtResult {
    final String last;
    final int primary;

    BwtResult(String last, int primary) {
        this.last = last;
        this.primary = primary;
    }
}

class BurrowsWheeler {
    static final char SENTINEL = '$';

    static BwtResult bwtCyclic(String s) {
        int n = s.length();
        if (n == 0) return new BwtResult("", 0);
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        final String doubled = s + s;
        Arrays.sort(order, (a, b) -> {
            for (int k = 0; k < n; k++) {
                char ca = doubled.charAt(a + k);
                char cb = doubled.charAt(b + k);
                if (ca != cb) return ca < cb ? -1 : 1;
            }
            return 0;                                    // equal rotations keep index order (stable)
        });
        StringBuilder last = new StringBuilder();
        int primary = -1;
        for (int i = 0; i < n; i++) {
            int start = order[i];
            last.append(s.charAt((start - 1 + n) % n));
            if (start == 0) primary = i;
        }
        return new BwtResult(last.toString(), primary);
    }

    static int[] lfMapping(String last) {
        int n = last.length();
        int[] lf = new int[n];
        Map<Character, Integer> starts = new TreeMap<>();
        int total = 0;
        for (char ch : sortedUnique(last)) {
            starts.put(ch, total);
            total += count(last, ch);
        }
        Map<Character, Integer> seen = new HashMap<>();
        for (int i = 0; i < n; i++) {
            char ch = last.charAt(i);
            int rank = seen.getOrDefault(ch, 0);
            lf[i] = starts.get(ch) + rank;
            seen.put(ch, rank + 1);
        }
        return lf;
    }

    private static List<Character> sortedUnique(String s) {
        TreeSet<Character> set = new TreeSet<>();
        for (int i = 0; i < s.length(); i++) set.add(s.charAt(i));
        return new ArrayList<>(set);
    }

    private static int count(String s, char ch) {
        int total = 0;
        for (int i = 0; i < s.length(); i++) if (s.charAt(i) == ch) total++;
        return total;
    }

    static String inverseCyclic(String last, int primary) {
        int n = last.length();
        if (n == 0) return "";
        int[] lf = lfMapping(last);
        int row = primary;
        StringBuilder collected = new StringBuilder();
        for (int step = 0; step < n; step++) {
            collected.append(last.charAt(row));
            row = lf[row];
        }
        if (row != primary) throw new IllegalStateException("LF walk must return to start");
        return collected.reverse().toString();
    }

    static String bwtSentinel(String s) {
        if (s.indexOf(SENTINEL) != -1) throw new IllegalArgumentException("text must not contain $");
        return bwtCyclic(s + SENTINEL).last;
    }

    static String inverseSentinel(String last) {
        int n = last.length() - 1;
        int start = last.indexOf(SENTINEL);
        int[] lf = lfMapping(last);
        int row = start;
        StringBuilder collected = new StringBuilder();
        for (int step = 0; step < n + 1; step++) {
            collected.append(last.charAt(row));
            row = lf[row];
        }
        return collected.reverse().substring(0, n);      // reversed join is s + "$"
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

    static String sortedChars(String s) {
        char[] chars = s.toCharArray();
        Arrays.sort(chars);
        return new String(chars);
    }

    static void checkBwt(String s, String expectedLast, int expectedPrimary) {
        BwtResult result = BurrowsWheeler.bwtCyclic(s);
        check(result.last.equals(expectedLast), "last " + s);
        check(result.primary == expectedPrimary, "primary " + s);
    }

    public static void main(String[] args) {
        checkBwt("banana", "nnbaaa", 3);
        checkBwt("abab", "bbaa", 0);
        checkBwt("aaaa", "aaaa", 0);
        checkBwt("", "", 0);
        checkBwt("a", "a", 0);
        checkBwt("ba", "ba", 1);
        checkBwt("ab", "ba", 0);

        // same last column, different primary index -> different original
        check(BurrowsWheeler.inverseCyclic("ba", 0).equals("ab"), "inverse ab");
        check(BurrowsWheeler.inverseCyclic("ba", 1).equals("ba"), "inverse ba");
        BwtResult banana = BurrowsWheeler.bwtCyclic("banana");
        check(BurrowsWheeler.inverseCyclic(banana.last, banana.primary).equals("banana"), "inverse banana");
        BwtResult miss = BurrowsWheeler.bwtCyclic("mississippi");
        check(BurrowsWheeler.inverseCyclic(miss.last, miss.primary).equals("mississippi"), "inverse mississippi");

        check(BurrowsWheeler.bwtSentinel("banana").equals("annb$aa"), "sentinel banana");
        check(BurrowsWheeler.bwtSentinel("").equals("$"), "sentinel empty");
        check(BurrowsWheeler.bwtSentinel("a").equals("a$"), "sentinel a");
        check(BurrowsWheeler.inverseSentinel("$").equals(""), "inverse sentinel empty");
        check(BurrowsWheeler.inverseSentinel("annb$aa").equals("banana"), "inverse sentinel banana");
        check(BurrowsWheeler.inverseSentinel(BurrowsWheeler.bwtSentinel("abracadabra")).equals("abracadabra"),
                "inverse sentinel abracadabra");

        check(sortedChars(miss.last).equals(sortedChars("mississippi")), "multiset");
        int[] lf = BurrowsWheeler.lfMapping(miss.last);
        int[] sortedLf = lf.clone();
        Arrays.sort(sortedLf);
        for (int i = 0; i < lf.length; i++) check(sortedLf[i] == i, "lf permutation " + i);

        for (String alphabet : new String[]{"ab", "abc"}) {
            for (int trial = 0; trial < 40; trial++) {
                int length = trial % 41;
                String s = lcgString(length, alphabet);
                BwtResult result = BurrowsWheeler.bwtCyclic(s);
                check(sortedChars(result.last).equals(sortedChars(s)), "lcg multiset " + s);
                check(BurrowsWheeler.inverseCyclic(result.last, result.primary).equals(s), "lcg inverse " + s);
                check(BurrowsWheeler.inverseSentinel(BurrowsWheeler.bwtSentinel(s)).equals(s), "lcg sentinel " + s);
            }
        }

        String medium = lcgString(300, "abc");
        BwtResult mediumResult = BurrowsWheeler.bwtCyclic(medium);
        check(mediumResult.last.length() == 300, "medium length");
        check(BurrowsWheeler.inverseCyclic(mediumResult.last, mediumResult.primary).equals(medium), "medium inverse");
        check(BurrowsWheeler.inverseSentinel(BurrowsWheeler.bwtSentinel(medium)).equals(medium), "medium sentinel");

        boolean threw = false;
        try {
            BurrowsWheeler.bwtSentinel("bad$text");
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "sentinel validation");
        System.out.println("PASS");
    }
}
