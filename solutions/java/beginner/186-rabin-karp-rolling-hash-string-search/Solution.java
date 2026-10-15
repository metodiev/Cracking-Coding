/** Task 186 - Rabin-Karp rolling hash string search. (Tier: Beginner, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

public class Solution {
    static final long DEFAULT_BASE = 256;
    static final long DEFAULT_MOD = 1_000_000_007L;

    static long powMod(long base, long exponent, long mod) {
        long result = 1;
        for (long i = 0; i < exponent; i++) result = result * base % mod;
        return result;
    }

    /**
     * Polynomial window hash h = c0 * base^(m-1) + ... + c(m-1) (mod mod). Sliding one step
     * drops the leaving character's weighted term and appends the new one in O(1). Equal
     * hashes can also mean a collision, so every hash hit is confirmed with a direct
     * substring comparison: a false positive never becomes a reported match.
     */
    static List<Integer> rabinKarp(String text, String pattern) {
        return rabinKarp(text, pattern, DEFAULT_BASE, DEFAULT_MOD);
    }

    static List<Integer> rabinKarp(String text, String pattern, long base, long mod) {
        List<Integer> matches = new ArrayList<>();
        if (pattern.isEmpty()) return matches;
        int m = pattern.length();
        int n = text.length();
        if (m > n) return matches;

        long highest = powMod(base, m - 1, mod);        // weight of a window's first character
        long patternHash = 0;
        long windowHash = 0;
        for (int i = 0; i < m; i++) {
            patternHash = (patternHash * base + pattern.charAt(i)) % mod;
            windowHash = (windowHash * base + text.charAt(i)) % mod;
        }

        if (windowHash == patternHash && text.regionMatches(0, pattern, 0, m)) matches.add(0);
        for (int start = 1; start + m <= n; start++) {
            windowHash = (windowHash - text.charAt(start - 1) * highest) % mod;
            if (windowHash < 0) windowHash += mod;
            windowHash = (windowHash * base + text.charAt(start + m - 1)) % mod;
            if (windowHash == patternHash && text.regionMatches(start, pattern, 0, m)) matches.add(start);
        }
        return matches;
    }

    /** Reference scan used by the self-check. */
    static List<Integer> rabinKarpNaive(String text, String pattern) {
        List<Integer> matches = new ArrayList<>();
        if (pattern.isEmpty()) return matches;
        for (int i = 0; i + pattern.length() <= text.length(); i++) {
            if (text.startsWith(pattern, i)) matches.add(i);
        }
        return matches;
    }

    static String repeat(String unit, int times) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < times; i++) sb.append(unit);
        return sb.toString();
    }

    static List<Integer> range(int start, int stop, int step) {
        List<Integer> values = new ArrayList<>();
        for (int i = start; i < stop; i += step) values.add(i);
        return values;
    }

    public static void main(String[] args) {
        check(rabinKarp("abracadabra", "abra").equals(Arrays.asList(0, 7)), "abracadabra");
        check(rabinKarp("aaaaa", "aa").equals(Arrays.asList(0, 1, 2, 3)), "aaaaa / aa");
        check(rabinKarp("abcabcabc", "abc").equals(Arrays.asList(0, 3, 6)), "repeated block");
        check(rabinKarp("abcd", "bcd").equals(Arrays.asList(1)), "suffix");
        check(rabinKarp("aaaa", "b").isEmpty(), "no match");
        check(rabinKarp("ab", "abc").isEmpty(), "pattern longer than text");
        check(rabinKarp("", "a").isEmpty(), "empty text");
        check(rabinKarp("", "").isEmpty(), "empty text and pattern");
        check(rabinKarp("anything", "").isEmpty(), "empty pattern");
        check(rabinKarp("a", "a").equals(Arrays.asList(0)), "single char");

        // repetition-heavy text
        String heavy = repeat("a", 1000) + "b";
        check(rabinKarp(heavy, repeat("a", 20)).equals(range(0, 981, 1)), "many a-windows");
        check(rabinKarp(heavy, "ab").equals(Arrays.asList(999)), "single ab crossing");
        check(rabinKarp(repeat("ab", 300), "abab").equals(range(0, 597, 2)), "overlapping abab");

        // agreement with the naive scan on pseudo-random strings
        Random rng = new Random(17);
        for (int t = 0; t < 200; t++) {
            String text = randomText(rng, rng.nextInt(41), 2);
            String pattern = randomText(rng, 1 + rng.nextInt(5), 2);
            check(rabinKarp(text, pattern).equals(rabinKarpNaive(text, pattern)), "cross-check " + text + " / " + pattern);
        }

        // a tiny modulus forces collisions; the substring verification must filter them out
        for (int t = 0; t < 200; t++) {
            String text = randomText(rng, rng.nextInt(41), 3);
            String pattern = randomText(rng, 1 + rng.nextInt(4), 3);
            check(rabinKarp(text, pattern, 256, 101).equals(rabinKarpNaive(text, pattern)),
                    "tiny modulus " + text + " / " + pattern);
        }
        String collidingText = randomText(rng, 60, 6);
        Map<Long, String> seen = new HashMap<>();
        boolean collided = false;
        for (int start = 0; start + 3 <= collidingText.length(); start++) {
            String window = collidingText.substring(start, start + 3);
            long h = 0;
            for (int i = 0; i < 3; i++) h = (h * 256 + window.charAt(i)) % 101;
            if (seen.containsKey(h) && !seen.get(h).equals(window)) collided = true;
            seen.putIfAbsent(h, window);
        }
        check(collided, "expected the tiny modulus to produce a real hash collision");
        System.out.println("PASS");
    }

    static String randomText(Random rng, int length, int alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) sb.append((char) ('a' + rng.nextInt(alphabet)));
        return sb.toString();
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
