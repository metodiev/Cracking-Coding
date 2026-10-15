/** Task 412 - Implement string factorization (LZ77 parse). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Greedy LZ77 factorization. Every token is a triple (offset, length, next):
 *   * offset = distance back to the match source (0 for a pure literal),
 *   * length = matched characters copied from that earlier position,
 *   * next   = the literal character that follows the match (-1 when the match
 *              consumes the end of the string).
 * Rules (documented, deterministic):
 *   * a match source j must satisfy max(0, i - window) <= j < i (window = 8),
 *   * the greedy step takes the LONGEST match, ties broken by the SMALLEST offset,
 *   * matches shorter than min_len = 3 are not worth a token, so a literal is emitted,
 *   * matches may overlap the source region (s[j+L] is read from the original string).
 * The brute-force oracle recomputes the same token list with explicit enumeration, and
 * a DP computes a lower bound on the token count of any parse.
 */
import java.util.*;

public class Solution {
    static final int MIN_LEN = 3;
    static final int WINDOW = 8;

    static int[][] lz77Parse(String s, int window, int minLen) {
        List<int[]> tokens = new ArrayList<>();
        int n = s.length();
        int i = 0;
        while (i < n) {
            int bestLen = 0, bestOff = 0;
            for (int j = Math.max(0, i - window); j < i; j++) {
                int L = 0;
                while (i + L < n && s.charAt(j + L) == s.charAt(i + L)) L++;
                if (L > bestLen) { bestLen = L; bestOff = i - j; }
            }
            if (bestLen >= minLen) {
                boolean atEnd = i + bestLen == n;
                tokens.add(new int[]{bestOff, bestLen, atEnd ? -1 : s.charAt(i + bestLen)});
                i += bestLen + (atEnd ? 0 : 1);
            } else {
                tokens.add(new int[]{0, 0, s.charAt(i)});
                i += 1;
            }
        }
        return tokens.toArray(new int[0][]);
    }

    /** Replays the tokens; overlapping copies read characters emitted moments ago. */
    static String lz77Decode(int[][] tokens) {
        StringBuilder out = new StringBuilder();
        for (int[] t : tokens) {
            if (t[1] > 0) {
                int start = out.length() - t[0];
                for (int k = 0; k < t[1]; k++) out.append(out.charAt(start + k));
            }
            if (t[2] != -1) out.append((char) t[2]);
        }
        return out.toString();
    }

    /** Same greedy parse written with explicit candidate enumeration over all lengths. */
    static int[][] lz77Bruteforce(String s, int window, int minLen) {
        List<int[]> tokens = new ArrayList<>();
        int n = s.length();
        int i = 0;
        while (i < n) {
            int bestLen = 0, bestOff = 0;
            for (int length = n - i; length >= 1 && bestLen == 0; length--) {
                for (int j = Math.max(0, i - window); j < i; j++) {
                    if (s.regionMatches(j, s, i, length)) {
                        bestLen = length;
                        bestOff = i - j;
                        break;
                    }
                }
            }
            if (bestLen >= minLen) {
                boolean atEnd = i + bestLen == n;
                tokens.add(new int[]{bestOff, bestLen, atEnd ? -1 : s.charAt(i + bestLen)});
                i += bestLen + (atEnd ? 0 : 1);
            } else {
                tokens.add(new int[]{0, 0, s.charAt(i)});
                i += 1;
            }
        }
        return tokens.toArray(new int[0][]);
    }

    /** DP: fewest tokens any parse of s can use; greedy can only be >= this. */
    static int optimalParseLen(String s, int window, int minLen) {
        int n = s.length();
        int[] dp = new int[n + 1];
        for (int i = n - 1; i >= 0; i--) {
            int best = 1 + dp[i + 1];
            for (int j = Math.max(0, i - window); j < i; j++) {
                int L = 0;
                while (i + L < n && s.charAt(j + L) == s.charAt(i + L)) L++;
                for (int length = minLen; length <= L; length++) {
                    int cand = i + length == n ? 1 : 1 + dp[i + length + 1];
                    if (cand < best) best = cand;
                }
            }
            dp[i] = best;
        }
        return dp[0];
    }

    private static int seed = 777;

    private static String lcgString(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int k = 0; k < length; k++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append(alphabet.charAt(seed % alphabet.length()));
        }
        return sb.toString();
    }

    static String tk(int[] t) {
        return "(" + t[0] + "," + t[1] + "," + (t[2] == -1 ? "None" : (char) t[2]) + ")";
    }

    static boolean sameTokens(int[][] a, int[][] b) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            if (a[i][0] != b[i][0] || a[i][1] != b[i][1] || a[i][2] != b[i][2]) return false;
        }
        return true;
    }

    public static void main(String[] args) {
        // exact greedy factorizations demanded by the task
        check(sameTokens(lz77Parse("abcabcabc", 8, 3),
                new int[][]{{0, 0, 'a'}, {0, 0, 'b'}, {0, 0, 'c'}, {3, 6, -1}}), "abcabcabc");
        check(sameTokens(lz77Parse("aaaaaaaa", 8, 3),
                new int[][]{{0, 0, 'a'}, {1, 7, -1}}), "aaaaaaaa");

        // round trips and hand-built corner cases
        check(lz77Decode(lz77Parse("abcabcabc", 8, 3)).equals("abcabcabc"), "rt abc");
        check(lz77Decode(lz77Parse("aaaaaaaa", 8, 3)).equals("aaaaaaaa"), "rt aaa");
        check(lz77Parse("", 8, 3).length == 0, "empty parse");
        check(lz77Decode(new int[0][]).isEmpty(), "empty decode");
        check(sameTokens(lz77Parse("ab", 8, 3), new int[][]{{0, 0, 'a'}, {0, 0, 'b'}}), "too short to match");
        check(sameTokens(lz77Parse("abcabcx", 8, 3),
                new int[][]{{0, 0, 'a'}, {0, 0, 'b'}, {0, 0, 'c'}, {3, 3, 'x'}}), "match then next char");
        check(sameTokens(lz77Parse("aabaaba", 8, 3),
                new int[][]{{0, 0, 'a'}, {0, 0, 'a'}, {0, 0, 'b'}, {3, 4, -1}}), "consume tail");

        // window bounds the match source: distance-3 source needs window >= 3
        check(sameTokens(lz77Parse("aabaab", 3, 3),
                new int[][]{{0, 0, 'a'}, {0, 0, 'a'}, {0, 0, 'b'}, {3, 3, -1}}), "window 3");
        check(sameTokens(lz77Parse("aabaab", 2, 3), new int[][]{
                {0, 0, 'a'}, {0, 0, 'a'}, {0, 0, 'b'}, {0, 0, 'a'}, {0, 0, 'a'}, {0, 0, 'b'}}), "window 2");

        // min length boundary: length-2 match is rejected, length-3 accepted
        check(sameTokens(lz77Parse("aaab", 8, 3),
                new int[][]{{0, 0, 'a'}, {0, 0, 'a'}, {0, 0, 'a'}, {0, 0, 'b'}}), "min len 3");
        check(sameTokens(lz77Parse("aaab", 8, 2),
                new int[][]{{0, 0, 'a'}, {1, 2, 'b'}}), "min len 2");

        // optimal parses of the two hand strings
        check(optimalParseLen("abcabcabc", 8, 3) == 4, "opt abc");
        check(optimalParseLen("aaaaaaaa", 8, 3) == 2, "opt aaa");
        check(optimalParseLen("aaab", 8, 3) == 4, "opt aaab");
        check(optimalParseLen("aabaab", 3, 3) == 4, "opt aabaab");

        for (int round = 0; round < 8; round++) {
            int length = 60 + seed % 100;
            String alphabet = seed % 2 == 0 ? "ab" : "abc";
            String text = lcgString(length, alphabet);
            int window = 4 + seed % 29;
            int[][] greedy = lz77Parse(text, window, 3);
            int[][] slow = lz77Bruteforce(text, window, 3);
            check(sameTokens(greedy, slow), "oracle equal round " + round);
            check(lz77Decode(greedy).equals(text), "round trip round " + round);
            for (int[] t : greedy) {
                check(t[0] <= window && (t[1] == 0 || t[1] >= 3), "invariants round " + round);
            }
            int optimal = optimalParseLen(text, window, 3);
            check(optimal <= greedy.length && greedy.length <= text.length(), "bound round " + round);
        }

        // greedy re-derivation: at every token boundary the token must be the longest
        // match, ties resolved to the smallest offset (or a literal when it is shorter)
        String text = "abababababab";
        int[][] tokens = lz77Parse(text, 8, 3);
        int pos = 0;
        for (int[] t : tokens) {
            int bestLen = 0, bestJ = -1;
            for (int j = Math.max(0, pos - 8); j < pos; j++) {
                int L = 0;
                while (pos + L < text.length() && text.charAt(j + L) == text.charAt(pos + L)) L++;
                if (L > bestLen) { bestLen = L; bestJ = j; }
            }
            if (t[1] > 0) {
                check(t[1] == bestLen && t[0] == pos - bestJ, "rederive match at " + pos);
            } else {
                check(bestLen < 3, "rederive literal at " + pos);
            }
            pos += t[1] + (t[2] == -1 ? 0 : 1);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
