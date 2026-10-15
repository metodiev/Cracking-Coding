/** Task 414 - Implement edit distance with low memory (Hirschberg). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Classic Levenshtein DP uses an O(n*m) table. Hirschberg's algorithm recovers an
 * optimal alignment with O(n) auxiliary memory and O(n*m) time:
 *   * split b in half (Myers-Miller variant of the split), compute the forward row
 *     f[i] = dist(a[:i], b[:mid]) and the backward row g[i] = dist(a[i:], b[mid:])
 *     with two rolling rows each — the backward row reverses both strings because
 *     reversal preserves edit distance and turns suffixes into prefixes,
 *   * pick the split k minimizing f[k] + g[k] (smallest k on ties), recurse on
 *     (a[:k], b[:mid]) and (a[k:], b[mid:]); base cases solve directly.
 * An alignment is a list of ops {'=', x, y} match; {'!', x, y} substitution;
 * {'-', x, ""} deletion; {'+', "", y} insertion. Replaying the ops from a rebuilds b.
 */
import java.util.*;

class AlignResult {
    final int dist;
    final List<String[]> ops;

    AlignResult(int dist, List<String[]> ops) {
        this.dist = dist;
        this.ops = ops;
    }
}

public class Solution {
    static final int BASE = 16;

    /** Oracle: full O(n*m) table over rolling rows. */
    static int fullDpDist(String a, String b) {
        int m = b.length();
        int[] prev = new int[m + 1];
        for (int j = 0; j <= m; j++) prev[j] = j;
        for (int i = 1; i <= a.length(); i++) {
            int[] cur = new int[m + 1];
            cur[0] = i;
            for (int j = 1; j <= m; j++) {
                int sub = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1;
                cur[j] = Math.min(Math.min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + sub);
            }
            prev = cur;
        }
        return prev[m];
    }

    /** f[i] = dist(a[:i], prefix) for every i; one rolling row over the prefix. */
    static int[] forwardRow(String a, String prefix) {
        int m = prefix.length();
        int[] prev = new int[m + 1];
        for (int j = 0; j <= m; j++) prev[j] = j;
        int[] out = new int[a.length() + 1];
        out[0] = prev[m];
        for (int i = 1; i <= a.length(); i++) {
            int[] cur = new int[m + 1];
            cur[0] = i;
            for (int j = 1; j <= m; j++) {
                int sub = a.charAt(i - 1) == prefix.charAt(j - 1) ? 0 : 1;
                cur[j] = Math.min(Math.min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + sub);
            }
            prev = cur;
            out[i] = prev[m];
        }
        return out;
    }

    /** g[i] = dist(a[i:], suffix) for every i (reversal turns suffixes into prefixes). */
    static int[] backwardRow(String a, String suffix) {
        String ra = new StringBuilder(a).reverse().toString();
        String rs = new StringBuilder(suffix).reverse().toString();
        int m = rs.length();
        int[] prev = new int[m + 1];
        for (int j = 0; j <= m; j++) prev[j] = j;
        int[] collected = new int[a.length() + 1];
        collected[0] = prev[m];
        for (int p = 1; p <= ra.length(); p++) {
            int[] cur = new int[m + 1];
            cur[0] = p;
            for (int j = 1; j <= m; j++) {
                int sub = ra.charAt(p - 1) == rs.charAt(j - 1) ? 0 : 1;
                cur[j] = Math.min(Math.min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + sub);
            }
            prev = cur;
            collected[p] = prev[m];
        }
        int n = a.length();
        int[] out = new int[n + 1];
        for (int i = 0; i <= n; i++) out[i] = collected[n - i];
        return out;
    }

    /** Full-table DP with backtracking; used for the recursion base cases. */
    static AlignResult directAlign(String a, String b) {
        int n = a.length(), m = b.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++) dp[i][0] = i;
        for (int j = 1; j <= m; j++) dp[0][j] = j;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                int sub = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1;
                dp[i][j] = Math.min(Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1), dp[i - 1][j - 1] + sub);
            }
        }
        LinkedList<String[]> ops = new LinkedList<>();
        int i = n, j = m;
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && a.charAt(i - 1) == b.charAt(j - 1) && dp[i][j] == dp[i - 1][j - 1]) {
                ops.addFirst(new String[]{"=", String.valueOf(a.charAt(i - 1)), String.valueOf(b.charAt(j - 1))});
                i--; j--;
            } else if (i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1) {
                ops.addFirst(new String[]{"!", String.valueOf(a.charAt(i - 1)), String.valueOf(b.charAt(j - 1))});
                i--; j--;
            } else if (i > 0 && dp[i][j] == dp[i - 1][j] + 1) {
                ops.addFirst(new String[]{"-", String.valueOf(a.charAt(i - 1)), ""});
                i--;
            } else {
                ops.addFirst(new String[]{"+", "", String.valueOf(b.charAt(j - 1))});
                j--;
            }
        }
        return new AlignResult(dp[n][m], ops);
    }

    static AlignResult hirschbergAlign(String a, String b, int base) {
        if (a.isEmpty()) {
            List<String[]> ops = new ArrayList<>();
            for (int k = 0; k < b.length(); k++) ops.add(new String[]{"+", "", String.valueOf(b.charAt(k))});
            return new AlignResult(b.length(), ops);
        }
        if (b.isEmpty()) {
            List<String[]> ops = new ArrayList<>();
            for (int k = 0; k < a.length(); k++) ops.add(new String[]{"-", String.valueOf(a.charAt(k)), ""});
            return new AlignResult(a.length(), ops);
        }
        if (Math.min(a.length(), b.length()) <= Math.max(base, 1)) return directAlign(a, b);
        int mid = b.length() / 2;
        int[] f = forwardRow(a, b.substring(0, mid));
        int[] g = backwardRow(a, b.substring(mid));
        int split = 0, best = Integer.MAX_VALUE;
        for (int i = 0; i <= a.length(); i++) {
            if (f[i] + g[i] < best) { best = f[i] + g[i]; split = i; }
        }
        AlignResult left = hirschbergAlign(a.substring(0, split), b.substring(0, mid), base);
        AlignResult right = hirschbergAlign(a.substring(split), b.substring(mid), base);
        List<String[]> ops = new ArrayList<>(left.ops);
        ops.addAll(right.ops);
        return new AlignResult(left.dist + right.dist, ops);
    }

    /** Apply an alignment to a; returns the produced string. */
    static String replay(String a, List<String[]> ops) {
        StringBuilder out = new StringBuilder();
        int ai = 0;
        for (String[] op : ops) {
            if (op[0].equals("=") || op[0].equals("!") || op[0].equals("-")) {
                if (a.charAt(ai) != op[1].charAt(0)) throw new AssertionError("bad alignment column");
                ai++;
            }
            if (op[0].equals("=") || op[0].equals("!") || op[0].equals("+")) out.append(op[2]);
        }
        if (ai != a.length()) throw new AssertionError("source not fully consumed");
        return out.toString();
    }

    static int countKind(List<String[]> ops, String kind) {
        int c = 0;
        for (String[] op : ops) if (op[0].equals(kind)) c++;
        return c;
    }

    private static int seed = 4242;

    private static String lcgString(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int k = 0; k < length; k++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append(alphabet.charAt(seed % alphabet.length()));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        // hand-derived distances (Levenshtein, substitution cost 1)
        check(fullDpDist("kitten", "sitting") == 3, "kitten sitting oracle");
        check(fullDpDist("flaw", "lawn") == 2, "flaw lawn oracle");
        check(fullDpDist("sunday", "saturday") == 3, "sunday saturday oracle");
        check(fullDpDist("", "") == 0, "empty oracle");
        check(fullDpDist("a", "b") == 1, "single sub");
        check(fullDpDist("abc", "") == 3, "deletions oracle");
        check(fullDpDist("abc", "abc") == 0, "equal oracle");
        check(fullDpDist("abc", "yabd") == 2, "abc yabd oracle");

        AlignResult res = hirschbergAlign("kitten", "sitting", BASE);
        check(res.dist == 3, "kitten sitting hirschberg");
        check(replay("kitten", res.ops).equals("sitting"), "kitten replay");
        check(countKind(res.ops, "=") == 4, "kitten matches forced");
        check(countKind(res.ops, "!") == 2, "kitten subs forced");
        check(countKind(res.ops, "+") == 1, "kitten insert forced");
        check(countKind(res.ops, "-") == 0, "kitten deletes none");

        res = hirschbergAlign("flaw", "lawn", 1);
        check(res.dist == 2, "flaw lawn hirschberg");
        check(replay("flaw", res.ops).equals("lawn"), "flaw replay");
        check(countKind(res.ops, "!") + countKind(res.ops, "-") + countKind(res.ops, "+") == 2, "flaw edits");
        check(countKind(res.ops, "=") == 3, "flaw matches law");

        // base-case paths: empty strings, pure insertions/deletions, equal strings
        AlignResult ins = hirschbergAlign("", "abc", BASE);
        check(ins.dist == 3 && ins.ops.size() == 3, "pure insertions");
        AlignResult del = hirschbergAlign("abc", "", BASE);
        check(del.dist == 3 && del.ops.size() == 3, "pure deletions");
        AlignResult none = hirschbergAlign("", "", BASE);
        check(none.dist == 0 && none.ops.isEmpty(), "empty pair");
        AlignResult same = hirschbergAlign("same", "same", BASE);
        check(same.dist == 0 && replay("same", same.ops).equals("same"), "equal alignment");
        check(countKind(same.ops, "=") == 4, "equal matches");

        // both base settings must agree with the full table on hand and random pairs
        String[][] pairs = {{"kitten", "sitting"}, {"flaw", "lawn"}, {"sunday", "saturday"}, {"abc", "yabd"}};
        int[] bases = {0, 1, 3, 16, 1000};
        for (String[] pair : pairs) {
            for (int base : bases) {
                AlignResult r2 = hirschbergAlign(pair[0], pair[1], base);
                check(r2.dist == fullDpDist(pair[0], pair[1]), "pair " + pair[0] + "/" + pair[1] + " base " + base);
                check(replay(pair[0], r2.ops).equals(pair[1]), "pair replay " + pair[0] + " base " + base);
            }
        }

        String alphabet = "abcd";
        for (int round = 0; round < 10; round++) {
            int la = 30 + seed % 60;
            String a = lcgString(la, alphabet);
            int lb = 25 + seed % 75;
            String b = lcgString(lb, alphabet);
            int want = fullDpDist(a, b);
            for (int base : new int[]{2, 16, 400}) {
                AlignResult r2 = hirschbergAlign(a, b, base);
                check(r2.dist == want, "lcg round " + round + " base " + base);
                check(replay(a, r2.ops).equals(b), "lcg replay " + round + " base " + base);
                check(countKind(r2.ops, "=") + countKind(r2.ops, "!") + countKind(r2.ops, "-") == a.length(), "lcg consume a");
                check(countKind(r2.ops, "=") + countKind(r2.ops, "!") + countKind(r2.ops, "+") == b.length(), "lcg consume b");
                check(countKind(r2.ops, "!") + countKind(r2.ops, "-") + countKind(r2.ops, "+") == want, "lcg cost");
                for (String[] op : r2.ops) {
                    if (op[0].equals("=")) check(op[1].equals(op[2]), "match equal");
                    if (op[0].equals("!")) check(!op[1].equals(op[2]), "sub unequal");
                }
            }
        }

        // recurrence sanity: mid rows of the recursion match the full-table boundary values
        String a = "alignment", b = "alignmentx";
        int mid = b.length() / 2;
        int[] f = forwardRow(a, b.substring(0, mid));
        int[] g = backwardRow(a, b.substring(mid));
        check(f[a.length()] == fullDpDist(a, b.substring(0, mid)), "f whole");
        check(g[0] == fullDpDist(a, b.substring(mid)), "g whole");
        for (int i = 0; i <= a.length(); i++) {
            check(f[i] == fullDpDist(a.substring(0, i), b.substring(0, mid)), "f row " + i);
            check(g[i] == fullDpDist(a.substring(i), b.substring(mid)), "g row " + i);
        }
        int split = 0, best = Integer.MAX_VALUE;
        for (int i = 0; i <= a.length(); i++) {
            if (f[i] + g[i] < best) { best = f[i] + g[i]; split = i; }
        }
        check(best == fullDpDist(a, b), "split optimal");
        check(split >= 0 && split <= a.length(), "split range");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
