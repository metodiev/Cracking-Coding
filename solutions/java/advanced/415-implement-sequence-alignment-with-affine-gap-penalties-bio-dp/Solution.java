/** Task 415 - Implement sequence alignment with affine gap penalties (bio-DP). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Gotoh's algorithm aligns two strings under affine gap costs. Cost model (documented):
 *   * substitution: MISMATCH = 2, matched characters cost 0,
 *   * a gap of length L costs OPEN + (L - 1) * EXTEND, with OPEN = 2 and EXTEND = 1:
 *     opening the run costs OPEN, every extra character costs EXTEND.
 * Three DP states drive the recurrence:
 *   * X - the column ends with a (mis)match of the last two characters,
 *   * Y - the column ends with a gap in b (a character of a is deleted),
 *   * Z - the column ends with a gap in a (a character of b is inserted).
 * Gap-to-gap switches are allowed so that with EXTEND == OPEN the optimum equals
 * Needleman-Wunsch with a linear gap cost of OPEN per character (asserted below).
 * An alignment is a list of ops {'=', x, y} match; {'!', x, y} substitution;
 * {'-', x, ""} deletion; {'+', "", y} insertion; replaying the ops rebuilds b.
 */
import java.util.*;

class GotohResult {
    final int cost;
    final List<String[]> ops;

    GotohResult(int cost, List<String[]> ops) {
        this.cost = cost;
        this.ops = ops;
    }
}

public class Solution {
    static final int MISMATCH = 2;
    static final int OPEN = 2;
    static final int EXTEND = 1;
    static final int INF = Integer.MAX_VALUE / 4;

    static int argMin3(int a, int b, int c) {
        if (a <= b && a <= c) return 0;
        if (b <= c) return 1;
        return 2;
    }

    /** Returns (cost, ops) for the optimal affine-gap global alignment. */
    static GotohResult gotoh(String a, String b, int mismatch, int openCost, int extend) {
        int n = a.length(), m = b.length();
        int[][] x = new int[n + 1][m + 1];
        int[][] y = new int[n + 1][m + 1];
        int[][] z = new int[n + 1][m + 1];
        for (int[] row : x) Arrays.fill(row, INF);
        for (int[] row : y) Arrays.fill(row, INF);
        for (int[] row : z) Arrays.fill(row, INF);
        x[0][0] = 0;
        for (int i = 1; i <= n; i++) y[i][0] = openCost + (i - 1) * extend;
        for (int j = 1; j <= m; j++) z[0][j] = openCost + (j - 1) * extend;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                int sub = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : mismatch;
                x[i][j] = Math.min(Math.min(x[i - 1][j - 1], y[i - 1][j - 1]), z[i - 1][j - 1]) + sub;
                y[i][j] = Math.min(Math.min(x[i - 1][j] + openCost, z[i - 1][j] + openCost), y[i - 1][j] + extend);
                z[i][j] = Math.min(Math.min(x[i][j - 1] + openCost, y[i][j - 1] + openCost), z[i][j - 1] + extend);
            }
        }
        int state = argMin3(x[n][m], y[n][m], z[n][m]);
        int cost = state == 0 ? x[n][m] : state == 1 ? y[n][m] : z[n][m];
        LinkedList<String[]> ops = new LinkedList<>();
        int i = n, j = m;
        while (i > 0 || j > 0) {
            if (state == 0) {
                int sub = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : mismatch;
                ops.addFirst(new String[]{sub == 0 ? "=" : "!", String.valueOf(a.charAt(i - 1)), String.valueOf(b.charAt(j - 1))});
                int k = argMin3(x[i - 1][j - 1], y[i - 1][j - 1], z[i - 1][j - 1]);
                int cand = k == 0 ? x[i - 1][j - 1] : k == 1 ? y[i - 1][j - 1] : z[i - 1][j - 1];
                if (cand + sub != x[i][j]) throw new AssertionError("X backtrack");
                state = k;
                i--;
                j--;
            } else if (state == 1) {
                ops.addFirst(new String[]{"-", String.valueOf(a.charAt(i - 1)), ""});
                int k = argMin3(x[i - 1][j] + openCost, z[i - 1][j] + openCost, y[i - 1][j] + extend);
                int cand = k == 0 ? x[i - 1][j] + openCost : k == 1 ? z[i - 1][j] + openCost : y[i - 1][j] + extend;
                if (cand != y[i][j]) throw new AssertionError("Y backtrack");
                state = k == 0 ? 0 : k == 1 ? 2 : 1;
                i--;
            } else {
                ops.addFirst(new String[]{"+", "", String.valueOf(b.charAt(j - 1))});
                int k = argMin3(x[i][j - 1] + openCost, y[i][j - 1] + openCost, z[i][j - 1] + extend);
                int cand = k == 0 ? x[i][j - 1] + openCost : k == 1 ? y[i][j - 1] + openCost : z[i][j - 1] + extend;
                if (cand != z[i][j]) throw new AssertionError("Z backtrack");
                state = k;
                j--;
            }
        }
        return new GotohResult(cost, ops);
    }

    /** Linear gap cost: every gap character costs `gap` regardless of run length. */
    static int needlemanWunsch(String a, String b, int gap, int mismatch) {
        int n = a.length(), m = b.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++) dp[i][0] = i * gap;
        for (int j = 1; j <= m; j++) dp[0][j] = j * gap;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                int sub = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : mismatch;
                dp[i][j] = Math.min(Math.min(dp[i - 1][j] + gap, dp[i][j - 1] + gap), dp[i - 1][j - 1] + sub);
            }
        }
        return dp[n][m];
    }

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

    /** Recompute the cost from the op pattern alone, validating the cost model. */
    static int opsCost(List<String[]> ops, int mismatch, int openCost, int extend) {
        int total = 0, i = 0;
        while (i < ops.size()) {
            String kind = ops.get(i)[0];
            if (kind.equals("!")) {
                total += mismatch;
                i++;
            } else if (kind.equals("-") || kind.equals("+")) {
                int run = 0;
                while (i < ops.size() && ops.get(i)[0].equals(kind)) { run++; i++; }
                total += openCost + (run - 1) * extend;
            } else {
                i++;
            }
        }
        return total;
    }

    private static int seed = 20270;

    private static String lcgString(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int k = 0; k < length; k++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append(alphabet.charAt(seed % alphabet.length()));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        // equal strings align for free
        GotohResult res = gotoh("GATTACA", "GATTACA", MISMATCH, OPEN, EXTEND);
        check(res.cost == 0 && res.ops.size() == 7, "equal strings");
        for (String[] op : res.ops) check(op[0].equals("="), "equal all matches");

        // a single substitution (2) beats delete+insert (2 + 2)
        res = gotoh("a", "b", MISMATCH, OPEN, EXTEND);
        check(res.cost == 2 && res.ops.size() == 1 && res.ops.get(0)[0].equals("!"), "single sub");

        // one insertion: a single gap of length 1 costs OPEN = 2
        res = gotoh("ACGT", "ACCGT", MISMATCH, OPEN, EXTEND);
        check(res.cost == 2 && opsCost(res.ops, MISMATCH, OPEN, EXTEND) == 2, "one insertion");
        check(replay("ACGT", res.ops).equals("ACCGT"), "one insertion replay");

        // one gap of length 2 costs OPEN + EXTEND = 3; two separate gaps would cost 4
        res = gotoh("acgt", "acgggt", MISMATCH, OPEN, EXTEND);
        check(res.cost == 3, "gap 2 cost");
        check(replay("acgt", res.ops).equals("acgggt"), "gap 2 replay");
        check(opsCost(res.ops, MISMATCH, OPEN, EXTEND) == 3, "gap 2 ops cost");
        check(needlemanWunsch("acgt", "acgggt", 2, MISMATCH) == 4, "linear two gaps");
        check(needlemanWunsch("ab", "aZZb", 2, MISMATCH) == 4, "linear ab");
        GotohResult ab = gotoh("ab", "aZZb", MISMATCH, OPEN, EXTEND);
        check(ab.cost == 3 && opsCost(ab.ops, MISMATCH, OPEN, EXTEND) == 3, "affine ab");
        check(gotoh("acgt", "acgggt", 2, 2, 2).cost == needlemanWunsch("acgt", "acgggt", 2, 2), "extend equal open");

        // empty-string cases: a gap of length L costs OPEN + (L-1)*EXTEND
        GotohResult empty = gotoh("", "", MISMATCH, OPEN, EXTEND);
        check(empty.cost == 0 && empty.ops.isEmpty(), "empty pair");
        res = gotoh("abc", "", MISMATCH, OPEN, EXTEND);
        check(res.cost == OPEN + 2 * EXTEND && res.cost == 4 && res.ops.size() == 3, "pure deletion");
        for (String[] op : res.ops) check(op[0].equals("-"), "deletion ops");
        res = gotoh("", "abc", MISMATCH, OPEN, EXTEND);
        check(res.cost == 4 && res.ops.size() == 3, "pure insertion");
        check(gotoh("", "AAAAA", MISMATCH, OPEN, EXTEND).cost == OPEN + 4 * EXTEND, "long insertion 6");
        check(gotoh("", "AAAAA", MISMATCH, OPEN, EXTEND).cost == 6, "long insertion value");
        check(needlemanWunsch("", "AAAAA", 2, MISMATCH) == 10, "linear long insertion");

        // longer hand case: 3-character insertion in the middle
        res = gotoh("ABCD", "ABXYZCD", MISMATCH, OPEN, EXTEND);
        check(res.cost == OPEN + 2 * EXTEND && res.cost == 4, "middle insertion cost");
        check(replay("ABCD", res.ops).equals("ABXYZCD"), "middle insertion replay");
        check(opsCost(res.ops, MISMATCH, OPEN, EXTEND) == 4, "middle insertion ops cost");

        // replay + cost-model consistency on the hand pairs
        String[][] hand = {{"kitten", "sitting"}, {"sunday", "saturday"}, {"flaw", "lawn"}};
        for (String[] pair : hand) {
            GotohResult r2 = gotoh(pair[0], pair[1], MISMATCH, OPEN, EXTEND);
            check(replay(pair[0], r2.ops).equals(pair[1]), "hand replay " + pair[0]);
            check(opsCost(r2.ops, MISMATCH, OPEN, EXTEND) == r2.cost, "hand ops cost " + pair[0]);
        }

        // with EXTEND == OPEN the affine optimum equals Needleman-Wunsch
        String alphabet = "acgt";
        for (int round = 0; round < 8; round++) {
            int la = 25 + seed % 40;
            String a = lcgString(la, alphabet);
            int lb = 20 + seed % 45;
            String b = lcgString(lb, alphabet);
            GotohResult affine = gotoh(a, b, 2, 3, 3);
            int linear = needlemanWunsch(a, b, 3, 2);
            check(affine.cost == linear, "affine == linear round " + round);
            check(replay(a, affine.ops).equals(b), "lcg replay " + round);
            check(opsCost(affine.ops, 2, 3, 3) == affine.cost, "lcg ops cost " + round);
            // a cheaper affine extension can never lose against the linear model
            check(gotoh(a, b, 2, 2, 1).cost <= linear, "affine bound round " + round);
        }

        // gap-run counting sanity: a long deletion is one open plus many extends
        res = gotoh("ABCDEFGH", "AB", MISMATCH, OPEN, EXTEND);
        check(res.cost == OPEN + 5 * EXTEND && res.cost == 7, "long deletion cost");
        int deletes = 0;
        for (String[] op : res.ops) if (op[0].equals("-")) deletes++;
        check(deletes == 6 && opsCost(res.ops, MISMATCH, OPEN, EXTEND) == 7, "long deletion run");
        check(replay("ABCDEFGH", res.ops).equals("AB"), "long deletion replay");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
