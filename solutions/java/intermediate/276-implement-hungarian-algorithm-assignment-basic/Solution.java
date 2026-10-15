/** Task 276 - Implement the Hungarian algorithm for assignment (basic). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    /** Returns {total cost, assignment...}: result[0] is the cost, result[1+i] the column of row i. */
    static long[] hungarian(int[][] cost) {
        int n = cost.length;
        if (n == 0) return new long[]{0};
        long infinity = Long.MAX_VALUE / 4;
        long[] u = new long[n + 1];
        long[] v = new long[n + 1];
        int[] p = new int[n + 1];
        int[] way = new int[n + 1];

        for (int i = 1; i <= n; i++) {
            p[0] = i;
            int j0 = 0;
            long[] minv = new long[n + 1];
            Arrays.fill(minv, infinity);
            boolean[] used = new boolean[n + 1];
            while (true) {
                used[j0] = true;
                int i0 = p[j0];
                long delta = infinity;
                int j1 = -1;
                for (int j = 1; j <= n; j++) {
                    if (!used[j]) {
                        long cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                    }
                }
                for (int j = 0; j <= n; j++) {
                    if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                    else minv[j] -= delta;
                }
                j0 = j1;
                if (p[j0] == 0) break;
            }
            while (j0 != 0) {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            }
        }

        int[] assignment = new int[n];
        Arrays.fill(assignment, -1);
        for (int j = 1; j <= n; j++) if (p[j] != 0) assignment[p[j] - 1] = j - 1;
        long total = 0;
        for (int i = 0; i < n; i++) total += cost[i][assignment[i]];
        long[] result = new long[n + 1];
        result[0] = total;
        for (int i = 0; i < n; i++) result[i + 1] = assignment[i];
        return result;
    }

    static long bruteForce(int[][] cost) {
        int n = cost.length;
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;
        return permute(cost, perm, 0);
    }

    private static long permute(int[][] cost, int[] perm, int start) {
        int n = perm.length;
        if (start == n) {
            long total = 0;
            for (int i = 0; i < n; i++) total += cost[i][perm[i]];
            return total;
        }
        long best = Long.MAX_VALUE;
        for (int i = start; i < n; i++) {
            int tmp = perm[start]; perm[start] = perm[i]; perm[i] = tmp;
            best = Math.min(best, permute(cost, perm, start + 1));
            tmp = perm[start]; perm[start] = perm[i]; perm[i] = tmp;
        }
        return best;
    }

    private static int seed = 4242;

    private static int[][] lcgMatrix(int n) {
        int[][] matrix = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                seed = (seed * 75 + 74) % 65537;
                matrix[i][j] = seed % 20;
            }
        }
        return matrix;
    }

    public static void main(String[] args) {
        check(Arrays.equals(hungarian(new int[][]{{1,2},{3,4}}), new long[]{5,0,1}), "2x2 tie");
        check(hungarian(new int[][]{{10,19},{20,11}})[0] == 21, "2x2 classic");
        check(hungarian(new int[][]{{1,2,3},{4,5,6},{7,8,9}})[0] == 15, "3x3");
        check(Arrays.equals(hungarian(new int[][]{{7}}), new long[]{7,0}), "1x1");
        check(hungarian(new int[][]{})[0] == 0, "empty");
        check(hungarian(new int[][]{{9,1,1},{1,9,1},{1,1,9}})[0] == 3, "anti-diagonal");

        for (int size = 2; size <= 5; size++) {
            for (int trial = 0; trial < 15; trial++) {
                int[][] matrix = lcgMatrix(size);
                long[] result = hungarian(matrix);
                long expected = bruteForce(matrix);
                check(result[0] == expected, "random size " + size + " trial " + trial);
                boolean[] seen = new boolean[size];
                for (int i = 0; i < size; i++) {
                    int column = (int) result[i + 1];
                    check(column >= 0 && column < size && !seen[column], "permutation " + trial);
                    seen[column] = true;
                }
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
