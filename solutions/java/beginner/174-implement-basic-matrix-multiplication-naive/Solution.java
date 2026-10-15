/** Task 174 - Implement basic matrix multiplication (naive). (Tier: Beginner, CTCI Ch 6) */
public class Solution {
    static int[][] matmul(int[][] a, int[][] b) {
        if (a.length == 0 || b.length == 0) return new int[0][];
        int n = a.length, m = a[0].length;
        int m2 = b.length, p = b[0].length;
        if (m != m2)
            throw new IllegalArgumentException("dimension mismatch: " + n + "x" + m + " vs " + m2 + "x" + p);
        int[][] c = new int[n][p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++) {
                int total = 0;
                for (int k = 0; k < m; k++) total += a[i][k] * b[k][j];
                c[i][j] = total;
            }
        return c;
    }

    public static void main(String[] args) {
        int[][] product = matmul(new int[][]{{1, 2, 3}, {4, 5, 6}}, new int[][]{{7, 8}, {9, 10}, {11, 12}});
        check(product.length == 2 && product[0][0] == 58 && product[0][1] == 64
                && product[1][0] == 139 && product[1][1] == 154, "2x3 * 3x2");

        int[][] identity = {{1, 0}, {0, 1}};
        int[][] m = {{5, 6}, {7, 8}};
        int[][] withId = matmul(m, identity);
        check(withId[0][0] == 5 && withId[1][1] == 8 && withId[0][1] == 6, "identity");

        check(matmul(new int[][]{{1, 2}}, new int[][]{{3}, {4}})[0][0] == 11, "1x2 * 2x1");

        boolean threw = false;
        try {
            matmul(new int[][]{{1, 2}}, new int[][]{{1, 2}});
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "mismatch rejected");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
