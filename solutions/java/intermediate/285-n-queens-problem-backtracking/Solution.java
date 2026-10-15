/** Task 285 - N-Queens via backtracking. (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    /**
     * All solutions as lists of column indices per row, in lexicographic order:
     * queens are placed row by row with columns tried 0..n-1.
     */
    static List<List<Integer>> solveNQueens(int n) {
        List<List<Integer>> solutions = new ArrayList<>();
        List<Integer> columns = new ArrayList<>();
        place(0, n, columns, solutions);
        return solutions;
    }

    private static void place(int row, int n, List<Integer> columns, List<List<Integer>> solutions) {
        if (row == n) {
            solutions.add(new ArrayList<>(columns));
            return;
        }
        for (int col = 0; col < n; col++) {
            boolean ok = true;
            for (int pr = 0; pr < columns.size(); pr++) {
                int pc = columns.get(pr);
                if (col == pc || Math.abs(col - pc) == row - pr) { ok = false; break; }
            }
            if (ok) {
                columns.add(col);
                place(row + 1, n, columns, solutions);
                columns.remove(columns.size() - 1);
            }
        }
    }

    static int countSolutions(int n) {
        return solveNQueens(n).size();
    }

    /** Lexicographically first solution, or null when no placement exists. */
    static int[] firstSolution(int n) {
        List<List<Integer>> solutions = solveNQueens(n);
        if (solutions.isEmpty()) return null;
        List<Integer> first = solutions.get(0);
        int[] result = new int[first.size()];
        for (int i = 0; i < result.length; i++) result[i] = first.get(i);
        return result;
    }

    /** One queen per row is structural; check columns are a permutation and no shared diagonal. */
    static boolean isValidSolution(int[] solution) {
        int n = solution.length;
        boolean[] seen = new boolean[n];
        for (int col : solution) {
            if (col < 0 || col >= n || seen[col]) return false;
            seen[col] = true;
        }
        for (int r = 0; r < n; r++)
            for (int s = r + 1; s < n; s++)
                if (Math.abs(solution[r] - solution[s]) == s - r) return false;
        return true;
    }

    /** Permutation oracle: count column permutations with no shared diagonal. */
    static int bruteForceCount(int n) {
        int[] permutation = new int[n];
        boolean[] used = new boolean[n];
        return countPermutations(0, n, permutation, used);
    }

    private static int countPermutations(int row, int n, int[] permutation, boolean[] used) {
        if (row == n) {
            for (int r = 0; r < n; r++)
                for (int s = r + 1; s < n; s++)
                    if (Math.abs(permutation[r] - permutation[s]) == s - r) return 0;
            return 1;
        }
        int total = 0;
        for (int col = 0; col < n; col++) {
            if (used[col]) continue;
            used[col] = true;
            permutation[row] = col;
            total += countPermutations(row + 1, n, permutation, used);
            used[col] = false;
        }
        return total;
    }

    public static void main(String[] args) {
        check(countSolutions(1) == 1, "n=1");
        check(countSolutions(2) == 0, "n=2");
        check(countSolutions(3) == 0, "n=3");
        check(countSolutions(4) == 2, "n=4");
        check(countSolutions(5) == 10, "n=5");
        check(countSolutions(6) == 4, "n=6");
        check(countSolutions(7) == 40, "n=7");
        check(countSolutions(8) == 92, "n=8");

        check(Arrays.equals(firstSolution(1), new int[]{0}), "first n=1");
        check(Arrays.equals(firstSolution(4), new int[]{1, 3, 0, 2}), "first n=4");
        check(firstSolution(2) == null, "first n=2 none");
        check(firstSolution(3) == null, "first n=3 none");

        for (int n = 1; n <= 8; n++) {
            List<List<Integer>> solutions = solveNQueens(n);
            check(solutions.size() == countSolutions(n), "count matches list " + n);
            for (int i = 1; i < solutions.size(); i++) {
                List<Integer> previous = solutions.get(i - 1), current = solutions.get(i);
                for (int k = 0; k < n; k++) {
                    if (current.get(k) < previous.get(k)) throw new AssertionError("FAILED: order " + n);
                    if (current.get(k) > previous.get(k)) break;
                }
            }
            for (List<Integer> solution : solutions) {
                int[] array = new int[n];
                for (int i = 0; i < n; i++) array[i] = solution.get(i);
                check(isValidSolution(array), "valid " + n + " " + solution);
            }
            check(bruteForceCount(n) == solutions.size(), "oracle " + n);
        }

        for (int n = 1; n <= 8; n++) {
            List<List<Integer>> solutions = solveNQueens(n);
            if (!solutions.isEmpty()) {
                int[] first = firstSolution(n);
                int[] expected = solutions.get(0).stream().mapToInt(Integer::intValue).toArray();
                check(Arrays.equals(first, expected), "first solution " + n);
            }
        }

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
