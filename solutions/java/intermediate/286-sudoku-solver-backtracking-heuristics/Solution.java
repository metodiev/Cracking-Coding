/** Task 286 - Sudoku solver with backtracking and constraint propagation. (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    private static final int FULL = 0x1FF;    // bits 0..8 for digits 1..9

    private static final int[][] PUZZLE = {
        {5, 3, 0, 0, 7, 0, 0, 0, 0},
        {6, 0, 0, 1, 9, 5, 0, 0, 0},
        {0, 9, 8, 0, 0, 0, 0, 6, 0},
        {8, 0, 0, 0, 6, 0, 0, 0, 3},
        {4, 0, 0, 8, 0, 3, 0, 0, 1},
        {7, 0, 0, 0, 2, 0, 0, 0, 6},
        {0, 6, 0, 0, 0, 0, 2, 8, 0},
        {0, 0, 0, 4, 1, 9, 0, 0, 5},
        {0, 0, 0, 0, 8, 0, 0, 7, 9},
    };

    private static final int[][] SOLUTION = {
        {5, 3, 4, 6, 7, 8, 9, 1, 2},
        {6, 7, 2, 1, 9, 5, 3, 4, 8},
        {1, 9, 8, 3, 4, 2, 5, 6, 7},
        {8, 5, 9, 7, 6, 1, 4, 2, 3},
        {4, 2, 6, 8, 5, 3, 7, 9, 1},
        {7, 1, 3, 9, 2, 4, 8, 5, 6},
        {9, 6, 1, 5, 3, 7, 2, 8, 4},
        {2, 8, 7, 4, 1, 9, 6, 3, 5},
        {3, 4, 5, 2, 8, 6, 1, 7, 9},
    };

    private static int[][] grid;
    private static int[] rowsMask, colsMask, boxesMask;

    /**
     * Candidates come from row/column/box bitmasks and the search always branches on the
     * most-constrained empty cell (MRV). Returns the solved board, or null if unsolvable.
     */
    static int[][] solveSudoku(int[][] board) {
        grid = new int[9][9];
        for (int r = 0; r < 9; r++) grid[r] = board[r].clone();
        rowsMask = new int[9];
        colsMask = new int[9];
        boxesMask = new int[9];
        for (int r = 0; r < 9; r++) {
            for (int c = 0; c < 9; c++) {
                int value = grid[r][c];
                if (value < 0 || value > 9) return null;
                if (value == 0) continue;
                int bit = 1 << (value - 1);
                int box = (r / 3) * 3 + c / 3;
                if ((rowsMask[r] & bit) != 0 || (colsMask[c] & bit) != 0 || (boxesMask[box] & bit) != 0) {
                    return null;                  // duplicate clue
                }
                rowsMask[r] |= bit;
                colsMask[c] |= bit;
                boxesMask[box] |= bit;
            }
        }
        return search() ? grid : null;
    }

    private static boolean search() {
        int bestR = -1, bestC = -1, bestMask = 0, bestCount = 10;
        for (int r = 0; r < 9; r++) {
            for (int c = 0; c < 9; c++) {
                if (grid[r][c] != 0) continue;
                int box = (r / 3) * 3 + c / 3;
                int mask = ~(rowsMask[r] | colsMask[c] | boxesMask[box]) & FULL;
                int count = Integer.bitCount(mask);
                if (count == 0) return false;     // dead end found by propagation
                if (count < bestCount) {          // most-constrained cell
                    bestCount = count;
                    bestR = r;
                    bestC = c;
                    bestMask = mask;
                }
            }
        }
        if (bestR == -1) return true;             // no empty cell left
        int box = (bestR / 3) * 3 + bestC / 3;
        int mask = bestMask;
        while (mask != 0) {
            int bit = mask & -mask;
            mask ^= bit;
            grid[bestR][bestC] = Integer.numberOfTrailingZeros(bit) + 1;
            rowsMask[bestR] |= bit;
            colsMask[bestC] |= bit;
            boxesMask[box] |= bit;
            if (search()) return true;
            grid[bestR][bestC] = 0;
            rowsMask[bestR] &= ~bit;
            colsMask[bestC] &= ~bit;
            boxesMask[box] &= ~bit;
        }
        return false;
    }

    /** Every row, column and 3x3 box must contain the digits 1..9 exactly once. */
    static boolean isValidSolution(int[][] board) {
        boolean[] seen = new boolean[10];
        for (int r = 0; r < 9; r++) {
            Arrays.fill(seen, false);
            for (int c = 0; c < 9; c++) {
                int value = board[r][c];
                if (value < 1 || value > 9 || seen[value]) return false;
                seen[value] = true;
            }
        }
        for (int c = 0; c < 9; c++) {
            Arrays.fill(seen, false);
            for (int r = 0; r < 9; r++) {
                int value = board[r][c];
                if (seen[value]) return false;
                seen[value] = true;
            }
        }
        for (int boxR = 0; boxR < 9; boxR += 3) {
            for (int boxC = 0; boxC < 9; boxC += 3) {
                Arrays.fill(seen, false);
                for (int dr = 0; dr < 3; dr++) {
                    for (int dc = 0; dc < 3; dc++) {
                        int value = board[boxR + dr][boxC + dc];
                        if (seen[value]) return false;
                        seen[value] = true;
                    }
                }
            }
        }
        return true;
    }

    private static int seed = 13579;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static int[] randomPermutation(int size) {
        int[] values = new int[size];
        for (int i = 0; i < size; i++) values[i] = i;
        for (int i = size - 1; i > 0; i--) {
            int j = nextSeed() % (i + 1);
            int swap = values[i];
            values[i] = values[j];
            values[j] = swap;
        }
        return values;
    }

    private static int[][] transform(int[][] source, int[] digitMap, int[] rowOrder, int[] colOrder) {
        int[][] result = new int[9][9];
        for (int r = 0; r < 9; r++)
            for (int c = 0; c < 9; c++)
                result[r][c] = digitMap[source[rowOrder[r]][colOrder[c]]];
        return result;
    }

    static boolean deepEquals(int[][] a, int[][] b) {
        if (a == null || b == null) return a == b;
        return Arrays.deepEquals(a, b);
    }

    public static void main(String[] args) {
        int[][] solved = solveSudoku(PUZZLE);
        check(deepEquals(solved, SOLUTION), "classic puzzle");
        check(isValidSolution(solved), "solution valid");
        for (int r = 0; r < 9; r++)
            for (int c = 0; c < 9; c++)
                check(PUZZLE[r][c] == 0 || solved[r][c] == PUZZLE[r][c], "clue changed " + r + "," + c);

        check(deepEquals(solveSudoku(SOLUTION), SOLUTION), "already solved input");

        int[][] duplicateInColumn = copy(SOLUTION);
        duplicateInColumn[0][0] = 6;              // column 0 already has a 6
        check(solveSudoku(duplicateInColumn) == null, "duplicate in column");

        int[][] duplicateInRow = copy(SOLUTION);
        duplicateInRow[0][1] = 5;                 // row 0 already has a 5
        check(solveSudoku(duplicateInRow) == null, "duplicate in row");

        int[][] duplicateInBox = copy(SOLUTION);
        duplicateInBox[1][1] = 5;                 // box 0 already has a 5
        check(solveSudoku(duplicateInBox) == null, "duplicate in box");

        int[][] forcedConflict = new int[9][9];
        forcedConflict[0] = new int[]{1, 2, 3, 4, 5, 6, 0, 8, 9};  // (0,6) can only be 7
        forcedConflict[3][6] = 7;                                  // but column 6 already has a 7
        check(solveSudoku(forcedConflict) == null, "empty candidate set without guessing");

        int[][] empty = new int[9][9];
        int[][] filled = solveSudoku(empty);
        check(filled != null && isValidSolution(filled), "empty board");

        for (int trial = 0; trial < 25; trial++) {
            int[] permutation = randomPermutation(9);
            int[] digitMap = new int[10];
            for (int i = 0; i < 9; i++) digitMap[i + 1] = permutation[i] + 1;
            int[] bands = randomPermutation(3);
            int[] stacks = randomPermutation(3);
            int[] rowOrder = new int[9];
            int[] colOrder = new int[9];
            for (int group = 0; group < 3; group++) {
                int[] inner = randomPermutation(3);
                for (int i = 0; i < 3; i++) rowOrder[group * 3 + i] = bands[group] * 3 + inner[i];
                inner = randomPermutation(3);
                for (int i = 0; i < 3; i++) colOrder[group * 3 + i] = stacks[group] * 3 + inner[i];
            }
            int[][] puzzleVariant = transform(PUZZLE, digitMap, rowOrder, colOrder);
            int[][] solutionVariant = transform(SOLUTION, digitMap, rowOrder, colOrder);
            check(isValidSolution(solutionVariant), "variant valid " + trial);
            check(deepEquals(solveSudoku(puzzleVariant), solutionVariant), "variant solved " + trial);
        }

        System.out.println("PASS");
    }

    private static int[][] copy(int[][] source) {
        int[][] result = new int[9][];
        for (int r = 0; r < 9; r++) result[r] = source[r].clone();
        return result;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
