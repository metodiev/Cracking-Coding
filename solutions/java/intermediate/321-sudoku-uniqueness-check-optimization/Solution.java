/** Task 321 - Sudoku uniqueness check optimization. (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

class SudokuCounter {
    static final int FULL = 0x1FF; // nine candidate bits

    private final int[] grid = new int[81];
    private final int[] rows = new int[9];
    private final int[] cols = new int[9];
    private final int[] boxes = new int[9];
    private final List<int[]> trail = new ArrayList<>();
    private final List<int[]> found = new ArrayList<>();
    private final int limit;

    SudokuCounter(int[] board, int limit) {
        System.arraycopy(board, 0, grid, 0, 81);
        this.limit = limit;
    }

    static int boxOf(int cell) {
        return (cell / 27) * 3 + (cell % 9) / 3;
    }

    private void place(int cell, int bit) {
        rows[cell / 9] |= bit;
        cols[cell % 9] |= bit;
        boxes[boxOf(cell)] |= bit;
        grid[cell] = Integer.numberOfTrailingZeros(bit) + 1;
        trail.add(new int[]{cell, bit});
    }

    private void undoTo(int mark) {
        while (trail.size() > mark) {
            int[] step = trail.remove(trail.size() - 1);
            int cell = step[0], bit = step[1];
            grid[cell] = 0;
            rows[cell / 9] ^= bit;
            cols[cell % 9] ^= bit;
            boxes[boxOf(cell)] ^= bit;
        }
    }

    private int candidates(int cell) {
        return FULL & ~(rows[cell / 9] | cols[cell % 9] | boxes[boxOf(cell)]);
    }

    /** Naked singles: a peer left with a single candidate is forced; false on contradiction. */
    private boolean propagate(int[] seeds) {
        Deque<Integer> work = new ArrayDeque<>();
        for (int seed : seeds) work.push(seed);
        while (!work.isEmpty()) {
            int cell = work.pop();
            for (int peer = 0; peer < 81; peer++) {
                if (peer == cell || grid[peer] != 0) continue;
                boolean sameUnit = peer / 9 == cell / 9 || peer % 9 == cell % 9 || boxOf(peer) == boxOf(cell);
                if (!sameUnit) continue;
                int cand = candidates(peer);
                if (cand == 0) return false;
                if ((cand & (cand - 1)) == 0) { // exactly one bit set -> forced
                    place(peer, cand);
                    work.push(peer);
                }
            }
        }
        return true;
    }

    int run() {
        for (int cell = 0; cell < 81; cell++) {
            if (grid[cell] != 0) {
                int bit = 1 << (grid[cell] - 1);
                if ((candidates(cell) & bit) == 0) return 0; // the givens already conflict
                place(cell, bit);
            }
        }
        int[] allSeeds = new int[81];
        for (int i = 0; i < 81; i++) allSeeds[i] = i;
        if (!propagate(allSeeds)) return 0;
        search();
        return found.size();
    }

    private void search() {
        int bestCell = -1, bestCand = 0, bestSize = 10;
        for (int cell = 0; cell < 81; cell++) {
            if (grid[cell] != 0) continue;
            int cand = candidates(cell);
            if (cand == 0) return; // dead end
            int size = Integer.bitCount(cand);
            if (size < bestSize) {
                bestCell = cell;
                bestCand = cand;
                bestSize = size;
                if (size == 2) break; // cannot do better
            }
        }
        if (bestCell == -1) {
            found.add(grid.clone());
            return;
        }
        while (bestCand != 0) {
            int bit = bestCand & -bestCand;
            bestCand ^= bit;
            int mark = trail.size();
            place(bestCell, bit);
            if (propagate(new int[]{bestCell})) search();
            undoTo(mark);
            if (found.size() >= limit) return;
        }
    }

    int[] firstFound() {
        return found.isEmpty() ? null : found.get(0);
    }
}

public class Solution {
    static int countSolutions(int[] board, int limit) {
        return new SudokuCounter(board, limit).run();
    }

    static int uniqueness(int[] board) {
        return new SudokuCounter(board, 2).run();
    }

    static int[] firstSolution(int[] board) {
        SudokuCounter counter = new SudokuCounter(board, 1);
        counter.run();
        return counter.firstFound();
    }

    /** The 27 units (rows, columns, boxes) as lists of nine values. */
    static List<int[]> units(int[] board) {
        List<int[]> result = new ArrayList<>();
        for (int r = 0; r < 9; r++) {
            int[] unit = new int[9];
            for (int c = 0; c < 9; c++) unit[c] = board[r * 9 + c];
            result.add(unit);
        }
        for (int c = 0; c < 9; c++) {
            int[] unit = new int[9];
            for (int r = 0; r < 9; r++) unit[r] = board[r * 9 + c];
            result.add(unit);
        }
        for (int boxRow = 0; boxRow < 3; boxRow++) {
            for (int boxCol = 0; boxCol < 3; boxCol++) {
                int[] unit = new int[9];
                int k = 0;
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        unit[k++] = board[(boxRow * 3 + r) * 9 + boxCol * 3 + c];
                result.add(unit);
            }
        }
        return result;
    }

    static boolean isSolvedSudoku(int[] board) {
        for (int[] unit : units(board)) {
            int[] sorted = unit.clone();
            Arrays.sort(sorted);
            for (int v = 1; v <= 9; v++) if (sorted[v - 1] != v) return false;
        }
        return true;
    }

    static boolean unitsAreDuplicateFree(int[] board) {
        for (int[] unit : units(board)) {
            Set<Integer> seen = new HashSet<>();
            for (int value : unit) if (value != 0 && !seen.add(value)) return false;
        }
        return true;
    }

    static final int[] PUZZLE = {
        5, 3, 0, 0, 7, 0, 0, 0, 0,
        6, 0, 0, 1, 9, 5, 0, 0, 0,
        0, 9, 8, 0, 0, 0, 0, 6, 0,
        8, 0, 0, 0, 6, 0, 0, 0, 3,
        4, 0, 0, 8, 0, 3, 0, 0, 1,
        7, 0, 0, 0, 2, 0, 0, 0, 6,
        0, 6, 0, 0, 0, 0, 2, 8, 0,
        0, 0, 0, 4, 1, 9, 0, 0, 5,
        0, 0, 0, 0, 8, 0, 0, 7, 9,
    };

    static final int[] KNOWN_SOLUTION = {
        5, 3, 4, 6, 7, 8, 9, 1, 2,
        6, 7, 2, 1, 9, 5, 3, 4, 8,
        1, 9, 8, 3, 4, 2, 5, 6, 7,
        8, 5, 9, 7, 6, 1, 4, 2, 3,
        4, 2, 6, 8, 5, 3, 7, 9, 1,
        7, 1, 3, 9, 2, 4, 8, 5, 6,
        9, 6, 1, 5, 3, 7, 2, 8, 4,
        2, 8, 7, 4, 1, 9, 6, 3, 5,
        3, 4, 5, 2, 8, 6, 1, 7, 9,
    };

    public static void main(String[] args) {
        int[] empty = new int[81];
        // empty board: plenty of solutions, so the capped counter reports "many" (2)
        check(countSolutions(empty, 2) == 2, "empty board has many solutions");
        check(uniqueness(empty) == 2, "empty board uniqueness is many");
        check(countSolutions(empty, 4) == 4, "empty board limit 4");

        // proper puzzle: exactly one completion; with limit 2 the whole tree is explored
        check(countSolutions(PUZZLE, 2) == 1, "proper puzzle is unique");
        check(uniqueness(PUZZLE) == 1, "proper puzzle uniqueness is 1");
        int[] solution = firstSolution(PUZZLE);
        check(Arrays.equals(solution, KNOWN_SOLUTION), "first solution matches the published grid");
        check(isSolvedSudoku(solution), "published grid is a valid sudoku");

        // unsatisfiable board: no completion, and it is not a trivially conflicting one
        int[] unsat = PUZZLE.clone();
        unsat[1] = 1;
        check(unitsAreDuplicateFree(unsat), "unsatisfiable board is unit-consistent");
        check(countSolutions(unsat, 2) == 0, "unsatisfiable board has no solution");
        check(uniqueness(unsat) == 0, "unsatisfiable uniqueness is 0");
        check(firstSolution(unsat) == null, "unsatisfiable has no first solution");

        // a plainly conflicting board is rejected while loading the givens
        int[] conflicting = PUZZLE.clone();
        conflicting[0] = 6; // two 6s in the top-left box
        check(countSolutions(conflicting, 2) == 0, "conflicting board has no solution");

        // sparse valid board (a few clues of the known solution) has multiple completions
        int[] sparse = new int[81];
        for (int index = 0; index < 81; index += 10) sparse[index] = KNOWN_SOLUTION[index];
        check(uniqueness(sparse) == 2, "sparse board has many completions");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
