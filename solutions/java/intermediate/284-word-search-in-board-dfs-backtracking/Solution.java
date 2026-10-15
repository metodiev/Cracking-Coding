/** Task 284 - Word search in a board with DFS backtracking. (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

class WordCase {
    String[] board;
    String word;

    WordCase(String[] board, String word) {
        this.board = board;
        this.word = word;
    }
}

public class Solution {
    private static final int[][] DIRECTIONS = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    private static boolean[][] visited;
    private static String[] currentBoard;
    private static String currentWord;

    /** Backtracking DFS; cells cannot be reused. Returns true iff such a path exists. */
    static boolean exist(String[] board, String word) {
        if (word.isEmpty()) return false;
        currentBoard = board;
        currentWord = word;
        visited = new boolean[board.length][board[0].length()];
        for (int r = 0; r < board.length; r++)
            for (int c = 0; c < board[0].length(); c++)
                if (search(r, c, 0)) return true;
        return false;
    }

    private static boolean search(int r, int c, int i) {
        if (i == currentWord.length()) return true;
        if (r < 0 || r >= currentBoard.length || c < 0 || c >= currentBoard[0].length()) return false;
        if (visited[r][c] || currentBoard[r].charAt(c) != currentWord.charAt(i)) return false;
        visited[r][c] = true;
        boolean found = false;
        for (int[] d : DIRECTIONS) {
            if (search(r + d[0], c + d[1], i + 1)) { found = true; break; }
        }
        visited[r][c] = false;                     // unmark: later branches may need this cell
        return found;
    }

    private static boolean[][] buggyVisited;
    private static String[] buggyBoard;
    private static String buggyWord;

    /** Deliberately never unmarks cells; only used to prove the trap case needs backtracking. */
    static boolean buggyExistWithoutUnmarking(String[] board, String word) {
        buggyBoard = board;
        buggyWord = word;
        buggyVisited = new boolean[board.length][board[0].length()];
        for (int r = 0; r < board.length; r++)
            for (int c = 0; c < board[0].length(); c++)
                if (buggySearch(r, c, 0)) return true;
        return false;
    }

    private static boolean buggySearch(int r, int c, int i) {
        if (i == buggyWord.length()) return true;
        if (r < 0 || r >= buggyBoard.length || c < 0 || c >= buggyBoard[0].length()) return false;
        if (buggyVisited[r][c] || buggyBoard[r].charAt(c) != buggyWord.charAt(i)) return false;
        buggyVisited[r][c] = true;
        for (int[] d : DIRECTIONS) {
            if (buggySearch(r + d[0], c + d[1], i + 1)) return true;
        }
        return false;
    }

    /** Independent BFS over states (row, col, visited-cell mask). */
    static boolean oracleExist(String[] board, String word) {
        int rows = board.length, cols = board[0].length();
        int shift = rows * cols;
        Deque<int[]> queue = new ArrayDeque<>();
        Set<Integer> seen = new HashSet<>();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (board[r].charAt(c) == word.charAt(0)) {
                    int state = ((r * cols + c) << shift) | (1 << (r * cols + c));
                    if (seen.add(state)) queue.add(new int[]{r, c, 1 << (r * cols + c)});
                }
            }
        }
        while (!queue.isEmpty()) {
            int[] state = queue.poll();
            int r = state[0], c = state[1], mask = state[2];
            int i = Integer.bitCount(mask);
            if (i == word.length()) return true;
            for (int[] d : DIRECTIONS) {
                int nr = r + d[0], nc = c + d[1];
                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
                if (((mask >> (nr * cols + nc)) & 1) != 0) continue;
                if (board[nr].charAt(nc) != word.charAt(i)) continue;
                int nextMask = mask | (1 << (nr * cols + nc));
                int encoded = ((nr * cols + nc) << shift) | nextMask;
                if (seen.add(encoded)) queue.add(new int[]{nr, nc, nextMask});
            }
        }
        return false;
    }

    private static int seed = 13579;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static String[] randomBoard(int rows, int cols) {
        String[] board = new String[rows];
        for (int r = 0; r < rows; r++) {
            StringBuilder row = new StringBuilder();
            for (int c = 0; c < cols; c++) row.append("ABC".charAt(nextSeed() % 3));
            board[r] = row.toString();
        }
        return board;
    }

    private static WordCase randomCase(int rows, int cols, int maxLen) {
        String[] board = randomBoard(rows, cols);
        int length = 1 + nextSeed() % maxLen;
        StringBuilder word = new StringBuilder();
        for (int i = 0; i < length; i++) word.append("ABC".charAt(nextSeed() % 3));
        return new WordCase(board, word.toString());
    }

    /** Reads a random simple path, so the word is guaranteed to exist. */
    private static WordCase plantedCase(int rows, int cols, int maxLen) {
        String[] board = randomBoard(rows, cols);
        int r = nextSeed() % rows, c = nextSeed() % cols;
        boolean[][] used = new boolean[rows][cols];
        used[r][c] = true;
        StringBuilder word = new StringBuilder();
        word.append(board[r].charAt(c));
        int length = 1 + nextSeed() % maxLen;
        while (word.length() < length) {
            List<int[]> candidates = new ArrayList<>();
            for (int[] d : DIRECTIONS) {
                int nr = r + d[0], nc = c + d[1];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !used[nr][nc]) candidates.add(new int[]{nr, nc});
            }
            if (candidates.isEmpty()) break;
            int[] pick = candidates.get(nextSeed() % candidates.size());
            r = pick[0];
            c = pick[1];
            used[r][c] = true;
            word.append(board[r].charAt(c));
        }
        return new WordCase(board, word.toString());
    }

    public static void main(String[] args) {
        String[] classic = {"ABCE", "SFCS", "ADEE"};
        check(exist(classic, "ABCCED"), "ABCCED found");
        check(exist(classic, "SEE"), "SEE found");
        check(!exist(classic, "ABCB"), "ABCB not found");

        String[] trap = {"ABCE", "SFES", "ADEE"};  // needs the visited-unmarking
        check(exist(trap, "ABCESEEEFS"), "trap found with backtracking");
        check(!buggyExistWithoutUnmarking(trap, "ABCESEEEFS"), "trap needs unmarking");

        check(exist(new String[]{"A"}, "A"), "single cell");
        check(!exist(new String[]{"A"}, "B"), "single cell wrong letter");
        check(!exist(new String[]{"A"}, "AA"), "cell cannot be reused");

        check(exist(new String[]{"AB", "CD"}, "ABDC"), "2x2 path");
        check(!exist(new String[]{"AB", "CD"}, "ABCD"), "2x2 not adjacent");

        for (int trial = 0; trial < 150; trial++) {
            WordCase testCase = randomCase(3, 3, 4);
            check(exist(testCase.board, testCase.word) == oracleExist(testCase.board, testCase.word),
                  "random " + trial + " " + testCase.word);
        }

        for (int trial = 0; trial < 50; trial++) {
            WordCase testCase = plantedCase(3, 3, 5);
            check(exist(testCase.board, testCase.word), "planted " + trial);
            check(oracleExist(testCase.board, testCase.word), "planted oracle " + trial);
        }

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
