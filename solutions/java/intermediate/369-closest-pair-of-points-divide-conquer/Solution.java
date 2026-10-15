/** Task 369 - Closest pair of points (divide & conquer). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
import java.util.*;

class ClosestPairSolver {
    // O(n log n) divide & conquer.  Each recursive call returns its slice sorted by y
    // (merged from the children), so the strip is already y-sorted; strip pairs with
    // dy^2 >= best_so_far cannot improve the answer and are skipped.  Distances are
    // squared exact integers -- no epsilon anywhere.
    long best;
    int[] bestA;
    int[] bestB;

    static long dist2(int[] a, int[] b) {
        long dx = a[0] - b[0];
        long dy = a[1] - b[1];
        return dx * dx + dy * dy;
    }

    void solve(int[][] pointsByX) {
        best = -1;
        bestA = bestB = null;
        rec(pointsByX);
    }

    int[][] rec(int[][] ptsByX) {
        int n = ptsByX.length;
        if (n <= 3) {
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    long d = dist2(ptsByX[i], ptsByX[j]);
                    if (best < 0 || d < best) {
                        best = d;
                        bestA = ptsByX[i];
                        bestB = ptsByX[j];
                    }
                }
            }
            return sortByY(ptsByX);
        }
        int mid = n / 2;
        int[][] leftY = rec(Arrays.copyOfRange(ptsByX, 0, mid));
        int[][] rightY = rec(Arrays.copyOfRange(ptsByX, mid, n));
        int[][] merged = mergeByY(leftY, rightY);
        int midX = ptsByX[mid][0];
        List<int[]> strip = new ArrayList<>();
        for (int[] p : merged) {
            long dx = p[0] - midX;
            if (dx * dx < best) strip.add(p);
        }
        for (int i = 0; i < strip.size(); i++) {
            for (int j = i + 1; j < strip.size(); j++) {
                long dy = strip.get(j)[1] - strip.get(i)[1];
                if (dy * dy >= best) break;
                long d = dist2(strip.get(i), strip.get(j));
                if (d < best) {
                    best = d;
                    bestA = strip.get(i);
                    bestB = strip.get(j);
                }
            }
        }
        return merged;
    }

    private static int[][] sortByY(int[][] pts) {
        int[][] copy = pts.clone();
        Arrays.sort(copy, Comparator.<int[]>comparingInt(p -> p[1]).thenComparingInt(p -> p[0]));
        return copy;
    }

    private static int[][] mergeByY(int[][] a, int[][] b) {
        int[][] merged = new int[a.length + b.length][];
        int i = 0, j = 0, k = 0;
        while (i < a.length || j < b.length) {
            if (j >= b.length || (i < a.length && (a[i][1] < b[j][1] ||
                    (a[i][1] == b[j][1] && a[i][0] <= b[j][0])))) {
                merged[k++] = a[i++];
            } else {
                merged[k++] = b[j++];
            }
        }
        return merged;
    }
}

public class Solution {
    private static int seed = 271828;

    private static int[][] sortedPoints(int[][] points) {
        int[][] copy = points.clone();
        Arrays.sort(copy, Comparator.<int[]>comparingInt(p -> p[0]).thenComparingInt(p -> p[1]));
        return copy;
    }

    /** Independent O(n^2) oracle: same (squared distance) semantics. */
    static long bruteForce(int[][] points) {
        long best = -1;
        for (int i = 0; i < points.length; i++) {
            for (int j = i + 1; j < points.length; j++) {
                long d = ClosestPairSolver.dist2(points[i], points[j]);
                if (best < 0 || d < best) best = d;
            }
        }
        return best;
    }

    static long closestPairSq(int[][] points) {
        if (points.length < 2) throw new IllegalArgumentException("need at least two points");
        ClosestPairSolver solver = new ClosestPairSolver();
        solver.solve(sortedPoints(points));
        return solver.best;
    }

    public static void main(String[] args) {
        // Hand case: (0,0)-(3,4) and (3,4)-(7,1) both have squared distance 25.
        int[][] hand = {{0, 0}, {3, 4}, {7, 1}};
        check(closestPairSq(hand) == 25, "hand case 25");
        check(bruteForce(hand) == 25, "hand case brute force 25");

        // Duplicate point: distance 0.
        int[][] dup = {{5, 5}, {5, 5}, {9, 9}};
        check(closestPairSq(dup) == 0, "duplicate distance 0");
        check(bruteForce(dup) == 0, "duplicate brute force");

        int[][] pair = {{0, 0}, {-3, -4}};
        check(closestPairSq(pair) == 25, "two points 3-4-5");

        int[][] grid = new int[16][];
        int at = 0;
        for (int x = 0; x < 4; x++) {
            for (int y = 0; y < 4; y++) grid[at++] = new int[]{x, y};
        }
        check(closestPairSq(grid) == 1 && bruteForce(grid) == 1, "unit grid");

        // LCG cross-check: sizes up to ~60.
        for (int trial = 0; trial < 30; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int size = 2 + seed % 59;
            int modulus = 10 + seed % 90;
            int[][] points = new int[size][];
            for (int i = 0; i < size; i++) {
                seed = (seed * 75 + 74) % 65537;
                int x = seed % modulus;
                seed = (seed * 75 + 74) % 65537;
                int y = seed % modulus;
                points[i] = new int[]{x, y};
            }
            check(closestPairSq(points) == bruteForce(points),
                    "random trial " + trial + " size " + size);
        }

        // Concentrated points: many duplicates and very small distances.
        for (int trial = 0; trial < 10; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int size = 2 + seed % 30;
            int[][] points = new int[size][];
            for (int i = 0; i < size; i++) {
                seed = (seed * 75 + 74) % 65537;
                int x = seed % 5;
                seed = (seed * 75 + 74) % 65537;
                int y = seed % 5;
                points[i] = new int[]{x, y};
            }
            check(closestPairSq(points) == bruteForce(points), "dense trial " + trial);
        }

        // Worst-case strip shape: a vertical line.
        int[][] line = new int[60][];
        for (int i = 0; i < 60; i++) line[i] = new int[]{7, i};
        check(closestPairSq(line) == 1, "vertical line");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
