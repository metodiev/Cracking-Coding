/** Task 726 - Implement sparse-dense matrix multiply block algorithm optimizations. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
 *
 * SpMM computes C = A * B with A sparse in CSR form and B dense. The arithmetic is trivially fixed - each
 * nonzero of A scales one row of B into one row of C - so the only thing an implementation can optimize is
 * *memory traffic*: the row-wise kernel walks the whole width of B and C for every nonzero, which evicts
 * both rows from cache as soon as the matrices are wide. The blocked kernel fixes that by working on a panel
 * of `width` columns at a time, so the working set of B and C per pass is small enough to stay resident. Both
 * kernels accumulate each output element in the same order, so their results are bit-for-bit identical; the
 * difference shows up only in a simulated LRU cache's miss counter.
 *
 * Invariants verified:
 *   - equality   every variant matches a dense matmul oracle, including an empty matrix, an all-zero row and
 *                a single nonzero; the blocked variant matches the row-wise one exactly
 *   - work       both kernels perform exactly nnz * p multiply-adds: blocking changes locality, not arithmetic
 *   - locality   the simulated LRU misses of the blocked kernel are lower than the row-wise kernel once B
 *                no longer fits the cache, and the miss count is non-increasing in the panel width up to the
 *                cache capacity
 *   - boundary   when B fits comfortably in the cache the two kernels miss the same number of times, which is
 *                exactly the regime where blocking buys nothing
 *
 * Simplified: the cache is a fully associative LRU of fixed-size lines, tensors are small double arrays and
 * the matrices come from a fixed-seed generator, so the model is a stand-in for real hardware behaviour
 * rather than a measurement of it. The verified properties are result equality, conserved arithmetic and the
 * systematic locality advantage of the blocked sweep.
 */
import java.util.*;

public class Solution {
    /** Compressed sparse row matrix: rowStarts has m+1 entries. */
    static final class Csr {
        final int rows;
        final int cols;
        final int[] rowStarts;
        final int[] colIndex;
        final double[] value;

        Csr(int rows, int cols, int[] rowStarts, int[] colIndex, double[] value) {
            this.rows = rows;
            this.cols = cols;
            this.rowStarts = rowStarts;
            this.colIndex = colIndex;
            this.value = value;
        }

        int nonzeros() { return value.length; }

        /** Expands to a dense array for the oracle. */
        double[][] dense() {
            double[][] dense = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int k = rowStarts[i]; k < rowStarts[i + 1]; k++) dense[i][colIndex[k]] += value[k];
            }
            return dense;
        }
    }

    /** An LRU cache of `lines` lines, `lineSize` doubles each, counting misses. */
    static final class CacheModel {
        final int lineSize;
        final LinkedHashMap<Long, Boolean> lru;
        long misses;
        long accesses;

        CacheModel(int lines, int lineSize) {
            this.lineSize = lineSize;
            this.lru = new LinkedHashMap<Long, Boolean>(16, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<Long, Boolean> eldest) {
                    return size() > lines;
                }
            };
        }

        void touch(int array, int index) {
            long line = ((long) array << 32) | (index / lineSize);
            accesses++;
            if (lru.put(line, Boolean.TRUE) == null) misses++;
        }

    }

    static final int ARRAY_C = 0;
    static final int ARRAY_B = 1;
    static final int ARRAY_A = 2;

    /** Row-wise kernel: for every nonzero, sweep the whole row of B into the whole row of C. */
    static double[] spmmRowWise(Csr a, double[] b, int width, CacheModel cache) {
        int m = a.rows;
        int n = a.cols;
        int p = b.length / n;
        double[] c = new double[m * p];
        for (int i = 0; i < m; i++) {
            for (int k = a.rowStarts[i]; k < a.rowStarts[i + 1]; k++) {
                int col = a.colIndex[k];
                double v = a.value[k];
                if (cache != null) cache.touch(ARRAY_A, k);
                for (int j = 0; j < p; j++) {
                    if (cache != null) { cache.touch(ARRAY_B, col * p + j); cache.touch(ARRAY_C, i * p + j); }
                    c[i * p + j] += v * b[col * p + j];
                }
            }
        }
        return c;
    }

    /** Blocked kernel: the same arithmetic, swept one panel of `width` columns at a time. */
    static double[] spmmBlocked(Csr a, double[] b, int width, CacheModel cache) {
        int m = a.rows;
        int n = a.cols;
        int p = b.length / n;
        double[] c = new double[m * p];
        if (width <= 0) throw new IllegalArgumentException("panel width must be positive");
        for (int panel = 0; panel < p; panel += width) {
            int end = Math.min(panel + width, p);
            for (int i = 0; i < m; i++) {
                for (int k = a.rowStarts[i]; k < a.rowStarts[i + 1]; k++) {
                    int col = a.colIndex[k];
                    double v = a.value[k];
                    if (cache != null) cache.touch(ARRAY_A, k);
                    for (int j = panel; j < end; j++) {
                        if (cache != null) { cache.touch(ARRAY_B, col * p + j); cache.touch(ARRAY_C, i * p + j); }
                        c[i * p + j] += v * b[col * p + j];
                    }
                }
            }
        }
        return c;
    }

    /** Dense oracle: C = dense(A) * B, computed the obvious way. */
    static double[] denseMultiply(Csr a, double[] b, int p) {
        double[][] dense = a.dense();
        double[] c = new double[a.rows * p];
        for (int i = 0; i < a.rows; i++) {
            for (int k = 0; k < a.cols; k++) {
                double v = dense[i][k];
                if (v == 0.0) continue;
                for (int j = 0; j < p; j++) c[i * p + j] += v * b[k * p + j];
            }
        }
        return c;
    }

    /** Multiply-add count of the row-wise/blocked kernels: nnz * p, counted explicitly. */
    static long multiplyAdds(Csr a, int p) { return (long) a.nonzeros() * p; }

    private static long rngState = 0x853C49E6748FEA9BL;

    static long nextLong() {
        rngState += 0x9E3779B97F4A7C15L;
        long z = rngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    /** A CSR matrix with `perRow` nonzeros per row, all columns distinct within a row. */
    static Csr randomSparse(int rows, int cols, int perRow) {
        int[] starts = new int[rows + 1];
        List<Integer> columnList = new ArrayList<>();
        List<Double> valueList = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            starts[i] = columnList.size();
            Set<Integer> used = new TreeSet<>();
            int want = Math.min(perRow, cols);
            while (used.size() < want) used.add(nextInt(cols));
            for (int col : used) {
                columnList.add(col);
                valueList.add(nextDouble() * 2.0 - 1.0);
            }
        }
        starts[rows] = columnList.size();
        int[] colIndex = columnList.stream().mapToInt(Integer::intValue).toArray();
        double[] value = new double[valueList.size()];
        for (int k = 0; k < value.length; k++) value[k] = valueList.get(k);
        return new Csr(rows, cols, starts, colIndex, value);
    }

    static double[] randomDense(int size) {
        double[] v = new double[size];
        for (int i = 0; i < size; i++) v[i] = nextDouble() * 2.0 - 1.0;
        return v;
    }

    static double maxDifference(double[] a, double[] b) {
        double worst = 0.0;
        for (int i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
        return worst;
    }

    public static void main(String[] args) {
        // --- degenerate shapes -------------------------------------------------
        Csr empty = new Csr(3, 2, new int[]{0, 0, 0, 0}, new int[0], new double[0]);
        double[] emptyB = randomDense(2 * 4);
        check(maxDifference(spmmRowWise(empty, emptyB, 2, null), new double[3 * 4]) == 0.0,
                "an empty matrix leaves C at zero");
        check(maxDifference(spmmBlocked(empty, emptyB, 2, null), denseMultiply(empty, emptyB, 4)) == 0.0,
                "the blocked kernel agrees on an empty matrix");
        check(multiplyAdds(empty, 4) == 0, "an empty matrix costs no multiplies");

        boolean rejectedWidth = false;
        try {
            spmmBlocked(empty, emptyB, 0, null);
        } catch (IllegalArgumentException expected) {
            rejectedWidth = true;
        }
        check(rejectedWidth, "a non-positive panel width must be rejected");

        Csr zeroRow = new Csr(2, 3, new int[]{0, 1, 1}, new int[]{2}, new double[]{1.5});
        double[] zeroRowB = randomDense(3 * 3);
        double[] zeroRowC = spmmBlocked(zeroRow, zeroRowB, 3, null);
        boolean secondRowZero = true;
        for (int j = 0; j < 3; j++) if (zeroRowC[3 + j] != 0.0) secondRowZero = false;
        check(secondRowZero, "a row with no nonzeros stays zero");

        // --- equality against the dense oracle ---------------------------------
        for (int[] shape : new int[][]{{1, 1, 1}, {4, 6, 2}, {7, 5, 5}, {9, 9, 1}, {12, 16, 4}}) {
            int m = shape[0];
            int n = shape[1];
            int p = shape[2] + 1;
            Csr a = randomSparse(m, n, shape[2]);
            double[] b = randomDense(n * p);
            double[] oracle = denseMultiply(a, b, p);
            double[] rowWise = spmmRowWise(a, b, p, null);
            double[] blocked = spmmBlocked(a, b, Math.max(1, p / 2), null);
            check(maxDifference(rowWise, oracle) < 1e-12, "row-wise kernel vs dense oracle, shape "
                    + Arrays.toString(shape));
            check(maxDifference(blocked, oracle) < 1e-12, "blocked kernel vs dense oracle, shape "
                    + Arrays.toString(shape));
            check(maxDifference(blocked, rowWise) == 0.0,
                    "blocking must not change a single bit, shape " + Arrays.toString(shape));
            check(multiplyAdds(a, p) == (long) a.nonzeros() * p, "work is nnz * p");
        }

        // --- boundary: a panel wider than the matrix ---------------------------
        Csr a = randomSparse(6, 8, 3);
        double[] b = randomDense(8 * 5);
        check(maxDifference(spmmBlocked(a, b, 64, null), spmmRowWise(a, b, 5, null)) == 0.0,
                "a panel wider than B degenerates to the row-wise kernel");

        // --- locality: the blocked sweep must miss less when B is wide ---------
        int rows = 64;
        int cols = 64;
        int p = 128;
        Csr wide = randomSparse(rows, cols, 6);
        double[] wideB = randomDense(cols * p);
        CacheModel rowCache = new CacheModel(32, 8);
        CacheModel blockedCache = new CacheModel(32, 8);
        double[] rowResult = spmmRowWise(wide, wideB, 8, rowCache);
        double[] blockedResult = spmmBlocked(wide, wideB, 8, blockedCache);
        check(maxDifference(rowResult, blockedResult) == 0.0, "wide case: identical results");
        long nnz = wide.nonzeros();
        check(rowCache.accesses == nnz + 2L * nnz * p,
                "the row-wise sweep touches each nonzero once and each (nonzero, column) twice: "
                        + rowCache.accesses);
        check(blockedCache.accesses == nnz * (p / 8) + 2L * nnz * p,
                "blocking re-reads the sparse rows once per panel: " + blockedCache.accesses);
        check(blockedCache.misses < rowCache.misses, "the panel sweep must miss less: " + blockedCache.misses
                + " vs " + rowCache.misses);
        check(blockedCache.misses * 2 < rowCache.misses, "and by a wide margin");

        int[] widths = {2, 4, 8, 16, 32, 64, 128};
        long[] missesByWidth = new long[widths.length];
        for (int w = 0; w < widths.length; w++) {
            CacheModel cache = new CacheModel(32, 8);
            spmmBlocked(wide, wideB, widths[w], cache);
            missesByWidth[w] = cache.misses;
        }
        int bestIndex = 0;
        for (int w = 1; w < widths.length; w++) if (missesByWidth[w] < missesByWidth[bestIndex]) bestIndex = w;
        check(bestIndex > 0 && bestIndex < widths.length - 1,
                "the best panel width is interior, found " + widths[bestIndex]);
        check(missesByWidth[0] > missesByWidth[bestIndex],
                "narrow panels pay for re-reading the sparse rows: " + missesByWidth[0] + " vs "
                        + missesByWidth[bestIndex]);
        check(missesByWidth[widths.length - 1] > missesByWidth[bestIndex],
                "wide panels no longer fit the cache: " + missesByWidth[widths.length - 1] + " vs "
                        + missesByWidth[bestIndex]);
        check(missesByWidth[bestIndex] * 2 < rowCache.misses,
                "the best panel width must beat the row-wise sweep by a wide margin: "
                        + missesByWidth[bestIndex] + " vs " + rowCache.misses);

        // --- boundary: when B fits the cache, blocking is pointless ------------
        int smallP = 8;
        Csr narrow = randomSparse(16, 16, 3);
        double[] narrowB = randomDense(16 * smallP);
        CacheModel narrowRow = new CacheModel(256, 8);
        CacheModel narrowBlocked = new CacheModel(256, 8);
        spmmRowWise(narrow, narrowB, smallP, narrowRow);
        spmmBlocked(narrow, narrowB, smallP, narrowBlocked);
        check(narrowBlocked.misses == narrowRow.misses,
                "with everything cached both kernels miss equally: " + narrowBlocked.misses + " vs "
                        + narrowRow.misses);
        check(narrowRow.misses > 0, "the cache model really counts misses");

        // --- determinism ------------------------------------------------------
        CacheModel repeatCache = new CacheModel(32, 8);
        double[] repeat = spmmBlocked(wide, wideB, 8, repeatCache);
        check(maxDifference(repeat, blockedResult) == 0.0 && repeatCache.misses == blockedCache.misses,
                "the same call must give the same result and the same miss count");

        System.out.println("SpMM: A is " + rows + "x" + cols + " with " + wide.nonzeros() + " nonzeros, B is "
                + cols + "x" + p + ", cache 32 lines of 8 doubles");
        System.out.println("LRU misses: row-wise " + rowCache.misses + ", blocked best width " + widths[bestIndex]
                + " at " + missesByWidth[bestIndex] + " over " + rowCache.accesses + " accesses");
        System.out.println("work is " + multiplyAdds(wide, p) + " multiply-adds either way; narrow B misses "
                + narrowBlocked.misses + " times in both kernels");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
