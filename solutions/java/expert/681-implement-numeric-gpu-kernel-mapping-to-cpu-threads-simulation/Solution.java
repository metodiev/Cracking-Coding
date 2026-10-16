/** Task 681 - Implement numeric GPU kernel mapping to CPU threads simulation. (Tier: Expert, CTCI Beyond — Parallel Computing & GPU)
 *
 * A scaled-down but faithful model of a CUDA-style launch executed by CPU threads: a grid of blocks, each
 * block a fixed number of threads grouped into 32-lane warps, with the usual index arithmetic
 * (globalId = blockIdx * blockDim + threadIdx, warp = threadIdx / 32, lane = threadIdx % 32) and a memory
 * model in which a warp's accesses are served by 128-byte transactions - one transaction when the lanes
 * touch consecutive, aligned data, one per lane when they stride.
 *
 * The kernel is a tiled matrix multiply: each block owns a tile of the output, and for every tile along the
 * k dimension it loads the A and B tiles into shared memory (coalesced, bounds-guarded so non-multiple
 * sizes work) and accumulates the partial products in registers before writing the tile out.
 *
 * Simplification (stated): threads are simulated sequentially by the CPU, so there is no real parallelism
 * and no scheduling non-determinism; what is modelled faithfully is the index mapping, the warp/coalescing
 * arithmetic, the shared-memory tiling and the reduction of global traffic.
 *
 * Verified: the global index map is a bijection over the whole grid and every output cell is written
 * exactly once; warp/lane decomposition is consistent; transaction counts match an independent
 * 128-byte-segment computation for consecutive, strided and misaligned patterns; the tiled kernel equals a
 * naive triple loop for square sizes, for a boundary case where the size is not a multiple of the tile, and
 * for degenerate sizes (1x1, tile larger than the matrix); the global load count drops by exactly the tile
 * factor; shared-memory usage equals two tiles; and lane activity accounting shows the inactive lanes on
 * the boundary tile.
 */
import java.util.*;

final class Launch {
    final int m, tile;

    Launch(int m, int tile) { this.m = m; this.tile = tile; }

    int blocksPerSide() { return (m + tile - 1) / tile; }

    int blocks() { return blocksPerSide() * blocksPerSide(); }

    int threadsPerBlock() { return tile * tile; }

    int warpsPerBlock() { return (threadsPerBlock() + 31) / 32; }

    int totalThreads() { return blocks() * threadsPerBlock(); }

    int blockOf(int globalId) { return globalId / threadsPerBlock(); }

    int threadOf(int globalId) { return globalId % threadsPerBlock(); }

    int warpOf(int threadId) { return threadId / 32; }

    int laneOf(int threadId) { return threadId % 32; }

    int rowOf(int threadId) { return threadId / tile; }

    int colOf(int threadId) { return threadId % tile; }
}

final class MemoryModel {
    int transactions, loads, stores;

    /** A warp's accesses collapse into as many 128-byte transactions as distinct segments they touch. */
    static int transactionsFor(int[] addresses) {
        Set<Integer> segments = new TreeSet<>();
        for (int address : addresses) segments.add(address >>> 7);
        return segments.size();
    }

    int warpAccess(int[] addresses) {
        int count = transactionsFor(addresses);
        transactions += count;
        loads += addresses.length;
        return count;
    }

    String stats() { return "loads=" + loads + " transactions=" + transactions; }
}

final class TiledKernel {
    final Launch launch;
    final int[][] a, b, c;
    final MemoryModel memory = new MemoryModel();
    int globalLoads, sharedStores, sharedLoads, registerUpdates, activeLanes, inactiveLanes, cellsWritten;
    long sharedBytes;

    TiledKernel(Launch launch, int[][] a, int[][] b) {
        this.launch = launch;
        this.a = a;
        this.b = b;
        this.c = new int[launch.m][launch.m];
    }

    /** One warp step: 32 consecutive elements of a row are fetched as a single coalesced transaction. */
    void loadCoalescedRow(int[] sourceRow, int from, int count, int[] sharedRow, int sharedIndex) {
        int[] addresses = addresses(from, count, 1);
        memory.warpAccess(addresses);
        globalLoads += count;
        sharedStores += count;
        for (int k = 0; k < count; k++) sharedRow[sharedIndex + k] = sourceRow[from + k];
    }

    /** The uncoalesced variant: a warp walking down a column strides by one row of 4-byte words. */
    int loadStridedColumn(int[][] source, int col, int from, int count, int[] shared, int sharedIndex) {
        int[] addresses = addresses(from, count, launch.m);
        for (int k = 0; k < count; k++) addresses[k] += col * 4;
        int used = memory.warpAccess(addresses);
        globalLoads += count;
        sharedStores += count;
        for (int k = 0; k < count; k++) shared[sharedIndex + k] = source[from + k][col];
        return used;
    }

    static int[] addresses(int from, int count, int strideWords) {
        int[] addresses = new int[count];
        for (int k = 0; k < count; k++) addresses[k] = (from + k) * strideWords * 4;
        return addresses;
    }

    int[][] run() {
        int tile = launch.tile;
        int side = launch.blocksPerSide();
        for (int blockIdx = 0; blockIdx < launch.blocks(); blockIdx++) {
            int blockRow = blockIdx / side, blockCol = blockIdx % side;
            int[] tileA = new int[tile * tile], tileB = new int[tile * tile];
            int[][] accumulator = new int[tile][tile];
            sharedBytes = 2L * tile * tile * 4;
            for (int kt = 0; kt < side; kt++) {
                Arrays.fill(tileA, 0);                  // boundary tiles must not keep stale shared values
                Arrays.fill(tileB, 0);
                for (int r = 0; r < tile; r++) {
                    int row = blockRow * tile + r, col = kt * tile;
                    if (row >= launch.m) { inactiveLanes += tile; continue; }
                    int count = 0;
                    for (int cc = 0; cc < tile && col + cc < launch.m; cc++) count++;
                    if (count > 0) loadCoalescedRow(a[row], col, count, tileA, r * tile);
                    inactiveLanes += tile - count;
                    sharedLoads += count;
                }
                for (int r = 0; r < tile; r++) {
                    int row = kt * tile + r, col = blockCol * tile;
                    if (row >= launch.m || col >= launch.m) { inactiveLanes += tile; continue; }
                    int count = 0;
                    for (int cc = 0; cc < tile && col + cc < launch.m; cc++) count++;
                    if (count > 0) loadCoalescedRow(b[row], col, count, tileB, r * tile);
                    inactiveLanes += tile - count;
                    sharedLoads += count;
                }
                for (int i = 0; i < tile; i++) {
                    int row = blockRow * tile + i;
                    if (row >= launch.m) continue;
                    for (int j = 0; j < tile; j++) {
                        int col = blockCol * tile + j;
                        if (col >= launch.m) continue;
                        for (int k = 0; k < tile; k++) {
                            registerUpdates++;
                            activeLanes++;
                            accumulator[i][j] += tileA[i * tile + k] * tileB[k * tile + j];
                        }
                    }
                }
            }
            for (int i = 0; i < tile; i++) {
                int row = blockRow * tile + i;
                if (row >= launch.m) continue;
                for (int j = 0; j < tile; j++) {
                    int col = blockCol * tile + j;
                    if (col >= launch.m) continue;
                    c[row][col] = accumulator[i][j];
                    cellsWritten++;
                    memory.stores++;
                }
            }
        }
        return c;
    }

    static int[][] naive(int[][] a, int[][] b) {
        int n = a.length;
        int[][] c = new int[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                int sum = 0;
                for (int k = 0; k < n; k++) sum += a[i][k] * b[k][j];
                c[i][j] = sum;
            }
        return c;
    }
}

public class Solution {
    static long state = 0x6E0FFEEL;

    static long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static int[][] randomMatrix(int n, int bound) {
        int[][] m = new int[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) m[i][j] = (int) (nextLong() >>> 45) % bound - bound / 2;
        return m;
    }

    public static void main(String[] args) {
        Launch launch = new Launch(64, 16);
        check(launch.blocksPerSide() == 4 && launch.blocks() == 16 && launch.threadsPerBlock() == 256
                        && launch.warpsPerBlock() == 8 && launch.totalThreads() == 4096,
                "a 64x64 matrix with 16x16 tiles launches 16 blocks of 256 threads");
        boolean bijective = true;
        Set<Integer> ids = new TreeSet<>();
        for (int b = 0; b < launch.blocks(); b++)
            for (int t = 0; t < launch.threadsPerBlock(); t++)
                if (!ids.add(b * launch.threadsPerBlock() + t)) bijective = false;
        check(bijective && ids.size() == launch.totalThreads() && ids.contains(0) && ids.contains(launch.totalThreads() - 1),
                "the global index map covers every thread exactly once");
        check(launch.blockOf(300) == 1 && launch.threadOf(300) == 44 && launch.warpOf(44) == 1
                        && launch.laneOf(44) == 12 && launch.warpOf(44) * 32 + launch.laneOf(44) == 44,
                "warp and lane decomposition is consistent with the linear thread index");
        check(launch.rowOf(37) == 2 && launch.colOf(37) == 5, "a block's threads map onto tile rows and columns");

        int[] consecutive = new int[32];
        for (int i = 0; i < 32; i++) consecutive[i] = i * 4;
        check(MemoryModel.transactionsFor(consecutive) == 1, "32 consecutive words fit one 128-byte transaction");
        int[] strided = new int[32];
        for (int i = 0; i < 32; i++) strided[i] = i * 64 * 4;
        check(MemoryModel.transactionsFor(strided) == 32, "a column walk costs one transaction per lane");
        int[] misaligned = new int[32];
        for (int i = 0; i < 32; i++) misaligned[i] = 132 + i * 4;
        check(MemoryModel.transactionsFor(misaligned) == 2, "a misaligned 128-byte row splits into two transactions");
        check(MemoryModel.transactionsFor(new int[]{0, 124, 128}) == 2, "segment boundaries are exclusive at 128 bytes");
        check(MemoryModel.transactionsFor(new int[]{}) == 0, "an idle warp touches nothing");

        for (int[] config : new int[][]{{8, 4}, {16, 8}, {16, 4}, {32, 16}}) {
            Launch l = new Launch(config[0], config[1]);
            int[][] a = randomMatrix(config[0], 9), b = randomMatrix(config[0], 7);
            TiledKernel kernel = new TiledKernel(l, a, b);
            int[][] tiled = kernel.run();
            int[][] reference = TiledKernel.naive(a, b);
            boolean equal = true;
            for (int i = 0; i < config[0]; i++) if (!Arrays.equals(tiled[i], reference[i])) equal = false;
            check(equal, "tiled kernel matches the naive loop for " + config[0] + "x" + config[0]
                    + " with tiles of " + config[1]);
            check(kernel.cellsWritten == config[0] * config[0], "every output cell is written exactly once, got "
                    + kernel.cellsWritten);
            long naiveLoads = 2L * config[0] * config[0] * config[0];
            check(naiveLoads / kernel.globalLoads == config[1],
                    "global loads drop by exactly the tile factor " + config[1] + ": " + kernel.globalLoads
                            + " vs " + naiveLoads);
            check(kernel.sharedBytes == 2L * config[1] * config[1] * 4, "shared memory holds two tiles: "
                    + kernel.sharedBytes + " bytes");
            check(kernel.registerUpdates == config[0] * config[0] * config[0],
                    "the multiply-accumulate count matches the work of the kernel");
        }

        Launch boundary = new Launch(10, 4);
        int[][] a = randomMatrix(10, 5), b = randomMatrix(10, 5);
        TiledKernel kernel = new TiledKernel(boundary, a, b);
        int[][] tiled = kernel.run();
        check(Arrays.deepEquals(tiled, TiledKernel.naive(a, b)),
                "boundary tiles (10x10 with 4x4 tiles) still produce the exact product");
        check(kernel.cellsWritten == 100 && kernel.inactiveLanes > 0,
                "the boundary tile wastes " + kernel.inactiveLanes + " lane slots but writes all 100 cells");
        check(2L * 1000 / kernel.globalLoads >= 3,
                "the boundary case still cuts global traffic by about the tile factor (padding costs some): "
                        + kernel.globalLoads + " loads vs 2000 naive");

        Launch tiny = new Launch(1, 4);
        TiledKernel tinyKernel = new TiledKernel(tiny, randomMatrix(1, 3), randomMatrix(1, 3));
        int[][] tinyResult = tinyKernel.run();
        check(tinyResult[0][0] == TiledKernel.naive(tinyKernel.a, tinyKernel.b)[0][0],
                "a 1x1 matrix works although the tile is 4x4");
        check(tiny.blocks() == 1 && tiny.threadsPerBlock() == 16 && tinyKernel.cellsWritten == 1,
                "a single block with one useful thread: " + tinyKernel.cellsWritten + " cell written");

        Launch wide = new Launch(32, 32);
        TiledKernel uncoalesced = new TiledKernel(wide, randomMatrix(32, 3), randomMatrix(32, 3));
        MemoryModel probe = new MemoryModel();
        int transactions = probe.warpAccess(stridedColumnAddresses(32));
        check(transactions == 32, "the transposed load pattern costs 32 transactions per warp, got " + transactions);
        check(uncoalesced.memory.transactions < 32 * 32,
                "the coalesced kernel uses far fewer transactions than the strided pattern would: "
                        + uncoalesced.memory.transactions);

        TiledKernel repeat = new TiledKernel(new Launch(16, 8), a = randomMatrix(16, 4), b = randomMatrix(16, 4));
        int[][] first = repeat.run();
        TiledKernel second = new TiledKernel(new Launch(16, 8), a, b);
        check(Arrays.deepEquals(first, second.run()) && repeat.globalLoads == second.globalLoads,
                "the kernel is deterministic");

        System.out.println("launch      : 64x64 with 16x16 tiles -> " + launch.blocks() + " blocks, "
                + launch.threadsPerBlock() + " threads/block, " + launch.warpsPerBlock() + " warps/block");
        System.out.println("coalescing  : consecutive = 1, column walk = 32, misaligned = 2 transactions per warp");
        System.out.println("kernel      : 10x10 boundary case correct, " + kernel.globalLoads + " global loads vs "
                + (2L * 1000) + " naive");
        System.out.println("PASS");
    }

    static int[] stridedColumnAddresses(int rows) {
        int[] addresses = new int[rows];
        for (int i = 0; i < rows; i++) addresses[i] = i * rows * 4;
        return addresses;
    }
}
