/** Task 699 - Implement GPU-style parallel prefix-sum algorithm simulation. (Tier: Expert, CTCI Beyond - Networking, OS & Real-Time)
 *
 * Three scan kernels, all simulated round by round so that "depth" means barrier-separated
 * rounds and "work" means the number of additions - exactly the two costs a GPU cares about:
 *
 *   HILLIS-STEELE  the classic n log n work / log n depth inclusive scan: in round k, element i
 *                  adds element i - 2^k from the previous round (double buffering stands in for
 *                  the barrier, so no element reads a value another thread just wrote - the
 *                  reason a real kernel needs two buffers);
 *   BLELLOCH       the work-efficient 2n scan: pad to a power of two and run an up-sweep that
 *                  builds the tree sums, then a down-sweep that turns it into an exclusive scan;
 *   BLOCK SCAN     the two-level pattern used for large arrays: scan each block with Blelloch,
 *                  scan the block totals, then add each block's offset back.
 *
 * Verified: all three kernels agree with the sequential scan for every length up to 33 (so the
 * power-of-two padding really is the identity), the exclusive/inclusive relationship, range
 * sums, the exact work counts (2n-2 for Blelloch, sum over rounds for Hillis-Steele), the exact
 * round counts, negatives and zeros, and a 4096-element run with its cost metrics.
 */
import java.util.*;

/** Scan output plus the parallel cost metrics of the kernel that produced it. */
class ScanResult {
    final int[] values;
    final int work;
    final int rounds;

    ScanResult(int[] values, int work, int rounds) {
        this.values = values;
        this.work = work;
        this.rounds = rounds;
    }
}

public class Solution {
    static int[] sequential(int[] values) {
        int[] out = new int[values.length];
        int running = 0;
        for (int i = 0; i < values.length; i++) {
            running += values[i];
            out[i] = running;
        }
        return out;
    }

    /** Inclusive scan, log2(n) rounds, one addition per element per round. */
    static ScanResult hillisSteele(int[] values) {
        int n = values.length;
        int[] current = values.clone();
        int work = 0;
        int rounds = 0;
        for (int step = 1; step < n; step <<= 1) {
            int[] next = current.clone();
            for (int i = step; i < n; i++) {
                next[i] = current[i] + current[i - step];
                work++;
            }
            current = next;
            rounds++;
        }
        return new ScanResult(current, work, rounds);
    }

    /** Pads to a power of two, then Blelloch exclusive scan; returns values and cost. */
    static ScanResult blelloch(int[] values) {
        int size = 1;
        while (size < Math.max(1, values.length)) size <<= 1;
        int[] tree = new int[size];
        for (int i = 0; i < values.length; i++) tree[i] = values[i];
        int work = 0;
        int rounds = 0;
        for (int step = 1; step < size; step <<= 1) {
            for (int i = (step << 1) - 1; i < size; i += (step << 1)) {
                tree[i] += tree[i - step];
                work++;
            }
            rounds++;
        }
        tree[size - 1] = 0;
        for (int step = size >> 1; step > 0; step >>= 1) {
            for (int i = (step << 1) - 1; i < size; i += (step << 1)) {
                int left = tree[i - step];
                tree[i - step] = tree[i];
                tree[i] += left;
                work++;
            }
            rounds++;
        }
        int[] out = new int[values.length];
        System.arraycopy(tree, 0, out, 0, values.length);
        return new ScanResult(out, work, rounds);
    }

    /** Two-level scan: Blelloch per block, a scan of the block totals, then offset each block. */
    static ScanResult blockScan(int[] values, int blockSize) {
        int n = values.length;
        int blocks = (n + blockSize - 1) / blockSize;
        int[] totals = new int[blocks];
        int[] out = new int[n];
        int work = 0;
        int rounds = 0;
        for (int block = 0; block < blocks; block++) {
            int from = block * blockSize;
            int to = Math.min(n, from + blockSize);
            int[] piece = Arrays.copyOfRange(values, from, to);
            ScanResult scanned = blelloch(piece);
            work += scanned.work;
            rounds = Math.max(rounds, scanned.rounds);
            int sum = to > from ? scanned.values[to - from - 1] + piece[to - from - 1] : 0;
            // values is the exclusive scan of `piece`, so the block total is the exclusive
            // prefix at the last position plus the last element.
            totals[block] = sum;
            for (int i = 0; i < to - from; i++) out[from + i] = scanned.values[i];
        }
        ScanResult blockTotals = blelloch(totals);
        work += blockTotals.work;
        rounds = rounds + blockTotals.rounds + 1;
        for (int block = 0; block < blocks; block++) {
            int from = block * blockSize;
            int to = Math.min(n, from + blockSize);
            for (int i = from; i < to; i++) {
                out[i] += blockTotals.values[block];
                work++;
            }
        }
        return new ScanResult(out, work, rounds);
    }

    /** Exclusive reference: the inclusive prefix shifted right with the identity in front. */
    static int[] exclusiveReference(int[] values) {
        int[] inclusive = sequential(values);
        int[] out = new int[values.length];
        for (int i = 1; i < values.length; i++) out[i] = inclusive[i - 1];
        return out;
    }

    static String text(int[] values) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(values[i]);
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        // 1. Hand-checked values.
        int[] small = {3, 1, 4, 1, 5};
        check(text(sequential(small)).equals("3,4,8,9,14"), "sequential reference: "
                + text(sequential(small)));
        check(text(hillisSteele(small).values).equals("3,4,8,9,14"), "Hillis-Steele inclusive scan");
        check(text(blelloch(small).values).equals("0,3,4,8,9"), "Blelloch produces an exclusive scan");
        for (int i = 1; i < small.length; i++) {
            check(blelloch(small).values[i] == sequential(small)[i - 1],
                    "exclusive prefix is the inclusive prefix one step behind");
        }
        check(blelloch(small).values[0] == 0, "the exclusive scan starts at the identity");

        // 2. Every kernel agrees with the sequential scan for every length up to 33.
        int agreements = 0;
        for (int length = 0; length <= 33; length++) {
            int[] data = new int[length];
            for (int i = 0; i < length; i++) data[i] = ((i * 37) % 11) - 5;
            int[] reference = sequential(data);
            int[] exclusive = exclusiveReference(data);
            if (!Arrays.equals(hillisSteele(data).values, reference)) {
                check(false, "Hillis-Steele mismatch at length " + length);
            }
            if (!Arrays.equals(blelloch(data).values, exclusive)) {
                check(false, "Blelloch mismatch at length " + length);
            }
            if (!Arrays.equals(blockScan(data, 4).values, exclusive)) {
                check(false, "block scan mismatch at length " + length);
            }
            agreements++;
        }
        check(agreements == 34, "all lengths from 0 to 33 pass for all three kernels");

        // 3. Exclusive/inclusive relationship and range sums on a longer array.
        int[] data = new int[64];
        for (int i = 0; i < data.length; i++) data[i] = 1 + (i % 7) * 3 - 5;
        int[] inclusive = hillisSteele(data).values;
        int[] exclusive = blelloch(data).values;
        int total = 0;
        for (int value : data) total += value;
        check(inclusive[63] == total, "the last inclusive prefix is the total: " + inclusive[63]);
        boolean relationship = true;
        for (int i = 1; i < data.length; i++) {
            if (exclusive[i] != inclusive[i - 1]) relationship = false;
        }
        check(relationship, "exclusive and inclusive scans are consistent");
        int rangeChecks = 0;
        for (int from = 0; from + 5 <= 64; from += 7) {
            int expected = 0;
            for (int i = from; i < from + 5; i++) expected += data[i];
            check(inclusive[from + 4] - (from > 0 ? inclusive[from - 1] : 0) == expected,
                    "range sum by prefix difference at " + from);
            rangeChecks++;
        }
        check(rangeChecks >= 8, "range sums were exercised at many offsets: " + rangeChecks);

        // 4. Cost metrics: 2n-2 work for Blelloch, the closed form for Hillis-Steele.
        ScanResult blelloch16 = blelloch(new int[16]);
        check(blelloch16.work == 2 * 16 - 2, "Blelloch does exactly 2n-2 additions: " + blelloch16.work);
        check(blelloch16.rounds == 8, "Blelloch has 2 log2(n) rounds: " + blelloch16.rounds);
        ScanResult hillis16 = hillisSteele(new int[16]);
        int expectedWork = 0;
        for (int step = 1; step < 16; step <<= 1) expectedWork += 16 - step;
        check(hillis16.work == expectedWork, "Hillis-Steele work matches the closed form: "
                + hillis16.work);
        check(hillis16.rounds == 4, "Hillis-Steele has log2(n) rounds: " + hillis16.rounds);
        check(hillis16.work > blelloch16.work,
                "Hillis-Steele trades work for depth: " + hillis16.work + " vs " + blelloch16.work);
        ScanResult blellochNonPower = blelloch(new int[10]);
        check(blellochNonPower.work == 2 * 16 - 2,
                "padding a 10-element array scans 16 leaves: " + blellochNonPower.work);

        // 5. The two-level kernel: correct for several block sizes, with a moderate cost.
        int[] longData = new int[1000];
        for (int i = 0; i < longData.length; i++) longData[i] = (i % 13) - 6;
        int[] longReference = exclusiveReference(longData);
        for (int blockSize : new int[]{1, 2, 4, 16, 32, 64}) {
            ScanResult result = blockScan(longData, blockSize);
            check(Arrays.equals(result.values, longReference),
                    "block scan with block size " + blockSize + " matches");
            check(result.work > 0 && result.rounds > 0, "block-cost metrics were recorded");
        }
        ScanResult block16 = blockScan(longData, 16);
        check(block16.work < hillisSteele(longData).work,
                "the two-level scan does less work than a flat Hillis-Steele: " + block16.work
                        + " vs " + hillisSteele(longData).work);

        // 6. Boundary cases.
        check(hillisSteele(new int[0]).values.length == 0, "empty input gives an empty scan");
        check(blelloch(new int[0]).values.length == 0, "Blelloch handles the empty input");
        check(blockScan(new int[0], 8).values.length == 0, "block scan handles the empty input");
        check(hillisSteele(new int[]{7}).values[0] == 7 && blelloch(new int[]{7}).values[0] == 0,
                "single element: inclusive is the element, exclusive is the identity");
        check(hillisSteele(new int[]{7}).rounds == 0 && blelloch(new int[]{7}).rounds == 0,
                "a one-element scan needs no parallel round at all");
        int[] zeros = new int[32];
        check(text(hillisSteele(zeros).values).equals(text(sequential(zeros))),
                "all zeros scan to all zeros");
        int[] negatives = {-5, -10, 3, -1, 0, 7};
        check(Arrays.equals(hillisSteele(negatives).values, sequential(negatives)),
                "negative values scan correctly");

        // 7. A 4096-element run with its cost metrics.
        int[] large = new int[4096];
        for (int i = 0; i < large.length; i++) large[i] = (i * 2654435761L % 97 < 48) ? 1 : -2;
        int[] largeReference = exclusiveReference(large);
        ScanResult largeBlelloch = blelloch(large);
        check(Arrays.equals(largeBlelloch.values, largeReference), "4096 elements scan correctly");
        check(largeBlelloch.work == 2 * 4096 - 2, "Blelloch work is 2n-2 at scale: "
                + largeBlelloch.work);
        check(largeBlelloch.rounds == 24, "12 levels means 24 rounds: " + largeBlelloch.rounds);
        ScanResult largeHillis = hillisSteele(large);
        check(largeHillis.rounds == 12, "Hillis-Steele needs 12 rounds: " + largeHillis.rounds);
        check(largeHillis.work > 4 * largeBlelloch.work,
                "the work gap at scale (about 5.5x): " + largeHillis.work + " vs " + largeBlelloch.work);
        ScanResult largeBlock = blockScan(large, 64);
        check(Arrays.equals(largeBlock.values, largeReference), "blocked kernel at scale");
        check(largeBlock.rounds <= 2 * largeHillis.rounds + 2,
                "the two-level kernel is depth-heavier than a flat scan, as expected: "
                        + largeBlock.rounds + " vs " + largeHillis.rounds);

        // 8. Determinism.
        int[] runA = hillisSteele(large).values;
        int[] runB = hillisSteele(large).values;
        check(Arrays.equals(runA, runB) && hillisSteele(large).work == largeHillis.work,
                "kernels are deterministic, cost metrics included");
        check(blelloch(large).work == largeBlelloch.work, "Blelloch cost is reproducible");

        System.out.println("PASS");
    }

    static final int edges = 1;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
