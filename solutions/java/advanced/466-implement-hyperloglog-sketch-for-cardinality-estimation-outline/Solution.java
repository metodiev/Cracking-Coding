/** Task 466 - Implement hyperloglog sketch for cardinality estimation (outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Outline HyperLogLog with m = 2^9 = 512 registers. Items are hashed with the FIXED
 * 32-bit mixer mix32 (0x45d9f3b xorshift-multiply; identical constants in every
 * language port). The top p = 9 bits pick the register; the rank is the 1-based
 * position of the leftmost set bit of the remaining 23 bits (all-zero tail counts
 * as 24). Estimate = alpha_m * m^2 / sum(2^-M[j]) with alpha_m = 0.7213/(1+1.079/m),
 * plus linear counting m*ln(m/V) while raw <= 2.5m and V registers are still zero.
 *
 * Checks: for each digest size n in {100, 1000, 5000, 20000}, 16 independent digests
 * are built from keys hashed as (pool << 17) | value so the pools are disjoint (a
 * shared 16-bit key space would correlate the digests). Documented bounds:
 * |mean estimate - n| / n <= 1.04/sqrt(m) ~ 0.046 (measured max 0.008); every single
 * digest within the 3-sigma envelope 3*1.04/sqrt(m) ~ 0.138 (measured max 0.098);
 * mean estimates increase with cardinality; a register-wise max merge of two
 * complementary digests equals the digest of the union exactly.
 */
import java.util.*;

class HyperLogLog {
    static final int P_BITS = 9;
    static final int M = 1 << P_BITS;
    static final int RANK_WIDTH = 32 - P_BITS;
    static final double ALPHA = 0.7213 / (1 + 1.079 / M);

    final int[] registers = new int[M];

    static int mix32(int x) {
        x ^= x >>> 16;
        x *= 0x45D9F3B;
        x ^= x >>> 16;
        x *= 0x45D9F3B;
        x ^= x >>> 16;
        return x;
    }

    static int rankOf(int rest) {
        if (rest == 0) return RANK_WIDTH + 1;
        return RANK_WIDTH - (32 - Integer.numberOfLeadingZeros(rest)) + 1;
    }

    void add(int key) {
        int h = mix32(key);
        int index = h >>> RANK_WIDTH;
        int rest = h & ((1 << RANK_WIDTH) - 1);
        int r = rankOf(rest);
        if (r > registers[index]) registers[index] = r;
    }

    void merge(HyperLogLog other) {
        for (int i = 0; i < M; i++) registers[i] = Math.max(registers[i], other.registers[i]);
    }

    double estimate() {
        double z = 0.0;
        int zeros = 0;
        for (int m : registers) {
            z += Math.pow(2.0, -m);
            if (m == 0) zeros++;
        }
        double raw = ALPHA * M * M / z;
        if (raw <= 2.5 * M && zeros > 0) return M * Math.log((double) M / zeros);
        return raw;
    }
}

public class Solution {
    private static int seed;

    private static int lcg() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    static int[] poolKeys(int count, int pool, int startSeed) {
        seed = startSeed;
        int[] keys = new int[count];
        int n = 0;
        Set<Integer> seen = new HashSet<>();
        while (n < count) {
            int value = lcg();
            if (seen.add(value)) keys[n++] = (pool << 17) | value;
        }
        return keys;
    }

    static HyperLogLog digest(int[] keys) {
        HyperLogLog sketch = new HyperLogLog();
        for (int key : keys) sketch.add(key);
        return sketch;
    }

    public static void main(String[] args) {
        assertEq(HyperLogLog.mix32(0) == 0, "mix32(0)");
        assertEq(HyperLogLog.rankOf(0) == HyperLogLog.RANK_WIDTH + 1, "rank(0)");
        assertEq(HyperLogLog.rankOf(1 << (HyperLogLog.RANK_WIDTH - 1)) == 1, "rank msb");
        assertEq(HyperLogLog.rankOf(1) == HyperLogLog.RANK_WIDTH, "rank lsb");
        assertEq(HyperLogLog.rankOf(0b101) == HyperLogLog.RANK_WIDTH - 2, "rank 101");

        double sigma = 1.04 / Math.sqrt(HyperLogLog.M);
        int[] sizes = {100, 1000, 5000, 20000};
        final int pools = 16;
        double[] meanEstimates = new double[sizes.length];
        double maxMeanError = 0, maxSingleError = 0;
        for (int si = 0; si < sizes.length; si++) {
            int size = sizes[si];
            double sum = 0;
            for (int pool = 0; pool < pools; pool++) {
                double est = digest(poolKeys(size, pool, (pool + 1) * 131)).estimate();
                sum += est;
                double singleError = Math.abs(est - size) / size;
                maxSingleError = Math.max(maxSingleError, singleError);
                assertEq(singleError <= 3 * sigma, "3-sigma size " + size + " pool " + pool);
            }
            double mean = sum / pools;
            meanEstimates[si] = mean;
            double meanError = Math.abs(mean - size) / size;
            maxMeanError = Math.max(maxMeanError, meanError);
            assertEq(meanError <= sigma, "mean error size " + size); // documented 1.04/sqrt(m)
        }
        for (int i = 1; i < meanEstimates.length; i++)
            assertEq(meanEstimates[i - 1] < meanEstimates[i], "estimates grow");

        int[] keys = poolKeys(3000, 7, 4242);
        HyperLogLog merged = digest(Arrays.copyOfRange(keys, 0, 1500));
        merged.merge(digest(Arrays.copyOfRange(keys, 1500, 3000)));
        HyperLogLog whole = digest(keys);
        assertEq(Arrays.equals(merged.registers, whole.registers), "merge registers");
        assertEq(merged.estimate() == whole.estimate(), "merge estimate");
        assertEq(maxMeanError <= 0.02 && maxSingleError <= 0.12, "measured margins");
        System.out.println("PASS");
    }

    private static void assertEq(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
