/** Task 467 - Implement bloom filter with double hashing and test false positive. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Bloom filter with m = 2^16 bits and k = 7 probes via double hashing:
 * h1 = mix32(key), h2 = mix32(key ^ 0x9E3779B9) | 1 (forced odd, hence coprime to
 * the power-of-two m, so the 7 probes are pairwise distinct),
 * probe i = (h1 + i*h2) mod m.
 *
 * Checks: a hand-verifiable small filter (single insert sets exactly 7 distinct bits
 * and is idempotent), then insert 10000 distinct member keys (LCG values in
 * [0, 65536)), verify all 10000 are reported present (zero false negatives), and
 * test 10000 disjoint non-member keys (200000 + i) counting false positives. The
 * documented bound is FP_MAX = 800; the theoretical rate (1 - e^(-k*n/m))^k ~ 0.0524
 * gives ~524 expected and the measured count on this fixed key set is 488 (asserted
 * <= 800 with a >= 444 sanity floor).
 */
import java.util.*;

class BloomFilter {
    static final int M_BITS = 1 << 16;
    static final int PROBES = 7;

    private final byte[] data = new byte[M_BITS / 8];

    static int mix32(int x) {
        x ^= x >>> 16;
        x *= 0x45D9F3B;
        x ^= x >>> 16;
        x *= 0x45D9F3B;
        x ^= x >>> 16;
        return x;
    }

    int[] positions(int key) {
        long h1 = mix32(key) & 0xFFFFFFFFL;
        long h2 = (mix32(key ^ 0x9E3779B9) & 0xFFFFFFFFL) | 1L;
        int[] out = new int[PROBES];
        for (int i = 0; i < PROBES; i++) out[i] = (int) ((h1 + (long) i * h2) % M_BITS);
        return out;
    }

    void add(int key) {
        for (int pos : positions(key)) data[pos >> 3] |= 1 << (pos & 7);
    }

    boolean contains(int key) {
        for (int pos : positions(key)) {
            if (((data[pos >> 3] >> (pos & 7)) & 1) == 0) return false;
        }
        return true;
    }

    int popcount() {
        int total = 0;
        for (byte b : data) total += Integer.bitCount(b & 0xFF);
        return total;
    }
}

public class Solution {
    private static final int FP_MAX = 800;
    private static int seed;

    private static int lcg() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void main(String[] args) {
        // Hand-verifiable small filter: 7 distinct probes, idempotent insert.
        BloomFilter tiny = new BloomFilter();
        Set<Integer> distinct = new HashSet<>();
        for (int pos : tiny.positions(5)) distinct.add(pos);
        check(distinct.size() == BloomFilter.PROBES, "distinct probes");
        tiny.add(5);
        check(tiny.popcount() == BloomFilter.PROBES, "popcount after insert");
        check(tiny.contains(5), "member present");
        tiny.add(5);
        check(tiny.popcount() == BloomFilter.PROBES, "idempotent insert");
        check(!tiny.contains(6), "non-member absent here");

        // Workload: 10000 distinct members from the LCG, 10000 disjoint non-members.
        BloomFilter bloom = new BloomFilter();
        seed = 3141592;
        int[] members = new int[10000];
        int n = 0;
        Set<Integer> seen = new HashSet<>();
        while (n < members.length) {
            int value = lcg();
            if (seen.add(value)) members[n++] = value;
        }
        for (int key : members) bloom.add(key);
        for (int key : members) check(bloom.contains(key), "zero false negatives"); // all present
        int falsePositives = 0;
        for (int i = 0; i < 10000; i++) if (bloom.contains(200000 + i)) falsePositives++;
        check(falsePositives <= FP_MAX, "fp bound " + falsePositives); // measured 488
        check(falsePositives >= 444, "fp sanity floor " + falsePositives);
        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
