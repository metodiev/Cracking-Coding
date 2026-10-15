/** Task 254 - Implement bloom filter insert/check (probabilistic). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
public class Solution {
    static final int M_BITS = 1024;
    static final int K_HASHES = 7;
    // Deterministic values for the exact test set below (cross-language check).
    static final int EXPECTED_BITS = 503;
    static final int EXPECTED_FP = 74;

    /** FNV-1a, 32-bit; int overflow wraps exactly like an unsigned 32-bit multiply. */
    static int fnv1a(String text) {
        int h = 0x811C9DC5;
        for (int i = 0; i < text.length(); i++) {
            h ^= text.charAt(i);
            h *= 0x01000193;
        }
        return h;
    }

    /** MurmurHash3 avalanche: raw FNV-1a low bits are poorly mixed for double hashing. */
    static int fmix32(int x) {
        x ^= x >>> 16;
        x *= 0x85EBCA6B;
        x ^= x >>> 13;
        x *= 0xC2B2AE35;
        x ^= x >>> 16;
        return x;
    }

    static int[] positions(String item, int bits, int hashes) {
        long h1 = fmix32(fnv1a(item)) & 0xFFFFFFFFL;
        long h2 = (fmix32(fnv1a(item + "#")) & 0xFFFFFFFFL) | 1L;
        int[] out = new int[hashes];
        for (int i = 0; i < hashes; i++) out[i] = (int) ((h1 + i * h2) % bits);
        return out;
    }

    static class BloomFilter {
        final int m;
        final int k;
        final boolean[] bits;

        BloomFilter(int m, int k) {
            this.m = m;
            this.k = k;
            this.bits = new boolean[m];
        }

        void add(String item) {
            for (int p : positions(item, m, k)) bits[p] = true;
        }

        boolean maybeContains(String item) {
            for (int p : positions(item, m, k)) if (!bits[p]) return false;
            return true;
        }

        int popcount() {
            int count = 0;
            for (boolean b : bits) if (b) count++;
            return count;
        }
    }

    public static void main(String[] args) {
        BloomFilter f = new BloomFilter(M_BITS, K_HASHES);
        check(f.popcount() == 0, "empty filter");
        for (String probe : new String[]{"item-0", "x", "hello"}) {
            check(!f.maybeContains(probe), "miss on empty: " + probe);
        }

        for (int i = 0; i < 100; i++) f.add("item-" + i);
        for (int i = 0; i < 100; i++) check(f.maybeContains("item-" + i), "no false negative: item-" + i);
        int filled = f.popcount();
        f.add("item-0");
        check(f.popcount() == filled, "insert is idempotent");

        int falsePositives = 0;
        for (int i = 0; i < 10000; i++) {
            if (f.maybeContains("other-" + i)) falsePositives++;
        }
        check(f.popcount() == EXPECTED_BITS, "bits set agree across languages");
        check(falsePositives == EXPECTED_FP, "false positives agree across languages");
        // Documented bound: theory gives (1 - e^(-k*n/m))^k ~ 0.7% for n=100, m=1024, k=7;
        // the measured rate must stay under 2% (200 out of 10000 queries).
        check(falsePositives <= 200, "measured false-positive rate under the documented bound");

        BloomFilter small = new BloomFilter(256, 4);
        for (String item : new String[]{"a", "b", "c"}) small.add(item);
        for (String item : new String[]{"a", "b", "c"}) check(small.maybeContains(item), "small filter: " + item);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
