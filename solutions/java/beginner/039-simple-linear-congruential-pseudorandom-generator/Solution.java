/** Task 039 - Simple linear congruential pseudorandom generator. (Tier: Beginner, CTCI Ch 6) */
public class Solution {
    static long[] lcg(long seed, int n) {
        long a = 1103515245L, c = 12345L, m = 1L << 31;
        long x = seed;
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            x = (a * x + c) % m;
            out[i] = x;
        }
        return out;
    }

    public static void main(String[] args) {
        long[] a = lcg(42, 5);
        check(a.length == 5, "length");
        check(java.util.Arrays.equals(a, lcg(42, 5)), "deterministic");
        check(!java.util.Arrays.equals(a, lcg(43, 5)), "seed-sensitive");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
