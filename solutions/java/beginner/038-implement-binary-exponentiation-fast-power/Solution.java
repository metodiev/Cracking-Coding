/** Task 038 - Implement binary exponentiation (fast power). (Tier: Beginner, CTCI Ch 6) */
public class Solution {
    static long powerFast(long base, int exp) {
        long result = 1;
        while (exp > 0) {
            if ((exp & 1) == 1) result *= base;
            base *= base;
            exp >>= 1;
        }
        return result;
    }

    public static void main(String[] args) {
        check(powerFast(3, 13) == 1594323L, "3^13");
        check(powerFast(2, 10) == 1024L, "2^10");
        check(powerFast(7, 0) == 1L, "7^0");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
