/** Task 029 - Check if integer is power of two. (Tier: Beginner, CTCI Ch 5) */
public class Solution {
    static boolean isPowerOfTwo(int x) {
        return x > 0 && (x & (x - 1)) == 0;
    }

    public static void main(String[] args) {
        check(isPowerOfTwo(1) && isPowerOfTwo(1024), "powers");
        check(!isPowerOfTwo(0) && !isPowerOfTwo(12) && !isPowerOfTwo(-4), "non-powers");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
