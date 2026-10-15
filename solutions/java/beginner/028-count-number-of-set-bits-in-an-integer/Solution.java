/** Task 028 - Count number of set bits in an integer. (Tier: Beginner, CTCI Ch 5) */
public class Solution {
    static int countSetBits(int x) {
        int count = 0;
        while (x != 0) { x &= x - 1; count++; }
        return count;
    }

    public static void main(String[] args) {
        check(countSetBits(0) == 0, "0");
        check(countSetBits(255) == 8, "255");
        check(countSetBits(0b101101) == 4, "0b101101");
        check(countSetBits(-1) == 32, "all bits (-1)");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
