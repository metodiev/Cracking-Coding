/** Task 014 - Swap two variables without a temporary variable. (Tier: Beginner, CTCI Ch 5) */
import java.util.Arrays;

public class Solution {
    static int[] swapArithmetic(int a, int b) {
        a = a + b;
        b = a - b;
        a = a - b;
        return new int[]{a, b};
    }

    static int[] swapXor(int a, int b) {
        a ^= b;
        b ^= a;
        a ^= b;
        return new int[]{a, b};
    }

    public static void main(String[] args) {
        check(Arrays.equals(swapArithmetic(5, 3), new int[]{3, 5}), "arithmetic");
        check(Arrays.equals(swapXor(5, 3), new int[]{3, 5}), "xor");
        check(Arrays.equals(swapXor(0, 42), new int[]{42, 0}), "xor zero");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
