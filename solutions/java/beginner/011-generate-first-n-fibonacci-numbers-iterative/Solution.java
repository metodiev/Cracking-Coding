/** Task 011 - Generate first N Fibonacci numbers (iterative). (Tier: Beginner, CTCI Ch 8) */
import java.util.Arrays;

public class Solution {
    static int[] fibSequence(int n) {
        int[] seq = new int[n];
        int a = 0, b = 1;
        for (int i = 0; i < n; i++) {
            seq[i] = a;
            int next = a + b;
            a = b;
            b = next;
        }
        return seq;
    }

    public static void main(String[] args) {
        check(fibSequence(0).length == 0, "n=0");
        check(Arrays.equals(fibSequence(8), new int[]{0, 1, 1, 2, 3, 5, 8, 13}), "n=8");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
