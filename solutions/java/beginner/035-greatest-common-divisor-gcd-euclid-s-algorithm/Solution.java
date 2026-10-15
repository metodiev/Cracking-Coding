/** Task 035 - Greatest common divisor (GCD) Euclid's algorithm. (Tier: Beginner, CTCI Ch 6) */
public class Solution {
    static int gcd(int a, int b) {
        a = Math.abs(a); b = Math.abs(b);
        while (b != 0) { int t = b; b = a % b; a = t; }
        return a;
    }

    public static void main(String[] args) {
        check(gcd(48, 18) == 6, "48/18");
        check(gcd(17, 5) == 1, "17/5");
        check(gcd(0, 9) == 9, "0/9");
        check(gcd(-24, 36) == 12, "negatives");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
