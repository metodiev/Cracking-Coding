/** Task 155 - Josephus problem simulation. (Tier: Beginner, CTCI Ch 2) */
import java.util.*;

public class Solution {
    /** Eliminate every k-th person in a circle of n; returns the 0-based survivor. O(n^2). */
    static int josephusSimulate(int n, int k) {
        if (n < 1 || k < 1) throw new IllegalArgumentException("n and k must be >= 1");
        List<Integer> people = new ArrayList<>();
        for (int i = 0; i < n; i++) people.add(i);
        int idx = 0;
        while (people.size() > 1) {
            idx = (idx + k - 1) % people.size();   // the k-th person counted from idx
            people.remove(idx);
        }
        return people.get(0);
    }

    /** Bottom-up J(n,k) = (J(n-1,k) + k) % n starting at J(1,k) = 0. O(n) time, O(1) space. */
    static int josephusRecurrence(int n, int k) {
        if (n < 1 || k < 1) throw new IllegalArgumentException("n and k must be >= 1");
        int survivor = 0;
        for (int i = 2; i <= n; i++) survivor = (survivor + k) % i;
        return survivor;
    }

    public static void main(String[] args) {
        // known checkpoints
        check(josephusSimulate(5, 2) == 2, "J(5,2)");
        check(josephusSimulate(7, 3) == 3, "J(7,3)");
        check(josephusSimulate(41, 3) == 30, "J(41,3)");

        // both variants agree, including the n = 1 base case
        for (int n : new int[] {1, 2, 3, 5, 7, 10, 20, 41}) {
            for (int k : new int[] {1, 2, 3, 4, 7, 11}) {
                check(josephusRecurrence(n, k) == josephusSimulate(n, k), "agree n=" + n + " k=" + k);
            }
        }
        check(josephusSimulate(1, 7) == 0 && josephusRecurrence(1, 999) == 0, "n=1 survivor 0");

        // k = 1 eliminates in order 0, 1, 2, ... so the last index survives
        check(josephusSimulate(10, 1) == 9 && josephusRecurrence(10, 1) == 9, "k=1");

        check(josephusRecurrence(5, 2) == 2, "recurrence J(5,2)");
        check(josephusRecurrence(7, 3) == 3, "recurrence J(7,3)");
        check(josephusRecurrence(41, 3) == 30, "recurrence J(41,3)");

        boolean threw = false;
        try { josephusSimulate(0, 2); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "n=0 throws");
        threw = false;
        try { josephusRecurrence(3, -1); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "k=-1 throws");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
