/** Task 358 - Dining philosophers solution using ordered locks. (Tier: Intermediate, CTCI Ch 15) */
import java.util.*;

public class Solution {
    static final int PHILOSOPHERS = 5;
    static final int MEALS = 40;

    /**
     * Dining philosophers solved by resource ordering: philosopher i uses forks i and
     * (i+1)%5 but always locks the lower-numbered fork first (philosopher 4 therefore takes
     * fork 0 before fork 4). Because every philosopher acquires its two forks in the same
     * global order, no circular wait can form, and each fork monitor is held by at most one
     * philosopher at any moment. Ownership accounting is recorded while the fork is still
     * held, so the per-fork single-holder invariant can be checked without races.
     */
    static class ForkTable {
        private final Object[] forks = new Object[PHILOSOPHERS];
        private final Object accounting = new Object();
        private final int[] owners = new int[PHILOSOPHERS];      // -1 when free
        private final int[] holders = new int[PHILOSOPHERS];
        private final int[] maxHolders = new int[PHILOSOPHERS];
        private final int[] acquisitions = new int[PHILOSOPHERS];
        private final int[] meals = new int[PHILOSOPHERS];
        private int violations = 0;

        ForkTable() {
            Arrays.fill(owners, -1);
            for (int i = 0; i < PHILOSOPHERS; i++) forks[i] = new Object();
        }

        /** Locks both forks in increasing index order and eats exactly one meal. */
        void eat(int philosopher, int first, int second) {
            synchronized (forks[first]) {
                recordAcquire(philosopher, first);
                synchronized (forks[second]) {
                    recordAcquire(philosopher, second);
                    synchronized (accounting) { meals[philosopher]++; }
                    recordRelease(philosopher, second);
                }
                recordRelease(philosopher, first);
            }
        }

        private void recordAcquire(int philosopher, int fork) {
            synchronized (accounting) {
                if (owners[fork] != -1) violations++;             // would mean the lock failed
                owners[fork] = philosopher;
                holders[fork]++;
                maxHolders[fork] = Math.max(maxHolders[fork], holders[fork]);
                acquisitions[fork]++;
            }
        }

        private void recordRelease(int philosopher, int fork) {
            synchronized (accounting) {
                if (owners[fork] != philosopher) violations++;
                owners[fork] = -1;
                holders[fork]--;
            }
        }

        int violations() { synchronized (accounting) { return violations; } }
        int[] meals() { return meals; }
        int[] owners() { return owners; }
        int[] holders() { return holders; }
        int[] maxHolders() { return maxHolders; }
        int[] acquisitions() { return acquisitions; }
    }

    /** Documented lock ordering: the lower-numbered of the two forks always comes first. */
    static int[] forkOrder(int philosopher) {
        int left = philosopher;
        int right = (philosopher + 1) % PHILOSOPHERS;
        return new int[]{Math.min(left, right), Math.max(left, right)};
    }

    public static void main(String[] args) {
        // Hand-checked ordering table: only philosopher 4 wraps around, taking fork 0 first.
        int[][] expectedOrder = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 4}};
        for (int i = 0; i < PHILOSOPHERS; i++) {
            check(Arrays.equals(forkOrder(i), expectedOrder[i]), "documented increasing fork order " + i);
        }

        ForkTable table = new ForkTable();
        Thread[] diners = new Thread[PHILOSOPHERS];
        for (int i = 0; i < PHILOSOPHERS; i++) {
            final int philosopher = i;
            diners[i] = new Thread(() -> {
                int[] order = forkOrder(philosopher);
                for (int meal = 0; meal < MEALS; meal++) table.eat(philosopher, order[0], order[1]);
            });
            diners[i].start();
        }
        for (Thread diner : diners) {
            join(diner, 10000);
            check(!diner.isAlive(), "philosopher did not finish in time (deadlock?)");
        }

        int totalMeals = 0;
        for (int count : table.meals()) {
            check(count == MEALS, "every philosopher ate MEALS times");
            totalMeals += count;
        }
        check(totalMeals == PHILOSOPHERS * MEALS, "total meals == 5 * MEALS");
        check(table.violations() == 0, "per-fork single-holder invariant never violated");
        for (int owner : table.owners()) check(owner == -1, "every fork released at the end");
        for (int count : table.holders()) check(count == 0, "no residual holders");
        for (int count : table.maxHolders()) check(count == 1, "one holder at a time per fork");
        for (int count : table.acquisitions()) {
            check(count == 2 * MEALS, "each fork acquired by exactly its two neighbours");
        }

        System.out.println("PASS");
    }

    static void join(Thread thread, long timeoutMillis) {
        try {
            thread.join(timeoutMillis);
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
        }
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
