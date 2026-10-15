/**
 * Task 350 - Thread-safe singleton via double-checked locking. (Tier: Intermediate, CTCI Ch 15)
 *
 * DCL with a volatile field: the unlocked fast-path read avoids the lock, the re-check inside the
 * synchronized block is what makes it correct, and the volatile write publishes the fully
 * constructed object (without volatile another thread could see a partially initialised instance
 * in the JMM).  32 threads are released through one gate; all must receive the same instance and
 * the private constructor must run exactly once.
 */
import java.util.*;
import java.util.concurrent.CountDownLatch;

class Singleton {
    private static volatile Singleton instance;
    private static int instantiations = 0;

    private Singleton() {
        instantiations++;                    // only reachable while holding the class monitor
    }

    static Singleton getInstance() {
        Singleton local = instance;          // fast path: one volatile read, no lock
        if (local == null) {
            synchronized (Singleton.class) {
                local = instance;
                if (local == null) {         // re-check under the lock
                    local = new Singleton();
                    instance = local;        // volatile write
                }
            }
        }
        return local;
    }

    static synchronized int instantiations() {
        return instantiations;
    }
}

public class Solution {
    static boolean join(Thread thread) {
        try {
            thread.join(15000);
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
        }
        return !thread.isAlive();
    }

    public static void main(String[] args) {
        check(Singleton.instantiations() == 0, "lazy: nothing constructed yet");

        int callers = 32;
        CountDownLatch gate = new CountDownLatch(1);
        Singleton[] results = new Singleton[callers];
        List<Thread> threads = new ArrayList<>();
        for (int index = 0; index < callers; index++) {
            final int i = index;
            Thread thread = new Thread(() -> {
                try {
                    gate.await();
                } catch (InterruptedException exc) {
                    Thread.currentThread().interrupt();
                    return;
                }
                results[i] = Singleton.getInstance();
            });
            thread.setDaemon(true);
            threads.add(thread);
        }
        for (Thread thread : threads) thread.start();
        gate.countDown();                     // release all callers at once
        for (Thread thread : threads) check(join(thread), "caller terminated");

        check(results[0] != null, "first caller received an instance");
        for (int i = 1; i < callers; i++) {
            check(results[i] == results[0], "caller " + i + " received the same instance");
        }
        check(Singleton.instantiations() == 1, "the constructor ran exactly once");
        check(Singleton.getInstance() == results[0], "later callers reuse the instance");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
