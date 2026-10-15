/** Task 356 - Implement lock-free stack outline (CAS). (Tier: Intermediate, CTCI Ch 15) */
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

public class Solution {
    static final int THREADS = 4;
    static final int PUSHES_PER_THREAD = 250;

    static class Node {
        final int value;
        Node next;

        Node(int value) { this.value = value; }
    }

    /**
     * Michael-Scott style lock-free stack: push/pop are CAS retry loops on the single head
     * pointer (AtomicReference.compareAndSet). pop() returns null, the documented sentinel,
     * on an empty stack; pushed values are ints and can never be null. The ABA hazard of
     * pointer CAS is documented but cannot occur here: nodes are never recycled during a
     * run, and any lost CAS is retried with a freshly read head.
     */
    static class LockFreeStack {
        private final AtomicReference<Node> head = new AtomicReference<>();
        private final AtomicLong casAttempts = new AtomicLong();
        private final AtomicLong casFailures = new AtomicLong();

        long casAttempts() { return casAttempts.get(); }
        long casFailures() { return casFailures.get(); }

        int push(int value) {
            int retries = 0;
            while (true) {
                Node current = head.get();
                Node node = new Node(value);
                node.next = current;
                casAttempts.incrementAndGet();
                if (head.compareAndSet(current, node)) return retries;
                casFailures.incrementAndGet();
                retries++;
            }
        }

        /** Pops the top value; returns null (the documented sentinel) when empty. */
        Integer pop() {
            while (true) {
                Node current = head.get();
                if (current == null) return null;
                casAttempts.incrementAndGet();
                if (head.compareAndSet(current, current.next)) return current.value;
                casFailures.incrementAndGet();
            }
        }

        boolean isEmpty() { return head.get() == null; }

        List<Integer> snapshotValues() {
            List<Integer> values = new ArrayList<>();
            for (Node node = head.get(); node != null; node = node.next) values.add(node.value);
            return values;
        }
    }

    public static void main(String[] args) {
        // Single-threaded script: LIFO order and the empty sentinel.
        LockFreeStack stack = new LockFreeStack();
        check(stack.pop() == null, "pop on an empty stack returns the sentinel");
        for (int value = 1; value <= 5; value++) stack.push(value);
        List<Integer> drained = new ArrayList<>();
        for (int i = 0; i < 5; i++) drained.add(stack.pop());
        check(drained.equals(List.of(5, 4, 3, 2, 1)), "LIFO order");
        check(stack.pop() == null, "empty again after draining");
        check(stack.casFailures() == 0, "single-threaded script never loses a CAS");

        // Concurrent pushes: no node is lost.
        LockFreeStack shared = new LockFreeStack();
        int total = THREADS * PUSHES_PER_THREAD;
        Thread[] pushers = new Thread[THREADS];
        for (int t = 0; t < THREADS; t++) {
            final int threadIndex = t;
            pushers[t] = new Thread(() -> {
                for (int offset = 0; offset < PUSHES_PER_THREAD; offset++) {
                    shared.push(threadIndex * PUSHES_PER_THREAD + offset);
                }
            });
            pushers[t].start();
        }
        for (Thread pusher : pushers) {
            join(pusher, 10000);
            check(!pusher.isAlive(), "pusher did not finish in time");
        }

        List<Integer> values = shared.snapshotValues();
        check(values.size() == total, "no node lost under concurrent pushes");
        List<Integer> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        check(sorted.equals(range(total)), "every pushed value present exactly once");

        // Concurrent pops: no node is lost or duplicated.
        List<List<Integer>> poppedPerThread = new ArrayList<>();
        for (int i = 0; i < THREADS; i++) poppedPerThread.add(Collections.synchronizedList(new ArrayList<>()));
        Thread[] poppers = new Thread[THREADS];
        for (int t = 0; t < THREADS; t++) {
            final int threadIndex = t;
            poppers[t] = new Thread(() -> {
                while (true) {
                    Integer value = shared.pop();
                    if (value == null) break;
                    poppedPerThread.get(threadIndex).add(value);
                }
            });
            poppers[t].start();
        }
        for (Thread popper : poppers) {
            join(popper, 10000);
            check(!popper.isAlive(), "popper did not finish in time");
        }

        List<Integer> allPopped = new ArrayList<>();
        for (List<Integer> chunk : poppedPerThread) allPopped.addAll(chunk);
        check(allPopped.size() == total, "no node lost under concurrent pops");
        Collections.sort(allPopped);
        check(allPopped.equals(range(total)), "every pushed value popped exactly once");
        check(shared.pop() == null, "stack is empty again");
        check(shared.isEmpty(), "empty state after the concurrent phase");

        // Exact CAS accounting: every successful push/pop is one attempt, every retry one
        // failure. The poppers that stopped with the sentinel made no CAS attempt at all.
        check(shared.casFailures() >= 0, "CAS failures are non-negative (they may be 0)");
        check(shared.casAttempts() == 2L * total + shared.casFailures(),
              "attempts == successful operations + failures");

        System.out.println("PASS");
    }

    static List<Integer> range(int total) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < total; i++) result.add(i);
        return result;
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
