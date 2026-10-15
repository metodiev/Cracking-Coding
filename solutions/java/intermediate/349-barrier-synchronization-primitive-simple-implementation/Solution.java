/**
 * Task 349 - Reusable barrier synchronization primitive. (Tier: Intermediate, CTCI Ch 15)
 *
 * Generation-based barrier: the last arriving participant resets the counter, flips the generation
 * and wakes everyone, so the object is instantly reusable.  One participant is a no-op.
 * await(timeout) returns false when the caller times out before all parties arrived and withdraws
 * its own count, leaving the barrier usable.  The scripted arena runs four real threads through
 * five rounds, records a global event sequence and asserts that no departure precedes every
 * arrival of the same round.
 */
import java.util.*;

class Barrier {
    private final int parties;
    private int count = 0;
    private int generation = 0;

    Barrier(int parties) {
        if (parties <= 0) throw new IllegalArgumentException("parties must be positive");
        this.parties = parties;
    }

    synchronized boolean await(long timeoutMillis) throws InterruptedException {
        int observed = generation;
        count++;
        if (count == parties) {
            count = 0;
            generation++;
            notifyAll();
            return true;
        }
        long deadline = timeoutMillis > 0 ? System.currentTimeMillis() + timeoutMillis : 0;
        while (observed == generation) {
            if (timeoutMillis <= 0) {
                wait();
            } else {
                long remaining = deadline - System.currentTimeMillis();
                if (remaining <= 0) {
                    count--;                  // withdraw; the barrier stays usable
                    return false;
                }
                wait(remaining);
            }
        }
        return true;
    }

    synchronized int[] state() {
        return new int[]{count, generation};
    }
}

public class Solution {
    static class Event {
        final int seq;
        final String kind;
        final int round;
        final String name;

        Event(int seq, String kind, int round, String name) {
            this.seq = seq;
            this.kind = kind;
            this.round = round;
            this.name = name;
        }
    }

    static void record(List<Event> events, String kind, int round, String name) {
        synchronized (events) {
            events.add(new Event(events.size(), kind, round, name));
        }
    }

    static Runnable participant(String name, Barrier barrier, int rounds, List<Event> events, List<String> failures) {
        return () -> {
            try {
                for (int round = 0; round < rounds; round++) {
                    record(events, "arrive", round, name);
                    if (!barrier.await(15000)) {
                        failures.add(name + ": timed out");
                        return;
                    }
                    record(events, "depart", round, name);
                }
            } catch (InterruptedException exc) {
                Thread.currentThread().interrupt();
                failures.add(name + ": interrupted");
            }
        };
    }

    static boolean join(Thread thread) {
        try {
            thread.join(15000);
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
        }
        return !thread.isAlive();
    }

    public static void main(String[] args) throws InterruptedException {
        int rounds = 5;
        Barrier barrier = new Barrier(4);
        List<Event> events = new ArrayList<>();
        List<String> failures = Collections.synchronizedList(new ArrayList<>());
        String[] names = {"P0", "P1", "P2", "P3"};
        List<Thread> threads = new ArrayList<>();
        for (String name : names) {
            Thread thread = new Thread(participant(name, barrier, rounds, events, failures), name);
            thread.setDaemon(true);
            threads.add(thread);
        }
        for (Thread thread : threads) thread.start();
        for (Thread thread : threads) check(join(thread), "participant terminated");
        check(failures.isEmpty(), "no participant failed: " + failures);

        int arrivals = 0;
        int departures = 0;
        for (Event event : events) {
            if (event.kind.equals("arrive")) arrivals++;
            else departures++;
        }
        check(arrivals == 4 * rounds, "20 arrivals");
        check(departures == 4 * rounds, "20 departures");

        for (int round = 0; round < rounds; round++) {
            int maxArrival = -1;
            int minDeparture = Integer.MAX_VALUE;
            int arrivalCount = 0;
            int departureCount = 0;
            for (Event event : events) {
                if (event.round != round) continue;
                if (event.kind.equals("arrive")) {
                    arrivalCount++;
                    maxArrival = Math.max(maxArrival, event.seq);
                } else {
                    departureCount++;
                    minDeparture = Math.min(minDeparture, event.seq);
                }
            }
            check(arrivalCount == 4 && departureCount == 4, "round " + round + " counts");
            check(maxArrival < minDeparture, "round " + round + " released early");
        }

        for (String name : names) {
            List<Integer> arrivalRounds = new ArrayList<>();
            List<Integer> departureRounds = new ArrayList<>();
            for (Event event : events) {
                if (!event.name.equals(name)) continue;
                if (event.kind.equals("arrive")) arrivalRounds.add(event.round);
                else departureRounds.add(event.round);
            }
            check(arrivalRounds.size() == rounds && departureRounds.size() == rounds, name + " passed every round");
            for (int round = 0; round < rounds; round++) {
                check(arrivalRounds.get(round) == round, name + " arrival round order");
                check(departureRounds.get(round) == round, name + " departure round order");
            }
        }

        check(Arrays.equals(barrier.state(), new int[]{0, rounds}), "barrier reset after every round");

        Barrier solo = new Barrier(1);
        for (int i = 0; i < 3; i++) check(solo.await(0), "single-participant barrier is a no-op");
        check(Arrays.equals(solo.state(), new int[]{0, 3}), "solo generation counted");

        Barrier pair = new Barrier(2);
        check(!pair.await(50), "timeout returns false");
        check(Arrays.equals(pair.state(), new int[]{0, 0}), "timed-out caller withdrew cleanly");

        List<Boolean> results = Collections.synchronizedList(new ArrayList<>());
        Thread threadA = new Thread(() -> {
            try {
                results.add(pair.await(15000));
            } catch (InterruptedException exc) {
                Thread.currentThread().interrupt();
            }
        });
        Thread threadB = new Thread(() -> {
            try {
                results.add(pair.await(15000));
            } catch (InterruptedException exc) {
                Thread.currentThread().interrupt();
            }
        });
        threadA.setDaemon(true);
        threadB.setDaemon(true);
        threadA.start();
        threadB.start();
        check(join(threadA) && join(threadB), "pair threads terminated");
        check(results.size() == 2 && results.get(0) && results.get(1), "both parties released");
        check(Arrays.equals(pair.state(), new int[]{0, 1}), "pair generation advanced once");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
