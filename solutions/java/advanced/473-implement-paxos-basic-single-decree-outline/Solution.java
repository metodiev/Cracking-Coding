/** Task 473 - Implement Paxos basic single-decree outline. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability) */
import java.util.*;

class Acceptor {
    final int id;
    int promised = -1;
    int acceptedRound = -1;
    Integer acceptedValue = null;
    final List<int[]> history = new ArrayList<>(); // {round, value}

    Acceptor(int id) {
        this.id = id;
    }
}

class Proposer {
    final int id;
    String phase = "idle"; // idle | P1 | P2 | done
    int roundNo = -1;
    Integer ownValue = null;
    Integer decidedValue = null;
    final List<Object[]> promises = new ArrayList<>(); // {acceptorId, acceptedRound, acceptedValue}
    final Set<Integer> accepts = new HashSet<>();
    int[] chosen = null; // {round, value}

    Proposer(int id) {
        this.id = id;
    }
}

class PaxosSim {
    static final int MAJORITY = 2; // of 3 acceptors

    final Acceptor[] acceptors = new Acceptor[3];
    final Proposer[] proposers = new Proposer[2];
    final ArrayDeque<Object[]> queue = new ArrayDeque<>(); // {destKind, destId, msg}
    final List<int[]> chosenEvents = new ArrayList<>();
    final Set<Integer> oracleChosen = new HashSet<>();
    final int[] lastPromised = {-1, -1, -1};

    PaxosSim() {
        for (int i = 0; i < 3; i++) acceptors[i] = new Acceptor(i);
        for (int i = 0; i < 2; i++) proposers[i] = new Proposer(i);
    }

    void propose(int p, int roundNo, int value) {
        Proposer pr = proposers[p];
        pr.phase = "P1";
        pr.roundNo = roundNo;
        pr.ownValue = value;
        pr.decidedValue = null;
        pr.promises.clear();
        pr.accepts.clear();
        for (int a = 0; a < 3; a++) {
            queue.addLast(new Object[]{"A", a, new Object[]{"prepare", roundNo, p}});
        }
    }

    void deliverOne() {
        Object[] item = queue.pollFirst();
        String destKind = (String) item[0];
        int destId = (Integer) item[1];
        Object[] msg = (Object[]) item[2];
        String kind = (String) msg[0];
        if (destKind.equals("A")) {
            Acceptor a = acceptors[destId];
            if (kind.equals("prepare")) {
                int roundNo = (Integer) msg[1];
                int p = (Integer) msg[2];
                if (roundNo > a.promised) {
                    a.promised = roundNo;
                    queue.addLast(new Object[]{"P", p, new Object[]{"promise", roundNo, a.id,
                            a.acceptedRound, a.acceptedValue}});
                } else {
                    queue.addLast(new Object[]{"P", p, new Object[]{"nack", roundNo, a.id}});
                }
            } else { // accept
                int roundNo = (Integer) msg[1];
                int value = (Integer) msg[2];
                int proposer = (Integer) msg[3];
                if (roundNo >= a.promised) {
                    a.promised = roundNo;
                    a.acceptedRound = roundNo;
                    a.acceptedValue = value;
                    a.history.add(new int[]{roundNo, value});
                    queue.addLast(new Object[]{"P", proposer, new Object[]{"accepted", roundNo, a.id, value}});
                } else {
                    queue.addLast(new Object[]{"P", proposer, new Object[]{"nack", roundNo, a.id}});
                }
            }
        } else {
            Proposer pr = proposers[destId];
            if (kind.equals("promise")) {
                int roundNo = (Integer) msg[1];
                int accId = (Integer) msg[2];
                Integer accRound = (Integer) msg[3];
                Integer accValue = (Integer) msg[4];
                if (!pr.phase.equals("P1") || roundNo != pr.roundNo) return;
                pr.promises.add(new Object[]{accId, accRound, accValue});
                if (pr.promises.size() >= MAJORITY) {
                    int bestRound = -1;
                    int bestValue = pr.ownValue;
                    for (Object[] promise : pr.promises) {
                        Integer value = (Integer) promise[2];
                        if (value != null && (Integer) promise[1] > bestRound) {
                            bestRound = (Integer) promise[1];
                            bestValue = value;
                        }
                    }
                    pr.decidedValue = bestValue;
                    pr.phase = "P2";
                    for (int a = 0; a < 3; a++) {
                        queue.addLast(new Object[]{"A", a, new Object[]{"accept", roundNo, bestValue, pr.id}});
                    }
                }
            } else if (kind.equals("accepted")) {
                int roundNo = (Integer) msg[1];
                int accId = (Integer) msg[2];
                int value = (Integer) msg[3];
                if (!pr.phase.equals("P2") || roundNo != pr.roundNo) return;
                pr.accepts.add(accId);
                if (pr.accepts.size() >= MAJORITY) {
                    pr.chosen = new int[]{roundNo, value};
                    pr.phase = "done";
                    chosenEvents.add(new int[]{roundNo, value});
                }
            }
        }
    }

    void deliver(int k) {
        for (int i = 0; i < k && !queue.isEmpty(); i++) deliverOne();
    }

    void drain() {
        while (!queue.isEmpty()) deliverOne();
    }

    void drop(int k) {
        for (int i = 0; i < k && !queue.isEmpty(); i++) queue.pollFirst();
    }

    void check() {
        Set<Integer> oracle = new HashSet<>();
        for (Acceptor a : acceptors) {
            for (int[] entry : a.history) {
                int count = 0;
                for (Acceptor b : acceptors) {
                    for (int[] other : b.history) {
                        if (other[0] == entry[0] && other[1] == entry[1]) count++;
                    }
                }
                if (count >= MAJORITY) oracle.add(entry[1]);
            }
        }
        oracleChosen.addAll(oracle);
        Set<Integer> chosenValues = new HashSet<>(oracleChosen);
        for (int[] event : chosenEvents) chosenValues.add(event[1]);
        check(chosenValues.size() <= 1, "at most one value may be chosen: " + chosenValues);
        for (Acceptor a : acceptors) {
            check(a.promised >= lastPromised[a.id], "promise regression");
            lastPromised[a.id] = a.promised;
            for (int i = 1; i < a.history.size(); i++) {
                check(a.history.get(i)[0] >= a.history.get(i - 1)[0], "accept rounds not monotone");
            }
        }
    }

    void run(List<Object[]> actions) {
        for (Object[] action : actions) {
            String kind = (String) action[0];
            if (kind.equals("propose")) propose((Integer) action[1], (Integer) action[2], (Integer) action[3]);
            else if (kind.equals("deliver")) deliver((Integer) action[1]);
            else if (kind.equals("drain")) drain();
            else if (kind.equals("drop")) drop((Integer) action[1]);
            check();
        }
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static List<Object[]> actions(Object[]... items) {
        return new ArrayList<>(Arrays.asList(items));
    }
}

public class Solution {
    static int lcg(int seed) {
        return (seed * 75 + 74) % 65537;
    }

    static List<Object[]> randomActions(int count, int seed, int startRound) {
        List<Object[]> actions = new ArrayList<>();
        int rnd = startRound;
        for (int i = 0; i < count; i++) {
            seed = lcg(seed);
            int a = seed % 8;
            seed = lcg(seed);
            int p = seed % 2;
            seed = lcg(seed);
            int value = seed % 20;
            seed = lcg(seed);
            int k = seed % 3;
            if (a < 3) {
                actions.add(new Object[]{"propose", p, rnd, value});
                rnd++;
            } else if (a == 3) {
                actions.add(new Object[]{"deliver", 1 + k});
            } else if (a < 6) {
                actions.add(new Object[]{"drain"});
            } else if (a == 6) {
                actions.add(new Object[]{"drop", k});
            } else {
                actions.add(new Object[]{"deliver", 1});
            }
        }
        return actions;
    }

    static Set<Integer> historyValues(PaxosSim sim) {
        Set<Integer> values = new HashSet<>();
        for (Acceptor a : sim.acceptors) {
            for (int[] entry : a.history) values.add(entry[1]);
        }
        return values;
    }

    public static void main(String[] args) {
        PaxosSim sim = new PaxosSim();

        // P0's value 7 is accepted by a majority (A0, A1) but all replies are dropped.
        sim.run(PaxosSim.actions(
                new Object[]{"propose", 0, 1000, 7}, new Object[]{"deliver", 3}, new Object[]{"deliver", 2},
                new Object[]{"deliver", 3}, new Object[]{"drop", 3}));
        PaxosSim.check(sim.chosenEvents.isEmpty(), "P0 unaware of the choice");
        PaxosSim.check(sim.oracleChosen.equals(new HashSet<>(Arrays.asList(7))), "7 chosen by oracle");
        PaxosSim.check(sim.acceptors[0].history.size() == 1 && sim.acceptors[0].history.get(0)[1] == 7, "A0 accepted 7");
        PaxosSim.check(sim.acceptors[1].history.size() == 1 && sim.acceptors[1].history.get(0)[1] == 7, "A1 accepted 7");
        PaxosSim.check(sim.acceptors[2].history.isEmpty(), "A2 has no accept");

        // P1 proposes 9 with a higher round; the promises carry (1000, 7), so it must re-propose 7.
        sim.run(PaxosSim.actions(new Object[]{"propose", 1, 1001, 9}, new Object[]{"drain"}));
        PaxosSim.check(sim.proposers[1].decidedValue == 7, "P1 must adopt the chosen value, not its own 9");
        PaxosSim.check(sim.proposers[1].chosen != null && sim.proposers[1].chosen[0] == 1001
                && sim.proposers[1].chosen[1] == 7, "P1 chose 7 at round 1001");
        PaxosSim.check(sim.chosenEvents.size() == 1, "one engine chosen event");

        // P0 retries with round 1002 and its own value 8; it must also re-propose 7.
        sim.run(PaxosSim.actions(new Object[]{"propose", 0, 1002, 8}, new Object[]{"drain"}));
        PaxosSim.check(sim.proposers[0].decidedValue == 7, "P0 retry must also re-propose 7");
        PaxosSim.check(sim.chosenEvents.size() == 2 && sim.chosenEvents.get(1)[0] == 1002
                && sim.chosenEvents.get(1)[1] == 7, "round 1002 chose 7");
        PaxosSim.check(historyValues(sim).equals(new HashSet<>(Arrays.asList(7))),
                "values 9 and 8 must never be accepted anywhere");

        // Randomized churn with losses: safety (at most one chosen value) must hold throughout.
        sim.run(randomActions(30, 13579, 2000));

        // Fair finale: fresh proposals with all messages delivered; liveness must return.
        for (int p = 0; p < 2; p++) {
            sim.run(PaxosSim.actions(new Object[]{"propose", p, 3000 + p, 5 + p}, new Object[]{"drain"}));
        }
        PaxosSim.check(!sim.chosenEvents.isEmpty(), "fair run must choose a value");
        Set<Integer> chosenValues = new HashSet<>();
        for (int[] event : sim.chosenEvents) chosenValues.add(event[1]);
        PaxosSim.check(chosenValues.equals(new HashSet<>(Arrays.asList(7))), "every choice is 7");
        PaxosSim.check(sim.oracleChosen.equals(new HashSet<>(Arrays.asList(7))), "oracle agrees");
        System.out.println("PASS");
    }
}
