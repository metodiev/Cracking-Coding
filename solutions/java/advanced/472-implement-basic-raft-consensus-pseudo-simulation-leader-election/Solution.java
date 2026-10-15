/** Task 472 - Implement basic Raft consensus pseudo-simulation (leader election outline). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability) */
import java.util.*;

class Entry {
    final int term;
    final String cmd;

    Entry(int term, String cmd) {
        this.term = term;
        this.cmd = cmd;
    }

    boolean same(Entry o) {
        return o != null && term == o.term && cmd.equals(o.cmd);
    }
}

class Node {
    static final int N = 5;
    final int id;
    int term = 0;
    int votedFor = -1;
    int commit = 0;
    String state = "F"; // F follower, C candidate, L leader
    final List<Entry> log = new ArrayList<>();
    final Set<Integer> voters = new HashSet<>();
    final int[] nextIndex = new int[N];
    final int[] matchIndex = new int[N];

    Node(int id) {
        this.id = id;
        Arrays.fill(matchIndex, -1);
    }
}

abstract class Msg {
    final int term;

    Msg(int term) {
        this.term = term;
    }
}

class RVMsg extends Msg {
    final int cand, lastIdx, lastTerm;

    RVMsg(int term, int cand, int lastIdx, int lastTerm) {
        super(term);
        this.cand = cand;
        this.lastIdx = lastIdx;
        this.lastTerm = lastTerm;
    }
}

class RVRespMsg extends Msg {
    final int voter;
    final boolean granted;

    RVRespMsg(int term, int voter, boolean granted) {
        super(term);
        this.voter = voter;
        this.granted = granted;
    }
}

class AEMsg extends Msg {
    final int leader, prevIdx, prevTerm, commit;
    final List<Entry> entries;

    AEMsg(int term, int leader, int prevIdx, int prevTerm, List<Entry> entries, int commit) {
        super(term);
        this.leader = leader;
        this.prevIdx = prevIdx;
        this.prevTerm = prevTerm;
        this.entries = entries;
        this.commit = commit;
    }
}

class AERespMsg extends Msg {
    final int follower;
    final boolean success;
    final int hint;

    AERespMsg(int term, int follower, boolean success, int hint) {
        super(term);
        this.follower = follower;
        this.success = success;
        this.hint = hint;
    }
}

class InFlight {
    final int from, to;
    final Msg msg;

    InFlight(int from, int to, Msg msg) {
        this.from = from;
        this.to = to;
        this.msg = msg;
    }
}

class RaftSim {
    static final int N = 5;
    static final int MAJORITY = 3;

    final List<Node> nodes = new ArrayList<>();
    final boolean[] alive = new boolean[N];
    List<Set<Integer>> partition = null;
    final ArrayDeque<InFlight> queue = new ArrayDeque<>();
    final Map<Integer, Integer> leadersByTerm = new TreeMap<>();
    final Map<Integer, Entry> committed = new TreeMap<>();
    int committedUpTo = 0;
    final int[] lastTerm = new int[N];
    final int[] lastCommit = new int[N];

    RaftSim() {
        for (int i = 0; i < N; i++) {
            nodes.add(new Node(i));
            alive[i] = true;
        }
    }

    boolean connected(int a, int b) {
        if (partition == null) return true;
        for (Set<Integer> g : partition) if (g.contains(a) && g.contains(b)) return true;
        return false;
    }

    void send(int frm, int to, Msg msg) {
        if (!alive[frm] || !alive[to]) return;
        if (!connected(frm, to)) return;
        queue.addLast(new InFlight(frm, to, msg));
    }

    void drain() {
        while (!queue.isEmpty()) {
            InFlight f = queue.pollFirst();
            if (!alive[f.from] || !alive[f.to] || !connected(f.from, f.to)) continue;
            deliver(f.from, f.to, f.msg);
        }
    }

    void crash(int i) {
        alive[i] = false;
    }

    void recover(int i) {
        alive[i] = true;
        nodes.get(i).state = "F";
    }

    void partitionOff(List<Set<Integer>> groups) {
        partition = groups;
    }

    void heal() {
        partition = null;
    }

    void timeout(int i) {
        if (!alive[i]) return;
        Node n = nodes.get(i);
        n.term += 1;
        n.votedFor = i;
        n.state = "C";
        n.voters.clear();
        n.voters.add(i);
        int lastIdx = n.log.size() - 1;
        int lastLogTerm = n.log.isEmpty() ? 0 : n.log.get(n.log.size() - 1).term;
        for (int j = 0; j < N; j++) {
            if (j != i) send(i, j, new RVMsg(n.term, i, lastIdx, lastLogTerm));
        }
    }

    boolean client(String cmd) {
        Node best = null;
        for (Node n : nodes) {
            if (n.state.equals("L") && alive[n.id] && (best == null || n.term > best.term)) best = n;
        }
        if (best == null) return false;
        best.log.add(new Entry(best.term, cmd));
        return true;
    }

    void replicate() {
        for (Node n : nodes) {
            if (!n.state.equals("L") || !alive[n.id]) continue;
            for (int f = 0; f < N; f++) {
                if (f == n.id) continue;
                int nn = Math.min(n.nextIndex[f], n.log.size());
                int prevIdx = nn - 1;
                int prevTerm = prevIdx >= 0 ? n.log.get(prevIdx).term : 0;
                send(n.id, f, new AEMsg(n.term, n.id, prevIdx, prevTerm,
                        new ArrayList<>(n.log.subList(nn, n.log.size())), n.commit));
            }
        }
    }

    void deliver(int frm, int to, Msg msg) {
        Node n = nodes.get(to);
        if (msg instanceof RVMsg) {
            RVMsg m = (RVMsg) msg;
            if (m.term < n.term) {
                send(to, frm, new RVRespMsg(n.term, to, false));
                return;
            }
            if (m.term > n.term) {
                n.term = m.term;
                n.votedFor = -1;
                if (!n.state.equals("F")) n.state = "F";
            }
            int ownIdx = n.log.size() - 1;
            int ownTerm = n.log.isEmpty() ? 0 : n.log.get(n.log.size() - 1).term;
            boolean upToDate = m.lastTerm > ownTerm || (m.lastTerm == ownTerm && m.lastIdx >= ownIdx);
            boolean granted = (n.votedFor == -1 || n.votedFor == m.cand) && upToDate;
            if (granted) n.votedFor = m.cand;
            send(to, frm, new RVRespMsg(n.term, to, granted));
        } else if (msg instanceof RVRespMsg) {
            RVRespMsg m = (RVRespMsg) msg;
            if (m.term > n.term) {
                n.term = m.term;
                n.votedFor = -1;
                n.state = "F";
                return;
            }
            if (n.state.equals("C") && m.term == n.term && m.granted && !n.voters.contains(m.voter)) {
                n.voters.add(m.voter);
                if (n.voters.size() >= MAJORITY) becomeLeader(n);
            }
        } else if (msg instanceof AEMsg) {
            AEMsg m = (AEMsg) msg;
            if (m.term < n.term) {
                send(to, frm, new AERespMsg(n.term, to, false, n.log.size()));
                return;
            }
            if (m.term > n.term) {
                n.term = m.term;
                n.votedFor = -1;
            }
            n.state = "F";
            if (m.prevIdx >= n.log.size() || (m.prevIdx >= 0 && n.log.get(m.prevIdx).term != m.prevTerm)) {
                send(to, frm, new AERespMsg(n.term, to, false, n.log.size()));
                return;
            }
            int idx = m.prevIdx + 1;
            for (Entry entry : m.entries) {
                if (idx < n.log.size()) {
                    if (n.log.get(idx).term != entry.term) {
                        while (n.log.size() > idx) n.log.remove(n.log.size() - 1);
                        n.log.add(entry);
                    }
                } else {
                    n.log.add(entry);
                }
                idx++;
            }
            int lastNew = m.prevIdx + m.entries.size() + 1;
            if (m.commit > n.commit) {
                int newCommit = Math.min(m.commit, lastNew);
                for (int i = n.commit; i < newCommit; i++) recordCommit(i, n.log.get(i));
                n.commit = Math.max(n.commit, newCommit);
            }
            send(to, frm, new AERespMsg(n.term, to, true, lastNew));
        } else {
            AERespMsg m = (AERespMsg) msg;
            if (m.term > n.term) {
                n.term = m.term;
                n.votedFor = -1;
                n.state = "F";
                return;
            }
            if (!n.state.equals("L") || m.term != n.term) return;
            if (m.success) {
                if (m.hint > n.matchIndex[m.follower]) {
                    n.matchIndex[m.follower] = m.hint;
                    n.nextIndex[m.follower] = m.hint + 1;
                }
                int newCommit = n.commit;
                for (int cand = n.log.size(); cand > n.commit; cand--) {
                    if (n.log.get(cand - 1).term != n.term) continue;
                    int votes = 0;
                    for (int j = 0; j < N; j++) {
                        if ((j == n.id && n.log.size() >= cand) || n.matchIndex[j] >= cand) votes++;
                    }
                    if (votes >= MAJORITY) {
                        newCommit = cand;
                        break;
                    }
                }
                if (newCommit > n.commit) {
                    for (int i = n.commit; i < newCommit; i++) recordCommit(i, n.log.get(i));
                    n.commit = newCommit;
                }
            } else {
                n.nextIndex[m.follower] = Math.max(0, Math.min(n.nextIndex[m.follower] - 1, m.hint));
            }
        }
    }

    void becomeLeader(Node n) {
        n.state = "L";
        Integer prev = leadersByTerm.get(n.term);
        check(prev == null || prev == n.id, "two leaders in same term");
        leadersByTerm.put(n.term, n.id);
        Arrays.fill(n.nextIndex, n.log.size());
        Arrays.fill(n.matchIndex, -1);
        for (int i = 0; i < committedUpTo; i++) {
            check(i < n.log.size() && n.log.get(i).same(committed.get(i)), "leader lacks committed entry " + i);
        }
    }

    void recordCommit(int i, Entry entry) {
        Entry prev = committed.get(i);
        check(prev == null || prev.same(entry), "committed entry changed at " + i);
        committed.put(i, entry);
        while (committed.containsKey(committedUpTo)) committedUpTo++;
    }

    void checkInvariants() {
        for (int a = 0; a < N; a++) {
            for (int b = a + 1; b < N; b++) {
                List<Entry> la = nodes.get(a).log, lb = nodes.get(b).log;
                int m = Math.min(la.size(), lb.size());
                for (int i = 0; i < m; i++) {
                    if (la.get(i).term == lb.get(i).term) {
                        for (int k = 0; k <= i; k++) {
                            check(la.get(k).same(lb.get(k)), "log matching violated " + a + "," + b + "@" + i);
                        }
                    }
                }
            }
        }
        for (int i = 0; i < N; i++) {
            Node n = nodes.get(i);
            check(n.commit <= n.log.size(), "commit beyond log");
            check(n.term >= lastTerm[i], "term regression");
            check(n.commit >= lastCommit[i], "commit regression");
            lastTerm[i] = n.term;
            lastCommit[i] = n.commit;
        }
    }

    void run(List<Object[]> actions) {
        for (Object[] action : actions) {
            String kind = (String) action[0];
            if (kind.equals("timeout")) timeout((Integer) action[1]);
            else if (kind.equals("client")) client((String) action[1]);
            else if (kind.equals("replicate")) replicate();
            else if (kind.equals("drain")) drain();
            else if (kind.equals("partition")) partitionOff(castGroups(action[1]));
            else if (kind.equals("heal")) heal();
            else if (kind.equals("crash")) crash((Integer) action[1]);
            else if (kind.equals("recover")) recover((Integer) action[1]);
            checkInvariants();
        }
    }

    @SuppressWarnings("unchecked")
    static List<Set<Integer>> castGroups(Object o) {
        List<Integer>[] raw = (List<Integer>[]) o;
        List<Set<Integer>> out = new ArrayList<>();
        for (List<Integer> g : raw) out.add(new TreeSet<>(g));
        return out;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}

public class Solution {
    static int lcg(int seed) {
        return (seed * 75 + 74) % 65537;
    }

    static List<Object[]> randomActions(int count, int seed) {
        List<Object[]> actions = new ArrayList<>();
        int counter = 0;
        for (int k = 0; k < count; k++) {
            seed = lcg(seed);
            int a = seed % 8;
            seed = lcg(seed);
            int node = seed % RaftSim.N;
            if (a == 0) {
                actions.add(new Object[]{"timeout", node});
            } else if (a == 1) {
                counter++;
                actions.add(new Object[]{"client", "c" + counter});
            } else if (a == 2) {
                actions.add(new Object[]{"replicate"});
            } else if (a == 3) {
                actions.add(new Object[]{"drain"});
            } else if (a == 4) {
                seed = lcg(seed);
                int mask = seed;
                List<Integer> g0 = new ArrayList<>();
                List<Integer> g1 = new ArrayList<>();
                for (int i = 0; i < RaftSim.N; i++) {
                    if (((mask >> i) & 1) == 1) g0.add(i);
                    else g1.add(i);
                }
                actions.add(new Object[]{"partition", new List[]{g0, g1}});
            } else if (a == 5) {
                actions.add(new Object[]{"heal"});
            } else if (a == 6) {
                actions.add(new Object[]{"crash", node});
            } else {
                actions.add(new Object[]{"recover", node});
            }
            if (a != 3) actions.add(new Object[]{"drain"});
        }
        return actions;
    }

    static boolean sameLog(List<Entry> a, List<Entry> b) {
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); i++) if (!a.get(i).same(b.get(i))) return false;
        return true;
    }

    @SuppressWarnings("unchecked")
    public static void main(String[] args) {
        RaftSim sim = new RaftSim();
        sim.run(actions(new Object[]{"timeout", 0}, new Object[]{"drain"}));
        check(sim.leadersByTerm.size() == 1 && sim.leadersByTerm.get(1) == 0, "term 1 leader");
        sim.run(actions(new Object[]{"client", "x=1"}, new Object[]{"replicate"}, new Object[]{"drain"},
                new Object[]{"replicate"}, new Object[]{"drain"}));
        check(sim.committedUpTo == 1, "x committed");
        check(sim.committed.get(0).same(new Entry(1, "x=1")), "x entry");

        sim.run(actions(new Object[]{"crash", 0}, new Object[]{"timeout", 1}, new Object[]{"drain"}));
        check(sim.leadersByTerm.get(2) != null && sim.leadersByTerm.get(2) == 1, "term 2 leader");
        sim.run(actions(new Object[]{"client", "y=2"}, new Object[]{"replicate"}, new Object[]{"drain"},
                new Object[]{"replicate"}, new Object[]{"drain"}));
        check(sim.committedUpTo == 2 && sim.committed.get(1).same(new Entry(2, "y=2")), "y committed");

        sim.run(actions(new Object[]{"partition", new List[]{Arrays.asList(0, 1, 2), Arrays.asList(3, 4)}},
                new Object[]{"client", "z=3"}, new Object[]{"replicate"}, new Object[]{"drain"},
                new Object[]{"replicate"}, new Object[]{"drain"}));
        check(sim.committedUpTo == 2, "z must not commit without quorum");
        Node n1 = sim.nodes.get(1);
        Node n2 = sim.nodes.get(2);
        check(n1.log.get(n1.log.size() - 1).same(new Entry(2, "z=3")), "z appended at leader");
        check(n2.log.get(n2.log.size() - 1).same(new Entry(2, "z=3")), "z replicated to peer");

        sim.run(actions(new Object[]{"timeout", 3}, new Object[]{"drain"}));
        check(!sim.leadersByTerm.containsKey(3), "minority cannot elect");
        sim.run(actions(new Object[]{"heal"}, new Object[]{"drain"}));
        sim.run(actions(new Object[]{"timeout", 1}, new Object[]{"drain"}));
        check(!sim.leadersByTerm.containsKey(3), "stale votes block term 3");
        sim.run(actions(new Object[]{"timeout", 1}, new Object[]{"drain"}));
        check(sim.leadersByTerm.get(4) != null && sim.leadersByTerm.get(4) == 1, "term 4 leader");

        sim.run(actions(new Object[]{"client", "w=4"}, new Object[]{"replicate"}, new Object[]{"drain"},
                new Object[]{"replicate"}, new Object[]{"drain"}, new Object[]{"replicate"}, new Object[]{"drain"}));
        check(sim.committedUpTo == 4, "w commits z");
        check(sim.committed.get(2).same(new Entry(2, "z=3")), "z committed");
        check(sim.committed.get(3).same(new Entry(4, "w=4")), "w committed");
        List<Entry> expected = Arrays.asList(new Entry(1, "x=1"), new Entry(2, "y=2"),
                new Entry(2, "z=3"), new Entry(4, "w=4"));
        for (int i = 1; i <= 4; i++) check(sameLog(sim.nodes.get(i).log, expected), "log mismatch node " + i);

        sim.run(randomActions(64, 24680));

        List<Object[]> finale = new ArrayList<>();
        finale.add(new Object[]{"heal"});
        for (int i = 0; i < RaftSim.N; i++) finale.add(new Object[]{"recover", i});
        finale.add(new Object[]{"drain"});
        finale.add(new Object[]{"drain"});
        sim.run(finale);
        int best = 0;
        for (int i = 1; i < RaftSim.N; i++) {
            Node a = sim.nodes.get(i), b = sim.nodes.get(best);
            int ta = a.log.isEmpty() ? 0 : a.log.get(a.log.size() - 1).term;
            int tb = b.log.isEmpty() ? 0 : b.log.get(b.log.size() - 1).term;
            if (ta > tb || (ta == tb && a.log.size() > b.log.size())) best = i;
        }
        for (int attempt = 0; attempt < 40; attempt++) {
            sim.run(actions(new Object[]{"timeout", best}, new Object[]{"drain"}));
            if (sim.nodes.get(best).state.equals("L")) break;
        }
        check(sim.nodes.get(best).state.equals("L"), "most up-to-date node must win");
        sim.run(actions(new Object[]{"client", "end"}));
        for (int attempt = 0; attempt < 12; attempt++) {
            sim.run(actions(new Object[]{"replicate"}, new Object[]{"drain"}, new Object[]{"replicate"}, new Object[]{"drain"}));
            if (sim.committedUpTo == sim.nodes.get(best).log.size()) break;
        }
        check(sim.committedUpTo == sim.nodes.get(best).log.size(), "final commit caught up");
        boolean sawEnd = false;
        for (Entry e : sim.committed.values()) if (e.same(new Entry(sim.nodes.get(best).term, "end"))) sawEnd = true;
        check(sawEnd, "end command committed");
        System.out.println("PASS");
    }

    static List<Object[]> actions(Object[]... items) {
        return new ArrayList<>(Arrays.asList(items));
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
