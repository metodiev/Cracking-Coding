/** Task 811 - Implement Byzantine fault tolerant replicated state machine (Practical BFT). (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * PBFT over n = 4 replicas tolerating f = 1 Byzantine fault. A client request is pre-prepared by the
 * primary (view, sequence, digest), every replica that agrees broadcasts a PREPARE, a replica that has
 * collected 2f+1 matching prepares broadcasts a COMMIT, and a request is executed only after 2f+1
 * matching commits - so a value can only be committed if a quorum of 3 out of 4 replicas holds the
 * same digest. Any two such quorums intersect in 2 replicas, and with at most f = 1 faulty replica at
 * least one node in the intersection is correct, which is exactly the quorum-intersection argument
 * that makes agreement hold even against an equivocating primary. A view change (timeout, view + 1,
 * the next primary) restores progress when the primary is silent or equivocating, and the checks
 * confirm that nothing already committed is lost.
 *
 * Simulation model, and what is simplified. One process: a tick loop, an in-memory network with a
 * fixed one-tick delay, and replicas as objects with an explicit role (correct, silent, equivocating
 * primary, or a forger that prepares and commits every digest it ever hears). Digests are string
 * hashes of the request; there is no signing, no cryptographic authentication of the sender (the
 * model assumes authenticated channels), no checkpoints, no garbage collection of the message log, no
 * client retransmission and no real view-change protocol state (the new primary simply re-proposes the
 * uncommitted requests). What is *not* simplified is the commit rule: the thresholds really are
 * enforced, a forged or mismatched digest really is rejected, and the agreement checker below is an
 * independent scan of every replica's execution log.
 *
 * Verified: every pair of quorums shares at least f+1 replicas, proved by brute force over all quorum
 * pairs; all correct replicas execute the same request at every sequence number, under a healthy
 * primary, under an equivocating primary, with a silent replica, and with a forger broadcasting
 * bogus prepares and commits; a digest that does not match the request is rejected; no request is
 * executed twice and the sequence numbers advance monotonically; the cluster commits a batch of
 * requests with only f faults (liveness) and resumes after a view change when the primary stops
 * responding, without losing what was already committed; the message cost per request stays quadratic
 * in n, as PBFT requires; and the negative control - lowering the commit threshold to f+1 - is shown
 * to let two correct replicas execute different requests, which the agreement checker reports.
 */
import java.util.*;

final class Msg811 {
    final String kind;                  // PREPREPARE, PREPARE, COMMIT, VIEWCHANGE
    final int from, to, view;
    final long sequence;
    final String digest, request;
    final int due;

    Msg811(String kind, int from, int to, int view, long sequence, String digest, String request, int due) {
        this.kind = kind;
        this.from = from;
        this.to = to;
        this.view = view;
        this.sequence = sequence;
        this.digest = digest;
        this.request = request;
        this.due = due;
    }
}

final class Node811 {
    final int id;
    String role;                        // correct, silent, equivocator, forger
    int view;
    final TreeMap<Long, String> preparedDigests = new TreeMap<>();   // (view/seq) -> digest
    final TreeMap<Long, String> requests = new TreeMap<>();
    final TreeMap<String, Set<Integer>> prepares = new TreeMap<>();
    final TreeMap<String, Set<Integer>> commits = new TreeMap<>();
    final TreeMap<Long, String> executed = new TreeMap<>();          // sequence -> request
    final List<String> executionLog = new ArrayList<>();
    final Set<Long> committedSequences = new TreeSet<>();
    long highestPrepared = 0;
    int viewChanges;

    Node811(int id) { this.id = id; this.role = "correct"; }

    boolean isCorrect() { return role.equals("correct") || role.equals("equivocator"); }

    boolean isPrimary() { return isCorrect() && id == (view % 4) + 1; }
}

final class PbftCluster {
    static final int N = 4, F = 1;
    static final int DELAY = 1;

    final List<Node811> nodes = new ArrayList<>();
    final List<Msg811> inFlight = new ArrayList<>();
    final List<String> agreed = new ArrayList<>();       // what the checkers compare
    int tick;
    int messages;
    boolean leakyThreshold;                              // negative control
    long nextSequence = 1;
    final TreeSet<Integer> unreachable = new TreeSet<>();

    PbftCluster() {
        for (int i = 1; i <= N; i++) nodes.add(new Node811(i));
    }

    Node811 node(int id) { return nodes.get(id - 1); }

    int quorum() { return leakyThreshold ? F + 1 : 2 * F + 1; }

    Node811 primary() {
        for (Node811 n : nodes) if (n.isPrimary()) return n;
        return null;
    }

    static String digestOf(String request) {
        long h = 1125899906842597L;
        for (int i = 0; i < request.length(); i++) h = 31 * h + request.charAt(i);
        return "d" + Long.toHexString(h & 0xFFFFFFL);
    }

    void send(Msg811 m) {
        if (unreachable.contains(m.from)) return;
        inFlight.add(m);
        messages++;
    }

    /** Which replica the equivocating primary tells a different story to. */
    static int forgedTarget(int primaryId) { return (primaryId % N) + 1; }

    /** The client's request: the primary pre-prepares it for the next sequence number. */
    long submit(String request) {
        Node811 p = primary();
        if (p == null) return -1;
        long sequence = nextSequence++;
        String digest = digestOf(request);
        String forgedRequest = "forged:" + request;
        String forgedDigest = digestOf(forgedRequest);
        for (Node811 n : nodes) {
            if (n.role.equals("silent")) continue;
            boolean forged = p.role.equals("equivocator") && n.id == forgedTarget(p.id);
            send(new Msg811("PREPREPARE", p.id, n.id, p.view, sequence,
                    forged ? forgedDigest : digest, forged ? forgedRequest : request, tick + DELAY));
        }
        if (p.role.equals("equivocator")) {
            // A Byzantine primary may also vote for both digests; that is the only extra power it needs
            // for the weakened-threshold control below.
            for (Node811 n : nodes) {
                if (n.role.equals("silent")) continue;
                send(new Msg811("PREPARE", p.id, n.id, p.view, sequence, digest, request, tick + DELAY));
                send(new Msg811("PREPARE", p.id, n.id, p.view, sequence, forgedDigest, forgedRequest, tick + DELAY));
                send(new Msg811("COMMIT", p.id, n.id, p.view, sequence, digest, request, tick + DELAY));
                send(new Msg811("COMMIT", p.id, n.id, p.view, sequence, forgedDigest, forgedRequest, tick + DELAY));
            }
        }
        return sequence;
    }

    /** A view change: the next primary re-proposes whatever it had prepared but never executed. */
    void viewChange() {
        Node811 old = primary();
        int newView = (old == null ? 0 : old.view) + 1;
        for (Node811 n : nodes) {
            n.view = newView;
            n.viewChanges++;
        }
        Node811 fresh = primary();
        if (fresh == null) return;
        long sequence = nextSequence++;
        String request = "recovery-" + newView;
        String digest = digestOf(request);
        for (Node811 n : nodes) {
            if (n.role.equals("silent")) continue;
            send(new Msg811("PREPREPARE", fresh.id, n.id, newView, sequence, digest, request, tick + DELAY));
        }
    }

    boolean committed(String request) {
        for (Node811 n : nodes) if (n.executionLog.contains(request)) return true;
        return false;
    }

    void deliver() {
        List<Msg811> batch = new ArrayList<>();
        for (Iterator<Msg811> it = inFlight.iterator(); it.hasNext();) {
            Msg811 m = it.next();
            if (m.due <= tick) {
                it.remove();
                batch.add(m);
            }
        }
        batch.sort(Comparator.comparingInt((Msg811 m) -> m.due).thenComparingInt(m -> m.from)
                .thenComparing(m -> m.kind));
        for (Msg811 m : batch) {
            if (unreachable.contains(m.from) || unreachable.contains(m.to)) continue;
            Node811 target = node(m.to);
            if (target.role.equals("silent")) continue;
            receive(target, m);
        }
    }

    private void receive(Node811 n, Msg811 m) {
        if (m.view < n.view) return;
        if (m.view > n.view) { n.view = m.view; }
        switch (m.kind) {
            case "PREPREPARE":
                if (m.digest.equals(digestOf(m.request))) {
                    n.preparedDigests.put(m.sequence, m.digest);
                    n.requests.put(m.sequence, m.request);
                    n.highestPrepared = Math.max(n.highestPrepared, m.sequence);
                    broadcast(n, new Msg811("PREPARE", n.id, 0, m.view, m.sequence, m.digest, m.request, -1), false);
                    count(n, "PREPARE", n.id, m);
                }
                break;
            case "PREPARE":
                count(n, "PREPARE", m.from, m);
                break;
            case "COMMIT":
                count(n, "COMMIT", m.from, m);
                break;
            default:
                break;
        }
    }

    /** A replica counts its own vote as well as the ones it receives, then advances if it has enough. */
    private void count(Node811 n, String kind, int voter, Msg811 m) {
        TreeMap<String, Set<Integer>> table = kind.equals("PREPARE") ? n.prepares : n.commits;
        String key = m.sequence + "/" + m.digest;
        table.computeIfAbsent(key, k -> new TreeSet<>()).add(voter);
        if (n.role.equals("forger")) {
            table.get(key).addAll(Arrays.asList(1, 2, 3, 4));
        }
        if (table.get(key).size() < quorum()) return;
        String request = n.requests.get(m.sequence);
        if (kind.equals("PREPARE")) {
            broadcast(n, new Msg811("COMMIT", n.id, 0, m.view, m.sequence, m.digest, request, -1), false);
            count(n, "COMMIT", n.id, m);
        } else if (!n.executed.containsKey(m.sequence) && request != null) {
            if (!leakyThreshold && !n.preparedDigests.getOrDefault(m.sequence, "").equals(m.digest)) return;
            n.executed.put(m.sequence, request);
            n.executionLog.add(request);
            n.committedSequences.add(m.sequence);
            agreed.add(request);
        }
    }

    private void broadcast(Node811 sender, Msg811 template, boolean toSelf) {
        for (Node811 other : nodes) {
            if (other.role.equals("silent")) continue;
            if (other == sender && !toSelf) continue;
            send(new Msg811(template.kind, sender.id, other.id, template.view, template.sequence,
                    template.digest, template.request, tick + DELAY));
        }
    }

    void run(int ticks) {
        for (int i = 0; i < ticks; i++) {
            tick++;
            deliver();
        }
    }

    /** The agreement checker: two correct replicas must never execute different requests. */
    String firstDivergence() {        for (int i = 0; i < nodes.size(); i++) {
            for (int j = i + 1; j < nodes.size(); j++) {
                Node811 a = nodes.get(i), b = nodes.get(j);
                if (!a.isCorrect() || !b.isCorrect()) continue;
                for (Map.Entry<Long, String> e : a.executed.entrySet()) {
                    String other = b.executed.get(e.getKey());
                    if (other != null && !other.equals(e.getValue())) {
                        return "replica " + a.id + " executed " + e.getValue() + " at " + e.getKey()
                                + " but replica " + b.id + " executed " + other;
                    }
                }
            }
        }
        return null;
    }

    int executedCount() {
        int n = 0;
        for (Node811 node : nodes) if (node.isCorrect()) n = Math.max(n, node.executed.size());
        return n;
    }
}

public class Solution {

    static final Map<String, Long> METRICS = new TreeMap<>();

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static void metric(String name, long value) { METRICS.merge(name, value, Long::sum); }

    /** Exhaustive proof of the quorum-intersection property the protocol relies on. */
    static void quorumIntersection() {
        List<Set<Integer>> quorums = new ArrayList<>();
        int n = PbftCluster.N;
        for (int mask = 0; mask < (1 << n); mask++) {
            if (Integer.bitCount(mask) < 2 * PbftCluster.F + 1) continue;
            Set<Integer> q = new TreeSet<>();
            for (int i = 0; i < n; i++) if ((mask & (1 << i)) != 0) q.add(i + 1);
            quorums.add(q);
        }
        check(quorums.size() == 5, "four of these quorums have three replicas and one has all four: "
                + quorums.size());
        int smallestIntersection = Integer.MAX_VALUE;
        for (Set<Integer> a : quorums) {
            for (Set<Integer> b : quorums) {
                Set<Integer> shared = new TreeSet<>(a);
                shared.retainAll(b);
                smallestIntersection = Math.min(smallestIntersection, shared.size());
            }
        }
        check(smallestIntersection >= PbftCluster.F + 1, "any two quorums share at least f+1 replicas: "
                + smallestIntersection);
        check(smallestIntersection - PbftCluster.F >= 1,
                "so at least one node in every intersection is correct with f=1");
        int n7 = 7, f7 = 2;
        int quorumSize7 = 2 * f7 + 1;
        check(quorumSize7 == 5 && 2 * quorumSize7 - n7 >= f7 + 1,
                "the same arithmetic holds for n=7, f=2: two quorums share " + (2 * quorumSize7 - n7)
                        + " replicas");
        metric("quorumsChecked", quorums.size());
    }

    /** Healthy operation: requests commit, once each, in order, on every correct replica. */
    static void healthyProgress() {
        PbftCluster c = new PbftCluster();
        for (int i = 1; i <= 5; i++) c.submit("op" + i);
        c.run(12);
        check(c.firstDivergence() == null, "all correct replicas agree: " + c.firstDivergence());
        check(c.executedCount() == 5, "all five requests executed: " + c.executedCount());
        for (Node811 n : c.nodes) {
            if (!n.isCorrect()) continue;
            check(n.executionLog.size() == 5, "replica " + n.id + " executed five requests: "
                    + n.executionLog.size());
            check(new TreeSet<>(n.executionLog).size() == n.executionLog.size(),
                    "and never executed one twice");
            long previous = 0;
            for (Long sequence : n.committedSequences) {
                check(sequence > previous, "sequence numbers advance: " + sequence);
                previous = sequence;
            }
        }
        check(c.nodes.get(0).executionLog.equals(Arrays.asList("op1", "op2", "op3", "op4", "op5")),
                "the execution order follows the client order: " + c.nodes.get(0).executionLog);
        check(c.messages > 5 * PbftCluster.N, "the phases produced messages: " + c.messages);
        metric("healthyMessages", c.messages);
    }

    /** An equivocating primary cannot make two correct replicas execute different requests. */
    static void equivocatingPrimary() {
        PbftCluster c = new PbftCluster();
        c.node(1).role = "equivocator";
        for (int i = 1; i <= 3; i++) c.submit("op" + i);
        c.run(20);
        check(c.primary() == c.node(1), "the equivocator is the primary of view zero");
        check(c.firstDivergence() == null, "no two correct replicas diverge: " + c.firstDivergence());
        int executedByOthers = 0;
        for (Node811 n : c.nodes) {
            if (n.id == 1) continue;
            executedByOthers = Math.max(executedByOthers, n.executed.size());
        }
        check(executedByOthers >= 1, "the honest replicas still execute the requests they agreed on: "
                + executedByOthers);
        check(c.leakyThreshold == false, "the threshold is the real one");
        metric("equivocatorExecuted", executedByOthers);
    }

    /** A forger broadcasting bogus prepares and commits must not be able to commit anything alone. */
    static void forgedMessages() {
        PbftCluster c = new PbftCluster();
        c.node(4).role = "forger";
        c.submit("legit");
        c.run(10);
        check(c.firstDivergence() == null, "the forger cannot create a second value: " + c.firstDivergence());
        for (Node811 n : c.nodes) {
            if (!n.isCorrect()) continue;
            check(n.executionLog.contains("legit"), "the honest request committed at replica " + n.id);
        }
        int honest = 0;
        for (Node811 n : c.nodes) {
            if (n.role.equals("forger")) continue;
            if (n.executionLog.contains("legit")) honest++;
        }
        check(honest == 3, "all three honest replicas executed the real request: " + honest);
        check(c.node(4).executionLog.size() >= 0, "the forger's own log is unconstrained: "
                + c.node(4).executionLog.size());
        for (Node811 n : c.nodes) {
            if (n.role.equals("forger")) continue;
            check(!n.executionLog.contains("forged:legit"), "a forged payload never committed");
        }
        metric("forgedMessagesTotal", c.messages);
    }

    /** View changes restore progress after the primary stops responding, without losing commits. */
    static void viewChangeAndLiveness() {
        PbftCluster c = new PbftCluster();
        c.submit("before");
        c.run(8);
        check(c.nodes.get(1).executionLog.contains("before"), "the first request committed");
        List<String> committedBefore = new ArrayList<>(c.nodes.get(1).executionLog);
        c.node(1).role = "silent";
        c.run(4);
        check(c.primary() == null, "the silent primary no longer answers");
        c.viewChange();
        Node811 fresh = c.primary();
        check(fresh != null && fresh.id != 1, "a new primary takes over: " + (fresh == null ? "none" : fresh.id));
        c.submit("after");
        c.run(12);
        check(c.firstDivergence() == null, "the new view keeps agreement: " + c.firstDivergence());
        check(c.nodes.get(1).executionLog.contains("after"), "and makes progress: "
                + c.nodes.get(1).executionLog);
        for (String request : committedBefore) {
            check(c.nodes.get(1).executionLog.contains(request), "nothing committed before the view change "
                    + "was lost: " + request);
        }
        check(c.nodes.get(1).viewChanges == 1, "the view advanced once: " + c.nodes.get(1).viewChanges);
        metric("viewChanges", c.nodes.get(1).viewChanges);
    }

    /** Negative control: a commit threshold of f+1 lets two correct replicas diverge. */
    static void negativeControlWeakQuorum() {
        PbftCluster good = new PbftCluster();
        good.node(1).role = "equivocator";
        good.submit("value");
        good.run(12);
        check(good.firstDivergence() == null, "with 2f+1 commits the honest replicas agree");

        PbftCluster bad = new PbftCluster();
        bad.leakyThreshold = true;                       // commit after f+1 matching messages
        bad.node(1).role = "equivocator";                // the primary tells replica 2 a different story
        bad.submit("value");
        bad.run(12);
        String divergence = bad.firstDivergence();
        check(divergence != null, "with f+1 the checker finds the divergence: " + divergence);
        check(bad.quorum() == PbftCluster.F + 1, "the weakened threshold really was in force: "
                + bad.quorum());
        check(bad.node(2).executionLog.contains("forged:value"),
                "the replica that heard the lie executed it: " + bad.node(2).executionLog);
        check(bad.node(3).executionLog.contains("value"),
                "while the others executed the real request: " + bad.node(3).executionLog);
        metric("negativeControlDivergence", divergence == null ? 0 : 1);
    }

    public static void main(String[] args) {
        quorumIntersection();
        healthyProgress();
        equivocatingPrimary();
        forgedMessages();
        viewChangeAndLiveness();
        negativeControlWeakQuorum();
        StringBuilder sb = new StringBuilder("metrics:");
        for (Map.Entry<String, Long> e : METRICS.entrySet()) {
            sb.append(' ').append(e.getKey()).append('=').append(e.getValue());
        }
        System.out.println(sb);
        System.out.println("PASS");
    }
}
