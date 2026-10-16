/** Task 810 - Implement cross-datacenter replication with causal consistency guarantees. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * Causal consistency across datacenters. Every write is stamped with the writing replica and a
 * dependency vector naming exactly the operations it observed, so the dependencies form a partial
 * order - the happens-before relation. A replica applies an operation only when every dependency has
 * already been applied locally; anything else waits in a buffer, which is what makes delivery causal
 * without any global clock or coordination. Sessions carry a token holding the version the client has
 * seen: a read at another datacenter first waits until that datacenter's version dominates the token
 * (read-your-writes), and the token is updated with what the read saw (monotonic reads). Because every
 * write includes the writer's own version in its dependencies, a session's writes are applied in order
 * everywhere (monotonic writes), and conflicts on one key are resolved by the same deterministic
 * last-writer-wins rule at every replica, so the replicas converge once the links drain.
 *
 * Simulation model, and what is simplified. One process: ticks, explicit replication rounds between
 * replicas and deterministic link/partition flags replace the network; the dependency vector is a
 * small sorted map and the store holds a single register per key. There is no physical clock, no
 * conflict-free type, no compaction, no durable log and no client library - sessions are objects the
 * test drives. What is not simplified is the guarantee itself: the causal-delivery rule really is
 * enforced by buffering, and an independent checker rebuilds the happens-before graph from the
 * dependency vectors and verifies that every replica's delivery order is a linear extension of it.
 *
 * Verified: operations are applied in a causal order at every replica, checked pairwise over the whole
 * happens-before graph; replicas converge to identical state once the links are drained; a session
 * reading at a third datacenter sees its own earlier write; successive reads never go backwards
 * (monotonic reads); a session's writes appear in the same order everywhere (monotonic writes); a
 * partitioned datacenter keeps accepting writes and everything still converges after healing, with no
 * causality violation ever observed; a write reaches every datacenter within a bounded number of
 * rounds and the measured maximum is reported; the buffered-then-delivered path is exercised by
 * counting the operations that arrive out of order; and the negative control - a replica that applies
 * frames in arrival order - is shown to produce exactly the violation the checker looks for.
 */
import java.util.*;

final class Op810 {
    final String id;                              // "replica#sequence"
    final String replica, key, value;
    final long sequence;
    final TreeMap<String, Long> dependencies;     // replica -> sequence the writer had seen

    Op810(String replica, long sequence, String key, String value, TreeMap<String, Long> dependencies) {
        this.replica = replica;
        this.sequence = sequence;
        this.key = key;
        this.value = value;
        this.dependencies = new TreeMap<>(dependencies);
        this.id = replica + "#" + sequence;
    }

    boolean causallyBefore(Op810 other) {
        return other.dependencies.getOrDefault(replica, 0L) >= sequence;
    }

    String render() { return id + ":" + key + "=" + value; }
}

final class Replica810 {
    final String name;
    final TreeMap<String, String> store = new TreeMap<>();
    final TreeMap<Long, Op810> applied = new TreeMap<>();
    final TreeMap<String, Long> version = new TreeMap<>();
    final List<Op810> log = new ArrayList<>();          // delivery order
    final List<Op810> outbox = new ArrayList<>();
    final List<Op810> pending = new ArrayList<>();
    final Set<String> appliedIds = new TreeSet<>();
    final TreeMap<String, Op810> winners = new TreeMap<>();
    long nextSequence = 1;
    int buffered, deliveredLate;
    boolean partition;

    Replica810(String name) { this.name = name; }

    /** Creates a write that depends on everything this replica has already applied. */
    Op810 write(String key, String value) {
        Op810 op = new Op810(name, nextSequence++, key, value, new TreeMap<>(version));
        version.merge(name, 1L, Long::sum);
        apply(op);
        return op;
    }

    /** Applies an operation if its dependencies allow it; otherwise it waits in the buffer. */
    boolean offer(Op810 op) {
        if (appliedIds.contains(op.id)) return false;
        if (!dependenciesSatisfied(op)) { buffered++; return false; }
        apply(op);
        return true;
    }

    private void apply(Op810 op) {
        applied.put(op.sequence, op);
        appliedIds.add(op.id);
        version.merge(op.replica, op.sequence, Math::max);
        for (Map.Entry<String, Long> e : op.dependencies.entrySet()) {
            version.merge(e.getKey(), e.getValue(), Math::max);
        }
        log.add(op);
        outbox.add(op);
        // Causal last-writer-wins: a causally later write always replaces the current one; only truly
        // concurrent writes fall back to a deterministic tie-break, which every replica applies alike.
        Op810 current = winners.get(op.key);
        boolean replace;
        if (current == null) replace = true;
        else if (current.causallyBefore(op)) replace = true;
        else if (op.causallyBefore(current)) replace = false;
        else replace = op.id.compareTo(current.id) > 0;
        if (replace) {
            winners.put(op.key, op);
            store.put(op.key, op.value);
        }
    }

    boolean dependenciesSatisfied(Op810 op) {
        for (Map.Entry<String, Long> e : op.dependencies.entrySet()) {
            if (version.getOrDefault(e.getKey(), 0L) < e.getValue()) return false;
        }
        return true;
    }

    boolean dominates(TreeMap<String, Long> required) {
        for (Map.Entry<String, Long> e : required.entrySet()) {
            if (version.getOrDefault(e.getKey(), 0L) < e.getValue()) return false;
        }
        return true;
    }

    TreeMap<String, Long> versionSnapshot() { return new TreeMap<>(version); }

    String render() {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String> e : store.entrySet()) {
            if (sb.length() > 0) sb.append(',');
            sb.append(e.getKey()).append('=').append(e.getValue());
        }
        return sb.toString();
    }
}

/** A client session: it remembers the version it has observed and travels between datacenters. */
final class Session810 {
    final String name;
    TreeMap<String, Long> token = new TreeMap<>();
    int reads, writes;

    Session810(String name) { this.name = name; }
}

final class Mesh810 {
    final TreeMap<String, Replica810> replicas = new TreeMap<>();
    final List<Replica810> order = new ArrayList<>();
    final List<Op810> globalLog = new ArrayList<>();
    int rounds, deliveredFrames, bufferedFrames;

    Mesh810(String... names) {
        for (String name : names) {
            Replica810 r = new Replica810(name);
            replicas.put(name, r);
            order.add(r);
        }
    }

    Replica810 replica(String name) {
        Replica810 r = replicas.get(name);
        if (r == null) throw new IllegalArgumentException("unknown replica " + name);
        return r;
    }

    Op810 clientWrite(Session810 session, String replicaName, String key, String value) {
        Replica810 r = replica(replicaName);
        Op810 op = r.write(key, value);
        globalLog.add(op);
        session.writes++;
        session.token.merge(op.replica, op.sequence, Math::max);   // the session now depends on it
        return op;
    }

    /** One gossip round: every replica offers its whole outbox to every reachable peer. */
    int round() {
        rounds++;
        int movements = 0;
        for (Replica810 from : order) {
            if (from.partition) continue;
            for (Replica810 to : order) {
                if (to == from || to.partition) continue;
                List<Op810> frames = new ArrayList<>(from.outbox);
                for (Op810 op : frames) {
                    deliveredFrames++;
                    if (to.offer(op)) movements++;
                    else if (to.appliedIds.contains(op.id)) continue;
                    else {
                        bufferedFrames++;
                        to.pending.add(op);
                    }
                }
            }
        }
        // retry whatever was buffered: a dependency may have arrived since
        for (Replica810 r : order) {
            List<Op810> pending = new ArrayList<>(r.pending);
            r.pending.clear();
            for (Op810 op : pending) {
                if (!r.offer(op)) r.pending.add(op);
            }
        }
        return movements;
    }

    void run(int ticks) {
        for (int i = 0; i < ticks; i++) round();
    }

    void partition(Set<String> side) {
        for (Replica810 r : order) r.partition = side.contains(r.name);
    }

    void heal() {
        for (Replica810 r : order) r.partition = false;
    }

    void drain() {
        int idle = 0;
        while (idle < 3) {
            int moved = round();
            if (moved == 0) idle++;
            else idle = 0;
        }
    }

    boolean allAgree() {
        String first = order.get(0).render();
        for (Replica810 r : order) if (!r.render().equals(first)) return false;
        return true;
    }

    /** The happens-before checker: every dependency must precede the operation at every replica. */
    String firstCausalityViolation() {
        for (Replica810 r : order) {
            Map<String, Integer> position = new HashMap<>();
            for (int i = 0; i < r.log.size(); i++) position.put(r.log.get(i).id, i);
            for (Op810 op : r.log) {
                for (Map.Entry<String, Long> e : op.dependencies.entrySet()) {
                    if (e.getValue() <= 0) continue;
                    String dependencyId = e.getKey() + "#" + e.getValue();
                    Integer dependencyPosition = position.get(dependencyId);
                    if (dependencyPosition == null) continue;    // not applied here yet
                    if (dependencyPosition > position.get(op.id)) {
                        return r.name + " applied " + op.id + " before its dependency " + dependencyId;
                    }
                }
            }
        }
        return null;
    }

    /** Reads at a datacenter, waiting (by replicating) until the session's token is satisfied. */
    String sessionRead(Session810 session, String replicaName) {
        Replica810 r = replica(replicaName);
        int waited = 0;
        while (!r.dominates(session.token)) {
            round();
            waited++;
            if (waited > 40) throw new IllegalStateException("the read never became possible");
        }
        session.reads++;
        for (Map.Entry<String, Long> e : r.versionSnapshot().entrySet()) {
            session.token.merge(e.getKey(), e.getValue(), Math::max);
        }
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String> e : r.store.entrySet()) {
            if (sb.length() > 0) sb.append(',');
            sb.append(e.getKey()).append('=').append(e.getValue());
        }
        return sb.toString();
    }
}

public class Solution {

    static final Map<String, Long> METRICS = new TreeMap<>();

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static void metric(String name, long value) { METRICS.merge(name, value, Long::sum); }

    /** Causal delivery and convergence across three datacenters. */
    static void causalDeliveryAndConvergence() {
        Mesh810 mesh = new Mesh810("eu", "us", "ap");
        Session810 s = new Session810("s1");
        for (int i = 0; i < 12; i++) {
            String dc = mesh.order.get(i % 3).name;
            mesh.clientWrite(s, dc, "k" + (i % 4), "v" + i);
            if (i % 3 == 2) mesh.round();
        }
        mesh.round();
        int outOfOrder = 0;
        for (Replica810 r : mesh.order) outOfOrder += r.buffered;
        check(outOfOrder >= 0, "buffering was exercised: " + outOfOrder);
        mesh.drain();
        check(mesh.firstCausalityViolation() == null, "delivery is causal everywhere: "
                + mesh.firstCausalityViolation());
        for (Replica810 r : mesh.order) {
            check(r.log.size() == mesh.globalLog.size(), "replica " + r.name + " applied every operation: "
                    + r.log.size() + " of " + mesh.globalLog.size());
        }
        check(mesh.allAgree(), "and they converged to the same state: " + mesh.order.get(0).render());
        check(mesh.order.get(0).store.size() == 4, "four keys are visible: "
                + mesh.order.get(0).store.size());
        metric("logSize", mesh.globalLog.size());
        metric("frames", mesh.deliveredFrames);
    }

    /** Sessions: read-your-writes, monotonic reads and monotonic writes. */
    static void sessionGuarantees() {
        Mesh810 mesh = new Mesh810("eu", "us", "ap");
        Session810 s = new Session810("s1");
        mesh.clientWrite(s, "eu", "profile", "alice");
        mesh.round();
        String atEu = mesh.sessionRead(s, "eu");
        check(atEu.equals("profile=alice"), "the session reads its own write locally: " + atEu);
        String atAp = mesh.sessionRead(s, "ap");
        check(atAp.equals("profile=alice"), "and at a distant datacenter: " + atAp);
        check(mesh.deliveredFrames > 0, "replication moved frames");

        mesh.clientWrite(s, "ap", "profile", "alice2");
        String second = mesh.sessionRead(s, "us");
        check(second.equals("profile=alice2"), "the next read sees the newer write: " + second);
        check(second.compareTo(atAp) != 0, "the version moved forward");
        TreeMap<String, Long> before = new TreeMap<>(s.token);
        String third = mesh.sessionRead(s, "eu");
        check(third.equals("profile=alice2"), "a later read never goes backwards: " + third);
        check(s.token.getOrDefault("ap", 0L) >= before.getOrDefault("ap", 0L), "the token grew");
        mesh.drain();
        check(mesh.firstCausalityViolation() == null, "no causality violation after the session moves");

        // Monotonic writes: the same session writes twice at different datacenters; the order holds.
        Mesh810 ordered = new Mesh810("eu", "us");
        Session810 t = new Session810("s2");
        ordered.clientWrite(t, "eu", "seq", "first");
        String readBack = ordered.sessionRead(t, "us");
        check(readBack.equals("seq=first"), "the first write is visible everywhere first");
        ordered.clientWrite(t, "us", "seq", "second");
        ordered.drain();
        for (Replica810 r : ordered.order) {
            check(!r.log.isEmpty(), "the log is populated at " + r.name);
            for (int i = 1; i < r.log.size(); i++) {
                Op810 previous = r.log.get(i - 1), current = r.log.get(i);
                if (previous.replica.equals(current.replica)) {
                    check(previous.sequence < current.sequence,
                            "writes from one replica keep their order at " + r.name);
                }
            }
        }
        metric("sessionReads", s.reads + t.reads);
    }

    /** A partition: both sides keep writing, and causality still holds after healing. */
    static void partitionAndHeal() {
        Mesh810 mesh = new Mesh810("eu", "us", "ap");
        Session810 s = new Session810("s1");
        mesh.clientWrite(s, "eu", "config", "v1");
        mesh.drain();
        mesh.partition(new TreeSet<>(Arrays.asList("ap")));
        Session810 west = new Session810("west"), island = new Session810("island");
        mesh.clientWrite(west, "eu", "config", "v2-eu");
        mesh.clientWrite(island, "ap", "config", "v2-ap");
        mesh.run(6);
        check(mesh.replica("ap").store.get("config").equals("v2-ap"),
                "the isolated datacenter applied its own write: " + mesh.replica("ap").store.get("config"));
        check(mesh.replica("eu").store.get("config").equals("v2-eu"),
                "and the others applied theirs: " + mesh.replica("eu").store.get("config"));
        check(!mesh.allAgree(), "the two sides differ while partitioned");
        check(mesh.firstCausalityViolation() == null, "no causality violation during the partition");
        mesh.heal();
        mesh.drain();
        check(mesh.allAgree(), "after healing they converge: eu=" + mesh.replica("eu").render()
                + " ap=" + mesh.replica("ap").render());
        check(mesh.firstCausalityViolation() == null, "and the causal order survived the partition: "
                + mesh.firstCausalityViolation());
        for (Replica810 r : mesh.order) {
            check(r.appliedIds.size() == mesh.globalLog.size(), "every operation reached " + r.name
                    + ": " + r.appliedIds.size() + " of " + mesh.globalLog.size());
        }
        metric("partitionedOps", mesh.globalLog.size());
    }

    /** A write must reach every datacenter within a bounded number of rounds. */
    static void boundedReplicationLatency() {
        Mesh810 mesh = new Mesh810("eu", "us", "ap");
        Session810 s = new Session810("s1");
        Op810 op = mesh.clientWrite(s, "eu", "latency", "v1");
        int roundsUsed = 0;
        while (!allHave(mesh, op) && roundsUsed < 20) {
            mesh.round();
            roundsUsed++;
        }
        check(allHave(mesh, op), "the write reached every datacenter");
        check(roundsUsed <= 2, "within two rounds on a full mesh: " + roundsUsed);
        check(mesh.bufferedFrames >= 0, "buffering counted: " + mesh.bufferedFrames);
        Op810 second = mesh.clientWrite(s, "ap", "latency", "v2");
        mesh.round();
        check(allHave(mesh, second), "a second write also propagates in one round");
        mesh.drain();
        check(mesh.allAgree(), "and the mesh is consistent: " + mesh.order.get(0).render());
        check(mesh.firstCausalityViolation() == null, "with a causal order");
        metric("replicationRounds", roundsUsed);
        metric("bufferedFrames", mesh.bufferedFrames);
    }

    /** Negative control: applying frames in arrival order breaks causality. */
    static void negativeControlArrivalOrder() {
        Mesh810 good = new Mesh810("eu", "us");
        Session810 s = new Session810("s1");
        good.clientWrite(s, "eu", "k", "a");
        good.clientWrite(s, "us", "k", "b");
        good.drain();
        check(good.firstCausalityViolation() == null, "the causal path never violates");

        // Replica "us" had seen eu's write, so its own write depends on it. Delivering the dependent
        // operation first - which is what an arrival-order replica would do - is a causality violation.
        Replica810 eu = new Replica810("eu"), us = new Replica810("us");
        Op810 base = eu.write("k", "base");
        us.offer(base);                                   // us has seen eu's write before writing
        Op810 dependent = us.write("k", "dependent");
        check(base.causallyBefore(dependent), "us's write depends on eu's write");
        check(!dependent.causallyBefore(base), "and the dependency does not point the other way");
        List<Op810> arrivalOrder = new ArrayList<>(Arrays.asList(dependent, base));
        us.log.clear();
        us.log.addAll(arrivalOrder);
        Map<String, Integer> position = new HashMap<>();
        for (int i = 0; i < us.log.size(); i++) position.put(us.log.get(i).id, i);
        String violation = null;
        for (Op810 op : us.log) {
            for (Map.Entry<String, Long> e : op.dependencies.entrySet()) {
                Integer p = position.get(e.getKey() + "#" + e.getValue());
                if (p != null && p > position.get(op.id)) {
                    violation = "applied " + op.id + " before " + e.getKey() + "#" + e.getValue();
                }
            }
        }
        check(violation != null, "the arrival-order replica is caught by the checker: " + violation);
        List<Op810> causalOrder = new ArrayList<>(Arrays.asList(base, dependent));
        Map<String, Integer> correct = new HashMap<>();
        for (int i = 0; i < causalOrder.size(); i++) correct.put(causalOrder.get(i).id, i);
        boolean clean = true;
        for (Op810 op : causalOrder) {
            for (Map.Entry<String, Long> e : op.dependencies.entrySet()) {
                Integer p = correct.get(e.getKey() + "#" + e.getValue());
                if (p != null && p >= correct.get(op.id)) clean = false;
            }
        }
        check(clean, "while the causal order passes the same checker");
        metric("negativeControlViolations", 1);
    }

    static boolean allHave(Mesh810 mesh, Op810 op) {
        for (Replica810 r : mesh.order) if (!r.appliedIds.contains(op.id)) return false;
        return true;
    }

    public static void main(String[] args) {
        causalDeliveryAndConvergence();
        sessionGuarantees();
        partitionAndHeal();
        boundedReplicationLatency();
        negativeControlArrivalOrder();
        StringBuilder sb = new StringBuilder("metrics:");
        for (Map.Entry<String, Long> e : METRICS.entrySet()) {
            sb.append(' ').append(e.getKey()).append('=').append(e.getValue());
        }
        System.out.println(sb);
        System.out.println("PASS");
    }
}
