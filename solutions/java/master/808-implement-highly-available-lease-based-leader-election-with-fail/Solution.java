/** Task 808 - Implement highly-available lease-based leader election with failover testing. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A quorum-based leader lease. A candidate collects votes; a voter only promises a lease when *its
 * own* clock says the previous lease it granted must have expired with certainty, which is why the
 * cooldown is the lease length plus twice the clock-drift bound plus the message delay. Because every
 * quorum overlaps the quorum that granted the previous lease in at least one node, at least one voter
 * remembers the old lease and refuses to break the guarantee - the standard argument that lets a
 * leader rely on a lease without synchronised clocks. The winner then holds the lease until the local
 * expiry and renews it on every heartbeat while it can still reach a quorum; a leader that loses its
 * quorum stops renewing, so its lease simply runs out in real time while the rest of the cluster still
 * has to wait out the cooldown before electing a successor.
 *
 * Simulation model, and what is simplified. No threads and no wall clock: real time is a tick counter,
 * each node has a fixed deterministic drift within the advertised bound, votes and heartbeats are
 * resolved procedurally by the driver rather than by a message queue, and a crash or a partition is a
 * flag. There is no disk, no leader-lease fencing token passed to clients and no failure detector
 * beyond "the peer did not answer". What is not simplified is the safety argument: the acquisition
 * condition really is derived from the drift bound and the delay, the brute-force checker below
 * converts every lease to the real-time interval it could possibly cover and proves that no two
 * leaders' intervals ever overlap, and the failover latency is measured against the same bound.
 *
 * Verified: exactly one leader at any time over a run with three failovers; leases never overlap in
 * real time, scanned exhaustively over every pair of leases (including the leases held by crashed and
 * by partitioned leaders); a lease is renewed while the leader is healthy and expires once it is not;
 * a partitioned minority can neither be elected nor renew; a new leader appears within the lease plus
 * cooldown bound after a crash, with the measured latency reported; terms only ever grow; an old
 * leader that has been superseded cannot win again with a stale lease; a single-node cluster trivially
 * elects itself; and the negative control - a lease shorter than the drift-plus-delay bound - is shown
 * to produce exactly the overlap the checker looks for.
 */
import java.util.*;

final class Node808 {
    static final long DRIFT_BOUND = 3;

    final int id;
    final long drift;                                  // fixed offset of this node's local clock
    String role = "follower";
    int term;
    int votedFor = -1;
    long leaseLocalStart = Long.MIN_VALUE / 4;          // start of the last lease this node helped grant
    long localElectionDeadline;
    boolean alive = true;
    boolean partitioned;
    int elections, votesGranted, renewals, preVotes;

    Node808(int id) {
        this.id = id;
        this.drift = Math.floorMod(id * 5, 2 * DRIFT_BOUND + 1) - DRIFT_BOUND;   // within +/- bound
    }

    long localClock(long tick) { return tick + drift; }

    boolean reachableFrom(Node808 other, Map<Integer, Boolean> links) {
        if (!alive || !other.alive) return false;
        if (partitioned != other.partitioned) return false;
        Boolean explicit = links.get(id * 100 + other.id);
        return explicit == null || explicit;
    }

    boolean isLeader() { return role.equals("leader") && alive; }
}

final class Lease {
    final int leader, term;
    long localStart, localEnd;

    Lease(int leader, int term, long localStart) {
        this.leader = leader;
        this.term = term;
        this.localStart = localStart;
        this.localEnd = localStart;
    }

    /** The real-time interval this lease could possibly cover, given the drift bound. */
    long realEarliestStart() { return localStart - Node808.DRIFT_BOUND; }

    long realLatestEnd() { return localEnd + Node808.DRIFT_BOUND; }

    boolean overlaps(Lease o) {
        return realEarliestStart() <= o.realLatestEnd() && o.realEarliestStart() <= realLatestEnd();
    }

    String render() { return "L" + leader + "@" + term + "[" + realEarliestStart() + ","
            + realLatestEnd() + "]"; }
}

final class Cluster808 {
    static final int DELAY = 2;
    static final int ELECTION_TIMEOUT = 8;
    static boolean TRACE;

    final List<Node808> nodes = new ArrayList<>();
    final List<Lease> leases = new ArrayList<>();
    final Map<Integer, Boolean> links = new HashMap<>();
    long tick;
    long leaseLength = 24;
    boolean leakyLeases;                                // negative control
    int elections, failovers, renewals;

    Cluster808(int n) {
        for (int i = 1; i <= n; i++) nodes.add(new Node808(i));
        for (Node808 node : nodes) node.localElectionDeadline = node.localClock(0) + ELECTION_TIMEOUT + node.id;
    }

    Node808 node(int id) { return nodes.get(id - 1); }

    List<Node808> alive() {
        List<Node808> out = new ArrayList<>();
        for (Node808 n : nodes) if (n.alive) out.add(n);
        return out;
    }

    Node808 leader() {
        for (Node808 n : nodes) if (n.isLeader()) return n;
        return null;
    }

    List<Node808> leaders() {
        List<Node808> out = new ArrayList<>();
        for (Node808 n : nodes) if (n.isLeader()) out.add(n);
        return out;
    }

    /**
     * The cooldown a voter insists on before it will help grant another lease: the old lease's length
     * plus four drift bounds (both the old and the new lease start are each uncertain by one bound)
     * plus the delay.
     */
    long cooldown() { return leaseLength + 4 * Node808.DRIFT_BOUND + DELAY; }

    /** True when a majority of the cluster is reachable from this node. */
    boolean hasQuorum(Node808 n) {
        int reachable = 0;
        for (Node808 other : nodes) if (other == n || n.reachableFrom(other, links)) reachable++;
        return reachable * 2 > nodes.size();
    }

    /**
     * A renewal is a fresh confirmation round: every reachable peer that still backs this term records
     * *its own* local time, which is what its cooldown will be measured from. That is what keeps a
     * leader from extending its lease beyond what the quorum remembers.
     */
    void renew(Node808 leader) {
        if (!leader.alive) return;
        for (Node808 peer : nodes) {
            if (peer == leader || !leader.reachableFrom(peer, links)) continue;
            if (peer.term > leader.term) {                 // it learned of a newer term: step down
                if (TRACE) System.err.println("t" + tick + " stepdown=" + leader.id + " " + leader.term
                        + " -> " + peer.term);
                leader.role = "follower";
                leader.term = peer.term;
                return;
            }
        }
        int confirms = 1;
        for (Node808 peer : nodes) {
            if (peer == leader || !leader.reachableFrom(peer, links)) continue;
            if (peer.term <= leader.term) confirms++;
        }
        if (confirms * 2 <= nodes.size()) return;        // no quorum: the lease simply runs out, and
        long now = leader.localClock(tick);              // no promise is recorded anywhere
        leader.leaseLocalStart = now;                    // the leader is a voter of its own quorum too
        for (Node808 peer : nodes) {
            if (peer == leader || !leader.reachableFrom(peer, links)) continue;
            if (peer.term <= leader.term) peer.leaseLocalStart = peer.localClock(tick);
        }
        leader.renewals++;
        renewals++;
        Lease lease = leases.get(leases.size() - 1);
        lease.localEnd = Math.max(lease.localEnd, now + leaseLength);
    }

    boolean pledge(Node808 peer, Node808 candidate) {
        boolean cooldownOver = peer.localClock(tick) > peer.leaseLocalStart + (leakyLeases ? 0 : cooldown());
        return peer.alive && candidate.reachableFrom(peer, links) && cooldownOver
                && peer.term <= candidate.term + 1;
    }

    /**
     * The candidate asks every reachable peer for a lease; a peer refuses while its cooldown runs.
     * A pre-vote round happens first, so a node that cannot win stays quiet instead of bumping its
     * term and deposing a healthy leader - the lease would otherwise be handed back and forth.
     */
    boolean campaign(Node808 candidate) {
        candidate.preVotes++;
        int pledges = 1;
        for (Node808 peer : nodes) {
            if (peer != candidate && pledge(peer, candidate)) pledges++;
        }
        if (pledges * 2 <= nodes.size()) return false;   // pre-vote failed: no term bump, no disruption
        elections++;
        candidate.elections++;
        candidate.term++;
        candidate.votedFor = candidate.id;
        int votes = 1;
        Set<Integer> grantedBy = new LinkedHashSet<>();
        for (Node808 peer : nodes) {
            if (peer == candidate || !candidate.reachableFrom(peer, links)) continue;
            if (pledge(peer, candidate) && peer.term <= candidate.term) {
                votes++;
                grantedBy.add(peer.id);
                peer.term = candidate.term;
                peer.leaseLocalStart = peer.localClock(tick);
                peer.votesGranted++;
            }
        }
        if (votes * 2 <= nodes.size()) return false;
        for (Node808 peer : nodes) if (peer.role.equals("leader") && peer != candidate) peer.role = "follower";
        candidate.role = "leader";
        candidate.leaseLocalStart = candidate.localClock(tick);
        for (int id : grantedBy) node(id).role = "follower";
        leases.add(new Lease(candidate.id, candidate.term, candidate.leaseLocalStart));
        return true;
    }

    /** One tick: renew leases, drop leaders that lost their quorum, then let timeouts campaign. */
    void step() {
        tick++;
        for (Node808 n : nodes) {
            if (n.isLeader() && !hasQuorum(n)) n.role = "follower";   // it stops renewing
        }
        Node808 current = leader();
        if (current != null) renew(current);
        for (Node808 n : nodes) {
            if (!n.alive || n.isLeader()) continue;
            if (n.localClock(tick) >= n.localElectionDeadline) {
                n.localElectionDeadline = n.localClock(tick) + ELECTION_TIMEOUT + n.id;
                campaign(n);
            }
        }
    }

    void run(int ticks) {
        for (int i = 0; i < ticks; i++) step();
    }

    void crash(int id) {
        node(id).alive = false;
        node(id).role = "dead";
    }

    void restart(int id) {
        node(id).alive = true;
        node(id).role = "follower";
        node(id).localElectionDeadline = node(id).localClock(tick) + ELECTION_TIMEOUT + id;
    }

    void partition(Set<Integer> side) {
        for (Node808 n : nodes) n.partitioned = side.contains(n.id);
    }

    void heal() {
        for (Node808 n : nodes) n.partitioned = false;
    }

    /** Safety: no two leaders' real-time lease intervals may overlap, ever. */
    String firstOverlap() {
        for (int i = 0; i < leases.size(); i++) {
            for (int j = i + 1; j < leases.size(); j++) {
                Lease a = leases.get(i), b = leases.get(j);
                if (a.leader == b.leader) continue;
                if (a.overlaps(b)) return a.render() + " overlaps " + b.render();
            }
        }
        return null;
    }

    long lastLeaseEndBy(int leaderId) {
        long end = Long.MIN_VALUE;
        for (Lease l : leases) if (l.leader == leaderId) end = Math.max(end, l.realLatestEnd());
        return end;
    }
}

public class Solution {

    static final Map<String, Long> METRICS = new TreeMap<>();

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static void metric(String name, long value) { METRICS.merge(name, value, Long::sum); }

    static String stateDump(Cluster808 c) {
        StringBuilder sb = new StringBuilder("tick=" + c.tick + " leases=" + c.leases.size());
        for (Node808 n : c.nodes) {
            sb.append(" | ").append(n.id).append(':').append(n.role).append('@').append(n.term)
              .append("/lease=").append(n.leaseLocalStart).append("/elections=").append(n.elections);
        }
        return sb.toString();
    }

    /** One leader at a time, renewed while it is healthy. */
    static void basicElectionAndRenewal() {
        Cluster808 c = new Cluster808(5);
        c.run(200);
        Node808 leader = c.leader();
        check(leader != null, "a five-node cluster elects a leader: " + stateDump(c));
        check(c.leaders().size() == 1, "exactly one leader: " + c.leaders().size());
        check(leader.term >= 1, "with a positive term: " + leader.term);
        check(leader.renewals > 0, "the lease was renewed while it was healthy: " + leader.renewals);
        Lease lease = c.leases.get(c.leases.size() - 1);
        check(lease.realLatestEnd() > lease.realEarliestStart(), "the lease covers real time");
        check(lease.localEnd >= lease.localStart + c.leaseLength,
                "a renewed lease reaches the lease length: " + (lease.localEnd - lease.localStart));
        check(c.firstOverlap() == null, "no two leases overlap: " + c.firstOverlap());
        for (Node808 n : c.nodes) {
            check(Math.abs(n.drift) <= Node808.DRIFT_BOUND, "the drift stays inside the bound");
        }
        metric("renewals", c.renewals);
    }

    /** Three failovers: a successor appears within the bound and leases never overlap. */
    static void failoverSafetyAndLatency() {
        Cluster808 c = new Cluster808(5);
        c.run(200);
        long worstLatency = 0;
        for (int round = 0; round < 3; round++) {
            Node808 leader = c.leader();
            check(leader != null, "round " + round + " starts with a leader");
            long crashTick = c.tick;
            c.crash(leader.id);
            check(c.leader() == null, "the crashed node is not a leader");
            long bound = c.cooldown() + 20;
            long electedAt = -1;
            for (long i = 0; i < bound; i++) {
                c.step();
                if (c.leader() != null) { electedAt = c.tick; break; }
            }
            check(electedAt > 0, "round " + round + " elected a successor within the bound");
            long latency = electedAt - crashTick;
            worstLatency = Math.max(worstLatency, latency);
            check(latency <= bound, "the failover took " + latency + " ticks, bound " + bound);
            check(c.leaders().size() == 1, "and exactly one leader emerged");
            check(c.firstOverlap() == null, "no lease overlap after failover " + round + ": "
                    + c.firstOverlap());
            c.restart(leader.id);
            c.run(20);
        }
        check(c.leases.size() >= 4, "four leases were issued over three failovers: " + c.leases.size());
        for (int i = 1; i < c.leases.size(); i++) {
            Lease prev = c.leases.get(i - 1), cur = c.leases.get(i);
            check(cur.realEarliestStart() > prev.realLatestEnd(),
                    "lease " + i + " starts after the previous lease's latest possible end");
            if (cur.leader == prev.leader) {
                check(cur.term > prev.term, "a leader that returns raised its term: "
                        + prev.term + " -> " + cur.term);
            }
        }
        metric("worstFailoverTicks", worstLatency);
        metric("leasesIssued", c.leases.size());
    }

    /** A partitioned leader stops renewing, and its successor may only start after the cooldown. */
    static void partitionedLeaderAndCooldown() {
        Cluster808 c = new Cluster808(5);
        c.run(200);
        Node808 old = c.leader();
        check(old != null, "a leader exists before the partition");
        long oldLeaseEnd = c.leases.get(c.leases.size() - 1).realLatestEnd();
        c.partition(new TreeSet<>(Arrays.asList(old.id, old.id == 5 ? 4 : 5)));
        int renewalsBefore = old.renewals;
        c.run(6);
        check(old.renewals == renewalsBefore, "the isolated leader no longer renews: " + old.renewals);
        check(!c.hasQuorum(old), "it cannot reach a quorum");
        c.run(60);
        Node808 fresh = c.leader();
        check(fresh != null && fresh != old, "the majority elected a successor: " + (fresh == null ? "none" : fresh.id));
        check(c.leaders().size() == 1, "the stale leader was demoted: " + c.leaders().size());
        check(c.firstOverlap() == null, "the successor's lease starts after the old one's real end: "
                + c.firstOverlap());
        Lease successor = c.leases.get(c.leases.size() - 1);
        check(successor.realEarliestStart() >= oldLeaseEnd,
                "the cooldown pushed the new lease past the old one: " + successor.realEarliestStart()
                        + " vs " + oldLeaseEnd);
        c.heal();
        c.run(60);
        check(c.leaders().size() == 1, "after healing there is one leader again: " + c.leaders().size());
        Node808 after = c.leader();
        check(after != null, "with a leader");
        int renewalsNow = after.renewals;
        c.run(12);
        check(c.leader() != after || after.renewals > renewalsNow,
                "the leader is renewing its lease or has been replaced");
        check(c.firstOverlap() == null, "and no lease ever overlapped: " + c.firstOverlap());
        if (after.id != fresh.id) {
            Lease last = c.leases.get(c.leases.size() - 1);
            check(last.realEarliestStart() > successor.realLatestEnd(),
                    "a fresh lease still starts after the previous one's latest possible end");
        } else {
            check(after.term >= fresh.term, "the surviving leader kept its term: " + after.term);
        }
        check(c.firstOverlap() == null, "and no lease overlapped: " + c.firstOverlap());
        metric("cooldownTicks", c.cooldown());
    }

    /** A minority can neither elect a leader nor renew one. */
    static void minorityCannotElect() {
        Cluster808 c = new Cluster808(5);
        c.run(200);
        Node808 leader = c.leader();
        check(leader != null, "a leader exists");
        int minorityId = leader.id == 1 ? 2 : 1;
        c.partition(new TreeSet<>(Arrays.asList(minorityId, minorityId == 2 ? 3 : 2)));
        int electionsBefore = c.node(minorityId).preVotes;
        int termBefore = c.node(minorityId).term;
        c.run(80);
        check(c.node(minorityId).preVotes > electionsBefore, "the isolated node keeps trying to campaign: "
                + c.node(minorityId).preVotes);
        check(c.node(minorityId).term == termBefore,
                "but its pre-vote fails, so it never bumps the term and cannot depose the leader: "
                        + c.node(minorityId).term);
        check(!c.node(minorityId).isLeader(), "but never wins with two of five nodes: " + c.node(minorityId).role);
        check(c.leader() != null && c.leader().id != minorityId, "the majority keeps its own leader");
        int renewals = c.leader().renewals;
        c.run(10);
        check(c.leader().renewals > renewals, "and that leader keeps renewing");
        c.heal();
        c.run(40);
        check(c.leaders().size() == 1, "after healing the cluster has a single leader");
        check(c.firstOverlap() == null, "with no lease overlap: " + c.firstOverlap());
        metric("minorityPreVotes", c.node(minorityId).preVotes);
    }

    /** A superseded node cannot come back with a stale lease. */
    static void staleLeaderCannotReturn() {
        Cluster808 c = new Cluster808(3);
        c.run(200);
        Node808 first = c.leader();
        check(first != null, "the first leader exists");
        int firstTerm = first.term;
        c.crash(first.id);
        c.run(80);
        Node808 second = c.leader();
        check(second != null && second.id != first.id, "a second leader took over");
        check(second.term > firstTerm, "in a higher term: " + second.term);
        c.restart(first.id);
        c.run(80);
        Node808 current = c.leader();
        check(current != null && c.leaders().size() == 1, "exactly one leader after the return");
        check(current.term >= second.term, "leadership did not go backwards: " + current.term);
        check(c.firstOverlap() == null, "no overlap when the old leader returns: " + c.firstOverlap());
        check(first.term >= firstTerm, "the returning node's term only grew");
        for (Lease l : c.leases) {
            check(l.leader != first.id || l.term >= firstTerm, "a stale lease was never reissued");
        }
        metric("staleReturns", 1);
    }

    /** Negative control: ignoring the cooldown lets a successor's lease start inside the old one. */
    static void negativeControlIgnoredCooldown() {
        Cluster808 good = new Cluster808(3);
        good.run(200);
        Node808 isolated = good.leader();
        good.partition(new TreeSet<>(Arrays.asList(isolated.id)));
        good.run(120);
        Node808 successor = good.leader();
        check(successor != null && successor.id != isolated.id,
                "with the cooldown honoured the majority waits it out and elects a successor");
        check(good.firstOverlap() == null, "and no lease ever overlapped: " + good.firstOverlap());
        check(good.leases.size() >= 2, "a second lease was issued: " + good.leases.size());
        Lease second = good.leases.get(good.leases.size() - 1);
        Lease first = good.leases.get(good.leases.size() - 2);
        check(second.realEarliestStart() > first.realLatestEnd(),
                "the successor's lease starts after the old lease's latest possible end: "
                        + second.realEarliestStart() + " vs " + first.realLatestEnd());

        Cluster808 bad = new Cluster808(3);
        bad.leakyLeases = true;                          // the voters stopped checking the cooldown
        bad.run(30);
        Node808 stale = bad.leader();
        bad.partition(new TreeSet<>(Arrays.asList(stale.id)));
        bad.run(40);
        String overlap = bad.firstOverlap();
        check(overlap != null, "ignoring the cooldown produces the overlapping leases the checker "
                + "looks for: " + overlap);
        check(bad.leases.size() >= 2, "two leases were issued: " + bad.leases.size());
        metric("negativeControlOverlaps", 1);

        Cluster808 single = new Cluster808(1);
        single.run(20);
        check(single.leader() != null, "a single node elects itself");
        check(single.leader().id == 1, "and it is node one");
        check(single.firstOverlap() == null, "there is nothing to overlap with");
        metric("singleNodeRenewals", single.leader().renewals);
    }

    public static void main(String[] args) {
        basicElectionAndRenewal();
        failoverSafetyAndLatency();
        partitionedLeaderAndCooldown();
        minorityCannotElect();
        staleLeaderCannotReturn();
        negativeControlIgnoredCooldown();
        StringBuilder sb = new StringBuilder("metrics:");
        for (Map.Entry<String, Long> e : METRICS.entrySet()) {
            sb.append(' ').append(e.getKey()).append('=').append(e.getValue());
        }
        System.out.println(sb);
        System.out.println("PASS");
    }
}
