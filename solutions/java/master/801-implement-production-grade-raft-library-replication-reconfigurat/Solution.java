/** Task 801 - Implement production-grade Raft library (replication, reconfiguration, snapshots). (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A deterministic, in-process Raft: leader election with randomized (splitmix64) timeouts, log
 * replication honouring the leader-completeness restriction (only a current-term entry may advance
 * the commit index), the no-op entry appended on election (Raft 8.3 - without it entries from an
 * older term could never be committed), log compaction into snapshots with InstallSnapshot
 * catch-up, and single-server membership changes driven through joint consensus (C_old,new).
 *
 * Simulation model, and what is simplified. There are no threads, no sockets and no wall clock: a
 * cluster is a tick loop, messages travel over an explicit in-memory network with a fixed one-tick
 * latency and are dropped when a partition separates sender and receiver, a crashed node simply
 * stops ticking, and election timeouts are counted in ticks. A snapshot is a whole state machine
 * (a small sorted map) rather than a serialized byte frame, and a reconfiguration is a log entry
 * carrying the voter sets as text. Servers that are about to be added are started with the
 * post-change voter set already known, which in real Raft they would learn from the leader. What is
 * *not* simplified is the protocol: quorums are computed against the live configuration
 * (majorities in both halves while joint), a leader accepts at most one uncommitted configuration
 * entry, and a configuration entry takes effect the moment it is appended, exactly as in the paper.
 *
 * Safety is observed after *every* tick by an observer that holds no protocol state: (a) election
 * safety - at most one leader per term; (b) no committed entry is ever replaced - an entry sitting
 * at a committed index whose term is at least the term that committed it must be that exact entry
 * (older-term leftovers of a partitioned leader are uncommitted and are legitimately truncated, so
 * they are allowed to linger until the next AppendEntries); (c) log matching - two nodes whose
 * entries differ at an index must disagree on that index's term; (d) applied prefixes only grow, all
 * nodes agree index by index, and the final state machines equal an oracle the test computes from
 * the commands it proposed; (e) a commit is announced with one value per index, ever. Liveness is
 * checked too: a leader is elected in every live majority, and a lagging server catches up by
 * snapshot. The negative control swaps in a commit rule without the majority requirement and the
 * observer must catch the divergence, which is what proves the checker is not vacuous.
 *
 * Verified: single-node cluster; election and replication of a known command sequence against the
 * oracle machine; compaction firing and preserving the commit prefix; a follower that slept through
 * a compaction repaired with InstallSnapshot and converging; joint-consensus growth 3 -> 5 and
 * shrinkage 5 -> 3 including the one-uncommitted-change-at-a-time rule; a partitioned minority that
 * can neither elect nor commit, a stale leader whose entries are replaced on healing, and quorum
 * safety when the majority moves on; plus the negative control above.
 */
import java.util.*;

/** splitmix64 - the only source of "randomness" (fixed seed, fully reproducible). */
final class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }
}

final class LogEntry {
    final int term;
    final String command;   // "key=value", or null for a configuration entry
    final String config;    // "j:a,b>c,d" (joint) or "c:a,b,c", or null for a command entry

    LogEntry(int term, String command, String config) {
        this.term = term;
        this.command = command;
        this.config = config;
    }

    boolean isConfig() { return config != null; }

    String label() { return isConfig() ? "cfg[" + config + "]" : command; }

    boolean sameAs(LogEntry other) {
        return other != null && term == other.term && Objects.equals(command, other.command)
                && Objects.equals(config, other.config);
    }
}

final class Msg {
    final String kind;
    final int from, to;
    int term;
    int prevLogIndex, prevLogTerm, leaderCommit;
    int matchIndex;
    int snapshotIndex, snapshotTerm;
    int lastLogIndex, lastLogTerm;
    boolean flag;                       // voteGranted / success
    List<LogEntry> entries;
    Map<String, String> snapshot;
    int due;

    Msg(String kind, int from, int to) { this.kind = kind; this.from = from; this.to = to; }
}

/** Explicit in-memory network: fixed latency, partitions and crashes observed at delivery time. */
final class Network {
    final List<Msg> inFlight = new ArrayList<>();
    final boolean[] down;
    final int[] group;
    int latency = 1;
    int whenSent;
    int dropped;

    Network(int nodes) {
        down = new boolean[nodes + 1];
        group = new int[nodes + 1];
        Arrays.fill(group, 1);
    }

    boolean link(int a, int b) { return !down[a] && !down[b] && group[a] == group[b]; }

    void send(int from, int to, Msg m) {
        if (!link(from, to)) { dropped++; return; }
        m.due = whenSent + latency;
        inFlight.add(m);
    }

    List<Msg> advance(int now) {
        List<Msg> out = new ArrayList<>();
        for (Iterator<Msg> it = inFlight.iterator(); it.hasNext();) {
            Msg m = it.next();
            if (m.due <= now) {
                it.remove();
                if (link(m.from, m.to)) out.add(m);
                else dropped++;          // the partition appeared while the packet was in flight
            }
        }
        out.sort(Comparator.comparingInt((Msg m) -> m.due).thenComparingInt(m -> m.from)
                .thenComparingInt(m -> m.to).thenComparing(m -> m.kind));
        return out;
    }
}

final class RaftNode {
    static final int HEARTBEAT_TICKS = 4;
    static final int ELECTION_MIN = 12;
    static final int ELECTION_JITTER = 12;

    final int id;
    final SplitMix64 rng;
    RaftCluster cluster;

    String role = "follower";
    int currentTerm;
    int votedFor = -1;

    final List<LogEntry> log = new ArrayList<>();   // log.get(i) holds index baseIndex + 1 + i
    int baseIndex;                                  // index covered by the local snapshot
    int snapshotTerm;
    TreeMap<String, String> snapshotState = new TreeMap<>();

    int commitIndex, lastApplied;
    final TreeMap<String, String> machine = new TreeMap<>();
    final TreeMap<Integer, String> applied = new TreeMap<>();

    int electionDeadline;
    int lastHeartbeatSent = -100000;
    final TreeMap<Integer, Integer> nextIndex = new TreeMap<>();
    final TreeMap<Integer, Integer> matchIndex = new TreeMap<>();
    final TreeSet<Integer> granted = new TreeSet<>();

    TreeSet<Integer> voters = new TreeSet<>();
    TreeSet<Integer> newVoters;                     // non-null only while the configuration is joint

    int elections, compactions, snapshotsInstalled;

    RaftNode(int id, SplitMix64 rng, Collection<Integer> config) {
        this.id = id;
        this.rng = rng;
        this.voters.addAll(config);
        matchIndex.put(id, 0);
    }

    int lastIndex() { return baseIndex + log.size(); }

    LogEntry entryAt(int index) {
        if (index <= baseIndex || index > lastIndex()) return null;
        return log.get(index - baseIndex - 1);
    }

    int termAt(int index) {
        if (index == baseIndex) return snapshotTerm;
        LogEntry e = entryAt(index);
        return e == null ? 0 : e.term;
    }

    TreeSet<Integer> members() {
        TreeSet<Integer> out = new TreeSet<>(voters);
        if (newVoters != null) out.addAll(newVoters);
        return out;
    }

    boolean isVoter(int who) {
        return voters.contains(who) || (newVoters != null && newVoters.contains(who));
    }

    static TreeSet<Integer> parseSet(String text) {
        TreeSet<Integer> out = new TreeSet<>();
        for (String part : text.split(",")) {
            String p = part.trim();
            if (!p.isEmpty()) out.add(Integer.parseInt(p));
        }
        return out;
    }

    /** A configuration entry takes effect the moment it is appended (Raft 6). */
    void applyConfig(String text) {
        if (text.startsWith("j:")) {
            String[] halves = text.substring(2).split(">");
            voters = parseSet(halves[0]);
            newVoters = parseSet(halves[1]);
        } else {
            voters = parseSet(text.substring(2));
            newVoters = null;
        }
    }

    /** After any log change the newest configuration entry present wins. */
    void refreshConfig() {
        for (int i = lastIndex(); i > baseIndex; i--) {
            LogEntry e = entryAt(i);
            if (e != null && e.isConfig()) { applyConfig(e.config); return; }
        }
    }

    static boolean majority(Set<Integer> config, Collection<Integer> acks) {
        int n = 0;
        for (int m : acks) if (config.contains(m)) n++;
        return n * 2 > config.size();
    }

    boolean hasQuorum(Collection<Integer> acks) {
        return majority(voters, acks) && (newVoters == null || majority(newVoters, acks));
    }

    void resetTimer(int now) {
        electionDeadline = now + ELECTION_MIN + rng.nextInt(ELECTION_JITTER);
    }

    void step(Network net, int now) {
        if (role.equals("leader")) {
            if (now - lastHeartbeatSent >= HEARTBEAT_TICKS) broadcastAppend(net, now);
        } else if (now >= electionDeadline) {
            startElection(net, now);
        }
    }

    void startElection(Network net, int now) {
        elections++;
        currentTerm++;
        role = "candidate";
        votedFor = id;
        granted.clear();
        if (isVoter(id)) granted.add(id);
        resetTimer(now);
        if (hasQuorum(granted)) { becomeLeader(net, now); return; }
        for (int m : members()) {
            if (m == id) continue;
            Msg m2 = new Msg("RequestVote", id, m);
            m2.term = currentTerm;
            m2.lastLogIndex = lastIndex();
            m2.lastLogTerm = termAt(lastIndex());
            net.send(id, m, m2);
        }
    }

    void becomeLeader(Network net, int now) {
        role = "leader";
        nextIndex.clear();
        matchIndex.clear();
        matchIndex.put(id, lastIndex());
        for (int m : members()) nextIndex.put(m, lastIndex() + 1);
        lastHeartbeatSent = now;
        log.add(new LogEntry(currentTerm, "noop", null));   // see the class comment
        matchIndex.put(id, lastIndex());
        advanceCommit();
        broadcastAppend(net, now);
    }

    void broadcastAppend(Network net, int now) {
        lastHeartbeatSent = now;
        for (int m : members()) if (m != id) sendAppend(net, m);
    }

    void sendAppend(Network net, int target) {
        int next = nextIndex.getOrDefault(target, lastIndex() + 1);
        if (next <= baseIndex) {
            Msg m = new Msg("InstallSnapshot", id, target);
            m.term = currentTerm;
            m.snapshotIndex = baseIndex;
            m.snapshotTerm = snapshotTerm;
            m.snapshot = new TreeMap<>(snapshotState);   // the state machine *at* baseIndex
            net.send(id, target, m);
            return;
        }
        Msg m = new Msg("AppendEntries", id, target);
        m.term = currentTerm;
        m.prevLogIndex = next - 1;
        m.prevLogTerm = termAt(next - 1);
        m.leaderCommit = commitIndex;
        m.entries = new ArrayList<>();
        for (int i = next; i <= lastIndex(); i++) m.entries.add(entryAt(i));
        net.send(id, target, m);
    }

    void handle(Network net, Msg m, int now) {
        if (m.term > currentTerm) becomeFollower(m.term);
        switch (m.kind) {
            case "RequestVote":      onRequestVote(net, m, now); break;
            case "RequestVoteReply": onVoteReply(net, m, now); break;
            case "AppendEntries":    onAppend(net, m, now); break;
            case "AppendReply":      onAppendReply(net, m, now); break;
            case "InstallSnapshot":  onSnapshot(net, m, now); break;
            case "SnapshotReply":    onSnapshotReply(m); break;
            default: throw new IllegalStateException("unknown message " + m.kind);
        }
    }

    void becomeFollower(int term) {
        if (term > currentTerm) { currentTerm = term; votedFor = -1; }
        role = "follower";
        granted.clear();
    }

    void onRequestVote(Network net, Msg m, int now) {
        int lastTerm = termAt(lastIndex());
        boolean upToDate = m.lastLogTerm > lastTerm
                || (m.lastLogTerm == lastTerm && m.lastLogIndex >= lastIndex());
        boolean grant = m.term >= currentTerm && upToDate && (votedFor == -1 || votedFor == m.from);
        if (grant) { votedFor = m.from; resetTimer(now); }
        Msg reply = new Msg("RequestVoteReply", id, m.from);
        reply.term = currentTerm;
        reply.flag = grant;
        net.send(id, m.from, reply);
    }

    void onVoteReply(Network net, Msg m, int now) {
        if (!role.equals("candidate") || m.term != currentTerm || !m.flag) return;
        granted.add(m.from);
        if (hasQuorum(granted)) becomeLeader(net, now);
    }

    void onAppend(Network net, Msg m, int now) {
        if (m.term < currentTerm) {
            Msg reply = new Msg("AppendReply", id, m.from);
            reply.term = currentTerm;
            net.send(id, m.from, reply);
            return;
        }
        role = "follower";
        currentTerm = m.term;
        resetTimer(now);
        boolean ok = m.prevLogIndex <= lastIndex() && m.prevLogIndex >= baseIndex
                && termAt(m.prevLogIndex) == m.prevLogTerm;
        if (ok) {
            int idx = m.prevLogIndex;
            for (LogEntry e : m.entries) {
                idx++;
                LogEntry local = entryAt(idx);
                if (local != null && local.sameAs(e)) continue;
                if (local != null) {
                    log.subList(idx - baseIndex - 1, log.size()).clear();
                    refreshConfig();
                }
                log.add(e);
            }
            refreshConfig();
            if (m.leaderCommit > commitIndex) {
                commitIndex = Math.min(m.leaderCommit, lastIndex());
                apply();
                maybeCompact();
            }
        }
        Msg reply = new Msg("AppendReply", id, m.from);
        reply.term = currentTerm;
        reply.flag = ok;
        reply.matchIndex = ok ? m.prevLogIndex + m.entries.size() : 0;
        net.send(id, m.from, reply);
    }

    void onAppendReply(Network net, Msg m, int now) {
        if (!role.equals("leader") || m.term != currentTerm) return;
        if (m.flag) {
            matchIndex.put(m.from, Math.max(matchIndex.getOrDefault(m.from, 0), m.matchIndex));
            nextIndex.put(m.from, matchIndex.get(m.from) + 1);
            advanceCommit();
        } else {
            int next = Math.max(baseIndex + 1, nextIndex.getOrDefault(m.from, lastIndex() + 1) - 1);
            nextIndex.put(m.from, next);
            sendAppend(net, m.from);
        }
    }

    void onSnapshot(Network net, Msg m, int now) {
        if (m.term < currentTerm) {
            Msg reply = new Msg("SnapshotReply", id, m.from);
            reply.term = currentTerm;
            net.send(id, m.from, reply);
            return;
        }
        role = "follower";
        currentTerm = m.term;
        resetTimer(now);
        machine.clear();
        machine.putAll(m.snapshot);
        snapshotState = new TreeMap<>(m.snapshot);
        baseIndex = m.snapshotIndex;
        snapshotTerm = m.snapshotTerm;
        log.clear();
        commitIndex = baseIndex;
        lastApplied = baseIndex;
        applied.tailMap(baseIndex + 1).clear();   // defensive: nothing beyond the snapshot may linger
        snapshotsInstalled++;
        Msg reply = new Msg("SnapshotReply", id, m.from);
        reply.term = currentTerm;
        reply.flag = true;
        reply.matchIndex = baseIndex;
        net.send(id, m.from, reply);
    }

    void onSnapshotReply(Msg m) {
        if (!role.equals("leader") || m.term != currentTerm || !m.flag) return;
        matchIndex.put(m.from, Math.max(matchIndex.getOrDefault(m.from, 0), m.matchIndex));
        nextIndex.put(m.from, matchIndex.get(m.from) + 1);
        advanceCommit();
    }

    /** Commit rule: only a current-term entry replicated on a quorum may advance the commit index. */
    void advanceCommit() {
        for (int n = lastIndex(); n > commitIndex; n--) {
            LogEntry e = entryAt(n);
            if (e.term != currentTerm) continue;
            TreeSet<Integer> acked = new TreeSet<>();
            acked.add(id);
            if (cluster.buggyCommit) {
                boolean oneFollower = false;
                for (int mm : members()) {
                    if (mm != id && matchIndex.getOrDefault(mm, 0) >= n) oneFollower = true;
                }
                if (!oneFollower) continue;
            } else {
                for (int mm : members()) if (matchIndex.getOrDefault(mm, 0) >= n) acked.add(mm);
                if (!hasQuorum(acked)) continue;
            }
            for (int i = commitIndex + 1; i <= n; i++) {
                commitIndex = i;
                cluster.recordCommit(i, entryAt(i));
            }
            apply();
            maybeCompact();
            return;
        }
    }

    void apply() {
        while (lastApplied < commitIndex) {
            int i = ++lastApplied;
            LogEntry e = entryAt(i);
            applied.put(i, e.label());
            if (!e.isConfig() && !e.command.equals("noop")) {
                int eq = e.command.indexOf('=');
                machine.put(e.command.substring(0, eq), e.command.substring(eq + 1));
            }
        }
    }

    void maybeCompact() {
        if (commitIndex - baseIndex < RaftCluster.SNAPSHOT_THRESHOLD) return;
        snapshotTerm = termAt(commitIndex);
        snapshotState = new TreeMap<>(machine);         // machine is applied exactly up to commitIndex
        log.subList(0, commitIndex - baseIndex).clear();
        baseIndex = commitIndex;
        compactions++;
        refreshConfig();
    }

    boolean appendCommand(String command) {
        if (!role.equals("leader")) return false;
        log.add(new LogEntry(currentTerm, command, null));
        matchIndex.put(id, lastIndex());
        advanceCommit();                    // a lone server is its own quorum
        return true;
    }

    boolean hasUncommittedConfig() {
        for (int i = commitIndex + 1; i <= lastIndex(); i++) {
            LogEntry e = entryAt(i);
            if (e != null && e.isConfig()) return true;
        }
        return false;
    }

    boolean appendConfig(String configText) {
        if (!role.equals("leader")) return false;
        if (hasUncommittedConfig()) return false;        // one uncommitted change at a time
        log.add(new LogEntry(currentTerm, null, configText));
        matchIndex.put(id, lastIndex());
        applyConfig(configText);
        advanceCommit();
        return true;
    }
}

final class RaftCluster {
    static final int SNAPSHOT_THRESHOLD = 10;

    final Network net;
    final RaftNode[] nodes;
    final SplitMix64 rng;
    int tick;
    boolean buggyCommit;                             // negative control only
    boolean expectViolations;                        // set only by the negative control

    final TreeMap<Integer, Integer> leaderByTerm = new TreeMap<>();
    final TreeMap<Integer, String> committed = new TreeMap<>();
    final TreeMap<Integer, Integer> committedTerm = new TreeMap<>();
    final TreeMap<Integer, String> appliedEver = new TreeMap<>();
    final List<String> commitOrder = new ArrayList<>();
    final List<String> violations = new ArrayList<>();
    int[] lastAppliedSeen;
    int divergence, leadersElected;

    RaftCluster(int n, long seed, Integer... initialConfig) {
        rng = new SplitMix64(seed);
        net = new Network(n);
        nodes = new RaftNode[n + 1];
        lastAppliedSeen = new int[n + 1];
        List<Integer> cfg = Arrays.asList(initialConfig);
        for (int i = 1; i <= n; i++) {
            nodes[i] = new RaftNode(i, rng, cfg);
            nodes[i].cluster = this;
            nodes[i].resetTimer(0);
        }
    }

    int size() { return nodes.length - 1; }

    boolean up(int id) { return !net.down[id]; }

    RaftNode node(int id) { return nodes[id]; }

    void crash(int id) { net.down[id] = true; }

    /** A rebooted server keeps term, vote and log but comes back as a follower with a fresh timer. */
    void restart(int id) {
        net.down[id] = false;
        nodes[id].role = "follower";
        nodes[id].granted.clear();
        nodes[id].resetTimer(tick);
    }

    void setInitialConfig(int id, Integer... cfg) {
        nodes[id].voters = RaftNode.parseSet(join(cfg));
        nodes[id].newVoters = null;
    }

    static String join(Integer... ids) {
        StringBuilder sb = new StringBuilder();
        for (Integer i : ids) {
            if (sb.length() > 0) sb.append(',');
            sb.append(i);
        }
        return sb.toString();
    }

    /** Cuts the given nodes off from everyone else; the rest stay fully connected. */
    void partition(Set<Integer> side) {
        for (int i = 1; i <= size(); i++) net.group[i] = side.contains(i) ? 2 : 1;
    }

    void heal() {
        for (int i = 1; i <= size(); i++) net.group[i] = 1;
    }

    /** The reachable leader with the highest term - the one a client would be routed to. */
    int leaderId() {
        int best = -1, bestTerm = -1;
        for (int i = 1; i <= size(); i++) {
            if (up(i) && nodes[i].role.equals("leader") && nodes[i].currentTerm > bestTerm) {
                best = i;
                bestTerm = nodes[i].currentTerm;
            }
        }
        return best;
    }

    boolean propose(String command) {
        int leader = leaderId();
        return leader > 0 && proposeTo(leader, command);
    }

    boolean proposeTo(int id, String command) {
        RaftNode leader = nodes[id];
        if (!leader.role.equals("leader")) return false;
        boolean ok = leader.appendCommand(command);
        if (ok) for (int m : leader.members()) if (m != id) leader.sendAppend(net, m);
        return ok;
    }

    boolean proposeConfig(String configText) {
        int leader = leaderId();
        if (leader < 0) return false;
        boolean ok = nodes[leader].appendConfig(configText);
        if (ok) for (int m : nodes[leader].members()) if (m != leader) nodes[leader].sendAppend(net, m);
        return ok;
    }

    void recordCommit(int index, LogEntry e) {
        String label = e.label();
        String prev = committed.putIfAbsent(index, label);
        if (prev == null) {
            commitOrder.add(label);
            committedTerm.put(index, e.term);
        } else if (!prev.equals(label)) {
            divergence++;
            violation("committed index " + index + " changed: " + prev + " -> " + label);
        }
    }

    void run(int ticks) {
        for (int t = 0; t < ticks; t++) {
            tick++;
            net.whenSent = tick;
            for (Msg m : net.advance(tick)) nodes[m.to].handle(net, m, tick);
            for (int i = 1; i <= size(); i++) if (up(i)) nodes[i].step(net, tick);
            observe();
        }
    }

    /** The independent observer: it only reads node state, it never drives the protocol. */
    void observe() {
        for (int i = 1; i <= size(); i++) {
            if (!up(i) || !nodes[i].role.equals("leader")) continue;
            Integer prev = leaderByTerm.putIfAbsent(nodes[i].currentTerm, i);
            if (prev == null) leadersElected++;
            else if (prev != i) violation("two leaders in term " + nodes[i].currentTerm + ": " + prev + ", " + i);
        }
        for (int i = 1; i <= size(); i++) {
            if (!up(i)) continue;
            RaftNode r = nodes[i];
            if (r.lastApplied < lastAppliedSeen[i]) {
                violation("node " + i + " un-applied index " + lastAppliedSeen[i]);
            }
            lastAppliedSeen[i] = Math.max(lastAppliedSeen[i], r.lastApplied);
            // An entry at a committed index whose term is at least the term that committed it must
            // *be* that entry: only a leader elected at or after the commit could have appended it,
            // and by leader completeness such a leader holds the committed value. An older-term
            // entry at that index is an uncommitted leftover of a partitioned leader, which the
            // next AppendEntries truncates - so it is not a violation while it lingers.
            for (Map.Entry<Integer, Integer> e : committedTerm.entrySet()) {
                LogEntry le = r.entryAt(e.getKey());
                if (le != null && le.term >= e.getValue()
                        && !le.label().equals(committed.get(e.getKey()))) {
                    violation("node " + i + " replaced committed index " + e.getKey() + ": "
                            + le.label() + "@" + le.term + " != " + committed.get(e.getKey())
                            + "@" + e.getValue());
                }
            }
            for (Map.Entry<Integer, String> e : r.applied.entrySet()) {
                String seen = appliedEver.putIfAbsent(e.getKey(), e.getValue());
                if (seen != null && !seen.equals(e.getValue())) {
                    violation("applied index " + e.getKey() + " changed from " + seen
                            + " to " + e.getValue());
                }
                String c = committed.get(e.getKey());
                if (c != null && !c.equals(e.getValue())) {
                    violation("node " + i + " applied " + e.getValue() + " at " + e.getKey()
                            + " but " + c + " was committed");
                }
            }
        }
        for (int a = 1; a <= size(); a++) {
            for (int b = a + 1; b <= size(); b++) {
                RaftNode x = nodes[a], y = nodes[b];
                int lo = Math.max(x.baseIndex, y.baseIndex) + 1;
                int hi = Math.min(x.lastIndex(), y.lastIndex());
                for (int i = lo; i <= hi; i++) {
                    if (!x.entryAt(i).sameAs(y.entryAt(i)) && x.termAt(i) == y.termAt(i)) {
                        violation("log matching broken at index " + i + " between " + a + " and " + b);
                    }
                }
                for (Map.Entry<Integer, String> e : x.applied.entrySet()) {
                    String other = y.applied.get(e.getKey());
                    if (other != null && !other.equals(e.getValue())) {
                        violation("applied index " + e.getKey() + " differs: " + e.getValue()
                                + " vs " + other);
                    }
                }
            }
        }
        for (int i = 1; i <= size(); i++) {
            if (!up(i)) continue;
            for (Map.Entry<Integer, String> e : nodes[i].applied.entrySet()) {
                String c = committed.get(e.getKey());
                if (c != null && !c.equals(e.getValue())) {
                    violation("node " + i + " applied " + e.getValue() + " at " + e.getKey()
                            + " but " + c + " was committed");
                }
            }
        }
        if (violations.size() > 25) throw new AssertionError("observer: " + violations.get(0));
    }

    /** Records one violation, stopping a cluster that is meant to be correct before the list grows. */
    void violation(String text) {
        if (violations.size() >= 25) return;
        violations.add(text);
        if (!expectViolations && violations.size() >= 25) {
            throw new AssertionError("observer: " + violations.get(0));
        }
    }

    static String render(RaftNode r) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String> e : r.machine.entrySet()) {
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

    static void checkQuiet(RaftCluster c, String what) {
        check(c.violations.isEmpty(), what + ": " + c.violations);
    }

    /** Election plus replication of a known command sequence, cross-checked against an oracle. */
    static void electionAndReplication() {
        RaftCluster c = new RaftCluster(3, 801001L, 1, 2, 3);
        c.run(60);
        check(c.leaderId() > 0, "a 3-node cluster elects a leader");
        check(c.leaderByTerm.size() == 1, "exactly one term produced a leader: " + c.leaderByTerm);
        check(c.leadersElected == 1, "exactly one leader was ever observed");
        checkQuiet(c, "no violations while electing");

        for (String cmd : new String[] {"a=1", "b=2", "c=3", "a=4", "d=5"}) {
            check(c.propose(cmd), "the leader accepts " + cmd);
        }
        c.run(40);
        String oracle = "a=4,b=2,c=3,d=5";                 // b,c,d once; a overwritten by a=4
        for (int i = 1; i <= 3; i++) {
            check(RaftCluster.render(c.node(i)).equals(oracle),
                    "node " + i + " machine is " + RaftCluster.render(c.node(i)) + ", oracle " + oracle);
        }
        check(c.commitOrder.equals(Arrays.asList("noop", "a=1", "b=2", "c=3", "a=4", "d=5")),
                "commit order is the proposal order: " + c.commitOrder);
        checkQuiet(c, "safety held under replication");

        c.crash(3);
        c.run(40);
        check(c.leaderId() > 0 && c.leaderId() != 3, "the surviving majority elected a new leader");
        for (int k = 0; k < 6; k++) { check(c.propose("k" + k + "=" + k), "proposal k" + k); c.run(6); }
        int before = c.node(3).lastApplied;
        c.restart(3);
        c.run(80);
        check(c.node(3).lastApplied > before, "the restarted follower made progress");
        check(RaftCluster.render(c.node(3)).equals(RaftCluster.render(c.node(1))),
                "the rejoined follower converges: " + RaftCluster.render(c.node(3)));
        checkQuiet(c, "no safety violation during catch-up");
        metric("commits", c.commitOrder.size());
        metric("elections", c.node(1).elections + c.node(2).elections + c.node(3).elections);
    }

    /** Compaction, snapshot installation and the durability of the committed prefix. */
    static void snapshotsAndCompaction() {
        RaftCluster c = new RaftCluster(3, 801002L, 1, 2, 3);
        c.run(40);
        for (int k = 0; k < 14; k++) { c.propose("s" + k + "=" + k); c.run(4); }
        c.run(30);
        int leader = c.leaderId();
        check(leader > 0, "the compacting cluster has a leader");
        check(c.node(leader).compactions > 0, "the leader compacted its log at least once");
        check(c.node(leader).baseIndex >= RaftCluster.SNAPSHOT_THRESHOLD,
                "the snapshot covers at least the threshold: " + c.node(leader).baseIndex);
        check(c.node(leader).commitIndex >= 14, "the commit index passed the compaction point");
        for (int k = 1; k <= 3; k++) {
            for (int j = 0; j < 14; j++) {
                check(c.node(k).machine.getOrDefault("s" + j, "-").equals(Integer.toString(j)),
                        "node " + k + " kept s" + j + " across compaction");
            }
        }
        check(c.divergence == 0, "no committed value was ever overwritten");

        // A follower that slept through the compaction must be repaired with a snapshot.
        RaftCluster d = new RaftCluster(3, 801003L, 1, 2, 3);
        d.run(40);
        for (int k = 0; k < 16; k++) { d.propose("t" + k + "=" + k); d.run(3); }
        d.crash(2);
        for (int k = 0; k < 16; k++) { d.propose("u" + k + "=" + k); d.run(3); }
        d.run(20);
        int dl = d.leaderId();
        check(dl > 0 && dl != 2, "the remaining majority kept a leader while node 2 was down");
        check(d.node(dl).baseIndex > 0, "the leader compacted while node 2 was down");
        d.restart(2);
        d.run(120);
        check(d.node(2).snapshotsInstalled > 0, "node 2 was repaired with an InstallSnapshot");
        check(RaftCluster.render(d.node(2)).equals(RaftCluster.render(d.node(dl))),
                "the repaired follower matches the leader exactly");
        check(d.commitOrder.contains("u15=15"), "entries committed while it was away survive");
        checkQuiet(d, "snapshot catch-up was safe");
        metric("compactions", c.node(leader).compactions + d.node(dl).compactions);
        metric("snapshots", d.node(2).snapshotsInstalled);
    }

    /** Joint consensus growth and shrinkage, including the one-change-at-a-time rule. */
    static void reconfiguration() {
        RaftCluster c = new RaftCluster(5, 801004L, 1, 2, 3);
        c.setInitialConfig(4, 1, 2, 3, 4, 5);
        c.setInitialConfig(5, 1, 2, 3, 4, 5);
        c.crash(4);
        c.crash(5);
        c.run(40);
        check(c.leaderId() > 0, "the original 3-node configuration elects a leader");
        for (int k = 0; k < 4; k++) { c.propose("p" + k + "=" + k); c.run(5); }

        check(c.proposeConfig("j:1,2,3>1,2,3,4,5"), "the joint configuration entry is appended");
        check(c.node(c.leaderId()).hasUncommittedConfig(), "the joint entry is not yet committed");
        check(!c.proposeConfig("c:1,2,3,4,5"),
                "a second configuration change is rejected while one is pending");
        c.run(30);
        check(!c.node(c.leaderId()).hasUncommittedConfig(), "the joint configuration committed");
        check(c.proposeConfig("c:1,2,3,4,5"), "the final configuration entry is appended");
        c.run(30);
        c.restart(4);
        c.restart(5);
        c.run(150);
        for (int i = 1; i <= 5; i++) {
            check(c.node(i).voters.equals(RaftNode.parseSet("1,2,3,4,5")),
                    "node " + i + " uses the grown configuration: " + c.node(i).voters);
            check(c.node(i).newVoters == null, "node " + i + " left joint consensus");
        }
        check(c.leaderId() > 0, "the grown configuration has a leader");
        for (int i = 1; i <= 5; i++) {
            check(c.node(i).machine.getOrDefault("p3", "-").equals("3"),
                    "node " + i + " retained pre-reconfiguration data");
        }
        checkQuiet(c, "growth was safe");

        check(c.proposeConfig("j:1,2,3,4,5>1,2,3"), "the shrink is proposed as a joint configuration");
        c.run(30);
        check(c.proposeConfig("c:1,2,3"), "the shrunk configuration entry is appended");
        c.run(40);
        c.crash(4);
        c.crash(5);
        c.run(80);
        for (int i = 1; i <= 3; i++) {
            check(c.node(i).voters.equals(RaftNode.parseSet("1,2,3")), "node " + i + " shrunk back");
        }
        check(c.leaderId() > 0, "the shrunk cluster still elects a leader");
        check(c.propose("z=9"), "and still accepts writes");
        c.run(30);
        for (int i = 1; i <= 3; i++) {
            check(c.node(i).machine.getOrDefault("z", "-").equals("9"),
                    "node " + i + " applied the post-shrink write");
        }
        checkQuiet(c, "shrinkage was safe");
        metric("configChanges", 4);
    }

    /** Partitions: a minority cannot commit, and healing replaces the stale leader's entries. */
    static void partitionsAndFailover() {
        RaftCluster c = new RaftCluster(5, 801005L, 1, 2, 3, 4, 5);
        c.run(60);
        int leader = c.leaderId();
        check(leader > 0, "a 5-node cluster elects a leader");
        check(c.propose("m=1"), "the leader accepts a write");
        c.run(20);
        check(c.node(leader).machine.containsKey("m"), "the write committed on the leader");

        int side = leader == 1 ? 2 : 1;
        TreeSet<Integer> minority = new TreeSet<>(Arrays.asList(leader, side));
        c.partition(minority);
        c.run(80);
        int before = c.commitOrder.size();
        for (int k = 0; k < 3; k++) { c.proposeTo(leader, "stale" + k + "=" + k); c.run(8); }
        check(c.commitOrder.size() == before,
                "the isolated leader could not commit anything: " + c.commitOrder);
        int majorityLeader = c.leaderId();
        check(majorityLeader > 0 && !minority.contains(majorityLeader),
                "the majority elected a leader of its own: " + majorityLeader);
        check(c.node(majorityLeader).machine.containsKey("m"), "it inherited the committed data");
        check(c.propose("m2=2"), "the majority accepts writes");
        c.run(20);
        check(c.commitOrder.contains("m2=2"), "and commits them");

        c.heal();
        c.run(150);
        check(c.commitOrder.stream().noneMatch(s -> s.startsWith("stale")),
                "the stale entries were never committed: " + c.commitOrder);
        String reference = RaftCluster.render(c.node(majorityLeader));
        for (int i = 1; i <= 5; i++) {
            check(RaftCluster.render(c.node(i)).equals(reference),
                    "node " + i + " converged to " + RaftCluster.render(c.node(i)) + ", expected " + reference);
        }
        check(c.node(leader).machine.containsKey("m2"), "the stale leader took over the majority's data");
        for (int i = 1; i <= c.node(leader).commitIndex; i++) {
            if (c.node(leader).entryAt(i) != null) {
                check(c.node(leader).entryAt(i).label().equals(c.committed.get(i)),
                        "the stale leader's index " + i + " was repaired to "
                                + c.node(leader).entryAt(i).label() + ", committed " + c.committed.get(i));
            }
        }
        checkQuiet(c, "no committed entry was lost across failover");
        metric("faults", 1);
    }

    /** No leader may be elected by a minority, however long it campaigns. */
    static void minorityCannotElect() {
        RaftCluster c = new RaftCluster(5, 801006L, 1, 2, 3, 4, 5);
        c.run(60);
        int splitTerm = 0;
        for (int i = 1; i <= 5; i++) splitTerm = Math.max(splitTerm, c.node(i).currentTerm);
        int leaderBefore = c.leaderId();
        check(leaderBefore > 0, "the healthy cluster has a leader");
        TreeSet<Integer> minority = new TreeSet<>(Arrays.asList(1, 2));
        boolean leaderCutOff = minority.contains(leaderBefore);
        int commitOne = c.node(1).commitIndex, commitTwo = c.node(2).commitIndex;
        c.partition(minority);
        c.run(200);
        Integer replacement = null;
        for (Map.Entry<Integer, Integer> e : c.leaderByTerm.entrySet()) {
            if (e.getKey() > splitTerm) {
                check(!minority.contains(e.getValue()),
                        "term " + e.getKey() + " was led by minority node " + e.getValue());
                replacement = e.getValue();
            }
        }
        if (leaderCutOff) {
            check(replacement != null, "the majority elected a replacement for the cut-off leader");
        } else {
            check(c.leaderId() > 0 && !minority.contains(c.leaderId()),
                    "the majority kept its unreachable-from-minority leader");
        }
        check(c.node(1).commitIndex == commitOne && c.node(2).commitIndex == commitTwo,
                "a minority can never advance its commit index");
        check(Math.max(c.node(1).currentTerm, c.node(2).currentTerm) >= splitTerm + 2,
                "the campaigning minority kept raising its term: " + c.node(1).currentTerm
                        + "/" + c.node(2).currentTerm);
        int commitsBefore = c.commitOrder.size();
        check(c.propose("mm=1"), "the majority accepts writes while the minority is away");
        c.run(30);
        check(c.commitOrder.size() > commitsBefore, "the majority committed without them");
        c.heal();
        c.run(150);
        check(c.leaderId() > 0, "the healed cluster has a leader again");
        check(c.commitOrder.contains("mm=1"), "the majority's writes survived the healing");
        for (int i = 1; i <= 5; i++) {
            check(c.node(i).machine.getOrDefault("mm", "-").equals("1"),
                    "node " + i + " converged after the minority rejoined");
        }
    }

    /** Negative control: committing with a single follower must be caught by the observer. */
    static void negativeControlNoMajority() {
        RaftCluster good = new RaftCluster(3, 801007L, 1, 2, 3);
        good.run(40);
        good.propose("x=1");
        good.run(20);
        check(good.divergence == 0, "the correct commit rule never diverges");

        RaftCluster bad = new RaftCluster(5, 801008L, 1, 2, 3, 4, 5);
        bad.run(60);
        int leader = bad.leaderId();
        check(leader > 0, "the cluster used by the negative control starts with a leader");
        bad.propose("base=0");
        bad.run(20);
        bad.buggyCommit = true;                       // drop the majority requirement
        bad.expectViolations = true;                  // the observer may now report freely
        int buddy = leader == 1 ? 2 : 1;
        TreeSet<Integer> cut = new TreeSet<>(Arrays.asList(leader, buddy));
        bad.partition(cut);
        bad.run(30);
        for (int k = 0; k < 3; k++) { bad.proposeTo(leader, "bad" + k + "=" + k); bad.run(6); }
        check(bad.commitOrder.stream().anyMatch(s -> s.startsWith("bad")),
                "the buggy leader committed inside the minority: " + bad.commitOrder);
        bad.buggyCommit = false;
        bad.run(40);
        int majorityLeader = bad.leaderId();
        check(majorityLeader > 0 && !cut.contains(majorityLeader),
                "once the rule is restored the majority elects its own leader");
        bad.propose("good0=0");
        bad.run(20);
        bad.heal();
        bad.run(150);
        check(bad.divergence > 0, "the observer caught the divergent commit: " + bad.divergence);
        check(!bad.violations.isEmpty(), "and recorded a concrete violation");
        metric("negativeControlDivergences", bad.divergence);
    }

    /** Degenerate cluster: one server is a quorum of one. */
    static void singleNode() {
        RaftCluster c = new RaftCluster(1, 801009L, 1);
        c.run(30);
        check(c.leaderId() == 1, "a single node elects itself immediately");
        check(c.node(1).elections == 1, "exactly one election happened: " + c.node(1).elections);
        check(c.propose("only=1"), "it accepts writes without peers");
        c.run(10);
        check(c.node(1).machine.getOrDefault("only", "-").equals("1"), "and applies them");
        check(c.node(1).commitIndex >= 2, "no-op plus write are committed: " + c.node(1).commitIndex);
        check(c.proposeConfig("c:1"), "a redundant configuration change is accepted");
        c.run(20);
        check(c.node(1).voters.equals(RaftNode.parseSet("1")), "the configuration stays {1}");
        check(!c.node(1).hasUncommittedConfig(), "and the change committed");
        checkQuiet(c, "no violations in the degenerate case");
    }

    public static void main(String[] args) {
        electionAndReplication();
        snapshotsAndCompaction();
        reconfiguration();
        partitionsAndFailover();
        minorityCannotElect();
        negativeControlNoMajority();
        singleNode();
        StringBuilder sb = new StringBuilder("metrics:");
        for (Map.Entry<String, Long> e : METRICS.entrySet()) {
            sb.append(' ').append(e.getKey()).append('=').append(e.getValue());
        }
        System.out.println(sb);
        System.out.println("PASS");
    }
}
