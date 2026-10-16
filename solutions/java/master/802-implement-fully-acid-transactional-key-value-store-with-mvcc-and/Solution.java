/** Task 802 - Implement fully ACID transactional key-value store with MVCC and 2PC. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * Four pieces make up the store. (1) MVCC: every key holds a column of versions stamped with the
 * commit timestamp that created them, so a transaction reads the newest version at or below its own
 * snapshot timestamp and never blocks a writer. (2) Snapshot isolation with first-committer-wins: a
 * committing transaction is rejected if any key it wrote gained a version after its snapshot.
 * (3) Backward validation of the read set (textbook OCC): a commit is also rejected if a key the
 * transaction *read* gained a version after its snapshot. Plain snapshot isolation permits write
 * skew - the doctors-on-call scenario is reproduced here - and read validation closes it: it aborts
 * more histories than serializable snapshot isolation would, but it never accepts a non-serializable
 * one. (4) Two-phase commit with a coordinator write-ahead log plus one journal per participant:
 * PREPARE records an intent, the coordinator's decision is written durably before any COMMIT/ABORT
 * frame leaves, and recovery re-derives every volatile map by replaying the journals and then
 * following the coordinator's decision.
 *
 * Simulation model, and what is simplified. No threads, no sockets, no wall clock: a "transaction"
 * is an object the test drives step by step, so interleavings are explicit and reproducible; the
 * network is an in-memory frame queue that loses frames when a link or a node is down; the timestamp
 * oracle is a counter; a "crash" drops the volatile state of a node while its journal survives; and
 * the store runs a handful of shards with text keys routed by floorMod(hashCode, shardCount). What
 * is not simplified is the commit protocol: the decision is logged before it is sent, participants
 * are idempotent, an undecided intent after a coordinator crash becomes an ABORT (presumed abort),
 * and a participant whose coordinator is unreachable stays blocked instead of guessing - the
 * blocking nature of 2PC is demonstrated rather than papered over.
 *
 * Verified: read-your-own-writes and no visibility before commit; snapshot scans equal an
 * independently replayed oracle at six timestamps, including time travel to a pre-delete snapshot;
 * tombstones; first-committer-wins rejects the loser of a write-write race and the loser leaves no
 * trace; forty overlapping increments of one counter end at exactly forty (serializable), while the
 * negative control with the conflict check disabled provably loses updates; write skew appears under
 * pure snapshot isolation and disappears under read validation; two-shard 2PC is all-or-nothing on
 * the happy path, with a dropped PREPARE, after a coordinator crash before the decision, after a
 * coordinator crash with the decision already logged, and after a participant crash whose journal
 * replays to the exact pre-crash state; and a full reboot preserves every committed value.
 */
import java.util.*;

final class Version {
    final long commitTs;
    final String value;                 // null marks a tombstone

    Version(long commitTs, String value) { this.commitTs = commitTs; this.value = value; }

    boolean tombstone() { return value == null; }
}

/** One shard: an MVCC table plus the participant half of 2PC (intents and a write-ahead journal). */
final class Shard {
    static final String DELETE = "~del~";

    final String name;
    final TreeMap<String, TreeMap<Long, Version>> data = new TreeMap<>();
    final TreeMap<Long, Map<String, String>> prepared = new TreeMap<>();
    final TreeSet<Long> applied = new TreeSet<>();
    final List<String> journal = new ArrayList<>();
    boolean down;
    int prepares, commitsMade, discards, dropped;

    Shard(String name) { this.name = name; }

    /** The newest committed version at or below `ts`; a tombstone hides older values. */
    String read(String key, long ts) {
        TreeMap<Long, Version> column = data.get(key);
        if (column == null) return null;
        Map.Entry<Long, Version> e = column.floorEntry(ts);
        return e == null || e.getValue().tombstone() ? null : e.getValue().value;
    }

    TreeMap<String, String> scan(long ts) {
        TreeMap<String, String> out = new TreeMap<>();
        for (String key : data.keySet()) {
            String value = read(key, ts);
            if (value != null) out.put(key, value);
        }
        return out;
    }

    boolean hasVersionAfter(String key, long ts) {
        TreeMap<Long, Version> column = data.get(key);
        return column != null && !column.isEmpty() && column.lastKey() > ts;
    }

    static String encode(Map<String, String> writes) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String> e : writes.entrySet()) {
            if (sb.length() > 0) sb.append(';');
            sb.append(e.getKey()).append('=').append(e.getValue() == null ? DELETE : e.getValue());
        }
        return sb.toString();
    }

    static Map<String, String> decode(String text) {
        Map<String, String> out = new LinkedHashMap<>();
        if (text.isEmpty()) return out;
        for (String pair : text.split(";")) {
            int eq = pair.indexOf('=');
            String value = pair.substring(eq + 1);
            out.put(pair.substring(0, eq), value.equals(DELETE) ? null : value);
        }
        return out;
    }

    void writeIntent(long txnId, Map<String, String> writes) {
        if (down) { dropped++; return; }
        journal.add("INTENT|" + txnId + "|" + encode(writes));
        prepared.put(txnId, new LinkedHashMap<>(writes));
        prepares++;
    }

    void commitIntent(long txnId, long commitTs) {
        if (down) { dropped++; return; }
        Map<String, String> writes = prepared.get(txnId);
        if (writes == null) return;                     // idempotent: applied already, or never prepared
        journal.add("APPLIED|" + txnId + "|" + commitTs);
        applyWrites(writes, commitTs);
        prepared.remove(txnId);
        applied.add(txnId);
        commitsMade++;
    }

    void abortIntent(long txnId) {
        if (down) { dropped++; return; }
        if (!prepared.containsKey(txnId)) return;
        journal.add("ABORTED|" + txnId);
        prepared.remove(txnId);
        discards++;
    }

    private void applyWrites(Map<String, String> writes, long commitTs) {
        for (Map.Entry<String, String> e : writes.entrySet()) {
            data.computeIfAbsent(e.getKey(), k -> new TreeMap<>()).put(commitTs, new Version(commitTs, e.getValue()));
        }
    }

    /** Crash: volatile state is lost, the journal is not. */
    void crash() {
        data.clear();
        prepared.clear();
        applied.clear();
        down = true;
    }

    void restore() { down = false; replayJournal(); }

    /** Rebuilds data, intents and the applied set from the journal alone (write-ahead log replay). */
    void replayJournal() {
        data.clear();
        prepared.clear();
        applied.clear();
        for (String record : journal) {
            String[] field = record.split("\\|");
            long txnId = Long.parseLong(field[1]);
            if (field[0].equals("INTENT")) {
                prepared.put(txnId, decode(field[2]));
            } else if (field[0].equals("APPLIED")) {
                applyWrites(prepared.getOrDefault(txnId, new LinkedHashMap<>()), Long.parseLong(field[2]));
                prepared.remove(txnId);
                applied.add(txnId);
            } else {
                prepared.remove(txnId);
            }
        }
    }

    /**
     * Garbage collection: drop versions older than the watermark but keep the newest one below it,
     * which is what a reader exactly at the watermark still needs. Reads strictly below the
     * watermark then fall back to that surviving version instead of failing.
     */
    int compact(long watermark) {
        int removed = 0;
        for (TreeMap<Long, Version> column : data.values()) {
            Long keep = column.lowerKey(watermark);     // newest version below the watermark
            if (keep == null) continue;
            while (!column.isEmpty() && column.firstKey() < keep) {
                column.pollFirstEntry();
                removed++;
            }
        }
        return removed;
    }

    int versionCount() {
        int n = 0;
        for (TreeMap<Long, Version> column : data.values()) n += column.size();
        return n;
    }

    String renderState() { return Coordinator.render(scan(Long.MAX_VALUE)); }
}

final class Txn {
    static final int ACTIVE = 0, COMMITTED = 1, ABORTED = 2;

    final long id, startTs;                 // startTs is the snapshot the reads use
    long commitTs;
    int state = ACTIVE;
    String abortReason = "";
    final Map<String, String> writes = new LinkedHashMap<>();
    final Set<String> reads = new TreeSet<>();

    Txn(long id, long startTs) { this.id = id; this.startTs = startTs; }

    void write(String key, String value) { writes.put(key, value); }

    void delete(String key) { writes.put(key, null); }
}

/** The transaction manager: timestamp oracle, validation, and the 2PC coordinator. */
final class Coordinator {
    static final String COMMIT = "COMMIT", ABORT = "ABORT", CRASHED = "CRASHED";

    final List<Shard> shards = new ArrayList<>();
    final Map<String, Shard> byName = new TreeMap<>();
    final List<String> wal = new ArrayList<>();
    final List<String> frames = new ArrayList<>();      // frames currently on the wire
    long clock;
    long nextTxnId = 1;
    boolean up = true;

    boolean validateReads = true;                       // toggled off to exhibit write skew
    boolean buggyConflictCheck;                         // negative control
    boolean crashAfterPrepare, crashAfterDecision;

    final Map<Long, Txn> live = new TreeMap<>();        // volatile coordinator cache
    int committed, aborted, wwConflicts, rwConflicts, prepareFailures;
    int recoveries, resends, droppedFrames;

    Coordinator(int shardCount) {
        for (int i = 0; i < shardCount; i++) {
            Shard s = new Shard("s" + i);
            shards.add(s);
            byName.put(s.name, s);
        }
    }

    Shard shardFor(String key) { return shards.get(Math.floorMod(key.hashCode(), shards.size())); }

    Txn begin() {
        Txn t = new Txn(nextTxnId++, ++clock);
        live.put(t.id, t);
        return t;
    }

    String read(Txn t, String key) {
        if (t.state != Txn.ACTIVE) throw new IllegalStateException("read outside a live transaction");
        if (t.writes.containsKey(key)) return t.writes.get(key);       // read your own writes
        t.reads.add(key);
        return shardFor(key).read(key, t.startTs);
    }

    Map<String, String> scan(Txn t, String prefix) {
        if (t.state != Txn.ACTIVE) throw new IllegalStateException("scan outside a live transaction");
        TreeMap<String, String> out = new TreeMap<>();
        for (Shard s : shards) {
            for (Map.Entry<String, String> e : s.scan(t.startTs).entrySet()) {
                if (!e.getKey().startsWith(prefix)) continue;
                out.put(e.getKey(), e.getValue());
                t.reads.add(e.getKey());
            }
        }
        for (Map.Entry<String, String> e : t.writes.entrySet()) {
            if (!e.getKey().startsWith(prefix)) continue;
            if (e.getValue() == null) out.remove(e.getKey());
            else out.put(e.getKey(), e.getValue());
        }
        return out;
    }

    void write(Txn t, String key, String value) { t.write(key, value); }

    void delete(Txn t, String key) { t.delete(key); }

    List<Shard> participantsOf(Txn t) {
        TreeSet<Shard> set = new TreeSet<>(Comparator.comparing(s -> s.name));
        for (String key : t.writes.keySet()) set.add(shardFor(key));
        return new ArrayList<>(set);
    }

    /** Runs validation and the whole 2PC protocol; returns COMMIT, ABORT or CRASHED. */
    String commit(Txn t) {
        if (!up) return CRASHED;
        if (t.state != Txn.ACTIVE) throw new IllegalStateException("transaction is not live");
        t.commitTs = ++clock;
        if (t.writes.isEmpty()) {                       // read-only: nothing to prepare
            t.state = Txn.COMMITTED;
            committed++;
            return COMMIT;
        }
        if (!buggyConflictCheck) {
            for (String key : t.writes.keySet()) {
                if (shardFor(key).hasVersionAfter(key, t.startTs)) {
                    return reject(t, "write-write conflict on " + key, true);
                }
            }
        }
        if (validateReads) {
            for (String key : t.reads) {
                if (t.writes.containsKey(key)) continue;
                if (shardFor(key).hasVersionAfter(key, t.startTs)) {
                    return reject(t, "read validation failed on " + key, false);
                }
            }
        }
        List<Shard> participants = participantsOf(t);
        wal.add("PREPARE|" + t.id);
        for (Shard s : participants) frames.add("PREPARE|" + t.id + "|" + s.name + "|0");
        deliver();
        boolean allPrepared = true;
        for (Shard s : participants) if (!s.prepared.containsKey(t.id)) allPrepared = false;
        if (!allPrepared) {                             // a participant is unreachable: nothing applied
            prepareFailures++;
            wal.add("DECIDE|" + t.id + "|ABORT|0");
            for (Shard s : participants) frames.add("ABORT|" + t.id + "|" + s.name + "|0");
            deliver();
            return reject(t, "prepare phase failed", false);
        }
        if (crashAfterPrepare) return CRASHED;          // the decision is never logged
        wal.add("DECIDE|" + t.id + "|COMMIT|" + t.commitTs);
        if (crashAfterDecision) return CRASHED;         // logged, but no frame was sent
        for (Shard s : participants) frames.add("COMMIT|" + t.id + "|" + s.name + "|" + t.commitTs);
        deliver();
        t.state = Txn.COMMITTED;
        committed++;
        return COMMIT;
    }

    private String reject(Txn t, String reason, boolean writeConflict) {
        if (writeConflict) wwConflicts++; else if (!reason.contains("prepare")) rwConflicts++;
        t.state = Txn.ABORTED;
        t.abortReason = reason;
        aborted++;
        live.remove(t.id);
        return ABORT;
    }

    /** Delivers every frame on the wire; frames addressed to a down node are lost. */
    void deliver() {
        List<String> pending = new ArrayList<>(frames);
        frames.clear();
        for (String frame : pending) {
            String[] field = frame.split("\\|");
            Shard target = byName.get(field[2]);
            long txnId = Long.parseLong(field[1]);
            if (target.down) { target.dropped++; droppedFrames++; continue; }
            if (field[0].equals("PREPARE")) {
                Txn t = live.get(txnId);
                if (t == null) { droppedFrames++; continue; }   // intent lost with the coordinator
                Map<String, String> forShard = new LinkedHashMap<>();
                for (Map.Entry<String, String> e : t.writes.entrySet()) {
                    if (shardFor(e.getKey()) == target) forShard.put(e.getKey(), e.getValue());
                }
                target.writeIntent(txnId, forShard);
            } else if (field[0].equals("COMMIT")) {
                target.commitIntent(txnId, Long.parseLong(field[3]));
            } else {
                target.abortIntent(txnId);
            }
        }
    }

    String decisionOf(long txnId) {
        for (String record : wal) {
            if (!record.startsWith("DECIDE|")) continue;
            String[] field = record.split("\\|");
            if (Long.parseLong(field[1]) == txnId) return field[2];
        }
        return null;
    }

    long commitTsOf(long txnId) {
        for (String record : wal) {
            if (!record.startsWith("DECIDE|")) continue;
            String[] field = record.split("\\|");
            if (Long.parseLong(field[1]) == txnId) return Long.parseLong(field[3]);
        }
        return -1;
    }

    int committedDecisionsInWal() {
        int n = 0;
        for (String record : wal) if (record.startsWith("DECIDE|") && record.contains("|COMMIT|")) n++;
        return n;
    }

    int abortDecisionsInWal() {
        int n = 0;
        for (String record : wal) if (record.startsWith("DECIDE|") && record.contains("|ABORT|")) n++;
        return n;
    }

    /** Coordinator failure: the volatile transaction cache dies, the write-ahead log survives. */
    void crash() {
        up = false;
        live.clear();
    }

    /** Recovery: every logged PREPARE without a decision becomes an ABORT, then participants follow. */
    int recover() {
        up = true;
        recoveries++;
        for (String record : new ArrayList<>(wal)) {
            if (!record.startsWith("PREPARE|")) continue;
            long txnId = Long.parseLong(record.split("\\|")[1]);
            if (decisionOf(txnId) == null) wal.add("DECIDE|" + txnId + "|ABORT|0");
        }
        int resolved = 0;
        for (Shard s : shards) resolved += resolve(s);
        return resolved;
    }

    /** One participant's recovery: replay its journal, then adopt the coordinator's decision. */
    int resolve(Shard s) {
        if (s.down || !up) return 0;
        s.replayJournal();
        int resolved = 0;
        for (Long txnId : new ArrayList<>(s.prepared.keySet())) {
            String decision = decisionOf(txnId);
            if (decision == null) continue;              // blocking: no decision, no resolution
            if (decision.equals("COMMIT")) s.commitIntent(txnId, commitTsOf(txnId));
            else s.abortIntent(txnId);
            resends++;
            resolved++;
        }
        return resolved;
    }

    int unresolvedIntents() {
        int n = 0;
        for (Shard s : shards) n += s.prepared.size();
        return n;
    }

    TreeMap<String, String> scanAll(long ts) {
        TreeMap<String, String> out = new TreeMap<>();
        for (Shard s : shards) out.putAll(s.scan(ts));
        return out;
    }

    static String render(Map<String, String> state) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String> e : state.entrySet()) {
            if (sb.length() > 0) sb.append(',');
            sb.append(e.getKey()).append('=').append(e.getValue());
        }
        return sb.toString();
    }
}

/** An independent replay of the committed transactions: the oracle for isolation and durability. */
final class Oracle {
    final List<Txn> commitLog = new ArrayList<>();

    void record(Txn t) { if (t.state == Txn.COMMITTED) commitLog.add(t); }

    TreeMap<String, String> at(long ts) {
        TreeMap<String, String> state = new TreeMap<>();
        for (Txn t : commitLog) {
            if (t.commitTs > ts) continue;
            for (Map.Entry<String, String> e : t.writes.entrySet()) {
                if (e.getValue() == null) state.remove(e.getKey());
                else state.put(e.getKey(), e.getValue());
            }
        }
        return state;
    }
}

public class Solution {

    static final Map<String, Long> METRICS = new TreeMap<>();

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static void metric(String name, long value) { METRICS.merge(name, value, Long::sum); }

    /** Happy path: read-your-own-writes, invisibility before commit, visibility after. */
    static void basics() {
        Coordinator c = new Coordinator(3);
        Oracle oracle = new Oracle();
        Txn t = c.begin();
        check(c.read(t, "alpha") == null, "a missing key reads as absent");
        c.write(t, "alpha", "1");
        check("1".equals(c.read(t, "alpha")), "a transaction reads its own writes");
        check(c.shardFor("alpha").read("alpha", t.startTs) == null,
                "the uncommitted write is invisible to snapshots");
        check(Coordinator.COMMIT.equals(c.commit(t)), "the transaction commits");
        check("1".equals(c.read(c.begin(), "alpha")), "it is visible to later snapshots");
        check("1".equals(c.shardFor("alpha").read("alpha", t.commitTs)), "visible at its commit timestamp");
        check(c.shardFor("alpha").read("alpha", t.startTs) == null, "still invisible at its own start snapshot");
        oracle.record(t);
        check(Coordinator.render(c.scanAll(Long.MAX_VALUE)).equals(Coordinator.render(oracle.at(Long.MAX_VALUE))),
                "the store matches the oracle");
        metric("basicCommits", c.committed);
    }

    /** Snapshot reads equal an independent replay at the same timestamp, including time travel. */
    static void snapshotIsolation() {
        Coordinator c = new Coordinator(4);
        Oracle oracle = new Oracle();
        for (int i = 1; i <= 6; i++) {
            Txn t = c.begin();
            c.write(t, "k" + (i % 3), Integer.toString(i));
            c.write(t, "shared", Integer.toString(i));
            check(Coordinator.COMMIT.equals(c.commit(t)), "write " + i + " commits");
            oracle.record(t);
        }
        for (long ts : new long[] {1, 3, 5, 7, 9, 100}) {
            check(Coordinator.render(c.scanAll(ts)).equals(Coordinator.render(oracle.at(ts))),
                    "snapshot at " + ts + " is " + Coordinator.render(c.scanAll(ts))
                            + " but the oracle says " + Coordinator.render(oracle.at(ts)));
        }
        Txn longReader = c.begin();
        Map<String, String> frozen = c.scan(longReader, "");
        for (int i = 0; i < 3; i++) {
            Txn w = c.begin();
            c.write(w, "shared", "late" + i);
            c.commit(w);
        }
        check(Coordinator.render(c.scan(longReader, "")).equals(Coordinator.render(frozen)),
                "an open transaction keeps its snapshot while others commit");
        check("6".equals(c.read(longReader, "shared")), "and still sees the old value");
        check(Coordinator.COMMIT.equals(c.commit(longReader)), "a read-only transaction commits trivially");
        metric("snapshotsChecked", 6);
    }

    /** Deletes are versions, so an older snapshot still sees the value that was deleted. */
    static void tombstones() {
        Coordinator c = new Coordinator(2);
        Txn t1 = c.begin();
        c.write(t1, "doc", "v1");
        c.commit(t1);
        long ts1 = t1.commitTs;
        Txn t2 = c.begin();
        c.write(t2, "doc", "v2");
        c.commit(t2);
        long ts2 = t2.commitTs;
        Txn t3 = c.begin();
        c.delete(t3, "doc");
        c.commit(t3);
        check(c.shardFor("doc").read("doc", t3.commitTs) == null, "the delete hides the key afterwards");
        check("v2".equals(c.shardFor("doc").read("doc", ts2)), "the previous snapshot still sees v2");
        check("v1".equals(c.shardFor("doc").read("doc", ts1)), "an older snapshot still sees v1");
        check(!c.shardFor("doc").scan(t3.commitTs).containsKey("doc"), "scans honour the tombstone");
        Txn t4 = c.begin();
        c.write(t4, "doc", "v3");
        c.commit(t4);
        check("v3".equals(c.shardFor("doc").read("doc", t4.commitTs)), "a later write resurrects the key");
        check(c.shardFor("doc").versionCount() == 4, "all four versions survive: "
                + c.shardFor("doc").versionCount());
    }

    /** First-committer-wins, the retry loop, and the negative control that loses updates. */
    static void conflictsAndLostUpdates() {
        Coordinator c = new Coordinator(4);
        Txn a = c.begin();
        Txn b = c.begin();
        c.write(a, "hot", "A");
        c.write(b, "hot", "B");
        check(Coordinator.COMMIT.equals(c.commit(a)), "the first committer wins");
        check(Coordinator.ABORT.equals(c.commit(b)), "the second is rejected on a write-write conflict");
        check(c.wwConflicts == 1, "one write-write conflict was counted: " + c.wwConflicts);
        check("A".equals(c.shardFor("hot").read("hot", Long.MAX_VALUE)), "the winner's value stuck");
        check(b.state == Txn.ABORTED && b.abortReason.contains("write-write"), "the loser reports why");
        check(!c.shardFor("hot").applied.contains(b.id), "the loser applied nothing anywhere");

        // Forty overlapping increments of one counter, retried until every one of them succeeds.
        Coordinator d = new Coordinator(2);
        Txn init = d.begin();
        d.write(init, "counter", "0");
        d.commit(init);
        int done = 0, retries = 0;
        while (done < 40) {
            List<Txn> group = new ArrayList<>();
            for (int i = 0; i < Math.min(8, 40 - done); i++) {
                Txn t = d.begin();
                int current = Integer.parseInt(d.read(t, "counter"));
                d.write(t, "counter", Integer.toString(current + 1));
                group.add(t);
            }
            for (Txn t : group) {
                if (Coordinator.COMMIT.equals(d.commit(t))) done++;
                else retries++;
            }
        }
        check(done == 40, "all forty increments eventually committed: " + done);
        check("40".equals(d.shardFor("counter").read("counter", Long.MAX_VALUE)),
                "the counter is exactly forty - no updates were lost");
        check(retries > 0, "the conflict rule did reject overlapping attempts: " + retries);
        check(d.aborted == retries, "every rejection was an abort: " + d.aborted);

        Coordinator e = new Coordinator(2);
        e.buggyConflictCheck = true;                    // negative control: no write-write check
        Txn e0 = e.begin();
        e.write(e0, "counter", "0");
        e.commit(e0);
        int accepted = 0;
        for (int round = 0; round < 8; round++) {
            List<Txn> group = new ArrayList<>();
            for (int i = 0; i < 5; i++) {
                Txn t = e.begin();
                int current = Integer.parseInt(e.read(t, "counter"));
                e.write(t, "counter", Integer.toString(current + 1));
                group.add(t);
            }
            for (Txn t : group) if (Coordinator.COMMIT.equals(e.commit(t))) accepted++;
        }
        check(accepted == 40, "the buggy store also reports forty commits: " + accepted);
        int finalValue = Integer.parseInt(e.shardFor("counter").read("counter", Long.MAX_VALUE));
        check(finalValue < 40, "with the conflict check disabled updates are lost: counter=" + finalValue);
        check(finalValue == 8, "exactly one update per round survived: counter=" + finalValue);
        metric("lostUpdates", 40 - finalValue);
        metric("conflictAborts", retries);
    }

    /** Snapshot isolation alone permits write skew; backward read validation closes it. */
    static void writeSkew() {
        Coordinator c = new Coordinator(2);
        c.validateReads = false;                        // pure snapshot isolation
        Txn setup = c.begin();
        c.write(setup, "oncall_a", "yes");
        c.write(setup, "oncall_b", "yes");
        c.commit(setup);
        Txn one = c.begin();
        Txn two = c.begin();
        check("yes".equals(c.read(one, "oncall_a")) && "yes".equals(c.read(one, "oncall_b")),
                "the first doctor sees both of them on call");
        check("yes".equals(c.read(two, "oncall_a")) && "yes".equals(c.read(two, "oncall_b")),
                "the second doctor sees both of them on call");
        c.write(one, "oncall_a", "no");
        c.write(two, "oncall_b", "no");
        check(Coordinator.COMMIT.equals(c.commit(one)), "under snapshot isolation the first commits");
        check(Coordinator.COMMIT.equals(c.commit(two)), "and so does the second - write skew");
        check(c.scanAll(Long.MAX_VALUE).values().stream().noneMatch("yes"::equals),
                "both doctors went off call: the invariant is broken");

        Coordinator d = new Coordinator(2);
        d.validateReads = true;                         // backward read validation
        Txn s = d.begin();
        d.write(s, "oncall_a", "yes");
        d.write(s, "oncall_b", "yes");
        d.commit(s);
        Txn x = d.begin();
        Txn y = d.begin();
        check("yes".equals(d.read(x, "oncall_a")) && "yes".equals(d.read(x, "oncall_b")), "x sees both");
        check("yes".equals(d.read(y, "oncall_a")) && "yes".equals(d.read(y, "oncall_b")), "y sees both");
        d.write(x, "oncall_a", "no");
        d.write(y, "oncall_b", "no");
        check(Coordinator.COMMIT.equals(d.commit(x)), "the first still commits");
        check(Coordinator.ABORT.equals(d.commit(y)), "read validation rejects the skewed pair");
        check(d.rwConflicts == 1, "one read-validation failure was counted: " + d.rwConflicts);
        check("yes".equals(d.shardFor("oncall_b").read("oncall_b", Long.MAX_VALUE)),
                "the invariant holds: somebody is still on call");
        metric("readValidationAborts", d.rwConflicts);
    }

    /** Two-phase commit across two shards: atomicity under every injected failure. */
    static void twoPhaseCommit() {
        Coordinator c = new Coordinator(3);
        String ka = "key_a", kb = "key_b";
        Shard sa = c.shardFor(ka), sb = c.shardFor(kb);
        check(!sa.name.equals(sb.name), "the two keys live on different shards: " + sa.name + "/" + sb.name);

        Txn t = c.begin();
        c.write(t, ka, "va");
        c.write(t, kb, "vb");
        check(Coordinator.COMMIT.equals(c.commit(t)), "the two-shard transaction commits");
        check(sa.applied.contains(t.id) && sb.applied.contains(t.id), "both shards applied it");
        check("va".equals(sa.read(ka, Long.MAX_VALUE)) && "vb".equals(sb.read(kb, Long.MAX_VALUE)),
                "both values are readable");
        check(c.abortDecisionsInWal() == 0, "no abort was logged on the happy path");

        Txn u = c.begin();
        c.write(u, ka, "ua");
        c.write(u, kb, "ub");
        sb.down = true;
        check(Coordinator.ABORT.equals(c.commit(u)), "a failed prepare aborts the transaction");
        sb.down = false;
        sb.restore();
        check(!sa.applied.contains(u.id) && !sb.applied.contains(u.id), "nothing was applied anywhere");
        check("va".equals(sa.read(ka, Long.MAX_VALUE)), "the earlier value is untouched");
        check(c.prepareFailures == 1, "the prepare failure was counted");

        Txn v = c.begin();
        c.write(v, ka, "va2");
        c.write(v, kb, "vb2");
        c.crashAfterPrepare = true;
        check(Coordinator.CRASHED.equals(c.commit(v)), "the coordinator dies before deciding");
        c.crashAfterPrepare = false;
        check(c.unresolvedIntents() == 2, "both shards are left holding an intent");
        check(c.decisionOf(v.id) == null, "no decision for it reached the log");
        c.crash();
        c.recover();
        check(c.unresolvedIntents() == 0, "recovery turned every undecided intent into an abort");
        check(!sa.applied.contains(v.id) && !sb.applied.contains(v.id), "the transaction applied nowhere");
        check(c.abortDecisionsInWal() == 2, "two abort decisions were logged: " + c.abortDecisionsInWal());

        Txn w = c.begin();
        c.write(w, ka, "va3");
        c.write(w, kb, "vb3");
        c.crashAfterDecision = true;
        check(Coordinator.CRASHED.equals(c.commit(w)), "the coordinator dies after logging the decision");
        c.crashAfterDecision = false;
        check(c.committedDecisionsInWal() == 2, "both commit decisions survived in the log: "
                + c.committedDecisionsInWal());
        c.crash();
        c.recover();
        check(sa.applied.contains(w.id) && sb.applied.contains(w.id),
                "recovery re-sent the commit to both shards");
        check("va3".equals(sa.read(ka, Long.MAX_VALUE)) && "vb3".equals(sb.read(kb, Long.MAX_VALUE)),
                "the logged decision was carried out");
        check(c.unresolvedIntents() == 0, "no intent is left behind");
        metric("resends", c.resends);
        metric("droppedFrames", c.droppedFrames);
    }

    /** Participant crash, the blocking nature of 2PC, journal replay and a full reboot. */
    static void participantRecoveryAndReboot() {
        Coordinator c = new Coordinator(2);
        Oracle oracle = new Oracle();
        Shard s0 = c.shardFor("x"), s1 = c.shardFor("y");
        check(!s0.name.equals(s1.name), "x and y are on different shards");
        Txn first = c.begin();
        c.write(first, "x", "1");
        c.write(first, "y", "1");
        check(Coordinator.COMMIT.equals(c.commit(first)), "a two-shard transaction commits");
        oracle.record(first);

        String s0Before = s0.renderState();
        s0.crash();
        check(s0.read("x", Long.MAX_VALUE) == null, "the crashed shard lost its volatile state");
        Txn blocked = c.begin();
        c.write(blocked, "x", "2");
        check(Coordinator.ABORT.equals(c.commit(blocked)), "no prepare can reach the crashed shard");
        check(!s1.applied.contains(blocked.id), "and nothing is applied on the surviving shard");
        s0.restore();
        check(s0.renderState().equals(s0Before), "journal replay restored the crashed shard bit for bit");

        // A participant with an intent whose coordinator is down stays blocked - 2PC does not guess.
        Txn wait = c.begin();
        c.write(wait, "x", "w");
        c.write(wait, "y", "w");
        c.crashAfterPrepare = true;
        check(Coordinator.CRASHED.equals(c.commit(wait)), "the coordinator dies after both prepares");
        c.crashAfterPrepare = false;
        check(c.unresolvedIntents() == 2, "both shards hold an undecided intent: " + c.unresolvedIntents());
        c.crash();
        int blockedResolves = 0;
        for (Shard s : c.shards) blockedResolves += c.resolve(s);
        check(blockedResolves == 0, "a participant never resolves on its own while the coordinator is down");
        check(c.unresolvedIntents() == 2, "so both intents stay pending");

        c.recover();
        check(c.unresolvedIntents() == 0, "once the coordinator is back every intent resolves");
        check(!c.shards.stream().anyMatch(s -> s.applied.contains(wait.id)),
                "and the undecided transaction is aborted everywhere");
        check("1".equals(s0.read("x", Long.MAX_VALUE)), "the aborted intents changed no data");

        Txn second = c.begin();
        c.write(second, "x", "3");
        c.write(second, "y", "3");
        check(Coordinator.COMMIT.equals(c.commit(second)), "another transaction commits before the reboot");
        oracle.record(second);
        for (Shard s : c.shards) s.crash();
        check(c.scanAll(Long.MAX_VALUE).isEmpty(), "the reboot wiped every shard");
        for (Shard s : c.shards) s.restore();
        String after = Coordinator.render(c.scanAll(Long.MAX_VALUE));
        String expected = Coordinator.render(oracle.at(Long.MAX_VALUE));
        check(after.equals(expected), "after the reboot the store is " + after + ", the oracle " + expected);
        check(after.contains("y=3"), "including the transaction committed last before the reboot");
        check("1".equals(s0.read("x", first.commitTs)), "old snapshots still resolve through the versions");
        metric("journalRecords", c.shards.stream().mapToInt(s -> s.journal.size()).sum());
    }

    /** Compaction drops obsolete versions without disturbing reads at or after the watermark. */
    static void compaction() {
        Coordinator c = new Coordinator(2);
        Oracle oracle = new Oracle();
        for (int i = 1; i <= 10; i++) {
            Txn t = c.begin();
            c.write(t, "seq", Integer.toString(i));
            check(Coordinator.COMMIT.equals(c.commit(t)), "write " + i + " commits");
            oracle.record(t);
        }
        long watermark = c.clock - 4;
        int before = c.shards.stream().mapToInt(Shard::versionCount).sum();
        int removed = c.shards.stream().mapToInt(s -> s.compact(watermark)).sum();
        int after = c.shards.stream().mapToInt(Shard::versionCount).sum();
        check(removed > 0, "compaction removed obsolete versions: " + removed);
        check(after == before - removed, "the version count dropped by exactly that many: "
                + before + " -> " + after);
        check(Coordinator.render(c.scanAll(watermark)).equals(Coordinator.render(oracle.at(watermark))),
                "a snapshot at the watermark is still exact");
        check(Coordinator.render(c.scanAll(Long.MAX_VALUE)).equals(Coordinator.render(oracle.at(Long.MAX_VALUE))),
                "and the newest state is untouched");
        Txn reader = c.begin();
        check("10".equals(c.read(reader, "seq")), "a fresh snapshot still sees the newest value");
        check(c.shardFor("seq").read("seq", watermark - 1) != null,
                "a read below the watermark degrades to the oldest surviving version instead of failing");
        metric("versionsCompacted", removed);
    }

    public static void main(String[] args) {
        basics();
        snapshotIsolation();
        tombstones();
        conflictsAndLostUpdates();
        writeSkew();
        twoPhaseCommit();
        participantRecoveryAndReboot();
        compaction();
        StringBuilder sb = new StringBuilder("metrics:");
        for (Map.Entry<String, Long> e : METRICS.entrySet()) {
            sb.append(' ').append(e.getKey()).append('=').append(e.getValue());
        }
        System.out.println(sb);
        System.out.println("PASS");
    }
}
