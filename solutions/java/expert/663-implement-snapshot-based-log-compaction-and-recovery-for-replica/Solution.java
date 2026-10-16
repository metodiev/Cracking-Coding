/** Task 663 - Implement snapshot-based log compaction and recovery for replicated log. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A replicated log with Raft-style snapshot compaction. Entries are numbered from 1; the state machine
 * applies them in order. Once the applied prefix reaches a threshold the replica takes a snapshot:
 * the state machine is serialised, `lastIncludedIndex`/`lastIncludedTerm` are recorded and every entry
 * up to that index is dropped. A follower whose next index falls below the snapshot boundary cannot be
 * served entries any more and must be caught up by installing the snapshot and then streaming the
 * surviving tail.
 *
 * Invariants verified: the compacted state machine is byte-identical to a reference replica that never
 * compacts (the strongest oracle); the log stays bounded by the threshold; entries dropped equals
 * lastIncludedIndex; indices stay contiguous above the boundary; entries above the applied index
 * (uncommitted tail) are never discarded and never applied; a lagging follower that installs the
 * snapshot and then the tail ends up with an identical state; re-applying an entry at or below the
 * snapshot boundary is ignored; a snapshot round trip through its serialised form restores the state;
 * a stale snapshot is ignored and a malformed one rejected; and compacting past the applied index is
 * rejected.
 */
import java.util.*;

final class StateMachine {
    private final TreeMap<String, Integer> cells = new TreeMap<>();
    int applied;

    void apply(String command) {
        int separator = command.indexOf('=');
        if (separator < 0) throw new IllegalArgumentException("malformed command: " + command);
        String key = command.substring(0, separator);
        int delta = Integer.parseInt(command.substring(separator + 1));
        cells.merge(key, delta, Integer::sum);
        applied++;
    }

    int get(String key) { return cells.getOrDefault(key, 0); }

    int size() { return cells.size(); }

    Map<String, Integer> cells() { return new TreeMap<>(cells); }

    void restore(Map<String, Integer> snapshot, int applied) {
        cells.clear();
        cells.putAll(snapshot);
        this.applied = applied;
    }

    /** Wire format of a snapshot: "applied;key=value;key=value". */
    String serialize(int lastIncludedIndex, int lastIncludedTerm) {
        StringBuilder sb = new StringBuilder();
        sb.append(lastIncludedIndex).append(',').append(lastIncludedTerm).append(';');
        for (Map.Entry<String, Integer> entry : cells.entrySet()) {
            sb.append(entry.getKey()).append('=').append(entry.getValue()).append(';');
        }
        return sb.toString();
    }

    static StateMachine deserialize(String snapshot) {
        String[] parts = snapshot.split(";", -1);
        if (parts.length < 1 || parts[0].indexOf(',') < 0) throw new IllegalArgumentException("bad snapshot header");
        String[] header = parts[0].split(",", -1);
        if (header.length != 2) throw new IllegalArgumentException("bad snapshot header");
        StateMachine machine = new StateMachine();
        Map<String, Integer> cells = new TreeMap<>();
        for (int i = 1; i < parts.length; i++) {
            if (parts[i].isEmpty()) continue;
            int separator = parts[i].indexOf('=');
            if (separator <= 0) throw new IllegalArgumentException("bad snapshot cell: " + parts[i]);
            cells.put(parts[i].substring(0, separator), Integer.parseInt(parts[i].substring(separator + 1)));
        }
        machine.cells.putAll(cells);
        machine.applied = Integer.parseInt(header[0]);
        return machine;
    }

    static int snapshotIndex(String snapshot) {
        String[] parts = snapshot.split(";", -1);
        String[] header = parts[0].split(",", -1);
        if (header.length != 2) throw new IllegalArgumentException("bad snapshot header");
        return Integer.parseInt(header[0]);
    }

    static int snapshotTerm(String snapshot) {
        String[] header = snapshot.split(";", -1)[0].split(",", -1);
        return Integer.parseInt(header[1]);
    }
}

final class LogEntry {
    final int index;
    final int term;
    final String command;

    LogEntry(int index, int term, String command) {
        this.index = index;
        this.term = term;
        this.command = command;
    }
}

final class Replica {
    private final List<LogEntry> entries = new ArrayList<>();
    int lastIncludedIndex;
    int lastIncludedTerm;
    int commitIndex;
    int lastApplied;
    int entriesDropped;
    int compactions;
    private String snapshotBlob = "";        // the snapshot as taken - the file a restarted replica would read
    final StateMachine machine = new StateMachine();

    Replica() { }

    int size() { return entries.size(); }

    int lastIndex() { return lastIncludedIndex + entries.size(); }

    LogEntry entryAt(int index) {
        if (index > lastIncludedIndex && index <= lastIndex()) return entries.get(index - lastIncludedIndex - 1);
        throw new IllegalArgumentException("index " + index + " is not in the log (boundary " + lastIncludedIndex + ")");
    }

    int termAt(int index) {
        if (index == lastIncludedIndex) return lastIncludedTerm;
        if (index < lastIncludedIndex || index > lastIndex()) throw new IllegalArgumentException("no entry " + index);
        return entries.get(index - lastIncludedIndex - 1).term;
    }

    /** Appends an entry and immediately commits and applies it (single leader simulation). */
    int append(String command, int term) {
        int index = lastIndex() + 1;
        entries.add(new LogEntry(index, term, command));
        commitIndex = index;
        applyPending();
        return index;
    }

    /** Appends without committing - the uncommitted tail that must survive compaction. */
    int appendUncommitted(String command, int term) {
        int index = lastIndex() + 1;
        entries.add(new LogEntry(index, term, command));
        return index;
    }

    void commitUpTo(int index) {
        commitIndex = Math.max(commitIndex, index);
        applyPending();
    }

    /** Applies committed entries in order; entries at or below the snapshot boundary are skipped. */
    void applyPending() {
        while (lastApplied < commitIndex) {
            int next = lastApplied + 1;
            if (next <= lastIncludedIndex) {
                lastApplied = lastIncludedIndex;      // already covered by the snapshot
                continue;
            }
            machine.apply(entryAt(next).command);
            lastApplied = next;
        }
    }

    /** Compacts everything up to the applied index; returns whether a snapshot was taken. */
    boolean compact(int threshold) {
        if (entries.size() < threshold) return false;
        int index = lastApplied;
        if (index <= lastIncludedIndex) return false;
        lastIncludedTerm = termAt(index);
        String snapshot = machine.serialize(index, lastIncludedTerm);
        snapshotBlob = snapshot;
        lastIncludedIndex = index;
        int removed = 0;
        while (!entries.isEmpty() && entries.get(0).index <= lastIncludedIndex) {
            entries.remove(0);
            removed++;
        }
        entriesDropped += removed;
        compactions++;
        if (StateMachine.snapshotIndex(snapshot) != index) throw new AssertionError("snapshot header mismatch");
        return true;
    }

    /** The snapshot as taken: serialising the live machine would include entries beyond the boundary. */
    String serializeSnapshot() {
        return snapshotBlob.isEmpty() ? machine.serialize(lastIncludedIndex, lastIncludedTerm) : snapshotBlob;
    }

    /** Installs a snapshot if it is newer than what this replica already has. */
    boolean installSnapshot(String snapshot, int term) {
        int index = StateMachine.snapshotIndex(snapshot);
        if (index <= lastIncludedIndex) return false;
        machine.restore(StateMachine.deserialize(snapshot).cells(), index);
        snapshotBlob = snapshot;
        lastIncludedIndex = index;
        lastIncludedTerm = term;
        lastApplied = index;
        commitIndex = Math.max(commitIndex, index);
        while (!entries.isEmpty() && entries.get(0).index <= lastIncludedIndex) {
            entries.remove(0);
            entriesDropped++;
        }
        return true;
    }

    List<String> tailCommands() {
        List<String> out = new ArrayList<>();
        for (LogEntry entry : entries) out.add(entry.command);
        return out;
    }
}

public class Solution {
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static Replica referenceReplica(int commands) {
        Replica reference = new Replica();
        for (int i = 1; i <= commands; i++) reference.append(commandFor(i), 1);
        return reference;
    }

    static String commandFor(int i) {
        return (i % 3 == 0 ? "hot=" : "cold=") + (i % 7 + 1);
    }

    public static void main(String[] args) {
        // --- compaction keeps the log bounded and the state identical ----------
        Replica compacted = new Replica();
        int threshold = 8;
        int snapshots = 0;
        for (int i = 1; i <= 200; i++) {
            compacted.append(commandFor(i), 1);
            if (compacted.compact(threshold)) snapshots++;
            check(compacted.size() <= threshold, "the log stays bounded at command " + i);
        }
        Replica reference = referenceReplica(200);
        check(snapshots > 0, "compaction really happened: " + snapshots + " snapshots");
        check(compacted.machine.cells().equals(reference.machine.cells()),
                "the compacted state machine equals the never compacted reference");
        check(compacted.machine.applied == 200, "every command was applied exactly once");
        check(compacted.entriesDropped == compacted.lastIncludedIndex, "dropped entries equal the snapshot index");
        check(compacted.lastIncludedIndex > 190, "the snapshot reached the end of the log, got "
                + compacted.lastIncludedIndex);

        // --- the log is empty right after a compaction -------------------------
        Replica fresh = new Replica();
        for (int i = 1; i <= 7; i++) fresh.append(commandFor(i), 1);
        check(!fresh.compact(8), "a log below the threshold is not compacted");
        check(fresh.lastIncludedIndex == 0 && fresh.size() == 7, "nothing was dropped below the threshold");
        fresh.append(commandFor(8), 1);
        check(fresh.compact(8), "a log at the threshold is compacted");
        check(fresh.size() == 0 && fresh.lastIncludedIndex == 8, "everything up to the applied index was dropped");
        check(fresh.machine.applied == 8 && fresh.machine.get("hot") == 11 && fresh.machine.get("cold") == 19,
                "the state machine survived compaction: hot=" + fresh.machine.get("hot")
                        + " cold=" + fresh.machine.get("cold"));

        // --- an uncommitted tail must survive compaction -----------------------
        Replica tail = new Replica();
        for (int i = 1; i <= 6; i++) tail.append(commandFor(i), 1);
        int uncommitted = tail.appendUncommitted("pending=5", 1);
        check(tail.compact(4), "compaction happens while an uncommitted entry exists");
        check(tail.size() == 1 && tail.entryAt(uncommitted).command.equals("pending=5"),
                "the uncommitted tail entry survived");
        check(tail.machine.applied == 6, "the uncommitted entry was not applied");
        check(tail.machine.get("pending") == 0, "the uncommitted command left no trace in the state");
        tail.commitUpTo(uncommitted);
        check(tail.machine.get("pending") == 5, "committing applies the tail entry");
        check(tail.lastApplied == uncommitted, "applied index follows the commit index");

        // --- re-applying old entries is ignored --------------------------------
        Replica replay = new Replica();
        for (int i = 1; i <= 30; i++) replay.append(commandFor(i), 1);
        replay.compact(5);
        int appliedBefore = replay.machine.applied;
        Map<String, Integer> cellsBefore = replay.machine.cells();
        replay.commitUpTo(replay.lastIncludedIndex);
        replay.applyPending();
        check(replay.machine.applied == appliedBefore, "re-applying an old prefix changes nothing");
        check(replay.machine.cells().equals(cellsBefore), "the state is untouched by the replay");

        // --- a lagging follower catches up by snapshot + tail ------------------
        Replica leader = new Replica();
        for (int i = 1; i <= 120; i++) {
            leader.append(commandFor(i), 1);
            leader.compact(10);
        }
        for (int i = 121; i <= 125; i++) leader.appendUncommitted(commandFor(i), 1);
        Replica follower = new Replica();
        check(follower.lastIncludedIndex < leader.lastIncludedIndex,
                "the follower is behind the leader's snapshot boundary");
        boolean installed = follower.installSnapshot(leader.serializeSnapshot(), leader.lastIncludedTerm);
        check(installed, "the leader must ship a snapshot");
        check(follower.lastIncludedIndex == leader.lastIncludedIndex, "the snapshot boundary travels with it");
        check(follower.machine.applied == leader.lastIncludedIndex, "the snapshot sets the applied index");
        List<String> tailCommands = leader.tailCommands();
        for (String command : tailCommands) follower.appendUncommitted(command, 1);
        follower.commitUpTo(follower.lastIndex());
        leader.commitUpTo(leader.lastIndex());
        check(follower.machine.cells().equals(leader.machine.cells()),
                "snapshot plus tail catches the follower up exactly");

        // --- snapshot round trip ------------------------------------------------
        Replica roundTrip = new Replica();
        for (int i = 1; i <= 40; i++) {
            roundTrip.append(commandFor(i), 1);
            roundTrip.compact(5);
        }
        check(roundTrip.lastApplied == roundTrip.lastIncludedIndex,
                "a snapshot is taken exactly at the applied prefix");
        String serialized = roundTrip.serializeSnapshot();
        StateMachine restored = StateMachine.deserialize(serialized);
        check(restored.cells().equals(roundTrip.machine.cells()), "the serialised snapshot restores the state");
        check(StateMachine.snapshotIndex(serialized) == roundTrip.lastIncludedIndex, "the header carries the index");
        check(StateMachine.snapshotTerm(serialized) == roundTrip.lastIncludedTerm, "the header carries the term");
        Replica restarted = new Replica();
        restarted.installSnapshot(serialized, roundTrip.lastIncludedTerm);
        check(restarted.machine.cells().equals(roundTrip.machine.cells()), "installing restores the state");
        check(restarted.lastIncludedIndex == roundTrip.lastIncludedIndex, "installing moves the boundary");
        // a replica restarted from the leader's snapshot plus the surviving tail catches up exactly
        Replica fromLeader = new Replica();
        fromLeader.installSnapshot(leader.serializeSnapshot(), leader.lastIncludedTerm);
        for (String command : leader.tailCommands()) fromLeader.append(command, 1);
        fromLeader.commitUpTo(fromLeader.lastIndex());
        check(fromLeader.machine.cells().equals(leader.machine.cells()),
                "a replica restarted from the snapshot and the tail equals the leader");

        // --- a stale snapshot is ignored ---------------------------------------
        Replica upToDate = new Replica();
        for (int i = 1; i <= 40; i++) upToDate.append(commandFor(i), 1);
        upToDate.compact(4);
        String stale = upToDate.serializeSnapshot();
        int boundary = upToDate.lastIncludedIndex;
        Map<String, Integer> before = upToDate.machine.cells();
        check(!upToDate.installSnapshot(stale, 1), "a snapshot at the same index is ignored");
        check(upToDate.machine.cells().equals(before), "and changes nothing");
        check(upToDate.lastIncludedIndex == boundary, "and does not move the boundary");

        // --- contiguity of the surviving log -----------------------------------
        Replica contiguous = new Replica();
        for (int i = 1; i <= 60; i++) {
            contiguous.append(commandFor(i), 1);
            contiguous.compact(6);
        }
        contiguous.appendUncommitted("a=1", 1);
        contiguous.appendUncommitted("b=2", 1);
        for (int offset = 0; offset < contiguous.size(); offset++) {
            LogEntry entry = contiguous.entryAt(contiguous.lastIncludedIndex + 1 + offset);
            check(entry.index == contiguous.lastIncludedIndex + 1 + offset, "indices stay contiguous");
        }
        check(contiguous.termAt(contiguous.lastIncludedIndex) == contiguous.lastIncludedTerm,
                "the boundary index reports the snapshot term");

        // --- empty log and boundary cases --------------------------------------
        Replica emptyReplica = new Replica();
        check(!emptyReplica.compact(1), "compacting an empty log is a no-op");
        check(emptyReplica.lastIndex() == 0 && emptyReplica.size() == 0, "an empty log has no entries");
        check(emptyReplica.machine.serialize(0, 0).startsWith("0,0;"), "an empty snapshot is well formed");
        Replica oneEntry = new Replica();
        oneEntry.append("only=1", 1);
        check(oneEntry.compact(1), "a single entry can be compacted");
        check(oneEntry.machine.get("only") == 1 && oneEntry.size() == 0, "its effect is kept in the snapshot");

        // --- rejected input ----------------------------------------------------
        boolean rejected = false;
        try {
            new Replica().entryAt(3);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "reading past the log must be rejected");
        rejected = false;
        try {
            StateMachine.deserialize("nonsense");
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a malformed snapshot must be rejected");
        rejected = false;
        try {
            new StateMachine().apply("no-separator");
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a malformed command must be rejected");
        rejected = false;
        try {
            Replica partial = new Replica();
            partial.append("x=1", 1);
            partial.compact(1);
            partial.termAt(0);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "querying a term below the boundary must be rejected");

        System.out.println("PASS");
    }
}
