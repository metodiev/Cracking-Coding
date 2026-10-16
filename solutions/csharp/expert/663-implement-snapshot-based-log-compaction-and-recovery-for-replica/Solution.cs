// Task 663 - Implement snapshot-based log compaction and recovery for replicated log.
// (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// A replicated log (index, term, command) in front of a deterministic state machine, plus the compaction path
// that keeps such a log from growing without bound: once the state machine has applied up to index S, the
// machine is serialised into a snapshot, the log prefix up to S is discarded, and a replica that has fallen
// behind is brought up to date by installing the snapshot and replaying only the retained suffix. The
// invariant this file leans on hardest is the one that makes that legal: a snapshot at S plus the entries
// after S reconstructs exactly the state that replaying the whole log from index 1 would produce. A replica
// also refuses to compact past what it has applied, which is what stops a snapshot from claiming state the
// machine never reached.
// What is simplified: leadership, terms, quorum agreement and the transport are out of scope (entries are
// appended in order), the snapshot payload is a hand-written text format with an FNV-1a checksum instead of a
// binary encoding, and the state machine is a small key/value store. The replay equivalence, the checksum and
// parse validation, the refuse-older-snapshot rule, the apply-before-compact rule and the contiguity of the
// retained log are all real.
// Verified: the full replay matches an independently recomputed state fingerprint; snapshot + suffix replay
// equals the full replay and also equals the live replica after it keeps appending; a follower that installs
// the snapshot and the suffix is identical to the leader; compaction moves the log's first index and drops
// exactly the prefix, and reading a compacted entry fails; compaction past the applied index and past the end
// of the log are both rejected; the snapshot round-trips through its serialised form while a tampered payload,
// a payload that does not parse and an older snapshot are all refused; an incremental second snapshot agrees
// with installing the first and replaying the difference; the retained log stays contiguous with
// non-decreasing terms; and an untouched machine snapshots to an empty but valid payload.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class LogEntry
{
    internal int Index;
    internal int Term;
    internal string Command = string.Empty;
}

internal static class Fnv
{
    internal static ulong Hash(string text)
    {
        ulong hash = 1469598103934665603UL;
        foreach (char c in text)
        {
            hash ^= (byte)(c & 0xFF);
            hash *= 1099511628211UL;
            hash ^= (byte)(c >> 8);
            hash *= 1099511628211UL;
        }
        return hash;
    }
}

internal sealed class StateMachine
{
    private readonly SortedDictionary<string, long> _values = new SortedDictionary<string, long>(StringComparer.Ordinal);

    /// <summary>Deterministic apply: `set key value` and `incr key delta`.</summary>
    internal void Apply(string command)
    {
        string[] parts = command.Split(' ');
        if (parts.Length != 3) throw new ArgumentException("malformed command: " + command);
        string key = parts[1];
        long amount = long.Parse(parts[2]);
        if (parts[0] == "set") { _values[key] = amount; return; }
        if (parts[0] == "incr")
        {
            _values[key] = _values.TryGetValue(key, out long current) ? current + amount : amount;
            return;
        }
        throw new ArgumentException("unknown command: " + command);
    }

    /// <summary>Canonical serialisation: sorted keys, each as key=value;.</summary>
    internal string Serialise()
    {
        var builder = new StringBuilder();
        foreach (var pair in _values) builder.Append(pair.Key).Append('=').Append(pair.Value).Append(';');
        return builder.ToString();
    }

    internal void Load(string payload)
    {
        _values.Clear();
        if (payload.Length == 0) return;
        foreach (string field in payload.Split(';'))
        {
            if (field.Length == 0) continue;
            int split = field.IndexOf('=');
            if (split <= 0) throw new ArgumentException("malformed snapshot field: " + field);
            _values[field.Substring(0, split)] = long.Parse(field.Substring(split + 1));
        }
    }

    internal long Get(string key) => _values.TryGetValue(key, out long value) ? value : 0;

    internal int Size => _values.Count;

    internal string Fingerprint() => Fnv.Hash(Serialise()).ToString("x16");
}

internal sealed class Snapshot
{
    internal int LastIncludedIndex;
    internal int LastIncludedTerm;
    internal string Payload = string.Empty;
    internal ulong Checksum;

    internal static Snapshot Take(StateMachine machine, int index, int term)
    {
        string payload = machine.Serialise();
        return new Snapshot
        {
            LastIncludedIndex = index,
            LastIncludedTerm = term,
            Payload = payload,
            Checksum = Fnv.Hash(index + "|" + term + "|" + payload),
        };
    }

    /// <summary>Returns null when the snapshot is intact, otherwise why it was refused.</summary>
    internal string? Validate()
    {
        if (LastIncludedIndex < 0) return "negative snapshot index";
        if (Fnv.Hash(LastIncludedIndex + "|" + LastIncludedTerm + "|" + Payload) != Checksum)
            return "checksum mismatch";
        try
        {
            var probe = new StateMachine();
            probe.Load(Payload);
        }
        catch (Exception) { return "payload does not parse"; }
        return null;
    }

    internal StateMachine Restore()
    {
        if (Validate() is string problem) throw new ArgumentException("refusing corrupt snapshot: " + problem);
        var machine = new StateMachine();
        machine.Load(Payload);
        return machine;
    }

    internal string Describe() => "snapshot@" + LastIncludedIndex + "/term" + LastIncludedTerm
        + " payload=" + Payload + " checksum=" + Checksum.ToString("x8");
}

internal sealed class ReplicaLog
{
    private readonly List<LogEntry> _entries = new List<LogEntry>();
    private readonly StateMachine _machine = new StateMachine();

    internal Snapshot? Installed { get; private set; }

    internal StateMachine Machine => _machine;

    internal int LastIncludedIndex => Installed?.LastIncludedIndex ?? 0;

    internal int LastIncludedTerm => Installed?.LastIncludedTerm ?? 0;

    internal int Count => _entries.Count;

    internal int LastIndex => _entries.Count == 0 ? LastIncludedIndex : _entries[_entries.Count - 1].Index;

    /// <summary>Highest index the state machine has applied.</summary>
    internal int AppliedIndex { get; private set; }

    internal void Append(LogEntry entry)
    {
        if (entry.Index != LastIndex + 1)
            throw new ArgumentException("log indexes must be contiguous: got " + entry.Index + ", expected " + (LastIndex + 1));
        _entries.Add(entry);
    }

    internal LogEntry? At(int index)
    {
        if (index <= LastIncludedIndex) return null;                 // compacted away
        int offset = index - LastIncludedIndex - 1;
        return offset >= 0 && offset < _entries.Count ? _entries[offset] : null;
    }

    /// <summary>Applies retained entries up to `index` (inclusive).</summary>
    internal void ApplyUpTo(int index)
    {
        if (index < AppliedIndex) throw new ArgumentException("the state machine cannot roll back to " + index);
        for (int position = AppliedIndex + 1; position <= index; position++)
        {
            LogEntry? entry = At(position);
            if (entry is null) throw new ArgumentException("entry " + position + " is not available to apply");
            _machine.Apply(entry.Command);
        }
        AppliedIndex = index;
    }

    /// <summary>Snapshots the applied state at `index` and drops the log prefix up to it.</summary>
    internal Snapshot Compact(int index)
    {
        if (index > LastIndex) throw new ArgumentException("cannot compact past the end of the log");
        if (index <= LastIncludedIndex) throw new ArgumentException("cannot compact below the installed snapshot");
        if (index != AppliedIndex) throw new ArgumentException("cannot compact an index the state machine has not applied");
        LogEntry? entry = At(index);
        int term = entry is null ? LastIncludedTerm : entry.Term;
        Snapshot snapshot = Snapshot.Take(_machine, index, term);
        _entries.RemoveRange(0, index - LastIncludedIndex);
        Installed = snapshot;
        return snapshot;
    }

    /// <summary>Installs a snapshot that supersedes the current one; returns the reason when refused.</summary>
    internal string? TryInstall(Snapshot snapshot)
    {
        if (snapshot.Validate() is string problem) return problem;
        if (snapshot.LastIncludedIndex <= LastIncludedIndex) return "snapshot is not newer than the current one";
        _entries.RemoveAll(entry => entry.Index <= snapshot.LastIncludedIndex);
        Installed = snapshot;
        // The installed snapshot already represents the state at that index, so the machine has "applied" it.
        AppliedIndex = Math.Max(AppliedIndex, snapshot.LastIncludedIndex);
        return null;
    }

    internal List<LogEntry> SuffixAfter(int index)
    {
        var suffix = new List<LogEntry>();
        foreach (LogEntry entry in _entries) if (entry.Index > index) suffix.Add(entry);
        return suffix;
    }

    /// <summary>Null when the retained log is contiguous, ordered and terms never decrease.</summary>
    internal string? InvariantViolation()
    {
        for (int i = 0; i < _entries.Count; i++)
        {
            int expected = LastIncludedIndex + 1 + i;
            if (_entries[i].Index != expected)
                return "entry " + i + " has index " + _entries[i].Index + ", expected " + expected;
            if (i > 0 && _entries[i].Term < _entries[i - 1].Term)
                return "terms decreased at entry " + i;
        }
        if (AppliedIndex < LastIncludedIndex) return "the state machine lags behind the snapshot";
        return null;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }

    private static ReplicaLog BuildLog(int entries, ulong seed)
    {
        ulong state = seed;
        var log = new ReplicaLog();
        for (int index = 1; index <= entries; index++)
        {
            string command = index % 3 == 0
                ? "set key" + Rng.NextInt(ref state, 6) + " " + Rng.NextInt(ref state, 100)
                : "incr key" + Rng.NextInt(ref state, 6) + " " + (1 + Rng.NextInt(ref state, 9));
            log.Append(new LogEntry { Index = index, Term = 1 + index / 10, Command = command });
        }
        return log;
    }

    /// <summary>Replays every entry of a log into a fresh machine without compacting anything.</summary>
    private static StateMachine FullReplay(ReplicaLog log, int lastIndex)
    {
        var machine = new StateMachine();
        for (int index = 1; index <= lastIndex; index++)
        {
            LogEntry? entry = log.At(index);
            if (entry is null) throw new ArgumentException("entry " + index + " is missing from the log");
            machine.Apply(entry.Command);
        }
        return machine;
    }

    /// <summary>Installs a snapshot and replays the suffix - the recovery path of a lagging replica.</summary>
    private static StateMachine Recover(Snapshot? snapshot, IReadOnlyList<LogEntry> suffix, int lastIndex)
    {
        StateMachine machine = snapshot is null ? new StateMachine() : snapshot.Restore();
        int expected = snapshot?.LastIncludedIndex ?? 0;
        foreach (LogEntry entry in suffix)
        {
            if (entry.Index != expected + 1) throw new ArgumentException("suffix is not contiguous at " + entry.Index);
            machine.Apply(entry.Command);
            expected = entry.Index;
        }
        if (expected != lastIndex) throw new ArgumentException("replay stopped at " + expected + ", expected " + lastIndex);
        return machine;
    }

    public static void Main()
    {
        // ---- full replay versus snapshot plus suffix --------------------------------------------
        {
            ReplicaLog log = BuildLog(120, 0x0001020304050607UL);
            StateMachine reference = FullReplay(log, 120);
            Check(log.LastIndex == 120 && log.Count == 120 && log.InvariantViolation() == null,
                "the freshly built log holds 120 contiguous entries with non-decreasing terms");
            Check(reference.Size > 0, "the reference replay produced a non-trivial state (" + reference.Size + " keys)");

            log.ApplyUpTo(70);
            Snapshot snapshot = log.Compact(70);
            Check(log.LastIncludedIndex == 70 && log.Count == 50 && log.AppliedIndex == 70,
                "compaction discards the fifty-entry prefix and records the snapshot index");
            Check(log.At(70) is null && log.At(71) is not null,
                "entries up to the snapshot are gone while the suffix is still addressable");
            Check(snapshot.Validate() == null, "the snapshot it produced is valid");

            log.ApplyUpTo(120);
            Check(log.Machine.Fingerprint() == reference.Fingerprint(),
                "a replica that compacts and keeps applying reaches the same state as a full replay");
            Check(Recover(snapshot, log.SuffixAfter(70), 120).Fingerprint() == reference.Fingerprint(),
                "snapshot plus suffix reconstructs the identical state (the compaction invariant)");
            Check(log.InvariantViolation() == null, "the log stays contiguous after compaction");
            Console.WriteLine("logEntries=120 snapshotAt=70 payloadBytes=" + snapshot.Payload.Length
                + " keys=" + reference.Size);
        }

        // ---- a follower catching up from a snapshot ---------------------------------------------
        {
            ReplicaLog leaderLog = BuildLog(200, 0x1112131415161718UL);
            leaderLog.ApplyUpTo(150);
            Snapshot snapshot = leaderLog.Compact(150);
            leaderLog.ApplyUpTo(200);
            StateMachine leaderState = leaderLog.Machine;

            var follower = new ReplicaLog();
            Check(follower.TryInstall(snapshot) == null, "the follower installs the leader's snapshot");
            Check(follower.LastIncludedIndex == 150 && follower.Count == 0 && follower.AppliedIndex == 150,
                "after installing, the follower holds only the snapshot and counts it as applied");
            foreach (LogEntry entry in leaderLog.SuffixAfter(150)) follower.Append(entry);
            follower.ApplyUpTo(200);
            Check(follower.Machine.Fingerprint() == leaderState.Fingerprint(),
                "the caught-up follower is byte-identical to the leader");
            Check(follower.InvariantViolation() == null && follower.LastIndex == 200,
                "the follower's log is contiguous up to the leader's last index");

            string? refusal = leaderLog.TryInstall(snapshot);
            Check(refusal != null && refusal.Contains("not newer"),
                "installing a snapshot the replica already has is refused: " + refusal);
            var stale = Snapshot.Take(new StateMachine(), 10, 1);
            Check(leaderLog.TryInstall(stale) != null, "an older snapshot is refused as well");
        }

        // ---- snapshot serialisation, corruption and truncation ---------------------------------
        {
            ReplicaLog log = BuildLog(60, 0x2122232425262728UL);
            log.ApplyUpTo(40);
            Snapshot snapshot = log.Compact(40);
            Check(snapshot.Payload.Contains("=") && snapshot.Payload.EndsWith(";"),
                "the payload is the canonical sorted key=value form: " + snapshot.Payload);
            StateMachine restored = snapshot.Restore();
            Check(restored.Fingerprint() == log.Machine.Fingerprint()
                    && restored.Serialise() == log.Machine.Serialise(),
                "the snapshot round-trips through its serialised form");

            string payload = snapshot.Payload;
            string tampered = payload.Length == 0
                ? "x"
                : payload.Substring(0, payload.Length - 1)
                  + (payload[payload.Length - 1] == '0' ? '1' : '0');
            var flipped = new Snapshot
            {
                LastIncludedIndex = snapshot.LastIncludedIndex,
                LastIncludedTerm = snapshot.LastIncludedTerm,
                Payload = tampered,
                Checksum = snapshot.Checksum,
            };
            Check(flipped.Validate() == "checksum mismatch",
                "a tampered payload is rejected by the checksum before it can poison the state");
            bool refused = false;
            try { flipped.Restore(); }
            catch (ArgumentException) { refused = true; }
            Check(refused, "restoring a corrupt snapshot throws instead of yielding a wrong state");

            var unparsable = new Snapshot
            {
                LastIncludedIndex = snapshot.LastIncludedIndex,
                LastIncludedTerm = snapshot.LastIncludedTerm,
                Payload = "brokenfield",
                Checksum = snapshot.Checksum,
            };
            Check(unparsable.Validate() != null, "a payload that does not parse is reported as invalid");

            var emptySnapshot = Snapshot.Take(new StateMachine(), 0, 0);
            Check(emptySnapshot.Validate() == null && emptySnapshot.Payload.Length == 0
                    && emptySnapshot.Restore().Size == 0,
                "snapshotting an untouched state machine yields an empty but valid snapshot");
        }

        // ---- incremental snapshots agree with replaying the difference --------------------------
        {
            ReplicaLog log = BuildLog(150, 0x3132333435363738UL);
            log.ApplyUpTo(50);
            Snapshot first = log.Compact(50);
            var between = new List<string>();
            log.ApplyUpTo(110);
            for (int index = 51; index <= 110; index++) between.Add(log.At(index)!.Command);
            Snapshot second = log.Compact(110);
            Check(second.LastIncludedIndex == 110 && log.Count == 40,
                "the second snapshot advances the compaction point to index 110");

            StateMachine fromFirst = first.Restore();
            foreach (string command in between) fromFirst.Apply(command);
            Check(fromFirst.Fingerprint() == second.Restore().Fingerprint(),
                "installing the first snapshot and replaying the difference equals the second snapshot");
            log.ApplyUpTo(150);
            Check(Recover(second, log.SuffixAfter(110), 150).Fingerprint() == log.Machine.Fingerprint(),
                "the newest snapshot plus the retained suffix still rebuilds the replica's state");

            bool refused = false;
            try { log.Compact(200); }
            catch (ArgumentException) { refused = true; }
            Check(refused, "compacting past the end of the log is rejected");
            refused = false;
            try { log.Compact(110); }
            catch (ArgumentException) { refused = true; }
            Check(refused, "compacting below the installed snapshot is rejected");
            log.Append(new LogEntry { Index = 151, Term = 16, Command = "incr later 1" });
            refused = false;
            try { log.Compact(151); }
            catch (ArgumentException) { refused = true; }
            Check(refused, "compacting an index the state machine has not applied is rejected");
            log.ApplyUpTo(151);
            Check(log.Compact(151).LastIncludedIndex == 151 && log.Count == 0,
                "once the entry has been applied, compacting it succeeds");
            Console.WriteLine("retainedAfterFinalCompaction=" + log.Count + " snapshotBytes=" + second.Payload.Length);
        }

        // ---- a long run compacted repeatedly ---------------------------------------------------
        {
            ulong state = 0x4142434445464748UL;
            var log = new ReplicaLog();
            var reference = new StateMachine();
            int compactions = 0;
            bool equivalent = true, contiguous = true, matchesReference = true;
            for (int index = 1; index <= 600; index++)
            {
                string command = "incr counter " + (1 + Rng.NextInt(ref state, 5));
                log.Append(new LogEntry { Index = index, Term = 1 + index / 50, Command = command });
                reference.Apply(command);
                log.ApplyUpTo(index);
                if (index % 75 == 0)
                {
                    log.Compact(index);
                    compactions++;
                }
                if (log.InvariantViolation() != null) contiguous = false;
                if (log.Machine.Fingerprint() != reference.Fingerprint()) matchesReference = false;
                if (Recover(log.Installed, log.SuffixAfter(log.LastIncludedIndex), index).Fingerprint()
                    != reference.Fingerprint()) equivalent = false;
            }
            Check(compactions == 8, "the run compacted eight times (got " + compactions + ")");
            Check(log.Count == 0 && log.LastIncludedIndex == 600,
                "after the last compaction the log is empty and the snapshot covers everything");
            Check(contiguous, "the retained log stayed contiguous with non-decreasing terms throughout");
            Check(matchesReference, "the live state machine tracked the reference replay at every entry");
            Check(equivalent, "after every entry, snapshot + suffix still equalled the full replay");
            Snapshot lastSnapshot = log.Installed!;
            Check(lastSnapshot.Validate() == null && lastSnapshot.Restore().Fingerprint() == reference.Fingerprint(),
                "the final snapshot alone restores the final state without any log entries");
            Console.WriteLine("compactions=" + compactions + " finalSnapshotBytes=" + lastSnapshot.Payload.Length
                + " finalCounter=" + reference.Get("counter"));
        }

        Console.WriteLine("PASS");
    }
}
