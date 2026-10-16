// Task 966 - Implement proctoring/anticheat measures for timed contests (privacy-aware).
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// A contest session emits privacy-relevant signals (focus lost, paste, tab switch, clipboard copy,
// disconnect, idle).  The monitor turns them into a risk score with a per-signal weight and a per-signal
// cap, so no single signal can dominate, and the score maps onto a decision: >= 60 is flagged for human
// review, 40..59 produces a soft note, below 40 is clear.  Privacy is enforced by construction: the audit
// record stores only a hash of the session id, per-signal counters and the score - never the pasted text.
//
// Verified: the incremental ("live") accumulator reproduces the direct recomputation on the same stream
// (the independent oracle), a session scoring exactly 60 is flagged while 59 is only a soft note, a
// benign session stays clear, a single oversized signal is capped, an empty stream scores 0, the payload
// never reaches the audit record, and hashing plus serialisation are deterministic.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }

    private sealed class Signal
    {
        public readonly string Kind;
        public readonly int Amount; // milliseconds, characters or event count, depending on the kind
        public readonly string Payload;
        public Signal(string kind, int amount, string payload) { Kind = kind; Amount = amount; Payload = payload; }
    }

    private sealed class Rule
    {
        public readonly double PerUnit;
        public readonly double Cap;
        public Rule(double perUnit, double cap) { PerUnit = perUnit; Cap = cap; }
    }

    private static readonly Dictionary<string, Rule> Rules = new Dictionary<string, Rule>(StringComparer.Ordinal)
    {
        ["focus_lost"] = new Rule(0.002, 20),     // 2 points per second away from the contest window
        ["paste"] = new Rule(0.5, 25),            // half a point per pasted character
        ["tab_switch"] = new Rule(5.0, 20),       // five points per switch
        ["clipboard_copy"] = new Rule(0.3, 10),   // 0.3 points per copied character
        ["disconnect"] = new Rule(0.002, 15),     // 2 points per disconnected second
        ["idle"] = new Rule(0.0, 0.0),            // thinking is not evidence of anything
    };

    /** Direct computation: group by kind, then cap each kind once. */
    private static double DirectScore(List<Signal> stream)
    {
        var perKind = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (Signal signal in stream)
        {
            if (!Rules.TryGetValue(signal.Kind, out Rule rule)) continue;
            perKind[signal.Kind] = (perKind.TryGetValue(signal.Kind, out double running) ? running : 0.0) + signal.Amount * rule.PerUnit;
        }
        double total = 0.0;
        foreach (KeyValuePair<string, double> pair in perKind) total += Math.Min(pair.Value, Rules[pair.Key].Cap);
        return total;
    }

    /** Live computation (the streaming path): accumulate the capped delta as each signal arrives. */
    private static double IncrementalScore(List<Signal> stream)
    {
        var perKind = new Dictionary<string, double>(StringComparer.Ordinal);
        double total = 0.0;
        foreach (Signal signal in stream)
        {
            if (!Rules.TryGetValue(signal.Kind, out Rule rule)) continue;
            double before = perKind.TryGetValue(signal.Kind, out double running) ? running : 0.0;
            double after = before + signal.Amount * rule.PerUnit;
            perKind[signal.Kind] = after;
            total += Math.Min(after, rule.Cap) - Math.Min(before, rule.Cap);
        }
        return total;
    }

    private static string Verdict(double score) => score >= 60 ? "flagged-for-human-review" : score >= 40 ? "soft-note" : "clear";

    /** FNV-1a: the record keeps this hash instead of the raw participant identity. */
    private static ulong StableHash(string text)
    {
        unchecked
        {
            ulong hash = 14695981039346656037UL;
            foreach (char ch in text) { hash ^= ch; hash *= 1099511628211UL; }
            return hash;
        }
    }

    private sealed class AuditRecord
    {
        public readonly ulong SessionHash;
        public readonly double Score;
        public readonly string Verdict;
        public readonly SortedDictionary<string, int> Counters = new SortedDictionary<string, int>(StringComparer.Ordinal);
        public AuditRecord(ulong hash, double score, string verdict) { SessionHash = hash; Score = score; Verdict = verdict; }
        public string Serialize()
        {
            var sb = new StringBuilder();
            sb.Append("session=").Append(SessionHash.ToString("x16", CultureInfo.InvariantCulture))
              .Append(";score=").Append(Score.ToString("F2", CultureInfo.InvariantCulture))
              .Append(";verdict=").Append(Verdict).Append(";signals=");
            foreach (KeyValuePair<string, int> pair in Counters) sb.Append(pair.Key).Append('=').Append(pair.Value).Append(',');
            return sb.ToString();
        }
    }

    private static AuditRecord Monitor(string sessionId, List<Signal> stream)
    {
        double score = IncrementalScore(stream);
        var record = new AuditRecord(StableHash(sessionId), score, Verdict(score));
        foreach (Signal signal in stream) record.Counters[signal.Kind] = record.Counters.TryGetValue(signal.Kind, out int c) ? c + 1 : 1;
        return record;
    }

    public static void Main()
    {
        var cheat = new List<Signal>
        {
            new Signal("focus_lost", 10_000, ""), new Signal("paste", 10_000, "secret-answer"), new Signal("tab_switch", 1, ""),
            new Signal("clipboard_copy", 1_000, ""), new Signal("tab_switch", 1, ""), new Signal("disconnect", 10_000, ""),
        };
        AuditRecord cheatRecord = Monitor("session-ada", cheat);
        Check(cheatRecord.Score >= 60 && cheatRecord.Verdict == "flagged-for-human-review",
            "a session that pastes, copies and disconnects is flagged for human review (" + cheatRecord.Score.ToString("F2", CultureInfo.InvariantCulture) + ")");

        var benign = new List<Signal>
        {
            new Signal("focus_lost", 4_000, ""), new Signal("tab_switch", 1, ""), new Signal("idle", 60_000, ""), new Signal("idle", 60_000, ""),
        };
        AuditRecord benignRecord = Monitor("session-grace", benign);
        Check(benignRecord.Score < 40 && benignRecord.Verdict == "clear",
            "an ordinary session stays clear (" + benignRecord.Score.ToString("F2", CultureInfo.InvariantCulture) + ")");

        // boundary: focus 10s + paste 50 chars + disconnect 7.5s == 20 + 25 + 15 == exactly 60.
        var exactly = new List<Signal>
        {
            new Signal("focus_lost", 10_000, ""), new Signal("paste", 50, ""), new Signal("disconnect", 7_500, ""),
        };
        AuditRecord boundary = Monitor("session-hedy", exactly);
        Check(Math.Abs(boundary.Score - 60.0) < 1e-9 && boundary.Verdict == "flagged-for-human-review",
            "a score of exactly 60 is flagged - the threshold is inclusive (" + boundary.Score.ToString("F2", CultureInfo.InvariantCulture) + ")");
        Check(Verdict(59.99) == "soft-note" && Verdict(40.0) == "soft-note" && Verdict(39.99) == "clear",
            "the soft-note band is [40, 60) and everything below 40 is clear");

        var soft = new List<Signal> { new Signal("focus_lost", 5_000, ""), new Signal("paste", 100, ""), new Signal("tab_switch", 2, "") };
        Check(Monitor("session-linus", soft).Verdict == "soft-note", "a mid-range session produces a soft note rather than a flag");

        Check(Math.Abs(DirectScore(cheat) - IncrementalScore(cheat)) < 1e-9 && Math.Abs(DirectScore(benign) - IncrementalScore(benign)) < 1e-9 &&
            Math.Abs(DirectScore(exactly) - IncrementalScore(exactly)) < 1e-9,
            "the live accumulator matches the direct recomputation on every session (the oracle)");

        var hugePaste = new List<Signal> { new Signal("paste", 100_000, "") };
        Check(Math.Abs(Monitor("session-edsger", hugePaste).Score - 25.0) < 1e-9,
            "one enormous paste is capped at 25 points, so a single signal cannot dominate");

        // privacy: the record never carries the payload, only a hash, counters and the score.
        string serialized = cheatRecord.Serialize();
        Check(!serialized.Contains("secret-answer", StringComparison.Ordinal) && !serialized.Contains("ANSWER", StringComparison.OrdinalIgnoreCase) &&
            serialized.Contains("session=") && serialized.Contains("paste=1"),
            "the audit record keeps a hash, counters and the score but never the pasted payload");
        Check(serialized.StartsWith("session=" + StableHash("session-ada").ToString("x16", CultureInfo.InvariantCulture), StringComparison.Ordinal),
            "the record identifies the session only through its stable hash");

        AuditRecord emptyRecord = Monitor("session-empty", new List<Signal>());
        Check(emptyRecord.Score == 0.0 && emptyRecord.Verdict == "clear" && emptyRecord.Counters.Count == 0,
            "an empty stream scores 0, is clear and records no counters");
        Check(Monitor("session-ada", cheat).Serialize() == serialized, "replaying the same stream reproduces the identical record");
        Check(StableHash("a") == StableHash("a") && StableHash("a") != StableHash("b"), "the session hash is stable and distinguishes sessions");

        Console.WriteLine("monitor: cheat " + cheatRecord.Score.ToString("F2", CultureInfo.InvariantCulture) + " (flagged), benign " +
            benignRecord.Score.ToString("F2", CultureInfo.InvariantCulture) + " (clear), boundary " + boundary.Score.ToString("F2", CultureInfo.InvariantCulture) + " (>=60)");
        Console.WriteLine("record: " + serialized);
        Console.WriteLine("privacy: payload dropped, only " + StableHash("session-ada").ToString("x16", CultureInfo.InvariantCulture) + " plus counters retained");
        Console.WriteLine("PASS");
    }
}
