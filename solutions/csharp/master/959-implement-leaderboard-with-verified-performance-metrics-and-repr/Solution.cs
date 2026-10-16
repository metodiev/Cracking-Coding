// Task 959 - Implement leaderboard with verified performance metrics and reproducible runs.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Every submission carries a claimed time, which is verified by re-running the benchmark a fixed number
// of times in a simulated environment: the true cost is jittered by a splitmix64 stream seeded from the
// run seed and the submission id, and the median is the verified time.  A claim is accepted only when it
// sits within 5% of the verified time and above the physical floor of the workload; anything faster than
// the hardware could possibly be is rejected as impossible rather than trusted.
//
// Verified: verified times stay within the modelled jitter band of the true cost, the board is ordered by
// verified time with a deterministic id tie-break, an independent selection-sort oracle reproduces the
// order, unverified and impossible claims are excluded with reasons, dense ranks and percentiles behave,
// the empty and single-entry boundaries hold, and the same seed reproduces the whole board byte-for-byte.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public static class Solution
{
    private static ulong _state;
    private static ulong NextUInt64()
    {
        unchecked
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }
    private static double NextUnit() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }
    private sealed class Submission
    {
        public readonly string Id;
        public readonly string User;
        public readonly double TrueMs;
        public readonly double ClaimedMs;
        public Submission(string id, string user, double trueMs, double claimedMs)
        {
            Id = id;
            User = user;
            TrueMs = trueMs;
            ClaimedMs = claimedMs;
        }
    }
    private sealed class Entry
    {
        public readonly Submission Source;
        public readonly double VerifiedMs;
        public readonly string Status;
        public readonly string Reason;
        public Entry(Submission source, double verifiedMs, string status, string reason)
        {
            Source = source;
            VerifiedMs = verifiedMs;
            Status = status;
            Reason = reason;
        }
    }
    private sealed class Board
    {
        public readonly List<Entry> Verified = new List<Entry>();
        public readonly List<Entry> Ranked = new List<Entry>();
    }
    private const double PhysicalFloorMs = 0.5;
    private const double ClaimTolerance = 0.05;
    private const int Runs = 5;
    /** FNV-1a over the submission id: String.GetHashCode is randomised per process, this is not. */
    private static ulong StableHash(string text)
    {
        unchecked
        {
            ulong hash = 14695981039346656037UL;
            foreach (char ch in text)
            {
                hash ^= ch;
                hash *= 1099511628211UL;
            }
            return hash;
        }
    }
    private static double VerifiedTime(ulong seed, Submission submission)
    {
        _state = seed ^ StableHash(submission.Id) ^ 0x5DEECE66DUL;
        var samples = new double[Runs];
        for (int i = 0; i < Runs; i++) samples[i] = submission.TrueMs * (0.98 + 0.04 * NextUnit());
        Array.Sort(samples);
        return samples[Runs / 2];
    }
    private static Board Build(ulong seed, List<Submission> submissions)
    {
        var board = new Board();
        foreach (Submission submission in submissions)
        {
            double verified = VerifiedTime(seed, submission);
            if (submission.ClaimedMs < PhysicalFloorMs)
            {
                board.Verified.Add(new Entry(submission, verified, "rejected",
                    "claim " + submission.ClaimedMs.ToString("F2", CultureInfo.InvariantCulture) + "ms is below the physical floor"));
            }
            else if (Math.Abs(submission.ClaimedMs - verified) / verified > ClaimTolerance)
            {
                board.Verified.Add(new Entry(submission, verified, "unverified",
                    "claim differs from the verified run by more than 5%"));
            }
            else
            {
                board.Verified.Add(new Entry(submission, verified, "accepted", ""));
            }
        }
        foreach (Entry entry in board.Verified)
        {
            if (entry.Status == "accepted") board.Ranked.Add(entry);
        }
        board.Ranked.Sort((a, b) => a.VerifiedMs != b.VerifiedMs
            ? a.VerifiedMs.CompareTo(b.VerifiedMs)
            : string.CompareOrdinal(a.Source.Id, b.Source.Id));
        return board;
    }
    /** Independent oracle: selection sort over the same accepted entries must reproduce the order. */
    private static List<Entry> SelectionOrder(List<Entry> accepted)
    {
        var remaining = new List<Entry>(accepted);
        var ordered = new List<Entry>();
        while (remaining.Count > 0)
        {
            int best = 0;
            for (int i = 1; i < remaining.Count; i++)
            {
                bool smaller = remaining[i].VerifiedMs < remaining[best].VerifiedMs;
                bool tieBreak = remaining[i].VerifiedMs == remaining[best].VerifiedMs &&
                    string.CompareOrdinal(remaining[i].Source.Id, remaining[best].Source.Id) < 0;
                if (smaller || tieBreak) best = i;
            }
            ordered.Add(remaining[best]);
            remaining.RemoveAt(best);
        }
        return ordered;
    }
    private static int Rank(List<Entry> ranked, int index)
    {
        int rank = 1;
        for (int i = 1; i <= index; i++)
        {
            if (ranked[i].VerifiedMs != ranked[i - 1].VerifiedMs) rank++;
        }
        return rank;
    }
    private static int Percentile(int count, int index) => (int)Math.Round(100.0 * (count - index) / count);
    private static string Fingerprint(Board board)
    {
        var sb = new StringBuilder();
        foreach (Entry entry in board.Ranked)
            sb.Append(entry.Source.Id).Append(':').Append(entry.VerifiedMs.ToString("F4", CultureInfo.InvariantCulture)).Append(';');
        return sb.ToString();
    }
    public static void Main()
    {
        var submissions = new List<Submission>
        {
            new Submission("s-101", "ada", 100.0, 100.4),
            new Submission("s-102", "grace", 200.0, 203.0),
            new Submission("s-103", "linus", 300.0, 341.0),
            new Submission("s-104", "hedy", 150.0, 0.2),
            new Submission("s-105", "edsger", 250.0, 251.0),
            new Submission("s-106", "barbara", 400.0, 402.0),
        };
        Board board = Build(20240901UL, submissions);
        List<Entry> accepted = SelectionOrder(new List<Entry>(board.Ranked));
        bool orderMatches = accepted.Count == board.Ranked.Count;
        for (int i = 0; orderMatches && i < accepted.Count; i++) orderMatches &= accepted[i].Source.Id == board.Ranked[i].Source.Id;
        Check(orderMatches, "the independent selection-sort oracle reproduces the leaderboard order");
        Check(board.Ranked.Count == 4, "exactly the four verifiable claims reach the board (" + board.Ranked.Count + " ranked)");
        bool sorted = true, withinBand = true;
        for (int i = 0; i < board.Ranked.Count; i++)
        {
            if (i > 0) sorted &= board.Ranked[i].VerifiedMs >= board.Ranked[i - 1].VerifiedMs;
            double trueMs = board.Ranked[i].Source.TrueMs;
            withinBand &= Math.Abs(board.Ranked[i].VerifiedMs - trueMs) / trueMs <= 0.021;
        }
        Check(sorted, "the board is non-decreasing in verified time");
        Check(withinBand, "every verified time lies within the +-2% modelled jitter band of the true cost");
        Entry notVerified = board.Verified.Find(e => e.Source.Id == "s-103")!;
        Check(notVerified.Status == "unverified" && !board.Ranked.Exists(e => e.Source.Id == "s-103"),
            "a claim 13% off the verified run is unverified and stays off the board");
        Entry impossible = board.Verified.Find(e => e.Source.Id == "s-104")!;
        Check(impossible.Status == "rejected" && impossible.Reason.Contains("physical floor"),
            "a 0.2ms claim for a 150ms workload is rejected as physically impossible");
        Check(Rank(board.Ranked, 0) == 1 && Percentile(board.Ranked.Count, 0) == 100,
            "the leader has dense rank 1 and percentile 100");
        bool ranksRise = true, percentilesFall = true;
        for (int i = 1; i < board.Ranked.Count; i++)
        {
            ranksRise &= Rank(board.Ranked, i) >= Rank(board.Ranked, i - 1);
            percentilesFall &= Percentile(board.Ranked.Count, i) <= Percentile(board.Ranked.Count, i - 1);
        }
        Check(ranksRise && percentilesFall, "dense ranks never decrease and percentiles never increase down the board");
        var tied = new List<Entry>
        {
            new Entry(new Submission("t-2", "b", 123.0, 123.0), 123.0, "accepted", ""),
            new Entry(new Submission("t-1", "a", 123.0, 123.0), 123.0, "accepted", ""),
        };
        tied.Sort((a, b) => a.VerifiedMs != b.VerifiedMs ? a.VerifiedMs.CompareTo(b.VerifiedMs) : string.CompareOrdinal(a.Source.Id, b.Source.Id));
        Check(tied[0].Source.Id == "t-1" && tied[1].Source.Id == "t-2" && Rank(tied, 1) == 1,
            "equal verified times break the tie by submission id and share dense rank 1");
        Board empty = Build(11UL, new List<Submission>());
        Check(empty.Ranked.Count == 0 && empty.Verified.Count == 0, "an empty submission list yields an empty board");
        Board single = Build(11UL, new List<Submission> { new Submission("only", "solo", 42.0, 42.0) });
        Check(single.Ranked.Count == 1 && Rank(single.Ranked, 0) == 1 && Percentile(1, 0) == 100,
            "the single-entry boundary ranks first with percentile 100");
        Board repeat = Build(20240901UL, submissions);
        Check(Fingerprint(board) == Fingerprint(repeat), "the same seed reproduces the board's verified times exactly");
        Board otherSeed = Build(424242UL, submissions);
        Check(Fingerprint(board) != Fingerprint(otherSeed), "a different seed changes the verified times");
        bool orderStable = otherSeed.Ranked.Count == board.Ranked.Count;
        for (int i = 0; orderStable && i < board.Ranked.Count; i++) orderStable &= otherSeed.Ranked[i].Source.Id == board.Ranked[i].Source.Id;
        Check(orderStable, "the ordering survives re-seeding because the jitter band is far smaller than the spread");
        Console.WriteLine("board: " + board.Ranked.Count + " verified entries of " + submissions.Count +
            " submissions; ranks " + string.Join(", ", RankString(board.Ranked)));
        Console.WriteLine("verification: median of " + Runs + " seeded runs, +-2% jitter band, 5% claim tolerance, floor " +
            PhysicalFloorMs.ToString("F1", CultureInfo.InvariantCulture) + "ms");
        Console.WriteLine("excluded: s-103 (unverified claim), s-104 (below physical floor)");
        Console.WriteLine("PASS");
    }
    private static List<string> RankString(List<Entry> ranked)
    {
        var labels = new List<string>();
        for (int i = 0; i < ranked.Count; i++) labels.Add("#" + Rank(ranked, i) + " " + ranked[i].Source.Id);
        return labels;
    }
}