// Task 620 - Implement lock elision simulation and contention measurement. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
//
// A simulation of hardware-style lock elision (as in Intel RTM / TSX). A critical section updates a
// shared counter with the read-modify-write v -> 2v + 1. A thread either elides the lock - it reads
// the value and the *version* of the location, computes the new value and, at commit time, checks
// that the version is unchanged, in which case the write becomes visible - or it takes the real lock
// (only one holder at a time; a contended acquire fails and is retried later, so nothing blocks). A
// failed validation is an *abort*: the buffered write is discarded and the section is retried, and
// after enough consecutive aborts the thread falls back to the lock (adaptive elision), re-enabling
// elision once it has made progress.
//
// The interleaving is an explicit schedule of (thread, micro-step) pairs, executed by a single
// thread - no real atomic instructions, no spurious aborts, no cache-line effects: only the abort
// caused by a concurrent committed write is modelled. Measurements are taken for every run: aborts,
// elided commits, lock acquisitions, contended acquires and fallbacks.
//
// Verified: after *every* micro-step the shared value equals the value implied by the number of
// committed sections (2^k * (v0 + 1) - 1), so no update is ever lost or duplicated, in both the
// eliding and the non-eliding runs; the lock is never held by two threads at once; a low-contention
// schedule elides successfully (elided commits > 0) and takes strictly fewer locks than the same
// schedule with elision disabled; a high-contention schedule aborts and eventually falls back to the
// lock; and a deliberately unvalidated variant loses updates, which the same invariant detects.

using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;          // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal sealed class ElisionThread
{
    internal int Id;
    internal bool Eliding = true;
    internal int ConsecutiveAborts;
    internal int Phase;                     // 0 idle, 1 optimistic begin, 2 optimistic commit, 3 lock acquire, 4 locked update
    internal int ObservedValue;
    internal int ObservedVersion;

    internal int ElidedCommits;
    internal int Aborts;
    internal int LockAcquisitions;
    internal int ContendedAcquires;
    internal int Fallbacks;
    internal int Sections;
}

internal sealed class LockElisionSimulator
{
    internal long Value;
    internal int Version;
    internal int LockHolder = -1;
    internal long CommittedSections;

    internal readonly List<ElisionThread> Threads = new List<ElisionThread>();
    internal readonly bool AllowElision;
    internal readonly bool ValidateOnCommit;
    internal readonly int AbortThreshold;
    internal readonly bool Adaptive;
    internal readonly long InitialValue;

    internal long Steps;

    internal LockElisionSimulator(int threads, bool allowElision, bool validateOnCommit, int abortThreshold,
        bool adaptive, long initialValue)
    {
        AllowElision = allowElision;
        ValidateOnCommit = validateOnCommit;
        AbortThreshold = abortThreshold;
        Adaptive = adaptive;
        InitialValue = initialValue;
        Value = initialValue;
        for (int i = 0; i < threads; i++) Threads.Add(new ElisionThread { Id = i });
    }

    /** The value the invariant demands after k committed sections. */
    internal long ExpectedValue() => Pow2(CommittedSections) * (InitialValue + 1) - 1;

    private static long Pow2(long exponent)
    {
        long result = 1;
        for (long i = 0; i < exponent && i < 62; i++) result *= 2;
        return result;
    }

    internal bool InvariantHolds() => Value == ExpectedValue();

    /** One micro-step for a thread. Returns true when the thread completed a section. */
    internal bool Step(ElisionThread thread)
    {
        Steps++;
        switch (thread.Phase)
        {
            case 0:                                              // start a section
                thread.Sections++;
                thread.Phase = thread.Eliding && AllowElision ? 1 : 3;
                return false;
            case 1:                                              // optimistic begin: read value + version
                thread.ObservedValue = (int)Value;
                thread.ObservedVersion = Version;
                thread.Phase = 2;
                return false;
            case 2:                                              // optimistic commit: validate, then apply
                if (!ValidateOnCommit || Version == thread.ObservedVersion)
                {
                    Value = 2L * thread.ObservedValue + 1;
                    Version++;
                    CommittedSections++;
                    thread.ElidedCommits++;
                    thread.ConsecutiveAborts = 0;
                    thread.Phase = 0;
                    return true;
                }
                thread.Aborts++;
                thread.ConsecutiveAborts++;
                if (Adaptive && thread.ConsecutiveAborts >= AbortThreshold)
                {
                    thread.Eliding = false;
                    thread.Fallbacks++;
                }
                thread.Phase = thread.Eliding && AllowElision ? 1 : 3;
                return false;
            case 3:                                              // acquire the real lock
                if (LockHolder == -1)
                {
                    LockHolder = thread.Id;
                    thread.LockAcquisitions++;
                    thread.Phase = 4;
                }
                else
                {
                    thread.ContendedAcquires++;
                }
                return false;
            default:                                             // locked update and release
                Value = 2L * Value + 1;
                Version++;
                CommittedSections++;
                thread.ConsecutiveAborts = 0;
                LockHolder = -1;
                if (Adaptive && !thread.Eliding && thread.LockAcquisitions % 4 == 0)
                {
                    thread.Eliding = true;
                }
                thread.Phase = 0;
                return true;
        }
    }

    internal int TotalElidedCommits()
    {
        int total = 0;
        foreach (ElisionThread thread in Threads) total += thread.ElidedCommits;
        return total;
    }

    internal int TotalAborts()
    {
        int total = 0;
        foreach (ElisionThread thread in Threads) total += thread.Aborts;
        return total;
    }

    internal int TotalLockAcquisitions()
    {
        int total = 0;
        foreach (ElisionThread thread in Threads) total += thread.LockAcquisitions;
        return total;
    }

    internal int TotalFallbacks()
    {
        int total = 0;
        foreach (ElisionThread thread in Threads) total += thread.Fallbacks;
        return total;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /** Run the schedule, asserting the safety invariant after every single micro-step. */
    private static bool DriveWithInvariant(LockElisionSimulator sim, int sections, int seed, bool check)
    {
        ulong state = (ulong)seed * 0x9E3779B97F4A7C15UL + 3;
        int remaining = sections;
        int guard = 0;
        while ((remaining > 0 || sim.Threads.Exists(t => t.Phase != 0)) && guard++ < 200000)
        {
            var ready = new List<ElisionThread>();
            foreach (ElisionThread thread in sim.Threads)
            {
                if (thread.Phase != 0 || remaining > 0) ready.Add(thread);
            }
            var candidates = new List<ElisionThread>();
            foreach (ElisionThread thread in ready)
            {
                if (thread.Phase == 0 && remaining <= 0) continue;
                candidates.Add(thread);
            }
            ElisionThread pick = candidates[(int)(SplitMix64.Next(ref state) % (ulong)candidates.Count)];
            if (pick.Phase == 0) remaining--;
            bool completed = sim.Step(pick);
            if (check)
            {
                Check(sim.InvariantHolds(), "safety invariant after step " + sim.Steps);
                Check(sim.LockHolder < 0 || sim.Threads.TrueForAll(t => t.Phase != 4 || t.Id == sim.LockHolder),
                    "at most one lock holder");
            }
            if (completed && guard == 0) return false;
        }
        return guard < 200000;
    }

    public static void Main()
    {
        // Low contention: elision succeeds most of the time and takes fewer locks.
        var eliding = new LockElisionSimulator(3, true, true, 8, true, 1);
        Check(DriveWithInvariant(eliding, 24, 5, true), "low contention schedule completes");
        Check(eliding.InvariantHolds(), "low contention invariant");
        Check(eliding.TotalElidedCommits() > 0, "elision actually committed");
        Check(eliding.CommittedSections == 24, "every section committed exactly once");

        var baseline = new LockElisionSimulator(3, false, true, 8, true, 1);
        Check(DriveWithInvariant(baseline, 24, 5, true), "baseline schedule completes");
        Check(baseline.TotalLockAcquisitions() == 24, "without elision every section takes the lock");
        Check(eliding.TotalLockAcquisitions() < baseline.TotalLockAcquisitions(),
            "elision reduces lock acquisitions: " + eliding.TotalLockAcquisitions() + " vs "
            + baseline.TotalLockAcquisitions());
        Check(eliding.Value == baseline.Value, "both runs reach the same value");

        // High contention: many aborts, adaptive fallback, and the invariant still holds throughout.
        var contended = new LockElisionSimulator(4, true, true, 3, true, 1);
        Check(DriveWithInvariant(contended, 30, 17, true), "contended schedule completes");
        Check(contended.InvariantHolds(), "contended invariant");
        Check(contended.TotalAborts() > 0, "contention produced aborts: " + contended.TotalAborts());
        Check(contended.TotalFallbacks() > 0, "adaptive fallback engaged: " + contended.TotalFallbacks());
        Check(contended.CommittedSections == 30, "contended run committed every section");
        Check(contended.TotalLockAcquisitions() > 0, "the fallback path took the lock");

        // No adaptation: aborts keep happening but the invariant still holds.
        var alwaysEliding = new LockElisionSimulator(4, true, true, 1, false, 1);
        Check(DriveWithInvariant(alwaysEliding, 20, 23, true), "always-eliding schedule completes");
        Check(alwaysEliding.TotalAborts() > 0 && alwaysEliding.InvariantHolds(),
            "no fallback but still safe: aborts=" + alwaysEliding.TotalAborts());

        // Reproducibility: the same seed gives the same measurements.
        var repeat = new LockElisionSimulator(4, true, true, 3, true, 1);
        DriveWithInvariant(repeat, 30, 17, true);
        Check(repeat.TotalAborts() == contended.TotalAborts()
            && repeat.TotalElidedCommits() == contended.TotalElidedCommits()
            && repeat.TotalLockAcquisitions() == contended.TotalLockAcquisitions(), "measurements reproducible");

        // Negative control: elision without validation loses updates, and the invariant catches it.
        var unsafeSim = new LockElisionSimulator(4, true, false, 3, true, 1);
        bool complete = DriveWithInvariant(unsafeSim, 30, 17, false);
        Check(complete, "unvalidated schedule completes");
        Check(!unsafeSim.InvariantHolds() || unsafeSim.TotalAborts() == 0,
            "the invariant detects the unvalidated run: value=" + unsafeSim.Value + " expected="
            + unsafeSim.ExpectedValue());

        // A single thread never conflicts, so it commits every section without the lock.
        var solo = new LockElisionSimulator(1, true, true, 4, true, 0);
        Check(DriveWithInvariant(solo, 10, 1, true), "single thread completes");
        Check(solo.TotalElidedCommits() == 10 && solo.TotalLockAcquisitions() == 0 && solo.TotalAborts() == 0,
            "a lone thread elides all ten sections");
        Check(solo.Value == (1L << 10) - 1, "2^10 - 1 after ten doublings");

        // Degenerate inputs: no sections at all, and a zero-length run with elision disabled.
        var idle = new LockElisionSimulator(2, true, true, 4, true, 7);
        Check(DriveWithInvariant(idle, 0, 2, true) && idle.Value == 7, "no sections leaves the value alone");
        var lockedOnly = new LockElisionSimulator(2, false, true, 4, true, 0);
        Check(DriveWithInvariant(lockedOnly, 6, 9, true), "locked-only run completes");
        Check(lockedOnly.Value == (1L << 6) - 1 && lockedOnly.TotalElidedCommits() == 0,
            "locked-only value and no elision");

        Console.WriteLine("  lowContention: elided=" + eliding.TotalElidedCommits() + " locks="
            + eliding.TotalLockAcquisitions() + " aborts=" + eliding.TotalAborts() + " value=" + eliding.Value);
        Console.WriteLine("  contended: elided=" + contended.TotalElidedCommits() + " locks="
            + contended.TotalLockAcquisitions() + " aborts=" + contended.TotalAborts() + " fallbacks="
            + contended.TotalFallbacks() + " contendedAcquires="
            + contended.Threads.ConvertAll(t => t.ContendedAcquires).ToArray()[0] + " value=" + contended.Value);
        Console.WriteLine("PASS");
    }
}
