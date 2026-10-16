// Task 600 - Implement advanced checkpoint-restore for long-running computations (save state). (Tier: Advanced, CTCI Beyond — Software Engineering & Tooling)
//
// Checkpointing and restart for a long-running iterative computation, with the two properties that
// actually matter in production: the resumed run must be *bit-identical* to the uninterrupted one, and a
// damaged checkpoint must be detected instead of silently poisoning the computation. The payload is a
// hand-rolled little-endian format -- magic, version, iteration number, generator state, the length, the
// IEEE-754 bit patterns of the state vector and an FNV-1a checksum over everything preceding it -- written
// with explicit byte shifting rather than BinaryFormatter, so the bytes are platform independent and every
// field is validated before use: wrong magic, wrong version, a truncated buffer and a checksum mismatch are
// four separately tested failure modes, and `TryDeserialize` reports which one fired instead of throwing an
// unchecked exception. Redundancy is handled by a mirrored two-slot store in which restore falls back from
// slot A to slot B (and counts the recovery), so a single corrupted slot costs nothing while two corrupted
// slots fail cleanly. The computation being checkpointed is a deterministic 1-D heat solver with an
// explicit-Euler sweep plus a running statistic fed by an LCG stream, whose generator state therefore has to
// travel inside the checkpoint -- otherwise the resumed run would diverge even with identical lattices,
// which the file demonstrates by also checking the generator continuation. Beyond the in-memory round trip
// there is a real file round trip written to the current directory and removed in a `finally` block (the
// runner executes in its own working directory, so nothing is left behind, and the file asserts its own
// cleanup). All numbers come from the fixed-seed 64-bit LCG in `Rng` (MMIX constants, documented because the
// repo's 16-bit LCG has poor low-bit structure); no wall-clock time enters the state, so replay is exact.
//
// Documented checks: a run checkpointed every 10 of 200 iterations and rebuilt purely from the serialized
// bytes produces a lattice, statistic and generator state digitally identical to the uninterrupted run
// (compared as 64-bit patterns, not as floats), and so does the variant that resumes through the file;
// serializing a deserialized state reproduces the same bytes; the state size is constant across iterations
// (equal at iteration 10 and 190); restoring twice from one checkpoint yields identical continuations;
// flipping any one bit in any one byte of the payload is detected by the checksum, for all eight bit
// positions of a sampled byte and for a single-bit scan over every other byte; a corrupted version field
// is rejected as a version
// mismatch and a truncated payload as a length error, each with its own message; one corrupted slot is
// recovered from its mirror (the recovery counter increments) while two corrupted slots fail; a fault
// injected into the middle of the run is absorbed by rolling back to the last checkpoint and replaying with
// the fault disabled, still landing on the identical final state; and the generator continues from the
// restored state exactly as it would have without any interruption.
using System;
using System.Collections.Generic;
using System.IO;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state = state * 6364136223846793005UL + 1442695040888963407UL;
        return state >> 11;
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);

    internal static double NextUnit(ref ulong state) => NextBelow(ref state, 2001) / 1000.0 - 1.0;
}

internal static class Checkpointing
{
    internal const uint Magic = 0x314B4343u;
    internal const int CurrentVersion = 2;
    internal const int HeaderBytes = 4 + 4 + 8 + 8 + 4;

    internal sealed class State
    {
        internal long Iteration;
        internal ulong RngState;
        internal double[] Values = Array.Empty<double>();

        internal State Clone()
        {
            return new State { Iteration = Iteration, RngState = RngState, Values = (double[])Values.Clone() };
        }
    }

    internal sealed class Failure
    {
        internal string Kind = string.Empty;
        internal string Detail = string.Empty;
    }

    internal static ulong Fnv1a(IReadOnlyList<byte> bytes, int count)
    {
        ulong hash = 1469598103934665603UL;
        for (int i = 0; i < count; i++)
        {
            hash ^= bytes[i];
            hash *= 1099511628211UL;
        }
        return hash;
    }

    private static void WriteInt(List<byte> buffer, int value)
    {
        for (int i = 0; i < 4; i++) buffer.Add((byte)(value >> (8 * i)));
    }

    private static void WriteLong(List<byte> buffer, long value)
    {
        for (int i = 0; i < 8; i++) buffer.Add((byte)(value >> (8 * i)));
    }

    private static int ReadInt(IReadOnlyList<byte> buffer, int offset)
    {
        int value = 0;
        for (int i = 0; i < 4; i++) value |= buffer[offset + i] << (8 * i);
        return value;
    }

    private static long ReadLong(IReadOnlyList<byte> buffer, int offset)
    {
        long value = 0;
        for (int i = 0; i < 8; i++) value |= (long)buffer[offset + i] << (8 * i);
        return value;
    }

    internal static byte[] Serialize(State state, int version)
    {
        var buffer = new List<byte>();
        WriteInt(buffer, (int)Magic);
        WriteInt(buffer, version);
        WriteLong(buffer, state.Iteration);
        WriteLong(buffer, unchecked((long)state.RngState));
        WriteInt(buffer, state.Values.Length);
        foreach (double value in state.Values) WriteLong(buffer, BitConverter.DoubleToInt64Bits(value));
        ulong checksum = Fnv1a(buffer, buffer.Count);
        WriteLong(buffer, unchecked((long)checksum));
        return buffer.ToArray();
    }

    internal static bool TryDeserialize(byte[]? bytes, out State? state, out Failure? failure)
    {
        state = null;
        failure = null;
        if (bytes is null)
        {
            failure = new Failure { Kind = "missing", Detail = "no checkpoint bytes were supplied" };
            return false;
        }
        if (bytes.Length < HeaderBytes + 8)
        {
            failure = new Failure { Kind = "truncated", Detail = "buffer of " + bytes.Length + " bytes is shorter than the header" };
            return false;
        }
        if ((uint)ReadInt(bytes, 0) != Magic)
        {
            failure = new Failure { Kind = "magic", Detail = "unexpected magic value" };
            return false;
        }
        int version = ReadInt(bytes, 4);
        if (version != CurrentVersion)
        {
            failure = new Failure { Kind = "version", Detail = "checkpoint version " + version + " is not " + CurrentVersion };
            return false;
        }
        long iteration = ReadLong(bytes, 8);
        ulong rngState = unchecked((ulong)ReadLong(bytes, 16));
        int length = ReadInt(bytes, 24);
        if (length < 0)
        {
            failure = new Failure { Kind = "corrupt", Detail = "negative state length " + length };
            return false;
        }
        long expected = HeaderBytes + 8L * length + 8L;
        if (bytes.Length != expected)
        {
            failure = new Failure { Kind = "truncated", Detail = "expected " + expected + " bytes for " + length
                    + " values but found " + bytes.Length };
            return false;
        }
        ulong stored = unchecked((ulong)ReadLong(bytes, bytes.Length - 8));
        ulong computed = Fnv1a(bytes, bytes.Length - 8);
        if (stored != computed)
        {
            failure = new Failure { Kind = "checksum", Detail = "stored " + stored + " but the payload hashes to " + computed };
            return false;
        }
        var values = new double[length];
        for (int i = 0; i < length; i++) values[i] = BitConverter.Int64BitsToDouble(ReadLong(bytes, HeaderBytes + 8 * i));
        state = new State { Iteration = iteration, RngState = rngState, Values = values };
        return true;
    }

    /** Mirrored two-slot store with fallback recovery. */
    internal sealed class DualSlotStore
    {
        internal byte[]? SlotA;
        internal byte[]? SlotB;
        internal int Recoveries;
        internal int Failures;

        internal void Save(byte[] payload)
        {
            SlotA = (byte[])payload.Clone();
            SlotB = (byte[])payload.Clone();
        }

        internal bool Restore(out State? state, out string report)
        {
            state = null;
            if (TryDeserialize(SlotA, out State? fromA, out Failure? failureA))
            {
                state = fromA;
                report = "primary slot accepted";
                return true;
            }
            if (TryDeserialize(SlotB, out State? fromB, out Failure? failureB))
            {
                state = fromB;
                Recoveries++;
                report = "primary slot rejected (" + failureA?.Kind + "), recovered from the mirror";
                return true;
            }
            Failures++;
            report = "both slots rejected (" + failureA?.Kind + " and " + failureB?.Kind + ")";
            return false;
        }
    }
}

internal sealed class Simulation
{
    internal const int Lattice = 40;

    internal long Iteration;
    internal readonly double[] Values = new double[Lattice + 2];
    internal ulong GeneratorState;
    internal int InjectedFaultAt;
    internal bool FaultEnabled;

    internal Simulation(ulong seed)
    {
        GeneratorState = seed;
        for (int i = 0; i < Lattice; i++) Values[i] = i == 0 ? 1.0 : 0.0;
    }

    internal void SetState(Checkpointing.State state)
    {
        Iteration = state.Iteration;
        GeneratorState = state.RngState;
        Array.Copy(state.Values, Values, Values.Length);
    }

    internal Checkpointing.State GetState()
    {
        return new Checkpointing.State { Iteration = Iteration, RngState = GeneratorState, Values = (double[])Values.Clone() };
    }

    /** One explicit-Euler heat sweep plus a running statistic fed by the generator. */
    internal void Step()
    {
        var next = new double[Lattice];
        for (int i = 1; i < Lattice - 1; i++) next[i] = Values[i] + 0.2 * (Values[i - 1] - 2.0 * Values[i] + Values[i + 1]);
        next[0] = 1.0;
        next[Lattice - 1] = 0.0;
        for (int i = 0; i < Lattice; i++) Values[i] = next[i];
        double noise = Rng.NextUnit(ref GeneratorState);
        Values[Lattice] += noise;
        Values[Lattice + 1] += noise * noise;
        Iteration++;
    }

    /** A step that throws before mutating anything, to exercise rollback and replay. */
    internal void StepWithFault()
    {
        if (FaultEnabled && Iteration == InjectedFaultAt)
        {
            throw new InvalidOperationException("injected fault before iteration " + Iteration);
        }
        Step();
    }
}

public static class Solution
{
    private const string CheckpointFile = "checkpoint-demo.bin";

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckClose(double actual, double expected, double tolerance, string label)
    {
        Check(Math.Abs(actual - expected) <= tolerance, label + " (got " + actual + ", expected " + expected + ")");
    }

    private static bool Identical(Simulation left, Simulation right)
    {
        if (left.Iteration != right.Iteration || left.GeneratorState != right.GeneratorState) return false;
        for (int i = 0; i < left.Values.Length; i++)
        {
            if (BitConverter.DoubleToInt64Bits(left.Values[i]) != BitConverter.DoubleToInt64Bits(right.Values[i])) return false;
        }
        return true;
    }

    private static void Run(Simulation simulation, int steps)
    {
        for (int i = 0; i < steps; i++) simulation.Step();
    }

    /** Interrupted run: checkpoint every `interval` iterations through a mirrored store, then resume. */
    private static Simulation RunWithCheckpoints(ulong seed, int steps, int interval, out int saves,
            out int recoveries, out int sizeAtTen, out int sizeAtLate)
    {
        var store = new Checkpointing.DualSlotStore();
        var current = new Simulation(seed);
        saves = 0;
        recoveries = 0;
        sizeAtTen = 0;
        sizeAtLate = 0;
        int done = 0;
        while (done < steps)
        {
            int block = Math.Min(interval, steps - done);
            Run(current, block);
            done += block;
            byte[] payload = Checkpointing.Serialize(current.GetState(), Checkpointing.CurrentVersion);
            store.Save(payload);
            saves++;
            if (done == 10) sizeAtTen = payload.Length;
            if (done == 190) sizeAtLate = payload.Length;
            // Rebuild the worker purely from the bytes, as a restart after a crash would.
            Check(store.Restore(out Checkpointing.State? restored, out _), "the store restores a fresh checkpoint");
            var rebuilt = new Simulation(seed);
            rebuilt.SetState(restored!);
            current = rebuilt;
        }
        recoveries = store.Recoveries;
        return current;
    }

    public static void Main()
    {
        const int Steps = 200;
        const int Interval = 10;
        const ulong Seed = 600001UL;

        // ---- uninterrupted reference run ---------------------------------------------------
        var uninterrupted = new Simulation(Seed);
        Run(uninterrupted, Steps);
        Check(uninterrupted.Iteration == Steps, "the reference run performed 200 iterations");

        // ---- checkpointed run, rebuilt from bytes only -------------------------------------
        Simulation checkpointed = RunWithCheckpoints(Seed, Steps, Interval, out int saves, out int recoveries,
                out int sizeAtTen, out int sizeAtLate);
        Check(saves == Steps / Interval, "the run saved one checkpoint per interval (" + saves + ", expected " + (Steps / Interval) + ")");
        Check(Identical(uninterrupted, checkpointed),
                "the checkpointed run is bit-identical to the uninterrupted one");
        Check(sizeAtTen == sizeAtLate && sizeAtTen > 0,
                "the state size is constant across the run (" + sizeAtTen + " bytes at iteration 10 and "
                + sizeAtLate + " at 190)");
        Check(recoveries == 0, "the healthy run needed no slot recovery");

        // ---- serialization round trip ------------------------------------------------------
        byte[] payload = Checkpointing.Serialize(uninterrupted.GetState(), Checkpointing.CurrentVersion);
        Check(Checkpointing.TryDeserialize(payload, out Checkpointing.State? state, out _), "the reference checkpoint parses");
        byte[] again = Checkpointing.Serialize(state!, Checkpointing.CurrentVersion);
        Check(again.Length == payload.Length, "the round trip preserves the length");
        bool byteIdentical = true;
        for (int i = 0; i < payload.Length; i++) if (payload[i] != again[i]) byteIdentical = false;
        Check(byteIdentical, "serialize(deserialize(bytes)) reproduces the same bytes");
        CheckClose(state!.Iteration, uninterrupted.Iteration, 0.0, "the iteration number survives the round trip");
        Check(state.RngState == uninterrupted.GeneratorState, "the generator state survives the round trip");
        for (int i = 0; i < state.Values.Length; i++)
        {
            CheckClose(state.Values[i], uninterrupted.Values[i], 0.0, "state value " + i + " survives exactly");
        }
        Check(payload.Length == Checkpointing.HeaderBytes + 8 * uninterrupted.Values.Length + 8,
                "the payload length matches the documented layout (" + payload.Length + " bytes)");

        // ---- corruption detection -----------------------------------------------------------
        int checkedBits = 0;
        for (int bit = 0; bit < 8; bit++)
        {
            byte[] damaged = (byte[])payload.Clone();
            damaged[Checkpointing.HeaderBytes + 3] ^= (byte)(1 << bit);
            Check(!Checkpointing.TryDeserialize(damaged, out _, out Checkpointing.Failure? failure), "corruption is detected");
            Check(failure!.Kind == "checksum", "the corrupted payload reports a checksum failure (bit " + bit + ")");
            checkedBits++;
        }
        Check(checkedBits == 8, "all eight bit positions of the sampled byte were probed");
        int detected = 0;
        for (int offset = 0; offset < payload.Length - 8; offset++)
        {
            byte[] damaged = (byte[])payload.Clone();
            damaged[offset] ^= 0x01;
            if (!Checkpointing.TryDeserialize(damaged, out _, out _)) detected++;
        }
        Check(detected == payload.Length - 8, "every single-byte flip outside the checksum itself is detected ("
                + detected + " of " + (payload.Length - 8) + ")");
        byte[] wrongVersion = Checkpointing.Serialize(uninterrupted.GetState(), Checkpointing.CurrentVersion + 1);
        Check(!Checkpointing.TryDeserialize(wrongVersion, out _, out Checkpointing.Failure? versionFailure),
                "a newer version field is rejected");
        Check(versionFailure!.Kind == "version", "the rejection is reported as a version mismatch");
        var truncated = new byte[payload.Length - 16];
        Array.Copy(payload, truncated, truncated.Length);
        Check(!Checkpointing.TryDeserialize(truncated, out _, out Checkpointing.Failure? truncatedFailure),
                "a truncated payload is rejected");
        Check(truncatedFailure!.Kind == "truncated", "the rejection is reported as truncation");
        Check(!Checkpointing.TryDeserialize(null, out _, out Checkpointing.Failure? missingFailure),
                "a missing checkpoint is rejected");
        Check(missingFailure!.Kind == "missing", "the rejection is reported as missing");
        byte[] wrongMagic = (byte[])payload.Clone();
        wrongMagic[0] ^= 0xFF;
        Check(!Checkpointing.TryDeserialize(wrongMagic, out _, out Checkpointing.Failure? magicFailure),
                "a wrong magic value is rejected");
        Check(magicFailure!.Kind == "magic", "the rejection is reported as a magic mismatch");

        // ---- mirrored slots -----------------------------------------------------------------
        var store = new Checkpointing.DualSlotStore();
        store.Save(payload);
        store.SlotA![5] ^= 0x20;
        Check(store.Restore(out Checkpointing.State? recovered, out string report), "the mirror rescues a damaged primary");
        Check(report.Contains("mirror"), "the recovery is reported (" + report + ")");
        Check(store.Recoveries == 1, "the recovery counter incremented");
        Check(recovered!.Iteration == uninterrupted.Iteration, "the recovered state is the right one");
        store.SlotB![7] ^= 0x10;
        Check(!store.Restore(out _, out string bothReport), "two damaged slots fail");
        Check(store.Failures == 1, "the failure counter incremented");
        Check(bothReport.Contains("both"), "the double failure is reported (" + bothReport + ")");

        // ---- double resume is idempotent -----------------------------------------------------
        var firstResume = new Simulation(Seed);
        firstResume.SetState(state!);
        var secondResume = new Simulation(Seed);
        secondResume.SetState(state!);
        Run(firstResume, 30);
        Run(secondResume, 30);
        Check(Identical(firstResume, secondResume), "restoring the same checkpoint twice gives the same continuation");
        var continued = new Simulation(Seed);
        continued.SetState(state!);
        Run(continued, 30);
        Check(Identical(continued, firstResume), "the continuation is deterministic");

        // ---- generator continuation -----------------------------------------------------------
        {
            var withRestore = new Simulation(Seed);
            Run(withRestore, 100);
            byte[] midPayload = Checkpointing.Serialize(withRestore.GetState(), Checkpointing.CurrentVersion);
            var fromBytes = new Simulation(Seed);
            Check(Checkpointing.TryDeserialize(midPayload, out Checkpointing.State? midState, out _), "the mid-run checkpoint parses");
            fromBytes.SetState(midState!);
            ulong firstGenerator = withRestore.GeneratorState;
            ulong secondGenerator = fromBytes.GeneratorState;
            Check(firstGenerator == secondGenerator, "the restored generator state matches the original");
            for (int i = 0; i < 50; i++)
            {
                double left = Rng.NextUnit(ref firstGenerator);
                double right = Rng.NextUnit(ref secondGenerator);
                CheckClose(left, right, 0.0, "the generator stream continues identically at draw " + i);
            }
        }

        // ---- fault injection with rollback and replay -----------------------------------------
        {
            var store2 = new Checkpointing.DualSlotStore();
            var worker = new Simulation(Seed);
            worker.FaultEnabled = true;
            worker.InjectedFaultAt = 75;
            int completed = 0;
            bool faultHit = false;
            while (completed < Steps)
            {
                try
                {
                    int block = Math.Min(Interval, Steps - completed);
                    for (int i = 0; i < block; i++)
                    {
                        worker.StepWithFault();
                        completed++;
                    }
                    store2.Save(Checkpointing.Serialize(worker.GetState(), Checkpointing.CurrentVersion));
                }
                catch (InvalidOperationException)
                {
                    faultHit = true;
                    Check(store2.Restore(out Checkpointing.State? rolled, out _), "the rollback restores the last checkpoint");
                    worker = new Simulation(Seed);
                    worker.SetState(rolled!);
                    worker.FaultEnabled = false;
                    // A mid-block fault: the checkpoint at 70 is restored and the five steps between 70 and
                    // the fault at 75 are replayed, which is what makes this a real rollback.
                    completed = (int)worker.Iteration;
                    Check(completed == 70, "the rollback landed on the last checkpoint before the fault (got "
                            + completed + ")");
                }
            }
            Check(faultHit, "the injected fault really fired");
            Check(Identical(uninterrupted, worker), "the faulted run converges to the identical final state");
        }

        // ---- a real file round trip -----------------------------------------------------------
        string path = Path.Combine(Directory.GetCurrentDirectory(), CheckpointFile);
        try
        {
            var fileRun = new Simulation(Seed);
            Run(fileRun, 75);
            File.WriteAllBytes(path, Checkpointing.Serialize(fileRun.GetState(), Checkpointing.CurrentVersion));
            Check(File.Exists(path), "the checkpoint file was written");
            byte[] fromDisk = File.ReadAllBytes(path);
            Check(Checkpointing.TryDeserialize(fromDisk, out Checkpointing.State? diskState, out _), "the file checkpoint parses");
            var resumed = new Simulation(Seed);
            resumed.SetState(diskState!);
            Run(resumed, Steps - 75);
            Check(Identical(uninterrupted, resumed), "the file-based resume is bit-identical too");
            File.WriteAllBytes(path, new byte[] { 1, 2, 3 });
            Check(!Checkpointing.TryDeserialize(File.ReadAllBytes(path), out _, out Checkpointing.Failure? fileFailure),
                    "a clobbered file is rejected");
            Check(fileFailure!.Kind == "truncated", "the clobbered file is reported as truncated");
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
        Check(!File.Exists(path), "the demo cleans up its own checkpoint file");

        Console.WriteLine("PASS");
    }
}
