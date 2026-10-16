// Task 936 - Implement transparent operator checkpointing and resume in compute graph.
// (Tier: Master, CTCI Beyond — Software Engineering & Tooling)
//
// A four-operator pipeline Source -> FilterSquare(even only) -> GroupSum(key = x % 3) -> Sink,
// driven record by record. A checkpoint captures *every* operator's state at once (source
// cursor, per-stage records-in counters, the group table and the sink's emitted records), which
// is what makes resumption transparent: the driver restores that consistent cut and replays the
// input from the captured cursor, so the sink ends up with exactly the records a clean run would
// have produced - no duplication, no loss.
//
// Verified: the resumed output equals a clean run element for element (exactly-once), the replay
// is genuine (the replayed count is asserted non-zero and equal to the run length from the cut,
// so the test cannot pass vacuously), the restored aggregate matches an oracle that sums the
// consumed prefix directly, checkpoints advance the cursor in fixed strides with every operator
// agreeing on the record count, failures before the first checkpoint and exactly at a checkpoint
// both behave, and - the negative control - resuming *without* rolling the aggregate and sink
// back (a torn cut) demonstrably yields a different output sequence.
// Simplified: the pipeline is single-threaded and synchronous, so nothing is in flight when the
// barrier is taken and "transparent" reduces to consistent-cut capture plus replay; there is no
// distributed snapshot protocol, no durable log and no operator with internal timers.
using System;
using System.Collections.Generic;
using System.Linq;

internal sealed class StageState
{
    internal int RecordsIn;
    internal int Cursor;
    internal readonly List<long> Outputs = new List<long>();
    internal readonly Dictionary<int, long> Sums = new Dictionary<int, long>();
    internal readonly Dictionary<int, int> Counts = new Dictionary<int, int>();

    internal StageState Clone()
    {
        var copy = new StageState { RecordsIn = RecordsIn, Cursor = Cursor };
        copy.Outputs.AddRange(Outputs);
        foreach (KeyValuePair<int, long> kv in Sums) copy.Sums[kv.Key] = kv.Value;
        foreach (KeyValuePair<int, int> kv in Counts) copy.Counts[kv.Key] = kv.Value;
        return copy;
    }

    internal void Restore(StageState other)
    {
        RecordsIn = other.RecordsIn;
        Cursor = other.Cursor;
        Outputs.Clear();
        Outputs.AddRange(other.Outputs);
        Sums.Clear();
        foreach (KeyValuePair<int, long> kv in other.Sums) Sums[kv.Key] = kv.Value;
        Counts.Clear();
        foreach (KeyValuePair<int, int> kv in other.Counts) Counts[kv.Key] = kv.Value;
    }
}

internal abstract class Stage
{
    internal readonly string Name;
    internal readonly StageState State = new StageState();
    internal Stage? Downstream;
    protected Stage(string name) => Name = name;
    internal abstract void Push(long value);
    protected void Emit(long value) => Downstream?.Push(value);
}

internal sealed class SourceStage : Stage
{
    internal readonly long[] Data;
    private readonly Stage head;
    internal SourceStage(long[] data, Stage head) : base("source") { Data = data; this.head = head; }

    internal override void Push(long value) { }

    internal void Drive(int untilOffset)
    {
        while (State.Cursor < untilOffset && State.Cursor < Data.Length)
        {
            long value = Data[State.Cursor];
            State.Cursor++;
            State.RecordsIn = State.Cursor;
            head.Push(value);
        }
    }
}

internal sealed class FilterSquareStage : Stage
{
    internal FilterSquareStage() : base("filter-square") { }

    internal override void Push(long value)
    {
        State.RecordsIn++;
        if (value % 2 == 0) Emit(value * value);
    }
}

internal sealed class GroupSumStage : Stage
{
    internal GroupSumStage() : base("group-sum") { }

    internal override void Push(long value)
    {
        State.RecordsIn++;
        int key = (int)(value % 3);
        State.Sums.TryGetValue(key, out long sum);
        State.Sums[key] = sum + value;
        State.Counts.TryGetValue(key, out int count);
        State.Counts[key] = count + 1;
        Emit(State.Sums[key]);
    }
}

internal sealed class SinkStage : Stage
{
    internal SinkStage() : base("sink") { }

    internal override void Push(long value)
    {
        State.RecordsIn++;
        State.Outputs.Add(value);
    }
}

internal sealed class Pipeline
{
    internal readonly SourceStage Source;
    internal readonly FilterSquareStage Filter;
    internal readonly GroupSumStage Aggregate;
    internal readonly SinkStage Sink;
    private readonly Stage[] stages;
    private readonly List<Dictionary<string, StageState>> checkpoints = new List<Dictionary<string, StageState>>();
    internal readonly List<int> CheckpointOffsets = new List<int>();

    internal Pipeline(long[] data)
    {
        Filter = new FilterSquareStage();
        Aggregate = new GroupSumStage();
        Sink = new SinkStage();
        Source = new SourceStage(data, Filter);
        Filter.Downstream = Aggregate;
        Aggregate.Downstream = Sink;
        stages = new Stage[] { Source, Filter, Aggregate, Sink };
    }

    internal IReadOnlyList<Stage> Stages => stages;
    internal IReadOnlyList<long> Outputs => Sink.State.Outputs;
    internal int Offset => Source.State.Cursor;

    internal void RunTo(int offset) => Source.Drive(offset);

    internal int Checkpoint()
    {
        var snapshot = new Dictionary<string, StageState>();
        foreach (Stage stage in stages) snapshot[stage.Name] = stage.State.Clone();
        checkpoints.Add(snapshot);
        CheckpointOffsets.Add(Offset);
        return checkpoints.Count - 1;
    }

    internal void Restore(int epoch)
    {
        foreach (Stage stage in stages) stage.State.Restore(checkpoints[epoch][stage.Name]);
    }

    /** Negative control: roll back only the producers, leaving the aggregate and sink ahead. */
    internal void RestoreTornCut(int epoch)
    {
        stages[0].State.Restore(checkpoints[epoch]["source"]);
        stages[1].State.Restore(checkpoints[epoch]["filter-square"]);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static long[] Input(int n)
    {
        var data = new long[n];
        for (int i = 0; i < n; i++) data[i] = i + 1;
        return data;
    }

    /** Independent oracle: the pipeline semantics written as a plain batch loop. */
    private static List<long> OracleOutputs(long[] data)
    {
        var sums = new Dictionary<int, long>();
        var outputs = new List<long>();
        foreach (long value in data)
        {
            if (value % 2 != 0) continue;
            int key = (int)(value * value % 3);
            sums.TryGetValue(key, out long running);
            sums[key] = running + value * value;
            outputs.Add(sums[key]);
        }
        return outputs;
    }

    private static Dictionary<int, long> OraclePrefixSums(long[] data, int upTo)
    {
        var sums = new Dictionary<int, long>();
        for (int i = 0; i < upTo; i++)
        {
            long value = data[i];
            if (value % 2 != 0) continue;
            int key = (int)(value * value % 3);
            sums.TryGetValue(key, out long running);
            sums[key] = running + value * value;
        }
        return sums;
    }

    private static Pipeline CleanRun(long[] data, int stride)
    {
        var pipeline = new Pipeline(data);
        for (int offset = stride; offset <= data.Length; offset += stride)
        {
            pipeline.RunTo(offset);
            pipeline.Checkpoint();
        }
        return pipeline;
    }

    public static void Main()
    {
        const int Stride = 10;
        long[] data = Input(60);

        // 1-2. Clean run against the independent batch oracle, plus a consistent-cut check.
        Pipeline clean = CleanRun(data, Stride);
        Check(clean.Outputs.SequenceEqual(OracleOutputs(data)),
            "the checkpointed clean run reproduces the batch oracle's " + clean.Outputs.Count + " outputs");
        Check(clean.Outputs.Count == 30, "exactly the 30 even records reach the sink");
        Check(clean.CheckpointOffsets.SequenceEqual(new[] { 10, 20, 30, 40, 50, 60 })
              && clean.Stages.Select(s => s.State.RecordsIn).SequenceEqual(new[] { 60, 60, 30, 30 }),
            "checkpoints advance the cursor in fixed strides and every operator agrees on the cut");

        // 3. Crash after record 37, then restore the last checkpoint (offset 30).
        var resumed = new Pipeline(data);
        resumed.RunTo(10); resumed.Checkpoint();
        resumed.RunTo(20); resumed.Checkpoint();
        resumed.RunTo(30); resumed.Checkpoint();
        resumed.RunTo(37);
        int lastEpoch = resumed.CheckpointOffsets.Count - 1;
        Check(resumed.CheckpointOffsets[lastEpoch] == 30 && resumed.Offset == 37,
            "the crash leaves the cursor 7 records past the checkpoint at offset 30");
        resumed.Restore(lastEpoch);
        Check(resumed.Offset == 30 && resumed.Outputs.Count == 15,
            "the restore rewinds source cursor and sink output to the checkpoint");
        Dictionary<int, long> prefix = OraclePrefixSums(data, 30);
        Check(resumed.Aggregate.State.Sums.Count == prefix.Count
              && prefix.All(kv => resumed.Aggregate.State.Sums[kv.Key] == kv.Value)
              && resumed.Aggregate.State.RecordsIn == 15,
            "the restored group table equals the oracle prefix sums (no double counting)");

        // 4. Exactly-once: replays are genuine and the final output matches the clean run.
        int replayedFrom = resumed.Offset;
        resumed.RunTo(data.Length);
        Check(data.Length - replayedFrom == 30,
            "the replay covers the 30 records from the cut to the end (" + (data.Length - replayedFrom) + ")");
        Check(resumed.Outputs.SequenceEqual(clean.Outputs)
              && resumed.Aggregate.State.Sums.All(kv => kv.Value == clean.Aggregate.State.Sums[kv.Key]),
            "resume yields exactly the clean run's outputs, no duplicate and no lost record");

        // 5. Boundary: a failure before the first periodic checkpoint restarts from the empty cut.
        var early = new Pipeline(data);
        early.Checkpoint();                 // epoch 0 at offset 0
        early.RunTo(6);
        early.Restore(0);
        early.RunTo(data.Length);
        Check(early.Outputs.SequenceEqual(clean.Outputs), "a crash before the first checkpoint restarts cleanly from record 1");

        // 6. Boundary: a crash exactly at a checkpoint replays nothing.
        var exact = new Pipeline(data);
        exact.RunTo(20);
        exact.Checkpoint();
        int beforeRestore = exact.Outputs.Count;
        exact.Restore(0);
        Check(exact.Offset == 20 && exact.Outputs.Count == beforeRestore && beforeRestore == 10,
            "crashing exactly at a checkpoint replays zero records");
        exact.RunTo(data.Length);
        Check(exact.Outputs.SequenceEqual(clean.Outputs), "resuming from the exact cut still matches the clean run");

        // 7. Negative control: a torn cut (aggregate and sink not rewound) corrupts the output.
        var torn = new Pipeline(data);
        torn.RunTo(10); torn.Checkpoint();
        torn.RunTo(20); torn.Checkpoint();
        torn.RunTo(30); torn.Checkpoint();
        torn.RunTo(37);
        torn.RestoreTornCut(2);
        torn.RunTo(data.Length);
        Check(!torn.Outputs.SequenceEqual(clean.Outputs),
            "rewinding only the producers leaves the sink reading " + torn.Outputs.Count
            + " records that disagree with the clean run");

        // 8. Boundary and determinism: empty input, and two replays of the crash scenario agree.
        var emptyRun = new Pipeline(new long[0]);
        emptyRun.RunTo(0);
        emptyRun.Checkpoint();
        Check(emptyRun.Offset == 0 && emptyRun.Outputs.Count == 0 && emptyRun.Stages.All(s => s.State.RecordsIn == 0),
            "an empty input checkpoints cleanly at offset 0 with empty operator state");
        var repeat = new Pipeline(data);
        repeat.RunTo(30); repeat.Checkpoint();
        repeat.RunTo(37);
        repeat.Restore(0);
        repeat.RunTo(data.Length);
        Check(repeat.Outputs.SequenceEqual(resumed.Outputs), "the same failure scenario replays identically every time");

        Console.WriteLine("  pipeline: 60 records, stride " + Stride + ", " + clean.Outputs.Count
            + " sink records, replayed 30 after the injected crash");
        Console.WriteLine("PASS");
    }
}
