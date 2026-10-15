// Task 480 - Implement write-ahead logging (WAL) + crash recovery simulation. (Tier: Advanced, CTCI Ch 14 — Databases)
using System;
using System.Collections.Generic;
using System.Text;

internal class RecoveryResult
{
    public readonly Dictionary<string, int> State;
    public readonly int Replayed;

    public RecoveryResult(Dictionary<string, int> state, int replayed)
    {
        State = state;
        Replayed = replayed;
    }
}

internal class WalEngine
{
    public Dictionary<string, int> State = new Dictionary<string, int>();
    private List<object[]> wal = new List<object[]>(); // {lsn, op, key, value}
    public int Lsn;
    public int DurableLsn;
    private Dictionary<string, int> checkpointState;
    private int checkpointLsn;
    public int CheckpointCount;
    public int CrashCount;
    public int LastReplayCount;

    public void Set(string key, int value)
    {
        Lsn++;
        wal.Add(new object[] { Lsn, "set", key, value });
        State[key] = value;
    }

    public void Delete(string key)
    {
        Lsn++;
        wal.Add(new object[] { Lsn, "del", key, null });
        State.Remove(key);
    }

    public void Flush() => DurableLsn = Lsn;

    public void CheckpointNow()
    {
        Flush();
        checkpointState = new Dictionary<string, int>(State);
        checkpointLsn = Lsn;
        CheckpointCount++;
    }

    public string WalText(string tornTail)
    {
        var sb = new StringBuilder();
        foreach (object[] record in wal)
        {
            int recordLsn = (int)record[0];
            string op = (string)record[1];
            string key = (string)record[2];
            int? value = (int?)record[3];
            sb.Append(recordLsn).Append('|').Append(op).Append('|').Append(key).Append('|');
            if (value.HasValue) sb.Append(value.Value);
            sb.Append('\n');
        }
        if (tornTail != null) sb.Append(tornTail);
        return sb.ToString();
    }

    public Dictionary<string, int> Crash() => Crash(null);

    public Dictionary<string, int> Crash(string tornTail)
    {
        string onDisk = WalText(tornTail);
        int durable = DurableLsn;
        RecoveryResult result = Recover(onDisk, durable, checkpointState, checkpointLsn);
        State = result.State;
        LastReplayCount = result.Replayed;
        Lsn = durable;
        wal = wal.GetRange(0, Math.Min(durable, wal.Count)); // truncate past watermark
        DurableLsn = durable;
        CrashCount++;
        return State;
    }

    public static RecoveryResult Recover(string walText, int durableLsn, Dictionary<string, int> ckptState, int ckptLsn)
    {
        var state = ckptState == null ? new Dictionary<string, int>() : new Dictionary<string, int>(ckptState);
        int snapshotLsn = ckptState == null ? 0 : ckptLsn;
        int replayed = 0;
        foreach (string line in walText.Split('\n'))
        {
            if (line.Length == 0) continue;
            string[] parts = line.Split('|');
            if (parts.Length != 4) break; // torn record: stop, drop it
            int recordLsn = int.Parse(parts[0]);
            if (recordLsn <= snapshotLsn) continue;
            if (recordLsn > durableLsn) break; // never made durable: lost
            if (parts[1] == "set") state[parts[2]] = int.Parse(parts[3]);
            else state.Remove(parts[2]);
            replayed++;
        }
        return new RecoveryResult(state, replayed);
    }
}

public static class Solution
{
    private static int Lcg(int seed) => (seed * 75 + 74) % 65537;

    private static bool SameMap(Dictionary<string, int> a, Dictionary<string, int> b)
    {
        if (a.Count != b.Count) return false;
        foreach (var pair in a)
        {
            if (!b.TryGetValue(pair.Key, out int other) || other != pair.Value) return false;
        }
        return true;
    }

    private static int[] RunRandom()
    {
        var eng = new WalEngine();
        var oracleLive = new Dictionary<string, int>();
        var oracleDurable = new Dictionary<string, int>();
        int crashes = 0;
        int seed = 8642;
        for (int i = 0; i < 60; i++)
        {
            seed = Lcg(seed);
            int action = seed % 10;
            seed = Lcg(seed);
            string key = "k" + seed % 12;
            seed = Lcg(seed);
            int value = seed % 1000;
            if (action < 5)
            {
                eng.Set(key, value);
                oracleLive[key] = value;
            }
            else if (action < 7)
            {
                eng.Delete(key);
                oracleLive.Remove(key);
            }
            else if (action == 7)
            {
                eng.Flush();
                oracleDurable = new Dictionary<string, int>(oracleLive);
            }
            else if (action == 8)
            {
                eng.CheckpointNow();
                oracleDurable = new Dictionary<string, int>(oracleLive);
            }
            else
            {
                crashes++;
                eng.Crash();
                Check(SameMap(eng.State, oracleDurable), "random crash " + i + " recovered the durable prefix");
                oracleLive = new Dictionary<string, int>(oracleDurable);
            }
        }
        Check(crashes >= 1, "random workload crashed");
        Check(eng.CheckpointCount >= 1, "random workload checkpointed");
        eng.CheckpointNow();
        oracleDurable = new Dictionary<string, int>(oracleLive);
        Check(SameMap(eng.State, oracleLive) && SameMap(eng.State, oracleDurable), "settled state");
        eng.Crash();
        Check(SameMap(eng.State, oracleLive), "a crash with nothing in flight is idempotent");
        return new[] { crashes, eng.CheckpointCount };
    }

    public static void Main()
    {
        // Direct parser checks: replay stops at the torn record, watermark and snapshot respected.
        string text = "1|set|a|1\n2|set|b|2\n3|del|a|\n4|set|c|3\n5|set|d";
        RecoveryResult r1 = WalEngine.Recover(text, 4, null, 0);
        Check(SameMap(r1.State, Map("b", 2, "c", 3)) && r1.Replayed == 4, "parser watermark");
        RecoveryResult r2 = WalEngine.Recover(text, 2, null, 0);
        Check(SameMap(r2.State, Map("a", 1, "b", 2)) && r2.Replayed == 2, "earlier watermark");
        RecoveryResult r3 = WalEngine.Recover(text, 5, Map("z", 9), 3);
        Check(SameMap(r3.State, Map("z", 9, "c", 3)) && r3.Replayed == 1, "snapshot skips LSNs <= 3");

        // Checkpoint + durable suffix replay, then a torn tail beyond the watermark.
        var eng = new WalEngine();
        for (int i = 1; i <= 5; i++) eng.Set("k" + i, i);
        Check(eng.Lsn == 5 && eng.DurableLsn == 0, "writes buffered");
        eng.CheckpointNow();
        Check(eng.DurableLsn == 5 && eng.CheckpointCount == 1, "checkpoint implies durability");
        for (int i = 6; i <= 10; i++) eng.Set("k" + i, i);
        eng.Flush();
        eng.Set("k11", 11);
        eng.Delete("k3");
        eng.Set("k12", 12);
        Check(eng.Lsn == 13, "three in-flight ops");
        eng.Crash("14|set|k13");
        var expected = new Dictionary<string, int>();
        for (int i = 1; i <= 10; i++) expected["k" + i] = i;
        Check(SameMap(eng.State, expected), "recovered = committed prefix k1..k10");
        Check(eng.LastReplayCount == 5, "only the five post-checkpoint durable records replay");
        Check(eng.Lsn == 10 && eng.DurableLsn == 10, "watermark restored");

        // The recovered engine keeps logging after the watermark.
        eng.Set("k100", 100);
        Check(eng.Lsn == 11, "lsn continues");
        eng.Flush();
        eng.Crash();
        expected["k100"] = 100;
        Check(SameMap(eng.State, expected), "second recovery");
        Check(eng.LastReplayCount == 6, "six replayed records");

        // Unflushed writes are lost: only the committed prefix survives.
        var eng2 = new WalEngine();
        eng2.Set("x", 1);
        eng2.Flush();
        eng2.Set("x", 2);
        eng2.Crash();
        Check(SameMap(eng2.State, Map("x", 1)), "unflushed write lost");

        // Recovery from a bare WAL without any checkpoint.
        var eng3 = new WalEngine();
        for (int i = 0; i < 4; i++) eng3.Set("a" + i, i);
        eng3.Flush();
        eng3.Crash();
        Check(SameMap(eng3.State, Map("a0", 0, "a1", 1, "a2", 2, "a3", 3)), "bare WAL replay");
        Check(eng3.LastReplayCount == 4, "bare WAL replayed all four");

        int[] random = RunRandom();
        Check(random[0] == 9, "random crash count (a==9 rolls at 5,9,15,24,27,32,47,51,55)");
        Check(random[1] == 11, "ten a==8 checkpoints plus the final settle checkpoint");
        Console.WriteLine("PASS");
    }

    private static Dictionary<string, int> Map(params object[] pairs)
    {
        var output = new Dictionary<string, int>();
        for (int i = 0; i < pairs.Length; i += 2) output[(string)pairs[i]] = (int)pairs[i + 1];
        return output;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
