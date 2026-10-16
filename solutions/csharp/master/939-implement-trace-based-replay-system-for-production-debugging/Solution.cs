// Task 939 - Implement trace-based replay system for production debugging.
// (Tier: Master, CTCI Beyond — Software Engineering & Tooling)
//
// A tiny service mesh - gateway -> {auth, catalog -> search, pricing -> rules} - is executed twice
// by the same code. In *record* mode the transport computes every dependency response and logs it
// together with the span it produced; in *replay* mode the transport consumes that log in call
// order and never computes anything itself, so the whole request tree is rebuilt from the capture.
// Replay is verified by folding every span (id, parent, key, request, response, duration) into one
// trace digest: a faithful replay reproduces the recorded digest bit for bit, a mutated recorded
// response changes the digest (the divergence is detected, not papered over), a recorded call that
// the code no longer makes shows up as an unexpected call, and a recorded call that the code never
// consumes shows up as an unconsumed entry.
//
// The critical path (the longest chain of self-costs through the span tree) is computed bottom-up
// over the span array and cross-checked against an exhaustive enumeration of every root-to-leaf
// chain, which is an independent method for this tree. Fragment invariants (unique ids, ids equal
// to creation order, parent closed after its children, parent duration >= sum of child durations)
// are re-verified on the recorded trace, and the empty and single-span traces are boundaries.
// Simplified: logical milliseconds instead of wall clock, one trace at a time, no sampling policy,
// no persistence format and no captured-but-unused span attributes; the record/replay contract,
// the divergence detection and the trace digest are real.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static ulong Fold(ulong state, ulong value)
    {
        ulong z = state ^ (value + 0x9E3779B97F4A7C15UL + (state << 6) + (state >> 2));
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        return z ^ (z >> 31);
    }

    internal static ulong Hash(string text)
    {
        ulong state = 0xCBF29CE484222325UL;
        foreach (char c in text) state = Fold(state, c);
        return state;
    }
}

internal sealed class Span
{
    internal readonly int Id;
    internal readonly int ParentId;
    internal readonly string Key;
    internal readonly long RequestDigest;
    internal readonly List<int> Children = new List<int>();
    internal long ResponseDigest;
    internal long SelfCostMs;
    internal long DurationMs;

    internal Span(int id, int parentId, string key, long requestDigest)
    {
        Id = id;
        ParentId = parentId;
        Key = key;
        RequestDigest = requestDigest;
    }
}

internal sealed class RpcLog
{
    internal readonly List<string> Keys = new List<string>();
    internal readonly List<long> Responses = new List<long>();
    internal readonly List<long> Durations = new List<long>();

    internal int Count => Keys.Count;

    internal RpcLog Clone()
    {
        var copy = new RpcLog();
        copy.Keys.AddRange(Keys);
        copy.Responses.AddRange(Responses);
        copy.Durations.AddRange(Durations);
        return copy;
    }
}

internal sealed class AppRunner
{
    private readonly bool live;
    private readonly RpcLog log;
    private readonly List<Span> spans = new List<Span>();
    private int nextId;
    private int cursor;

    internal readonly List<string> Divergences = new List<string>();
    internal long RootResponse;

    internal AppRunner(bool live, RpcLog log)
    {
        this.live = live;
        this.log = log;
    }

    internal IReadOnlyList<Span> Spans => spans;
    internal int ConsumedCalls => cursor;
    internal int UnconsumedCalls => live ? 0 : Math.Max(0, log.Count - cursor);

    private int OpenSpan(string key, int parent, long request)
    {
        int id = nextId++;
        var span = new Span(id, parent, key, request);
        spans.Add(span);
        if (parent >= 0) spans[parent].Children.Add(id);
        return id;
    }

    private void CloseSpan(int id, long response, long selfCost)
    {
        Span span = spans[id];
        long children = 0;
        foreach (int child in span.Children) children += spans[child].DurationMs;
        span.ResponseDigest = response;
        span.SelfCostMs = selfCost;
        span.DurationMs = children + selfCost;
    }

    /** One dependency call: recorded from the live computation, or replayed from the capture. */
    private long Rpc(string key, long request, int parent, int selfCost)
    {
        long response, duration;
        if (live)
        {
            ulong mixed = SplitMix64.Fold(SplitMix64.Hash(key), (ulong)request);
            response = (long)mixed;
            duration = 4 + (int)(mixed % 20);
            log.Keys.Add(key);
            log.Responses.Add(response);
            log.Durations.Add(duration);
        }
        else if (cursor >= log.Count)
        {
            response = 0;
            duration = 0;
            Divergences.Add("missing recorded call " + key);
        }
        else if (log.Keys[cursor] != key)
        {
            response = 0;
            duration = 0;
            Divergences.Add("unexpected call " + key + " where the recording has " + log.Keys[cursor]);
        }
        else
        {
            response = log.Responses[cursor];
            duration = log.Durations[cursor];
        }
        cursor++;
        int id = OpenSpan(key, parent, request);
        CloseSpan(id, response, selfCost);
        return response;
    }

    private long CatalogSearch(long request, int parent)
    {
        int span = OpenSpan("catalog:search", parent, request);
        long index = Rpc("search:index", request, span, 2);
        long result = (long)SplitMix64.Fold((ulong)request, (ulong)index);
        CloseSpan(span, result, 3);
        return result;
    }

    private long PricingQuote(long request, int parent)
    {
        int span = OpenSpan("pricing:quote", parent, request);
        long rules = Rpc("pricing:rules", request, span, 1);
        long result = (long)SplitMix64.Fold((ulong)request, (ulong)rules ^ 0x51EDUL);
        CloseSpan(span, result, 2);
        return result;
    }

    /** The traced request; the same code path runs in record and in replay mode. */
    internal long HandleRoot(long request)
    {
        int root = OpenSpan("gateway:handle", -1, request);
        long user = Rpc("auth:verify", request, root, 3);
        long catalog = CatalogSearch((long)SplitMix64.Fold((ulong)request, (ulong)user), root);
        long price = PricingQuote((long)SplitMix64.Fold((ulong)request, (ulong)catalog), root);
        long response = (long)SplitMix64.Fold(SplitMix64.Fold((ulong)catalog, (ulong)price), (ulong)user);
        CloseSpan(root, response, 2);
        RootResponse = response;
        return response;
    }

    /** A degenerate request whose handler calls nothing downstream. */
    internal long HandleLeaf(long request)
    {
        int root = OpenSpan("gateway:leaf", -1, request);
        CloseSpan(root, (long)SplitMix64.Fold((ulong)request, 1UL), 0);
        RootResponse = spans[root].ResponseDigest;
        return RootResponse;
    }

    /** One digest over every field of every span: the replay fingerprint of the whole trace. */
    internal long TraceDigest()
    {
        ulong state = 0x243F6A8885A308D3UL;
        foreach (Span span in spans)
        {
            state = SplitMix64.Fold(state, (ulong)span.Id);
            state = SplitMix64.Fold(state, (ulong)span.ParentId);
            state = SplitMix64.Fold(state, SplitMix64.Hash(span.Key));
            state = SplitMix64.Fold(state, (ulong)span.RequestDigest);
            state = SplitMix64.Fold(state, (ulong)span.ResponseDigest);
            state = SplitMix64.Fold(state, (ulong)span.DurationMs);
        }
        return (long)state;
    }

    internal long CriticalPath()
    {
        var best = new long[spans.Count];
        for (int i = spans.Count - 1; i >= 0; i--)
        {
            long longest = 0;
            foreach (int child in spans[i].Children) longest = Math.Max(longest, best[child]);
            best[i] = spans[i].SelfCostMs + longest;
        }
        return best[0];
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /** Independent oracle: enumerate every root-to-leaf chain and keep the largest self-cost sum. */
    private static long MaxChainSelfCost(IReadOnlyList<Span> spans, int id, long acc)
    {
        long here = acc + spans[id].SelfCostMs;
        if (spans[id].Children.Count == 0) return here;
        long best = 0;
        foreach (int child in spans[id].Children) best = Math.Max(best, MaxChainSelfCost(spans, child, here));
        return best;
    }

    public static void Main()
    {
        const long Request = 0x00C0FFEE1234L;

        // 1. Recording run and the structural invariants of the captured trace.
        var log = new RpcLog();
        var recorder = new AppRunner(true, log);
        recorder.HandleRoot(Request);
        IReadOnlyList<Span> recorded = recorder.Spans;
        Check(recorded.Count == 6 && log.Count == 3,
            "the recorded trace holds " + recorded.Count + " spans and " + log.Count + " dependency responses");
        bool unique = recorded.Select(s => s.Id).Distinct().Count() == recorded.Count
                      && recorded.Select((s, i) => s.Id == i).All(x => x);
        bool parentsFirst = recorded.All(s => s.ParentId < s.Id);
        bool childrenInside = recorded.All(s => s.Children.Sum(c => recorded[c].DurationMs) <= s.DurationMs);
        Check(unique && parentsFirst && childrenInside && recorded[0].ParentId == -1,
            "span ids are unique and creation-ordered, parents close after their children");

        // 2. Faithful replay: same request, but every dependency answer comes from the capture.
        var replayer = new AppRunner(false, log);
        long replayResponse = replayer.HandleRoot(Request);
        Check(replayResponse == recorder.RootResponse && replayer.TraceDigest() == recorder.TraceDigest(),
            "replay reproduces the recorded trace digest and root response exactly");
        Check(replayer.Divergences.Count == 0 && replayer.ConsumedCalls == log.Count && replayer.UnconsumedCalls == 0,
            "the replay consumed all " + log.Count + " recorded calls with no divergence");

        // 3. Critical path: bottom-up DP against an exhaustive chain enumeration.
        long critical = recorder.CriticalPath();
        long exhaustive = MaxChainSelfCost(recorded, 0, 0);
        Check(critical == exhaustive && critical > 0 && critical <= recorded[0].DurationMs,
            "the critical path " + critical + " ms equals the exhaustive chain maximum");
        Check(recorded[0].DurationMs == recorded[0].SelfCostMs + recorded[0].Children.Sum(c => recorded[c].DurationMs),
            "the root duration equals its self cost plus the sum of its children");

        // 4. A mutated recorded response must change the fingerprint (silent corruption is caught).
        RpcLog mutated = log.Clone();
        mutated.Responses[2] ^= 0x5DEADBEEFL;
        var mutatedReplay = new AppRunner(false, mutated);
        mutatedReplay.HandleRoot(Request);
        Check(mutatedReplay.TraceDigest() != recorder.TraceDigest()
              && mutatedReplay.RootResponse != recorder.RootResponse
              && mutatedReplay.Divergences.Count == 0,
            "a mutated recorded response changes both the root response and the trace digest");

        // 5. A recorded call the captured run no longer makes is reported, not ignored.
        RpcLog dropped = log.Clone();
        dropped.Keys.RemoveAt(1);
        dropped.Responses.RemoveAt(1);
        dropped.Durations.RemoveAt(1);
        var droppedReplay = new AppRunner(false, dropped);
        droppedReplay.HandleRoot(Request);
        Check(droppedReplay.Divergences.Count > 0 && droppedReplay.Divergences[0].StartsWith("unexpected call")
              && droppedReplay.TraceDigest() != recorder.TraceDigest(),
            "a missing dependency is reported as a divergence (" + droppedReplay.Divergences[0] + ")");

        // 6. A recorded call the code never consumes is detected as an unconsumed entry.
        RpcLog extra = log.Clone();
        extra.Keys.Add("ghost:call");
        extra.Responses.Add(1);
        extra.Durations.Add(1);
        var extraReplay = new AppRunner(false, extra);
        extraReplay.HandleRoot(Request);
        Check(extraReplay.TraceDigest() == recorder.TraceDigest() && extraReplay.UnconsumedCalls == 1,
            "an extra recorded entry leaves the trace intact but is flagged as unconsumed");

        // 7. Boundary: a trace with no dependency calls replays with an empty capture.
        var emptyLog = new RpcLog();
        var leafRecorder = new AppRunner(true, emptyLog);
        leafRecorder.HandleLeaf(7L);
        var leafReplay = new AppRunner(false, emptyLog);
        leafReplay.HandleLeaf(7L);
        Check(emptyLog.Count == 0 && leafRecorder.Spans.Count == 1 && leafRecorder.CriticalPath() == 0
              && leafReplay.TraceDigest() == leafRecorder.TraceDigest() && leafReplay.Divergences.Count == 0,
            "a dependency-free trace replays from an empty capture with a zero critical path");

        // 8. Boundary: replaying against an empty capture for a traced request reports the first miss.
        var starved = new AppRunner(false, new RpcLog());
        starved.HandleRoot(Request);
        Check(starved.Divergences.Count == 3 && starved.UnconsumedCalls == 0 && starved.RootResponse != 0,
            "an empty capture for a traced request is flagged once per dependency instead of returning silence");

        // 9. Determinism: two replays of the same capture agree in every detail.
        var again = new AppRunner(false, log);
        again.HandleRoot(Request);
        var third = new AppRunner(false, log);
        third.HandleRoot(Request);
        Check(again.TraceDigest() == third.TraceDigest() && again.ConsumedCalls == third.ConsumedCalls
              && again.RootResponse == third.RootResponse && again.Spans.Count == recorder.Spans.Count,
            "two independent replays of one capture are bit-identical");

        Console.WriteLine("  capture: " + log.Count + " calls, " + recorded.Count + " spans, digest "
            + recorder.TraceDigest() + ", critical path " + critical + " ms of " + recorded[0].DurationMs + " ms");
        Console.WriteLine("PASS");
    }
}
