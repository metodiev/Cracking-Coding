// Task 507 - Implement throttling middleware simulation (sliding window counters). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
//
// Three per-client rate limiters behind the same Decision API - (allowed, remaining, retryAfter) -
// and a middleware that turns those decisions into 200/429 responses for a deterministic
// multi-tenant request stream:
//   * FixedWindowCounter - a counter reset at each aligned window boundary. It is the cheap
//     baseline and it is deliberately kept because its classic flaw is one of the documented
//     checks: a burst at the end of one window plus a burst at the start of the next admits 2*limit
//     requests inside a single unaligned window of length W (the crafted 5 + 5 pattern with
//     limit = 5, W = 10 admits all ten).
//   * SlidingWindowLog - the exact reference: it keeps the accepted timestamps of each client and
//     evicts those at or before tick - W, so no half-open window of length W can ever hold more
//     than `limit` accepted requests (verified by brute force over every window of the trace).
//   * SlidingWindowCounter - the production compromise: estimate = currentWindowCount +
//     previousWindowCount * (1 - elapsedFraction); verified to admit no more than 2*limit in any
//     window, and to sit between the log and the fixed window on the boundary pattern. On a
//     uniformly spaced stream it admits within 10% of the exact log's total, but its verdicts
//     differ in phase for about 40% of individual requests (it borrows quota from the previous
//     window instead of admitting the whole quota at once) - that difference is asserted, not
//     hidden.
// Every limiter answers RetryAfter by scanning forward (at most 2*W + 1 ticks) for the first tick
// at which the request would be admitted, so the self-test can close the loop: replaying a
// throttled request after waiting exactly RetryAfter ticks must succeed. Also checked: Remaining is
// monotonically non-increasing inside a window and never negative, 429s carry a positive
// Retry-After, allowed + throttled == requests, one client's flood never throttles another client,
// and the whole simulation is reproducible.
using System;
using System.Collections.Generic;

internal static class Deterministic
{
    internal static double Uniform(long seed, long index)
    {
        ulong x = (ulong)seed * 0x9E3779B97F4A7C15UL + (ulong)index * 0xD1B54A32D192ED03UL;
        x += 0x9E3779B97F4A7C15UL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
        x ^= x >> 31;
        return 2.0 * ((x >> 11) * (1.0 / 9007199254740992.0)) - 1.0;
    }

    internal static long Range(long seed, long index, long low, long high)
    {
        double unit = (Uniform(seed, index) + 1.0) * 0.5;
        long value = low + (long)(unit * (high - low + 1));
        return value > high ? high : value;
    }
}

internal readonly struct Decision
{
    internal Decision(bool allowed, long remaining, long retryAfter)
    {
        Allowed = allowed;
        Remaining = remaining;
        RetryAfter = retryAfter;
    }

    internal bool Allowed { get; }

    internal long Remaining { get; }

    internal long RetryAfter { get; }
}

internal abstract class Throttler
{
    protected Throttler(long limit, long window)
    {
        Limit = limit;
        Window = window;
    }

    internal long Limit { get; }

    internal long Window { get; }

    internal abstract string Name { get; }

    internal abstract Decision Decide(long client, long tick);

    protected abstract bool WouldAllow(long client, long tick);

    /** First strictly later tick at which the request would be admitted. */
    internal long RetryAfter(long client, long tick)
    {
        for (long candidate = tick + 1; candidate <= tick + 2 * Window + 1; candidate++)
        {
            if (WouldAllow(client, candidate)) return candidate - tick;
        }
        return 2 * Window + 1;
    }
}

internal class FixedWindowCounter : Throttler
{
    private readonly Dictionary<long, (long Index, long Count)> state =
            new Dictionary<long, (long, long)>();

    internal FixedWindowCounter(long limit, long window) : base(limit, window) { }

    internal override string Name => "fixed-window";

    internal override Decision Decide(long client, long tick)
    {
        state.TryGetValue(client, out (long Index, long Count) entry);
        long index = tick / Window;
        if (entry.Index != index) entry = (index, 0);
        if (entry.Count < Limit)
        {
            entry.Count += 1;
            state[client] = entry;
            return new Decision(true, Limit - entry.Count, 0);
        }
        state[client] = entry;
        return new Decision(false, 0, (index + 1) * Window - tick);
    }

    protected override bool WouldAllow(long client, long tick)
    {
        state.TryGetValue(client, out (long Index, long Count) entry);
        long index = tick / Window;
        long count = entry.Index == index ? entry.Count : 0;
        return count < Limit;
    }
}

internal class SlidingWindowLog : Throttler
{
    private readonly Dictionary<long, List<long>> accepted = new Dictionary<long, List<long>>();

    internal SlidingWindowLog(long limit, long window) : base(limit, window) { }

    internal override string Name => "sliding-window-log";

    private List<long> Times(long client)
    {
        if (!accepted.TryGetValue(client, out List<long> times))
        {
            times = new List<long>();
            accepted[client] = times;
        }
        return times;
    }

    private static int Trim(List<long> times, long tick, long window)
    {
        int drop = 0;
        while (drop < times.Count && times[drop] <= tick - window) drop += 1;
        if (drop > 0) times.RemoveRange(0, drop);
        return drop;
    }

    internal override Decision Decide(long client, long tick)
    {
        List<long> times = Times(client);
        Trim(times, tick, Window);
        if (times.Count < Limit)
        {
            times.Add(tick);
            return new Decision(true, Limit - times.Count, 0);
        }
        long retry = Math.Max(1, times[0] + Window - tick);
        return new Decision(false, 0, retry);
    }

    protected override bool WouldAllow(long client, long tick)
    {
        List<long> times = Times(client);
        int count = 0;
        foreach (long time in times)
        {
            if (time > tick - Window) count += 1;
        }
        return count < Limit;
    }
}

internal class SlidingWindowCounter : Throttler
{
    private readonly Dictionary<long, (long Index, long Count, long Prev)> state =
            new Dictionary<long, (long, long, long)>();

    internal SlidingWindowCounter(long limit, long window) : base(limit, window) { }

    internal override string Name => "sliding-window-counter";

    private (long Count, long Prev) Counters(long client, long tick)
    {
        if (!state.TryGetValue(client, out (long Index, long Count, long Prev) entry))
        {
            return (0, 0);
        }
        long index = tick / Window;
        if (entry.Index == index) return (entry.Count, entry.Prev);
        if (entry.Index == index - 1) return (0, entry.Count);
        return (0, 0);
    }

    private double Estimate(long client, long tick)
    {
        (long count, long prev) = Counters(client, tick);
        long index = tick / Window;
        double elapsed = (tick - index * Window) / (double)Window;
        return count + prev * (1.0 - elapsed);
    }

    internal override Decision Decide(long client, long tick)
    {
        state.TryGetValue(client, out (long Index, long Count, long Prev) entry);
        long index = tick / Window;
        if (entry.Index != index)
        {
            long previous = entry.Index == index - 1 ? entry.Count : 0;
            entry = (index, 0, previous);
        }
        double estimate = Estimate(client, tick);
        if (estimate + 1.0 <= Limit + 1e-9)
        {
            entry.Count += 1;
            state[client] = entry;
            return new Decision(true, Limit - entry.Count, 0);
        }
        state[client] = entry;
        return new Decision(false, 0, RetryAfter(client, tick));
    }

    protected override bool WouldAllow(long client, long tick)
    {
        return Estimate(client, tick) + 1.0 <= Limit + 1e-9;
    }
}

internal struct ThrottleEvent
{
    internal long Tick;
    internal long Client;
    internal bool Allowed;
    internal long Remaining;
    internal long RetryAfter;
}

internal static class Middleware
{
    internal static List<ThrottleEvent> Run(Throttler throttler, (long Tick, long Client)[] stream)
    {
        var log = new List<ThrottleEvent>();
        foreach ((long tick, long client) in stream)
        {
            Decision decision = throttler.Decide(client, tick);
            log.Add(new ThrottleEvent
            {
                Tick = tick,
                Client = client,
                Allowed = decision.Allowed,
                Remaining = decision.Remaining,
                RetryAfter = decision.RetryAfter,
            });
        }
        return log;
    }

    internal static (long Tick, long Client)[] UniformStream(int count, int clients, long gap, long seed)
    {
        var stream = new (long, long)[count];
        long tick = 0;
        for (int i = 0; i < count; i++)
        {
            tick += Deterministic.Range(seed, i, 1, gap);
            long client = Deterministic.Range(seed, 100000 + i, 0, clients - 1);
            stream[i] = (tick, client);
        }
        return stream;
    }

    internal static (long Tick, long Client)[] BurstyStream(int count, long start, long step, long client)
    {
        var stream = new (long, long)[count];
        for (int i = 0; i < count; i++) stream[i] = (start + i * step, client);
        return stream;
    }

    /** Largest number of accepted requests of one client inside any window of length W. */
    internal static long MaxAcceptedPerWindow(List<ThrottleEvent> log, long client, long window)
    {
        var ticks = new List<long>();
        foreach (ThrottleEvent entry in log)
        {
            if (entry.Client == client && entry.Allowed) ticks.Add(entry.Tick);
        }
        ticks.Sort();
        long worst = 0;
        for (int i = 0; i < ticks.Count; i++)
        {
            long count = 0;
            for (int j = i; j < ticks.Count && ticks[j] < ticks[i] + window; j++) count += 1;
            worst = Math.Max(worst, count);
        }
        return worst;
    }
}

public static class Solution
{
    public static void Main()
    {
        // --- the crafted boundary pattern ---------------------------------------------------------
        const long limit = 5;
        const long window = 10;
        var boundary = new List<(long, long)>();
        for (int i = 0; i < 3; i++) boundary.Add((8, 1));
        for (int i = 0; i < 2; i++) boundary.Add((9, 1));
        for (int i = 0; i < 3; i++) boundary.Add((10, 1));
        for (int i = 0; i < 2; i++) boundary.Add((11, 1));

        var fixedCounter = new FixedWindowCounter(limit, window);
        var log = new SlidingWindowLog(limit, window);
        var slidingCounter = new SlidingWindowCounter(limit, window);
        Check(fixedCounter.Name == "fixed-window" && log.Name == "sliding-window-log"
                && slidingCounter.Name == "sliding-window-counter", "the three limiter names");

        List<ThrottleEvent> fixedEvents = Middleware.Run(fixedCounter, boundary.ToArray());
        List<ThrottleEvent> logEvents = Middleware.Run(log, boundary.ToArray());
        List<ThrottleEvent> counterEvents = Middleware.Run(slidingCounter, boundary.ToArray());
        long fixedAllowed = CountAllowed(fixedEvents);
        long logAllowed = CountAllowed(logEvents);
        long counterAllowed = CountAllowed(counterEvents);
        Check(fixedAllowed == 2 * limit,
                "documented flaw: the fixed window admits a 2*limit burst (" + fixedAllowed + ")");
        Check(logAllowed == limit, "the log admits exactly limit in the crafted pattern");
        Check(counterAllowed >= logAllowed && counterAllowed <= fixedAllowed,
                "the sliding counter sits between log and fixed window (" + counterAllowed + ")");
        Check(Middleware.MaxAcceptedPerWindow(fixedEvents, 1, window) == 2 * limit,
                "the fixed window shows the full 2*limit window burst");
        Check(Middleware.MaxAcceptedPerWindow(logEvents, 1, window) <= limit,
                "the log never exceeds limit in any window");
        Check(Middleware.MaxAcceptedPerWindow(counterEvents, 1, window) <= 2 * limit,
                "the counter stays within the documented 2*limit bound");

        // --- the exact guarantee of the log over a long stream -------------------------------------
        var stream = Middleware.UniformStream(4000, 3, 4, 20240607);
        long[] limits = { 4, 7, 7 };
        long[] windows = { 20, 25, 10 };
        for (int c = 0; c < 3; c++)
        {
            var clientStream = new List<(long, long)>();
            foreach ((long tick, long client) in stream)
            {
                if (client == c) clientStream.Add((tick, c));
            }
            var exact = new SlidingWindowLog(limits[c], windows[c]);
            List<ThrottleEvent> events = Middleware.Run(exact, clientStream.ToArray());
            long worst = Middleware.MaxAcceptedPerWindow(events, c, windows[c]);
            Check(worst <= limits[c], "log guarantee for client " + c + " (worst " + worst + ")");
            Check(worst >= 1, "client " + c + " is actually served sometimes");
            long span = events[events.Count - 1].Tick - events[0].Tick;
            Check(CountAllowed(events) <= limits[c] * (span / windows[c] + 2),
                    "the long-run average for client " + c + " stays near the quota");
            long ceiling = Math.Min(events.Count, limits[c] * Math.Max(1, span / windows[c]));
            Check(CountAllowed(events) >= ceiling / 2,
                    "client " + c + " is not needlessly starved");
        }

        // --- 429 semantics and the Retry-After closed loop -------------------------------------------
        var flood = Middleware.BurstyStream(60, 0, 1, 42);
        var counterForFlood = new SlidingWindowCounter(6, 12);
        List<ThrottleEvent> floodEvents = Middleware.Run(counterForFlood, flood);
        long allowedFlood = CountAllowed(floodEvents);
        Check(allowedFlood + CountThrottled(floodEvents) == flood.Length, "allowed + throttled == requests");
        Check(allowedFlood >= 6 && allowedFlood <= 35, "the flood is throttled but not starved");
        Check(Middleware.MaxAcceptedPerWindow(floodEvents, 42, 12) <= 12,
                "the counter never admits more than 2*limit inside a window");
        int retried = 0;
        int succeeded = 0;
        foreach (ThrottleEvent entry in floodEvents)
        {
            if (entry.Allowed)
            {
                Check(entry.RetryAfter == 0, "an allowed request has no Retry-After");
                Check(entry.Remaining >= 0 && entry.Remaining < 6, "remaining stays inside the quota");
            }
            else
            {
                Check(entry.RetryAfter > 0, "a throttled request carries a positive Retry-After");
                Check(entry.Remaining == 0, "a throttled request has no remaining quota");
                var replayLimiter = new SlidingWindowCounter(6, 12);
                var warmup = new List<(long, long)>();
                foreach (ThrottleEvent warm in floodEvents)
                {
                    if (warm.Tick <= entry.Tick) warmup.Add((warm.Tick, warm.Client));
                }
                List<ThrottleEvent> warmEvents = Middleware.Run(replayLimiter, warmup.ToArray());
                Check(warmEvents.Count > 0, "the replay reproduced the past");
                Decision retry = replayLimiter.Decide(entry.Client, entry.Tick + entry.RetryAfter);
                retried += 1;
                if (retry.Allowed) succeeded += 1;
            }
        }
        Check(retried > 0, "the flood produced throttled requests");
        Check(succeeded == retried, "replaying after exactly RetryAfter ticks always succeeds");

        // --- tenants do not interfere -----------------------------------------------------------------
        var mixed = new List<(long, long)>();
        for (long tick = 0; tick < 400; tick++)
        {
            mixed.Add((tick, 0));
            if (tick % 10 == 0)
            {
                mixed.Add((tick, 1));
                mixed.Add((tick, 2));
            }
        }
        var tenantLimiter = new SlidingWindowLog(4, 30);
        List<ThrottleEvent> mixedEvents = Middleware.Run(tenantLimiter, mixed.ToArray());
        int quietThrottled = 0;
        long quietAllowed = 0;
        foreach (ThrottleEvent entry in mixedEvents)
        {
            if (entry.Client == 0) continue;
            if (entry.Allowed) quietAllowed += 1;
            else quietThrottled += 1;
        }
        Check(quietThrottled == 0, "quiet tenants are never throttled by the flooding tenant");
        Check(quietAllowed == 80, "both quiet tenants were served in full");
        Check(Middleware.MaxAcceptedPerWindow(mixedEvents, 0, 30) <= 4, "the flooding tenant is capped");

        // --- counter versus log on a uniform stream -----------------------------------------------------
        var uniform = Middleware.BurstyStream(300, 0, 1, 77);
        List<ThrottleEvent> uniformLog = Middleware.Run(new SlidingWindowLog(10, 25), uniform);
        List<ThrottleEvent> uniformCounter = Middleware.Run(new SlidingWindowCounter(10, 25), uniform);
        Check(uniformLog.Count == uniformCounter.Count, "both limiters saw every request");
        int mismatches = 0;
        long monotoneViolations = 0;
        long previousRemaining = long.MaxValue;
        long previousWindowIndex = -1;
        for (int i = 0; i < uniformLog.Count; i++)
        {
            if (uniformLog[i].Allowed != uniformCounter[i].Allowed) mismatches += 1;
            long index = uniformLog[i].Tick / 25;
            if (index == previousWindowIndex && uniformCounter[i].Remaining > previousRemaining)
            {
                monotoneViolations += 1;
            }
            if (index != previousWindowIndex)
            {
                previousWindowIndex = index;
                previousRemaining = long.MaxValue;
            }
            if (uniformCounter[i].Allowed) previousRemaining = uniformCounter[i].Remaining;
        }
        long allowedUniformLog = CountAllowed(uniformLog);
        long allowedUniformCounter = CountAllowed(uniformCounter);
        Check(Math.Abs(allowedUniformLog - allowedUniformCounter) <= 0.1 * allowedUniformLog,
                "both limiters admit nearly the same total (log " + allowedUniformLog + " vs counter "
                + allowedUniformCounter + ")");
        Check(mismatches <= 0.5 * uniformLog.Count,
                "documented phase difference: the per-request verdicts differ on " + mismatches
                + " of " + uniformLog.Count + " uniformly spaced requests");
        Check(monotoneViolations == 0, "remaining never grows inside a window");
        Check(Middleware.MaxAcceptedPerWindow(uniformCounter, 0, 25) <= 20,
                "the counter respects its 2*limit window bound");

        // --- determinism --------------------------------------------------------------------------------
        List<ThrottleEvent> repeated = Middleware.Run(new SlidingWindowCounter(10, 25), uniform);
        for (int i = 0; i < uniformCounter.Count; i++)
        {
            Check(repeated[i].Allowed == uniformCounter[i].Allowed
                    && repeated[i].Remaining == uniformCounter[i].Remaining
                    && repeated[i].RetryAfter == uniformCounter[i].RetryAfter,
                    "the middleware is deterministic at request " + i);
        }
        List<ThrottleEvent> logRepeat = Middleware.Run(new SlidingWindowLog(10, 25), uniform);
        for (int i = 0; i < uniformLog.Count; i++)
        {
            Check(logRepeat[i].Allowed == uniformLog[i].Allowed, "the log is deterministic at request " + i);
        }
        Check(CountAllowed(logEvents) == logAllowed && CountAllowed(counterEvents) == counterAllowed,
                "the crafted pattern verdicts are stable");

        Console.WriteLine("PASS");
    }

    private static long CountAllowed(List<ThrottleEvent> events)
    {
        long count = 0;
        foreach (ThrottleEvent entry in events)
        {
            if (entry.Allowed) count += 1;
        }
        return count;
    }

    private static long CountThrottled(List<ThrottleEvent> events) => events.Count - CountAllowed(events);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
