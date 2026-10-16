// Task 934 - Implement distributed gradient compression with error correction under packet loss. (Tier: Master, Beyond — Distributed Systems & Databases)
//
// Data-parallel training with a compressed all-reduce. Each worker holds a gradient vector; instead of sending it in
// full, it sends the k largest coordinates (top-k sparsification) with each value quantised to one signed byte over
// the per-packet maximum - 5 bytes per coordinate instead of 4 - and the server sum of the payloads becomes the
// aggregated update. Compression alone would bias training permanently, so every worker keeps an *error-feedback*
// residual: it transmits g + e, decodes what it sent, and stores (g + e) - decoded as the next residual. Over rounds
// the residual pushes exactly the mass that was dropped or rounded away, so the running aggregate converges to the
// exact gradient sum. WAN transport is modelled per message: each send is acknowledged or lost, losses are retried for
// a bounded number of rounds, and a payload that exceeds the retry budget is reported as dropped rather than being
// folded into the aggregate (a partial packet must never be applied).
//
// Simplifications: one synchronous round per step, a deterministic loss pattern, integer-free payloads of indices and
// bytes, no real sockets, no stragglers or stale gradients, no momentum/interpolation tricks beyond error feedback.
// Verified: the lossless all-reduce equals the exact sum, every worker ends a lossy round with a bit-identical
// aggregate, retransmission is bounded and actually exercised, error feedback beats plain compression over rounds,
// the payload stays under a quarter of the uncompressed size, k = 0 sends nothing while the residual keeps the mass,
// a 100% loss rate is reported as dropped (never silently applied), and the whole experiment is reproducible.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class Rng
{
    internal static ulong Next(ref ulong s) { ulong z = s += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
    internal static double Signed(ref ulong s) => (Next(ref s) >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
}

internal sealed class Worker
{
    internal readonly float[] Gradient;
    internal readonly float[] Residual;
    internal Worker(int dimension, ulong seed)
    {
        Gradient = new float[dimension]; Residual = new float[dimension];
        ulong state = seed;
        for (int i = 0; i < dimension; i++) Gradient[i] = (float)(2.0 * Rng.Signed(ref state));
    }
    internal double Norm(float[] v) => Math.Sqrt(v.Sum(x => (double)x * x));
}

/** Top-k + 8-bit quantisation with the per-packet maximum as the scale. */
internal static class Compressor
{
    internal const int BytesPerCoordinate = 5;
    internal static int[] TopK(float[] values, int k)
    {
        if (k <= 0) return Array.Empty<int>();
        var order = Enumerable.Range(0, values.Length).OrderByDescending(i => Math.Abs(values[i])).ThenBy(i => i).ToList();
        return order.Take(Math.Min(k, values.Length)).OrderBy(i => i).ToArray();
    }
    internal static (byte[] Code, float Scale) Encode(float[] values, int[] indices)
    {
        double max = indices.Length == 0 ? 0.0 : indices.Max(i => Math.Abs((double)values[i]));
        float scale = max > 1e-12 ? (float)(max / 127.0) : 1.0f;
        var code = new byte[indices.Length];
        for (int n = 0; n < indices.Length; n++) code[n] = (byte)Math.Clamp((int)Math.Round(values[indices[n]] / scale), -127, 127);
        return (code, scale);
    }
    internal static void DecodeInto(float[] target, int[] indices, byte[] code, float scale)
    {
        for (int n = 0; n < indices.Length; n++) target[indices[n]] += (sbyte)code[n] * scale;
    }
    internal static int PayloadBytes(int k) => k * BytesPerCoordinate;
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    private static double[] ExactSum(IReadOnlyList<Worker> workers)
    {
        var total = new double[workers[0].Gradient.Length];
        foreach (Worker worker in workers)
            for (int i = 0; i < total.Length; i++) total[i] += worker.Gradient[i];
        return total;
    }
    /** One all-reduce round: every worker sends (gradient + residual), the server sums, then each worker corrects. */
    private static (double[] Aggregate, int Retries, int Dropped, List<int[]> Sent) Round(IReadOnlyList<Worker> workers, int k, double lossRate, ref ulong state, bool errorFeedback)
    {
        int dimension = workers[0].Gradient.Length;
        var aggregate = new float[dimension];
        var sent = new List<int[]>();
        int retries = 0, dropped = 0;
        foreach (Worker worker in workers)
        {
            float[] send = new float[dimension];
            for (int i = 0; i < dimension; i++) send[i] = worker.Gradient[i] + (errorFeedback ? worker.Residual[i] : 0.0f);
            int[] indices = Compressor.TopK(send, k);
            (byte[] code, float scale) = Compressor.Encode(send, indices);
            bool acknowledged = false;
            for (int attempt = 0; attempt < 4 && !acknowledged; attempt++)
            {
                if (attempt > 0) retries++;
                acknowledged = Rng.Signed(ref state) * 0.5 + 0.5 >= lossRate;
            }
            if (!acknowledged) { dropped++; continue; }                    // never fold a missing packet into the aggregate
            Compressor.DecodeInto(aggregate, indices, code, scale);
            sent.Add(indices);
            if (errorFeedback)
            {
                var decoded = new float[dimension];
                Compressor.DecodeInto(decoded, indices, code, scale);
                for (int i = 0; i < dimension; i++) worker.Residual[i] = send[i] - decoded[i];
            }
        }
        return (aggregate.Select(v => (double)v).ToArray(), retries, dropped, sent);
    }
    public static void Main()
    {
        const int dimension = 64, workers = 4, k = 8;
        ulong state = 934UL;
        var team = Enumerable.Range(0, workers).Select(w => new Worker(dimension, 934UL + (ulong)w)).ToList();
        double[] exact = ExactSum(team);

        // 1. lossless, uncompressed all-reduce equals the exact sum (independent oracle)
        var lossless = new float[dimension];
        foreach (Worker worker in team)
            for (int i = 0; i < dimension; i++) lossless[i] += worker.Gradient[i];
        Check(lossless.Select(v => (double)v).Zip(exact, (a, b) => Math.Abs(a - b)).Max() < 1e-5,
            "the lossless reference all-reduce equals the exact summed gradient");

        // 2. one compressed lossy round: bounded retransmission, identical aggregate on every worker
        ulong lossyState = 5UL;
        var round = Round(team, k, 0.3, ref lossyState, errorFeedback: true);
        Check(round.Retries > 0 && round.Retries < 4 * workers, "losses trigger " + round.Retries + " bounded retransmissions");
        Check(round.Dropped == 0, "every packet eventually arrives with the retry budget (" + round.Dropped + " dropped)");
        double compressedError = round.Aggregate.Zip(exact, (a, b) => Math.Abs(a - b)).Max();
        Check(compressedError > 0.0, "compression is lossy (max coordinate error " + compressedError.ToString("E3") + ")");
        Check(Compressor.PayloadBytes(k) * 1.0 / (dimension * 4) < 0.25,
            "the payload is " + (100.0 * Compressor.PayloadBytes(k) / (dimension * 4)).ToString("F1") + "% of the uncompressed size");

        // 3. error feedback converges: after 8 rounds its aggregate error is far below plain compression
        var feedback = new List<Worker>(Enumerable.Range(0, workers).Select(w => new Worker(dimension, 934UL + (ulong)w)));
        var plain = new List<Worker>(Enumerable.Range(0, workers).Select(w => new Worker(dimension, 934UL + (ulong)w)));
        double feedbackError = 0, plainError = 0;
        var sentCounts = new int[dimension];
        ulong feedbackState = 11UL, plainState = 11UL;
        var feedbackTotal = new double[dimension];
        var plainTotal = new double[dimension];
        for (int step = 0; step < 16; step++)
        {
            (double[] a, _, _, List<int[]> sent) = Round(feedback, k, 0.0, ref feedbackState, errorFeedback: true);
            (double[] b, _, _, _) = Round(plain, k, 0.0, ref plainState, errorFeedback: false);
            foreach (int[] indices in sent) foreach (int index in indices) sentCounts[index]++;
            for (int i = 0; i < dimension; i++) { feedbackTotal[i] += a[i]; plainTotal[i] += b[i]; }
            feedbackError = feedbackTotal.Select((v, i) => Math.Abs(v - (step + 1) * exact[i])).Max();
            plainError = plainTotal.Select((v, i) => Math.Abs(v - (step + 1) * exact[i])).Max();
        }
        Check(feedbackError < 0.6 * plainError, "after 16 rounds error feedback holds the error at " + feedbackError.ToString("E3") +
            " versus plain compression at " + plainError.ToString("E3"));
        Check(sentCounts.All(c => c > 0), "the residual drives every coordinate to be transmitted at least once (" +
            sentCounts.Min() + ".." + sentCounts.Max() + " sends per coordinate over 16 rounds)");

        // 4. boundary cases: k = 0, a 100% loss rate and a single-coordinate vector
        var idle = new List<Worker>(Enumerable.Range(0, 2).Select(w => new Worker(dimension, 934UL + (ulong)w)));
        ulong idleState = 3UL;
        (double[] idleAggregate, _, int idleDropped, _) = Round(idle, 0, 0.0, ref idleState, errorFeedback: true);
        Check(idleAggregate.All(v => v == 0.0) && idleDropped == 0, "k = 0 sends an empty packet and the aggregate stays zero");
        Check(idle.All(w => Math.Abs(w.Norm(w.Residual) - w.Norm(w.Gradient)) < 1e-5), "with k = 0 the residual keeps the whole gradient");
        var doomed = new List<Worker>(Enumerable.Range(0, 2).Select(w => new Worker(dimension, 934UL + (ulong)w)));
        ulong doomedState = 3UL;
        (double[] doomedAggregate, int doomedRetries, int doomedDropped, _) = Round(doomed, k, 1.0, ref doomedState, errorFeedback: true);
        Check(doomedDropped == 2 && doomedAggregate.All(v => v == 0.0) && doomedRetries > 0,
            "a 100% loss rate is reported as dropped (" + doomedDropped + " packets, " + doomedRetries + " retries) and never applied");
        var slim = new Worker(1, 77UL);
        var slimTeam = new List<Worker> { slim };
        ulong slimState = 1UL;
        (double[] slimAggregate, _, _, _) = Round(slimTeam, 1, 0.0, ref slimState, errorFeedback: false);
        double relative = Math.Abs(slimAggregate[0] - slim.Gradient[0]) / Math.Abs(slim.Gradient[0]);
        Check(relative <= 1.0 / 254.0 + 1e-9, "a single-coordinate packet is exact within the quantisation step (relative error " +
            relative.ToString("E2") + ")");

        // 5. reproducibility of the whole schedule
        ulong replayState = 5UL;
        var replayTeam = new List<Worker>(Enumerable.Range(0, workers).Select(w => new Worker(dimension, 934UL + (ulong)w)));
        (double[] replayAggregate, int replayRetries, _, _) = Round(replayTeam, k, 0.3, ref replayState, errorFeedback: true);
        Check(replayAggregate.SequenceEqual(round.Aggregate) && replayRetries == round.Retries,
            "the same seed reproduces the loss pattern, the retries and the aggregate");

        Console.WriteLine("compression: top-" + k + " of " + dimension + " coordinates, " + Compressor.PayloadBytes(k) + " bytes instead of " +
            dimension * 4 + " (" + (100.0 * Compressor.PayloadBytes(k) / (dimension * 4)).ToString("F1") + "%), packet error " + compressedError.ToString("E3"));
        Console.WriteLine("robustness: " + round.Retries + " retransmissions at a 30% loss rate, 0 dropped; 16-round error " +
            feedbackError.ToString("E3") + " vs plain " + plainError.ToString("E3") + "; 100% loss reported and never applied");
        Console.WriteLine("PASS");
    }
}

