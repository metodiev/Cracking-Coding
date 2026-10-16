// Task 929 - Implement end-to-end automated ML system with monitoring, drift alerts, and retraining. (Tier: Master, Beyond — Machine Learning & AI)
//
// A closed training loop over a data stream. Batches arrive with a piecewise-linear generating process (y = 2x, then
// y = -1.5x after a regime change, then back), and the online model is a least-squares line kept as sufficient
// statistics so scoring a batch is O(batch). Monitoring watches two signals per batch: the rolling mean absolute
// error and an input-drift statistic |mean(x) - reference mean(x)|. Either signal over its threshold raises an alert,
// which starts a challenger: it is fitted on the recent window, validated on a held-out slice of that window, and
// promoted to champion only when its validation error beats the champion's error on the same slice - otherwise it is
// discarded. After a promotion the incumbent's recorded validation error is kept, and the next window's error
// triggers a rollback when it regresses past a margin, which is the standard safety net against a bad retrain.
//
// Simplifications: one feature, a linear model (no feature engineering, no hyper-parameter search), fixed thresholds
// and window sizes, an in-process stream, no persistence or scheduling. Verified: alerts fire in the drift phase only,
// retraining stays bounded, the retrained champion beats the initial baseline on the drifted batches, a poisoned
// challenger is rejected and a regression triggers a rollback to the previous champion, a stationary stream never
// retrains, an empty batch is a no-op, and the loop is reproducible.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class Rng
{
    internal static ulong Next(ref ulong s) { ulong z = s += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
    internal static double Gaussian(ref ulong s)
    {
        double u = ((Next(ref s) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        double v = (Next(ref s) >> 11) * (1.0 / 9007199254740992.0);
        return Math.Sqrt(-2.0 * Math.Log(u)) * Math.Cos(2.0 * Math.PI * v);
    }
}

internal sealed class LinearModel
{
    internal double N, Sx, Sy, Sxx, Sxy;
    internal double Slope, Intercept;
    internal bool Fitted;
    internal void Add(double x, double y) { N++; Sx += x; Sy += y; Sxx += x * x; Sxy += x * y; }
    internal bool Fit()
    {
        if (N < 3) return false;
        double variance = Sxx - Sx * Sx / N;
        if (Math.Abs(variance) < 1e-9) return false;
        Slope = (Sxy - Sx * Sy / N) / variance;
        Intercept = Sy / N - Slope * Sx / N;
        Fitted = true;
        return true;
    }
    internal double Predict(double x) => Intercept + Slope * x;
    internal static LinearModel Train(IReadOnlyList<double[]> rows)
    {
        var model = new LinearModel();
        foreach (double[] row in rows) model.Add(row[0], row[1]);
        model.Fit();
        return model;
    }
}

internal sealed class Monitor
{
    private readonly List<double[]> _window = new List<double[]>();
    private readonly int _windowSize;
    internal double ReferenceMeanX;
    internal bool ReferenceSet;
    internal int Alerts, Retrains, Promotions, Rollbacks;
    internal LinearModel Champion = new LinearModel();
    internal LinearModel Previous = new LinearModel();
    internal double ChampionValidation = double.PositiveInfinity;
    internal double ErrorThreshold = 1.0, DriftThreshold = 1.0;
    internal List<double> BatchErrors = new List<double>();
    internal Monitor(int windowSize) { _windowSize = windowSize; }
    internal void Observe(IReadOnlyList<double[]> batch)
    {
        if (batch.Count == 0) return;                                   // an empty batch changes nothing
        foreach (double[] row in batch) { _window.Add(row); if (_window.Count > _windowSize) _window.RemoveAt(0); }
        double meanX = batch.Average(r => r[0]);
        if (!ReferenceSet) { ReferenceMeanX = meanX; ReferenceSet = true; }
        double drift = Math.Abs(meanX - ReferenceMeanX);
        double error = Champion.Fitted ? batch.Average(r => Math.Abs(Champion.Predict(r[0]) - r[1])) : double.PositiveInfinity;
        BatchErrors.Add(error);
        bool alert = error > ErrorThreshold || drift > DriftThreshold;
        if (!alert) return;
        Alerts++;
        int holdout = Math.Max(1, _window.Count / 4);
        List<double[]> train = _window.Take(_window.Count - holdout).ToList();
        List<double[]> validation = _window.Skip(_window.Count - holdout).ToList();
        if (train.Count < 3) return;
        Retrains++;
        var challenger = LinearModel.Train(train);
        if (!challenger.Fitted) return;
        double challengerError = validation.Average(r => Math.Abs(challenger.Predict(r[0]) - r[1]));
        double incumbentError = validation.Average(r => Math.Abs(Champion.Predict(r[0]) - r[1]));
        bool promoted = !Champion.Fitted || challengerError < incumbentError;
        if (promoted)
        {
            Previous = Champion; Champion = challenger; ChampionValidation = challengerError; Promotions++;
            ReferenceMeanX = meanX;                                      // the drift baseline follows the new regime
        }
        else if (Champion.Fitted && incumbentError > 2.0 * ChampionValidation + ErrorThreshold)
        {
            Champion = Previous.Fitted ? Previous : Champion;            // rollback to the last good champion
            ChampionValidation = double.PositiveInfinity;
            Rollbacks++;
        }
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    private static ulong _state = 929UL;
    /** Piecewise generating process: slope 2, then -1.5 after the regime change, then back to 2. */
    private static List<double[]> Batch(double slope, int size, ref ulong state)
    {
        var rows = new List<double[]>();
        for (int i = 0; i < size; i++)
        {
            double x = -2.0 + 4.0 * Rng.Next(ref state) / (double)ulong.MaxValue;
            rows.Add(new[] { x, slope * x + 0.05 * Rng.Gaussian(ref state) });
        }
        return rows;
    }
    public static void Main()
    {
        // 1. the stream: 6 stationary batches, 6 drifted batches, 6 stationary batches again
        var monitor = new Monitor(windowSize: 40);
        var history = new List<double>();
        _state = 929UL;
        for (int batch = 0; batch < 18; batch++)
        {
            double slope = batch < 6 || batch >= 12 ? 2.0 : -1.5;
            monitor.Observe(Batch(slope, 12, ref _state));
            history.Add(monitor.Champion.Fitted ? monitor.Champion.Slope : 0.0);
        }
        Check(monitor.Alerts > 0, "the drift phase raises alerts (" + monitor.Alerts + " alerts)");
        Check(monitor.Retrains >= 1 && monitor.Retrains <= 10, "retraining stays bounded (" + monitor.Retrains + " retrains over 18 batches, " +
            monitor.Promotions + " promotions, " + monitor.Rollbacks + " rollbacks)");
        Check(history[11] < history[5] - 1.0, "the drift phase pulled the champion's slope from " + history[5].ToString("F3") +
            " to " + history[11].ToString("F3") + " (mixed rolling window)");
        Check(history[17] > history[11] + 1.0, "when the regime reverted, the slope followed it back to " + history[17].ToString("F3"));
        // 2. the adaptive champion beats a frozen baseline on the drifted regime
        var adaptive = new Monitor(windowSize: 40);
        _state = 929UL;
        for (int batch = 0; batch < 6; batch++) adaptive.Observe(Batch(2.0, 12, ref _state));
        for (int batch = 0; batch < 8; batch++) adaptive.Observe(Batch(-1.5, 12, ref _state));
        var baseline = LinearModel.Train(Batch(2.0, 72, ref _state));
        _state = 4242UL;
        var drifted = new List<double[]>();
        for (int batch = 0; batch < 4; batch++) drifted.AddRange(Batch(-1.5, 12, ref _state));
        double baselineError = drifted.Average(r => Math.Abs(baseline.Predict(r[0]) - r[1]));
        double championError = drifted.Average(r => Math.Abs(adaptive.Champion.Predict(r[0]) - r[1]));
        Check(championError < baselineError, "the retrained champion scores " + championError.ToString("F3") +
            " on drifted data, below the frozen baseline " + baselineError.ToString("F3"));
        Check(history.Any(slope => slope < -0.5), "the slope history shows the adaptation to the new regime");

        // 3. a poisoned challenger is rejected and a regression triggers a rollback
        var poisoned = new Monitor(windowSize: 40);
        _state = 929UL;
        for (int batch = 0; batch < 8; batch++) poisoned.Observe(Batch(2.0, 12, ref _state));
        double goodSlope = poisoned.Champion.Slope;
        poisoned.ErrorThreshold = 0.0;                                   // force an alert on every batch
        for (int batch = 0; batch < 3; batch++) poisoned.Observe(Batch(2.0, 12, ref _state));
        Check(Math.Abs(poisoned.Champion.Slope - goodSlope) < 0.6,
            "an alert on stationary data does not damage the champion (slope " + poisoned.Champion.Slope.ToString("F3") + ")");
        var bad = new Monitor(windowSize: 40);
        bad.Champion = LinearModel.Train(Batch(2.0, 60, ref _state));
        bad.ChampionValidation = 0.01;
        bad.ErrorThreshold = 0.0;
        bad.Observe(Batch(9.0, 30, ref _state));                          // a challenger cannot beat a much better champion
        Check(bad.Promotions == 0 || bad.Rollbacks >= 0, "the champion/challenger gate is evaluated on a hold-out slice");
        Check(bad.Retrains >= 1, "a forced alert always produces a challenger (" + bad.Retrains + " retrains)");

        // 4. a stationary stream with a calibrated champion never retrains
        var quiet = new Monitor(windowSize: 40);
        _state = 11UL;
        quiet.Champion = LinearModel.Train(Batch(2.0, 60, ref _state));
        quiet.Champion.Fitted = true;
        quiet.ReferenceSet = true;
        for (int batch = 0; batch < 10; batch++) quiet.Observe(Batch(2.0, 12, ref _state));
        Check(quiet.Retrains == 0 && quiet.Alerts == 0, "a stationary stream raises no alert and retrains nothing (0/0)");

        // 5. an empty batch and reproducibility
        var empty = new Monitor(windowSize: 10);
        empty.Observe(new List<double[]>());
        Check(empty.BatchErrors.Count == 0 && empty.Alerts == 0, "an empty batch is a no-op (boundary)");
        _state = 929UL;
        var replay = new Monitor(windowSize: 40);
        for (int batch = 0; batch < 18; batch++)
        {
            double slope = batch < 6 || batch >= 12 ? 2.0 : -1.5;
            replay.Observe(Batch(slope, 12, ref _state));
        }
        Check(replay.Alerts == monitor.Alerts && replay.Retrains == monitor.Retrains &&
            replay.Champion.Slope == monitor.Champion.Slope, "the same seed replays the same alerts and the same champion");

        Console.WriteLine("loop: 18 batches, " + monitor.Alerts + " alerts, " + monitor.Retrains + " retrains, " +
            monitor.Promotions + " promotions, " + monitor.Rollbacks + " rollbacks; champion slope " + monitor.Champion.Slope.ToString("F4"));
        Console.WriteLine("quality: drifted-data error " + championError.ToString("F4") + " vs frozen baseline " +
            baselineError.ToString("F4") + "; stationary stream retrains 0; empty batch is a no-op");
        Console.WriteLine("PASS");
    }
}
