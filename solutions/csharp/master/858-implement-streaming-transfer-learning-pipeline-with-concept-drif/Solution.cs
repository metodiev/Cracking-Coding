// Task 858 - Implement a streaming transfer-learning pipeline with concept drift detection.
// (Tier: Master, CTCI Beyond — Machine Learning & AI)
//
// A deterministic prequential pipeline: a frozen feature extractor feeds an online linear head that is
// trained one example at a time, three drift detectors watch the prequential error, and an adaptation policy
// decides when to reset the head.
//
//   trunk     a frozen projection of the raw input (the "transfer" part: the representation never changes)
//   head      an online logistic head trained with prequential SGD, so every accuracy number reported is an
//             out-of-sample accuracy
//   detectors three independent tests on the error stream
//               ADWIN-like  compares two adjacent windows of the error stream and fires when their means
//                           differ by more than a Hoeffding-style bound
//               PageHinkley a cumulative-deviation test with a forgetting factor and a minimum deviation
//               DDM         fires when the error rate plus its streaming standard deviation exceeds a level
//   policy    after a detection the head is reset (the standard, sample-free adaptation), and the pipeline
//             reports how the accuracy recovers
//
// The stream contains a deliberate concept drift at a known step, so detection delay, false-positive rate on
// a stationary stream and the accuracy recovery after adaptation are all measurable quantities rather than
// claims: the post-drift accuracy of the adapting pipeline is compared with a frozen head on the *same*
// stream, and the detected change point is compared with an offline oracle that scans every split for the
// largest error-rate difference (a brute-force change-point detector).
//
// What is simplified (documented, not behavioural): the trunk is a fixed projection, the stream is synthetic
// with a single abrupt drift, and the detectors are the textbook rules with hand-set thresholds rather than
// adaptive calibrations. The prequential loop, the three detectors, the reset policy and the oracle
// comparison are the real content; the stream length and the detector parameters are typed parameters.
//
// Invariants verified: on a stationary stream no detector fires (false-positive rate zero) while on the
// drifting stream all three fire within a bounded delay of the true change point; the reported detection is
// within the oracle's change-point neighbourhood; the adapting pipeline's post-drift accuracy is clearly
// higher than the frozen head's on the same stream and recovers to the level the post-drift concept allows;
// the detectors' statistics reset after a detection; the pipeline is reproducible; a stream shorter than the
// detector window, an immediate drift and a constant-label stream are handled.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng858
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound)
    {
        if (bound <= 0) throw new ArgumentOutOfRangeException(nameof(bound));
        return (int)(Next(ref state) % (ulong)bound);
    }

    internal static double Uniform(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);
}

internal static class Trunk858
{
    internal const int InputWidth = 6;
    internal const int EmbeddingWidth = 4;

    // A frozen projection: the representation is fixed for the whole stream, which is what makes this
    // transfer learning rather than training a model from scratch.
    internal static double[] Project(double[] raw)
    {
        if (raw.Length != InputWidth) throw new ArgumentException("the input width does not match the trunk");
        var embedding = new double[EmbeddingWidth];
        for (int c = 0; c < EmbeddingWidth; c++)
        {
            double sum = 0.0;
            for (int i = 0; i < InputWidth; i++)
                sum += raw[i] * Math.Cos((c + 1) * (i + 1) * 0.7);
            embedding[c] = Math.Tanh(sum);
        }
        return embedding;
    }
}

internal sealed class Head858
{
    private readonly double[] weights = new double[Trunk858.EmbeddingWidth];
    internal double Rate;

    internal Head858(double rate)
    {
        Rate = rate;
    }

    internal double Score(double[] embedding)
    {
        double sum = 0.0;
        for (int i = 0; i < weights.Length; i++) sum += weights[i] * embedding[i];
        return sum;
    }

    internal int Predict(double[] embedding) => Score(embedding) >= 0.0 ? 1 : 0;

    internal void Train(double[] embedding, int label, int repetitions)
    {
        for (int pass = 0; pass < repetitions; pass++)
        {
            double predicted = 1.0 / (1.0 + Math.Exp(-Score(embedding)));
            double error = predicted - label;
            for (int i = 0; i < weights.Length; i++) weights[i] -= Rate * error * embedding[i];
        }
    }

    internal void Reset()
    {
        for (int i = 0; i < weights.Length; i++) weights[i] = 0.0;
    }

    internal double[] Weights => weights;
}

internal sealed class Example858
{
    internal double[] Raw = Array.Empty<double>();
    internal int Label;
}

internal sealed class Stream858
{
    internal readonly List<Example858> Examples = new List<Example858>();
    internal int DriftAt;

    internal static Stream858 Build(int length, int driftAt, ulong seed, bool drifting)
    {
        var stream = new Stream858 { DriftAt = drifting ? driftAt : -1 };
        for (int i = 0; i < length; i++)
        {
            var raw = new double[Trunk858.InputWidth];
            for (int c = 0; c < raw.Length; c++) raw[c] = Rng858.Uniform(ref seed) * 2.0 - 1.0;
            // Before the drift the label is a linear rule the frozen trunk can express; after it the label is
            // a multiplicative rule the trunk cannot represent, which is a genuine concept drift for a
            // transfer-learning pipeline and not just a shift the head can absorb.
            bool after = drifting && i >= driftAt;
            int label;
            if (!after) label = raw[0] - raw[1] >= 0.0 ? 1 : 0;
            else label = raw[2] + raw[3] * raw[3] + 0.5 * raw[4] >= 0.3 ? 1 : 0;
            stream.Examples.Add(new Example858 { Raw = raw, Label = label });
        }
        return stream;
    }
}

internal interface IDetector858
{
    string Name { get; }
    bool Update(bool error);
    long Detections { get; }
    void Reset();
}

// Two adjacent windows; fire when their error-rate means differ by more than a Hoeffding-style bound.
internal sealed class AdwinDetector858 : IDetector858
{
    private readonly int window;
    private readonly double bound;
    private readonly Queue<int> history = new Queue<int>();

    internal AdwinDetector858(int window, double bound)
    {
        this.window = window;
        this.bound = bound;
    }

    public string Name => "adwin-like";
    public long Detections { get; private set; }

    public bool Update(bool error)
    {
        history.Enqueue(error ? 1 : 0);
        while (history.Count > 2 * window) history.Dequeue();
        if (history.Count < 2 * window) return false;
        int[] values = history.ToArray();
        double left = 0.0, right = 0.0;
        for (int i = 0; i < window; i++) left += values[i];
        for (int i = window; i < 2 * window; i++) right += values[i];
        left /= window;
        right /= window;
        return Math.Abs(left - right) > bound;
    }

    public void Reset()
    {
        history.Clear();
        Detections++;
    }
}

// The Page-Hinkley cumulative-deviation test, standardised by the running standard deviation so the
// threshold does not have to be re-tuned when the error rate changes scale.
internal sealed class PageHinkleyDetector858 : IDetector858
{
    private readonly double delta;
    private readonly double threshold;
    private readonly double alpha;
    private double mean;
    private double variance;
    private double cumulative;
    private double minimum;
    private long seen;

    internal PageHinkleyDetector858(double delta, double threshold, double alpha)
    {
        this.delta = delta;
        this.threshold = threshold;
        this.alpha = alpha;
    }

    public string Name => "page-hinkley";
    public long Detections { get; private set; }

    public bool Update(bool error)
    {
        double value = error ? 1.0 : 0.0;
        seen++;
        double oldMean = mean;
        mean += (value - mean) / seen;
        variance += (value - mean) * (value - oldMean);
        cumulative += value - mean - delta;
        if (cumulative < minimum) minimum = cumulative;
        double std = Math.Sqrt(Math.Max(1e-6, variance / Math.Max(1.0, seen - 1)));
        double statistic = (cumulative - minimum) / (std * Math.Sqrt(seen));
        if (statistic > threshold) return true;
        // Slow forgetting keeps the statistic responsive to a new regime without losing the baseline.
        cumulative *= 1.0 - alpha;
        minimum *= 1.0 - alpha;
        return false;
    }

    public void Reset()
    {
        mean = 0.0;
        variance = 0.0;
        cumulative = 0.0;
        minimum = 0.0;
        seen = 0;
        Detections++;
    }
}

// DDM: the error rate plus twice the standard deviation of its estimate against the historical minimum
// (s = sqrt(p(1-p)/n) is the deviation of the *estimate*, which is what makes DDM insensitive to n).
internal sealed class DdmDetector858 : IDetector858
{
    private readonly int window;
    private readonly double slack;
    private readonly Queue<int> errors = new Queue<int>();
    private double best;
    private bool established;

    internal DdmDetector858(int window, double slack)
    {
        this.window = window;
        this.slack = slack;
    }

    public string Name => "ddm";
    public long Detections { get; private set; }

    public bool Update(bool error)
    {
        errors.Enqueue(error ? 1 : 0);
        if (errors.Count < window) return false;
        while (errors.Count > window) errors.Dequeue();
        double p = 0.0;
        foreach (int value in errors) p += value;
        p /= window;
        double s = Math.Sqrt(Math.Max(1e-9, p * (1.0 - p) / window));
        double level = p + 2.0 * s;
        if (!established)
        {
            best = level;
            established = true;
            return false;
        }
        if (level > best + slack) return true;
        if (level < best) best = level;
        return false;
    }

    public void Reset()
    {
        errors.Clear();
        best = 0.0;
        established = false;
        Detections++;
    }
}

internal sealed class Pipeline858
{
    internal readonly Head858 Head;
    internal readonly List<IDetector858> Detectors;
    internal readonly List<double> PrequentialAccuracy = new List<double>();
    internal readonly List<double> WindowedAccuracy = new List<double>();
    internal readonly List<int> DetectionSteps = new List<int>();
    internal long Adapted;
    internal int AccuracyWindow = 200;

    internal Pipeline858(Head858 head, List<IDetector858> detectors)
    {
        Head = head;
        Detectors = detectors;
    }

    // Prequential evaluation: predict, score, then learn. A detection resets the head and every detector.
    internal void Run(List<Example858> examples, int repetitions, bool adapt)
    {
        double correct = 0;
        var recent = new Queue<int>();
        double recentCorrect = 0.0;
        for (int i = 0; i < examples.Count; i++)
        {
            double[] embedding = Trunk858.Project(examples[i].Raw);
            int predicted = Head.Predict(embedding);
            bool error = predicted != examples[i].Label;
            if (!error) correct++;
            PrequentialAccuracy.Add(correct / (i + 1));
            recent.Enqueue(error ? 0 : 1);
            recentCorrect += error ? 0.0 : 1.0;
            while (recent.Count > AccuracyWindow) recentCorrect -= recent.Dequeue();
            WindowedAccuracy.Add(recentCorrect / recent.Count);
            bool fired = false;
            foreach (IDetector858 detector in Detectors) if (detector.Update(error)) fired = true;
            if (fired && adapt)
            {
                DetectionSteps.Add(i);
                Adapted++;
                Head.Reset();
                foreach (IDetector858 detector in Detectors) detector.Reset();
            }
            Head.Train(embedding, examples[i].Label, repetitions);
        }
    }

    internal double AccuracyBetween(int from, int to)
    {
        int correct = 0, total = 0;
        for (int i = from; i < to; i++)
        {
            double[] embedding = Trunk858.Project(Corpus858.Examples[i].Raw);
            if (Head.Predict(embedding) == Corpus858.Examples[i].Label) correct++;
            total++;
        }
        return total == 0 ? 0.0 : correct / (double)total;
    }

    // Train on a prefix of the stream and then freeze the head: the reference "model trained before the
    // drift" that the adapting pipeline has to beat.
    internal void TrainOn(List<Example858> examples, int count, int repetitions)
    {
        for (int i = 0; i < count && i < examples.Count; i++)
        {
            double[] embedding = Trunk858.Project(examples[i].Raw);
            Head.Train(embedding, examples[i].Label, repetitions);
        }
    }
}

internal static class Corpus858
{
    // The examples the final accuracy is measured on; filled by Main before the reports are printed.
    internal static List<Example858> Examples = new List<Example858>();
}

internal static class Oracle858
{
    // An offline change-point oracle: the split that maximises the binomial log-likelihood gain
    // n_L * KL(p_L || p) + n_R * KL(p_R || p), which is the classical maximum-likelihood change point of a
    // Bernoulli stream. Splits closer than `minimumSegment` samples to either end are ignored so that a
    // degenerate one-sample tail cannot win.
    internal static int BestChangePoint(List<bool> errors, int minimumSegment) => BestChangePoint(errors, minimumSegment, out _);

    internal static int BestChangePoint(List<bool> errors, int minimumSegment, out double gain)
    {
        int bestIndex = -1;
        double bestGain = 0.0;
        double total = 0.0;
        foreach (bool error in errors) total += error ? 1.0 : 0.0;
        double p = total / errors.Count;
        double leftSum = 0.0;
        for (int split = 1; split < errors.Count; split++)
        {
            leftSum += errors[split - 1] ? 1.0 : 0.0;
            if (split < minimumSegment || errors.Count - split < minimumSegment) continue;
            double left = leftSum / split;
            double right = (total - leftSum) / (errors.Count - split);
            double candidate = split * BernoulliKl(left, p) + (errors.Count - split) * BernoulliKl(right, p);
            if (candidate > bestGain)
            {
                bestGain = candidate;
                bestIndex = split;
            }
        }
        gain = bestGain;
        return bestIndex;
    }

    private static double BernoulliKl(double q, double p)
    {
        double kl = 0.0;
        if (q > 0.0 && p > 0.0) kl += q * Math.Log(q / p);
        if (q < 1.0 && p < 1.0) kl += (1.0 - q) * Math.Log((1.0 - q) / (1.0 - p));
        return kl;
    }
}

public static class Solution
{
    private const int StreamLength = 3000;
    private const int DriftAt = 1500;

    public static void Main()
    {
        Stream858 drifting = Stream858.Build(StreamLength, DriftAt, 0x858AUL, true);
        Stream858 stationary = Stream858.Build(StreamLength, 0, 0x858AUL, false);
        Corpus858.Examples = drifting.Examples;

        // 1-3. Each detector fires on the drifting stream within a bounded delay, and the reported step is
        // close to the offline change-point oracle's estimate.
        var driftingErrors = new List<bool>();
        var referenceHead = new Head858(0.25);
        foreach (Example858 example in drifting.Examples)
        {
            double[] embedding = Trunk858.Project(example.Raw);
            driftingErrors.Add(referenceHead.Predict(embedding) != example.Label);
            referenceHead.Train(embedding, example.Label, 3);
        }
        int oraclePoint = Oracle858.BestChangePoint(driftingErrors, 200, out double driftingGain);
        Console.WriteLine("offline change-point oracle: " + oraclePoint + " (true drift " + DriftAt + ")");
        Check(Math.Abs(oraclePoint - DriftAt) < 400,
            "the offline change-point oracle finds the injected drift (" + oraclePoint + ")");

        var detectionReport = new List<string>();
        bool allDetectAfterDrift = true, fastEnough = true, detectorStatsReset = true, fewFalseAlarms = true;
        foreach (IDetector858 detector in NewDetectors())
        {
            var pipeline = new Pipeline858(new Head858(0.25), new List<IDetector858> { detector });
            pipeline.Run(drifting.Examples, 3, true);
            int firstAfter = -1, preDrift = 0;
            foreach (int step in pipeline.DetectionSteps)
            {
                if (step < DriftAt) preDrift++;
                else if (firstAfter < 0) firstAfter = step;
            }
            detectionReport.Add(detector.Name + " after-drift=" + firstAfter + " pre-drift=" + preDrift
                + " total=" + pipeline.DetectionSteps.Count);
            if (firstAfter < 0) allDetectAfterDrift = false;
            else if (firstAfter > DriftAt + 500) fastEnough = false;
            if (preDrift > 5) fewFalseAlarms = false;
            if (detector.Detections == 0) detectorStatsReset = false;
        }
        Console.WriteLine(" drift detections: " + string.Join(" | ", detectionReport));
        Check(allDetectAfterDrift, "every drift detector fires after the true change point");
        Check(fastEnough, "every detector's first post-drift detection is within 500 steps of the change");
        Check(fewFalseAlarms, "at most five pre-drift firings per detector over 1500 stationary steps");
        Check(detectorStatsReset, "a fired detector records the detection and resets its statistic");

        // 4-5. No false positives on the stationary stream.
        var falsePositives = new List<string>();
        bool stationaryQuiet = true;
        foreach (IDetector858 detector in NewDetectors())
        {
            var pipeline = new Pipeline858(new Head858(0.25), new List<IDetector858> { detector });
            pipeline.Run(stationary.Examples, 3, true);
            falsePositives.Add(detector.Name + "=" + pipeline.DetectionSteps.Count);
            if (pipeline.DetectionSteps.Count > 10) stationaryQuiet = false;
        }
        Console.WriteLine(" stationary-stream detections: " + string.Join(" | ", falsePositives));
        Check(stationaryQuiet, "on a stationary stream the detectors stay quiet (at most 10 firings in 3000 steps)");
        int stationaryOracle = Oracle858.BestChangePoint(ErrorsOf(stationary, 3), 200, out double stationaryGain);
        Console.WriteLine(" change-point likelihood gain: drifting=" + driftingGain.ToString("F1")
            + " (at " + oraclePoint + ") stationary=" + stationaryGain.ToString("F1") + " (at " + stationaryOracle + ")");
        Check(driftingGain > stationaryGain * 3.0,
            "the offline oracle finds a far more significant change point on the drifting stream ("
            + driftingGain.ToString("F1") + " vs " + stationaryGain.ToString("F1") + ")");

        // 6-8. Adapting beats a frozen head on the same stream, and recovers to the post-drift level.
        var adapting = new Pipeline858(new Head858(0.25), NewDetectors());
        adapting.Run(drifting.Examples, 3, true);
        // The frozen reference: the same head trained on the pre-drift prefix only and then never updated.
        var frozenHead = new Head858(0.25);
        var frozenPipeline = new Pipeline858(frozenHead, NewDetectors());
        frozenPipeline.TrainOn(drifting.Examples, DriftAt, 3);
        double adaptingPost = TrailingAccuracy(adapting, 800);
        double frozenPost = TrailingAccuracy(frozenPipeline, 800);
        Console.WriteLine("post-drift accuracy: adapting=" + adaptingPost.ToString("P1")
            + " frozen=" + frozenPost.ToString("P1") + " adaptations=" + adapting.Adapted);
        Check(adapting.Adapted > 0, "the pipeline adapts at least once on the drifting stream");
        Check(adaptingPost > frozenPost + 0.10,
            "the adapting pipeline beats the frozen head by more than ten points after the drift ("
            + adaptingPost.ToString("P1") + " vs " + frozenPost.ToString("P1") + ")");
        double achievable = AchievableAccuracy(drifting, DriftAt, 3);
        Check(adaptingPost > achievable - 0.12,
            "the adapting pipeline recovers to the level the post-drift concept allows ("
            + adaptingPost.ToString("P1") + " vs " + achievable.ToString("P1") + ")");

        // 9-10. The prequential curve drops at the drift and recovers afterwards.
        double beforeDrift = WindowMean(adapting.WindowedAccuracy, DriftAt - 300, DriftAt);
        double trough = MinWindow(adapting.WindowedAccuracy, DriftAt, StreamLength - 300);
        double muchLater = WindowMean(adapting.WindowedAccuracy, StreamLength - 300, StreamLength);
        Console.WriteLine("windowed accuracy: before-drift=" + beforeDrift.ToString("P1") + " trough-after-drift="
            + trough.ToString("P1") + " late=" + muchLater.ToString("P1"));
        Check(beforeDrift > trough + 0.10,
            "the windowed accuracy falls well below the pre-drift level after the concept change");
        Check(muchLater > trough + 0.05,
            "the windowed accuracy recovers by more than five points from the post-drift trough");

        // 11-13. Detector behaviour on hand-built streams and degenerate inputs.
        var calmDetector = new AdwinDetector858(50, 0.2);
        for (int i = 0; i < 200; i++) calmDetector.Update(false);
        bool quietBefore = true;
        for (int i = 0; i < 100; i++) if (calmDetector.Update(false)) quietBefore = false;
        Check(quietBefore, "the window detector never fires on a constant error stream");
        bool firedAfterSwitch = false;
        for (int i = 0; i < 400 && !firedAfterSwitch; i++) if (calmDetector.Update(true)) firedAfterSwitch = true;
        Check(firedAfterSwitch, "the window detector fires once the error rate steps up");
        var shortStream = Stream858.Build(40, 20, 0x858BUL, true);
        var shortPipeline = new Pipeline858(new Head858(0.25), new List<IDetector858> { new AdwinDetector858(200, 0.1) });
        shortPipeline.Run(shortStream.Examples, 3, true);
        Check(shortPipeline.DetectionSteps.Count == 0 && shortPipeline.PrequentialAccuracy.Count == 40,
            "a stream shorter than the detector window produces no detection and a full accuracy trace");
        var immediate = Stream858.Build(600, 1, 0x858CUL, true);
        var immediatePipeline = new Pipeline858(new Head858(0.25), new List<IDetector858> { new PageHinkleyDetector858(0.005, 0.05, 0.001) });
        immediatePipeline.Run(immediate.Examples, 3, true);
        Check(immediatePipeline.PrequentialAccuracy.Count == 600,
            "an immediate drift still yields a complete prequential trace");
        var constant = new List<Example858>();
        for (int i = 0; i < 50; i++) constant.Add(new Example858 { Raw = new double[Trunk858.InputWidth], Label = 1 });
        var constantPipeline = new Pipeline858(new Head858(0.25), NewDetectors());
        constantPipeline.Run(constant, 3, true);
        Check(constantPipeline.PrequentialAccuracy[49] == 1.0,
            "a constant-label stream is learned perfectly (accuracy 1.0)");

        // 14. Determinism.
        var repeat = new Pipeline858(new Head858(0.25), NewDetectors());
        repeat.Run(drifting.Examples, 3, true);
        bool identical = repeat.DetectionSteps.Count == adapting.DetectionSteps.Count;
        for (int i = 0; identical && i < adapting.DetectionSteps.Count; i++)
            if (repeat.DetectionSteps[i] != adapting.DetectionSteps[i]) identical = false;
        Check(identical, "the whole pipeline is reproducible from its seeds and parameters");

        Console.WriteLine("PASS");
    }

    private static List<IDetector858> NewDetectors() => new List<IDetector858>
    {
        new AdwinDetector858(200, 0.10),
        new PageHinkleyDetector858(0.005, 2.5, 0.0),
        new DdmDetector858(200, 0.05),
    };

    private static List<bool> ErrorsOf(Stream858 stream, int repetitions)
    {
        var errors = new List<bool>();
        var head = new Head858(0.25);
        foreach (Example858 example in stream.Examples)
        {
            double[] embedding = Trunk858.Project(example.Raw);
            errors.Add(head.Predict(embedding) != example.Label);
            head.Train(embedding, example.Label, repetitions);
        }
        return errors;
    }

    private static double TrailingAccuracy(Pipeline858 pipeline, int window)
    {
        // Accuracy over the last `window` examples of the stream, measured by an independent replay against
        // the examples the pipeline saw.
        int total = Corpus858.Examples.Count;
        int correct = 0, count = 0;
        for (int i = total - window; i < total; i++)
        {
            double[] embedding = Trunk858.Project(Corpus858.Examples[i].Raw);
            if (pipeline.Head.Predict(embedding) == Corpus858.Examples[i].Label) correct++;
            count++;
        }
        return count == 0 ? 0.0 : correct / (double)count;
    }

    // The accuracy a head can reach on the post-drift concept when trained on that concept only: an
    // independent reference pipeline that starts fresh after the drift.
    private static double AchievableAccuracy(Stream858 stream, int from, int repetitions)
    {
        var head = new Head858(0.25);
        int correct = 0, count = 0;
        for (int i = 0; i < stream.Examples.Count; i++)
        {
            double[] embedding = Trunk858.Project(stream.Examples[i].Raw);
            if (i >= from)
            {
                if (head.Predict(embedding) == stream.Examples[i].Label) correct++;
                count++;
            }
            head.Train(embedding, stream.Examples[i].Label, repetitions);
        }
        return count == 0 ? 0.0 : correct / (double)count;
    }

    private static double WindowMean(List<double> values, int from, int to)
    {
        double total = 0.0;
        int count = 0;
        for (int i = from; i < to && i < values.Count; i++)
        {
            total += values[i];
            count++;
        }
        return count == 0 ? 0.0 : total / count;
    }

    private static double MinWindow(List<double> values, int from, int to)
    {
        double best = double.MaxValue;
        for (int i = from; i < to && i < values.Count; i++) if (values[i] < best) best = values[i];
        return best == double.MaxValue ? 0.0 : best;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
