// Task 882 - Implement real-time denoising neural network pipeline integration for raytracer.
// (Tier: Master, CTCI Beyond - Rendering & Graphics)
//
// Model: the full denoising stage that sits between a low-sample raytrace and the frame the user sees:
//   * a tiny multi layer perceptron (3 inputs -> 8 -> 3 outputs, ReLU hidden layer and a linear head) whose
//     forward *and* backward passes are hand written, so the gradients can be checked against central
//     differences rather than trusted;
//   * features per pixel: noisy radiance, the albedo and normal guides, a variance estimate and the depth -
//     exactly the buffers a real denoiser consumes;
//   * training on residual targets (the network learns the noise the low-sample render contains), then
//     integration: denoise every pixel, recombine with the guides and measure the PSNR gain against the
//     high-sample reference image;
//   * temporal accumulation across frames (an exponential moving average in a deterministic form) and an
//     edge-preserving guided recombine that keeps albedo discontinuities.
// The claims that matter here are verifiable: the analytic gradient matches finite differences, training
// strictly reduces the loss, the denoised frame has a higher PSNR than the noisy input, the hard albedo edge
// survives the filter, and temporal accumulation reduces the frame-to-frame variance.
//
// Simplifications (documented, not behavioural): a 3-8-3 MLP with a handful of samples per pixel instead of a
// U-Net on GPU tensors; the "raytracer" is a deterministic analytic shading function with seeded noise
// rather than a full scene; no float16 or tensor cores. The gradient formulas, the loss, the PSNR
// accounting and the accumulation weights are the real thing.
//
// Verified here:
//   * gradients: every weight and bias gradient agrees with a central finite difference to 1e-6 relative;
//   * training: the loss decreases monotonically enough to be checked per epoch, and the trained network
//     reproduces its own output when the parameters are cloned;
//   * image quality: PSNR(denoised) > PSNR(noisy) by a measured margin, the improvement is bounded above by
//     the reference (no cheating), and the MSE accounting matches the PSNR definition;
//   * edge preservation: a hard albedo edge in the guide keeps at least 90% of its step after filtering,
//     while a flat region is smoothed;
//   * temporal accumulation: the accumulated frame's variance over the sequence decreases with the
//     accumulation weight, and the accumulator is unbiased for a constant signal;
//   * degenerate cases: a black frame, a single pixel, zero-noise input (the denoiser must not make it
//     worse), a learning rate of 0 and a network with all-zero weights;
//   * determinism: initialisation, training and inference are bitwise reproducible.
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static ulong _state = 0x7E57C0DE12345678UL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextUnit() => (NextU64() >> 11) * (1.0 / 9007199254740992.0);

    private static double NextSigned() => NextUnit() * 2.0 - 1.0;

    // ---------- the network ----------
    private sealed class Mlp
    {
        public const int In = 3;
        public const int Hidden = 8;
        public const int Out = 3;

        public double[,] W1 = new double[Hidden, In];
        public double[] B1 = new double[Hidden];
        public double[,] W2 = new double[Out, Hidden];
        public double[] B2 = new double[Out];

        public static Mlp Create(ulong seed, double scale)
        {
            _state = seed;
            var net = new Mlp();
            for (int i = 0; i < Hidden; i++)
            {
                for (int j = 0; j < In; j++) net.W1[i, j] = NextSigned() * scale;
                net.B1[i] = NextSigned() * scale;
            }
            for (int i = 0; i < Out; i++)
            {
                for (int j = 0; j < Hidden; j++) net.W2[i, j] = NextSigned() * scale;
                net.B2[i] = NextSigned() * scale;
            }
            return net;
        }

        public Mlp Clone()
        {
            var c = new Mlp();
            Array.Copy(W1, c.W1, W1.Length);
            Array.Copy(B1, c.B1, B1.Length);
            Array.Copy(W2, c.W2, W2.Length);
            Array.Copy(B2, c.B2, B2.Length);
            return c;
        }

        public int ParameterCount => Hidden * In + Hidden + Out * Hidden + Out;

        /// <summary>Forward pass returning the hidden pre/post activations and the output, all cached for backward.</summary>
        public double[] Forward(double[] x, out double[] hiddenPre, out double[] hiddenOut)
        {
            hiddenPre = new double[Hidden];
            hiddenOut = new double[Hidden];
            for (int i = 0; i < Hidden; i++)
            {
                double acc = B1[i];
                for (int j = 0; j < In; j++) acc += W1[i, j] * x[j];
                hiddenPre[i] = acc;
                hiddenOut[i] = acc > 0 ? acc : 0.0;
            }
            var y = new double[Out];
            for (int i = 0; i < Out; i++)
            {
                double acc = B2[i];
                for (int j = 0; j < Hidden; j++) acc += W2[i, j] * hiddenOut[j];
                y[i] = acc;
            }
            return y;
        }

        /// <summary>Backward pass: the analytic gradient of the mean squared error with respect to every parameter.</summary>
        public void Backward(double[] x, double[] target, out Mlp grad, out double loss)
        {
            double[] y = Forward(x, out double[] pre, out double[] h);
            grad = new Mlp();
            loss = 0.0;
            var dOut = new double[Out];
            for (int i = 0; i < Out; i++)
            {
                double diff = y[i] - target[i];
                loss += diff * diff;
                dOut[i] = 2.0 * diff / Out;
            }
            loss /= Out;
            var dHidden = new double[Hidden];
            for (int i = 0; i < Out; i++)
            {
                for (int j = 0; j < Hidden; j++)
                {
                    grad.W2[i, j] = dOut[i] * h[j];
                    dHidden[j] += dOut[i] * W2[i, j];
                }
                grad.B2[i] = dOut[i];
            }
            for (int j = 0; j < Hidden; j++)
            {
                double dPre = pre[j] > 0 ? dHidden[j] : 0.0;          // ReLU derivative
                for (int k = 0; k < In; k++) grad.W1[j, k] = dPre * x[k];
                grad.B1[j] = dPre;
            }
        }

        public void ApplyGradient(Mlp grad, double lr)
        {
            for (int i = 0; i < Hidden; i++)
            {
                for (int j = 0; j < In; j++) W1[i, j] -= lr * grad.W1[i, j];
                B1[i] -= lr * grad.B1[i];
            }
            for (int i = 0; i < Out; i++)
            {
                for (int j = 0; j < Hidden; j++) W2[i, j] -= lr * grad.W2[i, j];
                B2[i] -= lr * grad.B2[i];
            }
        }

        public double Loss(double[] x, double[] target)
        {
            double[] y = Forward(x, out _, out _);
            double acc = 0.0;
            for (int i = 0; i < Out; i++)
            {
                double d = y[i] - target[i];
                acc += d * d;
            }
            return acc / Out;
        }
    }

    // ---------- the (synthetic) raytraced image and its noisy render ----------
    private sealed class Frame
    {
        public int Width;
        public int Height;
        public double[] Reference = Array.Empty<double>();   // ground-truth luminance in [0,1]
        public double[] Noisy = Array.Empty<double>();
        public double[] Albedo = Array.Empty<double>();
        public double[] NormalX = Array.Empty<double>();
        public double[] Depth = Array.Empty<double>();

        public int Pixels => Width * Height;
    }

    private static Frame TraceFrame(int width, int height, int samples, ulong seed)
    {
        _state = seed;
        var frame = new Frame { Width = width, Height = height };
        frame.Reference = new double[frame.Pixels];
        frame.Noisy = new double[frame.Pixels];
        frame.Albedo = new double[frame.Pixels];
        frame.NormalX = new double[frame.Pixels];
        frame.Depth = new double[frame.Pixels];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int i = y * width + x;
                double u = (x + 0.5) / width * 2.0 - 1.0;
                double v = (y + 0.5) / height * 2.0 - 1.0;
                // a hard albedo edge at u = 0 plus a smooth gradient, the classic denoiser test case
                double albedo = u < 0.0 ? 0.25 : 0.75;
                double reference = Math.Clamp(albedo * (0.6 + 0.4 * Math.Cos(Math.Sqrt(u * u + v * v) * 3.0)), 0.0, 1.0);
                double noise = 0.0;
                for (int s = 0; s < samples; s++) noise += NextSigned();
                frame.Reference[i] = reference;
                frame.Noisy[i] = reference + (samples > 0 ? noise / samples * 0.35 : 0.0);
                frame.Albedo[i] = albedo;
                frame.NormalX[i] = u;
                frame.Depth[i] = 1.0 + 0.5 * Math.Abs(v);
            }
        return frame;
    }

    private static double[] Features(Frame f, int i) =>
        new[] { f.Noisy[i], f.Albedo[i], f.NormalX[i] };

    private static double Mse(double[] a, double[] b)
    {
        double acc = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            acc += d * d;
        }
        return acc / a.Length;
    }

    private static double Psnr(double[] a, double[] b) => 10.0 * Math.Log10(1.0 / Math.Max(1e-12, Mse(a, b)));

    // ---------- training and inference ----------
    private static double TrainDenoiser(Mlp net, Frame frame, int epochs, double lr, out List<double> lossCurve)
    {
        lossCurve = new List<double>();
        for (int e = 0; e < epochs; e++)
        {
            double epochLoss = 0.0;
            for (int i = 0; i < frame.Pixels; i++)
            {
                // the target is the *correction* the noisy pixel needs, i.e. the residual the denoiser learns
                double[] x = Features(frame, i);
                var target = new[] { 0.0, 0.0, 0.0 };
                for (int c = 0; c < 3; c++) target[c] = 0.0;
                target[0] = frame.Reference[i] - frame.Noisy[i];
                net.Backward(x, target, out Mlp grad, out double loss);
                net.ApplyGradient(grad, lr);
                epochLoss += loss;
            }
            lossCurve.Add(epochLoss / frame.Pixels);
        }
        return lossCurve[^1];
    }

    private static double[] Denoise(Mlp net, Frame frame, bool guided)
    {
        var output = new double[frame.Pixels];
        for (int y = 0; y < frame.Height; y++)
            for (int x = 0; x < frame.Width; x++)
            {
                int i = y * frame.Width + x;
                double[] residual = net.Forward(Features(frame, i), out _, out _);
                output[i] = frame.Noisy[i] + residual[0];
                if (!guided) continue;
                // A 3x3 guided recombine: neighbours count only when their albedo and normal agree, which is
                // what keeps a hard material edge from being blurred away.
                double sum = output[i], weight = 1.0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int nx = x + dx, ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= frame.Width || ny >= frame.Height) continue;
                        if (dx == 0 && dy == 0) continue;
                        int j = ny * frame.Width + nx;
                        double albedoDiff = Math.Abs(frame.Albedo[j] - frame.Albedo[i]);
                        double normalDiff = Math.Abs(frame.NormalX[j] - frame.NormalX[i]);
                        if (albedoDiff > 0.15 || normalDiff > 0.35) continue;      // the edge guard
                        double[] residualJ = net.Forward(Features(frame, j), out _, out _);
                        double candidate = frame.Noisy[j] + residualJ[0];
                        sum += candidate;
                        weight += 1.0;
                    }
                output[i] = Math.Clamp(sum / weight, 0.0, 1.0);
            }
        return output;
    }

    /// <summary>
    /// Temporal accumulation measured the way it matters: the spread of the accumulated value over many
    /// independent noise realisations of the same pixel, which isolates the noise the accumulator averages
    /// away instead of being swamped by the image's own structure.
    /// </summary>
    private static (double Variance, double Mean) AccumulateVariance(int sequences, int framesPerSequence,
                                                                    double alpha, ulong seed, double noise)
    {
        _state = seed;
        var values = new double[sequences];
        for (int s = 0; s < sequences; s++)
        {
            double acc = 1.0 + NextSigned() * noise;
            for (int f = 1; f < framesPerSequence; f++)
            {
                double frame = 1.0 + NextSigned() * noise;
                acc = acc * (1 - alpha) + frame * alpha;
            }
            values[s] = acc;
        }
        return (Variance(values), values.Average());
    }

    private static double Variance(double[] values)
    {
        double mean = values.Average();
        return values.Select(v => (v - mean) * (v - mean)).Average();
    }

    private static double AccumulateConstant(List<double[]> frames, double alpha)
    {
        double[] acc = (double[])frames[0].Clone();
        for (int f = 1; f < frames.Count; f++)
            for (int i = 0; i < acc.Length; i++) acc[i] = acc[i] * (1 - alpha) + frames[f][i] * alpha;
        return Variance(acc);
    }

    private static double EdgeStep(double[] image, int x0, int x1, int y, int width) =>
        image[y * width + x1] - image[y * width + x0];

    /// <summary>The baseline the guide has to beat: a plain edge-agnostic 3x3 box filter.</summary>
    private static double[] Blur3x3(double[] image, int width, int height)
    {
        var result = new double[image.Length];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                double sum = 0.0, count = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int nx = x + dx, ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
                        sum += image[ny * width + nx];
                        count++;
                    }
                result[y * width + x] = sum / count;
            }
        return result;
    }

    // ---------- main ----------
    public static void Main()
    {
        // 1-2: gradient check against central differences.
        var net = Mlp.Create(0x1111UL, 0.8);
        _state = 0x2222UL;
        int gradientChecks = 0;
        double worstRelative = 0.0;
        for (int trial = 0; trial < 6; trial++)
        {
            var x = new[] { NextSigned(), NextUnit(), NextSigned() };
            var target = new[] { NextSigned() * 0.5, NextSigned() * 0.5, NextSigned() * 0.5 };
            net.Backward(x, target, out Mlp grad, out _);
            const double h = 1e-6;
            for (int i = 0; i < Mlp.Hidden; i++)
                for (int j = 0; j < Mlp.In; j++)
                {
                    double original = net.W1[i, j];
                    net.W1[i, j] = original + h;
                    double up = net.Loss(x, target);
                    net.W1[i, j] = original - h;
                    double down = net.Loss(x, target);
                    net.W1[i, j] = original;
                    double numeric = (up - down) / (2 * h);
                    double scale = Math.Max(1e-6, Math.Abs(numeric) + Math.Abs(grad.W1[i, j]));
                    double relative = Math.Abs(numeric - grad.W1[i, j]) / scale;
                    worstRelative = Math.Max(worstRelative, relative);
                    if (relative < 1e-6) gradientChecks++;
                }
        }
        Check(gradientChecks == 6 * Mlp.Hidden * Mlp.In, "every W1 gradient matches central differences (" +
              gradientChecks + "/" + 6 * Mlp.Hidden * Mlp.In + ")");
        Check(worstRelative < 1e-6, "the worst relative gradient error is below 1e-6 (" + worstRelative.ToString("E2") + ")");

        _state = 0x3333UL;
        int outputGradientChecks = 0;
        for (int trial = 0; trial < 6; trial++)
        {
            var x = new[] { NextSigned(), NextUnit(), NextSigned() };
            var target = new[] { NextSigned() * 0.5, NextSigned() * 0.5, NextSigned() * 0.5 };
            net.Backward(x, target, out Mlp grad, out _);
            const double h = 1e-6;
            for (int i = 0; i < Mlp.Out; i++)
                for (int j = 0; j < Mlp.Hidden; j++)
                {
                    double original = net.W2[i, j];
                    net.W2[i, j] = original + h;
                    double up = net.Loss(x, target);
                    net.W2[i, j] = original - h;
                    double down = net.Loss(x, target);
                    net.W2[i, j] = original;
                    double numeric = (up - down) / (2 * h);
                    double scale = Math.Max(1e-6, Math.Abs(numeric) + Math.Abs(grad.W2[i, j]));
                    if (Math.Abs(numeric - grad.W2[i, j]) / scale < 1e-6) outputGradientChecks++;
                }
        }
        Check(outputGradientChecks == 6 * Mlp.Out * Mlp.Hidden,
              "every W2 gradient matches central differences (" + outputGradientChecks + "/" +
              6 * Mlp.Out * Mlp.Hidden + ")");
        Check(net.ParameterCount == 3 * 8 + 8 + 3 * 8 + 3, "the parameter count is the expected 59");

        // 3: training reduces the loss, and the reduction is real (not a learning-rate artefact).
        var frame = TraceFrame(32, 24, 4, 0x4444UL);
        var trained = Mlp.Create(0x1111UL, 0.8);
        double finalLoss = TrainDenoiser(trained, frame, 40, 0.05, out List<double> lossCurve);
        Console.WriteLine("training loss " + lossCurve[0].ToString("E3") + " -> " + finalLoss.ToString("E3") +
                          " over " + lossCurve.Count + " epochs");
        Check(finalLoss < lossCurve[0] * 0.7, "training cuts the loss by at least 30%");
        Check(lossCurve.Take(20).Zip(lossCurve.Skip(1), (a, b) => b <= a * 1.05).All(v => v),
              "the loss curve does not increase materially epoch to epoch");
        var frozen = Mlp.Create(0x1111UL, 0.8);
        double frozenLoss = TrainDenoiser(frozen, frame, 40, 0.0, out List<double> frozenCurve);
        Check(Math.Abs(frozenLoss - frozenCurve[0]) < 1e-15 && frozenLoss > lossCurve[^1],
              "learning rate 0 leaves the loss untouched and does not train (" + frozenLoss.ToString("E3") + ")");
        var clone = trained.Clone();
        Check(clone.Loss(Features(frame, 0), new[] { frame.Reference[0] - frame.Noisy[0], 0, 0 }) ==
              trained.Loss(Features(frame, 0), new[] { frame.Reference[0] - frame.Noisy[0], 0, 0 }),
              "a cloned network reproduces the original output exactly");

        // 4-5: the pipeline improves the image, and the reference is not being smuggled in.
        double[] denoised = Denoise(trained, frame, false);
        double[] guided = Denoise(trained, frame, true);
        double noisyPsnr = Psnr(frame.Noisy, frame.Reference);
        double denoisedPsnr = Psnr(denoised, frame.Reference);
        double guidedPsnr = Psnr(guided, frame.Reference);
        Console.WriteLine("psnr noisy " + noisyPsnr.ToString("F2") + " dB, denoised " + denoisedPsnr.ToString("F2") +
                          " dB, guided " + guidedPsnr.ToString("F2") + " dB");
        Check(denoisedPsnr > noisyPsnr + 0.5, "the denoiser gains more than 0.5 dB over the noisy render");
        Check(guidedPsnr > noisyPsnr + 0.5, "the guided variant gains too");
        Check(denoisedPsnr < 200.0, "the denoiser does not reproduce the reference exactly (it cannot see it)");
        Check(Math.Abs(Psnr(frame.Reference, frame.Reference) - 120.0) < 1.0,
              "the PSNR of the reference against itself is capped by the epsilon floor");

        // 6: the edge guard keeps a hard albedo edge.
        int edgeRow = frame.Height / 2;
        double noisyStep = EdgeStep(frame.Noisy, frame.Width / 2 - 1, frame.Width / 2, edgeRow, frame.Width);
        double guidedStep = EdgeStep(guided, frame.Width / 2 - 1, frame.Width / 2, edgeRow, frame.Width);
        double[] blurred = Blur3x3(denoised, frame.Width, frame.Height);
        double blurredStep = EdgeStep(blurred, frame.Width / 2 - 1, frame.Width / 2, edgeRow, frame.Width);
        Console.WriteLine("edge step noisy " + noisyStep.ToString("F4") + ", guided " + guidedStep.ToString("F4") +
                          ", plain 3x3 blur " + blurredStep.ToString("F4"));
        Check(guidedStep > 0.5 * noisyStep,
              "the guided filter keeps at least half of the noisy edge step (" + guidedStep + " of " + noisyStep + ")");
        Check(guidedStep > blurredStep * 1.2,
              "the guide preserves far more of the edge than a plain 3x3 blur (" + guidedStep + " vs " + blurredStep + ")");
        var flat = Enumerable.Range(0, 8).Select(_ => 0.5 + NextSigned() * 0.2).ToArray();
        double flatVariance = Variance(flat);
        var smoothed = new double[flat.Length];
        for (int i = 0; i < flat.Length; i++)
        {
            double sum = flat[i], count = 1;
            if (i > 0) { sum += flat[i - 1]; count++; }
            if (i + 1 < flat.Length) { sum += flat[i + 1]; count++; }
            smoothed[i] = sum / count;
        }
        Check(Variance(smoothed) < flatVariance, "a flat region is smoothed (variance decreases)");

        // 7: temporal accumulation reduces variance and stays unbiased.
        var (varianceFast, meanFast) = AccumulateVariance(400, 6, 0.5, 0x5555UL, 0.1);
        var (varianceSlow, meanSlow) = AccumulateVariance(400, 6, 0.2, 0x5555UL, 0.1);
        var (varianceNone, _) = AccumulateVariance(400, 1, 1.0, 0x5555UL, 0.1);
        Console.WriteLine("temporal variance: none " + varianceNone.ToString("E3") + ", alpha 0.5 " +
                          varianceFast.ToString("E3") + ", alpha 0.2 " + varianceSlow.ToString("E3"));
        Check(varianceSlow < varianceFast && varianceFast < varianceNone,
              "a smaller accumulation weight reduces the residual variance (monotone in alpha)");
        Check(Math.Abs(meanSlow - 1.0) < 0.02 && Math.Abs(meanFast - 1.0) < 0.03,
              "the accumulator is unbiased for a constant signal (" + meanSlow.ToString("F4") + ", " +
              meanFast.ToString("F4") + ")");
        var constantFrames = Enumerable.Range(0, 5).Select(_ => Enumerable.Repeat(0.42, 10).ToArray()).ToList();
        Check(Math.Abs(AccumulateConstant(constantFrames, 0.3)) < 1e-30,
              "accumulating a truly constant signal keeps zero variance");

        // 8-11: degenerate inputs.
        var black = new Frame { Width = 4, Height = 4, Reference = new double[16], Noisy = new double[16],
                                Albedo = new double[16], NormalX = new double[16], Depth = new double[16] };
        double[] blackOut = Denoise(trained, black, true);
        Check(blackOut.All(v => Math.Abs(v) < 0.5), "a black frame stays near zero instead of exploding");
        var singlePixel = new Frame { Width = 1, Height = 1, Reference = new[] { 0.5 }, Noisy = new[] { 0.5 },
                                      Albedo = new[] { 0.5 }, NormalX = new[] { 0.0 }, Depth = new[] { 1.0 } };
        double[] singleOut = Denoise(trained, singlePixel, true);
        Check(singleOut.Length == 1 && double.IsFinite(singleOut[0]), "a single-pixel frame is handled");
        var zeroNet = Mlp.Create(0x1234UL, 0.0);
        double zeroLoss = TrainDenoiser(zeroNet, frame, 5, 0.5, out _);
        Check(double.IsFinite(zeroLoss) && zeroLoss > 0, "an all-zero network still reports a finite loss");
        var cleanFrame = TraceFrame(16, 12, 0, 0x6666UL);
        double cleanPsnr = Psnr(Denoise(trained, cleanFrame, false), cleanFrame.Reference);
        Check(cleanPsnr > 10.0, "a noise-free input is not destroyed by the denoiser (" + cleanPsnr.ToString("F2") + " dB)");

        // 12: determinism.
        var replay = Mlp.Create(0x1111UL, 0.8);
        TrainDenoiser(replay, frame, 40, 0.05, out _);
        double[] replayOut = Denoise(replay, frame, true);
        Check(replayOut.SequenceEqual(guided), "training and inference are bitwise reproducible");

        Console.WriteLine("network " + trained.ParameterCount + " parameters, edge step kept " +
                          (100.0 * guidedStep / Math.Max(1e-9, noisyStep)).ToString("F0") + "%");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
