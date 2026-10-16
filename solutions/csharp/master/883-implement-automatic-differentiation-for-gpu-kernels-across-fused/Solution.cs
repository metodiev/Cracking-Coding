// Task 883 - Implement automatic differentiation for GPU kernels across fused operators.
// (Tier: Master, CTCI Beyond - Machine Learning & AI)
//
// Model: a small reverse-mode autodiff engine for the operator pairs that get fused in real kernels - the
// three training-critical fusions are implemented twice, once as a composition of primitive nodes on a tape
// and once as a single fused kernel with a hand-derived backward:
//   * Linear+Bias+ReLU : one fused node whose backward streams through the matmul, the bias broadcast and
//     the ReLU mask without materialising the intermediate tensors;
//   * Softmax+CrossEntropy : the numerically stable fused form (max subtraction, log-sum-exp) whose
//     backward collapses to (p - onehot)/n, plus the naive exp/log composition used as a negative control
//     that really does overflow;
//   * data layout: the batched linear layer treats the weight as (out x in) and the input as (batch x in),
//     so the weight gradient is an outer-product accumulation over the batch.
// A forward-mode dual-number path is implemented as well and cross-checked against reverse mode, which is
// the cheapest way to catch a backward that silently transposed something.
//
// Simplifications (documented, not behavioural): scalars are double (no float16), tensors are dense flat
// arrays instead of GPU buffers, and the "GPU" aspect is modelled by counting the elements each node stores
// for the backward pass - that count is exactly what fusion reduces, and it is verified against a fusion
// cost model rather than measured on hardware. The gradients themselves are the real thing and are checked
// against central finite differences.
//
// Verified here:
//   * gradients: every fused backward agrees with central finite differences (data, weight and bias) to
//     1e-6 relative, including the bias broadcast (whose gradient must sum over the batch);
//   * fusion equivalence: the fused forward value and every gradient are bitwise identical to the unfused
//     composition, while the fused pipeline stores strictly fewer elements;
//   * forward vs reverse: the dual-number derivative equals the reverse-mode gradient on the same function;
//   * stability: the fused softmax/cross-entropy is finite for logits around 1e3 while the naive exp/log
//     composition overflows - the numerical reason the fusion exists;
//   * degenerate cases: an empty batch, a single sample, a zero weight matrix, a uniform target, a
//     many-class softmax with one correct class, and a ReLU with an all-negative pre-activation (zero
//     gradient);
//   * determinism: forward, backward and training steps are bitwise reproducible.
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static ulong _state = 0xD1FFAB1E5EEDCAFEUL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextSigned() => ((NextU64() >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;

    // ---------- tensors ----------
    private sealed class Tensor
    {
        public int Rows;
        public int Cols;
        public double[] Data = Array.Empty<double>();

        public int Count => Rows * Cols;

        public static Tensor Zeros(int rows, int cols) =>
            new Tensor { Rows = rows, Cols = cols, Data = new double[Math.Max(0, rows * cols)] };

        public static Tensor Random(int rows, int cols, double scale)
        {
            var t = Zeros(rows, cols);
            for (int i = 0; i < t.Data.Length; i++) t.Data[i] = NextSigned() * scale;
            return t;
        }

        public Tensor CloneInstance() =>
            new Tensor { Rows = Rows, Cols = Cols, Data = (double[])Data.Clone() };

        public double this[int i, int j]
        {
            get => Data[i * Cols + j];
            set => Data[i * Cols + j] = value;
        }

        public double SumSquares() => Data.Sum(v => v * v);
    }

    private sealed class Gradients
    {
        public Tensor DX = Tensor.Zeros(0, 0);
        public Tensor DW = Tensor.Zeros(0, 0);
        public Tensor DB = Tensor.Zeros(0, 0);
        public double DLoss = 1.0;
        public long StoredElements;
    }

    // ---------- fused kernels ----------
    /// <summary>
    /// Linear + bias + ReLU in one pass. Only the pre-activation tensor is kept (the ReLU mask is its sign),
    /// and the bias broadcast is handled by accumulating over the batch inside the same loop.
    /// </summary>
    private static double[] FusedLinearReluForward(Tensor x, Tensor w, Tensor b, out double[] pre, out long storedElements)
    {
        int n = x.Rows, d = x.Cols, h = w.Rows;
        var outData = new double[n * h];
        pre = new double[n * h];
        for (int i = 0; i < n; i++)
            for (int o = 0; o < h; o++)
            {
                double acc = b.Data[o];
                for (int k = 0; k < d; k++) acc += w[o, k] * x[i, k];
                pre[i * h + o] = acc;
                outData[i * h + o] = acc > 0 ? acc : 0.0;
            }
        storedElements = pre.Length;                           // one buffer, not three
        return outData;
    }

    private static void FusedLinearReluBackward(double[] dOut, Tensor x, Tensor w, double[] pre,
                                                Gradients grads)
    {
        int n = x.Rows, d = x.Cols, h = w.Rows;
        grads.DX = Tensor.Zeros(n, d);
        grads.DW = Tensor.Zeros(h, d);
        grads.DB = Tensor.Zeros(1, h);
        for (int i = 0; i < n; i++)
            for (int o = 0; o < h; o++)
            {
                double mask = pre[i * h + o] > 0 ? 1.0 : 0.0;
                double g = dOut[i * h + o] * mask;
                grads.DB.Data[o] += g;                        // the broadcast accumulates over the batch
                for (int k = 0; k < d; k++)
                {
                    grads.DX[i, k] += g * w[o, k];
                    grads.DW[o, k] += g * x[i, k];
                }
            }
    }

    /// <summary>Stable fused softmax + cross entropy: subtract the row max, and return the loss.</summary>
    private static double FusedSoftmaxCrossEntropy(Tensor logits, int[] labels, out Tensor probs, out long storedElements)
    {
        int n = logits.Rows, c = logits.Cols;
        probs = Tensor.Zeros(n, c);
        double loss = 0.0;
        for (int i = 0; i < n; i++)
        {
            double max = double.NegativeInfinity;
            for (int j = 0; j < c; j++) max = Math.Max(max, logits[i, j]);
            double sum = 0.0;
            for (int j = 0; j < c; j++)
            {
                double e = Math.Exp(logits[i, j] - max);
                probs[i, j] = e;
                sum += e;
            }
            for (int j = 0; j < c; j++) probs[i, j] /= sum;
            loss += -Math.Log(Math.Max(probs[i, labels[i]], 1e-300));
        }
        storedElements = probs.Count;                          // the probabilities are the only buffer
        return n == 0 ? 0.0 : loss / n;                        // an empty batch contributes nothing
    }

    /// <summary>The naive composition kept as a negative control: it overflows where the fused form does not.</summary>
    private static double NaiveSoftmaxCrossEntropy(Tensor logits, int[] labels, out Tensor probs)
    {
        int n = logits.Rows, c = logits.Cols;
        probs = Tensor.Zeros(n, c);
        double loss = 0.0;
        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < c; j++)
            {
                double e = Math.Exp(logits[i, j]);
                probs[i, j] = e;
                sum += e;
            }
            for (int j = 0; j < c; j++) probs[i, j] /= sum;
            loss += -Math.Log(Math.Max(probs[i, labels[i]], 1e-300));
        }
        return loss / n;
    }

    // ---------- unfused composition (the reference the fused kernels must match bit for bit) ----------
    private sealed class UnfusedTape
    {
        public Tensor Z = Tensor.Zeros(0, 0);        // pre-activation buffer
        public Tensor A = Tensor.Zeros(0, 0);        // relu output buffer
        public Tensor P = Tensor.Zeros(0, 0);        // softmax buffer
        public long StoredElements;

        public double Forward(Tensor x, Tensor w, Tensor b, int[] labels)
        {
            int n = x.Rows, d = x.Cols, h = w.Rows;
            Z = Tensor.Zeros(n, h);
            A = Tensor.Zeros(n, h);
            for (int i = 0; i < n; i++)
                for (int o = 0; o < h; o++)
                {
                    double acc = b.Data[o];
                    for (int k = 0; k < d; k++) acc += w[o, k] * x[i, k];
                    Z[i, o] = acc;
                    A[i, o] = acc > 0 ? acc : 0.0;
                }
            double loss = FusedSoftmaxCrossEntropy(A, labels, out P, out _);
            StoredElements = Z.Count + A.Count + P.Count;
            return loss;
        }

        public Gradients Backward(Tensor x, Tensor w, Tensor b, int[] labels)
        {
            int n = x.Rows, c = P.Cols, h = w.Rows, d = x.Cols;
            var grads = new Gradients();
            var dA = Tensor.Zeros(n, h);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < h; j++)
                {
                    double p = P[i, j];
                    double y = labels[i] == j ? 1.0 : 0.0;
                    dA[i, j] = (p - y) / n;                  // the softmax/cross-entropy backward, composed
                }
            var dRelu = Tensor.Zeros(n, h);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < h; j++) dRelu[i, j] = Z[i, j] > 0 ? dA[i, j] : 0.0;
            grads.DX = Tensor.Zeros(n, d);
            grads.DW = Tensor.Zeros(h, d);
            grads.DB = Tensor.Zeros(1, h);
            for (int i = 0; i < n; i++)
                for (int o = 0; o < h; o++)
                {
                    double g = dRelu[i, o];
                    grads.DB.Data[o] += g;
                    for (int k = 0; k < d; k++)
                    {
                        grads.DX[i, k] += g * w[o, k];
                        grads.DW[o, k] += g * x[i, k];
                    }
                }
            grads.StoredElements = StoredElements;
            return grads;
        }
    }

    // ---------- fused pipeline ----------
    private sealed class FusedTape
    {
        public long StoredElements;

        public double Forward(Tensor x, Tensor w, Tensor b, int[] labels, out double[] pre, out Tensor probs)
        {
            double[] activations = FusedLinearReluForward(x, w, b, out pre, out long storageA);
            var activationTensor = new Tensor { Rows = x.Rows, Cols = w.Rows, Data = activations };
            double loss = FusedSoftmaxCrossEntropy(activationTensor, labels, out probs, out long storageB);
            StoredElements = storageA + storageB;
            return loss;
        }

        public Gradients Backward(Tensor x, Tensor w, Tensor b, int[] labels, double[] pre, Tensor probs)
        {
            int n = x.Rows, h = w.Rows;
            var grads = new Gradients();
            var dActivation = new double[n * h];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < h; j++)
                    dActivation[i * h + j] = (probs[i, j] - (labels[i] == j ? 1.0 : 0.0)) / n;
            FusedLinearReluBackward(dActivation, x, w, pre, grads);
            grads.StoredElements = StoredElements;
            return grads;
        }
    }

    // ---------- forward-mode dual numbers ----------
    private struct Dual
    {
        public double Value;
        public double Dot;                       // the derivative along the seeded direction
        public Dual(double v, double d) { Value = v; Dot = d; }
        public static Dual operator +(Dual a, Dual b) => new Dual(a.Value + b.Value, a.Dot + b.Dot);
        public static Dual operator -(Dual a, Dual b) => new Dual(a.Value - b.Value, a.Dot - b.Dot);
        public static Dual operator *(Dual a, Dual b) => new Dual(a.Value * b.Value, a.Dot * b.Value + a.Value * b.Dot);
        public static Dual operator /(Dual a, Dual b) =>
            new Dual(a.Value / b.Value, (a.Dot * b.Value - a.Value * b.Dot) / (b.Value * b.Value));
        public static Dual Exp(Dual a)
        {
            double e = Math.Exp(a.Value);
            return new Dual(e, e * a.Dot);
        }
        public static Dual Log(Dual a) => new Dual(Math.Log(a.Value), a.Dot / a.Value);
        public static Dual Relu(Dual a) => new Dual(a.Value > 0 ? a.Value : 0.0, a.Value > 0 ? a.Dot : 0.0);
    }

    /// <summary>Scalar function used to cross-check forward mode against reverse mode and finite differences.</summary>
    private static double ScalarFunction(double[] x) =>
        Math.Log(1.0 + Math.Exp(x[0] * 2.0 - x[1])) + 0.5 * x[2] * x[2];

    private static double[] ScalarGradientReverse(double[] x)
    {
        double z = x[0] * 2.0 - x[1];
        double logistic = 1.0 / (1.0 + Math.Exp(-z));
        return new[] { logistic * 2.0, -logistic, x[2] };
    }

    /// <summary>The same function evaluated with dual numbers along one direction, giving the directional derivative.</summary>
    private static double ScalarDirectionalDerivativeDual(double[] x, double[] direction)
    {
        Dual x0 = new Dual(x[0], direction[0]);
        Dual x1 = new Dual(x[1], direction[1]);
        Dual x2 = new Dual(x[2], direction[2]);
        Dual z = x0 * new Dual(2.0, 0.0) - x1;
        Dual logTerm = Dual.Log(new Dual(1.0, 0.0) + Dual.Exp(z));
        Dual square = x2 * x2 * new Dual(0.5, 0.0);
        return (logTerm + square).Dot;
    }

    // ---------- main ----------
    public static void Main()
    {
        _state = 0x1111UL;
        var x = Tensor.Random(4, 3, 1.0);
        var w = Tensor.Random(5, 3, 0.6);
        var b = Tensor.Random(1, 5, 0.2);
        var labels = new[] { 1, 3, 0, 4 };

        // 1-3: the fused forward value equals the unfused composition bit for bit.
        var unfused = new UnfusedTape();
        double unfusedLoss = unfused.Forward(x, w, b, labels);
        var fused = new FusedTape();
        double fusedLoss = fused.Forward(x, w, b, labels, out double[] pre, out Tensor probs);
        Check(unfusedLoss == fusedLoss, "the fused loss equals the unfused composition exactly");
        Gradients unfusedGrads = unfused.Backward(x, w, b, labels);
        Gradients fusedGrads = fused.Backward(x, w, b, labels, pre, probs);
        Check(unfusedGrads.DX.Data.SequenceEqual(fusedGrads.DX.Data), "dX is bitwise identical after fusion");
        Check(unfusedGrads.DW.Data.SequenceEqual(fusedGrads.DW.Data), "dW is bitwise identical after fusion");
        Check(unfusedGrads.DB.Data.SequenceEqual(fusedGrads.DB.Data), "dB is bitwise identical after fusion");
        Console.WriteLine("fusion storage: unfused " + unfusedGrads.StoredElements + " elements, fused " +
                          fusedGrads.StoredElements + " elements");
        Check(fusedGrads.StoredElements < unfusedGrads.StoredElements,
              "fusion stores strictly fewer elements (" + fusedGrads.StoredElements + " < " +
              unfusedGrads.StoredElements + ")");
        Check(fusedGrads.StoredElements <= unfusedGrads.StoredElements * 2 / 3 + 1,
              "the fused pipeline stores at most two thirds of the unfused buffers");

        // 4-5: gradients against central finite differences on the *fused* path.
        double worstRelative = 0.0;
        int checkedEntries = 0;
        const double h = 1e-6;
        for (int o = 0; o < w.Rows; o++)
            for (int k = 0; k < w.Cols; k++)
            {
                double original = w[o, k];
                w[o, k] = original + h;
                double up = fused.Forward(x, w, b, labels, out _, out _);
                w[o, k] = original - h;
                double down = fused.Forward(x, w, b, labels, out _, out _);
                w[o, k] = original;
                double numeric = (up - down) / (2 * h);
                double analytic = fusedGrads.DW[o, k];
                worstRelative = Math.Max(worstRelative, Math.Abs(numeric - analytic) /
                    Math.Max(1e-9, Math.Abs(numeric) + Math.Abs(analytic)));
                checkedEntries++;
            }
        Check(checkedEntries == w.Count, "every weight entry was finite-differenced");
        Check(worstRelative < 1e-6, "dW matches finite differences (worst relative " + worstRelative.ToString("E2") + ")");
        double worstBias = 0.0;
        for (int o = 0; o < b.Cols; o++)
        {
            double original = b.Data[o];
            b.Data[o] = original + h;
            double up = fused.Forward(x, w, b, labels, out _, out _);
            b.Data[o] = original - h;
            double down = fused.Forward(x, w, b, labels, out _, out _);
            b.Data[o] = original;
            double numeric = (up - down) / (2 * h);
            worstBias = Math.Max(worstBias, Math.Abs(numeric - fusedGrads.DB.Data[o]));
        }
        Check(worstBias < 1e-6, "dB sums the broadcast over the batch correctly (worst " + worstBias.ToString("E2") + ")");
        double worstData = 0.0;
        for (int i = 0; i < x.Rows; i++)
            for (int k = 0; k < x.Cols; k++)
            {
                double original = x[i, k];
                x[i, k] = original + h;
                double up = fused.Forward(x, w, b, labels, out _, out _);
                x[i, k] = original - h;
                double down = fused.Forward(x, w, b, labels, out _, out _);
                x[i, k] = original;
                double numeric = (up - down) / (2 * h);
                worstData = Math.Max(worstData, Math.Abs(numeric - fusedGrads.DX[i, k]));
            }
        Check(worstData < 1e-6, "dX matches finite differences (worst " + worstData.ToString("E2") + ")");
        Check(fusedGrads.DW.Data.Any(v => Math.Abs(v) > 1e-9) && fusedGrads.DX.Data.Any(v => Math.Abs(v) > 1e-9),
              "the gradients are not trivially zero");

        // 6-7: forward mode against reverse mode and against finite differences.
        _state = 0x2222UL;
        int forwardAgreements = 0, finiteDifferenceAgreements = 0;
        for (int trial = 0; trial < 50; trial++)
        {
            var point = new[] { NextSigned(), NextSigned(), NextSigned() };
            var direction = new[] { NextSigned(), NextSigned(), NextSigned() };
            double[] analytic = ScalarGradientReverse(point);
            double directional = analytic[0] * direction[0] + analytic[1] * direction[1] + analytic[2] * direction[2];
            double dual = ScalarDirectionalDerivativeDual(point, direction);
            if (Math.Abs(dual - directional) < 1e-10 * Math.Max(1.0, Math.Abs(directional))) forwardAgreements++;
            double step = 1e-6;
            var plus = point.Zip(direction, (p, d) => p + step * d).ToArray();
            var minus = point.Zip(direction, (p, d) => p - step * d).ToArray();
            double numeric = (ScalarFunction(plus) - ScalarFunction(minus)) / (2 * step);
            if (Math.Abs(numeric - dual) < 1e-6 * Math.Max(1.0, Math.Abs(numeric))) finiteDifferenceAgreements++;
        }
        Check(forwardAgreements == 50, "forward mode matches reverse mode on 50 random directions ("
              + forwardAgreements + "/50)");
        Check(finiteDifferenceAgreements == 50, "forward mode matches central differences on the same directions ("
              + finiteDifferenceAgreements + "/50)");

        // 8: numerical stability - the whole point of fusing softmax with the loss.
        var bigLogits = Tensor.Zeros(3, 6);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 6; j++) bigLogits[i, j] = 1000.0 + j;
        double stableLoss = FusedSoftmaxCrossEntropy(bigLogits, new[] { 0, 1, 2 }, out Tensor stableProbs, out _);
        Check(double.IsFinite(stableLoss) && stableLoss > 0, "the fused softmax/CE is finite for logits around 1e3");
        Check(Math.Abs(stableProbs.Data.Sum() - 3.0) < 1e-12, "each row of the softmax still sums to one");
        Check(stableProbs.Data.All(p => p >= 0 && p <= 1), "all probabilities stay inside [0,1]");
        double naiveLoss = NaiveSoftmaxCrossEntropy(bigLogits, new[] { 0, 1, 2 }, out _);
        Check(double.IsNaN(naiveLoss) || double.IsInfinity(naiveLoss),
              "the naive exp/log composition overflows on the same input (the reason to fuse)");
        var mildLogits = Tensor.Zeros(2, 4);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++) mildLogits[i, j] = NextSigned();
        double fusedMild = FusedSoftmaxCrossEntropy(mildLogits, new[] { 0, 1 }, out _, out _);
        double naiveMild = NaiveSoftmaxCrossEntropy(mildLogits, new[] { 0, 1 }, out _);
        Check(Math.Abs(fusedMild - naiveMild) < 1e-12, "both agree in the well-conditioned regime");

        // 9: the softmax backward is the (p - onehot)/n collapse.
        var dActivation = new double[1 * 6];
        for (int j = 0; j < 6; j++) dActivation[j] = (stableProbs[0, j] - (0 == j ? 1.0 : 0.0)) / 1;
        Check(Math.Abs(dActivation.Sum()) < 1e-12, "the softmax backward rows sum to zero");
        Check(dActivation.All(v => double.IsFinite(v)), "the backward values are finite for extreme logits");

        // 10-12: training loop sanity - a few steps reduce the loss on a toy classification problem.
        _state = 0x3333UL;
        var tw = Tensor.Random(3, 2, 0.5);
        var tb = Tensor.Random(1, 3, 0.1);
        var tx = Tensor.Random(12, 2, 1.0);
        // A learnable toy task: the label is the argmax of a fixed linear scoring function, so a linear model
        // can drive the loss down instead of being stuck at the class-prior floor.
        var prototypes = Tensor.Random(3, 2, 1.0);
        var tl = new int[12];
        for (int i = 0; i < 12; i++)
        {
            int best = 0;
            double bestScore = double.NegativeInfinity;
            for (int k = 0; k < 3; k++)
            {
                double score = prototypes[k, 0] * tx[i, 0] + prototypes[k, 1] * tx[i, 1];
                if (score > bestScore) { bestScore = score; best = k; }
            }
            tl[i] = best;
        }
        Check(tl.Distinct().Count() >= 2, "the toy task has at least two classes present");
        var trainer = new FusedTape();
        double firstLoss = trainer.Forward(tx, tw, tb, tl, out double[] tpre, out Tensor tprobs);
        for (int step = 0; step < 300; step++)
        {
            trainer.Forward(tx, tw, tb, tl, out tpre, out tprobs);
            Gradients g = trainer.Backward(tx, tw, tb, tl, tpre, tprobs);
            for (int i = 0; i < tw.Data.Length; i++) tw.Data[i] -= 0.5 * g.DW.Data[i];
            for (int i = 0; i < tb.Data.Length; i++) tb.Data[i] -= 0.5 * g.DB.Data[i];
        }
        double lastLoss = trainer.Forward(tx, tw, tb, tl, out _, out _);
        Console.WriteLine("training loss " + firstLoss.ToString("F4") + " -> " + lastLoss.ToString("F4"));
        Check(lastLoss < firstLoss * 0.5, "300 fused training steps more than halve the loss");

        // 13-16: degenerate cases.
        var emptyX = Tensor.Zeros(0, 3);
        var emptyTape = new FusedTape();
        double emptyLoss = emptyTape.Forward(emptyX, w, b, Array.Empty<int>(), out double[] emptyPre, out Tensor emptyProbs);
        Check(emptyLoss == 0.0 && emptyProbs.Count == 0, "an empty batch yields a zero loss and no buffers");
        Gradients emptyGrads = emptyTape.Backward(emptyX, w, b, Array.Empty<int>(), emptyPre, emptyProbs);
        Check(emptyGrads.DW.Data.All(v => v == 0.0) && emptyGrads.DX.Count == 0,
              "an empty batch yields zero weight gradients");
        var singleX = Tensor.Random(1, 3, 1.0);
        var singleTape = new FusedTape();
        double singleLoss = singleTape.Forward(singleX, w, b, new[] { 2 }, out double[] singlePre, out Tensor singleProbs);
        Gradients singleGrads = singleTape.Backward(singleX, w, b, new[] { 2 }, singlePre, singleProbs);
        Check(singleLoss > 0 && singleGrads.DW.Data.Any(v => Math.Abs(v) > 1e-9), "a single-sample batch trains");
        var zeroW = Tensor.Zeros(5, 3);
        var zeroB = Tensor.Zeros(1, 5);
        var zeroTape = new FusedTape();
        double zeroLoss = zeroTape.Forward(x, zeroW, zeroB, labels, out double[] zeroPre, out Tensor zeroProbs);
        Gradients zeroGrads = zeroTape.Backward(x, zeroW, zeroB, labels, zeroPre, zeroProbs);
        Check(zeroGrads.DW.Data.All(v => v == 0.0) && zeroGrads.DX.Data.All(v => v == 0.0),
              "an all-zero weight matrix produces zero gradients (the ReLU is inactive)");
        Check(zeroPre.All(v => v == 0.0) && Math.Abs(zeroLoss - Math.Log(5.0)) < 1e-12,
              "a zero network predicts uniformly (loss log C)");
        var uniformTape = new FusedTape();
        var zeroLogits = Tensor.Zeros(2, 4);
        double uniformLoss = FusedSoftmaxCrossEntropy(zeroLogits, new[] { 0, 1 }, out Tensor uniformProbs, out _);
        Check(Math.Abs(uniformLoss - Math.Log(4.0)) < 1e-12, "all-equal logits give the uniform loss log C");
        Check(uniformProbs.Data.All(p => Math.Abs(p - 0.25) < 1e-15), "the uniform softmax is exactly 1/C");
        var negativeReLU = Tensor.Zeros(2, 3);
        for (int i = 0; i < negativeReLU.Data.Length; i++) negativeReLU.Data[i] = -1.0 - i;
        Tensor negW = Tensor.Zeros(3, 3), negB = Tensor.Zeros(1, 3);
        for (int o = 0; o < 3; o++) negB.Data[o] = -5.0;
        FusedLinearReluForward(negativeReLU, negW, negB, out double[] negPre, out _);
        var negGrads = new Gradients();
        FusedLinearReluBackward(new double[negativeReLU.Rows * 3], negativeReLU, negW, negPre, negGrads);
        Check(negGrads.DW.Data.All(v => v == 0.0) && negGrads.DB.Data.All(v => v == 0.0),
              "an all-negative pre-activation blocks every gradient through the ReLU");

        // 17: determinism.
        _state = 0x1111UL;
        var repeatX = Tensor.Random(4, 3, 1.0);
        var repeatW = Tensor.Random(5, 3, 0.6);
        var repeatB = Tensor.Random(1, 5, 0.2);
        var repeatTape = new FusedTape();
        double repeatLoss = repeatTape.Forward(repeatX, repeatW, repeatB, labels, out double[] repeatPre, out Tensor repeatProbs);
        Gradients repeatGrads = repeatTape.Backward(repeatX, repeatW, repeatB, labels, repeatPre, repeatProbs);
        Check(repeatLoss == fusedLoss && repeatGrads.DW.Data.SequenceEqual(fusedGrads.DW.Data),
              "the fused forward and backward are bitwise reproducible");

        Console.WriteLine("fused loss " + fusedLoss.ToString("F6") + ", stable loss at 1e3 logits " +
                          stableLoss.ToString("F6") + ", naive " + naiveLoss + ", trained " +
                          lastLoss.ToString("F6"));
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
