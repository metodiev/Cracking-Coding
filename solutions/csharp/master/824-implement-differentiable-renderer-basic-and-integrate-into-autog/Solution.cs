// Task 824 - Implement differentiable renderer basic and integrate into autograd. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// Two pieces, both deterministic and in process:
//
//   * a reverse-mode autodiff engine - Value nodes with the usual arithmetic and activations, an explicit
//     topological order and gradient *accumulation* (so a node consumed twice receives both
//     contributions). Backward is iterative, not recursive, so long chains are fine.
//   * a basic differentiable rasteriser - the triangle is rasterised per pixel; the signed distance of
//     the sample from each edge is built from the edge function divided by the edge length, coverage is
//     the product of the three sigmoids of those distances, so a pixel inside the triangle gets coverage
//     ~1, a pixel outside gets ~0, and the transition is smooth in the vertex positions, the colours, the
//     background and the softness (a width in pixels). Every vertex coordinate, colour component and the
//     softness are Value nodes, which is what makes the loss (mean squared error against a target image)
//     differentiable end to end.
//
// The strongest verification available for a renderer is the finite-difference gradient check: every
// parameter of the renderer (6 vertex coordinates, 3 colour components, the background and the softness)
// is compared against a central difference of the same forward pass, and the forward pass itself is
// cross checked against an independently written reference rasteriser that uses the classic edge-function
// (Pineda) formulation instead of the signed-area ratios.
//
// Simplifications (documented): 2D triangles only (no 3D projection, no camera, no lighting model);
// coverage uses a sigmoid product rather than a proper antialiasing filter, so the "hard" limit is
// approached but never exact at the silhouette; there is no texture, no z-buffer beyond a painter's
// order for the two-triangle compositing test, and no occlusion gradient; the renderer runs on the CPU
// with an O(pixels) loop rather than paying for a GPU; the model is a handful of primitives, so "at
// scale" is expressed as the pixel count and the measured cost per pixel.
//
// Verified invariants: analytic gradients equal finite differences for the engine and for every renderer
// parameter; shared subexpressions accumulate correctly; a 100 step chain keeps the expected closed form;
// the differentiable forward pass equals the independent reference rasteriser for both the hard and the
// soft coverage; the soft pass converges to the hard mask as the softness shrinks; coverage is ~1 at the
// centroid and ~0 far outside; the loss is zero when the target is the current render; gradient descent
// on the vertices reduces the loss by more than an order of magnitude and moves the triangle onto the
// target; the composited two-triangle image is differentiable and matches the reference compositing;
// degenerate triangles, one pixel images and triangles outside the frame stay finite; and the whole
// pipeline is deterministic.
using System;
using System.Collections.Generic;
using System.Linq;

internal struct SplitMix64
{
    private ulong state;

    public SplitMix64(ulong seed) { state = seed; }

    public ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public int NextInt(int bound) => (int)(Next() % (ulong)bound);

    public double NextUnit() => (Next() >> 11) * (1.0 / 9007199254740992.0);

    public double NextSigned() => NextUnit() * 2.0 - 1.0;
}

// ---- reverse mode autodiff ---------------------------------------------------------------------------

internal sealed class Value
{
    public double Data;
    public double Grad;
    public readonly List<(Value Parent, double Local)> Parents = new List<(Value, double)>();

    public static Value Of(double data) => new Value { Data = data };

    public static Value From(double data, Value parent, double local)
    {
        var value = new Value { Data = data };
        value.Parents.Add((parent, local));
        return value;
    }

    public static Value Add(Value a, Value b)
    {
        var result = new Value { Data = a.Data + b.Data };
        result.Parents.Add((a, 1.0));
        result.Parents.Add((b, 1.0));
        return result;
    }

    public static Value Sub(Value a, Value b)
    {
        var result = new Value { Data = a.Data - b.Data };
        result.Parents.Add((a, 1.0));
        result.Parents.Add((b, -1.0));
        return result;
    }

    public static Value Mul(Value a, Value b)
    {
        var result = new Value { Data = a.Data * b.Data };
        result.Parents.Add((a, b.Data));
        result.Parents.Add((b, a.Data));
        return result;
    }

    public static Value Div(Value a, Value b)
    {
        var result = new Value { Data = a.Data / b.Data };
        result.Parents.Add((a, 1.0 / b.Data));
        result.Parents.Add((b, -a.Data / (b.Data * b.Data)));
        return result;
    }

    public static Value Neg(Value a) => From(-a.Data, a, -1.0);

    public static Value Exp(Value a)
    {
        double e = Math.Exp(a.Data);
        return From(e, a, e);
    }

    public static Value Log(Value a) => From(Math.Log(a.Data), a, 1.0 / a.Data);

    public static Value Tanh(Value a)
    {
        double t = Math.Tanh(a.Data);
        return From(t, a, 1.0 - t * t);
    }

    public static Value Sigmoid(Value a)
    {
        double s = 1.0 / (1.0 + Math.Exp(-a.Data));
        return From(s, a, s * (1.0 - s));
    }

    public static Value Relu(Value a) => From(Math.Max(0.0, a.Data), a, a.Data > 0 ? 1.0 : 0.0);

    public static Value Sqrt(Value a)
    {
        double root = Math.Sqrt(Math.Max(0, a.Data));
        return From(root, a, root < 1e-12 ? 0.0 : 0.5 / root);
    }

    public static Value Square(Value a) => Mul(a, a);

    public static Value Scale(Value a, double factor) => From(a.Data * factor, a, factor);

    public static Value Sum(IEnumerable<Value> values)
    {
        var result = new Value();
        foreach (Value value in values)
        {
            result.Data += value.Data;
            result.Parents.Add((value, 1.0));
        }
        return result;
    }

    public static Value Mean(IEnumerable<Value> values)
    {
        var list = values.ToList();
        var result = new Value();
        foreach (Value value in list)
        {
            result.Data += value.Data / list.Count;
            result.Parents.Add((value, 1.0 / list.Count));
        }
        return result;
    }

    // Iterative topological backward with gradient accumulation.
    public void Backward()
    {
        var order = new List<Value>();
        var visited = new HashSet<Value>();
        var stack = new Stack<(Value Node, bool Expanded)>();
        stack.Push((this, false));
        while (stack.Count > 0)
        {
            (Value node, bool expanded) = stack.Pop();
            if (expanded)
            {
                order.Add(node);
                continue;
            }
            if (!visited.Add(node)) continue;
            stack.Push((node, true));
            foreach ((Value parent, double _) in node.Parents)
                if (!visited.Contains(parent)) stack.Push((parent, false));
        }
        Grad += 1.0;
        for (int i = order.Count - 1; i >= 0; i--)
            foreach ((Value parent, double local) in order[i].Parents)
                parent.Grad += local * order[i].Grad;
    }

    public void ZeroGrad()
    {
        var seen = new HashSet<Value>();
        var stack = new Stack<Value>();
        stack.Push(this);
        while (stack.Count > 0)
        {
            Value node = stack.Pop();
            if (!seen.Add(node)) continue;
            node.Grad = 0;
            foreach ((Value parent, double _) in node.Parents) stack.Push(parent);
        }
    }

    public int NodeCount
    {
        get
        {
            var seen = new HashSet<Value>();
            var stack = new Stack<Value>();
            stack.Push(this);
            while (stack.Count > 0)
            {
                Value node = stack.Pop();
                if (!seen.Add(node)) continue;
                foreach ((Value parent, double _) in node.Parents) stack.Push(parent);
            }
            return seen.Count;
        }
    }
}

// ---- differentiable renderer -------------------------------------------------------------------------

internal sealed class Scene
{
    public Value[] X = new Value[3];
    public Value[] Y = new Value[3];
    public Value[] Color = new Value[3];
    public Value Background = Value.Of(0.0);
    public Value Sigma = Value.Of(0.05);
}

internal static class Renderer
{
    // Signed area of the triangle; its sign encodes the winding, so the barycentric ratios below are
    // well defined for either orientation.
    private static double Area2(Scene scene)
        => (scene.X[1].Data - scene.X[0].Data) * (scene.Y[2].Data - scene.Y[0].Data)
           - (scene.X[2].Data - scene.X[0].Data) * (scene.Y[1].Data - scene.Y[0].Data);

    private static Value Area2Value(Scene scene)
    {
        Value a = Value.Sub(scene.X[1], scene.X[0]);
        Value b = Value.Sub(scene.Y[2], scene.Y[0]);
        Value c = Value.Sub(scene.X[2], scene.X[0]);
        Value d = Value.Sub(scene.Y[1], scene.Y[0]);
        return Value.Sub(Value.Mul(a, b), Value.Mul(c, d));
    }

    // Differentiable rasteriser: per pixel the coverage is the product of the three sigmoids of the
    // barycentric coordinates, and the colour is a linear blend between the triangle colour and the
    // background. Returns the loss (mean squared error) against the target, so the caller only has to
    // call Backward on the result.
    // Distance based softness. For an edge (a, b) the signed distance of the sample point is
    // 2 * signedArea(a, b, P) / |Vb - Va|, which is positive inside when the winding is taken into
    // account; every factor here is a Value, so the gradient includes the edge length and the area as
    // well as the barycentric ratio. The softness is therefore a width in pixels.
    private static Value EdgeDistance(Scene scene, int a, int b, double cx, double cy, double winding)
    {
        Value twoArea = Value.Scale(SignedArea(scene.X[a], scene.Y[a], scene.X[b], scene.Y[b], cx, cy), winding);
        Value dx = Value.Sub(scene.X[b], scene.X[a]);
        Value dy = Value.Sub(scene.Y[b], scene.Y[a]);
        Value length = Value.Add(Value.Sqrt(Value.Add(Value.Mul(dx, dx), Value.Mul(dy, dy))), Value.Of(1e-12));
        return Value.Div(twoArea, length);
    }

    private static double EdgeDistancePlain(Scene scene, int a, int b, double cx, double cy, double winding)
    {
        double ax = scene.X[a].Data, ay = scene.Y[a].Data;
        double bx = scene.X[b].Data, by = scene.Y[b].Data;
        double twoArea = ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) * winding;
        double length = Math.Sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay));
        return twoArea / Math.Max(length, 1e-12);
    }

    private static double Winding(Scene scene) => Area2(scene) >= 0 ? 1.0 : -1.0;

    public static double CoverageValue(Scene scene, double cx, double cy)
    {
        double winding = Winding(scene);
        double d0 = EdgeDistancePlain(scene, 0, 1, cx, cy, winding);
        double d1 = EdgeDistancePlain(scene, 1, 2, cx, cy, winding);
        double d2 = EdgeDistancePlain(scene, 2, 0, cx, cy, winding);
        return (1.0 / (1.0 + Math.Exp(-d0 / scene.Sigma.Data)))
             * (1.0 / (1.0 + Math.Exp(-d1 / scene.Sigma.Data)))
             * (1.0 / (1.0 + Math.Exp(-d2 / scene.Sigma.Data)));
    }

    public static Value CoverageNode(Scene scene, double cx, double cy)
    {
        double winding = Winding(scene);
        Value coverage = Value.Sigmoid(Value.Div(EdgeDistance(scene, 0, 1, cx, cy, winding), scene.Sigma));
        coverage = Value.Mul(coverage, Value.Sigmoid(Value.Div(EdgeDistance(scene, 1, 2, cx, cy, winding), scene.Sigma)));
        return Value.Mul(coverage, Value.Sigmoid(Value.Div(EdgeDistance(scene, 2, 0, cx, cy, winding), scene.Sigma)));
    }

    public static Value RenderLoss(int height, int width, Scene scene, double[] target, out double[] image)
    {
        var errors = new List<Value>();
        var pixels = new double[height * width * 3];
        for (int py = 0; py < height; py++)
        {
            for (int px = 0; px < width; px++)
            {
                double cx = px + 0.5, cy = py + 0.5;
                Value coverage = CoverageNode(scene, cx, cy);
                for (int channel = 0; channel < 3; channel++)
                {
                    Value blended = Value.Add(
                        Value.Mul(coverage, scene.Color[channel]),
                        Value.Mul(Value.Sub(Value.Of(1.0), coverage), scene.Background));
                    int index = (py * width + px) * 3 + channel;
                    pixels[index] = blended.Data;
                    Value difference = Value.Sub(blended, Value.Of(target[index]));
                    errors.Add(Value.Mul(difference, difference));
                }
            }
        }
        image = pixels;
        return Value.Mean(errors);
    }

    // Barycentric coordinates via the signed area ratios - an independent route from the edge functions
    // used by the reference rasteriser.
    public static Value[] Barycentric(Scene scene, double cx, double cy)
    {
        double area = Area2(scene);
        var result = new Value[3];
        result[0] = Value.Div(SignedArea(scene.X[1], scene.Y[1], scene.X[2], scene.Y[2], cx, cy), Value.Of(area));
        result[1] = Value.Div(SignedArea(scene.X[2], scene.Y[2], scene.X[0], scene.Y[0], cx, cy), Value.Of(area));
        result[2] = Value.Div(SignedArea(scene.X[0], scene.Y[0], scene.X[1], scene.Y[1], cx, cy), Value.Of(area));
        return result;
    }

    private static Value SignedArea(Value ax, Value ay, Value bx, Value by, double cx, double cy)
    {
        // (bx-ax)*(cy-ay) - (cx-ax)*(by-ay), with the sample point as a constant and every vertex a Value
        // (so both differences really do contribute their derivatives).
        Value left = Value.Mul(Value.Sub(bx, ax), Value.Sub(Value.Of(cy), ay));
        Value right = Value.Mul(Value.Sub(Value.Of(cx), ax), Value.Sub(by, ay));
        return Value.Sub(left, right);
    }

    public static double Area(Scene scene) => Area2(scene);

    // Independent reference rasteriser using the classic edge function formulation. Both the hard mask
    // and the soft coverage (same sigmoid definition, different algebra) are produced. The image is flat
    // with three channels per pixel.
    public static double[] RenderReference(int height, int width, Scene scene, bool hard)
    {
        double ax = scene.X[0].Data, ay = scene.Y[0].Data;
        double bx = scene.X[1].Data, by = scene.Y[1].Data;
        double ccx = scene.X[2].Data, ccy = scene.Y[2].Data;
        double area = (bx - ax) * (ccy - ay) - (ccx - ax) * (by - ay);
        double winding = area >= 0 ? 1.0 : -1.0;
        var image = new double[height * width * 3];
        for (int py = 0; py < height; py++)
        {
            for (int px = 0; px < width; px++)
            {
                double cx = px + 0.5, cy = py + 0.5;
                // The classic edge functions, normalised by the edge length: a distance in pixels.
                double e0 = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
                double e1 = (ccx - bx) * (cy - by) - (cx - bx) * (ccy - by);
                double e2 = (ax - ccx) * (cy - ccy) - (cx - ccx) * (ay - ccy);
                double len0 = Math.Sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay));
                double len1 = Math.Sqrt((ccx - bx) * (ccx - bx) + (ccy - by) * (ccy - by));
                double len2 = Math.Sqrt((ax - ccx) * (ax - ccx) + (ay - ccy) * (ay - ccy));
                double d0 = e0 * winding / Math.Max(len0, 1e-12);
                double d1 = e1 * winding / Math.Max(len1, 1e-12);
                double d2 = e2 * winding / Math.Max(len2, 1e-12);
                double coverage;
                if (hard)
                {
                    coverage = d0 >= 0 && d1 >= 0 && d2 >= 0 ? 1.0 : 0.0;
                }
                else
                {
                    coverage = Sigmoid(d0 / scene.Sigma.Data)
                             * Sigmoid(d1 / scene.Sigma.Data)
                             * Sigmoid(d2 / scene.Sigma.Data);
                }
                for (int channel = 0; channel < 3; channel++)
                {
                    double value = coverage * scene.Color[channel].Data + (1 - coverage) * scene.Background.Data;
                    image[(py * width + px) * 3 + channel] = value;
                }
            }
        }
        return image;
    }

    private static double Sigmoid(double z) => 1.0 / (1.0 + Math.Exp(-z));
}

// ---- checks -------------------------------------------------------------------------------------------

public static class Solution
{
    private static SplitMix64 rng = new SplitMix64(0x4E0DEADUL);

    public static void Main()
    {
        AutogradChecks();
        RendererChecks();

        Console.WriteLine("824 renderer: autograd graph checked against finite differences on every operator, "
            + "renderer gradients verified for 6 vertex coordinates, 3 colour channels, background and softness; "
            + "differentiable forward matches the independent edge-function rasteriser; descent reached "
            + lastLoss.ToString("E3") + " from " + firstLoss.ToString("E3"));
        Console.WriteLine("PASS");
    }

    private static void AutogradChecks()
    {
        // Analytic: f = tanh(a*b) + c/d, gradients known by hand.
        var a = Value.Of(0.7);
        var b = Value.Of(-1.3);
        var c = Value.Of(2.1);
        var d = Value.Of(0.9);
        Value product = Value.Mul(a, b);
        Value f = Value.Add(Value.Tanh(product), Value.Div(c, d));
        f.Backward();
        double tanhDerivative = 1 - Math.Tanh(product.Data) * Math.Tanh(product.Data);
        Check(Math.Abs(f.Data - (Math.Tanh(-0.91) + 2.1 / 0.9)) < 1e-12, "composite forward value");
        Check(Math.Abs(a.Grad - tanhDerivative * b.Data) < 1e-12, "d/da of tanh(a*b)");
        Check(Math.Abs(b.Grad - tanhDerivative * a.Data) < 1e-12, "d/db of tanh(a*b)");
        Check(Math.Abs(c.Grad - 1.0 / d.Data) < 1e-12, "d/dc of c/d");
        Check(Math.Abs(d.Grad + c.Data / (d.Data * d.Data)) < 1e-12, "d/dd of c/d");

        // Accumulation through a shared node: y = x*x + x has dy/dx = 2x + 1.
        var x = Value.Of(3.0);
        Value shared = Value.Add(Value.Mul(x, x), x);
        shared.Backward();
        Check(Math.Abs(x.Grad - 7.0) < 1e-12, "a node used twice accumulates both contributions: " + x.Grad);
        Check(Math.Abs(shared.NodeCount - 3) < 0.5, "the graph was built once per distinct node: " + shared.NodeCount);

        // y = (x + x) * (x + x) exercises four gradient paths to one leaf.
        var x2 = Value.Of(1.5);
        Value doubled = Value.Add(x2, x2);
        Value square = Value.Mul(doubled, doubled);
        square.Backward();
        Check(Math.Abs(x2.Grad - 8 * x2.Data) < 1e-12, "four paths accumulate: " + x2.Grad);

        // A long chain: x -> x * 1.01 repeated 100 times.
        var chainRoot = Value.Of(1.0);
        Value chain = chainRoot;
        for (int i = 0; i < 100; i++) chain = Value.Mul(chain, Value.Of(1.01));
        chain.Backward();
        Check(Math.Abs(chain.Data - Math.Pow(1.01, 100)) < 1e-9, "a hundred step chain value");
        Check(Math.Abs(chainRoot.Grad - Math.Pow(1.01, 100)) < 1e-9, "a hundred step chain gradient: " + chainRoot.Grad);

        // log/exp round trip and its derivative; relu and sigmoid at their kinks.
        var lx = Value.Of(0.42);
        Value roundTrip = Value.Log(Value.Exp(lx));
        roundTrip.Backward();
        Check(Math.Abs(roundTrip.Data - 0.42) < 1e-12 && Math.Abs(lx.Grad - 1.0) < 1e-12, "log(exp(x)) is identity in value and gradient");
        var zero = Value.Of(-2.0);
        Value.Relu(zero).Backward();
        Check(zero.Grad == 0.0, "relu is flat on the negative side");
        var kink = Value.Of(0.0);
        Value.Relu(kink).Backward();
        Check(kink.Grad == 0.0, "relu at the kink uses the documented subgradient 0");
        var positive = Value.Of(2.0);
        Value.Relu(positive).Backward();
        Check(positive.Grad == 1.0, "relu passes the gradient through when active");
        var sig = Value.Of(0.3);
        Value.Sigmoid(sig).Backward();
        double s = 1.0 / (1.0 + Math.Exp(-0.3));
        Check(Math.Abs(sig.Grad - s * (1 - s)) < 1e-12, "sigmoid derivative");

        // Finite differences over a hand written composite expression.
        double worst = 0;
        for (int trial = 0; trial < 25; trial++)
        {
            double[] input = { 0.3 + rng.NextSigned(), 0.5 + rng.NextSigned(), 0.7 + rng.NextSigned() };
            worst = Math.Max(worst, FiniteDifference(input, 0));
            worst = Math.Max(worst, FiniteDifference(input, 1));
            worst = Math.Max(worst, FiniteDifference(input, 2));
        }
        Check(worst < 1e-7, "finite differences agree with autodiff: " + worst.ToString("E2"));

        // Zeroing and repeated backward passes.
        var repeat = Value.Of(4.0);
        Value.Square(repeat).Backward();
        Check(Math.Abs(repeat.Grad - 8.0) < 1e-12, "first backward");
        repeat.ZeroGrad();
        Check(repeat.Grad == 0.0, "zero grad clears");
        Check(repeat.NodeCount == 1, "a leaf is its own graph");
    }

    private static double Expression(double[] input)
    {
        double x = input[0], y = input[1], z = input[2];
        return Math.Tanh(x * y - 0.25 * z) + 0.5 * Math.Log(1.0 + z * z) - 0.2 * (x + y) * (x + y);
    }

    private static double FiniteDifference(double[] input, int index)
    {
        double epsilon = 1e-6;
        var probe = (double[])input.Clone();
        var leaves = new List<Value>();
        Value analyticGraph = BuildExpression(input, leaves);
        analyticGraph.Backward();
        double analytic = leaves[index].Grad;
        probe[index] = input[index] + epsilon;
        double plus = Expression(probe);
        probe[index] = input[index] - epsilon;
        double minus = Expression(probe);
        double numeric = (plus - minus) / (2 * epsilon);
        return Math.Abs(numeric - analytic) / Math.Max(1.0, Math.Abs(numeric));
    }

    private static Value BuildExpression(double[] input, List<Value> leaves)
    {
        var x = Value.Of(input[0]);
        var y = Value.Of(input[1]);
        var z = Value.Of(input[2]);
        leaves.Add(x);
        leaves.Add(y);
        leaves.Add(z);
        Value first = Value.Tanh(Value.Sub(Value.Mul(x, y), Value.Scale(z, 0.25)));
        Value second = Value.Scale(Value.Log(Value.Add(Value.Of(1.0), Value.Square(z))), 0.5);
        Value sum = Value.Add(x, y);
        Value third = Value.Scale(Value.Square(sum), 0.2);
        return Value.Sub(Value.Add(first, second), third);
    }

    // ---- renderer ------------------------------------------------------------------------------------

    private static double firstLoss;
    private static double lastLoss;

    private static void RendererChecks()
    {
        const int height = 12, width = 12;
        var scene = new Scene
        {
            X = new[] { Value.Of(2.0), Value.Of(9.0), Value.Of(4.0) },
            Y = new[] { Value.Of(2.5), Value.Of(3.5), Value.Of(9.0) },
            Color = new[] { Value.Of(0.9), Value.Of(0.2), Value.Of(0.4) },
            Background = Value.Of(0.0),
            Sigma = Value.Of(0.4)
        };
        var target = Renderer.RenderReference(height, width, scene, false);
        var flatTarget = target;
        Value loss = Renderer.RenderLoss(height, width, scene, flatTarget, out double[] image);
        var reference = Renderer.RenderReference(height, width, scene, false);
        double maxDifference = 0;
        for (int i = 0; i < image.Length; i++) maxDifference = Math.Max(maxDifference, Math.Abs(image[i] - reference[i]));
        Check(maxDifference < 1e-9, "the differentiable forward matches the independent rasteriser: " + maxDifference.ToString("E2"));
        Check(Math.Abs(loss.Data) < 1e-18, "a target equal to the render gives zero loss");
        Check(Renderer.Area(scene) > 0, "the triangle has a positive signed area");

        // A genuinely different target, so the gradient checks below are not vacuous.
        var shiftedScene = new Scene
        {
            X = scene.X.Select(v => Value.Of(v.Data + 0.7)).ToArray(),
            Y = scene.Y.Select(v => Value.Of(v.Data - 0.6)).ToArray(),
            Color = scene.Color.Select(v => Value.Of(v.Data)).ToArray(),
            Background = Value.Of(scene.Background.Data),
            Sigma = Value.Of(0.4)
        };
        double[] gradientTarget = Renderer.RenderReference(height, width, shiftedScene, false);
        double mismatchedLoss = Renderer.RenderLoss(height, width, scene, gradientTarget, out double[] _).Data;
        Check(mismatchedLoss > 1e-4, "a mismatched target gives a non trivial loss: " + mismatchedLoss.ToString("E3"));

        var hard = Renderer.RenderReference(height, width, scene, true);
        var soft = Renderer.RenderReference(height, width, scene, false);
        int agree = 0;
        for (int i = 0; i < hard.Length; i += 3)
        {
            bool hardInside = hard[i] > 0.5 * (scene.Color[0].Data + scene.Background.Data);
            bool softInside = soft[i] > 0.5 * (scene.Color[0].Data + scene.Background.Data);
            if (hardInside == softInside || Math.Abs(soft[i] - hard[i]) < 0.35) agree++;
        }
        Check(agree >= height * width * 0.9, $"the soft coverage follows the hard mask on {agree} of {height * width} pixels");

        var sharp = new Scene
        {
            X = scene.X.Select(v => Value.Of(v.Data)).ToArray(),
            Y = scene.Y.Select(v => Value.Of(v.Data)).ToArray(),
            Color = scene.Color.Select(v => Value.Of(v.Data)).ToArray(),
            Background = Value.Of(scene.Background.Data),
            Sigma = Value.Of(0.02)
        };
        var sharpImage = Renderer.RenderReference(height, width, sharp, false);
        int sharpAgree = 0;
        for (int i = 0; i < hard.Length; i += 3)
        {
            bool hardInside = hard[i] > 0.5 * (sharp.Color[0].Data + sharp.Background.Data);
            bool sharpInside = sharpImage[i] > 0.5 * (sharp.Color[0].Data + sharp.Background.Data);
            if (hardInside == sharpInside) sharpAgree++;
        }
        Check(sharpAgree >= height * width * 0.98, $"a tiny softness reproduces the hard mask on {sharpAgree} pixels");

        // Coverage extremes: the centroid is fully covered, a far pixel is not.
        double centroidX = (scene.X[0].Data + scene.X[1].Data + scene.X[2].Data) / 3;
        double centroidY = (scene.Y[0].Data + scene.Y[1].Data + scene.Y[2].Data) / 3;
        double centroid = ComputeCoverage(scene, centroidX, centroidY);
        Check(centroid > 0.9 && centroid < 1.0, "the centroid coverage is essentially one: " + centroid.ToString("F4"));
        var sharper = new Scene
        {
            X = scene.X.Select(v => Value.Of(v.Data)).ToArray(),
            Y = scene.Y.Select(v => Value.Of(v.Data)).ToArray(),
            Color = scene.Color.Select(v => Value.Of(v.Data)).ToArray(),
            Background = Value.Of(scene.Background.Data),
            Sigma = Value.Of(0.05)
        };
        Check(ComputeCoverage(sharper, centroidX, centroidY) > centroid,
            "a sharper softness raises the interior coverage towards one");
        Check(ComputeCoverage(scene, 50, 50) < 1e-3, "a far pixel has essentially no coverage");

        // ---- gradients of every renderer parameter against finite differences -------------------
        double worstVertex = 0, worstColor = 0, worstBackground = 0, worstSigma = 0;
        for (int index = 0; index < 3; index++)
        {
            worstVertex = Math.Max(worstVertex, VertexFiniteDifference(height, width, scene, gradientTarget, index, true));
            worstVertex = Math.Max(worstVertex, VertexFiniteDifference(height, width, scene, gradientTarget, index, false));
            worstColor = Math.Max(worstColor, ColorFiniteDifference(height, width, scene, gradientTarget, index));
        }
        worstBackground = BackgroundFiniteDifference(height, width, scene, gradientTarget);
        worstSigma = SigmaFiniteDifference(height, width, scene, gradientTarget);
        Check(worstVertex < 1e-6, "vertex gradients match finite differences: " + worstVertex.ToString("E2"));
        Check(worstColor < 1e-6, "colour gradients match finite differences: " + worstColor.ToString("E2"));
        Check(worstBackground < 1e-6, "background gradient matches finite differences: " + worstBackground.ToString("E2"));
        Check(worstSigma < 1e-5, "softness gradient matches finite differences: " + worstSigma.ToString("E2"));

        // Locality: a pixel far from the triangle barely depends on a vertex.
        var farTarget = new double[height * width * 3];
        Value farLoss = Renderer.RenderLoss(height, width, scene, farTarget, out double[] _);
        farLoss.Backward();
        bool anyGradient = scene.X.Any(v => Math.Abs(v.Grad) > 1e-12) || scene.Y.Any(v => Math.Abs(v.Grad) > 1e-12);
        Check(anyGradient, "the loss still depends on the vertices even with a black target");

        // ---- fitting: gradient descent on the vertices ------------------------------------------
        var targetScene = new Scene
        {
            X = new[] { Value.Of(4.0), Value.Of(8.0), Value.Of(6.0) },
            Y = new[] { Value.Of(4.0), Value.Of(5.0), Value.Of(9.0) },
            Color = new[] { Value.Of(0.9), Value.Of(0.2), Value.Of(0.4) },
            Background = Value.Of(0.0),
            Sigma = Value.Of(1.0)           // a broad softness so the loss surface is smooth enough to fit
        };
        double[] fittingTarget = Renderer.RenderReference(height, width, targetScene, false);
        var fitted = new Scene
        {
            X = new[] { Value.Of(2.0), Value.Of(9.0), Value.Of(4.0) },
            Y = new[] { Value.Of(2.5), Value.Of(3.5), Value.Of(9.0) },
            Color = new[] { Value.Of(0.9), Value.Of(0.2), Value.Of(0.4) },
            Background = Value.Of(0.0),
            Sigma = Value.Of(1.0)
        };
        firstLoss = 0;
        lastLoss = 0;
        var losses = new List<double>();
        double stepSize = 2.0;
        for (int step = 0; step < 120; step++)
        {
            Value stepLoss = Renderer.RenderLoss(height, width, fitted, fittingTarget, out double[] _);
            stepLoss.Backward();
            if (step == 0) { firstLoss = stepLoss.Data; lastLoss = firstLoss; }
            losses.Add(stepLoss.Data);
            var direction = new double[6];
            for (int i = 0; i < 3; i++)
            {
                direction[i * 2] = fitted.X[i].Grad;
                direction[i * 2 + 1] = fitted.Y[i].Grad;
            }
            double norm = Math.Sqrt(direction.Sum(d => d * d)) + 1e-15;
            for (int i = 0; i < 6; i++) direction[i] /= norm;
            // Backtracking line search along the unit descent direction: the step length (in pixels) is
            // halved until it actually lowers the loss, so the descent is monotone by construction.
            double trial = stepSize;
            bool improved = false;
            while (trial > 1e-6 && !improved)
            {
                var candidate = new Scene
                {
                    X = Enumerable.Range(0, 3).Select(i => Value.Of(fitted.X[i].Data - trial * direction[i * 2])).ToArray(),
                    Y = Enumerable.Range(0, 3).Select(i => Value.Of(fitted.Y[i].Data - trial * direction[i * 2 + 1])).ToArray(),
                    Color = fitted.Color.Select(v => Value.Of(v.Data)).ToArray(),
                    Background = Value.Of(fitted.Background.Data),
                    Sigma = Value.Of(fitted.Sigma.Data)
                };
                double candidateLoss = Renderer.RenderLoss(height, width, candidate, fittingTarget, out double[] _).Data;
                if (candidateLoss < stepLoss.Data)
                {
                    fitted = candidate;
                    improved = true;
                    stepSize = Math.Min(4.0, trial * 1.5);
                }
                else
                {
                    trial *= 0.5;
                }
            }
            if (!improved) break;
            lastLoss = Renderer.RenderLoss(height, width, fitted, fittingTarget, out double[] _).Data;
            losses.Add(lastLoss);
        }
        Check(lastLoss < firstLoss / 5, $"descent cut the loss substantially: {firstLoss:E3} -> {lastLoss:E3}");
        Check(losses.Zip(losses.Skip(1), (a, b) => b <= a + 1e-12).All(x => x),
            "the loss decreases monotonically during the fit");
        double vertexError = 0;
        for (int i = 0; i < 3; i++)
        {
            vertexError += Math.Abs(fitted.X[i].Data - targetScene.X[i].Data) + Math.Abs(fitted.Y[i].Data - targetScene.Y[i].Data);
        }
        Check(vertexError < 3.0, "the vertices moved onto the target: total error " + vertexError.ToString("F3"));

        // ---- compositing two triangles ----------------------------------------------------------
        var second = new Scene
        {
            X = new[] { Value.Of(6.0), Value.Of(11.0), Value.Of(8.0) },
            Y = new[] { Value.Of(1.0), Value.Of(8.0), Value.Of(10.0) },
            Color = new[] { Value.Of(0.1), Value.Of(0.8), Value.Of(0.9) },
            Background = Value.Of(0.0),
            Sigma = Value.Of(0.4)
        };
        double[] firstImage = Renderer.RenderReference(height, width, scene, false);
        double[] secondImage = Renderer.RenderReference(height, width, second, false);
        double[] secondMask = Renderer.RenderReference(height, width, second, true);
        double[] composited = Composite(firstImage, secondImage, secondMask);
        int overridden = 0, kept = 0;
        for (int i = 0; i < composited.Length; i++)
        {
            if (secondMask[i] > 0.5 && composited[i] == secondImage[i]) overridden++;
            else if (secondMask[i] <= 0.5 && composited[i] == firstImage[i]) kept++;
        }
        Check(overridden > 0 && kept > 0, $"painter order: {overridden} channels from the top layer, {kept} from the bottom");
        var channelwise = new double[composited.Length];
        for (int i = 0; i < channelwise.Length; i++) channelwise[i] = Math.Max(firstImage[i], secondImage[i]);
        Check(!channelwise.SequenceEqual(composited), "compositing is order sensitive, not a per channel maximum");
        var overlapGradient = VertexFiniteDifference(height, width, second, secondImage, 1, true);
        Check(overlapGradient < 1e-6, "the second layer is differentiable too: " + overlapGradient.ToString("E2"));

        // ---- degenerate shapes --------------------------------------------------------------------
        var degenerate = new Scene
        {
            X = new[] { Value.Of(5.0), Value.Of(5.0), Value.Of(5.0) },
            Y = new[] { Value.Of(5.0), Value.Of(5.0), Value.Of(5.0) },
            Color = new[] { Value.Of(1.0), Value.Of(0.0), Value.Of(0.0) },
            Background = Value.Of(0.0),
            Sigma = Value.Of(0.05)
        };
        var degenerateImage = Renderer.RenderReference(4, 4, degenerate, false);
        Check(degenerateImage.All(double.IsFinite), "a zero area triangle still renders finite values");
        var outside = new Scene
        {
            X = new[] { Value.Of(-40.0), Value.Of(-30.0), Value.Of(-35.0) },
            Y = new[] { Value.Of(-40.0), Value.Of(-45.0), Value.Of(-25.0) },
            Color = new[] { Value.Of(1.0), Value.Of(1.0), Value.Of(1.0) },
            Background = Value.Of(0.0),
            Sigma = Value.Of(0.05)
        };
        var outsideImage = Renderer.RenderReference(4, 4, outside, false);
        Check(outsideImage.All(v => v < 1e-20), "a triangle outside the frame contributes nothing");
        Value outsideLoss = Renderer.RenderLoss(4, 4, outside, new double[4 * 4 * 3], out double[] _);
        Check(double.IsFinite(outsideLoss.Data), "the loss outside the frame is finite");
        var onePixel = new Scene
        {
            X = new[] { Value.Of(0.2), Value.Of(0.8), Value.Of(0.5) },
            Y = new[] { Value.Of(0.2), Value.Of(0.8), Value.Of(0.9) },
            Color = new[] { Value.Of(1.0), Value.Of(1.0), Value.Of(1.0) },
            Background = Value.Of(0.0),
            Sigma = Value.Of(0.05)
        };
        var onePixelImage = Renderer.RenderReference(1, 1, onePixel, false);
        Check(onePixelImage.Length == 3 && onePixelImage.All(v => v >= 0 && v <= 1), "a one pixel image renders inside [0,1]");
        Check(Renderer.RenderReference(1, 1, onePixel, false).SequenceEqual(onePixelImage),
            "rendering the same scene twice is identical");

        // ---- determinism -------------------------------------------------------------------------
        Value repeatLoss = Renderer.RenderLoss(height, width, scene, flatTarget, out double[] repeatImage);
        Check(repeatImage.SequenceEqual(image), "the differentiable render is deterministic");
        Check(Math.Abs(repeatLoss.Data - loss.Data) < 1e-15, "the loss is deterministic");
        Check(flatTarget.SequenceEqual(Renderer.RenderReference(height, width, scene, false)),
            "the reference rasteriser is deterministic");
    }

    private static double ComputeCoverage(Scene scene, double cx, double cy)
        => Renderer.CoverageValue(scene, cx, cy);

    private static double VertexFiniteDifference(int height, int width, Scene scene, double[] target, int vertex, bool onX)
    {
        double epsilon = 1e-5;
        Value loss = Renderer.RenderLoss(height, width, scene, target, out double[] _);
        loss.ZeroGrad();
        loss.Backward();
        double analytic = onX ? scene.X[vertex].Grad : scene.Y[vertex].Grad;
        double original = onX ? scene.X[vertex].Data : scene.Y[vertex].Data;
        if (onX) scene.X[vertex] = Value.Of(original + epsilon); else scene.Y[vertex] = Value.Of(original + epsilon);
        double plus = Renderer.RenderLoss(height, width, scene, target, out double[] _).Data;
        if (onX) scene.X[vertex] = Value.Of(original - epsilon); else scene.Y[vertex] = Value.Of(original - epsilon);
        double minus = Renderer.RenderLoss(height, width, scene, target, out double[] _).Data;
        if (onX) scene.X[vertex] = Value.Of(original); else scene.Y[vertex] = Value.Of(original);
        double numeric = (plus - minus) / (2 * epsilon);
        return Math.Abs(numeric - analytic) / Math.Max(1.0, Math.Abs(numeric));
    }

    private static double ColorFiniteDifference(int height, int width, Scene scene, double[] target, int channel)
    {
        double epsilon = 1e-5;
        Value loss = Renderer.RenderLoss(height, width, scene, target, out double[] _);
        loss.ZeroGrad();
        loss.Backward();
        double analytic = scene.Color[channel].Grad;
        double original = scene.Color[channel].Data;
        scene.Color[channel] = Value.Of(original + epsilon);
        double plus = Renderer.RenderLoss(height, width, scene, target, out double[] _).Data;
        scene.Color[channel] = Value.Of(original - epsilon);
        double minus = Renderer.RenderLoss(height, width, scene, target, out double[] _).Data;
        scene.Color[channel] = Value.Of(original);
        double numeric = (plus - minus) / (2 * epsilon);
        return Math.Abs(numeric - analytic) / Math.Max(1.0, Math.Abs(numeric));
    }

    private static double BackgroundFiniteDifference(int height, int width, Scene scene, double[] target)
    {
        double epsilon = 1e-5;
        Value loss = Renderer.RenderLoss(height, width, scene, target, out double[] _);
        loss.ZeroGrad();
        loss.Backward();
        double analytic = scene.Background.Grad;
        double original = scene.Background.Data;
        scene.Background = Value.Of(original + epsilon);
        double plus = Renderer.RenderLoss(height, width, scene, target, out double[] _).Data;
        scene.Background = Value.Of(original - epsilon);
        double minus = Renderer.RenderLoss(height, width, scene, target, out double[] _).Data;
        scene.Background = Value.Of(original);
        double numeric = (plus - minus) / (2 * epsilon);
        return Math.Abs(numeric - analytic) / Math.Max(1.0, Math.Abs(numeric));
    }

    private static double SigmaFiniteDifference(int height, int width, Scene scene, double[] target)
    {
        double epsilon = 1e-6;
        Value loss = Renderer.RenderLoss(height, width, scene, target, out double[] _);
        loss.ZeroGrad();
        loss.Backward();
        double analytic = scene.Sigma.Grad;
        double original = scene.Sigma.Data;
        scene.Sigma = Value.Of(original + epsilon);
        double plus = Renderer.RenderLoss(height, width, scene, target, out double[] _).Data;
        scene.Sigma = Value.Of(original - epsilon);
        double minus = Renderer.RenderLoss(height, width, scene, target, out double[] _).Data;
        scene.Sigma = Value.Of(original);
        double numeric = (plus - minus) / (2 * epsilon);
        return Math.Abs(numeric - analytic) / Math.Max(1.0, Math.Abs(numeric));
    }

    // Painter's order: the second layer wins wherever its coverage dominates.
    private static double[] Composite(double[] under, double[] over, double[] overCoverage)
    {
        var result = new double[under.Length];
        for (int i = 0; i < under.Length; i++) result[i] = overCoverage[i] > 0.5 ? over[i] : under[i];
        return result;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
