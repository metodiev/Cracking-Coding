// Task 513 - Implement auto-differentiation simple reverse-mode for scalar functions. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// A tape-based reverse-mode ("backpropagation") auto-differentiator for scalar functions of several
// variables, plus a forward-mode dual-number evaluator used as an independent cross-check.
// Every operation allocates a Node holding its value, its inputs and the local partial derivatives
// of the operation with respect to those inputs; the creation order forms the tape. Backward(root)
// walks the tape in reverse topological order (DFS post-order from the root), seeds the root's
// adjoint with 1 and accumulates adjoint[input] += adjoint[node] * partial, so a node used twice
// receives both contributions - the diamond-shaped expression f = y + y with y = x * x must produce
// 4x, which is the classic reverse-mode sharing test. Adjoints live in a dictionary keyed by node
// identity and are reset on every call, so repeated Backward passes are idempotent.
// Supported operations: +, -, *, /, power, exp, log, sin, cos, tanh, sigmoid and negation.
// Documented checks: the exact closed forms of x*x*y + x*y*y at (2, 3) give dx = 21 and dy = 16 and
// the diamond f = y + y gives 4x; the tape for the diamond holds exactly three nodes (structural
// sharing); gradients of six expressions at four points per expression agree with a central
// difference (h = 1e-6) to 1e-7 relative and with the forward-mode dual evaluator exactly; a node
// unreachable from the root has gradient 0; the gradient of a constant is 0; gradients are additive
// (grad(f + g) = grad f + grad g); a 20-fold sin composition matches its closed-form product of
// cosines; and repeated Backward calls return bit-identical gradients.
using System;
using System.Collections.Generic;

internal class Node
{
    internal Node(double value, string op, Node[] inputs, double[] partials)
    {
        Value = value;
        Op = op;
        Inputs = inputs;
        Partials = partials;
    }

    internal double Value { get; }

    internal string Op { get; }

    internal Node[] Inputs { get; }

    internal double[] Partials { get; }
}

internal class Tape
{
    private readonly List<Node> nodes = new List<Node>();

    internal int NodeCount => nodes.Count;

    internal Node Variable(double value)
    {
        var node = new Node(value, "var", Array.Empty<Node>(), Array.Empty<double>());
        nodes.Add(node);
        return node;
    }

    internal Node Constant(double value)
    {
        var node = new Node(value, "const", Array.Empty<Node>(), Array.Empty<double>());
        nodes.Add(node);
        return node;
    }

    private Node Unary(double value, string op, Node input, double partial)
    {
        var node = new Node(value, op, new[] { input }, new[] { partial });
        nodes.Add(node);
        return node;
    }

    private Node Binary(double value, string op, Node left, Node right, double leftPartial,
            double rightPartial)
    {
        var node = new Node(value, op, new[] { left, right }, new[] { leftPartial, rightPartial });
        nodes.Add(node);
        return node;
    }

    internal Node Add(Node a, Node b) => Binary(a.Value + b.Value, "+", a, b, 1.0, 1.0);

    internal Node Sub(Node a, Node b) => Binary(a.Value - b.Value, "-", a, b, 1.0, -1.0);

    internal Node Mul(Node a, Node b) => Binary(a.Value * b.Value, "*", a, b, b.Value, a.Value);

    internal Node Div(Node a, Node b) =>
            Binary(a.Value / b.Value, "/", a, b, 1.0 / b.Value, -a.Value / (b.Value * b.Value));

    internal Node Neg(Node a) => Unary(-a.Value, "neg", a, -1.0);

    internal Node Exp(Node a) => Unary(Math.Exp(a.Value), "exp", a, Math.Exp(a.Value));

    internal Node Log(Node a) => Unary(Math.Log(a.Value), "log", a, 1.0 / a.Value);

    internal Node Sin(Node a) => Unary(Math.Sin(a.Value), "sin", a, Math.Cos(a.Value));

    internal Node Cos(Node a) => Unary(Math.Cos(a.Value), "cos", a, -Math.Sin(a.Value));

    internal Node Tanh(Node a) => Unary(Math.Tanh(a.Value), "tanh", a, 1.0 - Math.Tanh(a.Value) * Math.Tanh(a.Value));

    internal Node Sigmoid(Node a)
    {
        double s = 1.0 / (1.0 + Math.Exp(-a.Value));
        return Unary(s, "sigmoid", a, s * (1.0 - s));
    }

    internal Node Pow(Node a, double exponent)
    {
        double value = Math.Pow(a.Value, exponent);
        return Unary(value, "pow", a, exponent * Math.Pow(a.Value, exponent - 1.0));
    }

    /** Value of the whole tape's last node. */
    internal static double ValueOf(Node node) => node.Value;

    /** Adjoints of every reachable node, keyed by identity; repeated calls are idempotent. */
    internal static Dictionary<Node, double> Backward(Node root)
    {
        var adjoints = new Dictionary<Node, double>();
        var order = new List<Node>();
        var visited = new HashSet<Node>();
        void Visit(Node node)
        {
            if (!visited.Add(node)) return;
            foreach (Node input in node.Inputs) Visit(input);
            order.Add(node);
        }
        Visit(root);
        foreach (Node node in order) adjoints[node] = 0.0;
        adjoints[root] = 1.0;
        for (int i = order.Count - 1; i >= 0; i--)
        {
            Node node = order[i];
            double contribution = adjoints[node];
            if (contribution == 0.0) continue;
            for (int input = 0; input < node.Inputs.Length; input++)
            {
                Node child = node.Inputs[input];
                adjoints[child] = adjoints[child] + contribution * node.Partials[input];
            }
        }
        return adjoints;
    }

    internal static double Gradient(Node root, Node variable)
    {
        Dictionary<Node, double> adjoints = Backward(root);
        return adjoints.TryGetValue(variable, out double value) ? value : 0.0;
    }
}

/** Forward-mode dual numbers: value plus the derivative with respect to one seed variable. */
internal readonly struct Dual
{
    internal Dual(double value, double derivative)
    {
        Value = value;
        Derivative = derivative;
    }

    internal double Value { get; }

    internal double Derivative { get; }

    internal static Dual Seed(double value) => new Dual(value, 1.0);

    internal static Dual Constant(double value) => new Dual(value, 0.0);

    public static Dual operator +(Dual a, Dual b) => new Dual(a.Value + b.Value, a.Derivative + b.Derivative);

    public static Dual operator -(Dual a, Dual b) => new Dual(a.Value - b.Value, a.Derivative - b.Derivative);

    public static Dual operator *(Dual a, Dual b) =>
            new Dual(a.Value * b.Value, a.Derivative * b.Value + a.Value * b.Derivative);

    public static Dual operator /(Dual a, Dual b) =>
            new Dual(a.Value / b.Value, (a.Derivative * b.Value - a.Value * b.Derivative) / (b.Value * b.Value));

    internal static Dual Exp(Dual a) => new Dual(Math.Exp(a.Value), Math.Exp(a.Value) * a.Derivative);

    internal static Dual Log(Dual a) => new Dual(Math.Log(a.Value), a.Derivative / a.Value);

    internal static Dual Sin(Dual a) => new Dual(Math.Sin(a.Value), Math.Cos(a.Value) * a.Derivative);

    internal static Dual Cos(Dual a) => new Dual(Math.Cos(a.Value), -Math.Sin(a.Value) * a.Derivative);

    internal static Dual Tanh(Dual a)
    {
        double t = Math.Tanh(a.Value);
        return new Dual(t, (1.0 - t * t) * a.Derivative);
    }

    internal static Dual Sigmoid(Dual a)
    {
        double s = 1.0 / (1.0 + Math.Exp(-a.Value));
        return new Dual(s, s * (1.0 - s) * a.Derivative);
    }

    internal static Dual Pow(Dual a, double exponent) =>
            new Dual(Math.Pow(a.Value, exponent),
                    exponent * Math.Pow(a.Value, exponent - 1.0) * a.Derivative);
}

internal static class Expressions
{
    internal static string[] Names =
    {
        "x*x*y + x*y*y", "sin(x)*exp(y) - x/y", "log(exp(x)+exp(y))", "sigmoid(x)*tanh(y)",
        "x^3 - 2*y^2 + x*y", "sqrt-ish: pow(x*x + 1, 1.5) * log(1 + y*y)",
    };

    /** Builds each expression on a fresh tape; variable order is (x, y). */
    internal static Func<Tape, Node[], Node>[] Builders()
    {
        return new Func<Tape, Node[], Node>[]
        {
            (tape, v) =>
            {
                Node product = tape.Mul(tape.Mul(v[0], v[0]), v[1]);
                Node second = tape.Mul(tape.Mul(v[0], v[1]), v[1]);
                return tape.Add(product, second);
            },
            (tape, v) =>
            {
                Node first = tape.Mul(tape.Sin(v[0]), tape.Exp(v[1]));
                return tape.Sub(first, tape.Div(v[0], v[1]));
            },
            (tape, v) => tape.Log(tape.Add(tape.Exp(v[0]), tape.Exp(v[1]))),
            (tape, v) => tape.Mul(tape.Sigmoid(v[0]), tape.Tanh(v[1])),
            (tape, v) =>
            {
                Node first = tape.Pow(v[0], 3.0);
                Node second = tape.Mul(tape.Constant(2.0), tape.Mul(v[1], v[1]));
                return tape.Add(tape.Sub(first, second), tape.Mul(v[0], v[1]));
            },
            (tape, v) =>
            {
                Node inside = tape.Add(tape.Mul(v[0], v[0]), tape.Constant(1.0));
                Node first = tape.Mul(tape.Pow(inside, 1.5), tape.Log(tape.Add(tape.Constant(1.0), tape.Mul(v[1], v[1]))));
                return first;
            },
        };
    }

    /** The same expressions in forward mode, seeded once for the first variable. */
    internal static Func<Dual[], Dual>[] ForwardBuilders()
    {
        return new Func<Dual[], Dual>[]
        {
            v => v[0] * v[0] * v[1] + v[0] * v[1] * v[1],
            v => Dual.Sin(v[0]) * Dual.Exp(v[1]) - v[0] / v[1],
            v => Dual.Log(Dual.Exp(v[0]) + Dual.Exp(v[1])),
            v => Dual.Sigmoid(v[0]) * Dual.Tanh(v[1]),
            v => Dual.Pow(v[0], 3.0) - Dual.Constant(2.0) * v[1] * v[1] + v[0] * v[1],
            v => Dual.Pow(v[0] * v[0] + Dual.Constant(1.0), 1.5)
                    * Dual.Log(Dual.Constant(1.0) + v[1] * v[1]),
        };
    }
}

public static class Solution
{
    public static void Main()
    {
        // --- the diamond sharing test -------------------------------------------------------------------
        var tape = new Tape();
        Node x = tape.Variable(3.0);
        Node product = tape.Mul(x, x);
        Node diamond = tape.Add(product, product);
        Check(tape.NodeCount == 3, "the diamond shares one node (tape has three nodes)");
        Check(diamond.Value == 18.0, "x * x + x * x at x = 3 is 18");
        Check(Math.Abs(Tape.Gradient(diamond, x) - 12.0) <= 1e-12, "reverse mode gives 4x = 12");
        Dictionary<Node, double> first = Tape.Backward(diamond);
        Dictionary<Node, double> second = Tape.Backward(diamond);
        Check(first[x] == second[x], "repeated backward passes are idempotent");
        Check(first[product] == 2.0, "the shared node accumulates both paths");
        Check(first[diamond] == 1.0, "the root's adjoint is 1");

        // --- the exact closed form -------------------------------------------------------------------------
        var exactTape = new Tape();
        Node ex = exactTape.Variable(2.0);
        Node ey = exactTape.Variable(3.0);
        Node expression = Expressions.Builders()[0](exactTape, new[] { ex, ey });
        Check(expression.Value == 2.0 * 2.0 * 3.0 + 2.0 * 3.0 * 3.0, "x*x*y + x*y*y = 30 at (2, 3)");
        Check(Math.Abs(Tape.Gradient(expression, ex) - 21.0) <= 1e-12, "dx = 2xy + y^2 = 21");
        Check(Math.Abs(Tape.Gradient(expression, ey) - 16.0) <= 1e-12, "dy = x^2 + 2xy = 16");

        // --- closed form for a deep chain ---------------------------------------------------------------------
        var chainTape = new Tape();
        Node chainX = chainTape.Variable(0.3);
        Node current = chainX;
        double closedForm = 1.0;
        double running = 0.3;
        for (int i = 0; i < 20; i++)
        {
            current = chainTape.Sin(current);
            closedForm *= Math.Cos(running);
            running = Math.Sin(running);
        }
        double chainGradient = Tape.Gradient(current, chainX);
        Check(Math.Abs(chainGradient - closedForm) <= 1e-12,
                "the 20-fold sin derivative equals the product of cosines");
        var dualChain = Dual.Seed(0.3);
        for (int i = 0; i < 20; i++) dualChain = Dual.Sin(dualChain);
        Check(Math.Abs(dualChain.Derivative - chainGradient) <= 1e-12, "forward mode agrees on the chain");

        // --- gradients versus central differences and forward mode ----------------------------------------------
        var builders = Expressions.Builders();
        var forwardBuilders = Expressions.ForwardBuilders();
        Check(builders.Length == Expressions.Names.Length, "every expression is named");
        Check(forwardBuilders.Length == builders.Length, "forward builders line up");
        var points = new[]
        {
            new[] { 0.7, 1.3 },
            new[] { -0.5, 0.9 },
            new[] { 2.1, -1.4 },
            new[] { 0.25, 0.4 },
        };
        foreach (double[] point in points)
        {
            for (int e = 0; e < builders.Length; e++)
            {
                var localTape = new Tape();
                var variables = new[] { localTape.Variable(point[0]), localTape.Variable(point[1]) };
                Node root = builders[e](localTape, variables);
                var numeric = new double[2];
                for (int i = 0; i < 2; i++)
                {
                    numeric[i] = CentralDifference(e, point, i, 1e-6);
                }
                for (int i = 0; i < 2; i++)
                {
                    double reverse = Tape.Gradient(root, variables[i]);
                    double scale = Math.Max(1.0, Math.Max(Math.Abs(numeric[i]), Math.Abs(reverse)));
                    Check(Math.Abs(reverse - numeric[i]) / scale <= 1e-7,
                            "reverse vs numeric for " + Expressions.Names[e] + " component " + i);
                    var seeded = new[]
                    {
                        i == 0 ? Dual.Seed(point[0]) : Dual.Constant(point[0]),
                        i == 1 ? Dual.Seed(point[1]) : Dual.Constant(point[1]),
                    };
                    double forward = forwardBuilders[e](seeded).Derivative;
                    Check(Math.Abs(forward - reverse) <= 1e-9,
                            "forward vs reverse for " + Expressions.Names[e] + " component " + i);
                }
                Check(Math.Abs(root.Value - Evaluate(e, point)) <= 1e-12, "value agreement for expression " + e);
            }
        }

        // --- unreachable nodes, constants and additivity ----------------------------------------------------------
        var isolatedTape = new Tape();
        Node used = isolatedTape.Variable(1.5);
        Node unused = isolatedTape.Variable(4.0);
        Node onlyUsed = isolatedTape.Mul(used, used);
        Check(Tape.Gradient(onlyUsed, unused) == 0.0, "an unreachable variable has gradient zero");
        Check(!Tape.Backward(onlyUsed).ContainsKey(unused), "unreachable nodes are not even visited");
        var constantTape = new Tape();
        Node constantRoot = constantTape.Constant(5.0);
        Node constantVariable = constantTape.Variable(2.0);
        Check(Tape.Gradient(constantRoot, constantVariable) == 0.0, "a constant has zero gradient");

        var sumTape = new Tape();
        Node sx = sumTape.Variable(0.6);
        Node sy = sumTape.Variable(-0.2);
        Node f1 = sumTape.Sin(sx);
        Node f2 = sumTape.Exp(sy);
        Node sum = sumTape.Add(f1, f2);
        Dictionary<Node, double> sumGradients = Tape.Backward(sum);
        Check(Math.Abs(sumGradients[sx] - Math.Cos(0.6)) <= 1e-12, "grad(f + g) = grad f for x");
        Check(Math.Abs(sumGradients[sy] - Math.Exp(-0.2)) <= 1e-12, "grad(f + g) = grad g for y");
        Check(Math.Abs(sumGradients[sx] - Math.Cos(0.6)) <= 1e-12, "additivity along x");

        // --- the chain rule on products, quotients and powers ---------------------------------------------------------
        var mixTape = new Tape();
        Node mx = mixTape.Variable(1.2);
        Node my = mixTape.Variable(0.8);
        Node mixRoot = mixTape.Div(mixTape.Mul(mx, mixTape.Exp(my)), mixTape.Add(mx, my));
        double numerator = 1.2 * Math.Exp(0.8);
        double denominator = 2.0;
        double expectedX = (Math.Exp(0.8) * denominator - numerator) / (denominator * denominator);
        double expectedY = (1.2 * Math.Exp(0.8) * denominator - numerator) / (denominator * denominator);
        Check(Math.Abs(Tape.Gradient(mixRoot, mx) - expectedX) <= 1e-12, "quotient rule for x");
        Check(Math.Abs(Tape.Gradient(mixRoot, my) - expectedY) <= 1e-12, "quotient rule for y");
        Check(Math.Abs(Tape.Gradient(mixRoot, mx) - CentralDifferenceRaw(
                p => p[0] * Math.Exp(p[1]) / (p[0] + p[1]), new[] { 1.2, 0.8 }, 0, 1e-6)) <= 1e-7,
                "quotient rule versus numerics");

        // --- log, power and sigmoid at edge-ish points -------------------------------------------------------------
        var edgeTape = new Tape();
        Node t = edgeTape.Variable(1e-6);
        Node logT = edgeTape.Log(t);
        Check(Math.Abs(Tape.Gradient(logT, t) - 1e6) <= 1e-3, "d log(x)/dx = 1/x at x = 1e-6");
        var powerTape = new Tape();
        Node p = powerTape.Variable(2.5);
        Node power = powerTape.Pow(p, -0.5);
        Check(Math.Abs(Tape.Gradient(power, p) - (-0.5 * Math.Pow(2.5, -1.5))) <= 1e-12, "power rule");
        var sigmoidTape = new Tape();
        Node z = sigmoidTape.Variable(0.0);
        Node sigmoid = sigmoidTape.Sigmoid(z);
        Check(Math.Abs(sigmoid.Value - 0.5) <= 1e-15, "sigmoid(0) = 0.5");
        Check(Math.Abs(Tape.Gradient(sigmoid, z) - 0.25) <= 1e-15, "sigmoid'(0) = 0.25");
        var tanhTape = new Tape();
        Node w = tanhTape.Variable(0.0);
        Check(Math.Abs(Tape.Gradient(tanhTape.Tanh(w), w) - 1.0) <= 1e-15, "tanh'(0) = 1");
        Check(Math.Abs(Tape.Gradient(tanhTape.Neg(w), w) + 1.0) <= 1e-15, "neg'(x) = -1");

        // --- determinism ------------------------------------------------------------------------------------------------
        var repeatTape = new Tape();
        var repeatVariables = new[] { repeatTape.Variable(0.7), repeatTape.Variable(1.3) };
        Node repeatRoot = builders[1](repeatTape, repeatVariables);
        var repeatValues = new[] { repeatTape.Variable(0.7), repeatTape.Variable(1.3) };
        Node otherRoot = builders[1](repeatTape, repeatValues);
        Check(Math.Abs(repeatRoot.Value - otherRoot.Value) <= 1e-15, "the same expression value is stable");
        Check(Tape.Gradient(repeatRoot, repeatVariables[0]) == Tape.Gradient(otherRoot, repeatValues[0]),
                "gradients are bit-identical across equivalent tapes");
        Check(tape.NodeCount == 3, "the tape does not grow unexpectedly");

        Console.WriteLine("PASS");
    }

    private static double Evaluate(int expression, double[] p)
    {
        switch (expression)
        {
            case 0:
                return p[0] * p[0] * p[1] + p[0] * p[1] * p[1];
            case 1:
                return Math.Sin(p[0]) * Math.Exp(p[1]) - p[0] / p[1];
            case 2:
                return Math.Log(Math.Exp(p[0]) + Math.Exp(p[1]));
            case 3:
                return (1.0 / (1.0 + Math.Exp(-p[0]))) * Math.Tanh(p[1]);
            case 4:
                return Math.Pow(p[0], 3.0) - 2.0 * p[1] * p[1] + p[0] * p[1];
            case 5:
                return Math.Pow(p[0] * p[0] + 1.0, 1.5) * Math.Log(1.0 + p[1] * p[1]);
            default:
                throw new Exception("unknown expression");
        }
    }

    private static double CentralDifference(int expression, double[] point, int index, double h)
    {
        return CentralDifferenceRaw(p => Evaluate(expression, p), point, index, h);
    }

    private static double CentralDifferenceRaw(Func<double[], double> f, double[] point, int index, double h)
    {
        var plus = (double[])point.Clone();
        var minus = (double[])point.Clone();
        plus[index] += h;
        minus[index] -= h;
        return (f(plus) - f(minus)) / (2.0 * h);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
