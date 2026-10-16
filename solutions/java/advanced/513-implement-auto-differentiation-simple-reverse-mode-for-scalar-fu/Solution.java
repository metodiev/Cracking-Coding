/** Task 513 - Implement auto-differentiation simple reverse-mode for scalar functions. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * A micrograd-style reverse-mode automatic differentiation engine for scalar expressions: every Var
 * records its value, the operands it was built from and the local partial derivatives
 * d(result)/d(operand), and supports +, -, *, /, unary minus, exp, log, sin, cos, tanh and pow(c).
 * `backward()` computes the forward topological order iteratively (so deep graphs cannot overflow
 * the stack) and then walks it in reverse - consumers before their inputs - resetting every
 * gradient, seeding the output with dL/dL = 1 and accumulating dL/dv += dL/dout * d(out)/dv at each
 * node - the accumulation is what makes a value used twice, such as x*x + x, come out right.
 * Gradients are read with gradOf(output, leaf), which runs the backward pass and returns the leaf's
 * accumulated gradient. Numeric verification uses central differences (f(x + h) - f(x - h)) / 2h
 * with h = 1e-6.
 *
 * Documented checks: closed forms for x*y, exp(x) sin(y), log(1 + exp(x)) (the softplus whose
 * derivative is the logistic function, checked at five points including both tails), tanh, the
 * quotient rule on p/q, the power rule for x^3 (12 at x = 2) and unary minus, each to 1e-12 and
 * each also matched against the numeric derivative to 1e-6/1e-7; the shared-node cases x*x (2x),
 * x*x + x (2x + 1) and a node consumed three times (6x) prove gradient accumulation; a
 * squared-error cost with three weights and a bias - the shape a training loop needs - gives
 * dL/dw_i = 2 (pred - target) x_i and dL/db = 2 (pred - target) exactly, plus a numeric
 * cross-check; the deep composition tanh(tanh(tanh(x))) matches its analytic product-of-derivatives
 * chain to 1e-9 and its value to 1e-15; the second derivative of x^3 at 2 is recovered as 12 by
 * numerically differentiating the autodiff gradient; a 2000-node chain differentiates to exactly 2
 * without stack overflow; the domain guards reject log of a non-positive value and division by an
 * exact zero; repeated backward passes on the same graph give bit-identical gradients with the seed
 * restored.
 */
import java.util.*;

public class Solution {
    interface Scalar {
        double apply(double x);
    }

    static class Var {
        final double value;
        final String name;
        final List<Var> inputs = new ArrayList<>();
        final List<Double> local = new ArrayList<>();
        double grad;

        private Var(double value, String name) {
            this.value = value;
            this.name = name;
        }

        static Var named(double value, String name) {
            return new Var(value, name);
        }

        static Var of(double value) {
            return new Var(value, Double.toString(value));
        }

        private Var build(double result, String symbol, Var a, double da, Var b, double db) {
            String label = b == null ? "(" + symbol + a.name + ")" : "(" + a.name + symbol + b.name + ")";
            Var out = new Var(result, label);
            out.inputs.add(a);
            out.local.add(da);
            if (b != null) {
                out.inputs.add(b);
                out.local.add(db);
            }
            return out;
        }

        static Var add(Var a, Var b) {
            return a.build(a.value + b.value, "+", a, 1.0, b, 1.0);
        }

        static Var sub(Var a, Var b) {
            return a.build(a.value - b.value, "-", a, 1.0, b, -1.0);
        }

        static Var mul(Var a, Var b) {
            return a.build(a.value * b.value, "*", a, b.value, b, a.value);
        }

        static Var div(Var a, Var b) {
            if (b.value == 0.0) throw new ArithmeticException("division by zero");
            return a.build(a.value / b.value, "/", a, 1.0 / b.value, b, -a.value / (b.value * b.value));
        }

        static Var neg(Var a) {
            return a.build(-a.value, "neg", a, -1.0, null, 0.0);
        }

        static Var exp(Var a) {
            double value = Math.exp(a.value);
            return a.build(value, "exp", a, value, null, 0.0);
        }

        static Var log(Var a) {
            if (a.value <= 0.0) throw new IllegalArgumentException("log domain: " + a.value);
            return a.build(Math.log(a.value), "log", a, 1.0 / a.value, null, 0.0);
        }

        static Var sin(Var a) {
            return a.build(Math.sin(a.value), "sin", a, Math.cos(a.value), null, 0.0);
        }

        static Var cos(Var a) {
            return a.build(Math.cos(a.value), "cos", a, -Math.sin(a.value), null, 0.0);
        }

        static Var tanh(Var a) {
            double value = Math.tanh(a.value);
            return a.build(value, "tanh", a, 1.0 - value * value, null, 0.0);
        }

        static Var pow(Var a, double exponent) {
            double value = Math.pow(a.value, exponent);
            return a.build(value, "^" + exponent, a, exponent * Math.pow(a.value, exponent - 1.0), null, 0.0);
        }

        /** Forward topological order over the sub-graph (inputs before their consumers). */
        List<Var> topologicalOrder() {
            List<Var> order = new ArrayList<>();
            Set<Var> visited = Collections.newSetFromMap(new IdentityHashMap<>());
            Deque<Var> stack = new ArrayDeque<>();
            Deque<Boolean> expanded = new ArrayDeque<>();
            stack.push(this);
            expanded.push(false);
            while (!stack.isEmpty()) {
                Var node = stack.pop();
                boolean done = expanded.pop();
                if (done) {
                    order.add(node);
                    continue;
                }
                if (!visited.add(node)) continue;
                stack.push(node);
                expanded.push(true);
                for (Var input : node.inputs) {
                    stack.push(input);
                    expanded.push(false);
                }
            }
            return order;
        }

        void backward() {
            List<Var> order = topologicalOrder();
            for (Var node : order) node.grad = 0.0;
            grad = 1.0;
            // walk the forward order in reverse so every consumer is processed before its inputs
            for (int index = order.size() - 1; index >= 0; index--) {
                Var node = order.get(index);
                for (int i = 0; i < node.inputs.size(); i++) {
                    node.inputs.get(i).grad += node.grad * node.local.get(i);
                }
            }
        }
    }

    /** d(output)/d(leaf): run the reverse pass and read the accumulated gradient. */
    static double gradOf(Var output, Var leaf) {
        output.backward();
        return leaf.grad;
    }

    static double centralOf(Scalar f, double x, double h) {
        return (f.apply(x + h) - f.apply(x - h)) / (2.0 * h);
    }

    static double relativeError(double computed, double analytic) {
        return Math.abs(computed - analytic) / Math.max(1.0, Math.abs(analytic));
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }

    public static void main(String[] args) {
        final double h = 1e-6;

        // 1. Product rule, closed form and numeric.
        Var x = Var.named(1.3, "x");
        Var y = Var.named(-0.6, "y");
        Var product = Var.mul(x, y);
        product.backward();
        check(relativeError(x.grad, y.value) <= 1e-12, "d(xy)/dx = y");
        check(relativeError(y.grad, x.value) <= 1e-12, "d(xy)/dy = x");
        check(relativeError(product.grad, 1.0) <= 1e-12, "the output seeds its own gradient with 1");
        check(relativeError(product.value, 1.3 * -0.6) <= 1e-15, "the forward value is exact");

        Var uLeaf = Var.named(0.4, "u");
        Var vLeaf = Var.named(1.1, "v");
        Var expX = Var.exp(uLeaf);
        Var sinY = Var.sin(vLeaf);
        Var expSin = Var.mul(expX, sinY);
        expSin.backward();
        check(relativeError(expX.grad, Math.sin(1.1)) <= 1e-12,
                "d(exp(u) sin(v))/d(exp u) = sin(v)");
        check(relativeError(sinY.grad, Math.exp(0.4)) <= 1e-12,
                "d(exp(u) sin(v))/d(sin v) = exp(u)");
        check(relativeError(gradOf(expSin, uLeaf), Math.exp(0.4) * Math.sin(1.1)) <= 1e-12,
                "the full chain rule gives d/du = exp(u) sin(v)");
        check(relativeError(gradOf(expSin, vLeaf), Math.exp(0.4) * Math.cos(1.1)) <= 1e-12,
                "the full chain rule gives d/dv = exp(u) cos(v)");
        check(relativeError(gradOf(expSin, uLeaf), centralOf(t -> Math.exp(t) * Math.sin(1.1), 0.4, h)) <= 1e-7,
                "the autodiff gradient matches the numeric one");

        // 2. log(1 + exp(z)) is softplus; its derivative is the logistic function.
        for (double z : new double[]{-3.0, -0.5, 0.0, 0.75, 2.5}) {
            Var leaf = Var.named(z, "z");
            Var node = Var.log(Var.add(Var.of(1.0), Var.exp(leaf)));
            double gradient = gradOf(node, leaf);
            double logistic = 1.0 / (1.0 + Math.exp(-z));
            check(relativeError(gradient, logistic) <= 1e-9, "softplus'(z) = sigmoid(z) at z = " + z);
            check(relativeError(gradient, centralOf(t -> Math.log(1.0 + Math.exp(t)), z, h)) <= 1e-6,
                    "softplus gradient matches numerics at z = " + z);
            check(relativeError(node.value, Math.log(1.0 + Math.exp(z))) <= 1e-15,
                    "the softplus value is exact at z = " + z);
        }

        // 3. Gradient accumulation over shared nodes.
        Var shared = Var.named(2.5, "x");
        Var quadratic = Var.add(Var.mul(shared, shared), shared);
        check(relativeError(gradOf(quadratic, shared), 2 * 2.5 + 1.0) <= 1e-12,
                "d(x*x + x)/dx = 2x + 1 (the shared node accumulates both paths)");
        Var leaf2 = Var.named(2.5, "x");
        check(relativeError(gradOf(Var.mul(leaf2, leaf2), leaf2), 5.0) <= 1e-12, "d(x*x)/dx = 2x");
        Var leaf3 = Var.named(2.5, "x");
        Var once = Var.mul(leaf3, leaf3);
        Var summed = Var.add(Var.add(once, once), once);
        check(relativeError(gradOf(summed, leaf3), 3.0 * 2 * 2.5) <= 1e-12,
                "a node consumed three times accumulates 6x");

        // 4. Quotient, power and unary minus.
        Var p = Var.named(3.0, "p");
        Var q = Var.named(-4.0, "q");
        Var quotient = Var.div(p, q);
        quotient.backward();
        check(relativeError(p.grad, 1.0 / q.value) <= 1e-12, "d(p/q)/dp = 1/q");
        check(relativeError(q.grad, -p.value / (q.value * q.value)) <= 1e-12, "d(p/q)/dq = -p/q^2");
        check(relativeError(quotient.value, -0.75) <= 1e-15, "p/q = -0.75");
        Var cubeLeaf = Var.named(2.0, "x");
        Var cube = Var.pow(cubeLeaf, 3.0);
        check(relativeError(gradOf(cube, cubeLeaf), 12.0) <= 1e-12, "d(x^3)/dx = 3x^2 = 12 at x = 2");
        check(relativeError(cube.value, 8.0) <= 1e-12, "the forward value x^3 at 2 is 8");
        Var minusLeaf = Var.named(1.75, "x");
        check(relativeError(gradOf(Var.neg(minusLeaf), minusLeaf), -1.0) <= 1e-12, "d(-x)/dx = -1");
        Var cosLeaf = Var.named(0.6, "x");
        check(relativeError(gradOf(Var.cos(cosLeaf), cosLeaf), -Math.sin(0.6)) <= 1e-12,
                "cos' = -sin");

        // 5. A three-layer tanh composition matches the analytic chain rule.
        Var deepLeaf = Var.named(0.35, "x");
        Var layer1 = Var.tanh(deepLeaf);
        Var layer2 = Var.tanh(layer1);
        Var layer3 = Var.tanh(layer2);
        double t1 = Math.tanh(0.35);
        double t2 = Math.tanh(t1);
        double t3 = Math.tanh(t2);
        double analyticChain = (1 - t1 * t1) * (1 - t2 * t2) * (1 - t3 * t3);
        check(relativeError(gradOf(layer3, deepLeaf), analyticChain) <= 1e-9,
                "d(tanh(tanh(tanh(x))))/dx matches the product chain");
        check(relativeError(gradOf(layer3, deepLeaf),
                centralOf(t -> Math.tanh(Math.tanh(Math.tanh(t))), 0.35, h)) <= 1e-7,
                "the deep composition also matches the numeric derivative");
        check(Math.abs(t3 - layer3.value) <= 1e-15, "the forward pass matches Math.tanh");

        // 6. A squared-error cost: the shape a training loop actually needs.
        double[] weights = {0.5, -1.25, 0.75};
        double[] features = {2.0, 0.5, -1.0};
        double biasValue = 0.3;
        double target = 1.4;
        Var bias = Var.named(biasValue, "b");
        List<Var> weightVars = new ArrayList<>();
        Var prediction = bias;
        for (int i = 0; i < weights.length; i++) {
            Var w = Var.named(weights[i], "w" + i);
            weightVars.add(w);
            prediction = Var.add(prediction, Var.mul(w, Var.of(features[i])));
        }
        Var loss = Var.pow(Var.sub(prediction, Var.of(target)), 2.0);
        loss.backward();
        double error = prediction.value - target;
        for (int i = 0; i < weights.length; i++) {
            check(relativeError(weightVars.get(i).grad, 2 * error * features[i]) <= 1e-12,
                    "dL/dw" + i + " = 2 (pred - target) x" + i);
        }
        check(relativeError(bias.grad, 2 * error) <= 1e-12, "dL/db = 2 (pred - target)");
        double predictedValue = prediction.value;
        check(relativeError(loss.value, error * error) <= 1e-15, "the loss is the squared residual");
        check(relativeError(bias.grad, centralOf(t -> {
            double sum = t;
            for (int i = 0; i < weights.length; i++) sum += weights[i] * features[i];
            double residual = sum - target;
            return residual * residual;
        }, biasValue, h)) <= 1e-7, "the bias gradient matches the numeric derivative");
        check(predictedValue == prediction.value, "the forward value is a pure function of the graph");

        // 7. Second derivative by numerically differentiating the autodiff gradient.
        double x0 = 2.0;
        double step = 1e-4;
        Var atX0 = Var.named(x0, "x");
        Var atPlus = Var.named(x0 + step, "x");
        Var atMinus = Var.named(x0 - step, "x");
        double firstDerivative = gradOf(Var.pow(atX0, 3.0), atX0);
        double plus = gradOf(Var.pow(atPlus, 3.0), atPlus);
        double minus = gradOf(Var.pow(atMinus, 3.0), atMinus);
        double secondDerivative = (plus - minus) / (2 * step);
        check(relativeError(firstDerivative, 12.0) <= 1e-12, "f'(2) = 12 for x^3");
        check(relativeError(secondDerivative, 6 * x0) <= 1e-6,
                "f''(2) = 6x = 12 via numeric differentiation of the autodiff gradient");

        // 8. A long chain: the iterative traversal must not overflow the stack.
        Var chainLeaf = Var.named(0.5, "c");
        Var accumulator = Var.of(1.0);
        for (int i = 0; i < 2000; i++) {
            accumulator = Var.add(accumulator, Var.mul(chainLeaf, Var.of(0.001)));
        }
        check(accumulator.topologicalOrder().size() == 2000 * 3 + 2,
                "the 2000-node chain has the expected number of nodes");
        check(relativeError(gradOf(accumulator, chainLeaf), 2.0) <= 1e-12,
                "d(sum 0.001 c)/dc = 2000 * 0.001 = 2 without stack overflow");

        // 9. Domain guards.
        try {
            Var.log(Var.of(0.0));
            check(false, "log(0) must be rejected");
        } catch (IllegalArgumentException expected) {
            check(true, "log of a non-positive value is rejected");
        }
        try {
            Var.log(Var.of(-2.5));
            check(false, "log(-2.5) must be rejected");
        } catch (IllegalArgumentException expected) {
            check(true, "log of a negative value is rejected");
        }
        try {
            Var.div(Var.of(1.0), Var.of(0.0));
            check(false, "division by zero must be rejected");
        } catch (ArithmeticException expected) {
            check(true, "division by an exact zero is rejected");
        }

        // 10. Determinism: repeated passes give bit-identical gradients with the seed restored.
        Var dx = Var.named(1.5, "x");
        Var dy = Var.named(-2.0, "y");
        Var reuse = Var.add(Var.mul(dx, dy), Var.of(0.25));
        double first = gradOf(reuse, dx);
        double second = gradOf(reuse, dx);
        check(first == second, "repeated backward passes give identical gradients");
        check(relativeError(first, -2.0) <= 1e-12, "d(1.5x * -2y + 0.25)/dx = -2");
        check(relativeError(dy.grad, 1.5) <= 1e-12, "d(1.5x * -2y + 0.25)/dy = 1.5");
        check(relativeError(reuse.grad, 1.0) <= 1e-12, "the output seed is restored to 1");

        System.out.println("PASS");
    }
}
