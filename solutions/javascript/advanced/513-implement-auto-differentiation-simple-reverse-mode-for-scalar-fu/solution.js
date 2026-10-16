/**
 * Task 513 - Implement auto-differentiation simple reverse-mode for scalar functions. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * A tiny tape-based reverse-mode autodiff engine for scalar functions of several variables.
 * Every operation allocates a `Node` in the `Tape`'s creation-ordered list, carrying its value and
 * a closure that pushes the local partials onto its inputs; because nodes are only ever created
 * after their parents, walking the tape backwards is a valid reverse topological order, so a
 * single backward pass over the list accumulates every adjoint (branching included: a node used
 * twice simply receives two contributions). Implemented primitives: +, -, *, /, negation, pow (any
 * real exponent), sqrt, exp, log, sin, cos, tanh, sigmoid, relu, abs and max(a, b). Values are
 * ordinary doubles, so the gradient is exact up to floating point - the only place a tolerance is
 * needed is where the derivation itself is approximate (log near a pole, relu at the kink).
 * A second, independent engine (forward mode over dual numbers, `dualGrad`) is included so the two
 * modes can be cross-checked against each other and against central finite differences.
 *
 * Documented checks: (a) the multinomial f = x y + y z + z x has the exact gradient (y + z, x + z,
 * x + y), matched to 1e-12 at three different points; (b) x^5 - 3x^3 + 2x has 5x^4 - 9x^2 + 2 to
 * 1e-9; (c) branching x*x and x*x + x*x give 2x and 4x exactly (tolerance 1e-12), while a node that
 * is not on the path to the output has adjoint exactly 0; (d) sigmoid'(z) = s (1 - s) and
 * tanh' = 1 - tanh^2 match to 1e-12, and relu' is 1 for positive and 0 for negative inputs;
 * (e) for sin(x y) + x^2 log(y), a ten-fold sigmoid chain, a mixture of exp/log/sqrt and a max(x, y)
 * node, reverse mode agrees with forward mode to 1e-12 and with central differences (h = 1e-6) to
 * 1e-7; (f) the gradient of a sum equals the sum of the gradients, and constants contribute zero;
 * (g) relu at exactly 0 returns the documented subgradient 0 and log of a non-positive value
 * throws.
 */
"use strict";

class Tape {
  constructor() {
    this.nodes = [];
  }

  constant(value) {
    const node = { value, backward: () => {} };
    this.nodes.push(node);
    return node;
  }

  variable(value, name = null) {
    const node = { value, backward: () => {}, name };
    this.nodes.push(node);
    return node;
  }

  /** A node whose forward value is `value` and whose backward pushes into `parents`. */
  node(value, parents, local) {
    const node = { value };
    node.backward = () => {
      for (let i = 0; i < parents.length; i++) {
        const [parent, partial] = local[i];
        parent.adjoint += node.adjoint * partial;
      }
    };
    this.nodes.push(node);
    return node;
  }

  /** Reverse sweep from `output`; returns the adjoint map keyed by node identity. */
  gradient(output, inputs) {
    for (const node of this.nodes) node.adjoint = 0.0;
    output.adjoint = 1.0;
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const node = this.nodes[i];
      if (node.adjoint !== 0.0) node.backward();
    }
    return inputs.map((input) => input.adjoint);
  }
}

function add(self, a, b) {
  return self.node(a.value + b.value, [a, b], [[a, 1.0], [b, 1.0]]);
}

function sub(self, a, b) {
  return self.node(a.value - b.value, [a, b], [[a, 1.0], [b, -1.0]]);
}

function mul(self, a, b) {
  return self.node(a.value * b.value, [a, b], [[a, b.value], [b, a.value]]);
}

function div(self, a, b) {
  return self.node(a.value / b.value, [a, b],
    [[a, 1.0 / b.value], [b, -a.value / (b.value * b.value)]]);
}

function neg(self, a) {
  return self.node(-a.value, [a], [[a, -1.0]]);
}

function powOp(self, a, exponent) {
  const value = a.value ** exponent;
  return self.node(value, [a], [[a, exponent * a.value ** (exponent - 1.0)]]);
}

function sqrtOp(self, a) {
  const value = Math.sqrt(a.value);
  return self.node(value, [a], [[a, 0.5 / value]]);
}

function expOp(self, a) {
  const value = Math.exp(a.value);
  return self.node(value, [a], [[a, value]]);
}

function logOp(self, a) {
  if (!(a.value > 0.0)) throw new RangeError("log is undefined for a non-positive input");
  return self.node(Math.log(a.value), [a], [[a, 1.0 / a.value]]);
}

function sinOp(self, a) {
  return self.node(Math.sin(a.value), [a], [[a, Math.cos(a.value)]]);
}

function cosOp(self, a) {
  return self.node(Math.cos(a.value), [a], [[a, -Math.sin(a.value)]]);
}

function tanhOp(self, a) {
  const value = Math.tanh(a.value);
  return self.node(value, [a], [[a, 1.0 - value * value]]);
}

function sigmoidOp(self, a) {
  const value = 1.0 / (1.0 + Math.exp(-a.value));
  return self.node(value, [a], [[a, value * (1.0 - value)]]);
}

/** ReLU with the documented subgradient 0 at exactly 0. */
function reluOp(self, a) {
  const value = Math.max(a.value, 0.0);
  return self.node(value, [a], [[a, a.value > 0.0 ? 1.0 : 0.0]]);
}

function absOp(self, a) {
  return self.node(Math.abs(a.value), [a], [[a, a.value > 0.0 ? 1.0 : (a.value < 0.0 ? -1.0 : 0.0)]]);
}

/** max(a, b) with ties going to `a` (documented), so the gradient is deterministic. */
function maxOp(self, a, b) {
  return a.value >= b.value
    ? self.node(a.value, [a, b], [[a, 1.0], [b, 0.0]])
    : self.node(b.value, [a, b], [[a, 0.0], [b, 1.0]]);
}

// ---------------------------------------------------------------------------------------------
// Forward mode over dual numbers, used as an independent cross-check of the reverse tape.

class Dual {
  constructor(value, deriv) {
    this.value = value;
    this.deriv = deriv;
  }
}

function dualBinary(a, b, value, da, db) {
  return new Dual(value, da * a.deriv + db * b.deriv);
}

const dual = {
  add: (a, b) => dualBinary(a, b, a.value + b.value, 1.0, 1.0),
  sub: (a, b) => dualBinary(a, b, a.value - b.value, 1.0, -1.0),
  mul: (a, b) => dualBinary(a, b, a.value * b.value, b.value, a.value),
  div: (a, b) => dualBinary(a, b, a.value / b.value, 1.0 / b.value,
    -a.value / (b.value * b.value)),
  pow: (a, n) => dualBinary(a, a, a.value ** n, n * a.value ** (n - 1.0), 0.0),
  sqrt: (a) => dualBinary(a, a, Math.sqrt(a.value), 0.5 / Math.sqrt(a.value), 0.0),
  exp: (a) => dualBinary(a, a, Math.exp(a.value), Math.exp(a.value), 0.0),
  log: (a) => dualBinary(a, a, Math.log(a.value), 1.0 / a.value, 0.0),
  sin: (a) => dualBinary(a, a, Math.sin(a.value), Math.cos(a.value), 0.0),
  cos: (a) => dualBinary(a, a, Math.cos(a.value), -Math.sin(a.value), 0.0),
  tanh: (a) => dualBinary(a, a, Math.tanh(a.value), 1.0 - Math.tanh(a.value) ** 2, 0.0),
  sigmoid: (a) => {
    const s = 1.0 / (1.0 + Math.exp(-a.value));
    return dualBinary(a, a, s, s * (1.0 - s), 0.0);
  },
  relu: (a) => dualBinary(a, a, Math.max(a.value, 0.0), a.value > 0.0 ? 1.0 : 0.0, 0.0),
  max: (a, b) => (a.value >= b.value
    ? dualBinary(a, b, a.value, 1.0, 0.0)
    : dualBinary(a, b, b.value, 0.0, 1.0)),
};

/** Forward-mode gradient of `f` (a function of duals) at `point`. */
function dualGrad(f, point) {
  const out = [];
  for (let i = 0; i < point.length; i++) {
    const args = point.map((value, j) => new Dual(value, i === j ? 1.0 : 0.0));
    out.push(f(...args).deriv);
  }
  return out;
}

/** Central-difference gradient of a plain-float function. */
function numericGrad(f, point, h = 1e-6) {
  return point.map((_, i) => {
    const plus = point.slice();
    const minus = point.slice();
    plus[i] += h;
    minus[i] -= h;
    return (f(...plus) - f(...minus)) / (2.0 * h);
  });
}

function maxAbsDiff(a, b) {
  return a.reduce((best, value, i) => Math.max(best, Math.abs(value - b[i])), 0.0);
}

// --- the four functions of the header, once on the tape and once over duals -------------------

function multinomial(tape, x, y, z) {
  return add(tape, add(tape, mul(tape, x, y), mul(tape, y, z)), mul(tape, z, x));
}

function multinomialDual(x, y, z) {
  return dual.add(dual.add(dual.mul(x, y), dual.mul(y, z)), dual.mul(z, x));
}

function quintic(tape, x) {
  return add(tape, sub(tape, powOp(tape, x, 5), mul(tape, tape.constant(3.0), powOp(tape, x, 3))),
    mul(tape, tape.constant(2.0), x));
}

function quinticDual(x) {
  return dual.add(dual.sub(dual.pow(x, 5), dual.mul(new Dual(3.0, 0.0), dual.pow(x, 3))),
    dual.mul(new Dual(2.0, 0.0), x));
}

function mixed(tape, x, y) {
  const xy = mul(tape, x, y);
  return add(tape, sinOp(tape, xy),
    mul(tape, mul(tape, x, x), logOp(tape, y)));
}

function mixedDual(x, y) {
  const xy = dual.mul(x, y);
  return dual.add(dual.sin(xy), dual.mul(dual.mul(x, x), dual.log(y)));
}

function sigmoidChain(tape, x, depth = 10) {
  let node = x;
  for (let i = 0; i < depth; i++) node = sigmoidOp(tape, node);
  return node;
}

function sigmoidChainDual(x, depth = 10) {
  let node = x;
  for (let i = 0; i < depth; i++) node = dual.sigmoid(node);
  return node;
}

function kitchenSink(tape, x, y, z) {
  const a = expOp(tape, mul(tape, x, y));
  const b = logOp(tape, add(tape, sqr(tape, z), tape.constant(1.0)));
  const c = sqrtOp(tape, reluOp(tape, add(tape, tape.constant(0.5), powOp(tape, x, 2))));
  const d = maxOp(tape, y, z);
  return add(tape, sub(tape, add(tape, a, b), c), tanhOp(tape, d));
}

function sqr(tape, x) {
  return mul(tape, x, x);
}

/** Plain-JS twin of `kitchenSink` for the finite-difference check. */
function kitchenSinkPlain(x, y, z) {
  const a = Math.exp(x * y);
  const b = Math.log(z * z + 1.0);
  const c = Math.sqrt(Math.max(0.5 + x * x, 0.0));
  const d = Math.max(y, z);
  return a + b - c + Math.tanh(d);
}

function kitchenSinkDual(x, y, z) {
  const a = dual.exp(dual.mul(x, y));
  const b = dual.log(dual.add(dual.mul(z, z), new Dual(1.0, 0.0)));
  const c = dual.sqrt(dual.relu(dual.add(new Dual(0.5, 0.0), dual.mul(x, x))));
  const d = dual.max(y, z);
  return dual.add(dual.sub(dual.add(a, b), c), dual.tanh(d));
}

function main() {
  const assert = require("node:assert");
  const close = (a, b, tol, message) =>
    assert.ok(Math.abs(a - b) <= tol, `${message}: |${a} - ${b}| > ${tol}`);

  // (a) multinomial: exact gradient at three points.
  for (const [px, py, pz] of [[2.0, -3.0, 0.5], [0.0, 0.0, 0.0], [1.5, 1.5, 1.5]]) {
    const tape = new Tape();
    const x = tape.variable(px);
    const y = tape.variable(py);
    const z = tape.variable(pz);
    const out = multinomial(tape, x, y, z);
    close(out.value, px * py + py * pz + pz * px, 1e-12, "multinomial forward value");
    const grad = tape.gradient(out, [x, y, z]);
    assert.deepStrictEqual(grad.map((value, i) =>
      Math.abs(value - [py + pz, px + pz, px + py][i]) <= 1e-12), [true, true, true],
      "exact multinomial gradient");
    close(maxAbsDiff(grad, dualGrad(multinomialDual, [px, py, pz])), 0.0, 1e-12,
      "reverse equals forward mode");
  }

  // (b) quintic: exact symbolic derivative.
  for (const px of [1.7, -0.4, 2.25]) {
    const tape = new Tape();
    const x = tape.variable(px);
    const out = quintic(tape, x);
    close(out.value, px ** 5 - 3.0 * px ** 3 + 2.0 * px, 1e-9, "quintic value");
    const [dx] = tape.gradient(out, [x]);
    assert.ok(Math.abs(dx - (5.0 * px ** 4 - 9.0 * px ** 2 + 2.0)) <= 1e-9,
      "quintic gradient matches 5x^4 - 9x^2 + 2");
    close(dx, dualGrad(quinticDual, [px])[0], 1e-12, "quintic: reverse equals forward");
  }

  // (c) branching and unused nodes.
  {
    const tape = new Tape();
    const x = tape.variable(3.0);
    const square = mul(tape, x, x);
    close(square.value, 9.0, 1e-12, "x * x value");
    close(tape.gradient(square, [x])[0], 6.0, 1e-12, "x * x gradient is 2x");
  }
  {
    const tape = new Tape();
    const x = tape.variable(-2.5);
    const twice = add(tape, mul(tape, x, x), mul(tape, x, x));
    close(tape.gradient(twice, [x])[0], 4.0 * -2.5, 1e-12, "x^2 + x^2 gradient is 4x");
  }
  {
    const tape = new Tape();
    const x = tape.variable(1.5);
    const y = tape.variable(4.0);
    const out = add(tape, mul(tape, x, tape.constant(2.0)), tape.constant(1.0));
    const [dx, dy] = tape.gradient(out, [x, y]);
    close(dx, 2.0, 1e-12, "gradient w.r.t. the used variable");
    assert.strictEqual(dy, 0.0, "an unused variable has an exactly zero adjoint");
  }
  {
    const tape = new Tape();
    const x = tape.variable(0.75);
    const y = tape.variable(2.0);
    const combined = add(tape, mul(tape, x, y), mul(tape, x, y));
    const [dx, dy] = tape.gradient(combined, [x, y]);
    close(dx, 2.0 * 2.0, 1e-12, "shared subexpression accumulates twice");
    close(dy, 2.0 * 0.75, 1e-12, "shared subexpression accumulates twice (second input)");
  }

  // (d) activation derivatives.
  for (const value of [-3.0, -0.5, 0.0, 0.25, 2.0]) {
    const tape = new Tape();
    const x = tape.variable(value);
    const s = sigmoidOp(tape, x);
    const [ds] = tape.gradient(s, [x]);
    close(ds, s.value * (1.0 - s.value), 1e-12, "sigmoid derivative");
    const tape2 = new Tape();
    const x2 = tape2.variable(value);
    const t = tanhOp(tape2, x2);
    close(tape2.gradient(t, [x2])[0], 1.0 - t.value ** 2, 1e-12, "tanh derivative");
    const tape3 = new Tape();
    const x3 = tape3.variable(value);
    const r = reluOp(tape3, x3);
    const expected = value > 0 ? 1.0 : 0.0;
    close(tape3.gradient(r, [x3])[0], expected, 0.0, "relu subgradient");
  }
  {
    const tape = new Tape();
    const x = tape.variable(0.0);
    close(reluOp(tape, x).value, 0.0, 0.0, "relu(0) = 0");
    close(tape.gradient(reluOp(tape, x), [x])[0], 0.0, 0.0, "documented subgradient at 0");
  }

  // (e) reverse vs forward vs finite differences on composite functions.
  const points = [[0.7, 1.4], [-1.2, 0.35], [2.0, 3.0]];
  for (const [px, py] of points) {
    const tape = new Tape();
    const x = tape.variable(px);
    const y = tape.variable(py);
    const out = mixed(tape, x, y);
    const grad = tape.gradient(out, [x, y]);
    const forward = dualGrad(mixedDual, [px, py]);
    close(maxAbsDiff(grad, forward), 0.0, 1e-12, "mixed: reverse equals forward");
    const numeric = numericGrad((a, b) => Math.sin(a * b) + a * a * Math.log(b), [px, py]);
    assert.ok(maxAbsDiff(grad, numeric) <= 1e-7, "mixed: matches central differences to 1e-7");
    const dxExact = py * Math.cos(px * py) + 2.0 * px * Math.log(py);
    const dyExact = px * Math.cos(px * py) + px * px / py;
    close(grad[0], dxExact, 1e-9, "mixed: closed form d/dx");
    close(grad[1], dyExact, 1e-9, "mixed: closed form d/dy");
  }
  for (const value of [0.0, 0.8, -1.5]) {
    const tape = new Tape();
    const x = tape.variable(value);
    const out = sigmoidChain(tape, x, 10);
    const [dx] = tape.gradient(out, [x]);
    const forward = dualGrad((x1) => sigmoidChainDual(x1, 10), [value])[0];
    close(dx, forward, 1e-12, "10-deep sigmoid chain: reverse equals forward");
    const numeric = numericGrad((x1) => {
      let value2 = x1;
      for (let i = 0; i < 10; i++) value2 = 1.0 / (1.0 + Math.exp(-value2));
      return value2;
    }, [value])[0];
    assert.ok(Math.abs(dx - numeric) <= 1e-7, "10-deep sigmoid chain vs finite differences");
    assert.ok(dx >= 0.0 && dx <= 1.0, "sigmoid chain gradient stays in [0, 1]");
  }
  for (const point of [[0.4, -0.8, 1.3], [-1.1, 2.2, 0.5], [0.9, -1.4, 1.1]]) {
    const tape = new Tape();
    const [px, py, pz] = point;
    const x = tape.variable(px);
    const y = tape.variable(py);
    const z = tape.variable(pz);
    const out = kitchenSink(tape, x, y, z);
    close(out.value, kitchenSinkPlain(px, py, pz), 1e-12, "kitchen sink forward value");
    const grad = tape.gradient(out, [x, y, z]);
    close(maxAbsDiff(grad, dualGrad(kitchenSinkDual, point)), 0.0, 1e-12,
      "kitchen sink: reverse equals forward");
    assert.ok(maxAbsDiff(grad, numericGrad(kitchenSinkPlain, point)) <= 1e-7,
      "kitchen sink: matches central differences to 1e-7");
  }
  {
    // max picks the larger branch and routes the gradient there
    const tape = new Tape();
    const x = tape.variable(2.0);
    const y = tape.variable(5.0);
    const out = maxOp(tape, x, y);
    const [dx, dy] = tape.gradient(out, [x, y]);
    close(out.value, 5.0, 1e-12, "max forward value");
    assert.deepStrictEqual([dx, dy], [0.0, 1.0], "max routes the gradient to the winner");
    const tape2 = new Tape();
    const x2 = tape2.variable(7.0);
    const y2 = tape2.variable(3.0);
    const out2 = maxOp(tape2, x2, y2);
    assert.deepStrictEqual(tape2.gradient(out2, [x2, y2]), [1.0, 0.0],
      "max routes to the other side when it wins");
    const tape3 = new Tape();                       // documented tie rule: equality favours `a`
    const x3 = tape3.variable(4.0);
    const y3 = tape3.variable(4.0);
    assert.deepStrictEqual(tape3.gradient(maxOp(tape3, x3, y3), [x3, y3]), [1.0, 0.0],
      "max ties favour the first argument");
  }

  // (f) linearity of the gradient and constant handling.
  {
    const tape = new Tape();
    const x = tape.variable(0.6);
    const y = tape.variable(1.9);
    const f = mul(tape, x, y);
    const g = add(tape, mul(tape, x, x), logOp(tape, y));
    const sum = add(tape, f, g);
    const gradSum = tape.gradient(sum, [x, y]);
    const tapeF = new Tape();
    const xf = tapeF.variable(0.6);
    const yf = tapeF.variable(1.9);
    const gradF = tapeF.gradient(mul(tapeF, xf, yf), [xf, yf]);
    const tapeG = new Tape();
    const xg = tapeG.variable(0.6);
    const yg = tapeG.variable(1.9);
    const gradG = tapeG.gradient(add(tapeG, mul(tapeG, xg, xg), logOp(tapeG, yg)), [xg, yg]);
    for (let i = 0; i < 2; i++) {
      close(gradSum[i], gradF[i] + gradG[i], 1e-12, "grad(f + g) = grad f + grad g");
    }
    const tapeC = new Tape();
    const xc = tapeC.variable(0.6);
    const out = add(tapeC, mul(tapeC, tapeC.constant(4.0), xc), tapeC.constant(7.0));
    close(out.value, 9.4, 1e-12, "constant scaling and offset");
    close(tapeC.gradient(out, [xc])[0], 4.0, 0.0, "a constant contributes no gradient");
  }

  // (g) documented error behaviour.
  {
    const tape = new Tape();
    const x = tape.variable(-1.0);
    assert.throws(() => logOp(tape, x), RangeError, "log of a negative value throws");
    const tape2 = new Tape();
    const zero = tape2.variable(0.0);
    assert.throws(() => logOp(tape2, zero), RangeError, "log(0) throws");
    const tape3 = new Tape();
    const x3 = tape3.variable(0.25);
    const root = sqrtOp(tape3, x3);
    close(tape3.gradient(root, [x3])[0], 1.0 / (2.0 * Math.sqrt(0.25)), 1e-12, "sqrt derivative");
    const tape4 = new Tape();
    const x4 = tape4.variable(0.8);
    const inverse = div(tape4, tape4.constant(1.0), x4);
    close(tape4.gradient(inverse, [x4])[0], -1.0 / (0.8 * 0.8), 1e-12, "1 / x derivative");
    const tape5 = new Tape();
    const x5 = tape5.variable(0.3);
    const absVal = absOp(tape5, neg(tape5, x5));
    close(absVal.value, 0.3, 1e-12, "abs of a negation");
    close(tape5.gradient(absVal, [x5])[0], 1.0, 1e-12, "d|u|/dx = 1 for u > 0 (through neg)");
  }
}

module.exports = { Tape, add, sub, mul, div, neg, powOp, sqrtOp, expOp, logOp, sinOp, cosOp,
  tanhOp, sigmoidOp, reluOp, absOp, maxOp, Dual, dual, dualGrad, numericGrad, maxAbsDiff, sqr,
  multinomial, multinomialDual, quintic, quinticDual, mixed, mixedDual, sigmoidChain,
  sigmoidChainDual, kitchenSink, kitchenSinkPlain, kitchenSinkDual };

if (require.main === module) {
  main();
  console.log("PASS");
}
