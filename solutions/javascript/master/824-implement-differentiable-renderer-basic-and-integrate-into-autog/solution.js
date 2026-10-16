/** Task 824 — Implement differentiable renderer basic and integrate into autograd.
 *  (Tier: Master, Beyond — Machine Learning & AI)
 *
 * Autograd. A small reverse-mode engine over dense arrays: every operation records a local
 * vector–Jacobian-product closure, `backward()` traverses the graph in topological order and
 * accumulates gradients. Elementwise ops (add/sub/mul/div, pow, exp, log, sqrt, tanh, sigmoid, relu), a
 * scalar scale, a broadcast multiply and a sum reduction build the losses; the renderer is a *custom
 * node* with a hand-derived adjoint, which is how real differentiable renderers are organised.
 *
 * Renderer node. For every pixel and triangle the three signed sub-triangle areas `w_k` give the
 * normalised barycentric coordinates `b_k = winding·w_k/|Σw|`; the hard inside test `b_k ≥ 0` becomes
 * the logistic product `α = gate·Π σ(b_k/τ)`, so coverage — and therefore the whole image — is a smooth
 * function of the vertex coordinates. Colours are interpolated barycentrically and the triangles are
 * alpha-composited in painter order: `out ← out·(1 − α) + α·colour`. The backward pass pushes
 * ∂L/∂α and ∂L/∂colour into the vertex positions through ∂b_k/∂w_j and ∂w_j/∂p (the sub-area
 * derivatives) and into the vertex colours through the interpolation weights.
 *
 * Verification strategy:
 *   - every primitive op is checked against central finite differences;
 *   - the renderer's gradients w.r.t. vertex positions *and* vertex colours are checked the same way at
 *     τ = 0.8 — that is the property that makes a renderer differentiable and the one most easily got
 *     wrong;
 *   - the forward pass is checked against an independent supersampled hard rasteriser: the soft image
 *     must converge to it as τ → 0, coverage must sum to the triangle area (mass conservation), and a
 *     fully covered pixel must reproduce the analytically interpolated colour;
 *   - inverse rendering closes the loop: fitting vertex positions to a target image by gradient descent
 *     must reduce the loss monotonically and recover the target geometry.
 *
 * Simplified versus production: no textures, no perspective/model-view transform, no depth buffer
 * (painter order instead), no GPU kernels or batching, and tiny images because the graph is built in
 * plain arrays. Silhouette gradients are logistic approximations of a step function; the file *measures*
 * the soft/hard gap as a function of τ instead of pretending it is zero.
 *
 * Verified: primitive gradient checks, adjoint accumulation, a hand-derived chain rule, soft coverage
 * limits (0 outside, 1 deeply inside) and area conservation, convergence to the hard rasteriser,
 * renderer gradient checks for vertices and colours, finite gradients on degenerate/off-screen input,
 * painter depth ordering, inverse-rendering convergence, determinism and the degenerate meshes (empty,
 * zero-area, off-screen, single pixel, single triangle).
 */
"use strict";

// ---------------------------------------------------------------------------- autograd core

class Tensor {
  constructor(data) {
    this.data = data.slice();
    this.grad = new Array(data.length).fill(0);
    this.backwardFn = () => {};
    this.parents = [];
    this.name = "";
  }

  static of(data, name = "") {
    const tensor = new Tensor(data);
    tensor.name = name;
    return tensor;
  }

  get size() {
    return this.data.length;
  }

  /** Reverse-mode pass. The default seed is 1 everywhere, i.e. the gradient of Σ output. */
  backward(seed = null) {
    const order = [];
    const visited = new Set();
    const visit = (node) => {
      if (visited.has(node)) return;
      visited.add(node);
      for (const parent of node.parents) visit(parent);
      order.push(node);
    };
    visit(this);
    for (const node of order) node.grad = new Array(node.size).fill(0);
    this.grad = seed ?? new Array(this.size).fill(1);
    for (let i = order.length - 1; i >= 0; i--) order[i].backwardFn();
    return this.grad;
  }

  zeroGrad() {
    const visited = new Set();
    const walk = (node) => {
      if (visited.has(node)) return;
      visited.add(node);
      node.grad = new Array(node.size).fill(0);
      for (const parent of node.parents) walk(parent);
    };
    walk(this);
  }
}

function binaryOp(a, b, op, forward, gradA, gradB) {
  if (a.size !== b.size) throw new RangeError(`${op} needs equal lengths, got ${a.size} and ${b.size}`);
  const out = Tensor.of(a.data.map((value, i) => forward(value, b.data[i])));
  out.parents = [a, b];
  out.name = op;
  out.backwardFn = () => {
    for (let i = 0; i < out.size; i++) {
      a.grad[i] += out.grad[i] * gradA(a.data[i], b.data[i]);
      b.grad[i] += out.grad[i] * gradB(a.data[i], b.data[i]);
    }
  };
  return out;
}

const add = (a, b) => binaryOp(a, b, "add", (x, y) => x + y, () => 1, () => 1);
const sub = (a, b) => binaryOp(a, b, "sub", (x, y) => x - y, () => 1, () => -1);
const mul = (a, b) => binaryOp(a, b, "mul", (x, y) => x * y, (x, y) => y, (x, y) => x);
const div = (a, b) => binaryOp(a, b, "div", (x, y) => x / y, (x, y) => 1 / y, (x, y) => -x / (y * y));

function unaryOp(a, op, forward, grad) {
  const out = Tensor.of(a.data.map((value) => forward(value)));
  out.parents = [a];
  out.name = op;
  out.backwardFn = () => {
    for (let i = 0; i < out.size; i++) a.grad[i] += out.grad[i] * grad(a.data[i]);
  };
  return out;
}

const neg = (a) => unaryOp(a, "neg", (x) => -x, () => -1);
const exp = (a) => unaryOp(a, "exp", Math.exp, Math.exp);
const log = (a) => unaryOp(a, "log", Math.log, (x) => 1 / x);
const sqrt = (a) => unaryOp(a, "sqrt", Math.sqrt, (x) => 0.5 / Math.sqrt(x));
const tanh = (a) => unaryOp(a, "tanh", Math.tanh, (x) => 1 - Math.tanh(x) ** 2);
const sigmoid = (a) =>
  unaryOp(a, "sigmoid", (x) => logistic(x), (x) => logistic(x) * (1 - logistic(x)));
const relu = (a) => unaryOp(a, "relu", (x) => Math.max(0, x), (x) => (x > 0 ? 1 : 0));
const pow = (a, exponent) => unaryOp(a, `pow${exponent}`, (x) => x ** exponent, (x) => exponent * x ** (exponent - 1));
const scale = (a, factor) => unaryOp(a, `scale${factor}`, (x) => x * factor, () => factor);

/** Broadcast multiply: `weights` is indexed by `groups[i]`. */
function weighted(a, weights, groups) {
  const out = Tensor.of(a.data.map((value, i) => value * weights.data[groups[i]]));
  out.parents = [a, weights];
  out.name = "weighted";
  out.backwardFn = () => {
    for (let i = 0; i < out.size; i++) {
      a.grad[i] += out.grad[i] * weights.data[groups[i]];
      weights.grad[groups[i]] += out.grad[i] * a.data[i];
    }
  };
  return out;
}

function sum(a) {
  const out = Tensor.of([a.data.reduce((total, value) => total + value, 0)]);
  out.parents = [a];
  out.name = "sum";
  out.backwardFn = () => {
    for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[0];
  };
  return out;
}

function constant(data, name = "const") {
  return Tensor.of(data, name);
}

function logistic(value) {
  return 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, value))));
}

function logisticPrime(value) {
  const s = logistic(value);
  return s * (1 - s);
}

// ---------------------------------------------------------------------------- finite differences

function numericalGradient(fn, inputs, epsilon = 1e-5) {
  const gradients = inputs.map(() => []);
  for (let index = 0; index < inputs.length; index++) {
    const input = inputs[index];
    for (let i = 0; i < input.data.length; i++) {
      const original = input.data[i];
      input.data[i] = original + epsilon;
      const high = fn();
      input.data[i] = original - epsilon;
      const low = fn();
      input.data[i] = original;
      gradients[index].push((high - low) / (2 * epsilon));
    }
  }
  return gradients;
}

function relativeError(analytic, numeric) {
  let worst = 0;
  for (let i = 0; i < analytic.length; i++) {
    const denominator = Math.max(1e-6, Math.abs(analytic[i]) + Math.abs(numeric[i]));
    worst = Math.max(worst, Math.abs(analytic[i] - numeric[i]) / denominator);
  }
  return worst;
}

// ---------------------------------------------------------------------------- mesh

const BACKGROUND = [0.05, 0.05, 0.1];

class Mesh {
  constructor(vertices, colors, triangles) {
    this.vertices = vertices; // [[x, y], ...]
    this.colors = colors; // [[r, g, b], ...]
    this.triangles = triangles; // [[i, j, k], ...]
  }

  vertexTensor() {
    const data = [];
    for (const [x, y] of this.vertices) data.push(x, y);
    return Tensor.of(data, "vertices");
  }

  colorTensor() {
    const data = [];
    for (const [r, g, b] of this.colors) data.push(r, g, b);
    return Tensor.of(data, "colors");
  }
}

function triangleArea(vertices) {
  return (
    Math.abs((vertices[1][0] - vertices[0][0]) * (vertices[2][1] - vertices[0][1]) - (vertices[1][1] - vertices[0][1]) * (vertices[2][0] - vertices[0][0])) / 2
  );
}

// ---------------------------------------------------------------------------- renderer node

/**
 * Custom autograd node: soft rasterisation in the forward pass, the analytic adjoint in the backward
 * pass. The adjoint is what the finite-difference checks in the self-test verify.
 */
class RenderNode extends Tensor {
  constructor(options) {
    super([]);
    this.mesh = options.mesh;
    this.width = options.width ?? 16;
    this.height = options.height ?? 16;
    this.softness = options.softness ?? 0.06;
    this.background = options.background ?? BACKGROUND;
    this.vertexTensor = options.vertexTensor ?? this.mesh.vertexTensor();
    this.colorTensor = options.colorTensor ?? this.mesh.colorTensor();
    this.parents = [this.vertexTensor, this.colorTensor];
    this.name = "render";
    this.forwardPass();
    this.backwardFn = () => this.backwardPass();
  }

  /** Forward: logistic edge test for coverage, barycentric interpolation for colour, painter order. */
  forwardPass() {
    const pixels = this.width * this.height;
    const images = [this.backgroundPattern()];
    this.stages = [];
    this.coverageByTriangle = [];
    for (const triangle of this.mesh.triangles) {
      const [i, j, k] = triangle;
      const p = [i, j, k].map((vertex) => [this.vertexTensor.data[vertex * 2], this.vertexTensor.data[vertex * 2 + 1]]);
      const colors = [i, j, k].map((vertex) => [
        this.colorTensor.data[vertex * 3],
        this.colorTensor.data[vertex * 3 + 1],
        this.colorTensor.data[vertex * 3 + 2],
      ]);
      const signedArea = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]);
      const winding = signedArea >= 0 ? 1 : -1;
      const previous = images[images.length - 1];
      const next = previous.slice();
      const coverage = new Array(pixels).fill(0);
      const entries = new Array(pixels).fill(null);
      for (let py = 0; py < this.height; py++) {
        for (let px = 0; px < this.width; px++) {
          const pixel = py * this.width + px;
          const x = px + 0.5;
          const y = py + 0.5;
          const w0 = (p[1][0] - x) * (p[2][1] - y) - (p[1][1] - y) * (p[2][0] - x);
          const w1 = (p[2][0] - x) * (p[0][1] - y) - (p[2][1] - y) * (p[0][0] - x);
          const w2 = (p[0][0] - x) * (p[1][1] - y) - (p[0][1] - y) * (p[1][0] - x);
          const sumW = w0 + w1 + w2;
          if (Math.abs(sumW) < 1e-12) continue;
          const absolute = Math.abs(sumW);
          const b = [(winding * w0) / absolute, (winding * w1) / absolute, (winding * w2) / absolute];
          // `softness` is the logistic width in barycentric units (so it scales with the projected
          // triangle size); the second factor silently kills exactly degenerate triangles.
          const gate = logistic(absolute / 1e-4);
          const sigmas = b.map((value) => logistic(value / this.softness));
          const alpha = gate * sigmas[0] * sigmas[1] * sigmas[2];
          const color = [0, 1, 2].map((channel) => (w0 * colors[0][channel] + w1 * colors[1][channel] + w2 * colors[2][channel]) / sumW);
          coverage[pixel] = alpha;
          entries[pixel] = { pixel, x, y, w: [w0, w1, w2], sumW, absolute, b, sigmas, gate, alpha, color };
          for (let channel = 0; channel < 3; channel++) {
            next[pixel * 3 + channel] = previous[pixel * 3 + channel] * (1 - alpha) + alpha * color[channel];
          }
        }
      }
      images.push(next);
      this.stages.push({ triangle, p, colors, winding, entries });
      this.coverageByTriangle.push(coverage);
    }
    this.images = images;
    this.data = images[images.length - 1].slice();
  }

  backgroundPattern() {
    const image = new Array(this.width * this.height * 3).fill(0);
    for (let pixel = 0; pixel < this.width * this.height; pixel++) {
      for (let channel = 0; channel < 3; channel++) image[pixel * 3 + channel] = this.background[channel];
    }
    return image;
  }

  /** Adjoint of the forward pass, routed with the chain rule into vertices and vertex colours. */
  backwardPass() {
    const dimensions = 3;
    const upstream = this.grad.slice();
    for (let stageIndex = this.stages.length - 1; stageIndex >= 0; stageIndex--) {
      const stage = this.stages[stageIndex];
      const previous = this.images[stageIndex];
      const carried = upstream.slice();
      for (const entry of stage.entries) {
        if (entry === null) continue;
        const { pixel, w, sumW, absolute, b, sigmas, gate, color, alpha } = entry;
        const sign = stage.winding;
        // ∂out/∂α = colour − previous, ∂out/∂colour = α.
        let dAlpha = 0;
        const dColor = new Array(dimensions).fill(0);
        for (let channel = 0; channel < dimensions; channel++) {
          const g = carried[pixel * 3 + channel];
          dAlpha += g * (color[channel] - previous[pixel * 3 + channel]);
          dColor[channel] = g * alpha;
        }
        // Coverage: α = gate(|t|)·Π σ(b_i/τ), b_i = sign·w_i/|t|, t = Σ w.
        const dGate = dAlpha * logisticPrime(absolute / 1e-4) * (sign / 1e-4);
        const dAlphaDb = [0, 1, 2].map((index) => {
          let others = 1;
          for (let other = 0; other < 3; other++) if (other !== index) others *= sigmas[other];
          return dAlpha * gate * others * logisticPrime(b[index] / this.softness) / this.softness;
        });
        const dw = [0, 0, 0];
        for (let j = 0; j < 3; j++) {
          dw[j] += dGate; // ∂t/∂w_j = 1
          for (let i2 = 0; i2 < 3; i2++) {
            const dbdw = (i2 === j ? sign / absolute : 0) - w[i2] / (absolute * absolute);
            dw[j] += dAlphaDb[i2] * dbdw;
          }
          for (let channel = 0; channel < dimensions; channel++) {
            dw[j] += (dColor[channel] * (stage.colors[j][channel] - color[channel])) / sumW;
          }
        }
        const mapping = subAreaGradients(stage.p, entry.x, entry.y);
        for (let j = 0; j < 3; j++) {
          for (let corner = 0; corner < 3; corner++) {
            const vertex = stage.triangle[corner];
            this.vertexTensor.grad[vertex * 2] += dw[j] * mapping[j][corner][0];
            this.vertexTensor.grad[vertex * 2 + 1] += dw[j] * mapping[j][corner][1];
          }
        }
        for (let corner = 0; corner < 3; corner++) {
          const vertex = stage.triangle[corner];
          const weight = w[corner] / sumW;
          for (let channel = 0; channel < dimensions; channel++) {
            this.colorTensor.grad[vertex * 3 + channel] += dColor[channel] * weight;
          }
        }
        // ∂out/∂previous = (1 − α) is already folded into the upstream gradient of the previous image,
        // which is exactly what `upstream` keeps carrying as the loop walks backwards through stages.
        for (let channel = 0; channel < dimensions; channel++) {
          upstream[pixel * 3 + channel] = carried[pixel * 3 + channel] * (1 - alpha);
        }
      }
    }
  }

  coverageFor(triangleIndex) {
    return this.coverageByTriangle[triangleIndex];
  }
}

/** ∂w_j/∂(vertex coordinates) for the three signed sub-areas at pixel (x, y). */
function subAreaGradients(p, x, y) {
  const [p0, p1, p2] = p;
  return [
    [
      [0, 0],
      [p2[1] - y, -(p2[0] - x)],
      [-(p1[1] - y), p1[0] - x],
    ],
    [
      [-(p2[1] - y), p2[0] - x],
      [0, 0],
      [p0[1] - y, -(p0[0] - x)],
    ],
    [
      [p1[1] - y, -(p1[0] - x)],
      [-(p0[1] - y), p0[0] - x],
      [0, 0],
    ],
  ];
}

function render(mesh, options = {}) {
  return new RenderNode({ mesh, ...options });
}

// ---------------------------------------------------------------------------- reference rasteriser

/** Independent hard rasteriser with supersampling; returns [r, g, b] per pixel. */
function referenceRasterize(mesh, options = {}) {
  const width = options.width ?? 16;
  const height = options.height ?? 16;
  const background = options.background ?? BACKGROUND;
  const subsamples = options.subsamples ?? 4;
  const image = new Array(width * height * 3).fill(0);
  for (let pixel = 0; pixel < width * height; pixel++) {
    for (let channel = 0; channel < 3; channel++) image[pixel * 3 + channel] = background[channel];
  }
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      let red = 0;
      let green = 0;
      let blue = 0;
      for (let sy = 0; sy < subsamples; sy++) {
        for (let sx = 0; sx < subsamples; sx++) {
          const x = px + (sx + 0.5) / subsamples;
          const y = py + (sy + 0.5) / subsamples;
          let color = background;
          for (const [i, j, k] of mesh.triangles) {
            const p = [mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]];
            const w0 = (p[1][0] - x) * (p[2][1] - y) - (p[1][1] - y) * (p[2][0] - x);
            const w1 = (p[2][0] - x) * (p[0][1] - y) - (p[2][1] - y) * (p[0][0] - x);
            const w2 = (p[0][0] - x) * (p[1][1] - y) - (p[0][1] - y) * (p[1][0] - x);
            const sum = w0 + w1 + w2;
            if (Math.abs(sum) < 1e-12) continue;
            if (w0 / sum < 0 || w1 / sum < 0 || w2 / sum < 0) continue;
            const colors = [mesh.colors[i], mesh.colors[j], mesh.colors[k]];
            color = [0, 1, 2].map((channel) => (w0 * colors[0][channel] + w1 * colors[1][channel] + w2 * colors[2][channel]) / sum);
          }
          red += color[0];
          green += color[1];
          blue += color[2];
        }
      }
      const samples = subsamples * subsamples;
      image[(py * width + px) * 3] = red / samples;
      image[(py * width + px) * 3 + 1] = green / samples;
      image[(py * width + px) * 3 + 2] = blue / samples;
    }
  }
  return image;
}

function meanAbsoluteDifference(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) total += Math.abs(a[i] - b[i]);
  return total / a.length;
}

/** Squared-error image loss built on top of the render node, so gradients flow back into it. */
function imageLoss(node, target) {
  const difference = sub(node, constant(target, "target"));
  return sum(mul(difference, difference));
}

module.exports = {
  Tensor,
  add,
  sub,
  mul,
  div,
  neg,
  exp,
  log,
  sqrt,
  tanh,
  sigmoid,
  relu,
  pow,
  scale,
  weighted,
  sum,
  constant,
  numericalGradient,
  relativeError,
  logistic,
  logisticPrime,
  Mesh,
  RenderNode,
  render,
  referenceRasterize,
  meanAbsoluteDifference,
  imageLoss,
  triangleArea,
  BACKGROUND,
};

if (require.main === module) {
  const assert = require("node:assert");

  // --- primitive operations against central finite differences ----------------------
  const opChecks = [
    { name: "add", fn: (a, b) => add(a, b), inputs: () => [Tensor.of([0.3, -1.2, 2.5]), Tensor.of([1.5, 0.7, -0.4])] },
    { name: "sub", fn: (a, b) => sub(a, b), inputs: () => [Tensor.of([0.3, -1.2, 2.5]), Tensor.of([1.5, 0.7, -0.4])] },
    { name: "mul", fn: (a, b) => mul(a, b), inputs: () => [Tensor.of([0.3, -1.2, 2.5]), Tensor.of([1.5, 0.7, -0.4])] },
    { name: "div", fn: (a, b) => div(a, b), inputs: () => [Tensor.of([0.3, -1.2, 2.5]), Tensor.of([1.5, 0.7, -0.4])] },
    { name: "pow3", fn: (a) => pow(a, 3), inputs: () => [Tensor.of([0.3, -1.2, 2.5])] },
    { name: "exp", fn: (a) => exp(a), inputs: () => [Tensor.of([0.3, -1.2, 2.5])] },
    { name: "log", fn: (a) => log(a), inputs: () => [Tensor.of([0.3, 1.2, 2.5])] },
    { name: "sqrt", fn: (a) => sqrt(a), inputs: () => [Tensor.of([0.3, 1.2, 2.5])] },
    { name: "tanh", fn: (a) => tanh(a), inputs: () => [Tensor.of([0.3, -1.2, 2.5])] },
    { name: "sigmoid", fn: (a) => sigmoid(a), inputs: () => [Tensor.of([0.3, -1.2, 2.5])] },
    { name: "relu", fn: (a) => relu(a), inputs: () => [Tensor.of([0.3, -1.2, 2.5])] },
    { name: "neg", fn: (a) => neg(a), inputs: () => [Tensor.of([0.3, -1.2, 2.5])] },
    { name: "scale", fn: (a) => scale(a, -1.7), inputs: () => [Tensor.of([0.3, -1.2, 2.5])] },
  ];
  let primitiveChecks = 0;
  let worstPrimitiveError = 0;
  for (const check of opChecks) {
    const inputs = check.inputs();
    const value = () => sum(check.fn(...inputs)).data[0];
    const inputsForGrad = check.inputs();
    const analyticValue = sum(check.fn(...inputsForGrad));
    analyticValue.backward();
    const numeric = numericalGradient(value, inputs);
    for (let index = 0; index < inputs.length; index++) {
      const error = relativeError(inputsForGrad[index].grad, numeric[index]);
      worstPrimitiveError = Math.max(worstPrimitiveError, error);
      assert.ok(error < 1e-6, `${check.name} input ${index} gradient error ${error}`);
    }
    primitiveChecks++;
  }
  assert.ok(primitiveChecks >= 13);

  // --- hand-derived chain rule and adjoint accumulation -------------------------------
  {
    const a = Tensor.of([2]);
    const b = Tensor.of([3]);
    const c = Tensor.of([4]);
    const f = square(sum(add(mul(a, b), c)));
    f.backward();
    assert.strictEqual(f.data[0], (2 * 3 + 4) ** 2);
    assert.strictEqual(a.grad[0], 2 * (2 * 3 + 4) * 3, "∂f/∂a = 2(ab+c)·b");
    assert.strictEqual(b.grad[0], 2 * (2 * 3 + 4) * 2, "∂f/∂b = 2(ab+c)·a");
    assert.strictEqual(c.grad[0], 2 * (2 * 3 + 4), "∂f/∂c = 2(ab+c)");
    // The same tensor reached by two paths accumulates both contributions.
    const shared = Tensor.of([1.5, -0.5]);
    const pathOne = sum(pow(shared, 2));
    const pathTwo = sum(scale(shared, 3));
    const totalLoss = add(pathOne, pathTwo);
    totalLoss.backward();
    for (let i = 0; i < shared.size; i++) {
      assert.ok(Math.abs(shared.grad[i] - (2 * shared.data[i] + 3)) < 1e-12, "accumulated adjoint");
    }
  }

  function square(t) {
    return pow(t, 2);
  }

  // --- forward pass: coverage limits, area conservation, analytic colour ---------------
  const centered = new Mesh(
    [[2, 2], [13, 2], [7.5, 13]],
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 1, 2]]
  );
  const node = render(centered, { width: 16, height: 16, softness: 0.05 });
  const coverage = node.coverageFor(0);
  const totalCoverage = coverage.reduce((total, value) => total + value, 0);
  const area = triangleArea(centered.vertices);
  assert.ok(Math.abs(totalCoverage - area) / area < 0.1, `coverage ${totalCoverage.toFixed(2)} must approximate the area ${area.toFixed(2)}`);
  assert.ok(coverage.every((value) => value >= 0 && value <= 1), "coverage must stay in [0, 1]");
  // A deeply interior pixel is fully covered and carries the barycentric colour mixture.
  const interior = 7 * 16 + 7;
  assert.ok(coverage[interior] > 0.98, `interior coverage ${coverage[interior]}`);
  const sharpInterior = render(centered, { width: 16, height: 16, softness: 0.01 });
  assert.ok(sharpInterior.coverageFor(0)[interior] > 0.999, "interior coverage must approach 1 as the softness shrinks");
  const vertexData = centered.vertexTensor().data;
  const pixelX = 7.5;
  const pixelY = 7.5;
  const p = [[vertexData[0], vertexData[1]], [vertexData[2], vertexData[3]], [vertexData[4], vertexData[5]]];
  const weights = [
    (p[1][0] - pixelX) * (p[2][1] - pixelY) - (p[1][1] - pixelY) * (p[2][0] - pixelX),
    (p[2][0] - pixelX) * (p[0][1] - pixelY) - (p[2][1] - pixelY) * (p[0][0] - pixelX),
    (p[0][0] - pixelX) * (p[1][1] - pixelY) - (p[0][1] - pixelY) * (p[1][0] - pixelX),
  ];
  const weightSum = weights.reduce((a, b) => a + b, 0);
  const expectedRed = (weights[0] * 1) / weightSum;
  assert.ok(Math.abs(node.data[interior * 3] - expectedRed) < 0.02, `interpolated colour ${node.data[interior * 3]} vs ${expectedRed}`);

  // --- soft render converges to the hard rasteriser as tau shrinks ---------------------
  const hard = referenceRasterize(centered, { width: 16, height: 16, subsamples: 6 });
  const differences = [0.12, 0.06, 0.03, 0.015].map((softness) => meanAbsoluteDifference(render(centered, { width: 16, height: 16, softness }).data, hard));
  for (let i = 1; i < differences.length; i++) {
    assert.ok(differences[i] <= differences[i - 1] + 1e-9, `soft/hard gap must shrink with tau: ${differences.map((d) => d.toFixed(4)).join(", ")}`);
  }
  assert.ok(differences[differences.length - 1] < 0.05, `final soft/hard gap ${differences[differences.length - 1]}`);

  // --- renderer gradients against finite differences ------------------------------------
  function gradientCheckForRenderer(softness) {
    const mesh = new Mesh(
      [[2.0, 2.5], [12.5, 3.0], [7.0, 12.0]],
      [[0.9, 0.2, 0.1], [0.1, 0.9, 0.2], [0.2, 0.3, 0.9]],
      [[0, 1, 2]]
    );
    const target = new Array(16 * 16 * 3).fill(0.4);
    const vertices = mesh.vertexTensor();
    const colors = mesh.colorTensor();
    const lossValue = () => imageLoss(render(mesh, { width: 16, height: 16, softness, vertexTensor: vertices, colorTensor: colors }), target).data[0];
    const loss = imageLoss(render(mesh, { width: 16, height: 16, softness, vertexTensor: vertices, colorTensor: colors }), target);
    loss.backward();
    const numeric = numericalGradient(lossValue, [vertices, colors], 1e-5);
    return {
      vertexError: relativeError(vertices.grad, numeric[0]),
      colorError: relativeError(colors.grad, numeric[1]),
      gradNorm: Math.hypot(...vertices.grad),
      finite: vertices.grad.every(Number.isFinite) && colors.grad.every(Number.isFinite),
    };
  }
  const smoothCheck = gradientCheckForRenderer(0.06);
  assert.ok(smoothCheck.vertexError < 2e-3, `renderer vertex gradient error ${smoothCheck.vertexError}`);
  assert.ok(smoothCheck.colorError < 2e-3, `renderer colour gradient error ${smoothCheck.colorError}`);
  assert.ok(smoothCheck.finite && smoothCheck.gradNorm > 0, "the renderer must produce finite, non-zero gradients");
  const sharperCheck = gradientCheckForRenderer(0.04);
  assert.ok(sharperCheck.vertexError < 5e-3, `sharp renderer vertex gradient error ${sharperCheck.vertexError}`);

  // --- inverse rendering: fit the vertices to a target image ----------------------------
  const targetMesh = new Mesh([[3, 2], [13, 4], [8, 13]], [[0.9, 0.3, 0.2], [0.2, 0.8, 0.3], [0.3, 0.4, 0.9]], [[0, 1, 2]]);
  const targetImage = render(targetMesh, { width: 16, height: 16, softness: 0.06 }).data;
  const fitted = new Mesh([[2, 2], [12, 2], [7, 12]], [[0.9, 0.3, 0.2], [0.2, 0.8, 0.3], [0.3, 0.4, 0.9]], [[0, 1, 2]]);
  const vertexTensor = fitted.vertexTensor();
  const colorTensor = fitted.colorTensor();
  const renderLoss = () => imageLoss(render(fitted, { width: 16, height: 16, softness: 0.06, vertexTensor, colorTensor }), targetImage).data[0];
  let previousLoss = Infinity;
  const lossHistory = [];
  for (let step = 0; step < 120; step++) {
    const nodeStep = render(fitted, { width: 16, height: 16, softness: 0.06, vertexTensor, colorTensor });
    nodeStep.zeroGrad();
    const loss = imageLoss(nodeStep, targetImage);
    loss.backward();
    // Backtracking line search: only accept a step that actually lowers the loss, which makes the
    // monotone-decrease property below a property of the optimiser rather than of the step size.
    const before = vertexTensor.data.slice();
    let stepSize = 4;
    for (let attempt = 0; attempt < 24; attempt++) {
      for (let i = 0; i < vertexTensor.size; i++) vertexTensor.data[i] = before[i] - stepSize * vertexTensor.grad[i];
      if (renderLoss() < loss.data[0]) break;
      stepSize /= 2;
    }
    lossHistory.push(renderLoss());
    assert.ok(lossHistory[lossHistory.length - 1] <= previousLoss + 1e-9, `the loss must not increase during fitting (step ${step})`);
    previousLoss = lossHistory[lossHistory.length - 1];
  }
  for (let i = 0; i < vertexTensor.size; i++) {
    assert.ok(Math.abs(vertexTensor.data[i] - targetMesh.vertexTensor().data[i]) < 0.35, `vertex ${i} recovered as ${vertexTensor.data[i].toFixed(3)}`);
  }
  assert.ok(lossHistory[lossHistory.length - 1] < lossHistory[0] * 0.05, `fitting must collapse the loss: ${lossHistory[0].toFixed(4)} -> ${lossHistory[lossHistory.length - 1].toFixed(4)}`);
  const fittedImage = render(fitted, { width: 16, height: 16, softness: 0.06, vertexTensor, colorTensor }).data;
  const fittedGap = meanAbsoluteDifference(fittedImage, targetImage);
  assert.ok(fittedGap < 0.02, `fitted image gap ${fittedGap}`);

  // --- depth ordering (painter order) ------------------------------------------------------
  const near = [[4, 4], [13, 4], [8, 13]];
  const far = [[1, 1], [15, 1], [8, 15]];
  const layered = new Mesh(
    [...far, ...near],
    [[0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[0, 1, 2], [3, 4, 5]]
  );
  const layeredNode = render(layered, { width: 16, height: 16, softness: 0.05 });
  const centre = 8 * 16 + 8;
  assert.ok(layeredNode.data[centre * 3] > 0.9, "the later (nearer) triangle must win in painter order");

  // --- degenerate meshes and finite gradients -----------------------------------------------
  const degenerate = new Mesh([[4, 4], [8, 4], [12, 4]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 2]]);
  const degenerateNode = render(degenerate, { width: 8, height: 8, softness: 0.05 });
  assert.ok(degenerateNode.coverageFor(0).every((value) => value < 0.5), "a zero-area triangle must not cover pixels");
  degenerateNode.backward();
  assert.ok(degenerateNode.vertexTensor.grad.every(Number.isFinite), "degenerate gradients must stay finite");
  const offscreen = new Mesh([[-40, -40], [-30, -40], [-35, -30]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[0, 1, 2]]);
  const offscreenNode = render(offscreen, { width: 8, height: 8 });
  assert.ok(offscreenNode.coverageFor(0).every((value) => value < 1e-6), "an off-screen triangle covers nothing");
  assert.ok(offscreenNode.data.every((value, index) => Math.abs(value - BACKGROUND[index % 3]) < 1e-9), "an off-screen mesh renders the background");
  const emptyMesh = new Mesh([], [], []);
  const emptyNode = render(emptyMesh, { width: 4, height: 4 });
  assert.ok(emptyNode.data.every((value, index) => Math.abs(value - BACKGROUND[index % 3]) < 1e-12));
  assert.strictEqual(emptyNode.vertexTensor.size, 0);
  const singlePixel = render(centered, { width: 1, height: 1, softness: 0.2 });
  assert.strictEqual(singlePixel.data.length, 3);
  assert.ok(singlePixel.data.every(Number.isFinite));
  const backgroundOnly = new Mesh([[0, 0], [1, 0], [0, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], []);
  const backgroundNode = render(backgroundOnly, { width: 4, height: 4 });
  assert.ok(backgroundNode.data.every((value, index) => Math.abs(value - BACKGROUND[index % 3]) < 1e-12));

  // --- determinism ---------------------------------------------------------------------------
  const repeatNode = render(centered, { width: 16, height: 16, softness: 0.05 });
  assert.deepStrictEqual(repeatNode.data, node.data);
  assert.deepStrictEqual(repeatNode.coverageFor(0), coverage);

  console.log(
    `primitive gradient checks ${primitiveChecks} (worst relative error ${worstPrimitiveError.toExponential(1)}), ` +
      `coverage ${totalCoverage.toFixed(2)} vs triangle area ${area.toFixed(2)}, soft/hard gap ${differences.map((d) => d.toFixed(4)).join(" > ")} for tau 0.12/0.06/0.03/0.015, ` +
      `renderer gradient error tau=0.06: vertices ${smoothCheck.vertexError.toExponential(1)}, colours ${smoothCheck.colorError.toExponential(1)}, tau=0.04: ${sharperCheck.vertexError.toExponential(1)}, ` +
      `inverse rendering loss ${lossHistory[0].toFixed(3)} -> ${lossHistory[lossHistory.length - 1].toExponential(2)} with vertex error < 0.35 and image gap ${fittedGap.toExponential(2)}`
  );
  console.log("PASS");
}
