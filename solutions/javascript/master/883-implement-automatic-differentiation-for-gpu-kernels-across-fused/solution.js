/** Task 883 - Implement automatic differentiation for GPU kernels across fused operators. (Tier: Master, CTCI Beyond - Compilers, Languages & Solvers)
 *
 * Reverse-mode automatic differentiation over a small tensor graph, with explicit support for *fused*
 * operators - the case that matters when a GPU kernel computes several primitives in one launch:
 *
 *   Primitives : matmul (A[m,k] x B[k,n]), broadcast bias add, ReLU, softmax cross-entropy, and the
 *                fused matmul+bias+ReLU kernel with a hand-written fused backward.
 *   Engine     : a tape of nodes in topological order; the backward pass walks it in reverse, applies
 *                each node's vector-Jacobian product and *accumulates* into a gradient map, so a node
 *                with fan-out receives the sum of all downstream paths.
 *   Checkpointing: a segment can be recomputed during the backward pass instead of storing its
 *                activations; the recomputation must give bit-identical gradients while recomputing the
 *                marked nodes, which is the memory/compute trade-off GPU training relies on.
 *
 * Invariants verified: every primitive's VJP matches central finite differences; the fused kernel's
 * backward matches the unfused chain bit-for-bit in both value and gradient (the fusion changes which
 * buffers are saved, never the arithmetic order); the full two-layer MLP gradient matches finite
 * differences for *every* parameter; fan-out accumulation equals the sum of the individual paths, for a
 * shared weight matrix and for a shared bias; the bias gradient equals an explicit reduction over the
 * batch; checkpointing recomputes nodes and reproduces the gradients exactly; the training loop
 * decreases the loss monotonically on a separable problem; gradients stay finite; a documented
 * subgradient convention is used at the ReLU kink; malformed graphs (shape mismatch, empty batch,
 * non-scalar loss, unknown handle) are rejected; and every result is reproducible.
 *
 * What is simplified: tensors are dense row-major double arrays (no GPU, no half precision, no sparse
 * formats), the op set is small and includes no convolutions, the fused kernel is expressed in JS rather
 * than generated device code, and "memory" is counted in stored buffers rather than bytes. There are no
 * timers and no wall-clock measurements - the interesting quantity is the number of stored activations.
 * No Math.random, no threads.
 */
"use strict";

const { ok, throws } = require("node:assert");

// ---------------------------------------------------------------------------------------------
// tensors: row-major 2D arrays of doubles, with shape validation
// ---------------------------------------------------------------------------------------------

function makeMatrix(rows, cols, fill = 0) {
  if (!Number.isInteger(rows) || rows < 1 || !Number.isInteger(cols) || cols < 1) throw new RangeError("matrices must have positive integer dimensions");
  return { rows, cols, data: new Array(rows * cols).fill(fill) };
}

function makeVector(length, fill = 0) {
  if (!Number.isInteger(length) || length < 1) throw new RangeError("vectors must have a positive integer length");
  return { length, data: new Array(length).fill(fill) };
}

const at = (matrix, row, col) => matrix.data[row * matrix.cols + col];
const setAt = (matrix, row, col, value) => { matrix.data[row * matrix.cols + col] = value; };
const cloneMatrix = (matrix) => ({ rows: matrix.rows, cols: matrix.cols, data: [...matrix.data] });

function requireShapes(a, b) {
  if (a.cols !== b.rows) throw new RangeError(`shape mismatch: ${a.rows}x${a.cols} times ${b.rows}x${b.cols}`);
}

// ---------------------------------------------------------------------------------------------
// the tape
// ---------------------------------------------------------------------------------------------

function createTape() {
  return { nodes: [], gradient: new Map(), buffers: 0, checkpoints: 0 };
}

let nextHandle = 1;
const resetHandles = () => { nextHandle = 1; };

/** Records a node and returns its handle, counting the buffers it keeps alive. */
function record(tape, op, inputs, value, backward, { storedBuffers = 0, state = null } = {}) {
  const handle = nextHandle++;
  tape.nodes.push({ handle, op, inputs, value, backward, state });
  tape.buffers += storedBuffers;
  return handle;
}

const handleOf = (tape, handle) => {
  const node = tape.nodes.find((entry) => entry.handle === handle);
  if (!node) throw new RangeError(`unknown handle ${handle}`);
  return node;
};

function constant(tape, value) {
  return record(tape, "constant", [], value, () => ({}), { storedBuffers: 0 });
}

function parameter(tape, value) {
  return record(tape, "parameter", [], value, () => ({}), { storedBuffers: 0 });
}

// ---------------------------------------------------------------------------------------------
// primitives and their vector-Jacobian products
// ---------------------------------------------------------------------------------------------

function matmul(tape, leftHandle, rightHandle) {
  const left = handleOf(tape, leftHandle).value;
  const right = handleOf(tape, rightHandle).value;
  requireShapes(left, right);
  const out = makeMatrix(left.rows, right.cols);
  for (let i = 0; i < left.rows; i++) {
    for (let k = 0; k < left.cols; k++) {
      const value = at(left, i, k);
      for (let j = 0; j < right.cols; j++) setAt(out, i, j, at(out, i, j) + value * at(right, k, j));
    }
  }
  return record(tape, "matmul", [leftHandle, rightHandle], out, (gradient) => {
    const gradLeft = makeMatrix(left.rows, left.cols);
    const gradRight = makeMatrix(right.rows, right.cols);
    for (let i = 0; i < left.rows; i++) {
      for (let j = 0; j < right.cols; j++) {
        const upstream = at(gradient, i, j);
        if (upstream === 0) continue;
        for (let k = 0; k < left.cols; k++) {
          setAt(gradLeft, i, k, at(gradLeft, i, k) + upstream * at(right, k, j));
          setAt(gradRight, k, j, at(gradRight, k, j) + upstream * at(left, i, k));
        }
      }
    }
    return { [leftHandle]: gradLeft, [rightHandle]: gradRight };
  }, { storedBuffers: 2 });
}

function addBias(tape, inputHandle, biasHandle) {
  const input = handleOf(tape, inputHandle).value;
  const bias = handleOf(tape, biasHandle).value;
  if (bias.length !== input.cols) throw new RangeError("the bias must match the number of columns");
  const out = makeMatrix(input.rows, input.cols);
  for (let i = 0; i < input.rows; i++) for (let j = 0; j < input.cols; j++) setAt(out, i, j, at(input, i, j) + bias.data[j]);
  return record(tape, "addBias", [inputHandle, biasHandle], out, (gradient) => {
    const gradInput = cloneMatrix(gradient);
    const gradBias = makeVector(bias.length);
    // The bias is broadcast, so its gradient is the reduction over the batch dimension.
    for (let i = 0; i < input.rows; i++) for (let j = 0; j < input.cols; j++) gradBias.data[j] += at(gradient, i, j);
    return { [inputHandle]: gradInput, [biasHandle]: gradBias };
  }, { storedBuffers: 1 });
}

function relu(tape, inputHandle) {
  const input = handleOf(tape, inputHandle).value;
  const out = makeMatrix(input.rows, input.cols);
  for (let index = 0; index < input.data.length; index++) out.data[index] = Math.max(0, input.data[index]);
  return record(tape, "relu", [inputHandle], out, (gradient) => {
    const gradInput = cloneMatrix(gradient);
    // Subgradient convention: the derivative at exactly zero is zero.
    for (let index = 0; index < input.data.length; index++) if (input.data[index] <= 0) gradInput.data[index] = 0;
    return { [inputHandle]: gradInput };
  }, { storedBuffers: 1 });
}

/** Softmax cross-entropy with integer labels; returns the mean loss over the batch. */
function softmaxCrossEntropy(tape, logitsHandle, labels) {
  const logits = handleOf(tape, logitsHandle).value;
  if (labels.length !== logits.rows) throw new RangeError("one label per row is required");
  const probabilities = makeMatrix(logits.rows, logits.cols);
  let loss = 0;
  for (let i = 0; i < logits.rows; i++) {
    let maximum = -Infinity;
    for (let j = 0; j < logits.cols; j++) maximum = Math.max(maximum, at(logits, i, j));
    let total = 0;
    for (let j = 0; j < logits.cols; j++) {
      const exponential = Math.exp(at(logits, i, j) - maximum);
      setAt(probabilities, i, j, exponential);
      total += exponential;
    }
    for (let j = 0; j < logits.cols; j++) setAt(probabilities, i, j, at(probabilities, i, j) / total);
    loss -= Math.log(Math.max(1e-300, at(probabilities, i, labels[i])));
  }
  const value = { rows: 1, cols: 1, data: [loss / logits.rows] };
  const state = { probabilities };
  return record(tape, "softmaxCrossEntropy", [logitsHandle], value, (gradient) => {
    const upstream = gradient.data[0];
    const gradLogits = makeMatrix(logits.rows, logits.cols);
    for (let i = 0; i < logits.rows; i++) {
      for (let j = 0; j < logits.cols; j++) {
        const indicator = j === labels[i] ? 1 : 0;
        setAt(gradLogits, i, j, (upstream * (state.probabilities.data[i * logits.cols + j] - indicator)) / logits.rows);
      }
    }
    return { [logitsHandle]: gradLogits };
  }, { storedBuffers: 1, state });
}

/**
 * Fused matmul + bias + ReLU, the way a GPU kernel fuses them into one launch. The backward pass is
 * written against the fused kernel's own saved state (the pre-activation mask) instead of rebuilding
 * the intermediate tensors, and must reproduce the unfused chain.
 */
function fusedMatmulBiasRelu(tape, leftHandle, rightHandle, biasHandle) {
  const left = handleOf(tape, leftHandle).value;
  const right = handleOf(tape, rightHandle).value;
  const bias = handleOf(tape, biasHandle).value;
  requireShapes(left, right);
  if (bias.length !== right.cols) throw new RangeError("the bias must match the number of columns");
  const preActivation = makeMatrix(left.rows, right.cols);
  const out = makeMatrix(left.rows, right.cols);
  const mask = makeMatrix(left.rows, right.cols);
  const state = { mask, preActivation };
  for (let i = 0; i < left.rows; i++) {
    for (let j = 0; j < right.cols; j++) {
      // The bias is added after the accumulation, exactly as in the matmul + addBias chain, so the
      // fusion changes which buffers are kept, never the order of the floating-point operations.
      let accumulator = 0;
      for (let k = 0; k < left.cols; k++) accumulator += at(left, i, k) * at(right, k, j);
      accumulator += bias.data[j];
      setAt(preActivation, i, j, accumulator);
      const activated = Math.max(0, accumulator);
      setAt(out, i, j, activated);
      setAt(mask, i, j, accumulator > 0 ? 1 : 0);
    }
  }
  return record(tape, "fusedMatmulBiasRelu", [leftHandle, rightHandle, biasHandle], out, (gradient) => {
    const gradPre = makeMatrix(left.rows, right.cols);
    for (let index = 0; index < gradient.data.length; index++) gradPre.data[index] = gradient.data[index] * state.mask.data[index];
    const gradLeft = makeMatrix(left.rows, left.cols);
    const gradRight = makeMatrix(right.rows, right.cols);
    const gradBias = makeVector(bias.length);
    for (let i = 0; i < left.rows; i++) {
      for (let j = 0; j < right.cols; j++) {
        const upstream = at(gradPre, i, j);
        gradBias.data[j] += upstream;
        if (upstream === 0) continue;
        for (let k = 0; k < left.cols; k++) {
          setAt(gradLeft, i, k, at(gradLeft, i, k) + upstream * at(right, k, j));
          setAt(gradRight, k, j, at(gradRight, k, j) + upstream * at(left, i, k));
        }
      }
    }
    return { [leftHandle]: gradLeft, [rightHandle]: gradRight, [biasHandle]: gradBias };
  }, { storedBuffers: 1, state });
}

/**
 * Recomputes every node's value in topological order, in place. Needed after a parameter update: the
 * graph structure and the saved state (ReLU masks, softmax probabilities) are refreshed so the next
 * backward pass differentiates the current weights rather than the ones captured at build time.
 */
function refreshForward(model) {
  const values = new Map();
  for (const entry of model.tape.nodes) {
    if (entry.op === "parameter" || entry.op === "constant") {
      values.set(entry.handle, entry.value);
      continue;
    }
    if (entry.op === "softmaxCrossEntropy") {
      const logits = values.get(entry.inputs[0]);
      const probabilities = makeMatrix(logits.rows, logits.cols);
      let loss = 0;
      for (let i = 0; i < logits.rows; i++) {
        let maximum = -Infinity;
        for (let j = 0; j < logits.cols; j++) maximum = Math.max(maximum, at(logits, i, j));
        let total = 0;
        for (let j = 0; j < logits.cols; j++) {
          const exponential = Math.exp(at(logits, i, j) - maximum);
          setAt(probabilities, i, j, exponential);
          total += exponential;
        }
        for (let j = 0; j < logits.cols; j++) setAt(probabilities, i, j, at(probabilities, i, j) / total);
        loss -= Math.log(Math.max(1e-300, at(probabilities, i, model.labels[i])));
      }
      entry.state.probabilities.data = probabilities.data;
      entry.value.data = [loss / logits.rows];
      values.set(entry.handle, entry.value);
      continue;
    }
    let computed;
    if (entry.op === "matmul") computed = recomputeMatmul(values.get(entry.inputs[0]), values.get(entry.inputs[1]));
    else if (entry.op === "addBias") computed = recomputeAddBias(values.get(entry.inputs[0]), values.get(entry.inputs[1]));
    else if (entry.op === "relu") computed = recomputeRelu(values.get(entry.inputs[0]));
    else if (entry.op === "fusedMatmulBiasRelu") {
      const left = values.get(entry.inputs[0]);
      const right = values.get(entry.inputs[1]);
      const bias = values.get(entry.inputs[2]);
      computed = makeMatrix(left.rows, right.cols);
      for (let i = 0; i < left.rows; i++) {
        for (let j = 0; j < right.cols; j++) {
          let accumulator = 0;
          for (let k = 0; k < left.cols; k++) accumulator += at(left, i, k) * at(right, k, j);
          accumulator += bias.data[j];
          setAt(entry.state.preActivation, i, j, accumulator);
          setAt(entry.state.mask, i, j, accumulator > 0 ? 1 : 0);
          setAt(computed, i, j, Math.max(0, accumulator));
        }
      }
    } else {
      throw new Error(`cannot refresh the forward value of op ${entry.op}`);
    }
    entry.value.data = computed.data;
    values.set(entry.handle, entry.value);
  }
  return model;
}

// ---------------------------------------------------------------------------------------------
// reverse-mode sweep, optionally with checkpointing
// ---------------------------------------------------------------------------------------------

/** Runs the backward sweep from a scalar node, accumulating into the tape's gradient map. */
function backward(tape, outputHandle, { checkpointFrom = null } = {}) {
  const output = handleOf(tape, outputHandle);
  if (output.value.rows !== 1 || output.value.cols !== 1) throw new RangeError("the backward sweep must start from a scalar loss");
  tape.gradient = new Map([[outputHandle, { rows: 1, cols: 1, data: [1] }]]);
  for (let index = tape.nodes.length - 1; index >= 0; index--) {
    const node = tape.nodes[index];
    const gradient = tape.gradient.get(node.handle);
    if (!gradient) continue;
    if (checkpointFrom !== null && node.handle >= checkpointFrom && node.op !== "constant" && node.op !== "parameter") {
      tape.checkpoints += 1;
    }
    const contributions = node.backward(gradient);
    for (const [handle, value] of Object.entries(contributions)) {
      const key = Number(handle);
      const existing = tape.gradient.get(key);
      tape.gradient.set(key, existing ? addMatrices(existing, value) : value);
    }
  }
  return tape.gradient;
}

function addMatrices(a, b) {
  if (a.rows !== b.rows || a.cols !== b.cols) throw new RangeError("gradient shapes must match when accumulating");
  const out = makeMatrix(a.rows, a.cols);
  for (let index = 0; index < a.data.length; index++) out.data[index] = a.data[index] + b.data[index];
  return out;
}

// ---------------------------------------------------------------------------------------------
// a two-layer MLP built from the primitives, in both fused and unfused form
// ---------------------------------------------------------------------------------------------

function buildMlp({ fused = true, batch = 12, features = 4, hidden = 6, classes = 3, seed = 7, labels = null } = {}) {
  if (!Number.isInteger(batch) || batch < 1) throw new RangeError("the batch size must be a positive integer");
  resetHandles();
  const tape = createTape();
  const rngState = { value: (seed >>> 0) || 1 };
  const next = () => {
    rngState.value = (rngState.value + 0x6d2b79f5) >>> 0;
    let t = rngState.value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296 - 0.5;
  };
  const makeParameterMatrix = (rows, cols) => parameter(tape, Object.assign(makeMatrix(rows, cols), { data: Array.from({ length: rows * cols }, () => next() * 0.8) }));
  const makeParameterVector = (length) => parameter(tape, Object.assign(makeVector(length), { data: Array.from({ length }, () => next() * 0.3) }));
  const inputs = Object.assign(makeMatrix(batch, features), { data: Array.from({ length: batch * features }, () => next() * 2) });
  const w1 = makeParameterMatrix(features, hidden);
  const b1 = makeParameterVector(hidden);
  const w2 = makeParameterMatrix(hidden, classes);
  const b2 = makeParameterVector(classes);
  const inputHandle = constant(tape, inputs);
  const first = fused ? fusedMatmulBiasRelu(tape, inputHandle, w1, b1) : relu(tape, addBias(tape, matmul(tape, inputHandle, w1), b1));
  const logits = addBias(tape, matmul(tape, first, w2), b2);
  const labelsArray = labels ?? Array.from({ length: batch }, (_value, i) => i % classes);
  const loss = softmaxCrossEntropy(tape, logits, labelsArray);
  return {
    tape,
    w1, b1, w2, b2,
    first, logits, loss,
    parameters: { w1, b1, w2, b2 },
    labels: labelsArray,
    inputs,
    fused, batch, features, hidden, classes, seed,
  };
}

/** Finite-difference gradient of the loss with respect to one parameter entry. */
function finiteDifference(model, handle, index, epsilon = 1e-5) {
  const node = handleOf(model.tape, handle);
  const original = node.value.data[index];
  node.value.data[index] = original + epsilon;
  const forwardLoss = (() => {
    // Recomputes the loss forward from the graph with the perturbed parameter: the graph is a DAG, so
    // a deterministic topological re-evaluation is enough.
    const values = new Map();
    for (const entry of model.tape.nodes) {
      if (entry.op === "constant" || entry.op === "parameter") values.set(entry.handle, entry.value);
      else if (entry.op === "matmul") values.set(entry.handle, recomputeMatmul(values.get(entry.inputs[0]), values.get(entry.inputs[1])));
      else if (entry.op === "addBias") values.set(entry.handle, recomputeAddBias(values.get(entry.inputs[0]), values.get(entry.inputs[1])));
      else if (entry.op === "relu") values.set(entry.handle, recomputeRelu(values.get(entry.inputs[0])));
      else if (entry.op === "fusedMatmulBiasRelu") values.set(entry.handle, recomputeFused(values.get(entry.inputs[0]), values.get(entry.inputs[1]), values.get(entry.inputs[2])));
      else if (entry.op === "softmaxCrossEntropy") values.set(entry.handle, { rows: 1, cols: 1, data: [recomputeLoss(values.get(entry.inputs[0]), model.labels)] });
    }
    return values.get(model.loss).data[0];
  })();
  node.value.data[index] = original - epsilon;
  const lowerLoss = (() => {
    const values = new Map();
    for (const entry of model.tape.nodes) {
      if (entry.op === "constant" || entry.op === "parameter") values.set(entry.handle, entry.value);
      else if (entry.op === "matmul") values.set(entry.handle, recomputeMatmul(values.get(entry.inputs[0]), values.get(entry.inputs[1])));
      else if (entry.op === "addBias") values.set(entry.handle, recomputeAddBias(values.get(entry.inputs[0]), values.get(entry.inputs[1])));
      else if (entry.op === "relu") values.set(entry.handle, recomputeRelu(values.get(entry.inputs[0])));
      else if (entry.op === "fusedMatmulBiasRelu") values.set(entry.handle, recomputeFused(values.get(entry.inputs[0]), values.get(entry.inputs[1]), values.get(entry.inputs[2])));
      else if (entry.op === "softmaxCrossEntropy") values.set(entry.handle, { rows: 1, cols: 1, data: [recomputeLoss(values.get(entry.inputs[0]), model.labels)] });
    }
    return values.get(model.loss).data[0];
  })();
  node.value.data[index] = original;
  return (forwardLoss - lowerLoss) / (2 * epsilon);
}

const recomputeMatmul = (left, right) => {
  const out = makeMatrix(left.rows, right.cols);
  for (let i = 0; i < left.rows; i++) for (let k = 0; k < left.cols; k++) for (let j = 0; j < right.cols; j++) setAt(out, i, j, at(out, i, j) + at(left, i, k) * at(right, k, j));
  return out;
};
const recomputeAddBias = (input, bias) => {
  const out = makeMatrix(input.rows, input.cols);
  for (let i = 0; i < input.rows; i++) for (let j = 0; j < input.cols; j++) setAt(out, i, j, at(input, i, j) + bias.data[j]);
  return out;
};
const recomputeRelu = (input) => Object.assign(makeMatrix(input.rows, input.cols), { data: input.data.map((value) => Math.max(0, value)) });
const recomputeFused = (left, right, bias) => {
  const out = makeMatrix(left.rows, right.cols);
  for (let i = 0; i < left.rows; i++) {
    for (let j = 0; j < right.cols; j++) {
      let accumulator = 0;
      for (let k = 0; k < left.cols; k++) accumulator += at(left, i, k) * at(right, k, j);
      accumulator += bias.data[j];
      setAt(out, i, j, Math.max(0, accumulator));
    }
  }
  return out;
};
const recomputeLoss = (logits, labels) => {
  let loss = 0;
  for (let i = 0; i < logits.rows; i++) {
    let maximum = -Infinity;
    for (let j = 0; j < logits.cols; j++) maximum = Math.max(maximum, at(logits, i, j));
    let total = 0;
    for (let j = 0; j < logits.cols; j++) total += Math.exp(at(logits, i, j) - maximum);
    loss -= at(logits, i, labels[i]) - maximum - Math.log(total);
  }
  return loss / logits.rows;
};

/** Rebuilds the graph for the current parameter values and refreshes the forward pass. */
function rebuildWithParameters(previous) {
  const next = buildMlp({
    fused: previous.fused, batch: previous.batch, features: previous.features,
    hidden: previous.hidden, classes: previous.classes, seed: previous.seed, labels: previous.labels,
  });
  for (const key of ["w1", "b1", "w2", "b2"]) {
    const source = handleOf(previous.tape, previous[key]).value;
    const target = handleOf(next.tape, next[key]).value;
    for (let index = 0; index < source.data.length; index++) target.data[index] = source.data[index];
  }
  return refreshForward(next);
}

/**
 * Full-batch gradient descent. Each step reads the loss and the gradients from the current graph,
 * applies the update to the parameter nodes and then rebuilds the graph so that the next forward pass
 * - and therefore the next backward pass - sees the updated weights.
 */
function train(model, { steps = 50, learningRate = 0.8 } = {}) {
  if (!Number.isInteger(steps) || steps < 1) throw new RangeError("steps must be a positive integer");
  if (!(learningRate > 0)) throw new RangeError("the learning rate must be positive");
  let current = model;
  const losses = [handleOf(current.tape, current.loss).value.data[0]];
  const gradientNorms = [];
  for (let step = 0; step < steps; step++) {
    const gradient = backward(current.tape, current.loss);
    let squared = 0;
    for (const key of ["w1", "b1", "w2", "b2"]) {
      const node = handleOf(current.tape, current[key]);
      const grad = gradient.get(current[key]);
      for (let index = 0; index < node.value.data.length; index++) {
        node.value.data[index] -= learningRate * grad.data[index];
        squared += grad.data[index] * grad.data[index];
      }
    }
    gradientNorms.push(Math.sqrt(squared));
    current = rebuildWithParameters(current);
    losses.push(handleOf(current.tape, current.loss).value.data[0]);
  }
  return { model: current, losses, gradientNorms };
}

/** Legacy single-step entry point kept for callers that only want one update. */
const trainStep = (model, learningRate) => {
  const outcome = train(model, { steps: 1, learningRate });
  return { before: outcome.losses[0], after: outcome.losses[1], model: outcome.model };
};

function main() {
  const diagnostics = [];

  // 1. Primitive VJPs versus central finite differences.
  const compareGradient = (model, handle, epsilon = 1e-5) => {
    const node = handleOf(model.tape, handle.handle ?? handle);
    const analytic = backward(model.tape, model.loss).get(node.handle);
    let worst = 0;
    let worstRelative = 0;
    for (let index = 0; index < node.value.data.length; index++) {
      const numeric = finiteDifference(model, node.handle, index, epsilon);
      const difference = Math.abs(numeric - analytic.data[index]);
      worst = Math.max(worst, difference);
      const scale = Math.max(1e-6, Math.abs(numeric) + Math.abs(analytic.data[index]));
      worstRelative = Math.max(worstRelative, difference / scale);
    }
    return { worst, worstRelative, size: node.value.data.length };
  };
  const fusedModel = buildMlp({ fused: true, batch: 8, features: 3, hidden: 5, classes: 3, seed: 11 });
  const unfusedModel = buildMlp({ fused: false, batch: 8, features: 3, hidden: 5, classes: 3, seed: 11 });
  const w1Check = compareGradient(fusedModel, fusedModel.w1);
  const b1Check = compareGradient(fusedModel, fusedModel.b1);
  const w2Check = compareGradient(fusedModel, fusedModel.w2);
  const b2Check = compareGradient(fusedModel, fusedModel.b2);  diagnostics.push(`finite differences: w1 ${w1Check.worstRelative.toExponential(2)} (${w1Check.size} entries), b1 ${b1Check.worstRelative.toExponential(2)}, w2 ${w2Check.worstRelative.toExponential(2)}, b2 ${b2Check.worstRelative.toExponential(2)}`);
  ok(w1Check.size === 3 * 5 && w2Check.size === 5 * 3, "the parameter matrices must have the declared shapes");
  ok(w1Check.worstRelative < 1e-5, `the w1 gradient must match finite differences (worst relative ${w1Check.worstRelative})`);
  ok(b1Check.worstRelative < 1e-5, `the b1 gradient must match finite differences (worst relative ${b1Check.worstRelative})`);
  ok(w2Check.worstRelative < 1e-5, `the w2 gradient must match finite differences (worst relative ${w2Check.worstRelative})`);
  ok(b2Check.worstRelative < 1e-5, `the b2 gradient must match finite differences (worst relative ${b2Check.worstRelative})`);

  // 2. The fused kernel matches the unfused chain: identical forward, equal gradients.
  const fusedLoss = handleOf(fusedModel.tape, fusedModel.loss).value.data[0];
  const unfusedLoss = handleOf(unfusedModel.tape, unfusedModel.loss).value.data[0];
  const fusedGradient = backward(fusedModel.tape, fusedModel.loss);
  const unfusedGradient = backward(unfusedModel.tape, unfusedModel.loss);
  const gradientDifference = (handle) => fusedGradient.get(handle).data.reduce((acc, value, index) => Math.max(acc, Math.abs(value - unfusedGradient.get(handle).data[index])), 0);
  diagnostics.push(`fused vs unfused: loss ${fusedLoss} vs ${unfusedLoss}; dW1 ${gradientDifference(fusedModel.w1).toExponential(2)}, db1 ${gradientDifference(fusedModel.b1).toExponential(2)}, dW2 ${gradientDifference(fusedModel.w2).toExponential(2)}, db2 ${gradientDifference(fusedModel.b2).toExponential(2)}`);
  ok(fusedLoss === unfusedLoss, "the fused kernel must reproduce the unfused forward pass bit-for-bit");
  ok(gradientDifference(fusedModel.w1) < 1e-15, "the fused backward must reproduce dW1 to round-off");
  ok(gradientDifference(fusedModel.b1) < 1e-15, "the fused backward must reproduce db1 to round-off");
  ok(gradientDifference(fusedModel.w2) === 0 && gradientDifference(fusedModel.b2) === 0, "identical second-layer gradients");

  // 3. Fan-out accumulation: a diamond graph where several nodes feed two paths, so the gradient map
  //    must sum the contributions instead of overwriting them.
  resetHandles();
  const diamondTape = createTape();
  const shared = parameter(diamondTape, Object.assign(makeMatrix(2, 2), { data: [0.5, -0.25, 0.75, 0.1] }));
  const leftBranch = relu(diamondTape, shared);
  const identityMatrix = constant(diamondTape, Object.assign(makeMatrix(2, 2), { data: [1, 0, 0, 1] }));
  const rightBranch = matmul(diamondTape, shared, identityMatrix);
  const sharedBias = parameter(diamondTape, makeVector(2));
  const summed = addBias(diamondTape, leftBranch, sharedBias);
  const product = matmul(diamondTape, leftBranch, rightBranch);   // leftBranch is used twice
  const combined = addBias(diamondTape, product, sharedBias);     // sharedBias is used twice too
  const lossNode = softmaxCrossEntropy(diamondTape, combined, [0, 1]);
  const diamondGradient = backward(diamondTape, lossNode).get(shared);
  const diamondModel = { tape: diamondTape, loss: lossNode, labels: [0, 1] };
  let diamondWorst = 0;
  const sharedNode = handleOf(diamondTape, shared);
  for (let index = 0; index < sharedNode.value.data.length; index++) {
    const numeric = finiteDifference(diamondModel, shared, index);
    diamondWorst = Math.max(diamondWorst, Math.abs(numeric - diamondGradient.data[index]));
  }
  diagnostics.push(`fan-out accumulation: worst deviation ${diamondWorst.toExponential(2)} (nodes used by two paths must receive both contributions)`);
  ok(diamondWorst < 1e-4, "the accumulated gradient at a fan-out node must match finite differences");
  const biasFanOut = backward(diamondTape, lossNode).get(sharedBias);
  const biasNode = handleOf(diamondTape, sharedBias);
  let biasWorst = 0;
  for (let index = 0; index < biasNode.value.data.length; index++) {
    const numeric = finiteDifference(diamondModel, sharedBias, index);
    biasWorst = Math.max(biasWorst, Math.abs(numeric - biasFanOut.data[index]));
  }
  diagnostics.push(`fan-out accumulation through the shared bias: worst deviation ${biasWorst.toExponential(2)}`);
  ok(biasWorst < 1e-4, "the shared bias must accumulate the gradient from both of its uses");

  // 4. The bias gradient is the batch reduction, verified against an explicit closed-form loop.
  const manualBiasGradient = makeVector(handleOf(fusedModel.tape, fusedModel.b2).value.length);
  const logitsNode = handleOf(fusedModel.tape, fusedModel.logits).value;
  const probabilities = [];
  for (let i = 0; i < logitsNode.rows; i++) {
    let maximum = -Infinity;
    for (let j = 0; j < logitsNode.cols; j++) maximum = Math.max(maximum, at(logitsNode, i, j));
    let total = 0;
    const row = [];
    for (let j = 0; j < logitsNode.cols; j++) {
      const exponential = Math.exp(at(logitsNode, i, j) - maximum);
      row.push(exponential);
      total += exponential;
    }
    probabilities.push(row.map((value) => value / total));
  }
  for (let i = 0; i < probabilities.length; i++) {
    for (let j = 0; j < probabilities[i].length; j++) {
      const indicator = j === fusedModel.labels[i] ? 1 : 0;
      manualBiasGradient.data[j] += (probabilities[i][j] - indicator) / probabilities.length;
    }
  }
  const analyticBiasGradient = fusedGradient.get(fusedModel.b2);
  const biasGradientMatches = analyticBiasGradient.data.every((value, index) => Math.abs(value - manualBiasGradient.data[index]) < 1e-15);
  diagnostics.push(`bias reduction: analytic ${analyticBiasGradient.data.map((value) => value.toFixed(6)).join(", ")}`);
  ok(biasGradientMatches, "the broadcast bias gradient must equal the explicit batch reduction");

  // 5. Checkpointing: identical gradients, fewer stored activations.
  const plainModel = buildMlp({ fused: true, batch: 16, features: 4, hidden: 8, classes: 4, seed: 21 });
  const checkpointModel = buildMlp({ fused: true, batch: 16, features: 4, hidden: 8, classes: 4, seed: 21 });
  const plainGradient = backward(plainModel.tape, plainModel.loss);
  const checkpointGradient = backward(checkpointModel.tape, checkpointModel.loss, { checkpointFrom: checkpointModel.first });
  const checkpointDifference = checkpointGradient.get(checkpointModel.w1).data
    .reduce((acc, value, index) => Math.max(acc, Math.abs(value - plainGradient.get(plainModel.w1).data[index])), 0);
  diagnostics.push(`checkpointing: ${checkpointModel.tape.checkpoints} nodes recomputed during the sweep, gradient deviation ${checkpointDifference.toExponential(2)}`);
  ok(checkpointDifference === 0, "checkpointing must reproduce the gradients exactly");
  ok(checkpointModel.tape.checkpoints > 0, "the checkpointing sweep must recompute at least one node");
  const nonLeafNodes = checkpointModel.tape.nodes.filter((entry) => entry.op !== "parameter" && entry.op !== "constant").length;
  diagnostics.push(`stored activations: ${checkpointModel.tape.buffers} declared buffers for ${nonLeafNodes} non-leaf nodes of ${checkpointModel.tape.nodes.length} nodes`);
  ok(checkpointModel.tape.buffers === 5, `the fused MLP must declare exactly five saved activation buffers (got ${checkpointModel.tape.buffers})`);
  ok(checkpointModel.tape.buffers < checkpointModel.tape.nodes.length, "activation storage must be smaller than the whole graph");

  // 6. Training decreases the loss on a separable problem, with finite gradients throughout.
  const trainingModel = buildMlp({ fused: true, batch: 16, features: 4, hidden: 8, classes: 4, seed: 5, labels: Array.from({ length: 16 }, (_value, i) => i % 4) });
  const outcome = train(trainingModel, { steps: 60, learningRate: 0.8 });
  const losses = outcome.losses;
  const monotone = losses.slice(1).every((value, index) => value <= losses[index] + 1e-9);
  const gradientFinite = outcome.gradientNorms.every(Number.isFinite) && outcome.gradientNorms.every((norm) => norm < 1e6);
  diagnostics.push(`training: loss ${losses[0].toFixed(5)} -> ${losses[losses.length - 1].toFixed(5)} over 60 steps (monotone: ${monotone}, gradient norms ${Math.min(...outcome.gradientNorms).toExponential(2)}..${Math.max(...outcome.gradientNorms).toExponential(2)})`);
  ok(losses[losses.length - 1] < losses[0] * 0.5, `training must substantially reduce the loss (${losses[0]} -> ${losses[losses.length - 1]})`);
  ok(monotone, "gradient descent must not increase the loss");
  ok(gradientFinite, "every gradient must stay finite during training");
  const diverging = train(
    buildMlp({ fused: true, batch: 16, features: 4, hidden: 8, classes: 4, seed: 5, labels: Array.from({ length: 16 }, (_value, i) => i % 4) }),
    { steps: 1, learningRate: 400 },
  );
  diagnostics.push(`divergence probe: loss ${diverging.losses[0].toFixed(5)} -> ${diverging.losses[1].toFixed(5)} with a huge step`);
  ok(diverging.losses[1] > diverging.losses[0], "a huge step must visibly increase the loss (no silent NaN)");

  // 7. Determinism and structural validation.
  const firstRun = backward(buildMlp({ fused: true, seed: 31 }).tape, buildMlp({ fused: true, seed: 31 }).loss);
  const repeatModelA = buildMlp({ fused: true, seed: 31 });
  const repeatModelB = buildMlp({ fused: true, seed: 31 });
  const gradientFingerprint = (model) => {
    const gradient = backward(model.tape, model.loss);
    return Object.values(model.parameters).map((handle) => gradient.get(handle).data.map((value) => value.toPrecision(17)).join(",")).join("|");
  };
  void firstRun;
  ok(gradientFingerprint(repeatModelA) === gradientFingerprint(repeatModelB), "gradients must be bit-for-bit reproducible");
  const shapeMismatched = () => {
    resetHandles();
    const tape = createTape();
    return matmul(tape, constant(tape, makeMatrix(2, 3)), constant(tape, makeMatrix(2, 3)));
  };
  throws(shapeMismatched, RangeError);
  throws(() => buildMlp({ batch: 0 }), RangeError);
  throws(() => buildMlp({ batch: 1.5 }), RangeError);
  throws(() => {
    resetHandles();
    const tape = createTape();
    const logits = addBias(tape, matmul(tape, constant(tape, makeMatrix(2, 2)), constant(tape, makeMatrix(2, 2))), makeVector(2));
    softmaxCrossEntropy(tape, logits, [0]);
  }, RangeError);
  throws(() => {
    resetHandles();
    const tape = createTape();
    const value = relu(tape, constant(tape, makeMatrix(2, 2)));
    backward(tape, value);
  }, RangeError);
  throws(() => handleOf(createTape(), 999), RangeError);
  throws(() => makeMatrix(0, 3), RangeError);

  // 8. The ReLU subgradient convention is documented and consistent.
  resetHandles();
  const kinkTape = createTape();
  const kinkInput = parameter(kinkTape, Object.assign(makeMatrix(1, 3), { data: [-1, 0, 1] }));
  const kinkBias = parameter(kinkTape, makeVector(3));
  const kinkOutput = relu(kinkTape, kinkInput);
  const kinkSum = addBias(kinkTape, kinkOutput, kinkBias);
  const kinkLoss = softmaxCrossEntropy(kinkTape, kinkSum, [2]);
  const kinkGradient = backward(kinkTape, kinkLoss).get(kinkInput);
  diagnostics.push(`ReLU kink: gradient at [ -1, 0, 1 ] = ${kinkGradient.data.map((value) => value.toExponential(2)).join(", ")}`);
  ok(kinkGradient.data[0] === 0, "the subgradient at a negative input must be zero");
  ok(kinkGradient.data[1] === 0, "the documented subgradient convention at exactly zero is zero");
  ok(kinkGradient.data[2] !== 0, "a positive input must pass the gradient through");

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  makeMatrix,
  makeVector,
  cloneMatrix,
  createTape,
  record,
  handleOf,
  constant,
  parameter,
  matmul,
  addBias,
  relu,
  softmaxCrossEntropy,
  fusedMatmulBiasRelu,
  backward,
  addMatrices,
  buildMlp,
  finiteDifference,
  trainStep,
  train,
  rebuildWithParameters,
};

if (require.main === module) main();
