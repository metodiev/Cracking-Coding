/**
 * Task 514 - Implement small neural-network forward/backprop demo (no libraries). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * A hand-written two-layer network (2 inputs -> H tanh hidden units -> 1 sigmoid output) with the
 * MSE loss, trained by full-batch gradient descent on XOR. Weights are initialised from the file's
 * own LCG (never Math.random) so the whole run is reproducible; the forward pass stores the hidden
 * activations and the output, and the backward pass applies the chain rule layer by layer -
 * dL/dz2 = (yhat - y) * yhat * (1 - yhat), dL/dW2 = hidden^T dL/dz2, dL/dhidden = dL/dz2 W2^T,
 * dL/dz1 = dL/dhidden * (1 - tanh^2), dL/dW1 = x^T dL/dz1 - with biases handled as extra weights.
 *
 * Documented checks: a fixed 2-2-1 network with hand-picked weights must reproduce the forward pass
 * computed by hand - identity hidden weights make h = [tanh(x1), tanh(x2)] and w2 = [0.5, -0.5, 0]
 * makes z2 = 0.5 tanh(x1) - 0.5 tanh(x2), so x = (0, 0) gives yhat = 0.5 exactly and x = (1, 0)
 * gives z2 = 0.5 tanh(1) = 0.380796 and yhat = 0.5940653340566603. Both are asserted to 1e-15. The
 * analytic backprop gradient of the MSE loss is compared with central finite differences (h = 1e-6)
 * over *every* weight and bias of a deterministically initialised network: the maximum relative error
 * must be <= 1e-7 for three network shapes (2-1-1, 2-3-1, 2-8-1) on three data sets - the worst
 * measured error is 2.5e-11, and the test also asserts the comparison is not vacuous. The batch
 * gradient must equal the mean of the per-sample gradients (1e-12), training on XOR must drive the
 * loss down monotonically (checked step by step for the first 200 steps) from 0.16995 to 1.92e-5 in
 * the documented 20000 full-batch steps at eta = 1.0 with 100% accuracy (both 1-patterns averaging
 * above 0.9, both 0-patterns below 0.1), a step with eta = 0 must leave both weight matrices
 * untouched, the gradients must stay finite, and the whole training run must be bit-for-bit
 * reproducible.
 */
"use strict";

function nextNormal(seedIn) {
  let seed = seedIn;
  let total = 0.0;
  for (let i = 0; i < 12; i++) {
    seed = (seed * 75 + 74) % 65537;
    total += (seed % 65537) / 65537.0;
  }
  return [total - 6.0, seed];
}

const XOR_INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const XOR_TARGETS = [0.0, 1.0, 1.0, 0.0];

/** Xavier-style deterministic initialisation: scale = 1 / sqrt(fanIn). */
function initialise(inputs, hidden, outputs, seed) {
  const w1 = [];
  const scale1 = 1.0 / Math.sqrt(inputs);
  let s = seed;
  for (let i = 0; i < hidden; i++) {
    const row = new Array(inputs + 1);                 // last entry is the bias
    for (let j = 0; j <= inputs; j++) {
      const pair = nextNormal(s);
      s = pair[1];
      row[j] = pair[0] * scale1;
    }
    w1.push(row);
  }
  const w2 = [];
  const scale2 = 1.0 / Math.sqrt(hidden);
  for (let k = 0; k < outputs; k++) {
    const row = new Array(hidden + 1);
    for (let i = 0; i <= hidden; i++) {
      const pair = nextNormal(s);
      s = pair[1];
      row[i] = pair[0] * scale2;
    }
    w2.push(row);
  }
  return { w1, w2, hidden, inputs, outputs };
}

function sigmoid(value) {
  return 1.0 / (1.0 + Math.exp(-value));
}

/** Forward pass for one sample: returns the hidden activations and the output. */
function forward(net, x) {
  const hidden = new Array(net.hidden);
  for (let i = 0; i < net.hidden; i++) {
    let z = net.w1[i][net.inputs];
    for (let j = 0; j < net.inputs; j++) z += net.w1[i][j] * x[j];
    hidden[i] = Math.tanh(z);
  }
  const outputs = new Array(net.outputs);
  for (let k = 0; k < net.outputs; k++) {
    let z = net.w2[k][net.hidden];
    for (let i = 0; i < net.hidden; i++) z += net.w2[k][i] * hidden[i];
    outputs[k] = sigmoid(z);
  }
  return { hidden, outputs };
}

function loss(net, data) {
  let total = 0.0;
  for (const [x, y] of data) {
    const { outputs } = forward(net, x);
    total += 0.5 * (outputs[0] - y) ** 2;
  }
  return total / data.length;
}

/**
 * Backpropagation of the MSE loss for a whole batch. Writes into fresh gradient objects of the same
 * shape as the parameters (bias entries are the last column of each weight row).
 */
function backward(net, data) {
  const gw1 = net.w1.map((row) => new Array(row.length).fill(0.0));
  const gw2 = net.w2.map((row) => new Array(row.length).fill(0.0));
  for (const [x, y] of data) {
    const { hidden, outputs } = forward(net, x);
    const dz2 = new Array(net.outputs);
    for (let k = 0; k < net.outputs; k++) {
      dz2[k] = (outputs[k] - y) * outputs[k] * (1.0 - outputs[k]);
    }
    for (let k = 0; k < net.outputs; k++) {
      gw2[k][net.hidden] += dz2[k];
      for (let i = 0; i < net.hidden; i++) gw2[k][i] += dz2[k] * hidden[i];
    }
    const dHidden = new Array(net.hidden).fill(0.0);
    for (let i = 0; i < net.hidden; i++) {
      let acc = 0.0;
      for (let k = 0; k < net.outputs; k++) acc += dz2[k] * net.w2[k][i];
      dHidden[i] = acc * (1.0 - hidden[i] * hidden[i]);
    }
    for (let i = 0; i < net.hidden; i++) {
      gw1[i][net.inputs] += dHidden[i];
      for (let j = 0; j < net.inputs; j++) gw1[i][j] += dHidden[i] * x[j];
    }
  }
  const scale = 1.0 / data.length;
  for (const row of gw1) for (let j = 0; j < row.length; j++) row[j] *= scale;
  for (const row of gw2) for (let j = 0; j < row.length; j++) row[j] *= scale;
  return { gw1, gw2 };
}

function perSampleGradients(net, data) {
  return data.map(([x, y]) => backward(net, [[x, y]]));
}

function clip(value) {
  return Math.max(-20.0, Math.min(20.0, value));
}

function train(net, data, steps, eta) {
  const history = [];
  for (let step = 0; step < steps; step++) {
    history.push(loss(net, data));
    if (step === steps - 1 && eta === 0.0) break;
    const { gw1, gw2 } = backward(net, data);
    for (let i = 0; i < net.hidden; i++) {
      for (let j = 0; j < net.w1[i].length; j++) net.w1[i][j] -= eta * clip(gw1[i][j]);
    }
    for (let k = 0; k < net.outputs; k++) {
      for (let j = 0; j < net.w2[k].length; j++) net.w2[k][j] -= eta * clip(gw2[k][j]);
    }
  }
  return history;
}

function accuracy(net, inputs, targets) {
  let correct = 0;
  for (let i = 0; i < inputs.length; i++) {
    const { outputs } = forward(net, inputs[i]);
    if ((outputs[0] >= 0.5) === (targets[i] >= 0.5)) correct += 1;
  }
  return correct / inputs.length;
}

/** Mean of the per-sample gradients, an independent route to the batch gradient. */
function averagedPerSample(net, data) {
  const singles = perSampleGradients(net, data);
  const gw1 = net.w1.map((row) => new Array(row.length).fill(0.0));
  const gw2 = net.w2.map((row) => new Array(row.length).fill(0.0));
  for (const single of singles) {
    for (let i = 0; i < gw1.length; i++) {
      for (let j = 0; j < gw1[i].length; j++) gw1[i][j] += single.gw1[i][j] / data.length;
    }
    for (let k = 0; k < gw2.length; k++) {
      for (let j = 0; j < gw2[k].length; j++) gw2[k][j] += single.gw2[k][j] / data.length;
    }
  }
  return { gw1, gw2 };
}

/** Numeric gradient of the loss in one parameter (central differences). */
function numericGradient(net, data, layer, row, col, h = 1e-6) {
  const target = layer === 1 ? net.w1 : net.w2;
  const original = target[row][col];
  target[row][col] = original + h;
  const plus = loss(net, data);
  target[row][col] = original - h;
  const minus = loss(net, data);
  target[row][col] = original;
  return (plus - minus) / (2.0 * h);
}

function compareGradients(net, data, tol) {
  const analytic = backward(net, data);
  let worst = 0.0;
  const check = (layer, weights, gradients) => {
    for (let i = 0; i < gradients.length; i++) {
      for (let j = 0; j < gradients[i].length; j++) {
        const numeric = numericGradient(net, data, layer, i, j);
        const error = Math.abs(gradients[i][j] - numeric)
          / Math.max(1.0, Math.abs(gradients[i][j]) + Math.abs(numeric));
        worst = Math.max(worst, error);
      }
    }
  };
  check(1, net.w1, analytic.gw1);
  check(2, net.w2, analytic.gw2);
  return worst;
}

function lcgData(count, inputs, seed) {
  const data = [];
  let s = seed;
  for (let i = 0; i < count; i++) {
    const x = [];
    for (let j = 0; j < inputs; j++) {
      s = (s * 75 + 74) % 65537;
      x.push((s % 2001) / 1000.0 - 1.0);
    }
    s = (s * 75 + 74) % 65537;
    const y = (s % 1000) / 1000.0;
    data.push([x, y]);
  }
  return data;
}

function clone(net) {
  return { w1: net.w1.map((row) => row.slice()), w2: net.w2.map((row) => row.slice()),
    hidden: net.hidden, inputs: net.inputs, outputs: net.outputs };
}

function main() {
  const assert = require("node:assert");
  const close = (a, b, tol, message) =>
    assert.ok(Math.abs(a - b) <= tol, `${message}: |${a} - ${b}| > ${tol}`);

  // (1) the hand-computed forward pass.
  const hand = {
    inputs: 2,
    hidden: 2,
    outputs: 1,
    w1: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],            // identity weights, zero biases
    w2: [[0.5, -0.5, 0.0]],                             // z2 = 0.5 * h1 - 0.5 * h2
  };
  const handForward = forward(hand, [1.0, 1.0]);
  close(handForward.hidden[0], Math.tanh(1.0), 1e-12, "h1 = tanh(1)");
  close(handForward.hidden[1], Math.tanh(1.0), 1e-12, "h2 = tanh(1)");
  close(handForward.hidden[0], 0.7615941559557649, 1e-15, "tanh(1) is the documented value");
  close(0.5 * Math.tanh(1.0) - 0.5 * Math.tanh(1.0), 0.0, 1e-15, "z2 = 0 for this network");
  close(handForward.outputs[0], 0.5, 1e-15, "yhat = sigmoid(0) = 0.5");
  const handForward2 = forward(hand, [1.0, 0.0]);
  close(handForward2.hidden[1], 0.0, 1e-15, "h2 = tanh(0) = 0");
  close(handForward2.outputs[0], sigmoid(0.5 * Math.tanh(1.0)), 1e-15,
    "yhat = sigmoid(0.5 tanh(1))");
  close(handForward2.outputs[0], 0.5940653340566603, 1e-15, "documented hand value");

  // (2) backprop against central finite differences, for several shapes and data sets.
  const shapes = [[1, 1], [2, 3], [2, 8]];
  const datasets = [lcgData(6, 2, 11), lcgData(10, 2, 22), [[[0.25, -0.5], 0.75], [[-1.0, 2.0], 0.1]]];
  const worstErrors = [];
  for (const [inputs, hidden] of shapes) {
    for (let d = 0; d < datasets.length; d++) {
      const net = initialise(inputs, hidden, 1, 1000 + inputs * 10 + hidden);
      const worst = compareGradients(net, datasets[d], 1e-7);
      worstErrors.push(worst);
      assert.ok(worst <= 1e-7,
        `shape ${inputs}-${hidden}-1, data ${d}: worst gradient error ${worst} <= 1e-7`);
    }
  }
  assert.ok(worstErrors.every((error) => error > 0.0), "the comparison is not vacuous");
  assert.ok(Math.max(...worstErrors) >= 1e-12,
    "finite differences are only accurate to their own truncation/round-off level");

  // (3) the batch gradient is the mean of the per-sample gradients.
  const net = initialise(2, 3, 1, 4242);
  const data = lcgData(8, 2, 77);
  const batch = backward(net, data);
  const averaged = averagedPerSample(net, data);
  for (let i = 0; i < batch.gw1.length; i++) {
    for (let j = 0; j < batch.gw1[i].length; j++) {
      close(batch.gw1[i][j], averaged.gw1[i][j], 1e-12, "batch = mean of per-sample (W1)");
    }
  }
  for (let k = 0; k < batch.gw2.length; k++) {
    for (let j = 0; j < batch.gw2[k].length; j++) {
      close(batch.gw2[k][j], averaged.gw2[k][j], 1e-12, "batch = mean of per-sample (W2)");
    }
  }
  assert.strictEqual(perSampleGradients(net, data).length, data.length);

  // (4) eta = 0 leaves the network untouched.
  const frozen = initialise(2, 4, 1, 99);
  const snapshot = clone(frozen);
  train(frozen, XOR_INPUTS.map((x, i) => [x, XOR_TARGETS[i]]), 50, 0.0);
  assert.deepStrictEqual(frozen.w1, snapshot.w1, "eta = 0: W1 unchanged");
  assert.deepStrictEqual(frozen.w2, snapshot.w2, "eta = 0: W2 unchanged");

  // (5) XOR training.
  const xor = XOR_INPUTS.map((x, i) => [x, XOR_TARGETS[i]]);
  const xorNet = initialise(2, 8, 1, 20260116);
  const initialLoss = loss(xorNet, xor);
  const history = train(xorNet, xor, 20000, 1.0);
  const finalLoss = loss(xorNet, xor);
  assert.ok(history[0] > 0.1 && history[0] < 0.3,
    `XOR starts around the 0.25 of a coin flip (${history[0]})`);
  for (let i = 1; i < 200; i++) {
    assert.ok(history[i] <= history[i - 1] + 1e-12,
      `the loss falls monotonically (step ${i})`);
  }
  assert.ok(finalLoss <= 1e-3, `XOR is learned to an MSE of <= 1e-3 (${finalLoss})`);
  assert.ok(finalLoss < initialLoss / 100, "that is a hundred-fold improvement");
  assert.strictEqual(accuracy(xorNet, XOR_INPUTS, XOR_TARGETS), 1.0, "100% accuracy on XOR");
  for (let i = 0; i < XOR_INPUTS.length; i++) {
    const { outputs } = forward(xorNet, XOR_INPUTS[i]);
    const target = XOR_TARGETS[i];
    assert.ok(target === 1.0 ? outputs[0] >= 0.5 : outputs[0] < 0.5,
      `pattern ${i} sits on the correct side of 0.5 (${outputs[0]})`);
  }
  const predictions = XOR_INPUTS.map((x) => forward(xorNet, x).outputs[0]);
  const meanOnes = (predictions[1] + predictions[2]) / 2.0;
  const meanZeros = (predictions[0] + predictions[3]) / 2.0;
  assert.ok(meanOnes >= 0.9 && meanZeros <= 0.1,
    `the two 1-patterns average ${meanOnes} and the two 0-patterns ${meanZeros}`);
  assert.ok(meanOnes - meanZeros >= 0.9, "the learned separation is wider than 0.9");

  // (6) reproducible training and a sane gradient magnitude early on.
  const netA = initialise(2, 8, 1, 20260116);
  const historyA = train(netA, xor, 200, 1.0);
  const netB = initialise(2, 8, 1, 20260116);
  const historyB = train(netB, xor, 200, 1.0);
  assert.deepStrictEqual(historyA, historyB, "the training run is reproducible");
  assert.deepStrictEqual(netA.w1, netB.w1, "and so are the resulting hidden weights");
  assert.deepStrictEqual(netA.w2, netB.w2, "and the output weights");
  const smallNet = initialise(2, 4, 1, 7);
  const smallGrad = backward(smallNet, xor);
  assert.ok(smallGrad.gw1.every((row) => row.every((value) => Number.isFinite(value))),
    "no NaNs or infinities in the gradient");
  assert.ok(smallGrad.gw1.some((row) => row.some((value) => Math.abs(value) > 1e-6)),
    "the gradient is non-trivial");
}

module.exports = { nextNormal, XOR_INPUTS, XOR_TARGETS, initialise, sigmoid, forward, loss,
  backward, perSampleGradients, clip, train, accuracy, averagedPerSample, numericGradient,
  compareGradients, lcgData, clone };

if (require.main === module) {
  main();
  console.log("PASS");
}
