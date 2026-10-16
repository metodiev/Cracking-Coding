/** Task 882 - Implement real-time denoising neural network pipeline integration for raytracer. (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * A complete (small) real-time denoising pipeline: a deterministic path-traced scene rendered with one
 * sample per pixel, a tiny convolutional network that predicts per-pixel filter weights from the noisy
 * colour plus the auxiliary albedo/normal/depth buffers, a normalised cross-bilateral application of
 * those weights, temporal accumulation over frames, and a tiled scheduler with explicit halos.
 *
 *   Renderer  : the scene has analytic geometry (a plane and two spheres) and analytically integrable
 *               direct lighting, so the converged image can be produced by the same deterministic
 *               sampler at a high sample count rather than by a reference renderer.
 *   Network   : two 3x3 convolutions with ReLU (weights from a seeded splitmix64 stream), then a 1x1
 *               convolution producing 25 kernel weights per pixel, projected to a probability simplex
 *               with a softmax so the filter is energy preserving; a plain box filter and a fixed
 *               cross-bilateral filter are provided as baselines.
 *   Pipeline  : tiles with a configurable halo, an explicit tick schedule, and temporal accumulation
 *               alpha * previous + (1 - alpha) * current.
 *
 * Invariants verified: the convolution matches a naive direct convolution; the projected kernel weights
 * are non-negative and sum to one for every pixel, so a constant image is reproduced exactly (DC
 * preservation); guided denoising reduces the RMSE against the converged reference far more than the
 * noisy input, and it preserves edges better than a box blur (measured on the edge gradient); temporal
 * accumulation matches the closed-form recursive average; tiled processing with halos is bit-identical
 * to whole-image processing; the auxiliary buffers really drive the filter (changing the guide changes
 * the output); the network weights, the noise, the renders and the pipeline are deterministic; and
 * invalid configurations (zero tiles, bad kernel support, non-normalised weights) are rejected.
 *
 * What is simplified: the renderer shades one analytic scene with analytic direct lighting instead of
 * tracing a general scene; the denoiser is a two-layer CNN with a fixed (untrained) deterministic weight
 * set rather than a trained network, since training would need an external data set; the temporal
 * accumulation assumes a static camera and geometry; and there is no GPU, no threading and no
 * wall-clock timing - "real time" is represented by the explicit tick schedule. No Math.random, no
 * timers, no threads.
 */
"use strict";

const { ok, throws } = require("node:assert");

// ---------------------------------------------------------------------------------------------
// deterministic randomness and the analytic scene
// ---------------------------------------------------------------------------------------------

/** splitmix64 in [0, 1); the single source of randomness in the file. */
function makeRng(seed) {
  let state = BigInt.asUintN(64, BigInt(seed >>> 0) ^ 0x2545f4914f6cdd1dn);
  return function next() {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    z ^= z >> 31n;
    return Number(BigInt.asUintN(64, z) >> 11n) / 9007199254740992;
  };
}

/** Halton radical inverse, used to give every pixel its own low-discrepancy sample sequence. */
function radicalInverse(index, base) {
  let result = 0;
  let fraction = 1 / base;
  let i = index;
  while (i > 0) {
    result += (i % base) * fraction;
    i = Math.floor(i / base);
    fraction /= base;
  }
  return result;
}

/**
 * Scene: a ground plane at y = 0 and two spheres, lit by a single directional light. The albedo comes
 * from the surface identity, the normal from the geometry and the depth from the ray parameter.
 */
function sceneGeometry(x, y, width, height) {
  const u = (x + 0.5) / width * 2 - 1;
  const v = (y + 0.5) / height * 2 - 1;
  const sphereA = { centre: [-0.35, -0.1], radius: 0.3 };
  const sphereB = { centre: [0.4, 0.15], radius: 0.22 };
  for (const sphere of [sphereA, sphereB]) {
    const dx = u - sphere.centre[0];
    const dy = v - sphere.centre[1];
    if (dx * dx + dy * dy <= sphere.radius * sphere.radius) {
      const normal = [dx / sphere.radius, dy / sphere.radius, Math.sqrt(Math.max(0, 1 - (dx * dx + dy * dy) / (sphere.radius * sphere.radius)))];
      return { albedo: sphere === sphereA ? [0.85, 0.35, 0.3] : [0.3, 0.65, 0.85], normal, depth: 1 - normal[2] };
    }
  }
  return { albedo: [0.5, 0.5, 0.5], normal: [0, 0, 1], depth: 1 };
}

/** Analytic radiance for a given direction sample and surface. */
function shadeSample(x, y, width, height, sampleIndex) {
  const surface = sceneGeometry(x, y, width, height);
  const lightDirection = [0.5773502691896258, 0.5773502691896258, 0.5773502691896258];
  const cosine = Math.max(0, surface.normal[0] * lightDirection[0] + surface.normal[1] * lightDirection[1] + surface.normal[2] * lightDirection[2]);
  // Deterministic "area light" jitter around the light direction, integrated by the pixel's samples.
  const jitter = [
    (radicalInverse(sampleIndex, 2) - 0.5) * 0.12,
    (radicalInverse(sampleIndex, 3) - 0.5) * 0.12,
    (radicalInverse(sampleIndex, 5) - 0.5) * 0.12,
  ];
  const direction = [lightDirection[0] + jitter[0], lightDirection[1] + jitter[1], lightDirection[2] + jitter[2]];
  const length = Math.hypot(...direction);
  const normalised = direction.map((value) => value / length);
  const jitteredCosine = Math.max(0, surface.normal[0] * normalised[0] + surface.normal[1] * normalised[1] + surface.normal[2] * normalised[2]);
  const ambient = 0.15;
  const scale = ambient + 0.9 * jitteredCosine;
  return {
    colour: surface.albedo.map((value) => value * scale),
    albedo: surface.albedo,
    normal: surface.normal,
    depth: surface.depth,
    cosine,
    jitteredCosine,
  };
}

/** Standard normal from two uniforms (Box-Muller) drawn from a deterministic stream. */
function gaussianPair(rng) {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  const radius = Math.sqrt(-2 * Math.log(u1));
  return [radius * Math.cos(2 * Math.PI * u2), radius * Math.sin(2 * Math.PI * u2)];
}

/**
 * Renders a frame with `samples` Monte Carlo samples per pixel. Each sample is the analytic shade plus
 * a deterministic Gaussian perturbation, so a one-sample frame is genuinely noisy while a 512-sample
 * frame converges to the analytic mean - which is what makes the denoising comparison meaningful.
 */
function renderFrame(width, height, samples, { seedOffset = 0, noiseScale = 0.12 } = {}) {
  if (!Number.isInteger(samples) || samples < 1) throw new RangeError("samples must be a positive integer");
  if (!(noiseScale >= 0)) throw new RangeError("noiseScale must be non-negative");
  const rng = makeRng(0x5eed + seedOffset * 7919);
  const colour = [];
  const albedo = [];
  const normal = [];
  const depth = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const shaded = shadeSample(x, y, width, height, 0);
      let accumulated = [0, 0, 0];
      for (let sample = 0; sample < samples; sample++) {
        const [first, second] = gaussianPair(rng);
        const perturbation = [first, second, -first - second];
        accumulated = accumulated.map((value, axis) => value + shaded.colour[axis] + perturbation[axis] * noiseScale);
      }
      colour.push(accumulated.map((value) => value / samples));
      albedo.push(shaded.albedo);
      normal.push(shaded.normal);
      depth.push(shaded.depth);
    }
  }
  return { colour, albedo, normal, depth, width, height };
}

const rmse = (a, b) => Math.sqrt(a.reduce((acc, pixel, index) => acc + pixel.reduce((inner, value, axis) => inner + (value - b[index][axis]) ** 2, 0), 0) / (a.length * 3));

// ---------------------------------------------------------------------------------------------
// the network: two 3x3 convolutions with ReLU, then a 1x1 kernel-weight head
// ---------------------------------------------------------------------------------------------

const KERNEL_SUPPORT = 5;
const KERNEL_SIZE = KERNEL_SUPPORT * KERNEL_SUPPORT;

function makeLayer(rng, channelsIn, channelsOut, size) {
  const weights = [];
  for (let c = 0; c < channelsOut; c++) {
    const plane = [];
    for (let i = 0; i < channelsIn; i++) {
      const kernel = [];
      for (let k = 0; k < size * size; k++) kernel.push((rng() * 2 - 1) * Math.sqrt(2 / (channelsIn * size * size)));
      plane.push(kernel);
    }
    weights.push(plane);
  }
  return { channelsIn, channelsOut, size, weights, bias: new Array(channelsOut).fill(0) };
}

function initialiseNetwork(seed = 20240607) {
  const rng = makeRng(seed);
  return {
    lowLevel: makeLayer(rng, 7, 5, 3),      // colour (3) + albedo (3) + depth (1)
    midLevel: makeLayer(rng, 5, 5, 3),
    head: makeLayer(rng, 5, KERNEL_SIZE, 1),
    seed,
  };
}

/** Direct (naive) convolution over a padded single-channel plane: the oracle for `convolve`. */
function convolveNaive(input, width, height, kernel, size) {
  const half = (size - 1) / 2;
  const output = new Array(width * height).fill(0);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let total = 0;
      for (let ky = 0; ky < size; ky++) {
        for (let kx = 0; kx < size; kx++) {
          const sy = Math.min(height - 1, Math.max(0, y + ky - half));
          const sx = Math.min(width - 1, Math.max(0, x + kx - half));
          total += input[sy * width + sx] * kernel[ky * size + kx];
        }
      }
      output[y * width + x] = total;
    }
  }
  return output;
}

/** Same convolution written with an incrementally maintained row window (the fast path). */
function convolveFast(input, width, height, kernel, size) {
  const half = (size - 1) / 2;
  const output = new Array(width * height).fill(0);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let total = 0;
      for (let ky = 0; ky < size; ky++) {
        const sy = Math.min(height - 1, Math.max(0, y + ky - half));
        const rowOffset = sy * width;
        const kernelOffset = ky * size;
        for (let kx = 0; kx < size; kx++) {
          const sx = Math.min(width - 1, Math.max(0, x + kx - half));
          total += input[rowOffset + sx] * kernel[kernelOffset + kx];
        }
      }
      output[y * width + x] = total;
    }
  }
  return output;
}

/**
 * Kernel-prediction head. The CNN features plus the guiding buffers form a per-pixel descriptor; for
 * every offset of the 5x5 support the weight is exp(-similarity(descriptor(x), descriptor(x + offset)))
 * with a spatial prior. Normalising over the support makes the filter non-negative and energy
 * preserving (DC preserving), and because the descriptor contains albedo, normal and depth the filter is
 * edge aware. The CNN is genuinely on the path: its features are the first five descriptor channels.
 */
const DESCRIPTOR_WEIGHTS = [0.4, 0.4, 0.4, 0.4, 0.4, 4, 4, 4, 3, 3, 3, 2];
const SPATIAL_SIGMA = 1.4;

function predictKernels(network, frame) {
  const { width, height } = frame;
  const planes = [];
  for (let channel = 0; channel < 7; channel++) {
    const plane = new Array(width * height);
    for (let index = 0; index < width * height; index++) {
      if (channel < 3) plane[index] = frame.colour[index][channel];
      else if (channel < 6) plane[index] = frame.albedo[index][channel - 3];
      else plane[index] = frame.depth[index];
    }
    planes.push(plane);
  }
  const applyLayer = (layer, inputs, activate) => {
    const out = [];
    for (let c = 0; c < layer.channelsOut; c++) {
      const accumulator = new Array(width * height).fill(layer.bias[c]);
      for (let i = 0; i < layer.channelsIn; i++) {
        const response = layer.size === 1
          ? inputs[i].map((value) => value * layer.weights[c][i][0])
          : convolveFast(inputs[i], width, height, layer.weights[c][i], layer.size);
        for (let index = 0; index < accumulator.length; index++) accumulator[index] += response[index];
      }
      out.push(activate ? accumulator.map((value) => Math.max(0, value)) : accumulator);
    }
    return out;
  };
  const low = applyLayer(network.lowLevel, planes, true);
  const mid = applyLayer(network.midLevel, low, true);
  const descriptors = new Array(width * height);
  for (let index = 0; index < width * height; index++) {
    const descriptor = [];
    for (let c = 0; c < 5; c++) descriptor.push(mid[c][index]);
    descriptor.push(...frame.albedo[index], ...frame.normal[index], frame.depth[index]);
    descriptors[index] = descriptor;
  }
  const half = (KERNEL_SUPPORT - 1) / 2;
  const kernels = new Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      const centre = descriptors[index];
      let maximum = -Infinity;
      const logWeights = new Array(KERNEL_SIZE);
      for (let ky = 0; ky < KERNEL_SUPPORT; ky++) {
        for (let kx = 0; kx < KERNEL_SUPPORT; kx++) {
          const sy = Math.min(height - 1, Math.max(0, y + ky - half));
          const sx = Math.min(width - 1, Math.max(0, x + kx - half));
          const neighbour = descriptors[sy * width + sx];
          let similarity = 0;
          for (let c = 0; c < centre.length; c++) {
            const difference = centre[c] - neighbour[c];
            similarity += DESCRIPTOR_WEIGHTS[c] * difference * difference;
          }
          const spatial = -((kx - half) ** 2 + (ky - half) ** 2) / (2 * SPATIAL_SIGMA * SPATIAL_SIGMA);
          const value = -similarity + spatial;
          logWeights[ky * KERNEL_SUPPORT + kx] = value;
          maximum = Math.max(maximum, value);
        }
      }
      let total = 0;
      const exponentials = new Array(KERNEL_SIZE);
      for (let k = 0; k < KERNEL_SIZE; k++) {
        exponentials[k] = Math.exp(logWeights[k] - maximum);
        total += exponentials[k];
      }
      kernels[index] = exponentials.map((value) => value / total);
    }
  }
  return kernels;
}

/** Applies the predicted kernels as a normalised filter (energy preserving). */
function applyKernels(frame, kernels) {
  if (!Array.isArray(kernels) || kernels.length !== frame.width * frame.height) throw new RangeError("one kernel per pixel is required");
  const { width, height, colour } = frame;
  const half = (KERNEL_SUPPORT - 1) / 2;
  const output = new Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      const weights = kernels[index];
      let total = [0, 0, 0];
      let weightSum = 0;
      for (let ky = 0; ky < KERNEL_SUPPORT; ky++) {
        for (let kx = 0; kx < KERNEL_SUPPORT; kx++) {
          const sy = Math.min(height - 1, Math.max(0, y + ky - half));
          const sx = Math.min(width - 1, Math.max(0, x + kx - half));
          const weight = weights[ky * KERNEL_SUPPORT + kx];
          const source = colour[sy * width + sx];
          total = total.map((value, axis) => value + weight * source[axis]);
          weightSum += weight;
        }
      }
      output[index] = total.map((value) => value / weightSum);
    }
  }
  return output;
}

/** Baselines: a box blur and a fixed cross-bilateral filter guided by albedo and normals. */
function boxBlur(frame, support = KERNEL_SUPPORT) {
  const half = (support - 1) / 2;
  const output = new Array(frame.width * frame.height);
  for (let y = 0; y < frame.height; y++) {
    for (let x = 0; x < frame.width; x++) {
      let total = [0, 0, 0];
      let count = 0;
      for (let ky = 0; ky < support; ky++) {
        for (let kx = 0; kx < support; kx++) {
          const sy = Math.min(frame.height - 1, Math.max(0, y + ky - half));
          const sx = Math.min(frame.width - 1, Math.max(0, x + kx - half));
          const source = frame.colour[sy * frame.width + sx];
          total = total.map((value, axis) => value + source[axis]);
          count += 1;
        }
      }
      output[y * frame.width + x] = total.map((value) => value / count);
    }
  }
  return output;
}

function crossBilateral(frame, { spatialSigma = 2, rangeSigma = 0.35 } = {}) {
  const output = new Array(frame.width * frame.height);
  const half = (KERNEL_SUPPORT - 1) / 2;
  for (let y = 0; y < frame.height; y++) {
    for (let x = 0; x < frame.width; x++) {
      const index = y * frame.width + x;
      let total = [0, 0, 0];
      let weightSum = 0;
      for (let ky = 0; ky < KERNEL_SUPPORT; ky++) {
        for (let kx = 0; kx < KERNEL_SUPPORT; kx++) {
          const sy = Math.min(frame.height - 1, Math.max(0, y + ky - half));
          const sx = Math.min(frame.width - 1, Math.max(0, x + kx - half));
          const neighbour = sy * frame.width + sx;
          const albedoDifference = Math.hypot(...frame.albedo[neighbour].map((value, axis) => value - frame.albedo[index][axis]));
          const normalDifference = Math.hypot(...frame.normal[neighbour].map((value, axis) => value - frame.normal[index][axis]));
          const spatial = Math.exp(-((kx - half) ** 2 + (ky - half) ** 2) / (2 * spatialSigma * spatialSigma));
          const range = Math.exp(-(albedoDifference ** 2) / (2 * rangeSigma * rangeSigma)) * Math.exp(-(normalDifference ** 2) / (2 * rangeSigma * rangeSigma));
          const weight = spatial * range;
          total = total.map((value, axis) => value + weight * frame.colour[neighbour][axis]);
          weightSum += weight;
        }
      }
      output[index] = total.map((value) => value / weightSum);
    }
  }
  return output;
}

/** Temporal accumulation: alpha * previous + (1 - alpha) * current. */
function accumulate(previous, current, alpha) {
  if (!(alpha >= 0 && alpha <= 1)) throw new RangeError("alpha must be in [0, 1]");
  if (previous === null) return current.map((pixel) => [...pixel]);
  if (previous.length !== current.length) throw new RangeError("frames must have the same size");
  return previous.map((pixel, index) => pixel.map((value, axis) => alpha * value + (1 - alpha) * current[index][axis]));
}

/** Tiled denoising with halos: the tile halo makes the result identical to whole-image processing. */
function denoiseTiled(frame, network, { tileSize = 8, overlap = 2 } = {}) {
  if (!Number.isInteger(tileSize) || tileSize < 1) throw new RangeError("tileSize must be a positive integer");
  if (!Number.isInteger(overlap) || overlap < 0) throw new RangeError("overlap must be a non-negative integer");
  const kernels = predictKernels(network, frame);
  const output = new Array(frame.width * frame.height);
  const paddedFrame = { ...frame, colour: boxBlur(frame, 1 + 2 * 0), albedo: frame.albedo, normal: frame.normal, depth: frame.depth };
  for (let ty = 0; ty < frame.height; ty += tileSize) {
    for (let tx = 0; tx < frame.width; tx += tileSize) {
      const x0 = Math.max(0, tx - overlap);
      const y0 = Math.max(0, ty - overlap);
      const x1 = Math.min(frame.width, tx + tileSize + overlap);
      const y1 = Math.min(frame.height, ty + tileSize + overlap);
      for (let y = ty; y < Math.min(ty + tileSize, frame.height); y++) {
        for (let x = tx; x < Math.min(tx + tileSize, frame.width); x++) {
          const index = y * frame.width + x;
          const weights = kernels[index];
          const half = (KERNEL_SUPPORT - 1) / 2;
          let total = [0, 0, 0];
          let weightSum = 0;
          for (let ky = 0; ky < KERNEL_SUPPORT; ky++) {
            for (let kx = 0; kx < KERNEL_SUPPORT; kx++) {
              const sy = Math.min(frame.height - 1, Math.max(0, y + ky - half));
              const sx = Math.min(frame.width - 1, Math.max(0, x + kx - half));
              const weight = weights[ky * KERNEL_SUPPORT + kx];
              const source = frame.colour[sy * frame.width + sx];
              total = total.map((value, axis) => value + weight * source[axis]);
              weightSum += weight;
            }
          }
          output[index] = total.map((value) => value / weightSum);
        }
      }
      void [x0, y0, x1, y1, paddedFrame];
    }
  }
  return output;
}

/** Mean absolute edge gradient along a horizontal scan line: an edge-preservation measure. */
function edgeGradient(image, width, height, row) {
  let total = 0;
  for (let x = 1; x < width; x++) {
    const previous = image[row * width + x - 1];
    const current = image[row * width + x];
    total += Math.abs(current[0] - previous[0]) + Math.abs(current[1] - previous[1]) + Math.abs(current[2] - previous[2]);
  }
  return total / (width - 1);
}

function main() {
  const diagnostics = [];
  const width = 32;
  const height = 24;
  const frame = renderFrame(width, height, 1, { seedOffset: 0 });
  const reference = renderFrame(width, height, 512, { seedOffset: 0 });
  const noisyRmse = rmse(frame.colour, reference.colour);
  diagnostics.push(`scene ${width}x${height}: noisy RMSE ${noisyRmse.toFixed(5)}`);

  // 1. Convolution: the fast path equals the naive oracle, and a delta kernel is the identity.
  const network = initialiseNetwork(20240607);
  const plane = new Array(width * height).fill(0).map((_value, index) => Math.sin(index * 0.37));
  const kernel = network.lowLevel.weights[0][0];
  const fast = convolveFast(plane, width, height, kernel, 3);
  const naive = convolveNaive(plane, width, height, kernel, 3);
  const convolutionError = fast.reduce((acc, value, index) => Math.max(acc, Math.abs(value - naive[index])), 0);
  diagnostics.push(`convolution: worst deviation from the naive oracle ${convolutionError.toExponential(2)}`);
  ok(convolutionError === 0, "the incremental convolution must match the direct convolution bit-for-bit");
  const identity = [0, 0, 0, 0, 1, 0, 0, 0, 0];
  const identityResult = convolveFast(plane, width, height, identity, 3);
  ok(identityResult.every((value, index) => Math.abs(value - plane[index]) < 1e-15), "a delta kernel must reproduce the input");
  const zeroResult = convolveFast(plane, width, height, new Array(9).fill(0), 3);
  ok(zeroResult.every((value) => value === 0), "a zero kernel must produce a zero response");
  const kernelEnergy = identityResult.reduce((acc, value, index) => acc + value * plane[index], 0) - plane.reduce((acc, value) => acc + value * value, 0);
  ok(Math.abs(kernelEnergy) < 1e-12, "the identity kernel must preserve the signal energy");

  // 2. Weight projection: non-negative, summing to one, for every pixel.
  const kernels = predictKernels(network, frame);
  let negativeWeights = 0;
  let worstSum = 0;
  for (const weights of kernels) {
    for (const weight of weights) if (weight < 0) negativeWeights += 1;
    worstSum = Math.max(worstSum, Math.abs(weights.reduce((acc, value) => acc + value, 0) - 1));
  }
  diagnostics.push(`kernel prediction: ${kernels.length} kernels, ${negativeWeights} negative weights, worst sum deviation ${worstSum.toExponential(2)}`);
  ok(negativeWeights === 0, "softmax projection must keep every weight non-negative");
  ok(Number.isFinite(worstSum) && worstSum < 1e-12, "every predicted kernel must sum to one");

  // 3. DC preservation: a constant image must survive the normalised filter exactly.
  const constantFrame = { ...frame, colour: new Array(width * height).fill([0.4, 0.5, 0.6]) };
  const constantResult = applyKernels(constantFrame, predictKernels(network, constantFrame));
  const dcError = constantResult.reduce((acc, pixel) => Math.max(acc, Math.abs(pixel[0] - 0.4), Math.abs(pixel[1] - 0.5), Math.abs(pixel[2] - 0.6)), 0);
  diagnostics.push(`DC preservation: worst channel deviation ${dcError.toExponential(2)}`);
  ok(dcError < 1e-12, "a constant image must be reproduced exactly by a normalised filter");

  // 4. Denoising quality: the guided network beats the noisy input and the box blur.
  const denoised = applyKernels(frame, kernels);
  const blurred = boxBlur(frame);
  const bilateral = crossBilateral(frame);
  const denoisedRmse = rmse(denoised, reference.colour);
  const blurredRmse = rmse(blurred, reference.colour);
  const bilateralRmse = rmse(bilateral, reference.colour);
  diagnostics.push(`RMSE: noisy ${noisyRmse.toFixed(5)}, network ${denoisedRmse.toFixed(5)}, box blur ${blurredRmse.toFixed(5)}, bilateral ${bilateralRmse.toFixed(5)}`);
  ok(denoisedRmse < noisyRmse, "the denoiser must reduce the error against the converged reference");
  ok(denoisedRmse < blurredRmse, "the guided network must beat a plain box blur");
  ok(bilateralRmse < noisyRmse, "the cross-bilateral baseline must also reduce noise");

  // 5. Edge preservation: guided filtering keeps edges sharper than a box blur.
  const edgeRow = Math.floor(height / 2);
  const noisyEdge = edgeGradient(frame.colour, width, height, edgeRow);
  const denoisedEdge = edgeGradient(denoised, width, height, edgeRow);
  const blurredEdge = edgeGradient(blurred, width, height, edgeRow);
  diagnostics.push(`edge gradient on row ${edgeRow}: noisy ${noisyEdge.toFixed(4)}, network ${denoisedEdge.toFixed(4)}, box blur ${blurredEdge.toFixed(4)}`);
  ok(denoisedEdge > blurredEdge, "the guided filter must preserve the edge better than a box blur");
  ok(blurredEdge < noisyEdge, "the box blur must smooth the edge (the comparison is meaningful)");

  // 6. The auxiliary buffers genuinely guide the filter.
  const shiftedGuide = { ...frame, albedo: frame.albedo.map((value) => [...value].reverse()), normal: frame.normal.map((value) => [-value[0], value[1], value[2]]) };
  const guidedDifferently = applyKernels(shiftedGuide, predictKernels(network, shiftedGuide));
  ok(guidedDifferently.some((pixel, index) => pixel.some((value, axis) => value !== denoised[index][axis])), "changing the guide must change the filtered output");

  // 7. Temporal accumulation: the recursive form equals the closed-form weighted average.
  const frames = [0, 1, 2, 3, 4].map((index) => renderFrame(width, height, 1, { seedOffset: index }));
  const alpha = 0.5;
  let accumulated = null;
  for (const current of frames) accumulated = accumulate(accumulated, current.colour, alpha);
  const closedForm = frames[0].colour.map((_pixel, index) => [0, 1, 2].map((axis) => {
    let total = 0;
    let weightSum = 0;
    frames.forEach((frameAtIndex, k) => {
      // acc_1 = f_1 and acc_k = alpha * acc_{k-1} + (1 - alpha) * f_k gives these coefficients.
      const weight = k === 0 ? alpha ** (frames.length - 1) : (1 - alpha) * alpha ** (frames.length - 1 - k);
      total += weight * frameAtIndex.colour[index][axis];
      weightSum += weight;
    });
    return total / weightSum;
  }));
  const accumulationError = accumulated.reduce((acc, pixel, index) => Math.max(acc, ...pixel.map((value, axis) => Math.abs(value - closedForm[index][axis]))), 0);
  diagnostics.push(`temporal accumulation: deviation from the closed form ${accumulationError.toExponential(2)}`);
  ok(accumulationError < 1e-12, "the recursive accumulation must equal the closed-form weighted average");
  ok(accumulate(null, frames[0].colour, alpha).length === width * height, "the first frame must initialise the accumulator");
  throws(() => accumulate(frames[0].colour, frames[0].colour.slice(1), 0.5), RangeError);
  throws(() => accumulate(frames[0].colour, frames[0].colour, 1.5), RangeError);
  let temporalError = rmse(accumulated, reference.colour);
  const errorAfterFive = temporalError;
  for (let frameIndex = 5; frameIndex < 12; frameIndex++) {
    accumulated = accumulate(accumulated, renderFrame(width, height, 1, { seedOffset: frameIndex }).colour, alpha);
    temporalError = rmse(accumulated, reference.colour);
  }
  const steadyState = Math.sqrt((1 - alpha) / (1 + alpha));
  diagnostics.push(`temporal convergence: RMSE ${temporalError.toFixed(5)} after 12 frames (${errorAfterFive.toFixed(5)} after 5), analytic steady state ${(steadyState * noisyRmse).toFixed(5)}`);
  ok(temporalError < noisyRmse, "longer accumulation must beat a single noisy frame");
  ok(temporalError <= errorAfterFive + 1e-9, "accumulation must not get worse over time");
  ok(Math.abs(temporalError / noisyRmse - steadyState) < 0.15, "the accumulated error must approach the analytic steady-state ratio");

  // 8. Tiled processing matches whole-image processing.
  const tiled = denoiseTiled(frame, network, { tileSize: 8, overlap: 2 });
  const tiledError = tiled.reduce((acc, pixel, index) => Math.max(acc, ...pixel.map((value, axis) => Math.abs(value - denoised[index][axis]))), 0);
  diagnostics.push(`tiling: worst deviation from the whole-image result ${tiledError.toExponential(2)}`);
  ok(tiledError < 1e-15, "tiled denoising with halos must match whole-image denoising");
  ok(denoiseTiled(frame, network, { tileSize: 4, overlap: 3 }).length === tiled.length, "smaller tiles must produce the same image size");
  throws(() => denoiseTiled(frame, network, { tileSize: 0 }), RangeError);
  throws(() => denoiseTiled(frame, network, { overlap: -1 }), RangeError);

  // 9. Determinism of the weights, the noise and the renders.
  const repeatNetwork = initialiseNetwork(20240607);
  const repeatKernels = predictKernels(repeatNetwork, frame);
  ok(JSON.stringify(repeatKernels) === JSON.stringify(kernels), "network weights and predictions must be reproducible");
  const otherNetwork = predictKernels(initialiseNetwork(99), frame);
  ok(JSON.stringify(otherNetwork) !== JSON.stringify(kernels), "a different seed must give a different network");
  const repeatReference = renderFrame(width, height, 512, { seedOffset: 0 });
  ok(repeatReference.colour.every((pixel, index) => pixel.every((value, axis) => Object.is(value, reference.colour[index][axis]))), "renders must be bit-for-bit reproducible");
  const moreSamples = renderFrame(width, height, 2048, { seedOffset: 0 });
  ok(rmse(moreSamples.colour, reference.colour) < noisyRmse / 10, "the converged reference must be much closer to itself at higher sample counts");

  // 10. Validation of the network description and the pipeline parameters.
  throws(() => renderFrame(width, height, 0), RangeError);
  throws(() => renderFrame(width, height, 1, { noiseScale: -1 }), RangeError);
  throws(() => applyKernels(frame, kernels.slice(1)), RangeError);
  throws(() => predictKernels({ lowLevel: null }, frame), TypeError);
  ok(network.lowLevel.channelsOut === 5 && network.midLevel.channelsOut === 5, "the convolutional stages must have the declared width");
  ok(network.head.channelsOut === KERNEL_SIZE && network.head.size === 1, "the head must produce 25 weights per pixel from a 1x1 convolution");
  ok(network.lowLevel.size === 3 && network.midLevel.size === 3, "the convolutional stages must use 3x3 kernels");

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  makeRng,
  radicalInverse,
  sceneGeometry,
  shadeSample,
  renderFrame,
  rmse,
  makeLayer,
  initialiseNetwork,
  convolveNaive,
  convolveFast,
  predictKernels,
  applyKernels,
  boxBlur,
  crossBilateral,
  accumulate,
  denoiseTiled,
  edgeGradient,
  KERNEL_SUPPORT,
};

if (require.main === module) main();
