/** Task 301 - FFT outline and small convolution (conceptual). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

function fft(real, imag, invert) {
  const n = real.length;
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j |= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }

  for (let length = 2; length <= n; length <<= 1) {
    const angle = ((2 * Math.PI) / length) * (invert ? -1 : 1);
    const wlenReal = Math.cos(angle);
    const wlenImag = Math.sin(angle);
    const half = length >> 1;
    for (let start = 0; start < n; start += length) {
      let wReal = 1.0;
      let wImag = 0.0;
      for (let k = 0; k < half; k++) {
        const uReal = real[start + k];
        const uImag = imag[start + k];
        const vReal = real[start + k + half] * wReal - imag[start + k + half] * wImag;
        const vImag = real[start + k + half] * wImag + imag[start + k + half] * wReal;
        real[start + k] = uReal + vReal;
        imag[start + k] = uImag + vImag;
        real[start + k + half] = uReal - vReal;
        imag[start + k + half] = uImag - vImag;
        const nextReal = wReal * wlenReal - wImag * wlenImag;
        wImag = wReal * wlenImag + wImag * wlenReal;
        wReal = nextReal;
      }
    }
  }

  if (invert) {
    for (let i = 0; i < n; i++) {
      real[i] /= n;
      imag[i] /= n;
    }
  }
}

function convolutionFft(a, b) {
  if (!a.length || !b.length) return [];
  let size = 1;
  while (size < a.length + b.length - 1) size <<= 1;
  const realA = new Array(size).fill(0);
  const imagA = new Array(size).fill(0);
  const realB = new Array(size).fill(0);
  const imagB = new Array(size).fill(0);
  for (let i = 0; i < a.length; i++) realA[i] = a[i];
  for (let i = 0; i < b.length; i++) realB[i] = b[i];

  fft(realA, imagA, false);
  fft(realB, imagB, false);
  for (let i = 0; i < size; i++) {
    const re = realA[i] * realB[i] - imagA[i] * imagB[i];
    const im = realA[i] * imagB[i] + imagA[i] * realB[i];
    realA[i] = re;
    imagA[i] = im;
  }
  fft(realA, imagA, true);

  const out = [];
  for (let i = 0; i < a.length + b.length - 1; i++) {
    const value = Math.round(realA[i]);
    out.push(value === 0 ? 0 : value);   // normalise -0 so deep equals 0
  }
  return out;
}

function convolutionNaive(a, b) {
  if (!a.length || !b.length) return [];
  const out = new Array(a.length + b.length - 1).fill(0);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) out[i + j] += a[i] * b[j];
  }
  return out;
}

function lcgVectors(count, seed = 1123) {
  const vectors = [];
  for (let k = 0; k < count; k++) {
    const size = 1 + (seed % 6);
    const vector = [];
    for (let i = 0; i < size; i++) {
      seed = (seed * 75 + 74) % 65537;
      vector.push((seed % 11) - 5);
    }
    vectors.push(vector);
  }
  return vectors;
}

module.exports = { fft, convolutionFft, convolutionNaive, lcgVectors };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(convolutionFft([1, 2, 3], [0, 1, 1]), [0, 1, 3, 5, 3]);
  assert.deepStrictEqual(convolutionFft([1, 1], [1, 1]), [1, 2, 1]);
  assert.deepStrictEqual(convolutionFft([7], [9]), [63]);
  assert.deepStrictEqual(convolutionFft([], [1, 2]), []);
  assert.deepStrictEqual(convolutionFft([4, 5], [6]), [24, 30]);
  assert.deepStrictEqual(convolutionFft([1], [3, 4, 5]), [3, 4, 5]);

  const vectors = lcgVectors(40);
  for (let i = 0; i + 1 < vectors.length; i += 2) {
    const a = vectors[i];
    const b = vectors[i + 1];
    assert.deepStrictEqual(convolutionFft(a, b), convolutionNaive(a, b), `forward ${i}`);
    assert.deepStrictEqual(convolutionFft(b, a), convolutionNaive(a, b), `swapped ${i}`);
  }
  console.log("PASS");
}
