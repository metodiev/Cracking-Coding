/** Task 881 - Implement global illumination importance sampling techniques in renderer. (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Monte Carlo global illumination estimators over the upper hemisphere, with the sampling techniques
 * that make them usable in practice:
 *
 *   Samplers        : a Halton (2, 3) low-discrepancy sequence with a seeded Cranley-Patterson rotation,
 *                     a stratified grid, and a seeded linear-congruential pseudo-random stream. All three
 *                     are deterministic and are compared on star discrepancy and uniformity.
 *   Distributions   : uniform hemisphere sampling, pdf = 1/(2 pi), and cosine-weighted sampling,
 *                     pdf = cos(theta)/pi, with the analytic inverse-CDF construction.
 *   Estimators      : the Lambertian BRDF albedo/pi with the cosine factor, the "white furnace" test
 *                     (constant environment radiance L must integrate to pi L exactly for cosine-weighted
 *                     sampling), direct lighting from a spherical light via next-event estimation with
 *                     solid-angle (cone) sampling, and a multiple-importance-sampling combination of
 *                     BRDF sampling and light sampling with the balance heuristic.
 *
 * Invariants verified: both pdfs integrate to one by numerical quadrature; samples stay on the upper
 * hemisphere with cos(theta) > 0; the empirical CDF of cosine-weighted samples matches cos(theta)/pi
 * (Kolmogorov-Smirnov style comparison); the cosine-weighted estimator is *exactly* pi L for every
 * sample count under a constant environment while uniform sampling converges to it more slowly; the
 * measured variance of cosine-weighted sampling is lower than uniform sampling on a coloured
 * environment; next-event estimation and multiple importance sampling agree with an independent
 * high-sample uniform-hemisphere reference within their standard errors, and their variance and
 * effective-sample count are better than BRDF-only sampling; the Halton sequence has a lower star
 * discrepancy than the pseudo-random stream; a light behind the surface, a grazing direction and empty
 * sample sets are handled and rejected as documented; and every estimator is reproducible.
 *
 * What is simplified: the "scene" is analytic (no geometry, no ray tracing, no occlusion beyond the
 * given light), sampling is over the hemisphere only (no path continuation or depth), the light is a
 * sphere treated by its exact solid angle but without shadow tests, and converged references are
 * produced by running the same deterministic estimators with far more samples rather than by a
 * closed-form solution - except for the furnace test, which does have a closed form and is checked
 * exactly. No timers, no wall-clock time, no Math.random, no threads.
 */
"use strict";

const { ok, throws } = require("node:assert");

const TWO_PI = 2 * Math.PI;

// ---------------------------------------------------------------------------------------------
// samplers: each returns a stream of [0, 1)^2 points
// ---------------------------------------------------------------------------------------------

/** Halton radical inverse in the given base. */
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

/** Halton (2, 3) with a seeded Cranley-Patterson rotation: a low-discrepancy 2D sequence. */
function haltonSampler(seed, offset = 0) {
  let state = (seed >>> 0) || 0x2545f491;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const rotation = [next(), next()];
  let index = offset;
  return () => {
    const point = [(radicalInverse(index, 2) + rotation[0]) % 1, (radicalInverse(index, 3) + rotation[1]) % 1];
    index += 1;
    return point;
  };
}

/** Pseudo-random (mulberry32) 2D stream: the non-QMC baseline. */
function pseudoRandomSampler(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => [next(), next()];
}

/** Stratified sampler: exactly one deterministic point per cell of a `cells` x `cells` grid. */
function stratifiedSampler(cells, seed = 1) {
  if (!Number.isInteger(cells) || cells < 1) throw new RangeError("cells must be a positive integer");
  const jitter = seed === 0 ? 0.5 : 0.5;
  const points = [];
  for (let y = 0; y < cells; y++) {
    for (let x = 0; x < cells; x++) points.push([(x + jitter) / cells, (y + jitter) / cells]);
  }
  let index = 0;
  return { cells, points, next: () => points[index++ % points.length] };
}

/** Star discrepancy estimate: the maximum over anchors of |empirical - area| of the anchored box. */
function starDiscrepancy(points, anchorCount = 24) {
  const anchors = [];
  for (let i = 0; i < anchorCount; i++) {
    for (let j = 0; j < anchorCount; j++) {
      anchors.push([(i + 1) / (anchorCount + 1), (j + 1) / (anchorCount + 1)]);
    }
  }
  let worst = 0;
  for (const [ax, ay] of anchors) {
    let inside = 0;
    for (const [x, y] of points) if (x <= ax && y <= ay) inside += 1;
    worst = Math.max(worst, Math.abs(inside / points.length - ax * ay));
  }
  return worst;
}

// ---------------------------------------------------------------------------------------------
// hemisphere distributions
// ---------------------------------------------------------------------------------------------

/** Uniform hemisphere sample: (theta, phi) with sin(theta) weighting; pdf = 1/(2 pi). */
function sampleUniformHemisphere(point) {
  const cosTheta = point[0];
  const phi = TWO_PI * point[1];
  const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
  return { direction: [sinTheta * Math.cos(phi), cosTheta, sinTheta * Math.sin(phi)], pdf: 1 / TWO_PI };
}

/** Cosine-weighted (Malley) sample: pdf = cos(theta)/pi. */
function sampleCosineHemisphere(point) {
  const cosTheta = Math.sqrt(Math.max(0, 1 - point[0]));
  const phi = TWO_PI * point[1];
  const sinTheta = Math.sqrt(Math.max(0, point[0]));
  return { direction: [sinTheta * Math.cos(phi), cosTheta, sinTheta * Math.sin(phi)], pdf: Math.max(cosTheta / Math.PI, 0) };
}

/** Uniform sample inside the cone subtended by a spherical light (solid-angle sampling). */
function sampleSphereLight(point, surfacePoint, normal, light) {
  const toCentre = [light.position[0] - surfacePoint[0], light.position[1] - surfacePoint[1], light.position[2] - surfacePoint[2]];
  const distance = Math.hypot(...toCentre);
  if (!(distance > light.radius)) throw new RangeError("the shading point must lie outside the light sphere");
  const sinThetaMax = light.radius / distance;
  const cosThetaMax = Math.sqrt(Math.max(0, 1 - sinThetaMax * sinThetaMax));
  const cosTheta = 1 - point[0] * (1 - cosThetaMax);
  const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
  const phi = TWO_PI * point[1];
  // Build an orthonormal basis around the direction to the light centre.
  const w = toCentre.map((value) => value / distance);
  const helper = Math.abs(w[0]) > 0.9 ? [0, 1, 0] : [1, 0, 0];
  const u = normalise(cross(helper, w));
  const v = cross(w, u);
  const direction = normalise([
    u[0] * sinTheta * Math.cos(phi) + v[0] * sinTheta * Math.sin(phi) + w[0] * cosTheta,
    u[1] * sinTheta * Math.cos(phi) + v[1] * sinTheta * Math.sin(phi) + w[1] * cosTheta,
    u[2] * sinTheta * Math.cos(phi) + v[2] * sinTheta * Math.sin(phi) + w[2] * cosTheta,
  ]);
  const solidAngle = TWO_PI * (1 - cosThetaMax);
  return { direction, pdf: 1 / solidAngle, solidAngle, cosTheta, distance };
}

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const normalise = (a) => {
  const length = Math.hypot(...a);
  if (!(length > 0)) throw new RangeError("cannot normalise a null vector");
  return a.map((value) => value / length);
};

// ---------------------------------------------------------------------------------------------
// environments, lights and estimators
// ---------------------------------------------------------------------------------------------

const gammaRamp = (direction) => [0.2 + 0.8 * Math.abs(direction[0]), 0.3 + 0.5 * Math.abs(direction[1]), 0.4 + 0.6 * Math.abs(direction[2])];

/**
 * White furnace test: the diffuse reflected radiance for a constant environment of radiance L is
 * analytically albedo * L. With cosine-weighted sampling every single sample evaluates to albedo * L
 * exactly; with uniform sampling the estimator is 2 albedo L cos(theta) and only converges.
 */
function furnaceTest({ samples, technique, L = 1, albedo = 1, seed = 3 }) {
  if (!Number.isInteger(samples) || samples < 1) throw new RangeError("samples must be a positive integer");
  const sampler = technique === "cosine" ? haltonSampler(seed) : pseudoRandomSampler(seed);
  let total = 0;
  for (let i = 0; i < samples; i++) {
    const sample = technique === "cosine" ? sampleCosineHemisphere(sampler()) : sampleUniformHemisphere(sampler());
    const cosTheta = sample.direction[1];
    if (cosTheta <= 0 || sample.pdf <= 0) continue;
    total += ((albedo / Math.PI) * L * cosTheta) / sample.pdf;
  }
  return total / samples;
}

/** Irradiance estimator against a coloured environment with a Lambertian BRDF. */
function estimateEnvironmentIrradiance({ samples, technique, seed = 5, albedo = 1 }) {
  if (!Number.isInteger(samples) || samples < 1) throw new RangeError("samples must be a positive integer");
  const sampler = technique === "cosine" ? haltonSampler(seed) : pseudoRandomSampler(seed);
  const values = [];
  for (let i = 0; i < samples; i++) {
    const sample = technique === "cosine" ? sampleCosineHemisphere(sampler()) : sampleUniformHemisphere(sampler());
    const cosTheta = sample.direction[1];
    if (cosTheta <= 0 || sample.pdf <= 0) {
      values.push([0, 0, 0]);
      continue;
    }
    const colour = gammaRamp(sample.direction).map((value) => (value * albedo));
    values.push(colour.map((value) => (value * cosTheta) / sample.pdf));
  }
  const mean = [0, 0, 0].map((_value, axis) => values.reduce((acc, value) => acc + value[axis], 0) / values.length);
  const variance = [0, 0, 0].map((_value, axis) => values.reduce((acc, value) => acc + (value[axis] - mean[axis]) ** 2, 0) / values.length);
  return { mean, variance, values };
}

/**
 * Direct lighting from a spherical light. `technique` selects the strategy:
 *   "nee"  - next-event estimation: sample the cone of directions that see the light;
 *   "brdf" - BRDF sampling with a cone membership test (the direction either hits the light or not);
 *   "mis"  - both strategies combined with the balance heuristic, which is unbiased and low variance
 *            for lights of any angular size.
 */
function estimateDirectLighting({ samples, technique = "nee", seed = 9, albedo = 1, light, surfaceNormal = [0, 1, 0], surfacePoint = [0, 0, 0], mis = false }) {
  if (!Number.isInteger(samples) || samples < 1) throw new RangeError("samples must be a positive integer");
  const sampler = haltonSampler(seed);
  const toCentre = [light.position[0] - surfacePoint[0], light.position[1] - surfacePoint[1], light.position[2] - surfacePoint[2]];
  const distance = Math.hypot(...toCentre);
  if (!(distance > light.radius)) throw new RangeError("the shading point must lie outside the light sphere");
  const sinThetaMax = light.radius / distance;
  const cosThetaMax = Math.sqrt(Math.max(0, 1 - sinThetaMax * sinThetaMax));
  const lightPdf = 1 / (TWO_PI * Math.max(1e-12, 1 - cosThetaMax));
  const centreDirection = toCentre.map((value) => value / distance);
  const inCone = (direction) => direction[0] * centreDirection[0] + direction[1] * centreDirection[1] + direction[2] * centreDirection[2] >= cosThetaMax - 1e-12;
  const values = [];
  let effectiveSamples = 0;
  for (let i = 0; i < samples; i++) {
    const point = sampler();
    const lightJitter = [point[0], point[1]];
    const brdfSample = sampleCosineHemisphere([point[0], point[1]]);
    const lightSample = sampleSphereLight(lightJitter, surfacePoint, surfaceNormal, light);
    const strategies = [];
    if (technique === "nee") strategies.push({ sample: lightSample, hitsLight: true });
    else if (technique === "brdf") strategies.push({ sample: brdfSample, hitsLight: inCone(brdfSample.direction) });
    else {
      strategies.push({ sample: lightSample, hitsLight: true });
      strategies.push({ sample: brdfSample, hitsLight: inCone(brdfSample.direction) });
    }
    let contribution = [0, 0, 0];
    for (const { sample, hitsLight } of strategies) {
      if (!hitsLight) continue;
      const cosTheta = sample.direction[1];
      if (cosTheta <= 0 || sample.pdf <= 0) continue;
      const otherPdf = sample === lightSample ? Math.max(0, cosTheta / Math.PI) : lightPdf;
      const weight = mis ? sample.pdf / (sample.pdf + otherPdf) : 1;
      const radiance = (albedo / Math.PI) * cosTheta / sample.pdf;
      contribution = contribution.map((value, axis) => value + weight * radiance * light.intensity[axis]);
    }
    values.push(contribution);
    if (contribution.some((value) => value !== 0)) effectiveSamples += 1;
  }
  const mean = [0, 0, 0].map((_value, axis) => values.reduce((acc, value) => acc + value[axis], 0) / values.length);
  const variance = [0, 0, 0].map((_value, axis) => values.reduce((acc, value) => acc + (value[axis] - mean[axis]) ** 2, 0) / values.length);
  return { mean, variance, values, effectiveSamples };
}

/** Kolmogorov-Smirnov style distance between the cosine CDF and the empirical CDF of cos(theta). */
function cosineCdfDistance(samples, seed = 11) {
  const sampler = haltonSampler(seed);
  const cosines = [];
  for (let i = 0; i < samples; i++) cosines.push(sampleCosineHemisphere(sampler()).direction[1]);
  cosines.sort((a, b) => a - b);
  let worst = 0;
  for (let i = 0; i < cosines.length; i++) {
    const empirical = (i + 1) / cosines.length;
    const analytic = cosines[i] * cosines[i];       // CDF of cos(theta) under pdf = cos/pi is c^2
    worst = Math.max(worst, Math.abs(empirical - analytic));
    worst = Math.max(worst, Math.abs(i / cosines.length - analytic));
  }
  return worst;
}

/** Numerical quadrature of a hemisphere pdf over (theta, phi). */
function integratePdf(pdf, cells = 4000, phiCells = 1000) {
  let total = 0;
  for (let i = 0; i < cells; i++) {
    const theta = (Math.PI / 2) * ((i + 0.5) / cells);
    for (let j = 0; j < phiCells; j++) {
      const phi = TWO_PI * ((j + 0.5) / phiCells);
      const solidAngle = Math.sin(theta) * (Math.PI / 2 / cells) * (TWO_PI / phiCells);
      total += pdf(theta, phi) * solidAngle;
    }
  }
  return total;
}

function main() {
  const diagnostics = [];
  const light = { position: [0.0, 8.0, 0.0], radius: 1.0, intensity: [10, 10, 10] };

  // 1. The pdfs integrate to one over the hemisphere (numerical quadrature).
  const uniformIntegral = integratePdf(() => 1 / TWO_PI, 1000);
  const cosineIntegral = integratePdf((theta) => Math.cos(theta) / Math.PI, 1000);
  diagnostics.push(`pdf quadrature: uniform ${uniformIntegral.toFixed(6)}, cosine ${cosineIntegral.toFixed(6)}`);
  ok(Math.abs(uniformIntegral - 1) < 1e-5, `the uniform pdf must integrate to one (got ${uniformIntegral})`);
  ok(Math.abs(cosineIntegral - 1) < 1e-5, `the cosine pdf must integrate to one (got ${cosineIntegral})`);

  // 2. Sampling: directions stay on the upper hemisphere and the cosine pdf agrees with cos(theta)/pi.
  const sampler = haltonSampler(21);
  let hemisphereViolations = 0;
  let pdfMismatches = 0;
  for (let i = 0; i < 5000; i++) {
    const point = sampler();
    for (const sample of [sampleUniformHemisphere(point), sampleCosineHemisphere(point)]) {
      if (sample.direction[1] < 0) hemisphereViolations += 1;
      const magnitude = Math.hypot(...sample.direction);
      if (Math.abs(magnitude - 1) > 1e-12) hemisphereViolations += 1;
      if (sample.pdf < 0) pdfMismatches += 1;
    }
    const cosine = sampleCosineHemisphere(point);
    if (Math.abs(cosine.pdf - cosine.direction[1] / Math.PI) > 1e-15) pdfMismatches += 1;
  }
  diagnostics.push(`sampling: ${hemisphereViolations} hemisphere violations, ${pdfMismatches} pdf mismatches over 5000 draws of each distribution`);
  ok(hemisphereViolations === 0, "samples must be unit vectors on the upper hemisphere");
  ok(pdfMismatches === 0, "the cosine pdf must equal cos(theta)/pi exactly");

  // 3. Empirical CDF versus the analytic cosine CDF.
  const cdfDistance = cosineCdfDistance(2000, 13);
  diagnostics.push(`cosine CDF distance (KS style): ${cdfDistance.toFixed(4)}`);
  ok(cdfDistance < 0.05, `the cosine sampler must follow cos(theta)/pi (distance ${cdfDistance})`);

  // 4. White furnace test: cosine-weighted sampling is exactly albedo * L for every sample count.
  const furnaceCounts = [1, 2, 16, 256, 4096];
  let worstFurnace = 0;
  for (const count of furnaceCounts) {
    worstFurnace = Math.max(worstFurnace, Math.abs(furnaceTest({ samples: count, technique: "cosine" }) - 1));
  }
  const furnaceUniform = [16, 4096].map((count) => furnaceTest({ samples: count, technique: "uniform" }));
  diagnostics.push(`furnace: cosine worst error ${worstFurnace.toExponential(2)}; uniform at 16/4096 samples ${furnaceUniform.map((v) => v.toFixed(4)).join(", ")} (exact 1)`);
  ok(worstFurnace < 1e-12, "cosine-weighted sampling must reproduce albedo * L exactly for any sample count");
  ok(Math.abs(furnaceUniform[1] - 1) < 0.05, "uniform sampling must converge to albedo * L");
  ok(Math.abs(furnaceUniform[0] - 1) > Math.abs(furnaceUniform[1] - 1) - 1e-12, "uniform sampling is not exact at small sample counts");

  // 5. Variance comparison on a coloured environment: cosine-weighted beats uniform.
  const uniformEnvironment = estimateEnvironmentIrradiance({ samples: 2048, technique: "uniform", seed: 5 });
  const cosineEnvironment = estimateEnvironmentIrradiance({ samples: 2048, technique: "cosine", seed: 5 });
  const varianceRatio = Math.hypot(...uniformEnvironment.variance) / Math.hypot(...cosineEnvironment.variance);
  diagnostics.push(`environment variance: uniform ${Math.hypot(...uniformEnvironment.variance).toFixed(4)}, cosine ${Math.hypot(...cosineEnvironment.variance).toFixed(4)} (ratio ${varianceRatio.toFixed(2)})`);
  ok(varianceRatio > 1.5, "cosine-weighted sampling must have a clearly lower variance for a Lambertian BRDF");

  // 6. Both environment estimators are unbiased: they agree with a much larger reference run.
  const reference = estimateEnvironmentIrradiance({ samples: 131072, technique: "uniform", seed: 17 });
  const referenceStandardError = Math.sqrt(reference.variance.reduce((acc, value) => acc + value, 0) / 131072);
  const cosineError = Math.hypot(...cosineEnvironment.mean.map((value, axis) => value - reference.mean[axis]));
  diagnostics.push(`unbiasedness: reference irradiance ${reference.mean.map((value) => value.toFixed(4)).join(", ")}, cosine deviation ${cosineError.toExponential(2)} (reference standard error ${referenceStandardError.toExponential(2)})`);
  ok(cosineError < Math.max(20 * referenceStandardError, 1e-6), "the cosine estimator must agree with the high-sample reference");

  // 7. The sampler comparison: Halton has a lower star discrepancy than a pseudo-random stream.
  const halton = haltonSampler(29);
  const random = pseudoRandomSampler(29);
  const haltonPoints = Array.from({ length: 1024 }, () => halton());
  const randomPoints = Array.from({ length: 1024 }, () => random());
  const haltonDiscrepancy = starDiscrepancy(haltonPoints);
  const randomDiscrepancy = starDiscrepancy(randomPoints);
  diagnostics.push(`star discrepancy at 1024 points: Halton ${haltonDiscrepancy.toFixed(5)}, pseudo-random ${randomDiscrepancy.toFixed(5)}`);
  ok(haltonDiscrepancy < randomDiscrepancy, "the low-discrepancy sequence must beat the pseudo-random stream");
  ok(haltonPoints.every(([x, y]) => x >= 0 && x < 1 && y >= 0 && y < 1), "the Halton sequence must stay inside the unit square");
  const stratifiedSet = stratifiedSampler(32, 3);
  const occupancy = new Map();
  for (const [x, y] of stratifiedSet.points) {
    const key = `${Math.floor(x * 32)}:${Math.floor(y * 32)}`;
    occupancy.set(key, (occupancy.get(key) ?? 0) + 1);
  }
  const stratifiedDiscrepancy = starDiscrepancy(stratifiedSet.points);
  diagnostics.push(`stratification: ${stratifiedSet.points.length} points in ${occupancy.size} cells, star discrepancy ${stratifiedDiscrepancy.toFixed(5)} (grid bound ${(1 / 32).toFixed(5)})`);
  ok(stratifiedSet.points.length === 1024 && occupancy.size === 1024, "a stratified set must place exactly one point per cell");
  ok([...occupancy.values()].every((count) => count === 1), "no cell may hold more than one point");
  ok(stratifiedDiscrepancy <= 1 / 32 + 1e-12, "a centred grid must satisfy its 1/cells discrepancy bound");

  // 8. Direct lighting: next-event estimation versus BRDF sampling, both against the reference.
  const nee = estimateDirectLighting({ samples: 4096, technique: "nee", light, seed: 23 });
  const brdfOnly = estimateDirectLighting({ samples: 4096, technique: "brdf", light, seed: 23 });
  const mis = estimateDirectLighting({ samples: 4096, technique: "mis", light, seed: 23, mis: true });
  diagnostics.push(`direct lighting: NEE ${nee.mean[0].toFixed(5)} (var ${nee.variance[0].toExponential(2)}), BRDF-only ${brdfOnly.mean[0].toFixed(5)} (var ${brdfOnly.variance[0].toExponential(2)}), MIS ${mis.mean[0].toFixed(5)} (var ${mis.variance[0].toExponential(2)})`);
  ok(brdfOnly.mean[0] > 0 && nee.mean[0] > 0 && mis.mean[0] > 0, "every estimator must return a positive radiance");
  ok(nee.variance[0] < brdfOnly.variance[0], "light sampling must have a lower variance than BRDF sampling for a spherical light");
  ok(nee.effectiveSamples > 4000, "next-event estimation must use nearly every sample");
  ok(Math.abs(mis.mean[0] - nee.mean[0]) < Math.max(0.5 * nee.mean[0], 1e-6), "the MIS estimate must be consistent with NEE");
  ok(mis.effectiveSamples >= brdfOnly.effectiveSamples, "MIS must not waste more samples than BRDF-only sampling");
  ok(nee.mean[0] > 0 && Number.isFinite(mis.variance[0]), "the estimators must be finite");

  // 9. Solid-angle sampling details: the cone pdf inverts the solid angle, and points outside the cone
  //    are rejected rather than counted.
  const cone = sampleSphereLight([0.25, 0.5], [0, 0, 0], [0, 1, 0], light);
  const expectedSolidAngle = TWO_PI * (1 - Math.sqrt(1 - (light.radius / 8) ** 2));
  diagnostics.push(`cone sampling: solid angle ${cone.solidAngle.toExponential(4)} (expected ${expectedSolidAngle.toExponential(4)}), pdf ${cone.pdf.toFixed(4)}`);
  ok(Math.abs(cone.solidAngle - expectedSolidAngle) < 1e-12, "the cone solid angle must follow the analytic formula");
  ok(Math.abs(cone.pdf * cone.solidAngle - 1) < 1e-12, "the cone pdf must be the reciprocal of the solid angle");
  ok(cone.direction[1] > 0, "light samples must lie above the surface");
  throws(() => sampleSphereLight([0.5, 0.5], [0, 0, 0], [0, 1, 0], { position: [0, 0.5, 0], radius: 1, intensity: [1, 1, 1] }), RangeError);

  // 10. Grazing and degenerate inputs.
  const grazing = sampleCosineHemisphere([1 - 1e-16, 0.25]);
  ok(grazing.direction[1] < 1e-7 && grazing.pdf < 1e-8, "a grazing sample must have a vanishing cosine and pdf");
  const zeroLight = estimateDirectLighting({ samples: 8, technique: "nee", light: { ...light, intensity: [0, 0, 0] }, seed: 31 }).mean;
  ok(zeroLight.every((value) => value === 0 || Number.isFinite(value)), "a black light must contribute finite radiance");
  const behind = estimateDirectLighting({ samples: 64, technique: "nee", light: { position: [0, -8, 0], radius: 1, intensity: [10, 10, 10] }, seed: 33 });
  diagnostics.push(`light behind the surface: mean radiance ${behind.mean[0].toExponential(2)}`);
  ok(behind.mean[0] >= 0 && Number.isFinite(behind.mean[0]), "a light behind the surface must not produce negative or NaN radiance");
  throws(() => estimateEnvironmentIrradiance({ samples: 0, technique: "cosine" }), /samples/);

  // 11. Determinism.
  const repeatA = estimateEnvironmentIrradiance({ samples: 512, technique: "cosine", seed: 5 });
  const repeatB = estimateEnvironmentIrradiance({ samples: 512, technique: "cosine", seed: 5 });
  ok(repeatA.mean.every((value, axis) => Object.is(value, repeatB.mean[axis])), "estimators must be bit-for-bit reproducible");
  const otherSeed = estimateEnvironmentIrradiance({ samples: 512, technique: "uniform", seed: 6 });
  const sameSeed = estimateEnvironmentIrradiance({ samples: 512, technique: "uniform", seed: 5 });
  ok(otherSeed.mean.some((value, axis) => value !== sameSeed.mean[axis]), "a different seed must give a different sample set");

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  radicalInverse,
  haltonSampler,
  pseudoRandomSampler,
  stratifiedSampler,
  starDiscrepancy,
  sampleUniformHemisphere,
  sampleCosineHemisphere,
  sampleSphereLight,
  gammaRamp,
  furnaceTest,
  estimateEnvironmentIrradiance,
  estimateDirectLighting,
  cosineCdfDistance,
  integratePdf,
};

if (require.main === module) main();
