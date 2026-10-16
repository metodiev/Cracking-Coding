/**
 * Task 573 - Implement geometric hashing for object recognition small demo. (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * Geometric hashing: for every ordered basis pair (b1, b2) of a model and every remaining model
 * point p, store the coordinates of p in the affine frame spanned by the basis,
 * u = ((p - b1) . (b2 - b1)) / |b2 - b1|^2 and v = cross(b2 - b1, p - b1) / |b2 - b1|^2, quantised
 * into a bin grid.  Because u and v are dimensionless ratios inside the basis frame they are exactly
 * invariant to translation, rotation *and* scale; a reflected copy flips the sign of v, so the
 * scheme deliberately distinguishes chirality.  Recognition works the other way round: for every
 * ordered pair of scene points, every other scene point is quantised and used to look up the table,
 * and the (model, basis) pair collecting the most votes wins; that basis then defines the similarity
 * transform from model to scene, which is used to verify the match by transferring all model points.
 *
 * Documented checks: every model contributes exactly n(n-1)(n-2) entries; the basis coordinates are
 * asserted invariant to rotation by 0.7 rad, translation by (13, -7), scale 2.5 and any composition
 * (to 1e-9), while mirroring flips v and therefore changes the quantised key; a noisy, scaled,
 * rotated scene of a six-point model is recognised with at least n - 2 votes for the correct basis
 * and the recovered transform transfers every model point to within 0.02 (the noise level is 0.001
 * against a bin size of 0.05); a two-model table keeps them apart (the wrong model's best score is
 * asserted to be strictly smaller); degenerate bases with coincident points are skipped instead of
 * dividing by zero; the 3x3 neighbourhood lookup is asserted to collect at least as many votes as the
 * exact-bin lookup (and, on the shipped noisy scene, to beat it); adding three unrelated scene points
 * does not disturb the winner; and an unrelated random scene produces a much lower best score.
 * Randomised quantities come from a documented 64-bit splitmix64 generator rather than the repo's
 * 16-bit LCG, per the wave-3 note on that LCG's short cycles and low-bit structure.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;

/** Documented 64-bit splitmix64 generator (long period, well-mixed low and high bits). */
class Rng {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    const z = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    this.state = z;
    let x = z;
    x = ((x ^ (x >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    x = ((x ^ (x >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return x ^ (x >> 31n);
  }

  nextFloat() {
    return Number(this.next() >> 11n) / 9007199254740992.0;
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

/** Coordinates of p in the affine frame (b1; b2 - b1, perpendicular). */
function basisCoordinates(b1, b2, p) {
  const dx = b2[0] - b1[0];
  const dy = b2[1] - b1[1];
  const norm2 = dx * dx + dy * dy;
  if (norm2 === 0) return null; // coincident basis points carry no frame
  const px = p[0] - b1[0];
  const py = p[1] - b1[1];
  // v is the signed cross product of the basis direction with the offset, so the frame
  // (b2 - b1, perpendicular) is right-handed and a reflection negates v.
  return [(px * dx + py * dy) / norm2, (dx * py - dy * px) / norm2];
}

function binKey(u, v, binSize) {
  return `${Math.floor(u / binSize)},${Math.floor(v / binSize)}`;
}

function quantize(u, v, binSize) {
  return [Math.floor(u / binSize), Math.floor(v / binSize)];
}

/** Translate, rotate by `angle` and scale about the origin, then translate by (tx, ty). */
function transformPoints(points, angle, tx, ty, scale) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return points.map(([x, y]) => [scale * (cos * x - sin * y) + tx, scale * (sin * x + cos * y) + ty]);
}

/** Reflection about the x-axis: this is what geometric hashing is not invariant to. */
function mirrorPoints(points) {
  return points.map(([x, y]) => [x, -y]);
}

/** Hash table: quantised (u, v) -> list of (model, ordered basis, point) entries. */
function buildTable(models, binSize) {
  const table = new Map();
  let entries = 0;
  for (const model of models) {
    const points = model.points;
    for (let i = 0; i < points.length; i++) {
      for (let j = 0; j < points.length; j++) {
        if (i === j) continue;
        for (let k = 0; k < points.length; k++) {
          if (k === i || k === j) continue;
          const coordinates = basisCoordinates(points[i], points[j], points[k]);
          if (coordinates === null) continue;
          const key = binKey(coordinates[0], coordinates[1], binSize);
          if (!table.has(key)) table.set(key, []);
          table.get(key).push({ model: model.name, basis: [i, j], point: k });
          entries += 1;
        }
      }
    }
  }
  return { table, entries };
}

/**
 * Recognise the scene.  Votes are accumulated per (model, model basis, scene basis) triple, so the
 * winning triple already contains a two-point correspondence.  `tolerant` also inspects the eight
 * neighbouring bins (the standard remedy at quantisation boundaries).  When `verifyTolerance` is a
 * number the candidates are re-ranked by how many of the model's points the induced similarity
 * transform actually transfers onto a scene point - the verification step that suppresses the false
 * positives an accumulator alone produces.  Returns the ranked candidates, best first.
 */
function recognize(models, scene, binSize, tolerant = false, verifyTolerance = null) {
  const { table, entries } = buildTable(models, binSize);
  const scores = new Map();
  const meta = new Map();
  const offsets = tolerant
    ? [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]
    : [[0, 0]];
  for (let i = 0; i < scene.length; i++) {
    for (let j = 0; j < scene.length; j++) {
      if (i === j) continue;
      for (let k = 0; k < scene.length; k++) {
        if (k === i || k === j) continue;
        const coordinates = basisCoordinates(scene[i], scene[j], scene[k]);
        if (coordinates === null) continue;
        const [bu, bv] = quantize(coordinates[0], coordinates[1], binSize);
        for (const [du, dv] of offsets) {
          const bucket = table.get(`${bu + du},${bv + dv}`);
          if (!bucket) continue;
          for (const entry of bucket) {
            const id = `${entry.model}|${entry.basis[0]},${entry.basis[1]}|${i},${j}`;
            scores.set(id, (scores.get(id) || 0) + 1);
            if (!meta.has(id)) {
              meta.set(id, { model: entry.model, basis: entry.basis, sceneBasis: [i, j] });
            }
          }
        }
      }
    }
  }
  const byName = new Map(models.map((model) => [model.name, model]));
  const ranked = [...scores.entries()].map(([id, votes]) => {
    const info = meta.get(id);
    const record = { model: info.model, basis: info.basis, sceneBasis: info.sceneBasis, votes,
      matched: null, worst: null, transform: null };
    if (verifyTolerance !== null) {
      const transform = similarityFromBasis(byName.get(info.model).points, scene, info.basis[0],
        info.basis[1], info.sceneBasis[0], info.sceneBasis[1]);
      if (transform === null) {
        record.matched = -1;
        record.worst = Infinity;
      } else {
        const checked = transferScore(transform, byName.get(info.model).points, scene,
          verifyTolerance);
        record.matched = checked.matched;
        record.worst = checked.worst;
        record.transform = transform;
      }
    }
    return record;
  }).sort((a, b) => {
    if (verifyTolerance !== null && a.matched !== b.matched) return b.matched - a.matched;
    if (a.votes !== b.votes) return b.votes - a.votes;
    if (a.model !== b.model) return a.model < b.model ? -1 : 1;
    if (a.basis[0] !== b.basis[0]) return a.basis[0] - b.basis[0];
    return a.basis[1] - b.basis[1];
  });
  return { ranked, tableEntries: entries, tableSize: table.size, best: ranked[0] || null,
    scoreOf: (model, basis) => {
      let total = 0;
      for (const [id, votes] of scores) if (id.startsWith(`${model}|${basis[0]},${basis[1]}|`)) total = Math.max(total, votes);
      return total;
    } };
}

/** Similarity transform mapping model[i] -> scene[a] and model[j] -> scene[b]. */
function similarityFromBasis(modelPoints, scenePoints, i, j, a, b) {
  const mdx = modelPoints[j][0] - modelPoints[i][0];
  const mdy = modelPoints[j][1] - modelPoints[i][1];
  const sdx = scenePoints[b][0] - scenePoints[a][0];
  const sdy = scenePoints[b][1] - scenePoints[a][1];
  const modelNorm2 = mdx * mdx + mdy * mdy;
  if (modelNorm2 === 0) return null;
  const scale = Math.sqrt((sdx * sdx + sdy * sdy) / modelNorm2);
  const angle = Math.atan2(sdy, sdx) - Math.atan2(mdy, mdx);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const tx = scenePoints[a][0] - scale * (cos * modelPoints[i][0] - sin * modelPoints[i][1]);
  const ty = scenePoints[a][1] - scale * (sin * modelPoints[i][0] + cos * modelPoints[i][1]);
  return { scale, angle, tx, ty };
}

function applySimilarity(transform, point) {
  const cos = Math.cos(transform.angle);
  const sin = Math.sin(transform.angle);
  return [transform.scale * (cos * point[0] - sin * point[1]) + transform.tx,
    transform.scale * (sin * point[0] + cos * point[1]) + transform.ty];
}

/** How many model points transfer onto some scene point within `tolerance`. */
function transferScore(transform, modelPoints, scenePoints, tolerance) {
  let matched = 0;
  let worst = 0;
  for (const point of modelPoints) {
    const mapped = applySimilarity(transform, point);
    let best = Infinity;
    for (const target of scenePoints) {
      best = Math.min(best, Math.hypot(mapped[0] - target[0], mapped[1] - target[1]));
    }
    if (best <= tolerance) matched += 1;
    worst = Math.max(worst, best);
  }
  return { matched, worst };
}

const MODEL_L = { name: "L", points: [[0.0, 0.0], [2.0, 0.0], [2.0, 0.5], [0.5, 0.5], [0.5, 2.0],
  [0.0, 2.0]] };
const MODEL_T = { name: "T", points: [[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [1.0, 1.6], [0.9, 1.6],
  [1.1, 1.6]] };
// The "L" and "T" shapes happen to be mirror-symmetric (their mirror image is a rotation of
// themselves), so chirality is demonstrated with this genuinely chiral "F" shape instead.
const MODEL_F = { name: "F", points: [[0.0, 0.0], [0.0, 3.0], [1.5, 3.0], [1.5, 2.0], [0.6, 2.0],
  [0.6, 1.0], [1.2, 1.0], [1.2, 0.0]] };
const MODELS = [MODEL_L, MODEL_T];

function main() {
  const assert = require("node:assert");
  const rng = new Rng(0x5eed);
  const binSize = 0.05;

  // (1) The hash coordinates are invariant to rotation, translation and scale, and chirality-
  //     sensitive: mirroring flips v.
  {
    const model = MODEL_L.points;
    const moved = transformPoints(model, 0.7, 13.0, -7.0, 2.5);
    for (let i = 0; i < model.length; i++) {
      for (let j = 0; j < model.length; j++) {
        if (i === j) continue;
        for (let k = 0; k < model.length; k++) {
          if (k === i || k === j) continue;
          const original = basisCoordinates(model[i], model[j], model[k]);
          const transformed = basisCoordinates(moved[i], moved[j], moved[k]);
          assert.ok(Math.abs(original[0] - transformed[0]) <= 1e-9,
            `u is invariant (${original[0]} vs ${transformed[0]})`);
          assert.ok(Math.abs(original[1] - transformed[1]) <= 1e-9,
            `v is invariant (${original[1]} vs ${transformed[1]})`);
        }
      }
    }
    assert.deepStrictEqual(basisCoordinates([0, 0], [1, 0], [0, 1]), [0, 1]);
    assert.deepStrictEqual(basisCoordinates([0, 0], [1, 0], [3, -2]), [3, -2]);
    assert.deepStrictEqual(basisCoordinates([2, 2], [4, 2], [3, 5]), [0.5, 1.5]);
    assert.deepStrictEqual(basisCoordinates([1, 1], [2, 2], [1, 2]), [0.5, 0.5]);
    const mirrored = mirrorPoints(model);
    const flip = basisCoordinates(mirrored[0], mirrored[1], mirrored[2]);
    const straight = basisCoordinates(model[0], model[1], model[2]);
    assert.ok(Math.abs(flip[0] - straight[0]) <= 1e-12, "mirroring keeps u");
    assert.ok(Math.abs(flip[1] + straight[1]) <= 1e-12, "and negates v: the hash senses chirality");
    assert.notStrictEqual(binKey(flip[0], flip[1], binSize), binKey(straight[0], straight[1], binSize));
    assert.strictEqual(basisCoordinates([1, 1], [1, 1], [5, 5]), null,
      "coincident basis points carry no frame");
  }

  // (2) The table has exactly n(n - 1)(n - 2) entries per model, and the ordered-pair count is
  //     checked against a closed form.
  {
    const { entries, table } = buildTable(MODELS, binSize);
    const tableSize = table.size;
    let expected = 0;
    for (const model of MODELS) {
      const n = model.points.length;
      expected += n * (n - 1) * (n - 2);
    }
    assert.strictEqual(entries, expected, "every ordered basis and third point is inserted");
    assert.strictEqual(expected, 6 * 5 * 4 * 2);
    assert.ok(tableSize > 0 && tableSize <= entries, "several entries share a bin (injectivity fails)");
    assert.ok(tableSize >= 0.5 * entries, "and the keys are far from all-colliding");
    const degenerate = buildTable([{ name: "flat", points: [[1, 1], [1, 1], [1, 1]] }], binSize);
    assert.strictEqual(degenerate.entries, 0, "all-degenerate models insert nothing");
  }

  // (3) A noisy, scaled, rotated, translated scene of the L model is recognised and verified.
  {
    const truth = transformPoints(MODEL_L.points, -1.1, -4.0, 9.0, 2.5);
    const scene = truth.map(([x, y]) => [x + 0.001 * (rng.nextFloat() - 0.5),
      y + 0.001 * (rng.nextFloat() - 0.5)]);
    const result = recognize(MODELS, scene, binSize, false, 0.02);
    assert.strictEqual(result.best.model, "L", "the correct model wins");
    assert.ok(result.best.votes >= 4,
      `the winning triple collects at least n - 2 = 4 votes (measured ${result.best.votes})`);
    assert.strictEqual(result.best.matched, 6, "and verification transfers all six model points");
    assert.ok(result.best.worst <= 0.02, `worst transfer error ${result.best.worst}`);
    // The basis transfer is only as accurate as the noise on those two scene points (+/- 0.0005).
    assert.ok(Math.abs(result.best.transform.scale - 2.5) <= 0.01,
      `the recovered scale is ${result.best.transform.scale}`);
    assert.ok(Math.abs(result.best.transform.angle - (-1.1)) <= 0.01,
      `the recovered rotation is ${result.best.transform.angle} rad`);
    assert.deepStrictEqual(recognize(MODELS, scene, binSize, false, 0.02).ranked, result.ranked,
      "recognition is deterministic");
    const withoutVerification = recognize(MODELS, scene, binSize);
    assert.strictEqual(withoutVerification.best.matched, null,
      "the vote-only ranking carries no verification field");
    assert.strictEqual(withoutVerification.best.model, "L");
  }

  // (4) Two models: a scene of T is attributed to T, and the wrong model is verified out.
  {
    const scene = transformPoints(MODEL_T.points, 0.4, 3.0, 3.0, 1.0);
    const verified = recognize(MODELS, scene, binSize, false, 0.02);
    assert.strictEqual(verified.best.model, "T");
    assert.strictEqual(verified.best.matched, 6);
    assert.ok(verified.best.votes >= 4, "the winning basis collects at least n - 2 votes");
    let bestL = 0;
    for (const entry of verified.ranked) if (entry.model === "L") bestL = Math.max(bestL, entry.votes);
    assert.ok(bestL < verified.best.votes, `L's best score ${bestL} < T's ${verified.best.votes}`);
    const wrong = similarityFromBasis(MODEL_L.points, scene, 0, 1, 0, 1);
    assert.ok(transferScore(wrong, MODEL_L.points, scene, 0.02).matched <= 2,
      "and the L model cannot be transferred onto the scene");
  }

  // (5) Rotation, translation and scale invariance of the recognition itself.
  {
    for (const [angle, tx, ty, scale] of [[0.0, 0.0, 0.0, 1.0], [2.0, -20.0, 5.0, 0.5],
      [-0.3, 100.0, -100.0, 10.0]]) {
      const scene = transformPoints(MODEL_L.points, angle, tx, ty, scale);
      const result = recognize(MODELS, scene, binSize, false, 0.02);
      assert.strictEqual(result.best.model, "L",
        `recognised after rotation ${angle}, translation (${tx}, ${ty}), scale ${scale}`);
      assert.strictEqual(result.best.matched, 6, "with all points transferred");
      assert.ok(Math.abs(result.best.transform.scale - scale) <= 1e-6,
        `the recovered scale is ${result.best.transform.scale} vs ${scale}`);
    }
    // Chirality, tested on the genuinely chiral F shape (the L shape is mirror-symmetric, so its
    // mirror image is merely a rotation of itself and cannot serve as a reflection test).
    const fScene = transformPoints(MODEL_F.points, 0.3, 5.0, -2.0, 1.7);
    const fResult = recognize([MODEL_F], fScene, binSize, false, 0.02);
    assert.strictEqual(fResult.best.model, "F");
    assert.strictEqual(fResult.best.matched, 8, "the chiral model is verified on its own scene");
    assert.strictEqual(fResult.best.votes, 6);
    assert.ok(fResult.best.worst <= 1e-9, "with an exact transfer");
    const fMirrored = recognize([MODEL_F], mirrorPoints(fScene), binSize, false, 0.02);
    assert.ok(fMirrored.best === null || fMirrored.best.votes <= 2,
      "the mirrored scene collects far fewer votes (2 against 6)");
    assert.ok(fMirrored.best === null || fMirrored.best.matched <= 4,
      "and verification leaves half of the points unmatched (measured 4 of 8)");
    assert.ok(fMirrored.best === null || fMirrored.best.worst > 0.1,
      `with a visible residual (measured ${fMirrored.best === null ? "n/a" : fMirrored.best.worst})`);
  }

  // (6) Quantisation: sub-bin jitter keeps the key, a bigger jitter leaves it, and the tolerant
  //     lookup recovers votes that the exact lookup misses.
  {
    const [u, v] = basisCoordinates(MODEL_L.points[0], MODEL_L.points[1], MODEL_L.points[3]);
    assert.strictEqual(binKey(u + 0.4 * binSize, v + 0.4 * binSize, binSize), binKey(u, v, binSize),
      "a 0.4-bin jitter cannot leave the cell");
    assert.notStrictEqual(binKey(u + 2.5 * binSize, v + 2.5 * binSize, binSize),
      binKey(u, v, binSize), "a 2.5-bin jitter does leave it");
    assert.strictEqual(quantize(0.0, 0.0, binSize).join(","), "0,0");
    assert.strictEqual(quantize(-0.01, -0.01, binSize).join(","), "-1,-1",
      "bins start at the cell containing 0, so negatives are handled consistently");

    const truth = transformPoints(MODEL_L.points, 0.2, 1.0, 2.0, 1.0);
    const scene = truth.map(([x, y], index) => [x + (index % 2 === 0 ? 0.03 : -0.03),
      y + (index % 3 === 0 ? 0.03 : -0.02)]);
    // The jitter (+/- 0.03 on scene points) is larger than the verification tolerance 0.02, so the
    // transfer tolerance is set to 0.1 here: the estimate comes from just two jittered points.
    const exact = recognize(MODELS, scene, binSize, false, 0.1);
    const tolerant = recognize(MODELS, scene, binSize, true, 0.1);
    assert.strictEqual(exact.best.model, "L", "the exact-bin lookup still wins here");
    assert.strictEqual(tolerant.best.model, "L");
    assert.strictEqual(exact.best.matched, 6, "and all six model points still transfer");
    assert.strictEqual(tolerant.best.matched, 6);
    assert.ok(tolerant.best.votes >= exact.best.votes,
      `tolerant lookup ${tolerant.best.votes} votes >= exact ${exact.best.votes}`);
    assert.strictEqual(exact.best.votes, 3, "the jitter costs the exact lookup most of its votes");
    assert.strictEqual(tolerant.best.votes, 4, "the neighbourhood lookup recovers one of them");
  }

  // (7) Unrelated scene points do not disturb the winner; an unrelated scene scores much lower.
  {
    const truth = transformPoints(MODEL_L.points, 0.9, 2.0, 2.0, 1.5);
    const withExtras = truth.concat([[40.0, 41.0], [-30.0, 12.0], [7.5, -9.25]]);
    const result = recognize(MODELS, withExtras, binSize, false, 0.02);
    assert.strictEqual(result.best.model, "L", "three extra scene points do not derail it");
    assert.strictEqual(result.best.matched, 6);
    assert.ok(result.best.votes >= 4);
    // Documented: the extras create false positives, so the raw-vote margin collapses from 4 vs 0
    // to 4 vs 3 (the T model contains a cluster of almost-coincident points that produces many
    // colliding keys), even though verification cleanly separates 6 transferred points from 2.
    const raw = recognize(MODELS, withExtras, binSize);
    assert.strictEqual(raw.best.model, "L");
    assert.strictEqual(raw.best.votes, 4, "the winner keeps its 4 votes");
    let bestRawT = 0;
    let bestVerifiedT = 0;
    for (const entry of raw.ranked) if (entry.model === "T") bestRawT = Math.max(bestRawT, entry.votes);
    for (const entry of result.ranked) {
      if (entry.model === "T") bestVerifiedT = Math.max(bestVerifiedT, entry.matched);
    }
    assert.strictEqual(bestRawT, 3, "while the wrong model climbs to 3 raw votes");
    assert.ok(bestRawT >= raw.best.votes - 1, "leaving only a one-vote margin without verification");
    assert.strictEqual(bestVerifiedT, 2, "verification separates them: 6 against 2 transferred points");

    const unrelated = [];
    for (let i = 0; i < 6; i++) unrelated.push([rng.nextInt(100) / 10.0, rng.nextInt(100) / 10.0]);
    const noiseResult = recognize(MODELS, unrelated, binSize, false, 0.02);
    const scored = noiseResult.best ? noiseResult.best.votes : 0;
    assert.ok(scored <= 2, `an unrelated scene scores at most 2 votes (measured ${scored})`);
    assert.ok((noiseResult.best ? noiseResult.best.matched : 0) <= 3,
      "and verification finds few transferred points");
  }

  // (8) Deterministic generation and helper sanity checks.
  {
    const stream = [];
    const first = new Rng(0x5eed);
    for (let i = 0; i < 12; i++) stream.push(first.nextFloat());
    const replay = [];
    const second = new Rng(0x5eed);
    for (let i = 0; i < 12; i++) replay.push(second.nextFloat());
    assert.deepStrictEqual(replay, stream, "the splitmix64 stream is deterministic");
    assert.notStrictEqual(new Rng(0x5eed + 1).nextFloat(), stream[0],
      "a different seed starts elsewhere");
    assert.strictEqual(new Rng(1).nextInt(100), new Rng(1).nextInt(100));
    assert.deepStrictEqual(transformPoints([[1, 0]], 0, 0, 0, 1), [[1, 0]]);
    assert.ok(Math.abs(transformPoints([[1, 0]], Math.PI / 2, 0, 0, 1)[0][0]) <= 1e-15);
    assert.ok(Math.abs(transformPoints([[1, 0]], Math.PI / 2, 0, 0, 1)[0][1] - 1) <= 1e-15);
    assert.deepStrictEqual(mirrorPoints([[1, 2], [3, 4]]), [[1, -2], [3, -4]]);
    const identity = { scale: 1.0, angle: 0.0, tx: 0.0, ty: 0.0 };
    assert.deepStrictEqual(applySimilarity(identity, [2, 3]), [2, 3]);
    assert.strictEqual(transferScore(identity, [[2, 3]], [[2, 3]], 1e-9).matched, 1);
    assert.strictEqual(transferScore(identity, [[5, 5]], [[2, 3]], 0.5).matched, 0);
    assert.strictEqual(similarityFromBasis([[1, 1], [1, 1]], [[0, 0], [1, 1]], 0, 1, 0, 1), null);
  }
}

module.exports = { MASK64, Rng, basisCoordinates, binKey, quantize, transformPoints, mirrorPoints,
  buildTable, recognize, similarityFromBasis, applySimilarity, transferScore, MODEL_L, MODEL_T,
  MODEL_F, MODELS };

if (require.main === module) {
  main();
  console.log("PASS");
}
