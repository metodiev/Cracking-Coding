/** Task 394 - Implement skyline merging with a multiset (detailed). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
"use strict";

function sweep(events) {
  const byX = new Map();
  for (const [x, delta] of events) {
    if (!byX.has(x)) byX.set(x, []);
    byX.get(x).push(delta);
  }
  const order = [...byX.keys()].sort((a, b) => a - b);
  const counts = new Map();
  const points = [];
  let current = 0;
  for (const x of order) {
    for (const delta of byX.get(x)) {
      if (delta >= 0) counts.set(delta, (counts.get(delta) || 0) + 1);
      else {
        const height = -delta;
        const remaining = counts.get(height) - 1;
        if (remaining === 0) counts.delete(height);
        else counts.set(height, remaining);
      }
    }
    const top = counts.size ? Math.max(...counts.keys()) : 0;
    if (top !== current) {
      points.push([x, top]);
      current = top;
    }
  }
  return points;
}

function skylineOfRectangles(rectangles) {
  const events = [];
  for (const [x1, x2, height] of rectangles) {
    if (x1 < x2 && height > 0) {
      events.push([x1, height]);
      events.push([x2, -height]);
    }
  }
  return sweep(events);
}

function mergeSkylines(skylines) {
  const events = [];
  for (const skyline of skylines) {
    for (let i = 0; i + 1 < skyline.length; i++) {
      const [x1, height] = skyline[i];
      const x2 = skyline[i + 1][0];
      if (height > 0 && x1 < x2) {
        events.push([x1, height]);
        events.push([x2, -height]);
      }
    }
  }
  return sweep(events);
}

function lcgRectangles(count, seed, span = 20) {
  const rectangles = [];
  for (let k = 0; k < count; k++) {
    seed = (seed * 75 + 74) % 65537;
    const x1 = seed % span;
    seed = (seed * 75 + 74) % 65537;
    const length = 1 + (seed % 6);
    seed = (seed * 75 + 74) % 65537;
    const height = 1 + (seed % 10);
    rectangles.push([x1, x1 + length, height]);
  }
  return rectangles;
}

const key = (points) => points.map((p) => p.join(",")).join(";");

module.exports = { sweep, skylineOfRectangles, mergeSkylines, lcgRectangles };

if (require.main === module) {
  const assert = require("node:assert");
  const canonical = [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]];
  const expected = "2,10;3,15;7,12;12,0;15,10;20,8;24,0";
  assert.strictEqual(key(skylineOfRectangles(canonical)), expected);

  assert.strictEqual(key(mergeSkylines([[[0, 3], [4, 0]], [[1, 1], [2, 4], [5, 0]]])), "0,3;2,4;5,0");
  const expectedPoints = skylineOfRectangles(canonical);
  assert.strictEqual(key(mergeSkylines([expectedPoints, expectedPoints])), expected);

  assert.strictEqual(key(skylineOfRectangles([[0, 2, 3], [2, 4, 3]])), "0,3;4,0");
  assert.strictEqual(key(skylineOfRectangles([[0, 10, 5], [2, 5, 3]])), "0,5;10,0");
  assert.deepStrictEqual(skylineOfRectangles([]), []);

  let seed = 777;
  for (let trial = 0; trial < 40; trial++) {
    const rectangles = lcgRectangles(9, seed);
    seed = (seed * 75 + 74) % 65537;
    const cut = seed % (rectangles.length + 1);
    const combined = key(skylineOfRectangles(rectangles));
    const merged = key(mergeSkylines([skylineOfRectangles(rectangles.slice(0, cut)), skylineOfRectangles(rectangles.slice(cut))]));
    assert.strictEqual(merged, combined, `split ${trial}`);
  }

  assert.deepStrictEqual(skylineOfRectangles([[3, 1, 5]]), []);
  assert.deepStrictEqual(skylineOfRectangles([[0, 5, 0]]), []);
  console.log("PASS");
}
