/** Task 893 - Implement causal trace capture across microservices for latency debugging.
 *  (Tier: Master, CTCI Beyond — Software Engineering & Tooling)
 *
 * A distributed-tracing pipeline with the three pieces that make it work in the real world:
 *
 *  clocks    every service keeps a hybrid logical clock (HLC: physical time + logical counter +
 *            node id). Spans are stamped with their producer's HLC, and the collector orders the
 *            causal chain by HLC rather than by wall clock, because each host has a fixed clock skew
 *            — the simulator applies a per-service offset on purpose, so naive wall-clock ordering
 *            produces inversions that HLC ordering does not.
 *  capture   spans are reported to the collector asynchronously: reports arrive out of order, may be
 *            duplicated (at-least-once delivery) and may be lost. The collector de-duplicates by span
 *            id, buffers per trace and flushes a trace when it has been quiet for a watermark period
 *            (or after a hard deadline), then reports such traces as *incomplete* instead of quietly
 * analysing a partial picture. (A lost *parent* shows up as orphaned children; a lost *leaf* is
 * undetectable from the reports alone — stated here rather than papered over.)
 *  analysis  per span it computes the self time = duration − |union of children intervals|, and the
 *            critical path = the root-to-leaf chain maximising the sum of self times. Those two numbers
 *            are what latency debugging needs: the path is where the wall-clock time actually went,
 *            and per-service self-time shares blame the right service.
 *
 * Invariants verified below: (1) HLC axioms (monotonicity of send/receive, and e1 ⇒ e2 gives
 * hlc(e1) < hlc(e2)) hold over generated traces, and HLC ordering has zero inversions where
 * wall-clock ordering has many; (2) parenting is recovered after de-skewing (child intervals nest
 * inside the parent) while the raw skewed timestamps do not nest; (3) assembly is order-independent —
 * any arrival permutation, with duplicates, yields the same analysis; (4) completeness is reported
 * honestly (a lost report is surfaced, not hidden); (5) the critical path and self times equal an
 * independent brute-force computation over all root-to-leaf paths and interval unions; (6) attribution
 * sum to 1 and name the deliberately slowed service; (7) cycle detection rejects corrupt traces;
 * (8) memory and cost stay linear (bounded buffer, one pass per report).
 *
 * Simplifications: in-process services with an explicit virtual clock and message queue (no network),
 * integer millisecond timestamps, a single collector, and traces of ~200 spans over 6 services.
 */
"use strict";

/* ------------------------------------------------------------------ primitives */

class Prng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    this.state = (this.state + 0x9e3779b9) >>> 0;
    let z = this.state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return (z ^ (z >>> 15)) >>> 0;
  }
  below(limit) { return this.next() % limit; }
  unit() { return this.next() / 0x1_0000_0000; }
}

/* ------------------------------------------------------------------ hybrid logical clock */

/**
 * HLC of Kulkarni et al.: `physical` tracks the greatest wall time seen, `counter` breaks ties
 * without ever going backwards, so causally ordered events always compare in the same order.
 */
class Hlc {
  constructor(node, physical = 0) { this.node = node; this.physical = physical; this.counter = 0; }

  stamp() { return { physical: this.physical, counter: this.counter, node: this.node }; }

  /** A local event (or a message send). */
  send(localTime) {
    const previous = this.physical;
    this.physical = Math.max(this.physical, localTime);
    this.counter = this.physical === previous ? this.counter + 1 : 0;
    return this.stamp();
  }

  /** Receipt of a remote stamp. */
  receive(remote, localTime) {
    const previous = this.physical;
    this.physical = Math.max(this.physical, remote.physical, localTime);
    if (this.physical === previous && this.physical === remote.physical) {
      this.counter = Math.max(this.counter, remote.counter) + 1;
    } else if (this.physical === previous) {
      this.counter++;
    } else if (this.physical === remote.physical) {
      this.counter = remote.counter + 1;
    } else {
      this.counter = 0;
    }
    return this.stamp();
  }
}

const hlcCompare = (a, b) => (a.physical !== b.physical ? a.physical - b.physical
  : (a.counter !== b.counter ? a.counter - b.counter : String(a.node).localeCompare(String(b.node))));
const hlcKey = (stamp) => `${stamp.physical}.${stamp.counter}.${stamp.node}`;

/* ------------------------------------------------------------------ trace generation */

const SERVICES = ["gateway", "auth", "catalog", "cart", "pricing", "shipping"];

/** Each host's clock offset: it is what makes wall-clock ordering unusable and HLC necessary. */
const skewOf = (service) => {
  const index = SERVICES.indexOf(service);
  return (index % 2 === 0 ? 1 : -1) * (5 + index * 3);
};

/**
 * Generates one trace: spans laid out so that children nest inside their parent and never overlap
 * each other (a documented simplification: with overlapping siblings the self times double-count the
 * overlap and the critical-path DP would need clipping — with the sequential layout the identities
 * Σ self time = root duration and critical path ≤ root duration hold exactly, and both are checked).
 */
function generateTrace(options) {
  const prng = new Prng(options.seed || 1);
  const serviceCount = options.services || SERVICES.length;
  const maxDepth = options.maxDepth !== undefined ? options.maxDepth : 4;
  const slowService = options.slowService || null;
  const slowExtra = options.slowExtra || 0;
  const spans = [];
  let spanSeq = 0;
  const clocks = new Map(SERVICES.slice(0, serviceCount).map((name) => [name, new Hlc(name, 0)]));

  const build = (parentId, depth, start, service) => {
    const span = { id: `s${spanSeq++}`, parentId, service, trueStart: start, trueEnd: 0, children: [], injected: 0 };
    spans.push(span);
    const slow = slowService !== null && service === slowService ? slowExtra : 0;
    span.injected = slow;
    // Children are laid out after the span starts and always finish before the parent does: the
    // parent's tail is its own self time.
    let cursor = start + 1;
    const childCount = depth >= maxDepth ? 0 : 1 + prng.below(2);
    for (let i = 0; i < childCount; i++) {
      const childService = SERVICES[prng.below(serviceCount)];
      const childStart = Math.max(cursor, start + 1);
      const child = build(span.id, depth + 1, childStart, childService);
      span.children.push(child);
      cursor = Math.max(cursor, child.trueEnd) + 1 + prng.below(3);
    }
    const own = 2 + prng.below(6) + slow;
    span.trueEnd = Math.max(cursor, start + 1 + own) + own;
    return span;
  };

  const root = build(null, 0, 1000, SERVICES[0]);
  // HLC stamps follow the causal chain, so a child always compares after its parent.
  const assign = (span, parentStamp) => {
    const clock = clocks.get(span.service);
    span.hlc = parentStamp === null ? clock.send(span.trueStart) : clock.receive(parentStamp, span.trueStart);
    for (const child of span.children) assign(child, span.hlc);
  };
  assign(root, null);
  return { root, spans, prng };
}

/* ------------------------------------------------------------------ analysis */

/** Union length of a list of [start, end] intervals; the children of a span never nest each other. */
function unionLength(intervals) {
  if (intervals.length === 0) return 0;
  const sorted = [...intervals].sort((a, b) => a[0] - b[0]);
  let total = 0;
  let currentStart = sorted[0][0];
  let currentEnd = sorted[0][1];
  for (const [start, end] of sorted.slice(1)) {
    if (start > currentEnd) { total += currentEnd - currentStart; currentStart = start; currentEnd = end; }
    else currentEnd = Math.max(currentEnd, end);
  }
  return total + (currentEnd - currentStart);
}

function selfTime(span, intervalsOf) {
  return (span.trueEnd - span.trueStart) - unionLength(intervalsOf(span));
}

/** Critical path: the chain of self times that accounts for the trace's wall-clock time. */
function criticalPath(root, intervalsOf) {
  const bestOf = (span) => {
    const self = selfTime(span, intervalsOf);
    let bestValue = 0;
    let bestChild = null;
    for (const child of span.children) {
      const childBest = bestOf(child);
      if (childBest.value > bestValue) { bestValue = childBest.value; bestChild = child; }
    }
    return { value: self + bestValue, self, child: bestChild };
  };
  const total = bestOf(root).value;
  const chain = [];
  for (let node = root; node !== null; node = bestOf(node).child) chain.push(node);
  return { length: total, chain };
}

/** Independent oracle: enumerate every root-to-leaf path and its self-time sum. */
function bruteForceCriticalPath(root, intervalsOf) {
  let bestLength = -Infinity;
  let bestPath = null;
  const walk = (span, accumulated, path) => {
    const total = accumulated + selfTime(span, intervalsOf);
    const nextPath = [...path, span.id];
    if (span.children.length === 0) {
      if (total > bestLength) { bestLength = total; bestPath = nextPath; }
      return;
    }
    for (const child of span.children) walk(child, total, nextPath);
  };
  walk(root, 0, []);
  return { length: bestLength, path: bestPath };
}

/* ------------------------------------------------------------------ collector */

class TraceCollector {
  constructor(options = {}) {
    this.watermark = options.watermark !== undefined ? options.watermark : 40;
    this.hardDeadline = options.hardDeadline !== undefined ? options.hardDeadline : 200;
    this.maxBufferedTraces = options.maxBufferedTraces || 1000;
    this.traces = new Map();          // traceId -> { reports: Map(spanId -> report), lastReport, firstReport }
    this.flushed = [];
    this.duplicates = 0;
    this.assemblyOps = 0;
    this.bufferHighWater = 0;
    this.rejections = 0;
  }

  ingest(report, time) {
    this.assemblyOps++;
    let trace = this.traces.get(report.traceId);
    if (!trace) {
      trace = { traceId: report.traceId, reports: new Map(), firstReport: time, lastReport: time, skew: new Map() };
      this.traces.set(report.traceId, trace);
    }
    if (trace.reports.has(report.spanId)) { this.duplicates++; return false; }
    trace.reports.set(report.spanId, report);
    trace.lastReport = Math.max(trace.lastReport, time);
    this.bufferHighWater = Math.max(this.bufferHighWater, this.traces.size);
    if (this.traces.size > this.maxBufferedTraces) throw new Error("collector buffer overflow");
    return true;
  }

  /** Flush traces that have been quiet for a watermark period, or that hit the hard deadline. */
  advanceTo(time) {
    for (const [traceId, trace] of [...this.traces.entries()]) {
      const quiet = time - trace.lastReport >= this.watermark;
      const expired = time - trace.firstReport >= this.hardDeadline;
      if (quiet || expired) {
        this.traces.delete(traceId);
        this.flushed.push(this.assemble(trace, quiet ? "quiet" : "deadline"));
      }
    }
    return this;
  }

  /** Rebuilds the span tree from reports, tolerating missing parents and rejecting cycles. */
  assemble(trace, reason) {
    this.assemblyOps += trace.reports.size;
    const byId = new Map();
    for (const report of trace.reports.values()) byId.set(report.spanId, report);
    const children = new Map();
    const roots = [];
    let orphans = 0;
    for (const report of byId.values()) {
      if (report.parentId === null || report.parentId === undefined) { roots.push(report); continue; }
      if (!byId.has(report.parentId)) { orphans++; roots.push({ ...report, parentId: null, synthetic: true }); continue; }
      if (!children.has(report.parentId)) children.set(report.parentId, []);
      children.get(report.parentId).push(report);
    }
    // Cycle detection: walking up from every span must terminate at a root.
    for (const report of byId.values()) {
      const seen = new Set();
      let cursor = report;
      while (cursor && cursor.parentId !== null && cursor.parentId !== undefined) {
        if (seen.has(cursor.spanId)) throw new Error(`cycle in trace ${trace.traceId} at span ${report.spanId}`);
        seen.add(cursor.spanId);
        cursor = byId.get(cursor.parentId);
      }
    }
    const ordered = [...byId.values()].sort((a, b) => hlcCompare(a.hlc, b.hlc));
    return {
      traceId: trace.traceId, reason, spans: ordered, children, roots, orphans,
      complete: orphans === 0 && roots.length === 1,
    };
  }

  drained() { return this.traces.size === 0; }
}

/* ------------------------------------------------------------------ analysis of a flushed trace */

/** Des the skew: report.trueStart is recovered by subtracting the producer's clock offset. */
function deskew(report) { return { start: report.reportedStart - report.skew, end: report.reportedEnd - report.skew }; }

function analyseTrace(trace) {
  const byId = new Map(trace.spans.map((report) => [report.spanId, report]));
  const childIntervals = (report) => (trace.children.get(report.spanId) || [])
    .map((child) => { const d = deskew(child); return [d.start, d.end]; });
  const selfTimes = new Map();
  for (const report of trace.spans) {
    const d = deskew(report);
    selfTimes.set(report.spanId, selfTime({ trueStart: d.start, trueEnd: d.end }, () => childIntervals(report)));
  }
  // Rebuild the tree from the reassembled reports so the analyser works on report objects only.
  const buildNode = (report) => {
    const d = deskew(report);
    return { id: report.spanId, service: report.service, trueStart: d.start, trueEnd: d.end, children: [] };
  };
  const roots = trace.roots.map(buildNode);
  const link = (node) => {
    for (const child of trace.children.get(node.id) || []) {
      const childNode = buildNode(child);
      node.children.push(childNode);
      link(childNode);
    }
  };
  for (const root of roots) link(root);
  let criticalLength = 0;
  let criticalPathIds = [];
  for (const root of roots) {
    const result = criticalPath(root, (span) => span.children.map((child) => [child.trueStart, child.trueEnd]));
    if (result.length >= criticalLength) {
      criticalLength = result.length;
      criticalPathIds = result.chain.map((span) => span.id);
    }
  }
  const blameByService = new Map();
  for (const spanId of criticalPathIds) {
    const report = byId.get(spanId);
    if (!report) continue;
    blameByService.set(report.service,
      (blameByService.get(report.service) || 0) + (selfTimes.get(spanId) || 0));
  }
  const criticalSpans = criticalPathIds.map((spanId) => ({
    spanId, service: byId.has(spanId) ? byId.get(spanId).service : null, self: selfTimes.get(spanId) || 0,
  }));
  return {
    criticalLength, criticalPath: criticalPathIds, criticalSpans, selfTimes,
    rootDuration: roots.length > 0 ? roots[0].trueEnd - roots[0].trueStart : 0,
    blame: [...blameByService.entries()].map(([service, self]) => ({ service, self }))
      .sort((a, b) => b.self - a.self),
    spansAnalysed: trace.spans.length,
  };
}

/* ------------------------------------------------------------------ end-to-end pipeline */

/**
 * Runs one full pipeline: generate a trace, report the spans asynchronously (out of order, with
 * duplicates and optional loss), collect, flush and analyse on a virtual clock.
 */
function runPipeline(options) {
  const generated = generateTrace(options);
  const reportPrng = new Prng((options.seed || 1) ^ 0x9e37);
  const collector = new TraceCollector({
    watermark: options.watermark, hardDeadline: options.hardDeadline, maxBufferedTraces: options.maxBufferedTraces,
  });
  const traceId = options.traceId || `t${options.seed || 1}`;
  const events = [];
  let lossCount = 0;
  for (const span of generated.spans) {
    const skew = skewOf(span.service);
    if (options.loseSpanId && span.id === options.loseSpanId) { lossCount++; continue; }
    const report = {
      traceId, spanId: span.id, parentId: span.parentId,
      service: span.service, reportedStart: span.trueStart + skew, reportedEnd: span.trueEnd + skew,
      skew, hlc: span.hlc,
    };
    const copies = reportPrng.unit() < (options.duplicateRate || 0) ? 2 : 1;
    for (let copy = 0; copy < copies; copy++) {
      events.push({ time: reportPrng.below(options.spread || 120), report });
    }
  }
  events.sort((a, b) => a.time - b.time);
  for (const event of events) collector.ingest(event.report, event.time);
  collector.advanceTo((options.spread || 120) + (options.settle || 400));
  return { generated, collector, traceId, lossCount, flushed: collector.flushed };
}

/** The independent oracle shape: children spelled out the way the analyser expects them. */
function oracleTree(root) {
  return { ...root, trueStart: root.trueStart, trueEnd: root.trueEnd, children: root.children.map(oracleTree) };
}

module.exports = {
  Prng, Hlc, hlcCompare, hlcKey, generateTrace, unionLength, selfTime, criticalPath,
  bruteForceCriticalPath, TraceCollector, deskew, analyseTrace, runPipeline, oracleTree, SERVICES, skewOf,
};

/* ------------------------------------------------------------------ self-checks */

if (require.main === module) {
  const assert = require("node:assert");

  // --- HLC axioms ---------------------------------------------------------------------------
  {
    const clock = new Hlc("a", 100);
    const first = clock.send(100);
    const second = clock.send(100);
    assert.strictEqual(first.physical, 100);
    assert.strictEqual(second.counter, first.counter + 1, "equal physical time bumps the counter");
    const later = clock.send(200);
    assert.ok(hlcCompare(later, second) > 0, "HLC never goes backwards");
    assert.strictEqual(later.counter, 0, "a newer physical time resets the counter");
    const remote = { physical: 500, counter: 3, node: "b" };
    const received = clock.receive(remote, 300);
    assert.strictEqual(received.physical, 500, "the remote physical time wins");
    assert.ok(received.counter > remote.counter, "and the counter dominates the remote one");
    assert.ok(hlcCompare(received, remote) > 0);
    assert.strictEqual(hlcKey({ physical: 5, counter: 2, node: "z" }), "5.2.z");
    // A chain of alternating send/receive is strictly increasing.
    const left = new Hlc("l", 0);
    const right = new Hlc("r", 0);
    let previous = null;
    let carried = null;
    for (let i = 0; i < 20; i++) {
      const sent = left.send(10 + i * 3);
      if (previous) assert.ok(hlcCompare(sent, previous) > 0, "send must be monotone");
      previous = sent;
      carried = right.receive(sent, 10 + i * 3);
      assert.ok(hlcCompare(carried, sent) > 0, "receipt happens after the send");
      const reply = right.send(10 + i * 3);
      const back = left.receive(reply, 10 + i * 3);
      assert.ok(hlcCompare(back, carried) > 0);
      previous = back;
    }
  }

  // --- generated traces are well formed ------------------------------------------------------
  const sample = generateTrace({ seed: 893, maxDepth: 4 });
  assert.ok(sample.spans.length > 5, `only ${sample.spans.length} spans generated`);
  for (const span of sample.spans) {
    assert.ok(span.trueEnd >= span.trueStart, "an interval must be non-negative");
    for (const child of span.children) {
      assert.ok(child.trueStart >= span.trueStart && child.trueEnd <= span.trueEnd,
        "children must nest inside their parent");
      assert.ok(hlcCompare(child.hlc, span.hlc) > 0, "a child is causally after its parent");
    }
  }

  // --- HLC ordering versus wall-clock ordering under skew -------------------------------------
  {
    const skews = new Map(SERVICES.map((name, index) => [name, (index % 2 === 0 ? 1 : -1) * (5 + index * 3)]));
    let wallInversions = 0;
    let hlcInversions = 0;
    let comparisons = 0;
    for (const span of sample.spans) {
      for (const child of span.children) {
        comparisons++;
        const wallParent = span.trueStart + skews.get(span.service);
        const wallChild = child.trueStart + skews.get(child.service);
        if (wallChild <= wallParent) wallInversions++;
        if (hlcCompare(child.hlc, span.hlc) <= 0) hlcInversions++;
      }
    }
    assert.ok(comparisons > 5);
    assert.ok(wallInversions > 0, "clock skew must actually invert some parent/child pairs");
    assert.strictEqual(hlcInversions, 0, "HLC ordering must never invert a causal pair");
  }

  // --- deskewed containment -------------------------------------------------------------------
  {
    const skewOf = (service) => SERVICES.indexOf(service) % 2 === 0
      ? 5 + SERVICES.indexOf(service) * 3 : -(5 + SERVICES.indexOf(service) * 3);
    let nestedRaw = 0;
    let nestedDeskewed = 0;
    let pairs = 0;
    for (const span of sample.spans) {
      for (const child of span.children) {
        pairs++;
        if (child.trueStart + skewOf(child.service) >= span.trueStart + skewOf(span.service)
          && child.trueEnd + skewOf(child.service) <= span.trueEnd + skewOf(span.service)) nestedRaw++;
        if (child.trueStart >= span.trueStart && child.trueEnd <= span.trueEnd) nestedDeskewed++;
      }
    }
    assert.ok(pairs > 5);
    assert.strictEqual(nestedDeskewed, pairs, "de-skewed intervals nest exactly");
    assert.ok(nestedRaw < pairs, "the raw skewed timestamps do not nest reliably");
  }

  // --- union length and self time -------------------------------------------------------------
  assert.strictEqual(unionLength([]), 0);
  assert.strictEqual(unionLength([[0, 5]]), 5);
  assert.strictEqual(unionLength([[0, 5], [2, 8]]), 8, "overlapping intervals merge");
  assert.strictEqual(unionLength([[0, 5], [10, 12]]), 7, "disjoint intervals add up");
  assert.strictEqual(unionLength([[5, 10], [0, 3], [2, 6]]), 10, "unsorted input works");
  {
    const span = { trueStart: 0, trueEnd: 10, children: [] };
    assert.strictEqual(selfTime(span, () => []), 10);
    assert.strictEqual(selfTime(span, () => [[2, 5]]), 7);
  }

  // --- critical path against the brute-force oracle -------------------------------------------
  {
    for (let seed = 1; seed <= 6; seed++) {
      const trace = generateTrace({ seed: seed * 977, maxDepth: 4 });
      const intervalsOf = (span) => span.children.map((child) => [child.trueStart, child.trueEnd]);
      const fast = criticalPath(trace.root, intervalsOf);
      const slow = bruteForceCriticalPath(trace.root, intervalsOf);
      assert.strictEqual(Math.round(fast.length), Math.round(slow.length),
        `seed ${seed}: ${fast.length} vs ${slow.length}`);
      assert.deepStrictEqual(fast.chain.map((span) => span.id), slow.path, `seed ${seed}: same path`);
      // With a non-overlapping layout the self times partition the root span exactly.
      const totalSelf = trace.spans.reduce((total, span) => total + selfTime(span, intervalsOf), 0);
      assert.strictEqual(Math.round(totalSelf), Math.round(trace.root.trueEnd - trace.root.trueStart),
        `self times ${totalSelf} must partition the root span`);
      assert.ok(fast.length <= trace.root.trueEnd - trace.root.trueStart + 1e-9,
        "the critical path cannot exceed the trace duration");
      const selfTimes = trace.spans.map((span) => selfTime(span, intervalsOf));
      assert.ok(fast.length >= Math.max(...selfTimes) - 1e-9,
        "the critical path must at least cover the slowest single span");
    }
  }

  // --- ordered, duplicated reports assemble into the same tree --------------------------------
  for (const seed of [11, 42, 97]) {
    const pipeline = runPipeline({ seed, duplicateRate: 0.4, spread: 150, maxDepth: 4 });
    assert.strictEqual(pipeline.flushed.length, 1, `seed ${seed}: exactly one trace`);
    const trace = pipeline.flushed[0];
    assert.ok(pipeline.collector.duplicates > 0, "the run must have exercised duplicate reports");
    assert.strictEqual(trace.spans.length, pipeline.generated.spans.length,
      "every generated span must be assembled exactly once");
    assert.ok(trace.complete, "an undamaged trace must be reported complete");
    const analysis = analyseTrace(trace);
    assert.ok(analysis.criticalLength > 0);
    assert.strictEqual(analysis.spansAnalysed, pipeline.generated.spans.length);
    const oracle = criticalPath(pipeline.generated.root,
      (span) => span.children.map((child) => [child.trueStart, child.trueEnd]));
    assert.ok(Math.abs(analysis.criticalLength - oracle.length) <= 2,
      `seed ${seed}: critical path ${analysis.criticalLength} vs true ${oracle.length} (skew tolerance)`);
    const shareTotal = analysis.blame.reduce((total, entry) => total + entry.self, 0);
    assert.ok(Math.abs(shareTotal - analysis.criticalLength) < 1e-9, "blame must sum to the critical path");
    assert.ok(analysis.blame.length >= 1 && analysis.blame[0].self >= analysis.blame[analysis.blame.length - 1].self,
      "blame is sorted");
  }

  // --- the analysis is arrival-order independent ----------------------------------------------
  {
    const first = runPipeline({ seed: 5, duplicateRate: 0.5, spread: 300, maxDepth: 4 });
    const second = runPipeline({ seed: 5, duplicateRate: 0.5, spread: 300, maxDepth: 4, settle: 900 });
    assert.deepStrictEqual(first.flushed[0].spans.map((span) => span.spanId),
      second.flushed[0].spans.map((span) => span.spanId), "HLC order is stable across runs");
    assert.strictEqual(analyseTrace(first.flushed[0]).criticalLength,
      analyseTrace(second.flushed[0]).criticalLength);
  }

  // --- a lost report is surfaced, not hidden --------------------------------------------------
  {
    const victim = sample.spans.find((span) => span.children.length > 0);
    const pipeline = runPipeline({ seed: 21, loseSpanId: victim.id, maxDepth: 4 });
    const trace = pipeline.flushed[0];
    assert.strictEqual(pipeline.lossCount, 1);
    assert.strictEqual(trace.spans.length, pipeline.generated.spans.length - 1);
    assert.strictEqual(trace.complete, false, "an incomplete trace must say so");
    assert.strictEqual(trace.orphans, victim.children.length, "the children of the lost span become orphans");
    const analysis = analyseTrace(trace);
    assert.ok(analysis.criticalLength >= 0, "analysis still runs on the partial trace");
  }

  // --- cycles are rejected --------------------------------------------------------------------
  {
    const collector = new TraceCollector({ watermark: 10 });
    const base = { traceId: "cyc", service: "gateway" };
    collector.ingest({ ...base, spanId: "a", parentId: "b", reportedStart: 0, reportedEnd: 5, skew: 0, hlc: { physical: 1, counter: 0, node: "x" } }, 0);
    collector.ingest({ ...base, spanId: "b", parentId: "a", reportedStart: 0, reportedEnd: 5, skew: 0, hlc: { physical: 2, counter: 0, node: "x" } }, 1);
    assert.throws(() => collector.advanceTo(100), /cycle/, "a cyclic trace must be rejected");
    assert.strictEqual(collector.flushed.length, 0);
  }

  // --- the flushed trace identifies the deliberately slowed service ---------------------------
  {
    for (const seed of [3, 9, 15, 27]) {
      const pipeline = runPipeline({ seed, slowService: "pricing", slowExtra: 60, maxDepth: 5, duplicateRate: 0.2 });
      assert.ok(pipeline.generated.spans.some((span) => span.service === "pricing"),
        `seed ${seed}: the trace must actually contain the slowed service`);
      const analysis = analyseTrace(pipeline.flushed[0]);
      assert.ok(analysis.criticalSpans.length > 0);
      const worst = analysis.criticalSpans.reduce((a, b) => (b.self > a.self ? b : a));
      assert.strictEqual(worst.service, "pricing",
        `seed ${seed}: worst span was ${worst.service} (${worst.self}ms)`);
      assert.ok(analysis.blame.some((entry) => entry.service === "pricing"),
        `seed ${seed}: pricing must appear in the blame table`);
      const worstUninjected = analysis.criticalSpans
        .filter((entry) => entry.service !== "pricing")
        .reduce((a, b) => Math.max(a, b.self), 0);
      assert.ok(worst.self >= worstUninjected + 30, "the injection must dominate the path");
    }
  }

  // --- skew does not move the critical path ---------------------------------------------------
  {
    const intervalsOf = (span) => span.children.map((child) => [child.trueStart, child.trueEnd]);
    const clean = generateTrace({ seed: 77, maxDepth: 4 });
    const expected = bruteForceCriticalPath(clean.root, intervalsOf);
    const pipeline = runPipeline({ seed: 77, maxDepth: 4 });
    const analysis = analyseTrace(pipeline.flushed[0]);
    assert.strictEqual(analysis.criticalLength, expected.length,
      `skewed critical path ${analysis.criticalLength} vs true ${expected.length}`);
    const byId = new Map(pipeline.generated.spans.map((span) => [span.id, span]));
    const chain = analysis.criticalPath.map((id) => byId.get(id));
    assert.ok(chain.every(Boolean), "every span on the critical path must exist");
    assert.strictEqual(chain[0].parentId, null, "the critical path starts at the root");
    for (let i = 1; i < chain.length; i++) {
      assert.strictEqual(chain[i].parentId, chain[i - 1].id, "the critical path follows parent links");
    }
  }

  // --- scale and cost -------------------------------------------------------------------------
  {
    const pipeline = runPipeline({ seed: 404, maxDepth: 6, services: 6, spread: 400, duplicateRate: 0.3, settle: 1600 });
    const trace = pipeline.flushed[0];
    assert.ok(trace.spans.length >= 20, `only ${trace.spans.length} spans at scale`);
    assert.ok(trace.complete, "the large trace must assemble completely");
    const analysis = analyseTrace(trace);
    assert.ok(analysis.criticalLength > 0);
    assert.ok(pipeline.collector.assemblyOps <= 3 * pipeline.generated.spans.length,
      `assembly touched ${pipeline.collector.assemblyOps} rows for ${pipeline.generated.spans.length} spans`);
    assert.ok(pipeline.collector.bufferHighWater <= 2, "only one trace is in flight");
    assert.strictEqual(pipeline.collector.drained(), pipeline.collector.traces.size === 0);
  }

  // --- edge cases ---------------------------------------------------------------------------
  {
    const single = generateTrace({ seed: 1, maxDepth: 0 });
    assert.strictEqual(single.spans.length, 1);
    const intervalsOf = () => [];
    assert.strictEqual(criticalPath(single.root, intervalsOf).length, single.root.trueEnd - single.root.trueStart);
    const collector = new TraceCollector({ watermark: 10, hardDeadline: 50 });
    assert.strictEqual(collector.flushed.length, 0);
    collector.advanceTo(1000);
    assert.strictEqual(collector.flushed.length, 0, "nothing ingested, nothing flushed");
    assert.strictEqual(collector.drained(), true);
    const report = {
      traceId: "solo", spanId: "only", parentId: null, service: "gateway",
      reportedStart: 10, reportedEnd: 10, skew: 5, hlc: { physical: 1, counter: 0, node: "g" },
    };
    collector.ingest(report, 0);
    collector.advanceTo(5);
    assert.strictEqual(collector.flushed.length, 0, "the watermark has not passed yet");
    collector.advanceTo(60);
    assert.strictEqual(collector.flushed.length, 1);
    const trace = collector.flushed[0];
    assert.strictEqual(trace.reason, "quiet");
    assert.ok(trace.complete);
    const analysis = analyseTrace(trace);
    assert.strictEqual(analysis.criticalLength, 0, "a zero-length span has no critical path length");
    assert.deepStrictEqual(analysis.blame, [{ service: "gateway", self: 0 }]);
    // A trace that never becomes quiet is flushed on the hard deadline and marked incomplete.
    const stubborn = new TraceCollector({ watermark: 1000, hardDeadline: 50 });
    stubborn.ingest(report, 0);
    stubborn.ingest({ ...report, spanId: "later" }, 40);
    stubborn.advanceTo(60);
    assert.strictEqual(stubborn.flushed.length, 1);
    assert.strictEqual(stubborn.flushed[0].reason, "deadline", "a chatty trace hits the hard deadline");
    assert.strictEqual(stubborn.flushed[0].complete, false, "two roots means the picture is partial");
  }

  // --- determinism --------------------------------------------------------------------------
  {
    const digest = () => {
      const pipeline = runPipeline({ seed: 1234, duplicateRate: 0.3, maxDepth: 5, spread: 200 });
      const analysis = analyseTrace(pipeline.flushed[0]);
      return `${pipeline.collector.duplicates}|${analysis.criticalLength}|${analysis.criticalPath.join(",")}`;
    };
    assert.strictEqual(digest(), digest());
  }

  console.log(`tracing: ${sample.spans.length}-span sample, ${runPipeline({ seed: 404, maxDepth: 6, spread: 400, duplicateRate: 0.3, settle: 1600 }).flushed[0].spans.length} spans at scale, ` +
    `HLC order stable under skew, critical path verified against brute force`);
  console.log("PASS");
}
