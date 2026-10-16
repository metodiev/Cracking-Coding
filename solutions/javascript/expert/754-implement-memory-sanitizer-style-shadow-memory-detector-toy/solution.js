/** Task 754 - Implement memory sanitizer style shadow memory detector (toy). (Tier: Expert, CTCI Beyond — Security & Privacy)
 *
 * A toy arena plus the two shadow planes a sanitizer needs. (1) *Addressability*: every byte is
 * free, a payload byte, a redzone byte or a freed payload byte, so an access is classified as
 * in-bounds, out-of-bounds (redzone/free) or use-after-free. (2) *Definedness*: one mask per
 * byte, one bit per data bit, following the MSan rule that a value is as defined as its least
 * defined operand — `copy`/`memcpy` transport the shadow bit-for-bit, `memset` sets every bit, a
 * fresh allocation starts fully undefined (origin = the allocation name) and `dst = lhs op rhs`
 * has the *bitwise AND* of the operand masks. Reports name the first offending byte and the
 * allocation that owns it, which turns "uninitialized value" into an actionable finding.
 *
 * What is simplified: a bump-allocated byte arena instead of a real heap, host bytes instead of
 * bit-interleaved shadow, no compiler instrumentation (every operation is an explicit call) and
 * no stack tracking, so a clean shadow models MSan rather than guaranteeing anything about
 * machine code. What is verified: a fresh allocation reports as an uninitialized read naming its
 * origin; a partially written buffer reports the exact undefined offsets and undefined bit count;
 * `memcpy` transports partial definedness unchanged; the AND rule holds for arithmetic; redzone
 * and freed accesses are classified out-of-bounds / use-after-free; double free and interior free
 * are rejected; a fully initialised program reports nothing. Finally 40 seeded random operation
 * sequences are replayed against an independent reference model (interval scan, no per-byte
 * planes) that must agree on definedness, byte classification and the sequence of report kinds.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

const PAYLOAD = 1, REDZONE = 2, FREED = 3;
const STATE_NAMES = ["free", "payload", "redzone", "freed"];
const FULL = 0xff;
const popcount = (mask) => { let count = 0; for (let bits = mask; bits; bits >>= 1) count += bits & 1; return count; };
class ShadowMemory {
  constructor(size, redzone = 4) {
    this.size = size;
    this.redzone = redzone;
    this.init = new Uint8Array(size);            // definedness: one bit per data bit
    this.state = new Uint8Array(size);           // addressability plane
    this.owner = new Array(size).fill(null);
    this.byBase = new Map();
    this.names = new Set();
    this.reports = [];
    this.cursor = 0;
  }
  /** Bump allocation with a redzone on both sides; the payload starts fully undefined. */
  alloc(name, size) {
    if (!Number.isInteger(size) || size < 0) throw new RangeError(`bad allocation size ${size}`);
    if (this.names.has(name)) throw new Error(`duplicate allocation '${name}'`);
    const base = this.cursor + this.redzone;
    if (base + size + this.redzone > this.size) throw new RangeError("arena exhausted");
    for (let addr = base - this.redzone; addr < base + size + this.redzone; addr++) {
      const inPayload = addr >= base && addr < base + size;
      this.state[addr] = inPayload ? PAYLOAD : REDZONE;
      this.owner[addr] = name;
      this.init[addr] = 0;
    }
    this.cursor = base + size + this.redzone;
    const allocation = { name, base, size };
    this.names.add(name);
    this.byBase.set(base, allocation);
    return allocation;
  }
  classifyByte(addr) {
    if (!(addr >= 0 && addr < this.size)) return { kind: "outside", owner: null };
    return { kind: STATE_NAMES[this.state[addr]], owner: this.owner[addr] };
  }
  /** Addressability gate; reports the first offending byte once and returns false. */
  isAddressable(addr, length, access) {
    for (let i = 0; i < length; i++) {
      const { kind, owner } = this.classifyByte(addr + i);
      if (kind === "payload") continue;
      if (kind === "freed") this.reports.push({ kind: "use-after-free", access, addr: addr + i, allocation: owner });
      else this.reports.push({ kind: "out-of-bounds", access, addr: addr + i, allocation: owner });
      return false;
    }
    return true;
  }
  masks(addr, length) { return Array.from({ length }, (_, i) => this.init[addr + i]); }
  writeShadow(addr, masks) {
    if (!this.isAddressable(addr, masks.length, "write")) return false;
    masks.forEach((mask, i) => { this.init[addr + i] = mask & 0xff; });
    return true;
  }
  memset(addr, length) { return this.writeShadow(addr, new Array(length).fill(FULL)); }
  memcpy(dst, src, length) {
    if (length === 0) return true;
    if (!this.isAddressable(src, length, "read") || !this.isAddressable(dst, length, "write")) return false;
    return this.writeShadow(dst, this.masks(src, length));
  }
  /** `dst = lhs <op> rhs`; the result is defined exactly where both operands are. */
  arithmetic(dst, lhs, rhs, length) {
    if (!this.isAddressable(lhs, length, "read") || !this.isAddressable(rhs, length, "read")) return false;
    return this.writeShadow(dst, Array.from({ length }, (_, i) => this.init[lhs + i] & this.init[rhs + i]));
  }
  read(addr, length) {
    if (!this.isAddressable(addr, length, "read")) return null;
    const masks = this.masks(addr, length);
    const undefinedOffsets = masks.map((mask, i) => (mask === FULL ? -1 : i)).filter((i) => i >= 0);
    if (undefinedOffsets.length === 0) return masks;
    this.reports.push({
      kind: "uninitialized-read", addr, origin: this.owner[addr], undefinedOffsets,
      undefinedBits: masks.reduce((sum, mask) => sum + (mask === FULL ? 0 : 8 - popcount(mask)), 0),
    });
    return null;
  }
  branch(addr, length) {
    if (!this.isAddressable(addr, length, "read")) return false;
    const first = this.masks(addr, length).findIndex((mask) => mask !== FULL);
    if (first < 0) return true;
    this.reports.push({ kind: "branch-on-uninitialized", addr: addr + first, origin: this.owner[addr] });
    return false;
  }
  free(addr) {
    const allocation = this.byBase.get(addr);
    if (allocation === undefined) { this.reports.push({ kind: "invalid-free", addr }); return false; }
    if (this.state[addr] === FREED) { this.reports.push({ kind: "double-free", addr, allocation: allocation.name }); return false; }
    for (let i = 0; i < allocation.size; i++) { this.state[addr + i] = FREED; this.init[addr + i] = 0; }
    return true;
  }
  clearReports() { this.reports = []; }
}
/** Independent reference model: addressability by interval scan, definedness by a plain array. */
class ReferenceArena {
  constructor(size, redzone = 4) {
    this.size = size;
    this.redzone = redzone;
    this.layout = [];
    this.freed = new Set();
    this.init = new Array(size).fill(0);
    this.reportKinds = [];
    this.cursor = 0;
  }
  alloc(name, size) {
    const base = this.cursor + this.redzone;
    this.cursor = base + size + this.redzone;
    if (this.cursor > this.size) throw new RangeError("arena exhausted");
    for (let i = 0; i < size; i++) this.init[base + i] = 0;
    this.layout.push({ name, base, size });
    return { name, base, size };
  }
  classifyByte(addr) {
    if (addr < 0 || addr >= this.size) return { kind: "outside", owner: null };
    for (let i = this.layout.length - 1; i >= 0; i--) {
      const { name, base, size } = this.layout[i];
      if (addr >= base - this.redzone && addr < base + size + this.redzone) {
        const inPayload = addr >= base && addr < base + size;
        return { kind: inPayload ? (this.freed.has(name) ? "freed" : "payload") : "redzone", owner: name };
      }
    }
    return { kind: "free", owner: null };
  }

  gate(addr, length) {
    for (let i = 0; i < length; i++) {
      const { kind } = this.classifyByte(addr + i);
      if (kind === "payload") continue;
      this.reportKinds.push(kind === "freed" ? "use-after-free" : "out-of-bounds");
      return false;
    }
    return true;
  }

  writeShadow(addr, masks) {
    if (!this.gate(addr, masks.length)) return false;
    masks.forEach((mask, i) => { this.init[addr + i] = mask & 0xff; });
    return true;
  }

  memset(addr, length) { return this.writeShadow(addr, new Array(length).fill(FULL)); }

  masks(addr, length) { return Array.from({ length }, (_, i) => this.init[addr + i]); }

  memcpy(dst, src, length) {
    if (length === 0) return true;
    if (!this.gate(src, length) || !this.gate(dst, length)) return false;
    return this.writeShadow(dst, this.masks(src, length));
  }

  arithmetic(dst, lhs, rhs, length) {
    if (!this.gate(lhs, length) || !this.gate(rhs, length)) return false;
    return this.writeShadow(dst, Array.from({ length }, (_, i) => this.init[lhs + i] & this.init[rhs + i]));
  }

  read(addr, length) {
    if (!this.gate(addr, length)) return null;
    const masks = this.masks(addr, length);
    if (masks.every((mask) => mask === FULL)) return masks;
    this.reportKinds.push("uninitialized-read");
    return null;
  }

  branch(addr, length) {
    if (!this.gate(addr, length)) return false;
    if (this.masks(addr, length).every((mask) => mask === FULL)) return true;
    this.reportKinds.push("branch-on-uninitialized");
    return false;
  }

  free(addr) {
    const allocation = this.layout.find((entry) => entry.base === addr);
    if (allocation === undefined) { this.reportKinds.push("invalid-free"); return false; }
    if (this.freed.has(allocation.name)) { this.reportKinds.push("double-free"); return false; }
    this.freed.add(allocation.name);
    for (let i = 0; i < allocation.size; i++) this.init[addr + i] = 0;
    return true;
  }
}

/** Deterministic xorshift32. */
function makeRng(seed) {
  let state = seed >>> 0 || 0x754754;
  return (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

module.exports = { ShadowMemory, ReferenceArena, makeRng, popcount, FULL };

if (require.main === module) {
  const ARENA = 128;
  // 1. a fresh allocation is fully undefined, and reading it names the origin
  const s1 = new ShadowMemory(ARENA);
  const buf = s1.alloc("buf", 4);
  deepStrictEqual(s1.masks(buf.base, 4), [0, 0, 0, 0], "a new allocation starts undefined");
  strictEqual(s1.read(buf.base, 4), null);
  deepStrictEqual(s1.reports, [{ kind: "uninitialized-read", addr: buf.base, origin: "buf", undefinedOffsets: [0, 1, 2, 3], undefinedBits: 32 }]);
  // 2. after a memset the same read is clean: no false positives
  s1.clearReports();
  strictEqual(s1.memset(buf.base, 4), true);
  deepStrictEqual(s1.read(buf.base, 4), [FULL, FULL, FULL, FULL]);
  deepStrictEqual(s1.reports, [], "a fully initialised buffer must not report");
  // 3. partial initialisation reports the exact offsets and undefined bit count
  const s3 = new ShadowMemory(ARENA);
  const half = s3.alloc("half", 4);
  s3.writeShadow(half.base, [FULL, FULL, 0x0f, 0]);
  strictEqual(s3.read(half.base, 4), null);
  deepStrictEqual(s3.reports[0].undefinedOffsets, [2, 3]);
  strictEqual(s3.reports[0].undefinedBits, 12, "4 undefined bits in byte 2 plus 8 in byte 3");
  deepStrictEqual(s3.read(half.base, 2), [FULL, FULL], "the initialised prefix reads cleanly");
  // 4. memcpy transports the shadow bit-for-bit
  const s4 = new ShadowMemory(ARENA);
  const from = s4.alloc("from", 6);
  const to = s4.alloc("to", 6);
  s4.writeShadow(from.base, [FULL, 0x01, 0, 0x80, FULL, 0x3c]);
  strictEqual(s4.memcpy(to.base, from.base, 6), true);
  deepStrictEqual(s4.masks(to.base, 6), s4.masks(from.base, 6), "copy must preserve definedness");
  // 5. the arithmetic rule: the result is defined exactly where both operands are
  const mixed = s4.alloc("mixed", 6);
  s4.writeShadow(mixed.base, [FULL, 0, FULL, 0, FULL, 0]);
  const sum = s4.alloc("sum", 6);
  strictEqual(s4.arithmetic(sum.base, from.base, mixed.base, 6), true);
  deepStrictEqual(s4.masks(sum.base, 6), [FULL, 0, 0, 0, FULL, 0], "AND of both operand masks");
  deepStrictEqual(s4.reports, [], "merging definedness is not itself a finding");
  // 6. redzone, unallocated bytes and arena overflow are distinguished
  strictEqual(s4.classifyByte(from.base - 1).kind, "redzone");
  strictEqual(s4.classifyByte(ARENA - 1).kind, "free");
  throws(() => s4.alloc("from", 4), Error);
  throws(() => s4.alloc("huge", ARENA), RangeError);
  throws(() => s4.alloc("bad", -1), RangeError);
  // 7. out-of-bounds is detected at the first byte past the payload, in both directions
  const s7 = new ShadowMemory(ARENA);
  const seven = s7.alloc("seven", 4);
  s7.memset(seven.base, 4);
  ok(s7.read(seven.base + 3, 1) !== null, "the last payload byte is in bounds");
  s7.clearReports();
  strictEqual(s7.read(seven.base + 4, 1), null);
  deepStrictEqual(s7.reports, [{ kind: "out-of-bounds", access: "read", addr: seven.base + 4, allocation: "seven" }]);
  s7.clearReports();
  strictEqual(s7.memset(seven.base - 1, 2), false);
  deepStrictEqual([s7.reports[0].kind, s7.reports[0].access], ["out-of-bounds", "write"]);
  s7.clearReports();
  strictEqual(s7.read(ARENA + 10, 1), null, "reads outside the arena are reported too");
  strictEqual(s7.reports[0].allocation, null);
  // 8. use-after-free, double free and freeing an interior pointer
  const s8 = new ShadowMemory(ARENA);
  const live = s8.alloc("live", 8);
  const other = s8.alloc("other", 8);
  strictEqual(s8.free(live.base), true);
  s8.clearReports();
  strictEqual(s8.read(live.base, 4), null);
  deepStrictEqual(s8.reports, [{ kind: "use-after-free", access: "read", addr: live.base, allocation: "live" }]);
  strictEqual(s8.free(live.base), false);
  strictEqual(s8.reports[1].kind, "double-free");
  strictEqual(s8.free(other.base + 2), false);
  strictEqual(s8.reports[2].kind, "invalid-free");
  strictEqual(s8.free(other.base), true, "a well-formed free of the live allocation still works");
  // 9. a branch on an undefined condition is a control-flow finding
  const s9 = new ShadowMemory(ARENA);
  const flag = s9.alloc("flag", 1);
  strictEqual(s9.branch(flag.base, 1), false);
  deepStrictEqual(s9.reports[0], { kind: "branch-on-uninitialized", addr: flag.base, origin: "flag" });
  s9.clearReports();
  s9.memset(flag.base, 1);
  strictEqual(s9.branch(flag.base, 1), true);
  deepStrictEqual(s9.reports, [], "an initialised condition is fine");
  // 10. boundary: zero-length accesses touch nothing at all
  const s10 = new ShadowMemory(ARENA);
  const empty = s10.alloc("empty", 0);
  strictEqual(empty.base, 4);
  ok(s10.read(empty.base, 0) !== null && s10.memcpy(empty.base + 1, empty.base + 1, 0), "zero-length accesses are safe");
  deepStrictEqual(s10.reports, []);
  // 11. randomized differential test against the independent reference model
  const rng = makeRng(0x754754);
  let operations = 0;
  for (let trial = 0; trial < 40; trial++) {
    const model = new ShadowMemory(ARENA);
    const reference = new ReferenceArena(ARENA);
    const names = [];
    for (let i = 0, count = 4 + rng(4); i < count; i++) {
      const name = `a${trial}_${i}`;
      const size = 1 + rng(9);
      const mine = model.alloc(name, size);
      strictEqual(mine.base, reference.alloc(name, size).base, `layout differs for ${name}`);
      names.push(mine);
    }
    for (let step = 0; step < 30; step++) {
      const target = names[rng(names.length)];
      const addr = target.base + rng(target.size + 1);       // +1 deliberately lands in the redzone
      const length = rng(4);
      const choice = rng(7);
      operations++;
      if (choice === 0) strictEqual(model.memset(addr, length), reference.memset(addr, length));
      else if (choice === 1) strictEqual(model.read(addr, length) === null, reference.read(addr, length) === null);
      else if (choice === 2) strictEqual(model.branch(addr, length), reference.branch(addr, length));
      else if (choice === 3) {
        const source = names[rng(names.length)];
        strictEqual(model.memcpy(addr, source.base, length), reference.memcpy(addr, source.base, length));
      } else if (choice === 4) {
        const right = names[rng(names.length)];
        strictEqual(model.arithmetic(addr, target.base, right.base, length), reference.arithmetic(addr, target.base, right.base, length));
      } else if (choice === 5) {
        const masks = Array.from({ length }, () => rng(256));
        strictEqual(model.writeShadow(addr, masks), reference.writeShadow(addr, masks));
      } else {
        strictEqual(model.free(target.base), reference.free(target.base));
      }
      deepStrictEqual(model.reports.map((r) => r.kind), reference.reportKinds, `report kinds, trial ${trial} step ${step}`);
      for (let byte = 0; byte < ARENA; byte++) {
        const here = model.classifyByte(byte);
        const there = reference.classifyByte(byte);
        ok(model.init[byte] === reference.init[byte] && here.kind === there.kind && here.owner === there.owner,
          `shadow diverged at byte ${byte}, trial ${trial} step ${step}`);
      }
    }
  }
  console.log(`arena=${ARENA} differentialTrials=40 operations=${operations}`);
  console.log("PASS");
}
