/** Task 629 - Implement dynamic reconfiguration in Raft (joint consensus). (Tier: Expert, CTCI Beyond — Distributed Systems)
 *
 * Raft changes membership through an intermediate *joint configuration* C_old,new: an entry that
 * requires majorities of both C_old and C_new to be committed. That single entry is what closes the
 * door on two disjoint majorities: while it is the effective configuration, any two quorums overlap
 * because both share a majority of C_old and of C_new, and after it commits, every later quorum of
 * C_new still overlaps the joint quorum that decided the transition. Only then does the leader append
 * the plain C_new entry, after which majorities of C_new alone suffice. Membership therefore moves
 * one step at a time (one node added or removed per change), which the checks verify exhaustively.
 *
 * What is simplified: the quorum mathematics is verified by exhaustive enumeration of the quorum
 * families of small configurations (two to five voters), so "any two quorums intersect" is a proof
 * for those configurations rather than sampling. The log/ack simulation drives configuration entries
 * through the same rule with explicit acknowledgement sets - no timers, no network, no proposal
 * queue: a check hands the ack set to `commit` and asks whether the entry becomes committed.
 *
 * Verified invariants: (1) two quorums of one configuration always intersect; (2) any joint quorum
 * intersects any other joint quorum, any quorum of C_old and any quorum of C_new; (3) consequently
 * a joint quorum intersects any quorum of the *final* C_new - the property that makes the transition
 * safe; (4) a direct C_old -> C_new replacement is shown to be unsafe when the two sets are disjoint,
 * and safe when they differ by one node (why increments are restricted to one node at a time);
 * (5) the C_new entry cannot be committed before the joint entry, and a removed node's ack stops
 * counting once C_new is in effect.
 */
"use strict";

/** All 2^n subsets - small configurations only, which is exactly what exhaustive checks need. */
function subsets(items) {
  let result = [[]];
  for (const item of items) result = result.concat(result.map((subset) => subset.concat(item)));
  return result;
}

function quorums(voters) {
  return subsets(voters).filter((subset) => subset.length * 2 > voters.length);
}

function intersects(a, b) {
  return a.some((item) => b.includes(item));
}

function label(quorum) {
  return quorum.slice().sort().join(",") || "(none)";
}

/** A configuration: a plain voter set, or a joint one that needs both majorities. */
function config(oldVoters, newVoters = null) {
  return { old: oldVoters.slice().sort(), new: newVoters === null ? null : newVoters.slice().sort() };
}

function isJoint(configuration) {
  return configuration.new !== null;
}

function jointQuorums(configuration) {
  const result = [];
  for (const fromOld of quorums(configuration.old)) {
    for (const fromNew of quorums(configuration.new)) {
      const union = [...new Set(fromOld.concat(fromNew))].sort();
      result.push({ union, fromOld, fromNew });
    }
  }
  return result;
}

function majorityOf(acknowledgements, voters) {
  const acked = voters.filter((voter) => acknowledgements.includes(voter));
  return acked.length * 2 > voters.length;
}

function hasQuorum(acknowledgements, configuration) {
  if (!isJoint(configuration)) return majorityOf(acknowledgements, configuration.old);
  return majorityOf(acknowledgements, configuration.old) && majorityOf(acknowledgements, configuration.new);
}

/** A log of configuration entries plus the ack bookkeeping that decides them. */
class ConfigLog {
  constructor(initialVoters) {
    this.entries = [];
    this.jointCommitted = false;
    this.append({ kind: "config", configuration: config(initialVoters) });
  }

  append(entry) {
    const effective = this.effective();
    if (entry.kind === "config" && this.entries.length > 0) {
      const previous = effective.configuration;
      const next = entry.configuration;
      if (isJoint(previous) && !this.jointCommitted) throw new Error("previous configuration change is unfinished");
      if (!isJoint(next) && !this.jointCommitted && this.hadJoint()) {
        throw new Error("the joint configuration must be committed before the final one");
      }
      if (Math.abs(next.old.length - previous.old.length) > 1 && next.new === null && previous.new === null) {
        throw new Error("only one server may be added or removed at a time");
      }
    }
    entry.committed = false;
    entry.index = this.entries.length + 1;
    this.entries.push(entry);
    return entry;
  }

  hadJoint() {
    return this.entries.some((entry) => entry.kind === "config" && isJoint(entry.configuration));
  }

  effective() {
    return this.entries[this.entries.length - 1];
  }

  /** Commit an entry if the acknowledgements satisfy the configuration in effect at that index. */
  commit(entry, acknowledgements) {
    const configuration = this.configurationAt(entry.index);
    if (!hasQuorum(acknowledgements, configuration)) return false;
    entry.committed = true;
    if (entry.kind === "config" && isJoint(entry.configuration)) this.jointCommitted = true;
    return true;
  }

  /** The configuration an entry is decided against: the newest config entry at or before it. */
  configurationAt(index) {
    for (let position = index - 1; position >= 0; position--) {
      const entry = this.entries[position];
      if (entry.kind === "config") return entry.configuration;
    }
    throw new Error("no configuration in the log");
  }

  canElect(votes) {
    return hasQuorum(votes, this.configurationAt(this.entries.length));
  }

  committedConfigurations() {
    return this.entries.filter((entry) => entry.kind === "config" && entry.committed).map((entry) => entry.configuration);
  }
}

function lcg(seed) {
  return (seed * 48271) % 2147483647;
}

module.exports = {
  subsets, quorums, intersects, label, config, isJoint, jointQuorums, hasQuorum, majorityOf, ConfigLog, lcg,
};

if (require.main === module) {
  const assert = require("node:assert");
  const oldVoters = ["a", "b", "c"];
  const newVoters = ["c", "d", "e"]; // one server overlapping, one removed, two added
  const joint = config(oldVoters, newVoters);
  const single = config(newVoters);

  // --- exhaustive quorum-intersection proof for the joint configuration ---------------------
  const oldQuorums = quorums(oldVoters);
  const newQuorums = quorums(newVoters);
  const jointFamilies = jointQuorums(joint);
  assert.deepStrictEqual(oldQuorums.map(label), ["a,b", "a,c", "b,c", "a,b,c"], "a majority of three is any two or all three");
  assert.strictEqual(jointFamilies.length, 16, "four quorums of each side give sixteen joint quorums");
  for (const left of oldQuorums) for (const right of oldQuorums) assert.ok(intersects(left, right), "old quorums overlap");
  for (const left of newQuorums) for (const right of newQuorums) assert.ok(intersects(left, right), "new quorums overlap");
  for (const left of jointFamilies) {
    for (const right of jointFamilies) assert.ok(intersects(left.union, right.union), "joint quorums overlap");
    for (const quorum of oldQuorums) assert.ok(intersects(left.union, quorum), "joint overlaps every old quorum");
    for (const quorum of newQuorums) assert.ok(intersects(left.union, quorum), "joint overlaps every new quorum");
  }
  assert.ok(jointFamilies.some((family) => label(family.union) === "a,b,c,d,e"), "the widest joint quorum needs both majorities");

  // --- a direct C_old -> C_new swap is unsafe, a one-server change is safe -------------------
  let unsafePairs = 0;
  for (const left of oldQuorums) for (const right of quorums(["d", "e", "f"])) if (!intersects(left, right)) unsafePairs++;
  assert.strictEqual(unsafePairs, oldQuorums.length * quorums(["d", "e", "f"]).length,
    "replacing the set wholesale admits disjoint majorities");
  let safeAdded = 0, safeRemoved = 0;
  for (const left of oldQuorums) for (const right of quorums(["a", "b", "c", "d"])) safeAdded += intersects(left, right) ? 1 : 0;
  for (const left of oldQuorums) for (const right of quorums(["b", "c"])) safeRemoved += intersects(left, right) ? 1 : 0;
  assert.strictEqual(safeAdded, oldQuorums.length * quorums(["a", "b", "c", "d"]).length,
    "adding one server keeps every pair of quorums overlapping");
  assert.strictEqual(safeRemoved, oldQuorums.length * quorums(["b", "c"]).length,
    "removing one server keeps every pair of quorums overlapping");
  for (const size of [1, 2, 4, 5]) {
    const voters = ["v0", "v1", "v2", "v3", "v4"].slice(0, size);
    for (const left of quorums(voters)) for (const right of quorums(voters)) {
      assert.ok(intersects(left, right), `majorities of ${size} voters overlap`);
    }
  }

  // --- the transition in a log ---------------------------------------------------------------
  const log = new ConfigLog(oldVoters);
  const jointEntry = log.append({ kind: "config", configuration: joint });
  assert.strictEqual(log.configurationAt(jointEntry.index).new === null, false, "the joint config is effective");
  assert.strictEqual(log.commit(jointEntry, ["c", "d"]), false, "a C_new majority alone is not enough");
  assert.strictEqual(log.commit(jointEntry, ["d", "e"]), false, "nor is another C_new majority");
  assert.strictEqual(log.commit(jointEntry, ["c", "d", "a"]), true, "both majorities together commit it");
  assert.strictEqual(log.jointCommitted, true);
  assert.ok(!log.canElect(["d", "e"]), "an election needs a majority of C_old too");
  assert.ok(log.canElect(["d", "e", "a", "b"]), "with both majorities the election succeeds");
  assert.ok(!log.canElect(["d", "e", "b"]), "a single C_old voter is not a majority of three");

  const finalEntry = log.append({ kind: "config", configuration: single });
  assert.strictEqual(log.commit(finalEntry, ["d", "e"]), true, "only a C_new majority is needed now");
  assert.deepStrictEqual(log.committedConfigurations().length, 2, "both configuration entries are committed");
  assert.ok(log.canElect(["c", "d"]), "elections use C_new alone");
  assert.ok(!log.canElect(["a", "b"]), "the removed servers can no longer elect anyone");
  assert.ok(!log.canElect(["a", "b", "c"]), "even with an old majority, the quorum must be inside C_new");

  // --- ordering and single-step rules are enforced -------------------------------------------
  const broken = new ConfigLog(oldVoters);
  broken.append({ kind: "config", configuration: joint });
  assert.throws(() => broken.append({ kind: "config", configuration: single }), Error,
    "C_new may not be appended before the joint entry is committed");
  const wide = new ConfigLog(oldVoters);
  assert.throws(() => wide.append({ kind: "config", configuration: config(["a", "b", "c", "d", "e"]) }), Error,
    "two servers cannot be added at once");
  const trimmed = new ConfigLog(oldVoters);
  assert.throws(() => trimmed.append({ kind: "config", configuration: config(["a"]) }), Error,
    "two servers cannot be removed at once");
  const shrinkJoint = trimmed.append({ kind: "config", configuration: config(oldVoters, ["a", "b"]) });
  assert.strictEqual(trimmed.commit(shrinkJoint, ["a", "b"]), true, "a shrink moves through joint consensus");
  assert.strictEqual(trimmed.commit(trimmed.append({ kind: "config", configuration: config(["a", "b"]) }), ["a", "b"]), true);
  assert.ok(!trimmed.canElect(["b"]), "the surviving pair needs both votes");

  // --- no two leaders can be elected with disjoint acknowledgement sets ---------------------
  const electionLog = new ConfigLog(oldVoters);
  electionLog.append({ kind: "config", configuration: joint });
  let splitBrain = 0;
  for (const left of jointFamilies) {
    for (const right of jointFamilies) if (!intersects(left.union, right.union)) splitBrain++;
  }
  assert.strictEqual(splitBrain, 0, "no two joint-majority elections can be disjoint");
  const votesA = jointFamilies[0].union;
  const votesB = jointFamilies[1].union;
  assert.ok(electionLog.canElect(votesA) && electionLog.canElect(votesB));
  assert.ok(intersects(votesA, votesB), "and they necessarily share a voter, so one of them voted twice");
  const acksWithoutOverlap = ["d", "e"];
  assert.ok(!electionLog.canElect(acksWithoutOverlap), "the C_new-only side cannot elect against the joint config");

  // --- boundaries: one voter, degenerate joint config, randomized quorum sampling -----------
  const solo = new ConfigLog(["only"]);
  assert.strictEqual(solo.canElect(["only"]), true);
  assert.strictEqual(solo.canElect([]), false);
  const same = config(["a", "b", "c"], ["a", "b", "c"]);
  for (const family of jointQuorums(same)) assert.ok(hasQuorum(family.union, same), "a degenerate joint config is its own quorum");
  let seed = 20260101;
  let sampled = 0;
  for (let trial = 0; trial < 40; trial++) {
    seed = lcg(seed);
    const size = 2 + (seed % 4);
    const voters = ["v0", "v1", "v2", "v3", "v4"].slice(0, size);
    const all = quorums(voters);
    seed = lcg(seed);
    const first = all[seed % all.length];
    seed = lcg(seed);
    const second = all[seed % all.length];
    assert.ok(intersects(first, second), `trial ${trial}: ${label(first)} vs ${label(second)}`);
    const acked = voters.filter((voter) => first.includes(voter));
    assert.ok(hasQuorum(acked, config(voters)), `trial ${trial}: a quorum commits`);
    if (first.length === Math.floor(size / 2) + 1) {
      assert.ok(!hasQuorum(acked.slice(0, acked.length - 1), config(voters)), `trial ${trial}: one short fails`);
    }
    sampled++;
  }

  console.log(`joint quorums=${jointFamilies.length} unsafe-direct-pairs=${unsafePairs} sampled=${sampled}`);
  console.log(`config entries committed=${log.committedConfigurations().length} split-brain elections=${splitBrain}`);
  console.log("PASS");
}
