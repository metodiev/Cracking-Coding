/**
 * Task 350 - Thread-safe singleton via double-checked locking. (Tier: Intermediate, CTCI Ch 15)
 *
 * Node runs one thread per process: there are no shared-memory races, so double-checked locking is
 * not needed for correctness.  The single-threaded equivalent is shown below - lazy creation with
 * a guarded check-then-set that cannot interleave because synchronous JavaScript runs to the next
 * await without preemption; the DCL shape is kept so it can be compared with the Java/C# files.
 *
 * The real Node singleton idiom is the module cache: `require` returns the same `module.exports`
 * object to every caller, so a module-level instance is process-wide - asserted below by
 * re-requiring this very file and checking identity.  `instantiationCount` proves the constructor
 * ran exactly once even though many tasks raced to obtain the instance.
 */
"use strict";

let lazyInstantiations = 0;

class Singleton {
  constructor() {
    lazyInstantiations++;
  }

  static getInstance() {
    if (Singleton.instance === undefined) {
      // A threaded runtime would take a lock here and re-check inside it (that is the
      // "double-checked" part); single-threaded JavaScript cannot interleave here at all.
      Singleton.instance = new Singleton();
    }
    return Singleton.instance;
  }
}

let moduleLevelInstantiations = 0;

class ModuleLevelSingleton {
  constructor() {
    moduleLevelInstantiations++;
  }
}

const moduleSingleton = new ModuleLevelSingleton();   // the module cache makes this the only one

module.exports = {
  Singleton,
  ModuleLevelSingleton,
  moduleSingleton,
  counts: () => ({ lazy: lazyInstantiations, moduleLevel: moduleLevelInstantiations }),
};

if (require.main === module) {
  const main = async () => {
    const assert = require("node:assert");

    assert.deepStrictEqual(module.exports.counts(), { lazy: 0, moduleLevel: 1 },
      "lazy singleton untouched, module-level singleton already constructed");

    // many "concurrent" callers: 64 async tasks all released through one gate
    let release;
    const start = new Promise((resolve) => { release = resolve; });
    const results = [];
    const tasks = [];
    for (let i = 0; i < 64; i++) {
      tasks.push((async () => {
        await start;
        results.push(Singleton.getInstance());
      })());
    }
    release();
    await Promise.all(tasks);

    assert.strictEqual(results.length, 64, "all callers finished");
    for (const result of results) assert.strictEqual(result, results[0], "every caller got the same instance");
    assert.strictEqual(module.exports.counts().lazy, 1, "the constructor ran exactly once");
    assert.strictEqual(Singleton.getInstance(), results[0], "later callers reuse the instance");

    // module cache: require of this very file returns the same exports object
    const again = require(module.filename);
    assert.strictEqual(again, module.exports, "require returned the cached module");
    assert.strictEqual(again.moduleSingleton, moduleSingleton, "module-level singleton identity across requires");
    assert.strictEqual(again.counts().moduleLevel, 1, "the module-level constructor also ran exactly once");
  };

  main().then(() => console.log("PASS"), (err) => {
    console.error(err);
    process.exit(1);
  });
}
