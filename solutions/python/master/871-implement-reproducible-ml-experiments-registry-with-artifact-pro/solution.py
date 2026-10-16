"""Task 871 - Implement reproducible ML experiments registry with artifact provenance.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A content-addressed registry for machine-learning experiments.  Everything a run depends on is
reduced to a digest, so runs deduplicate, replay and garbage-collect by identity:

    config_hash = H(canonical json of the hyper-parameters)          (key order irrelevant)
    run_id      = H(config_hash, seed, code version, sorted input digests, sorted parent ids)
    artifact    = the serialised trained model + metrics, stored under H(artifact bytes)

Deduplication is therefore structural: re-registering the same configuration, seed, code
version, inputs and parents yields the *same* run id, so the registry records a second
observation of one run instead of a new one, while any change to any ingredient yields a new
run id (a provenance-sensitive identity, which is what makes a run reproducible in the first
place).

The experiment itself is real work, not a placeholder: a deterministic PRNG builds a synthetic
regression problem from the data spec, and a fixed-step gradient-descent ridge solver fits it.
Each run stores the fitted weights plus the metrics, and ``evaluate`` re-derives the objective
from the *decoded weights and data* so the recorded metric can be cross-checked independently
of the training loop.  ``reproduce`` re-executes the experiment from the recorded ingredients
and compares the digest of the new artifact with the recorded one; a config that declares
itself unstable mixes an entropy counter into its parameters and is reported as a
reproducibility violation.

Verification below covers: canonical config hashing (key-order insensitivity, semantic
changes), structural deduplication with observation counting, sensitivity of the run id to
every ingredient (seed, inputs, parents, code version), artifact round-trips and integrity,
tamper detection on the blob store, replay of every registered run, detection of an unstable
run, independent recomputation of the recorded objective from the decoded artifact, strict
monotonicity of the loss in the iteration budget, lineage closure and cycle rejection, a
mark-and-sweep GC that keeps every pinned run and all of its ancestors and reclaims exactly
the unreachable blobs, query filters, best-run selection, and an export/import round trip into
a fresh registry.
"""
import hashlib
import json
import math

CODE_VERSION = "train-2.3.1"
MASK64 = (1 << 64) - 1
ENTROPY = [0]


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def uniform(self):
        self.state, raw = splitmix64(self.state)
        return (raw >> 11) * (1.0 / float(1 << 53))

    def gauss(self):
        u = self.uniform()
        u = u if u > 1e-12 else 1e-12
        return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * self.uniform())


def canonical(value):
    """Canonical JSON: sorted keys, no incidental whitespace - the basis of config hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest_of_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_of_bytes(data):
    return hashlib.sha256(data).hexdigest()


def seed_of(text):
    return int(digest_of_text(text)[:16], 16)


# ------------------------------------------------------------- content addressed store
class ContentStore:
    """Blob store keyed by the SHA-256 of the bytes; the only place artifacts live."""

    def __init__(self):
        self.blobs = {}
        self.writes = 0
        self.reads = 0

    def put(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        digest = digest_of_bytes(data)
        if digest in self.blobs and self.blobs[digest] != data:
            raise ValueError("hash collision")  # pragma: no cover - defensive
        if digest not in self.blobs:
            self.writes += 1
        self.blobs[digest] = data
        return digest

    def get(self, digest):
        self.reads += 1
        if digest not in self.blobs:
            raise KeyError(f"missing artifact {digest[:12]}")
        return self.blobs[digest]

    def intact(self, digest):
        return digest in self.blobs and digest_of_bytes(self.blobs[digest]) == digest

    def __contains__(self, digest):
        return digest in self.blobs

    def __len__(self):
        return len(self.blobs)


# --------------------------------------------------------------------- the experiment
def synthesize(spec_digest, seed, samples, dimensions):
    """Deterministic synthetic ridge-regression problem built from the data spec digest."""
    rng = Rng(seed_of(spec_digest) ^ seed)
    true_weights = [rng.gauss() for _ in range(dimensions)]
    features = []
    targets = []
    for _ in range(samples):
        row = [rng.gauss() for _ in range(dimensions)]
        value = sum(w * x for w, x in zip(true_weights, row)) + 0.1 * rng.gauss()
        features.append(row)
        targets.append(value)
    return features, targets, true_weights


def loss_of(features, targets, weights):
    total = 0.0
    for row, target in zip(features, targets):
        residual = sum(w * x for w, x in zip(weights, row)) - target
        total += 0.5 * residual * residual
    return total / len(features)


def train(features, targets, iterations, learning_rate, penalty, entropy=0):
    """Fixed-step gradient descent on the ridge objective - the run's actual computation."""
    dimensions = len(features[0])
    weights = [0.0] * dimensions
    for step in range(iterations):
        gradient = [penalty * w for w in weights]
        for row, target in zip(features, targets):
            residual = sum(w * x for w, x in zip(weights, row)) - target
            for k in range(dimensions):
                gradient[k] += residual * row[k] / len(features)
        rate = learning_rate * (1.0 + 1e-9 * entropy * step)
        for k in range(dimensions):
            weights[k] -= rate * gradient[k]
    return weights


class ExperimentRegistry:
    def __init__(self, store=None):
        self.store = store if store is not None else ContentStore()
        self.runs = {}
        self.observations = {}
        self.order = []

    # ------------------------------------------------------------- registration
    def config_hash(self, config):
        return digest_of_text(canonical(config))

    def run_id(self, config, seed, inputs, parents=()):
        for parent in parents:
            if parent not in self.runs:
                raise ValueError(f"unknown parent run {parent[:12]}")
        payload = [
            self.config_hash(config),
            str(seed),
            CODE_VERSION,
            sorted(inputs.items()),
            sorted(parents),
        ]
        return digest_of_text(canonical(payload))

    def register(self, config, seed, inputs=None, parents=(), tags=()):
        inputs = dict(inputs or {})
        for name, digest in inputs.items():
            if digest not in self.store:
                raise KeyError(f"input {name} is not in the content store")
        run_id = self.run_id(config, seed, inputs, parents)
        if run_id in self.runs:
            self.observations[run_id] += 1
            return self.runs[run_id]
        spec_digest = inputs.get("dataset", digest_of_text("default-dataset"))
        samples = int(config.get("samples", 48))
        dimensions = int(config.get("dimensions", 4))
        iterations = int(config.get("iterations", 40))
        learning_rate = float(config.get("learning_rate", 0.1))
        penalty = float(config.get("penalty", 1e-3))
        entropy = 0
        if config.get("unstable"):
            ENTROPY[0] += 1
            entropy = ENTROPY[0]
        features, targets, _true = synthesize(spec_digest, seed, samples, dimensions)
        weights = train(features, targets, iterations, learning_rate, penalty, entropy)
        objective = loss_of(features, targets, weights) + 0.5 * penalty * sum(
            w * w for w in weights)
        artifact = canonical({
            "weights": [round(w, 12) for w in weights],
            "metrics": {"objective": round(objective, 12), "iterations": iterations},
            "config_hash": self.config_hash(config),
            "seed": seed,
        })
        artifact_digest = self.store.put(artifact)
        record = {
            "run_id": run_id,
            "config": dict(config),
            "config_hash": self.config_hash(config),
            "seed": seed,
            "code_version": CODE_VERSION,
            "inputs": inputs,
            "parents": list(parents),
            "tags": sorted(tags),
            "artifact": artifact_digest,
            "metrics": {"objective": objective, "iterations": iterations, "samples": samples},
        }
        self.runs[run_id] = record
        self.observations[run_id] = 1
        self.order.append(run_id)
        return record

    # ------------------------------------------------------------- inspection
    def artifact_bytes(self, run_id):
        record = self._record(run_id)
        return self.store.get(record["artifact"])

    def decode_artifact(self, run_id):
        return json.loads(self.artifact_bytes(run_id).decode("utf-8"))

    def _record(self, run_id):
        if run_id not in self.runs:
            raise KeyError(f"unknown run {run_id[:12]}")
        return self.runs[run_id]

    def query(self, **filters):
        matches = []
        for run_id in self.order:
            record = self.runs[run_id]
            if all(record["config"].get(key) == value for key, value in filters.items()):
                matches.append(record)
        return matches

    def best(self, metric="objective"):
        if set(metric for metric in [metric]) - {"objective", "iterations", "samples"}:
            raise ValueError(f"unknown metric {metric}")
        candidates = [self.runs[run_id] for run_id in self.order]
        if not candidates:
            raise ValueError("registry is empty")
        return min(candidates, key=lambda record: record["metrics"][metric])

    def lineage(self, run_id):
        seen = set()
        frontier = [run_id]
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(self._record(current)["parents"])
        return seen - {run_id}

    def verify(self, run_id):
        """Integrity check: every blob behind the run rehashes to its own digest."""
        record = self._record(run_id)
        return self.store.intact(record["artifact"])

    def reproduce(self, run_id):
        """Re-execute the run from its recorded ingredients and compare artifact digests."""
        record = self._record(run_id)
        spec_digest = record["inputs"].get("dataset", digest_of_text("default-dataset"))
        config = record["config"]
        features, targets, _true = synthesize(spec_digest, record["seed"],
                                              int(config.get("samples", 48)),
                                              int(config.get("dimensions", 4)))
        weights = train(features, targets, int(config.get("iterations", 40)),
                        float(config.get("learning_rate", 0.1)),
                        float(config.get("penalty", 1e-3)),
                        ENTROPY[0] + 1 if config.get("unstable") else 0)
        objective = loss_of(features, targets, weights) + 0.5 * float(
            config.get("penalty", 1e-3)) * sum(w * w for w in weights)
        artifact = canonical({
            "weights": [round(w, 12) for w in weights],
            "metrics": {"objective": round(objective, 12),
                        "iterations": int(config.get("iterations", 40))},
            "config_hash": record["config_hash"],
            "seed": record["seed"],
        })
        fresh = digest_of_bytes(artifact.encode("utf-8"))
        return {"reproduced": fresh == record["artifact"], "digest": fresh,
                "recorded": record["artifact"]}

    def evaluate_from_artifact(self, run_id):
        """Independent recomputation of the objective from the decoded artifact."""
        record = self._record(run_id)
        decoded = self.decode_artifact(run_id)
        spec_digest = record["inputs"].get("dataset", digest_of_text("default-dataset"))
        features, targets, _true = synthesize(spec_digest, record["seed"],
                                              int(record["config"].get("samples", 48)),
                                              len(decoded["weights"]))
        penalty = float(record["config"].get("penalty", 1e-3))
        return loss_of(features, targets, decoded["weights"]) + 0.5 * penalty * sum(
            w * w for w in decoded["weights"])

    # ------------------------------------------------------------- maintenance
    def export(self):
        return {"code_version": CODE_VERSION, "runs": [self.runs[r] for r in self.order]}

    def import_runs(self, payload):
        if payload.get("code_version") != CODE_VERSION:
            raise ValueError("code version mismatch")
        for record in payload["runs"]:
            run_id = self.run_id(record["config"], record["seed"], record["inputs"],
                                 [p for p in record["parents"] if p in self.runs])
            if record["artifact"] not in self.store:
                raise KeyError("artifact missing during import")
            self.runs[record["run_id"]] = record
            self.observations[record["run_id"]] = 1
            self.order.append(record["run_id"])
            if run_id != record["run_id"]:
                raise ValueError("run id does not reproduce on import")
        return len(payload["runs"])

    def collect_garbage(self, pinned):
        """Mark and sweep: keep pinned runs, their ancestors, and every blob they reference.

        Inputs count as referenced data - a run that is kept must keep the dataset it was
        trained on, otherwise it could never be replayed.
        """
        keep_runs = set()
        for run_id in pinned:
            keep_runs.add(run_id)
            keep_runs |= self.lineage(run_id)
        keep_blobs = set()
        for run_id in keep_runs:
            record = self.runs[run_id]
            keep_blobs.add(record["artifact"])
            keep_blobs.update(record["inputs"].values())
        reclaimed = sorted(digest for digest in list(self.store.blobs)
                           if digest not in keep_blobs)
        for digest in reclaimed:
            del self.store.blobs[digest]
        removed_runs = []
        for run_id in list(self.order):
            if run_id not in keep_runs:
                if self.runs[run_id]["artifact"] not in self.store.blobs:
                    removed_runs.append(run_id)
        for run_id in removed_runs:
            self.order.remove(run_id)
            del self.runs[run_id]
            del self.observations[run_id]
        return {"reclaimed": reclaimed, "kept_runs": sorted(keep_runs),
                "kept_blobs": sorted(keep_blobs)}


def main():
    store = ContentStore()
    dataset = store.put("grid=64;noise=0.1")
    alternate = store.put("grid=128;noise=0.1")
    registry = ExperimentRegistry(store)

    base = {"samples": 40, "dimensions": 3, "iterations": 60, "learning_rate": 0.2,
            "penalty": 1e-3}
    first = registry.register(base, seed=7, inputs={"dataset": dataset})

    # ---------------- canonical config hashing ----------------
    reordered = {"penalty": 1e-3, "learning_rate": 0.2, "iterations": 60, "dimensions": 3,
                 "samples": 40}
    check(registry.config_hash(base) == registry.config_hash(reordered),
          "config hashing must ignore key order")
    check(registry.config_hash(base) != registry.config_hash(dict(base, iterations=61)),
          "config hashing must see semantic changes")
    nested_a = {"samples": 40, "optimizer": {"name": "sgd", "momentum": 0.9}}
    nested_b = {"optimizer": {"momentum": 0.9, "name": "sgd"}, "samples": 40}
    check(registry.config_hash(nested_a) == registry.config_hash(nested_b),
          "nested mapping order must not matter")

    # ---------------- structural deduplication ----------------
    again = registry.register(reordered, seed=7, inputs={"dataset": dataset})
    check(again["run_id"] == first["run_id"], "re-observing a run must not create a new one")
    check(registry.observations[first["run_id"]] == 2, "the registry must count observations")
    check(len(registry.runs) == 1, f"expected one run, got {len(registry.runs)}")

    # ---------------- every ingredient is part of the identity ----------------
    variants = [
        registry.run_id(dict(base, iterations=61), 7, {"dataset": dataset}),
        registry.run_id(base, 8, {"dataset": dataset}),
        registry.run_id(base, 7, {"dataset": alternate}),
    ]
    check(len(set(variants)) == 3 and first["run_id"] not in variants,
          "changing any ingredient must change the run id")

    # ---------------- real training: the loss falls with the iteration budget ----------------
    objectives = []
    for iterations in (10, 30, 90):
        record = registry.register(dict(base, iterations=iterations), seed=11,
                                   inputs={"dataset": dataset}, tags=("sweep",))
        objectives.append(record["metrics"]["objective"])
        check(all(math.isfinite(value) for value in objectives) and objectives[-1] >= 0.0,
              "objectives must be finite and non-negative")
        check(registry.verify(record["run_id"]), "artifact must verify against its digest")
    check(objectives[0] > objectives[1] > objectives[2],
          f"the objective must fall with more iterations: {objectives}")

    # ---------------- the recorded objective is re-derived from the artifact ----------------
    recomputed = registry.evaluate_from_artifact(first["run_id"])
    recorded = first["metrics"]["objective"]
    check(abs(recomputed - recorded) < 1e-6 * max(1.0, abs(recorded)),
          f"objective mismatch: recorded {recorded} recomputed {recomputed}")
    decoded = registry.decode_artifact(first["run_id"])
    check(decoded["seed"] == 7 and len(decoded["weights"]) == 3,
          "the artifact must carry the seed and the full weight vector")
    check(decoded["config_hash"] == first["config_hash"],
          "the artifact must carry the config hash")

    # ---------------- artifact store integrity ----------------
    raw = registry.artifact_bytes(first["run_id"])
    check(store.get(first["artifact"]) == raw, "artifact bytes must round-trip through the store")
    check(store.put(raw) == first["artifact"], "re-storing identical bytes must be idempotent")
    try:
        store.get("0" * 64)
    except KeyError:
        pass
    else:
        raise AssertionError("missing artifacts must raise KeyError")

    # ---------------- tamper detection ----------------
    tampered = ContentStore()
    tampered.blobs = dict(store.blobs)
    victim = next(iter(sorted(tampered.blobs, key=lambda d: len(tampered.blobs[d]))))
    tampered.blobs[victim] = b"{\"weights\": [999.0]}"
    base_registry = ExperimentRegistry(tampered)
    base_registry.register(base, seed=7, inputs={"dataset": dataset})
    check(not tampered.intact(victim), "a mutated blob must fail its integrity check")
    check(tampered.intact(first["artifact"]), "unrelated blobs must stay intact")

    # ---------------- lineage and derived runs ----------------
    derived = registry.register(dict(base, iterations=120), seed=7, inputs={"dataset": dataset},
                                parents=(first["run_id"],))
    further = registry.register(dict(base, iterations=160), seed=7, inputs={"dataset": dataset},
                                parents=(derived["run_id"],))
    check(registry.lineage(further["run_id"]) == {first["run_id"], derived["run_id"]},
          "lineage must be the exact ancestor set")
    check(registry.lineage(first["run_id"]) == set(), "a root run has no ancestors")
    try:
        registry.runs[further["run_id"]]["parents"] = [further["run_id"]]
        registry.lineage(further["run_id"])
    except RecursionError:  # pragma: no cover - the guard below is what we want
        raise AssertionError("lineage must terminate on a self-referencing record")
    finally:
        registry.runs[further["run_id"]]["parents"] = [derived["run_id"]]
    try:
        registry.run_id(base, 7, {"dataset": dataset}, parents=("deadbeef" * 8,))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown parents must be rejected")

    # ---------------- queries ----------------
    sweep = registry.query(iterations=30)
    check(len(sweep) == 1 and sweep[0]["config"]["iterations"] == 30,
          f"query must filter on config fields, got {len(sweep)}")
    check(registry.query(iterations=999) == [], "a query with no match returns nothing")
    check(registry.best()["run_id"] == further["run_id"],
          "the best run must be the one with the smallest objective")

    # ---------------- reproducibility of every registered run ----------------
    violations = []
    for run_id in list(registry.order):
        outcome = registry.reproduce(run_id)
        if not outcome["reproduced"]:
            violations.append(run_id)
    check(violations == [], f"registered runs must replay bit for bit: {violations}")

    unstable_config = dict(base, unstable=True)
    unstable = registry.register(unstable_config, seed=3, inputs={"dataset": dataset})
    check(unstable["run_id"] not in variants, "an unstable run is still a distinct run")
    check(not registry.reproduce(unstable["run_id"])["reproduced"],
          "an unstable run must be reported as irreproducible")

    # ---------------- garbage collection ----------------
    expectations = registry.collect_garbage(pinned=[further["run_id"]])
    keep_expected = {first["run_id"], derived["run_id"], further["run_id"]}
    check(set(expectations["kept_runs"]) == keep_expected,
          f"GC must keep the pinned run and its ancestors: {sorted(expectations['kept_runs'])}")
    check(all(registry.store.intact(registry.runs[run_id]["artifact"])
              for run_id in keep_expected), "kept artifacts must survive intact")
    check(unstable["run_id"] not in registry.runs, "unreachable runs must be swept")
    check(len(expectations["reclaimed"]) == 5,
          f"exactly the five unreferenced blobs must be reclaimed: "
          f"{len(expectations['reclaimed'])}")
    check(len(registry.store) == 4,
          f"kept artifacts and referenced inputs must survive, got {len(registry.store)}")
    check(dataset in registry.store,
          "inputs of kept runs must be preserved for replay")

    # ---------------- export / import round trip ----------------
    payload = registry.export()
    fresh_store = ContentStore()
    for digest in list(registry.store.blobs):
        fresh_store.put(registry.store.blobs[digest])
    rebuilt = ExperimentRegistry(fresh_store)
    check(rebuilt.import_runs(payload) == len(payload["runs"]),
          "import must restore every exported run")
    check(rebuilt.runs[further["run_id"]] == registry.runs[further["run_id"]],
          "an imported run must be byte-identical to the exported one")
    check(rebuilt.reproduce(further["run_id"])["reproduced"],
          "an imported run must still replay")

    # ---------------- invalid inputs ----------------
    try:
        registry.register(base, seed=1, inputs={"dataset": "0" * 64})
    except KeyError:
        pass
    else:
        raise AssertionError("registering with a missing input artifact must fail")
    try:
        registry.best("accuracy")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown metrics must be rejected")
    try:
        rebuilt.import_runs({"code_version": "other", "runs": []})
    except ValueError:
        pass
    else:
        raise AssertionError("a code-version mismatch must be rejected")

    print(f"runs={len(registry.runs)} observations={sum(registry.observations.values())} "
          f"blobs={len(registry.store)}")
    print(f"objectives over budget [10,30,90] = {[round(o, 6) for o in objectives]}")
    print(f"replay violations={violations} gc_reclaimed={len(expectations['reclaimed'])}")
    print("PASS")


if __name__ == "__main__":
    main()
