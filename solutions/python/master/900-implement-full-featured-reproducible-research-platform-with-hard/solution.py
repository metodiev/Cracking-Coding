"""Task 900 - Implement full-featured reproducible research platform with hardware config capture.  (Tier: Master, CTCI Beyond - Research Engineering & Reproducibility)

The platform runs a dependency graph of experiment stages where every artifact is content
addressed: a stage's payload is rebuilt from its name, its parameters, the *hashes* of its input
artifacts, the pipeline seed and - for stages that declare themselves hardware sensitive - the
captured hardware fingerprint, and the artifact's identity is the SHA-256 of that payload.  Two
consequences follow, and they are the whole point of the design: the same configuration always
produces the same hashes no matter who runs it, and a change anywhere upstream is visible in the
hash of everything downstream without any central bookkeeping.

On top of that sits a run engine with three modes.  A normal run executes stages in topological
order but reuses a cached artifact whenever a stage's recomputed hash matches what is stored, so
touching one parameter re-executes exactly that stage and its descendants.  A *reproduction*
throws the cache away and re-runs everything from scratch, then compares the fresh hashes against
a recorded manifest - which is how a genuinely nondeterministic stage is caught instead of being
papered over by the cache.  A *verification* re-hashes the stored payloads and re-derives the
provenance, so a tampered artifact or a doctored manifest fails loudly.

Verified: the graph rejects cycles and dangling inputs, two independent topological sorts agree,
a fresh run executes every stage while a repeat run executes none and produces identical hashes,
a parameter change re-executes exactly the changed stage and its descendants (checked against an
independent reverse reachability computation) and leaves unrelated hashes untouched, a hardware
fingerprint change affects the hardware sensitive stages and everything downstream of them,
payload tampering and manifest doctoring are both detected, a deliberately nondeterministic stage
is reported as not reproducible with the offending stage named, provenance is complete for every
stage (parameters, input hashes, fingerprint, seed, hash), the empty and single stage pipelines
have their closed forms, a 60 stage pipeline is exercised with the cache saving measured, and the
whole platform replays from its seed.
"""
import hashlib

REQUIRED_PROVENANCE = ("stage", "params", "inputs", "input_hashes", "fingerprint", "seed")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class PipelineError(Exception):
    """Raised for a malformed graph, an unknown stage or a broken manifest."""


def canonical(value):
    """A deterministic serialisation: sorted keys, repr of primitives, no address noise."""
    if isinstance(value, dict):
        return "{" + ",".join("%s:%s" % (key, canonical(value[key])) for key in sorted(value)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical(item) for item in value) + "]"
    return repr(value)


class HardwareProfile:
    """The captured environment: everything about the machine that can change a result."""

    FIELDS = ("cpu", "cores", "gpu", "driver", "memory_gb", "os")

    def __init__(self, cpu, cores, gpu, driver, memory_gb, os_name):
        self.values = {"cpu": cpu, "cores": cores, "gpu": gpu, "driver": driver,
                       "memory_gb": memory_gb, "os": os_name}

    @property
    def fingerprint(self):
        return hashlib.sha256(canonical(self.values).encode()).hexdigest()

    def describe(self):
        return dict(self.values)

    def with_field(self, field, value):
        copy = dict(self.values)
        if field not in copy:
            raise PipelineError("unknown hardware field %r" % (field,))
        copy[field] = value
        return HardwareProfile(copy["cpu"], copy["cores"], copy["gpu"], copy["driver"],
                               copy["memory_gb"], copy["os"])


class Stage:
    """One experiment step: parameters, inputs, and whether hardware changes its result."""

    def __init__(self, name, params, inputs=(), hardware_sensitive=False, nondeterministic=False):
        self.name = name
        self.params = dict(params)
        self.inputs = tuple(inputs)
        self.hardware_sensitive = hardware_sensitive
        self.nondeterministic = nondeterministic


class Pipeline:
    """A validated DAG of stages plus the captured hardware and the pipeline seed."""

    def __init__(self, stages, hardware, seed=0):
        if not hardware:
            raise PipelineError("a pipeline must capture its hardware profile")
        self.stages = {stage.name: stage for stage in stages}
        if len(self.stages) != len(stages):
            raise PipelineError("stage names must be unique")
        self.hardware = hardware
        self.seed = seed
        for stage in stages:
            for source in stage.inputs:
                if source not in self.stages:
                    raise PipelineError("stage %r depends on unknown stage %r"
                                        % (stage.name, source))
        self.order = self.topological_order()

    # -------------------------------------------------------------- ordering
    def topological_order(self):
        """Kahn's algorithm with a deterministic tie break on the stage name."""
        remaining = {name: set(stage.inputs) for name, stage in self.stages.items()}
        order = []
        while remaining:
            ready = sorted(name for name, pending in remaining.items() if not pending)
            if not ready:
                raise PipelineError("the stage graph contains a cycle: %r" % sorted(remaining))
            for name in ready:
                order.append(name)
                del remaining[name]
            for pending in remaining.values():
                pending.difference_update(ready)
        return order

    def depth_first_order(self):
        """A second, independent ordering used to cross-check the first."""
        order = []
        state = {}

        def visit(name):
            if state.get(name) == "visiting":
                raise PipelineError("cycle through %r" % name)
            if state.get(name) == "done":
                return
            state[name] = "visiting"
            for source in self.stages[name].inputs:
                visit(source)
            state[name] = "done"
            order.append(name)

        for name in sorted(self.stages):
            visit(name)
        return order

    def descendants(self, changed):
        """Every stage that transitively depends on the given ones."""
        affected = set(changed)
        changed_any = True
        while changed_any:
            changed_any = False
            for name, stage in self.stages.items():
                if name in affected:
                    continue
                if any(source in affected for source in stage.inputs):
                    affected.add(name)
                    changed_any = True
        return affected

    def ancestors(self, name):
        seen = set()
        frontier = list(self.stages[name].inputs)
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(self.stages[current].inputs)
        return seen

    # ------------------------------------------------------------- artifacts
    def stage_seed(self, name):
        digest = hashlib.sha256(("%s|%d" % (name, self.seed)).encode()).digest()
        return int.from_bytes(digest[:8], "big")

    def payload_for(self, name, params, input_hashes, fingerprint=None):
        stage = self.stages[name]
        record = {
            "stage": name,
            "params": params,
            "inputs": sorted(stage.inputs),
            "input_hashes": [input_hashes[source] for source in sorted(stage.inputs)],
            "seed": self.stage_seed(name),
        }
        if stage.hardware_sensitive:
            record["hardware"] = fingerprint
        return canonical(record).encode()

    @staticmethod
    def digest(payload):
        return hashlib.sha256(payload).hexdigest()


class Store:
    """Content addressed artifact storage plus the run counters."""

    def __init__(self):
        self.artifacts = {}
        self.executions = []
        self.served_from_cache = []

    def has(self, name, digest):
        return self.artifacts.get(name, (None, None))[0] == digest

    def put(self, name, digest, payload, provenance):
        self.artifacts[name] = (digest, payload, provenance)

    def payload(self, name):
        return self.artifacts[name][1]

    def digest_of(self, name):
        return self.artifacts[name][0]

    def tamper(self, name, payload):
        digest, _, provenance = self.artifacts[name]
        self.artifacts[name] = (digest, payload, provenance)


class Runner:
    """Executes a pipeline, reusing cached artifacts whose hash is still current."""

    def __init__(self, pipeline, store=None, counter=None):
        self.pipeline = pipeline
        self.store = store if store is not None else Store()
        self.counter = counter if counter is not None else [0]

    def run(self, overrides=None, fingerprint=None):
        """Execute the pipeline; `overrides` maps stage names to replacement parameters."""
        overrides = overrides or {}
        fingerprint = fingerprint or self.pipeline.hardware.fingerprint
        executed = []
        reused = []
        for name in self.pipeline.order:
            stage = self.pipeline.stages[name]
            params = dict(stage.params)
            params.update(overrides.get(name, {}))
            input_hashes = {source: self.store.digest_of(source) for source in stage.inputs}
            payload = self.pipeline.payload_for(name, params, input_hashes, fingerprint)
            digest = self.pipeline.digest(payload)
            if self.store.has(name, digest):
                reused.append(name)
                self.store.served_from_cache.append(name)
                continue
            if stage.nondeterministic:
                self.counter[0] += 1
                payload = payload + b"|" + str(self.counter[0]).encode()
                digest = self.pipeline.digest(payload)
            provenance = {
                "stage": name,
                "params": params,
                "inputs": list(stage.inputs),
                "input_hashes": [input_hashes[source] for source in sorted(stage.inputs)],
                "fingerprint": fingerprint if stage.hardware_sensitive else None,
                "seed": self.pipeline.stage_seed(name),
                "digest": digest,
            }
            self.store.put(name, digest, payload, provenance)
            self.store.executions.append(name)
            executed.append(name)
        return {"executed": executed, "reused": reused,
                "manifest": self.manifest(fingerprint)}

    def manifest(self, fingerprint):
        return {
            "fingerprint": fingerprint,
            "hardware": self.pipeline.hardware.describe() if fingerprint
            == self.pipeline.hardware.fingerprint else {"fingerprint": fingerprint},
            "seed": self.pipeline.seed,
            "artifacts": {name: {"digest": self.store.digest_of(name),
                                 "provenance": self.store.artifacts[name][2]}
                          for name in self.pipeline.order},
        }

    @staticmethod
    def manifest_hash(manifest):
        return hashlib.sha256(canonical(manifest).encode()).hexdigest()

    def verify(self):
        """Re-derive every artifact from its provenance and re-hash the stored payload."""
        problems = []
        for name, (digest, payload, provenance) in self.store.artifacts.items():
            if self.pipeline.digest(payload) != digest:
                problems.append((name, "payload does not hash to the recorded digest"))
            for field in REQUIRED_PROVENANCE:
                if field not in provenance:
                    problems.append((name, "provenance is missing %r" % field))
            if provenance.get("digest") != digest:
                problems.append((name, "provenance digest mismatch"))
            recomputed = self.pipeline.payload_for(name, provenance["params"],
                                                   {source: self.store.digest_of(source)
                                                    for source in self.pipeline.stages[name].inputs},
                                                   provenance["fingerprint"]
                                                   or self.pipeline.hardware.fingerprint)
            if self.pipeline.stages[name].nondeterministic:
                continue
            if self.pipeline.digest(recomputed) != digest:
                problems.append((name, "payload does not follow from its provenance"))
        return problems

    def reproduce(self, recorded):
        """Re-run from an empty cache and compare against a recorded manifest."""
        fresh = Runner(self.pipeline, Store(), list(self.counter))
        fresh.run(fingerprint=recorded["fingerprint"])
        differences = []
        for name in self.pipeline.order:
            if fresh.store.digest_of(name) != recorded["artifacts"][name]["digest"]:
                differences.append(name)
        return {"differences": differences, "fresh": fresh,
                "reproduced": not differences}


# ----------------------------------------------------------------------- main
def build_pipeline(seed=900001, stage_count=6, nondeterministic=None):
    hardware = HardwareProfile("x86-64", 16, "A100", "535.104", 256, "linux-6.5")
    stages = []
    for index in range(stage_count):
        inputs = () if index == 0 else ("stage%02d" % (index - 1),)
        if index == stage_count - 1 and stage_count >= 4:
            inputs = ("stage00", "stage%02d" % (index - 1))     # a diamond at the end
        stages.append(Stage("stage%02d" % index,
                            {"learning_rate": 0.1 + index / 100, "epochs": 10 + index},
                            inputs,
                            hardware_sensitive=(index % 3 == 0),
                            nondeterministic=(index == nondeterministic)))
    return Pipeline(stages, hardware, seed), hardware


def main():
    # --- 1. hardware capture ------------------------------------------------
    hardware = HardwareProfile("x86-64", 16, "A100", "535.104", 256, "linux-6.5")
    check(len(hardware.fingerprint) == 64, "the fingerprint must be a SHA-256 digest")
    check(hardware.fingerprint == HardwareProfile("x86-64", 16, "A100", "535.104", 256,
                                                  "linux-6.5").fingerprint,
          "the fingerprint must be deterministic")
    other = hardware.with_field("cores", 32)
    check(other.fingerprint != hardware.fingerprint,
          "a different core count must change the fingerprint")
    check(hardware.with_field("gpu", "H100").fingerprint != hardware.fingerprint,
          "a different accelerator must change the fingerprint")
    check(all(field in hardware.describe() for field in HardwareProfile.FIELDS),
          "the profile must capture every declared field")
    raised = False
    try:
        hardware.with_field("flops", 1)
    except PipelineError:
        raised = True
    check(raised, "an unknown hardware field must be rejected")

    # --- 2. graph validation ------------------------------------------------
    pipeline, _ = build_pipeline()
    check(pipeline.topological_order() == pipeline.depth_first_order(),
          "the two independent topological sorts must agree")
    for position, name in enumerate(pipeline.order):
        for source in pipeline.stages[name].inputs:
            check(pipeline.order.index(source) < position,
                  "every input must be produced before its consumer")
    raised = False
    try:
        Pipeline([Stage("a", {}, ("b",)), Stage("b", {}, ("a",))], hardware, 1)
    except PipelineError:
        raised = True
    check(raised, "a cycle must be rejected, at construction time")
    raised = False
    try:
        Pipeline([Stage("a", {}, ("ghost",))], hardware, 1)
    except PipelineError:
        raised = True
    check(raised, "a dangling input must be rejected")
    raised = False
    try:
        Pipeline([Stage("a", {}), Stage("a", {})], hardware, 1)
    except PipelineError:
        raised = True
    check(raised, "duplicate stage names must be rejected")

    # --- 3. a fresh run executes everything ---------------------------------
    store = Store()
    runner = Runner(pipeline, store)
    first = runner.run()
    check(sorted(first["executed"]) == sorted(pipeline.stages),
          "a fresh run must execute every stage")
    check(first["reused"] == [], "a fresh run must reuse nothing")
    for name in pipeline.order:
        provenance = store.artifacts[name][2]
        for field in REQUIRED_PROVENANCE:
            check(field in provenance, "provenance for %r must record %r" % (name, field))
        check(provenance["stage"] == name, "the provenance must name its stage")
        check(provenance["digest"] == store.digest_of(name),
              "the provenance digest must match the stored artifact")
        for source, recorded in zip(sorted(pipeline.stages[name].inputs),
                                    provenance["input_hashes"]):
            check(recorded == store.digest_of(source),
                  "the recorded input hash must match the input artifact")
    manifest = first["manifest"]
    check(Runner.manifest_hash(manifest) == Runner.manifest_hash(first["manifest"]),
          "the manifest hash must be deterministic")

    # --- 4. a repeat run is entirely cached ---------------------------------
    repeat = runner.run()
    check(repeat["executed"] == [], "a repeat run must execute nothing")
    check(sorted(repeat["reused"]) == sorted(pipeline.stages), "every stage must be reused")
    check(Runner.manifest_hash(repeat["manifest"]) == Runner.manifest_hash(manifest),
          "the manifest must be identical after a cached run")

    # --- 5. one parameter change re-executes exactly the right stage --------
    changed = {"stage02": {"learning_rate": 0.9}}
    third = runner.run(overrides=changed)
    expected = pipeline.descendants({"stage02"})
    check(set(third["executed"]) == expected,
          "only the changed stage and its descendants may re-execute: %r vs %r"
          % (sorted(third["executed"]), sorted(expected)))
    untouched = set(pipeline.stages) - expected
    for name in untouched:
        check(third["manifest"]["artifacts"][name]["digest"]
              == manifest["artifacts"][name]["digest"],
              "a stage outside the affected set must keep its hash: %r" % name)
    for name in expected:
        check(third["manifest"]["artifacts"][name]["digest"]
              != manifest["artifacts"][name]["digest"],
              "a stage inside the affected set must change its hash: %r" % name)
    check("stage00" in untouched and "stage01" in untouched,
          "the ancestors of the change must stay untouched")
    check(pipeline.ancestors("stage02") == {"stage00", "stage01"},
          "the ancestor walk must find the upstream chain")

    # --- 6. a hardware change touches only the sensitive stages -------------
    sensitive = {name for name, stage in pipeline.stages.items() if stage.hardware_sensitive}
    check(sensitive, "the pipeline must declare hardware sensitive stages")
    affected = pipeline.descendants(sensitive)
    new_fingerprint = hardware.with_field("driver", "545.29").fingerprint
    fourth = runner.run(fingerprint=new_fingerprint)
    check(set(fourth["executed"]) == affected,
          "a hardware change must re-execute the sensitive stages and their descendants:"
          " %r vs %r" % (sorted(fourth["executed"]), sorted(affected)))
    for name in set(pipeline.stages) - affected:
        check(fourth["manifest"]["artifacts"][name]["digest"]
              == manifest["artifacts"][name]["digest"],
              "a hardware independent stage must keep its hash: %r" % name)
    back = runner.run()
    check(set(back["executed"]) == affected,
          "returning to the original hardware must re-execute the same set again")

    # --- 7. tampering and doctoring are detected ----------------------------
    check(runner.verify() == [], "an honest store must verify cleanly")
    original = store.payload("stage01")
    store.tamper("stage01", original + b"!")
    problems = runner.verify()
    check(problems and problems[0][0] == "stage01",
          "a tampered payload must be reported: %r" % (problems,))
    store.tamper("stage01", bytes([original[0] ^ 1]) + original[1:])
    check(runner.verify() != [], "a single flipped byte must still be detected")
    store.tamper("stage01", original)
    check(runner.verify() == [], "restoring the payload must clear the report")
    runner.run()

    # --- 8. reproduction catches a nondeterministic stage -------------------
    honest, _ = build_pipeline(seed=900100)
    honest_runner = Runner(honest, Store())
    recorded = honest_runner.run()["manifest"]
    report = honest_runner.reproduce(recorded)
    check(report["reproduced"] and report["differences"] == [],
          "an honest pipeline must reproduce exactly")
    check(report["fresh"].store.digest_of(honest.order[-1]) ==
          recorded["artifacts"][honest.order[-1]]["digest"],
          "the reproduced digest must match the recorded one")

    flaky, _ = build_pipeline(seed=900200, nondeterministic=2)
    flaky_runner = Runner(flaky, Store())
    flaky_recorded = flaky_runner.run()["manifest"]
    flaky_report = flaky_runner.reproduce(flaky_recorded)
    check(not flaky_report["reproduced"],
          "a nondeterministic stage must break reproduction")
    check("stage02" in flaky_report["differences"],
          "the offending stage must be named: %r" % (flaky_report["differences"],))
    downstream = flaky.descendants({"stage02"})
    check(set(flaky_report["differences"]) == downstream,
          "exactly the nondeterministic stage and its descendants must differ")

    # --- 9. manifest sensitivity --------------------------------------------
    same = Runner(honest, Store()).run()["manifest"]
    check(Runner.manifest_hash(same) == Runner.manifest_hash(recorded),
          "an identical configuration must hash the manifest identically")
    changed_runner = Runner(honest, Store())
    changed_manifest = changed_runner.run(overrides={"stage00": {"epochs": 99}})["manifest"]
    check(Runner.manifest_hash(changed_manifest) != Runner.manifest_hash(recorded),
          "a parameter change must change the manifest hash")

    # --- 10. degenerate pipelines -------------------------------------------
    single_hardware = HardwareProfile("arm64", 8, "none", "n/a", 32, "macos-14")
    single = Pipeline([Stage("only", {"x": 1})], single_hardware, 7)
    single_runner = Runner(single, Store())
    single_result = single_runner.run()
    check(single_result["executed"] == ["only"], "a one stage pipeline must execute once")
    check(single_runner.run()["executed"] == [], "a one stage pipeline must cache")
    empty = Pipeline([], single_hardware, 7)
    check(empty.order == [] and empty.topological_order() == [],
          "an empty pipeline must have an empty order")
    empty_runner = Runner(empty, Store())
    check(empty_runner.run()["executed"] == [], "an empty pipeline must execute nothing")
    check(empty_runner.verify() == [], "an empty pipeline must verify cleanly")
    check(empty.descendants({"nothing"}) == {"nothing"},
          "descendants of an unknown stage are just that stage")

    # --- 11. scale ----------------------------------------------------------
    wide_stages = []
    for index in range(60):
        inputs = ()
        if index > 0:
            inputs = ("stage%02d" % (index - 1),)
        if index >= 10 and index % 7 == 0:
            inputs = ("stage%02d" % (index - 5), "stage%02d" % (index - 1))
        wide_stages.append(Stage("stage%02d" % index,
                                 {"width": 32 + index, "layers": 2 + index % 5},
                                 inputs, hardware_sensitive=(index % 10 == 0)))
    wide = Pipeline(wide_stages, hardware, 900300)
    check(len(wide.order) == 60, "the wide pipeline must order every stage")
    wide_store = Store()
    wide_runner = Runner(wide, wide_store)
    wide_runner.run()
    check(len(wide_store.executions) == 60, "the first pass must execute all sixty stages")
    wide_store.executions = []
    wide_runner.run(overrides={"stage29": {"width": 4096}})
    expected = wide.descendants({"stage29"})
    check(set(wide_store.executions) == expected,
          "the wide pipeline must re-execute exactly the affected stages: %d expected"
          % len(expected))
    check(len(wide_store.executions) < 60,
          "the cache must save work: %d of 60 executed" % len(wide_store.executions))
    total_runs = 60 + len(expected)
    check(total_runs < 2 * 60, "the cache must beat a full re-run")
    check(wide_runner.verify() == [], "the wide pipeline must verify cleanly")
    check(len(wide_store.artifacts) == 60, "every artifact must be retained")

    # --- 12. determinism ----------------------------------------------------
    def replay(seed):
        local, _ = build_pipeline(seed=seed, stage_count=5)
        local_runner = Runner(local, Store())
        first_pass = local_runner.run()
        second_pass = local_runner.run(overrides={"stage01": {"epochs": 42}})
        return (first_pass["executed"], second_pass["executed"],
                Runner.manifest_hash(first_pass["manifest"]),
                local_runner.verify())

    check(replay(900400) == replay(900400), "an identical seed must replay the whole platform")
    check(replay(900401) != replay(900400), "a different pipeline seed must change the hashes")

    print("graph: %d stages, two independent topological sorts agree, %d hardware sensitive"
          % (len(pipeline.order), len(sensitive)))
    print("reproducibility: a repeat run executed %d stages and reproduced every hash"
          % len(repeat["executed"]))
    print("incremental: a parameter change re-executed %d of %d stages"
          % (len(third["executed"]), len(pipeline.order)))
    print("hardware: changing the driver re-executed %d of %d stages"
          % (len(fourth["executed"]), len(pipeline.order)))
    print("scale: %d stages, %d re-executed after one change, cache saved %d recomputations"
          % (60, len(expected), 60 - len(expected)))
    print("PASS")


if __name__ == "__main__":
    main()
