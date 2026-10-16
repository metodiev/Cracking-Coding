"""Task 990 - Implement reproducible Docker images for canonical environment with JDK/test harness.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A reproducible build model for the canonical learner image (JDK + test harness).  A spec is
normalised before hashing: packages sorted by name, environment iterated in sorted key order,
layers kept in their meaningful order, and every input pinned (base image by digest, JDK build,
exact package versions).  The image digest is SHA-256 over that canonical text, so two builds
of the same logical spec agree bit-for-bit, while cumulative layer digests let the builder
report how many layers a rebuild can reuse from cache.

Simplified versus production: layers are declarative instead of executed by a container
runtime, so sizes and real filesystem contents are out of scope, and the pinning rules are a
hand-written subset of a lockfile-driven build.  Verified: permutation invariance of
packages/env, layer-order significance, a closed-form digest oracle, round-trip stability
through the normal form, pinning and network-fetch rejections, cache reuse after appending a
layer, and the minimal-image boundary.
"""
import hashlib
import json
import re

DIGEST_REFERENCE = re.compile(r"@sha256:[0-9a-f]{64}$")
PINNED_VERSION = re.compile(r"^\d+(\.\d+)*([-+][A-Za-z0-9.]+)*$")
FORBIDDEN_ENV = ("BUILD_TIMESTAMP", "BUILD_TIME", "BUILD_DATE")


class ReproducibilityError(ValueError):
    """Raised when a spec could not be rebuilt byte-for-byte on another machine."""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_error(action, needle):
    try:
        action()
    except ReproducibilityError as exc:
        check(needle in str(exc), "error %r does not mention %r" % (str(exc), needle))
        return
    raise AssertionError("expected a ReproducibilityError mentioning %r" % needle)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate(spec):
    """Every input that could drift between machines must be pinned."""
    reference = spec["base"]["ref"]
    if not DIGEST_REFERENCE.search(reference):
        raise ReproducibilityError("base image %r must be digest-pinned (@sha256:<64 hex>)" % reference)
    if not PINNED_VERSION.match(spec["jdk"]["version"]):
        raise ReproducibilityError("JDK version %r must be pinned to an exact build"
                                   % spec["jdk"]["version"])
    names = set()
    for package in spec.get("packages", ()):
        if package["name"] in names:
            raise ReproducibilityError("duplicate package %r" % package["name"])
        names.add(package["name"])
        if not PINNED_VERSION.match(package.get("version", "")):
            raise ReproducibilityError("package %r is not version-pinned (got %r)"
                                       % (package["name"], package.get("version")))
    for key in spec.get("env", {}):
        if key.upper() in FORBIDDEN_ENV:
            raise ReproducibilityError("env key %r would change on every build" % key)
    for layer in spec.get("layers", ()):
        if "http://" in layer["command"] or "https://" in layer["command"]:
            raise ReproducibilityError("layer %r fetches over the network; vendor the input"
                                       % layer["name"])


def normalize(spec):
    """Order-insensitive inputs are sorted, so the canonical form depends only on content."""
    validate(spec)
    return {"base": spec["base"]["ref"],
            "jdk": "%s-%s" % (spec["jdk"]["distro"], spec["jdk"]["version"]),
            "packages": ["%s=%s" % (package["name"], package["version"])
                         for package in sorted(spec.get("packages", ()), key=lambda p: p["name"])],
            "env": {key: spec["env"][key] for key in sorted(spec.get("env", {}))},
            "layers": ["%s: %s" % (layer["name"], layer["command"]) for layer in spec.get("layers", ())],
            "harness": {"entrypoint": spec["harness"]["entrypoint"],
                        "script": sha256_text(spec["harness"]["script"])}}


def from_normalized(normalized, script_text):
    """Rebuild a spec from its canonical form; used to prove the normal form round-trips."""
    distro, version = normalized["jdk"].split("-", 1)
    return {"base": {"ref": normalized["base"]},
            "jdk": {"distro": distro, "version": version},
            "packages": [{"name": item.split("=", 1)[0], "version": item.split("=", 1)[1]}
                         for item in normalized["packages"]],
            "env": dict(normalized["env"]),
            "layers": [{"name": text.split(": ", 1)[0], "command": text.split(": ", 1)[1]}
                       for text in normalized["layers"]],
            "harness": {"entrypoint": normalized["harness"]["entrypoint"], "script": script_text}}


def image_digest(spec):
    return sha256_text(canonical(normalize(spec)))


def layer_digests(spec):
    """Cumulative digests: base first, then one digest per layer over its predecessor."""
    normalized = normalize(spec)
    digests = [sha256_text("base|" + normalized["base"])]
    for layer in normalized["layers"]:
        digests.append(sha256_text(digests[-1] + "|" + layer))
    return digests


def rebuild_cache_hits(previous, spec):
    """How many leading layers an incremental rebuild can reuse from the previous image."""
    old, new = layer_digests(previous), layer_digests(spec)
    hits = 0
    for before, after in zip(old, new):
        if before != after:
            break
        hits += 1
    return hits


def build_manifest(spec):
    normalized = normalize(spec)
    return {"base": normalized["base"], "jdk": normalized["jdk"],
            "packages": normalized["packages"], "layers": normalized["layers"],
            "test_harness": dict(normalized["harness"]),
            "layer_digests": layer_digests(spec), "digest": image_digest(spec)}


JDK_REF = ("eclipse-temurin:21.0.4_7-jdk@sha256:"
           "6b3a4c9d1e2f3a4b5c6d7e8f90112233445566778899aabbccddeeff00112233")


def harness_script():
    return "#!/bin/sh\njava -version\nexec python3 -m pytest \"$@\"\n"

MINIMAL_HARNESS_SHA256 = "306c6ca7407560340797866e077e053627ad409277d1b9da58106fce4cf717cb"


def image_spec():
    return {"base": {"ref": JDK_REF},
            "jdk": {"distro": "temurin", "version": "21.0.4+7"},
            "packages": [{"name": "curl", "version": "8.5.0-2"},
                         {"name": "python3", "version": "3.12.3-1"},
                         {"name": "ca-certificates", "version": "20240203"}],
            "env": {"JAVA_HOME": "/opt/java/openjdk", "PYTHONUNBUFFERED": "1"},
            "layers": [{"name": "apt", "command": "apt-get install -y curl python3"},
                       {"name": "jdk", "command": "copy /opt/java/openjdk"},
                       {"name": "harness", "command": "copy /opt/harness"}],
            "harness": {"entrypoint": "/opt/harness/run.sh", "script": harness_script()}}


def minimal_spec():
    return {"base": {"ref": JDK_REF},
            "jdk": {"distro": "temurin", "version": "21.0.4+7"},
            "packages": [], "env": [], "layers": [],
            "harness": {"entrypoint": "/opt/harness/run.sh", "script": "#!/bin/sh\nexit 0\n"}}


def main():
    spec = image_spec()
    digest = image_digest(spec)

    # Closed-form oracle: the minimal spec's canonical text is written out by hand (the harness
    # digest is the literal sha256 of "#!/bin/sh\nexit 0\n", computed independently).
    minimal = minimal_spec()
    expected_text = ('{"base":"%s","env":{},"harness":{"entrypoint":"/opt/harness/run.sh",'
                     '"script":"%s"},"jdk":"temurin-21.0.4+7","layers":[],"packages":[]}'
                     % (JDK_REF, MINIMAL_HARNESS_SHA256))
    check(canonical(normalize(minimal)) == expected_text,
          "canonical form of the minimal spec differs from the hand-written expectation")
    check(image_digest(minimal) == sha256_text(expected_text),
          "minimal image digest must equal sha256 of the hand-written canonical text")

    # Permutation invariance: packages and env are sets, so their order cannot change a build.
    permuted = image_spec()
    permuted["packages"] = list(reversed(permuted["packages"]))
    permuted["env"] = {"PYTHONUNBUFFERED": "1", "JAVA_HOME": "/opt/java/openjdk"}
    check(image_digest(permuted) == digest, "package/env order must not change the image digest")
    check(normalize(permuted) == normalize(spec), "normalisation must be order-insensitive")
    check(normalize(permuted) == normalize(spec), "normalisation must be order-insensitive")
    round_trip = from_normalized(normalize(spec), harness_script())
    check(image_digest(round_trip) == digest,
          "a spec rebuilt from its normal form must produce the same digest")
    check(build_manifest(permuted)["packages"] == ["ca-certificates=20240203", "curl=8.5.0-2",
                                                   "python3=3.12.3-1"],
          "packages must be reported sorted by name")

    # Layer order is semantically meaningful: permuting layers must change the digest.
    reordered = image_spec()
    reordered["layers"] = list(reversed(reordered["layers"]))
    check(image_digest(reordered) != digest, "reordering layers must change the image digest")
    check(layer_digests(reordered) != layer_digests(spec)
          and len(layer_digests(spec)) == len(spec["layers"]) + 1,
          "layer digests must follow layer order and cover every layer plus the base")

    # Pinning rules: base tag, JDK build, package versions, duplicate packages.
    expect_error(lambda: validate(dict(image_spec(), base={"ref": "eclipse-temurin:21-jdk"})),
                 "digest-pinned")
    expect_error(lambda: validate(dict(image_spec(), base={"ref": "eclipse-temurin:21@sha256:abc"})),
                 "digest-pinned")
    unpinned_jdk = image_spec()
    unpinned_jdk["jdk"] = {"distro": "temurin", "version": "latest"}
    expect_error(lambda: validate(unpinned_jdk), "pinned")
    floating = image_spec()
    floating["packages"] = [{"name": "curl", "version": "latest"}]
    expect_error(lambda: validate(floating), "curl")
    duplicated = image_spec()
    duplicated["packages"] = [{"name": "curl", "version": "8.5.0-2"},
                              {"name": "curl", "version": "8.6.0-1"}]
    expect_error(lambda: validate(duplicated), "duplicate package")

    # Build hygiene: no network fetch inside a layer, no wall-clock env values.
    fetching = image_spec()
    fetching["layers"] = [{"name": "install", "command": "curl -O https://example.com/jdk.tgz"}]
    expect_error(lambda: validate(fetching), "network")
    stamped = image_spec()
    stamped["env"] = {"BUILD_TIMESTAMP": "2026-10-16T12:00:00Z"}
    expect_error(lambda: validate(stamped), "BUILD_TIMESTAMP")

    # Cache: appending a layer reuses every earlier digest, and the harness layer stays pinned.
    extended = image_spec()
    extended["layers"] = spec["layers"] + [{"name": "extra", "command": "copy /opt/cache"}]
    check(rebuild_cache_hits(spec, extended) == len(spec["layers"]) + 1,
          "appending a layer must reuse the base and every earlier layer")
    check(rebuild_cache_hits(spec, reordered) == 1,
          "a different first layer must reuse only the base digest")
    check(image_digest(extended) != digest, "the extended image must get a new digest")
    manifest = build_manifest(spec)
    check(manifest["jdk"] == "temurin-21.0.4+7" and manifest["test_harness"]["entrypoint"].endswith("run.sh"),
          "the manifest must pin the JDK build and the harness entrypoint")
    check(manifest["test_harness"]["script"] == sha256_text(harness_script()),
          "the harness script must be recorded by digest, not by content")

    # Boundary: the minimal image (no packages, no layers) builds and is stable.
    check(build_manifest(minimal)["packages"] == [] and build_manifest(minimal)["layers"] == []
          and image_digest(minimal) == image_digest(minimal_spec()),
          "the minimal image must report no packages/layers and hash reproducibly")
    print("digest=%s layers=%d cache_hits=%d jdk=%s\nPASS"
          % (digest[:16], len(spec["layers"]), rebuild_cache_hits(spec, extended), manifest["jdk"]))


if __name__ == "__main__":
    main()
