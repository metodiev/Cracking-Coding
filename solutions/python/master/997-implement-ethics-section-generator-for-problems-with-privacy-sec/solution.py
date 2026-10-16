"""Task 997 - Implement ethics section generator for problems with privacy/security implications.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Generates the ethics/privacy section of a problem.  Declared risk tags (PII, credentials,
network, cryptography, biometric, location) each contribute a required disclosure and a
mitigation, and the sample data is independently scanned for personal data - e-mail addresses,
64-hex API keys and Luhn-valid card numbers - so a problem cannot claim to be harmless while
shipping real identifiers.  Publication candidates are redacted in place: addresses, keys and
card numbers become numbered placeholders, and the scan is re-run to prove nothing survived.

Simplified versus production: detection is a small set of regular expressions plus a Luhn
check rather than a full data-classification engine, and the section is text assembly rather
than natural-language generation.  Verified: disclosure completeness against the declared tags,
risk-level thresholds, Luhn's valid/invalid boundary, detection recall, redaction completeness
and idempotence, determinism, and the no-risk and all-risk boundaries.
"""
import hashlib
import json
import re

RISK_TAGS = ("PII", "credentials", "network", "cryptography", "biometric", "location")
CLAUSES = {
    "PII": "The task must use synthetic records; no real personal data may be shipped.",
    "credentials": "Secrets must be injected at runtime; no key may appear in the test data.",
    "network": "Network access is disabled during evaluation; the task must stay offline.",
    "cryptography": "Only vetted primitives may be used; rolling your own is out of scope.",
    "biometric": "Biometric inputs require explicit consent and must be irreversible.",
    "location": "Location data must be coarsened or removed before publication.",
}
MITIGATIONS = {
    "PII": "Replace identifiers with stable pseudonyms before publication.",
    "credentials": "Rotate any leaked credential and add a secret-scanning check to CI.",
    "network": "Stub the remote endpoint behind a local fixture server.",
    "cryptography": "Reference the library API instead of implementing primitives.",
    "biometric": "Reduce precision so individuals cannot be re-identified.",
    "location": "Round coordinates to the nearest city before publishing.",
}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
KEY = re.compile(r"\b[0-9a-fA-F]{64}\b")
PAN = re.compile(r"\b\d{16}\b")


class EthicsError(ValueError):
    """Raised when an ethics section cannot be produced safely."""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_error(action, needle):
    try:
        action()
    except EthicsError as exc:
        check(needle in str(exc), "error %r does not mention %r" % (str(exc), needle))
        return
    raise AssertionError("expected an EthicsError mentioning %r" % needle)


def digest_of(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def luhn_ok(digits):
    """Independent checksum oracle for card-like numbers (right-to-left doubling)."""
    total, double = 0, False
    for char in reversed(digits):
        value = int(char)
        if double:
            value *= 2
            if value > 9:
                value -= 9
        total += value
        double = not double
    return total % 10 == 0


def detect(text):
    """Findings as sorted (kind, value) pairs; cards only count when Luhn-valid."""
    findings = [("email", match) for match in EMAIL.findall(text)]
    findings += [("api-key", match) for match in KEY.findall(text)]
    findings += [("card", match) for match in PAN.findall(text) if luhn_ok(match)]
    return sorted(findings)


def redact(text):
    """Replace every detected identifier; e-mail placeholders are numbered to stay distinct."""
    counter = [0]

    def substitute_email(match):
        counter[0] += 1
        return "[REDACTED-EMAIL-%d]" % counter[0]

    text = EMAIL.sub(substitute_email, text)
    text = KEY.sub("[REDACTED-KEY]", text)
    return PAN.sub(lambda match: "[REDACTED-PAN]" if luhn_ok(match.group(0)) else match.group(0), text)


def validate(problem):
    if not problem.get("id"):
        raise EthicsError("problem without an id")
    if not isinstance(problem.get("sample_data"), str):
        raise EthicsError("problem %r needs sample_data as text" % (problem["id"],))
    for tag in problem.get("tags", ()):
        if tag not in RISK_TAGS:
            raise EthicsError("problem %r declares unknown risk tag %r" % (problem["id"], tag))


def risk_level(tags, findings):
    if findings or len(tags) >= 3:
        return "high"
    if tags:
        return "moderate"
    return "low"


def generate(problem):
    validate(problem)
    tags = [tag for tag in RISK_TAGS if tag in set(problem.get("tags", ()))]
    findings = detect(problem["sample_data"])
    kinds = sorted({kind for kind, _ in findings})
    disclosures = [CLAUSES[tag] for tag in tags]
    disclosures += ["Sample data contains %s values (%d occurrence(s))." % (kind, sum(1 for k, _ in findings if k == kind))
                    for kind in kinds]
    cleaned = redact(problem["sample_data"])
    section = {
        "id": problem["id"],
        "risk_level": risk_level(tags, findings),
        "reviewed_tags": tags,
        "disclosures": disclosures,
        "mitigations": [MITIGATIONS[tag] for tag in tags],
        "sample_policy": ("No personal data detected in the sample; it may be published as is."
                          if not findings else
                          "Sample data was redacted for publication: %s" % cleaned),
        "residual_findings": detect(cleaned),
    }
    section["digest"] = digest_of({key: section[key] for key in section})
    return section


def problem(pid, tags, sample_data):
    return {"id": pid, "tags": tags, "sample_data": sample_data}


VALID_CARD = "4111111111111111"
INVALID_CARD = "4111111111111112"
SECRET = "a" * 32 + "f" * 32


def main():
    # Luhn is the boundary between a plausible number and a card number.
    check(luhn_ok(VALID_CARD) and not luhn_ok(INVALID_CARD),
          "Luhn must accept %s and reject %s" % (VALID_CARD, INVALID_CARD))
    check(luhn_ok("79927398713") and not luhn_ok("79927398710"),
          "the classic Luhn vectors must hold")
    dirty = "student alice@school.example and bob@example.org solved it; key %s; card %s; note %s" % (
        SECRET, VALID_CARD, INVALID_CARD)
    findings = detect(dirty)
    check([kind for kind, _ in findings] == ["api-key", "card", "email", "email"],
          "detection must find one key, one Luhn-valid card and both addresses")
    check(("card", INVALID_CARD) not in findings and ("card", VALID_CARD) in findings,
          "a Luhn-invalid 16-digit string must not be treated as a card number")

    # Redaction completeness: the scan over the redacted text comes back empty, and stays empty.
    cleaned = redact(dirty)
    check(detect(cleaned) == [], "redaction must leave no finding behind: %r" % (detect(cleaned),))
    check(redact(cleaned) == cleaned, "redaction must be idempotent")
    check("[REDACTED-EMAIL-1]" in cleaned and "[REDACTED-EMAIL-2]" in cleaned
          and "[REDACTED-KEY]" in cleaned and "[REDACTED-PAN]" in cleaned and INVALID_CARD in cleaned,
          "redaction must replace identifiers, number them and leave non-identifiers untouched")

    # Completeness oracle: the disclosed tags are exactly the declared ones, in guide order.
    risky = problem("triage", ["location", "PII", "credentials"],
                    "call 555 and mail bob@example.org with key %s" % SECRET)
    section = generate(risky)
    check(section["reviewed_tags"] == sorted(section["reviewed_tags"],
                                            key=RISK_TAGS.index) == ["PII", "credentials", "location"],
          "disclosures must follow the guide order for exactly the declared tags")
    check(all(CLAUSES[tag] in section["disclosures"] for tag in section["reviewed_tags"])
          and len(section["mitigations"]) == len(section["reviewed_tags"]),
          "every declared tag needs one clause and one mitigation")
    check(not any(CLAUSES[tag] in section["disclosures"] for tag in RISK_TAGS
                  if tag not in section["reviewed_tags"]),
          "an undeclared tag must not produce a clause")
    check(section["risk_level"] == "high" and section["residual_findings"] == [],
          "personal data in the sample must raise the level and still be fully redacted")
    check(any("api-key" in disclosure for disclosure in section["disclosures"]),
          "the sample-data findings must appear in the disclosures")

    # Risk-level thresholds and the no-risk boundary.
    moderate = generate(problem("io", ["network"], "no identifiers here"))
    check(moderate["risk_level"] == "moderate" and len(moderate["mitigations"]) == 1,
          "one declared tag without findings is a moderate risk")
    clean = generate(problem("plain", [], "just numbers 4 8 15 16 23 42"))
    check(clean["risk_level"] == "low" and clean["disclosures"] == []
          and "No personal data detected" in clean["sample_policy"],
          "a problem with no tags and clean data must be declared low risk with no disclosures")
    everything = generate(problem("all", list(RISK_TAGS), ""))
    check(everything["risk_level"] == "high" and len(everything["disclosures"]) == len(RISK_TAGS),
          "declaring every tag must produce one clause per tag")

    # Determinism: same input, same section and digest; new tags change it.
    check(generate(risky) == section and generate(risky)["digest"] == section["digest"],
          "regenerating the same problem must be byte-identical")
    check(generate(problem("triage", ["PII"], risky["sample_data"]))["digest"] != section["digest"],
          "dropping a risk tag must change the section digest")
    check(len(section["digest"]) == 64, "the section digest must be a full sha256")

    # Validation: unknown tags and missing sample data are rejected.
    expect_error(lambda: generate(problem("bad", ["genetics"], "x")), "unknown risk tag")
    expect_error(lambda: generate({"id": "bad", "tags": []}), "sample_data")

    print("level=%s tags=%s disclosures=%d residual=%d digest=%s"
          % (section["risk_level"], ",".join(section["reviewed_tags"]), len(section["disclosures"]),
             len(section["residual_findings"]), section["digest"][:16]))
    print("PASS")


if __name__ == "__main__":
    main()
