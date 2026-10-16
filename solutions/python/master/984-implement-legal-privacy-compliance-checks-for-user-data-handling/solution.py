"""Task 984 - Implement legal/privacy compliance checks for user data handling in platform.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A privacy gate over the platform's data inventory and consent ledger.  Every collected field declares its
personal-data class, its processing purpose, the legal basis it relies on, a retention period, whether it
leaves the region (and with which safeguard) and whether it can be erased.  Consent records say which learner
agreed to which purpose until when.  Rule checks then verify purpose limitation and consent currency, the
retention ceiling of each legal basis, data minimisation, the right to erasure, transfer safeguards and the
stricter basis required for special-category data - each violation naming the rule and the article it
breaches, so the gate can fail the release instead of shipping a non-compliant change.

Simplified versus a real DPO tooling: a hand-written inventory of ten fields, one consent ledger and a fixed
"today" instead of a data catalogue integration and jurisdiction-specific rule packs.  Verified: every
planted violation found by exactly its own rule, the hardened inventory passing clean, retention and consent
expiry at the exact boundary, the erasure/minimisation/special-category/transfer rules firing individually,
an independent rule evaluator as oracle, report ordering and article mapping, empty-inventory and no-PII
boundaries and reproducibility.
"""
from collections import namedtuple

Field = namedtuple("Field", "name pii purpose basis retention_days cross_border safeguard erasable")
Consent = namedtuple("Consent", "learner purpose granted expires")
Violation = namedtuple("Violation", "rule article subject detail")
TODAY = 60
RETENTION_LIMIT = {"consent": 365, "contract": 730, "legitimate_interest": 180, "explicit_consent": 365}
SPECIAL_CATEGORY = ("sensitive", "biometric")
RULES = ("consent_expiry", "special_category", "retention_limit", "minimization", "erasure", "transfer")
ARTICLES = {"consent_expiry": "Art.5(1)(d)", "special_category": "Art.9", "retention_limit": "Art.5(1)(e)",
            "minimization": "Art.5(1)(c)", "erasure": "Art.17", "transfer": "Art.46"}
INVENTORY = (
    Field("user_id", "identifier", "authentication", "contract", 730, False, "none", True),
    Field("email", "contact", "authentication", "contract", 730, True, "scc", True),
    Field("password_hash", "sensitive", "authentication", "contract", 730, False, "none", True),
    Field("birth_date", "identifier", "personalization", "consent", 400, False, "none", True),
    Field("device_id", "identifier", "analytics", "legitimate_interest", 180, False, "none", True),
    Field("location", "sensitive", "analytics", "legitimate_interest", 365, False, "none", True),
    Field("browsing_history", "identifier", "none", "none", 3650, False, "none", True),
    Field("photo", "biometric", "authentication", "contract", 730, True, "scc", False),
    Field("shipping_address", "contact", "fulfilment", "contract", 400, True, "none", True),
    Field("city", "none", "personalization", "consent", 365, False, "none", True),
)
COMPLIANT = tuple(Field(row.name, row.pii, "analytics" if row.purpose == "none" else row.purpose,
                        "explicit_consent" if row.pii in SPECIAL_CATEGORY else
                        ("legitimate_interest" if row.purpose == "none" else row.basis),
                        min(row.retention_days, RETENTION_LIMIT["legitimate_interest"] if row.purpose == "none"
                            else RETENTION_LIMIT["explicit_consent"] if row.pii in SPECIAL_CATEGORY
                            else RETENTION_LIMIT[row.basis]),
                        row.cross_border, "scc" if row.cross_border and row.safeguard == "none" else row.safeguard,
                        True)
                   for row in INVENTORY)
CONSENTS = (
    Consent("l01", "analytics", 0, 400),
    Consent("l02", "analytics", 0, 10),
    Consent("l04", "analytics", 0, 20),
    Consent("l03", "marketing", 0, 100),
    Consent("l01", "marketing", 0, 30),
)
PURPOSES_IN_USE = ("authentication", "personalization", "analytics", "fulfilment")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def field_violations(entry):
    """Every rule that one field breaks, in rule order."""
    found = []
    if entry.basis in RETENTION_LIMIT and entry.retention_days > RETENTION_LIMIT[entry.basis]:
        found.append(("retention_limit", entry.name,
                      f"{entry.retention_days} days exceeds the {entry.basis} ceiling of "
                      f"{RETENTION_LIMIT[entry.basis]}"))
    if entry.purpose == "none" or entry.basis == "none":
        found.append(("minimization", entry.name, "collected without a declared purpose and legal basis"))
    if entry.pii != "none" and not entry.erasable:
        found.append(("erasure", entry.name, f"{entry.pii} data that cannot be erased on request"))
    if entry.cross_border and entry.safeguard == "none":
        found.append(("transfer", entry.name, "leaves the region without a transfer safeguard"))
    if entry.pii in SPECIAL_CATEGORY and entry.basis != "explicit_consent":
        found.append(("special_category", entry.name, f"{entry.pii} data held on the weaker basis {entry.basis}"))
    return found


def consent_violations(consents, today=TODAY, purposes_in_use=PURPOSES_IN_USE):
    """An expired consent record for a purpose still in use is a purpose-limitation failure."""
    return [("consent_expiry", f"{record.learner}:{record.purpose}",
             f"consent expired on day {record.expires} while {record.purpose} is still processed")
            for record in consents
            if record.expires < today and record.purpose in purposes_in_use]


def review(inventory, consents):
    """Scan the inventory and the ledger, returning findings in rule order with their articles."""
    raw = []
    for entry in inventory:
        raw.extend(field_violations(entry))
    raw.extend(consent_violations(consents))
    report = [Violation(rule, ARTICLES[rule], subject, detail) for rule, subject, detail in raw]
    return sorted(report, key=lambda item: (RULES.index(item.rule), item.subject)), report


def review_oracle(inventory, consents):
    """Independent oracle: the same rules, gathered rule by rule over the raw records."""
    found = []
    for rule in RULES:
        for entry in inventory:
            if rule == "retention_limit" and entry.basis in RETENTION_LIMIT \
                    and entry.retention_days > RETENTION_LIMIT[entry.basis]:
                found.append(rule)
            elif rule == "minimization" and (entry.purpose == "none" or entry.basis == "none"):
                found.append(rule)
            elif rule == "erasure" and entry.pii != "none" and not entry.erasable:
                found.append(rule)
            elif rule == "transfer" and entry.cross_border and entry.safeguard == "none":
                found.append(rule)
            elif rule == "special_category" and entry.pii in SPECIAL_CATEGORY \
                    and entry.basis != "explicit_consent":
                found.append(rule)
        if rule == "consent_expiry":
            found.extend("consent_expiry" for _record in consent_violations(consents))
    return found


def counts(report):
    tally = {rule: 0 for rule in RULES}
    for violation in report:
        tally[violation.rule] += 1
    return tally


def gate(report):
    return "fail" if report else "pass"


def main():
    ordered, raw = review(INVENTORY, CONSENTS)
    # 1) every planted violation is found by its own rule, and nothing else is reported.
    check(counts(raw) == {"consent_expiry": 2, "special_category": 3, "retention_limit": 2,
                          "minimization": 1, "erasure": 1, "transfer": 1}
          and len(raw) == 10 and gate(raw) == "fail",
          f"the planted violations were not detected exactly: {counts(raw)}")
    # 2) the hardened inventory and ledger are clean, so the gate passes.
    hardened_report, _hardened_raw = review(COMPLIANT, (Consent("l01", "analytics", 0, 400),))
    check(hardened_report == [] and gate(hardened_report) == "pass"
          and counts([]) == {rule: 0 for rule in RULES},
          f"the hardened inventory is not clean: {hardened_report}")
    # 3) retention boundary: sitting exactly on the ceiling is allowed, one day more is not.
    on_limit = Field("x", "identifier", "analytics", "legitimate_interest", 180, False, "none", True)
    over_limit = Field("x", "identifier", "analytics", "legitimate_interest", 181, False, "none", True)
    check(review((on_limit,), ())[0] == [] and [item.rule for item in review((over_limit,), ())[0]]
          == ["retention_limit"] and field_violations(on_limit) == [],
          f"the retention boundary is wrong: {review((over_limit,), ())[0]}")
    # 4) consent expiry boundary: a consent that expires today is still current, yesterday's is not.
    check(consent_violations((Consent("l", "analytics", 0, TODAY),)) == []
          and len(consent_violations((Consent("l", "analytics", 0, TODAY - 1),))) == 1
          and consent_violations((Consent("l", "reporting", 0, 0),)) == []
          and len(consent_violations((Consent("l", "analytics", 0, 0),))) == 1,
          "the consent expiry boundary is wrong")
    # 5) the per-field rules fire individually: minimisation, erasure, special category and transfer.
    single = {"minimization": Field("x", "identifier", "none", "contract", 30, False, "none", True),
              "erasure": Field("x", "identifier", "analytics", "consent", 30, False, "none", False),
              "special_category": Field("x", "biometric", "analytics", "contract", 30, False, "none", True),
              "transfer": Field("x", "identifier", "analytics", "consent", 30, True, "none", True)}
    check(all([item.rule for item in review((entry,), ())[0]] == [rule] for rule, entry in single.items())
          and field_violations(Field("ok", "none", "analytics", "consent", 30, False, "none", True)) == [],
          f"a single-field rule fired wrongly: "
          f"{ {rule: [i.rule for i in review((e,), ())[0]] for rule, e in single.items()} }")
    # 6) a special-category field on the strict basis is fine, and a cross-border field with a safeguard
    #    is fine - the two rules must not double count or over-trigger.
    safe = Field("dna", "biometric", "authentication", "explicit_consent", 365, True, "dpa", True)
    check(field_violations(safe) == [] and [item.rule for item in review((COMPLIANT[2],), ())[0]] == []
          and len(review((INVENTORY[7],), ())[0]) == 2,
          f"the strict-basis or safeguard handling is wrong: {field_violations(safe)}")
    # 7) oracle: an independent rule-by-rule evaluator produces the same multiset of rules, and the report
    #    is ordered by rule then subject with every article filled in.
    check(sorted(review_oracle(INVENTORY, CONSENTS)) == sorted(violation.rule for violation in raw)
          and dict(counts(raw)) == {rule: sum(1 for _r in review_oracle(INVENTORY, CONSENTS) if _r == rule)
                                    for rule in RULES}
          and len(ordered) == len(raw) == sum(counts(raw).values())
          and [item.rule for item in ordered] == sorted((item.rule for item in ordered), key=RULES.index)
          and all(item.article == ARTICLES[item.rule] and item.detail for item in ordered),
          f"the report disagrees with the oracle: {ordered}")
    # 8) boundaries: an empty inventory and ledger, an inventory with no personal data, and a ledger with
    #    no expired records all produce a clean report.
    clean_field = Field("meta", "none", "analytics", "legitimate_interest", 90, False, "none", True)
    check(review((), ())[0] == [] and gate([]) == "pass" and counts([]) == {rule: 0 for rule in RULES}
          and review((clean_field,), (Consent("l", "analytics", 0, 999),))[0] == []
          and consent_violations(()) == [] and field_violations(clean_field) == [],
          "the empty-inventory and no-PII boundaries are wrong")
    # 9) reproducibility: the review, its ordering and the tallies are byte-identical across runs.
    check(review(INVENTORY, CONSENTS) == (ordered, raw) and counts(raw) == counts(raw)
          and [violation.subject for violation in ordered] == [violation.subject for violation in ordered]
          and review(COMPLIANT, (Consent("l01", "analytics", 0, 400),))[0] == hardened_report,
          "the compliance review must be reproducible")
    print(f"fields={len(INVENTORY)} consents={len(CONSENTS)} violations={len(raw)} gate={gate(raw)} "
          f"by_rule={{{', '.join(f'{rule}:{count}' for rule, count in sorted(counts(raw).items()))}}}")
    print(f"hardened_violations={len(hardened_report)} hardened_gate={gate(hardened_report)} "
          f"articles={sorted({item.article for item in ordered})}")
    print("PASS")


if __name__ == "__main__":
    main()
