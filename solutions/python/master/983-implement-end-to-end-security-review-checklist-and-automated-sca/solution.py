"""Task 983 - Implement end-to-end security review checklist and automated scans for repository code.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A security gate for a repository: a small source corpus is scanned by independent detectors - hard-coded secrets (entropy plus assignment shape),
string-built SQL, dynamic code execution, insecure randomness, weak hashing, disabled TLS verification, unauthenticated request handlers and
disabled authz decorators - and every finding is mapped onto a checklist item with a severity.  The gate decides pass or fail from the highest
severity found, and the report lists what was found, where, and which checklist item it violates.

Simplified versus a real SAST tool: line-oriented regular expressions over a tiny in-memory corpus instead of a parser and taint analysis, only
eight rules and no data flow across functions.  Verified: every planted vulnerability caught by exactly the detector that owns it, the hardened
file producing no finding (no false positives), the Shannon entropy of a sample against a brute-force recomputation, the gate decision following
the severity policy, the checklist covering every detector, ordering and line numbers, empty and comment-only files being clean, and reproducibility.
"""
import math
import re
from collections import Counter, namedtuple

Finding = namedtuple("Finding", "rule path line severity cwe")
ChecklistItem = namedtuple("ChecklistItem", "rule title cwe severity")
CHECKLIST = (
    ChecklistItem("secret", "No credentials in source control", "CWE-798", "critical"),
    ChecklistItem("sql_injection", "Queries must be parameterised", "CWE-89", "critical"),
    ChecklistItem("dynamic_code", "No eval/exec of untrusted input", "CWE-95", "high"),
    ChecklistItem("weak_randomness", "Secrets must come from a CSPRNG", "CWE-338", "medium"),
    ChecklistItem("weak_hash", "Passwords must use a slow KDF", "CWE-916", "high"),
    ChecklistItem("tls_verification", "TLS verification must stay enabled", "CWE-295", "critical"),
    ChecklistItem("missing_authz", "Every route needs an authorization check", "CWE-862", "high"),
    ChecklistItem("authz_disabled", "Authorization decorators must not be bypassed", "CWE-863", "high"),
)
GATE = {"critical": "fail", "high": "fail", "medium": "review", "low": "pass"}
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
SECRET_PATTERN = re.compile(r"(?i)\b(api_key|secret|password|token|passwd)\w*\s*=\s*['\"]([^'\"]{8,})['\"]")
HASH_ENTROPY_THRESHOLD = 3.2
SQL_PATTERN = re.compile(r"(?i)['\"][^'\"]*\b(select|insert|update|delete)\b[^'\"]*['\"]\s*(%|\+|\.format\s*\()")
DYNAMIC_PATTERN = re.compile(r"\b(eval|exec|pickle\.loads)\s*\(")
RANDOM_PATTERN = re.compile(r"\brandom\.(choice|choices|randint|random|randbytes)\s*\(")
HASH_PATTERN = re.compile(r"\b(md5|sha1)\s*\(")
TLS_PATTERN = re.compile(r"verify\s*=\s*False|ssl\._create_unverified_context")
AUTHZ_PATTERN = re.compile(r"@(login_required|requires_role|authorize)")
AUTHZ_OFF_PATTERN = re.compile(r"#\s*nosec:?\s*authz|@public_route|\bauthz\s*=\s*False")
ROUTE_PATTERN = re.compile(r"@(app|router)\.(route|get|post)\s*\(")
DETECTORS = (("sql_injection", SQL_PATTERN), ("dynamic_code", DYNAMIC_PATTERN),
             ("weak_randomness", RANDOM_PATTERN), ("weak_hash", HASH_PATTERN),
             ("tls_verification", TLS_PATTERN), ("authz_disabled", AUTHZ_OFF_PATTERN))
SEVERITY = {item.rule: item.severity for item in CHECKLIST}
CWE = {item.rule: item.cwe for item in CHECKLIST}
SEVERITY_NAMES = ("low", "medium", "high", "critical")
GATE = {"critical": "fail", "high": "fail", "medium": "review", "low": "pass"}
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
CORPUS = {
    "app/config.py": (
        "import os",
        "API_KEY = 'sk_live_9f3ac7b1d2e4'",
        "DEBUG = os.environ.get('DEBUG', '0') == '1'",
    ),
    "app/db.py": (
        "def lookup(conn, user_id):",
        "    query = 'SELECT * FROM users WHERE id = %s' % user_id",
        "    return conn.execute(query)",
    ),
    "app/rules.py": (
        "def apply_rule(expression, payload):",
        "    return eval(expression)",
    ),
    "app/tokens.py": (
        "import random",
        "def new_reset_token():",
        "    return ''.join(random.choice('abcdef0123456789') for _ in range(16))",
    ),
    "app/auth.py": (
        "import hashlib",
        "def fingerprint(password):",
        "    return hashlib.md5(password.encode()).hexdigest()",
    ),
    "app/edge.py": (
        "import requests",
        "def fetch(url):",
        "    return requests.get(url, verify=False)",
    ),
    "app/routes.py": (
        "@app.route('/admin/users')",
        "@login_required",
        "def admin_users():",
        "    return list_users()",
        "@app.post('/admin/impersonate')",
        "def impersonate(user_id):",
        "    return impersonate_user(user_id)  # nosec:authz",
        "@app.get('/health')",
        "def health():",
        "    return {'ok': True}",
    ),
}
HARDENED = {
    "app/config.py": ("import os", "API_KEY = os.environ['API_KEY']"),
    "app/db.py": ("def lookup(conn, user_id):", "    return conn.execute('SELECT 1 WHERE id = %s', (user_id,))"),
    "app/rules.py": ("import ast", "def apply_rule(expression, payload):", "    return ast.literal_eval(expression)"),
    "app/tokens.py": ("import secrets", "def new_reset_token():",
                      "    return secrets.token_hex(16)"),
    "app/auth.py": ("from werkzeug.security import generate_password_hash", "def fingerprint(password):",
                    "    return generate_password_hash(password)"),
    "app/edge.py": ("import requests", "def fetch(url):", "    return requests.get(url)"),
    "app/routes.py": ("@app.route('/admin/users')", "@login_required", "def admin_users():",
                      "    return list_users()"),
}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def shannon_entropy(text):
    """Shannon entropy in bits per character, computed from the symbol frequencies."""
    if not text:
        return 0.0
    counts = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def entropy_oracle(text):
    """Independent oracle: the same entropy from math.log with an explicit bit conversion."""
    if not text:
        return 0.0
    frequencies = {}
    for char in text:
        frequencies[char] = frequencies.get(char, 0) + 1
    total = len(text)
    result = 0.0
    for count in frequencies.values():
        share = count / total
        result -= share * (math.log(share) / math.log(2.0))
    return result


def scan_file(path, lines):
    """Every finding in one file, detected line by line in line order."""
    findings = []
    for number, line in enumerate(lines, start=1):
        if line.strip().startswith("#"):
            continue
        for match in SECRET_PATTERN.finditer(line):
            if shannon_entropy(match.group(2)) >= HASH_ENTROPY_THRESHOLD:
                findings.append(("secret", path, number))
        for rule, pattern in DETECTORS:
            if pattern.search(line):
                findings.append((rule, path, number))
    findings.extend(unauthenticated_routes(path, lines))
    return findings


def unauthenticated_routes(path, lines):
    """A route is unauthenticated when neither it nor the next non-empty line carries an authz decorator."""
    findings = []
    for number, line in enumerate(lines):
        if not ROUTE_PATTERN.search(line):
            continue
        following = [row for row in lines[number + 1:number + 4] if row.strip()]
        if not (following and any(AUTHZ_PATTERN.search(row) for row in following)):
            findings.append(("missing_authz", path, number + 1))
    return findings


def review(corpus):
    """Scan the corpus and apply the severity policy: the mapped findings plus the gate decision."""
    findings = []
    for path in sorted(corpus):
        findings.extend(scan_file(path, corpus[path]))
    report = [Finding(rule, path, line, SEVERITY[rule], CWE[rule])
              for rule, path, line in sorted(findings, key=lambda row: (row[1], row[2], row[0]))]
    worst = max((SEVERITY_RANK[finding.severity] for finding in report), default=-1)
    return report, "pass" if worst < 0 else GATE[SEVERITY_NAMES[worst]]


def per_rule(report):
    return dict(Counter(finding.rule for finding in report))


def main():
    report, decision = review(CORPUS)
    counts = per_rule(report)
    # 1) every planted vulnerability is caught by the detector that owns it, exactly once, and the gate fails.
    check(counts == {"secret": 1, "sql_injection": 1, "dynamic_code": 1, "weak_randomness": 1,
                     "weak_hash": 1, "tls_verification": 1, "missing_authz": 2, "authz_disabled": 1}
          and len(report) == 9 and decision == "fail",
          f"the scan missed or duplicated a finding: {counts}")
    # 2) the hardened corpus (the same files, fixed) is clean - no false positives at all.
    hardened_report, hardened_decision = review(HARDENED)
    check(hardened_report == [] and hardened_decision == "pass" and per_rule(hardened_report) == {},
          f"the hardened corpus is not clean: {hardened_report}")
    # 3) each finding carries the checklist item's cwe and severity, and the gate follows the policy.
    check(all(finding.cwe == CWE[finding.rule] and finding.severity == SEVERITY[finding.rule]
              for finding in report)
          and {finding.severity for finding in report} <= {"critical", "high", "medium"}
          and GATE["medium"] == "review" and GATE["low"] == "pass" and len({item.rule for item in CHECKLIST}) == 8,
          "a finding is not mapped onto its checklist item")
    # 4) the secret detector uses entropy: a low-entropy placeholder and an env lookup are ignored, the
    #    value matches an independent recomputation and a random-looking literal fires.
    quiet_report, quiet_decision = review({"app/x.py": ("API_KEY = 'aaaaaaaaaaaaaaa'", "SECRET = os.environ['SECRET']")})
    salted_report, _salted = review({"app/y.py": ("password = 'Zq7!vL2#pR9$'",)})
    check(quiet_report == [] and quiet_decision == "pass" and len(salted_report) == 1
          and salted_report[0].rule == "secret"
          and abs(shannon_entropy("sk_live_9f3ac7b1d2e4") - entropy_oracle("sk_live_9f3ac7b1d2e4")) < 1e-12
          and abs(shannon_entropy("")) < 1e-12 and abs(shannon_entropy("aaaa")) < 1e-12
          and abs(shannon_entropy("abcd") - 2.0) < 1e-12,
          f"the entropy-based secret detection is wrong: {quiet_report} / {salted_report}")
    # 5) line numbers and ordering: findings are sorted by path then line, and each points at the line that
    #    actually contains the offending construct.
    by_path = {path: [finding.line for finding in report if finding.path == path] for path in CORPUS}
    check(by_path["app/routes.py"] == [5, 7, 8] and by_path["app/db.py"] == [2] and by_path["app/auth.py"] == [3]
          and [finding.path for finding in report] == sorted(finding.path for finding in report)
          and all(0 < line <= len(CORPUS[path]) for path, lines in by_path.items() for line in lines)
          and "eval" in CORPUS["app/rules.py"][1] and "verify=False" in CORPUS["app/edge.py"][2],
          f"the reported lines are wrong: {by_path}")
    # 6) the route rule: a decorated route is trusted even with a blank line in between, while the
    #    undecorated one is reported.
    variants = {"app/r.py": ("@app.get('/ok')", "", "@login_required", "def ok():", "    return 1",
                             "@app.get('/public')", "def public():", "    return 2")}
    check([(finding.rule, finding.line) for finding in review(variants)[0]] == [("missing_authz", 6)],
          f"the route rule is wrong: {review(variants)[0]}")
    # 7) the checklist is complete and consistent: one item per detector, unique rule names and codes, and
    #    a severity that never falls below the review floor.
    check(len(CHECKLIST) == 8 and len({item.rule for item in CHECKLIST}) == 8
          and {finding.rule for finding in report} == {item.rule for item in CHECKLIST}
          and all(re.fullmatch(r"CWE-\d{2,3}", item.cwe) for item in CHECKLIST)
          and {item.severity for item in CHECKLIST} == {"critical", "high", "medium"}
          and sum(1 for item in CHECKLIST if item.severity == "critical") == 3
          and all(SEVERITY_RANK[item.severity] >= SEVERITY_RANK["medium"] for item in CHECKLIST)
          and len({item.cwe for item in CHECKLIST}) == 8,
          "the checklist is not consistent")
    # 8) boundaries: an empty corpus, a comment-only file and a file with no lines are all clean, and a
    #    single-rule corpus only reports that rule.
    check(review({}) == ([], "pass") and review({"app/empty.py": ()}) == ([], "pass")
          and review({"app/comments.py": ("# API_KEY = 'sk_live_1'", "# eval(x)")}) == ([], "pass")
          and [finding.rule for finding in review({"app/one.py": ("eval(payload)",)})[0]] == ["dynamic_code"]
          and scan_file("app/none.py", ()) == [] and counts == per_rule(report),
          "the empty and comment-only boundaries are wrong")
    # 9) reproducibility: the scan, the ordering and the decision are byte-identical across runs.
    check(review(CORPUS) == (report, decision) and scan_file("app/db.py", CORPUS["app/db.py"])
          == scan_file("app/db.py", CORPUS["app/db.py"]) and entropy_oracle("abab") == shannon_entropy("abab") == 1.0
          and review(HARDENED) == (hardened_report, hardened_decision) and hardened_decision == "pass",
          "the scanner must be reproducible")
    print(f"files={len(CORPUS)} findings={len(report)} decision={decision} "
          f"by_rule={{{', '.join(f'{rule}:{count}' for rule, count in sorted(counts.items()))}}} "
          f"hardened_findings={len(hardened_report)} hardened_decision={hardened_decision} "
          f"entropy_of_planted_secret={shannon_entropy('sk_live_9f3ac7b1d2e4'):.3f}")
    print("PASS")


if __name__ == "__main__":
    main()
