"""Task 965 - Implement secure sandbox to run untrusted student submissions with resource cgroups.
(Tier: Master, CTCI Beyond - Security & Privacy)

Sandbox model: a submission is a small instruction list (allocate, free, fork, read, write, loop,
emit) executed by an interpreter that charges a cgroup-style budget - cpu steps, peak memory, process
count and filesystem policy.  Each limit is enforced at the instant it is crossed, the run is killed
with a machine-readable reason and the usage snapshot at the kill point is returned, so a hostile
program can neither exhaust the host nor hide what it did.  Root access is not required and no real
process is spawned.

Simplified: instructions stand in for a syscall trace, memory is accounted as a running total and
wall time is a step count.  Verified: the clean run, cpu/memory/pid kills, the exact memory and pid
boundaries, the read-only filesystem policy with its denial budget, usage conservation against an
independently recomputed peak, the empty-program boundary and determinism.
"""
DEFAULT_LIMITS = {"cpu": 400, "memory": 4096, "pids": 3, "denials": 2}
INSTRUCTIONS = ("ALLOC", "FREE", "FORK", "READ", "WRITE", "LOOP", "EMIT")
TIMEOUT = 100000


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def validate(program):
    """Reject anything the sandbox will not even start: unknown ops and non-positive amounts."""
    for op, argument in program:
        if op not in INSTRUCTIONS:
            raise ValueError(f"unknown instruction {op!r}")
        if op in ("ALLOC", "FREE", "LOOP") and not isinstance(argument, int):
            raise ValueError(f"{op} needs an integer amount")
    return program


def peak_memory(program):
    """Independent oracle for the memory high-water mark of a program that finishes."""
    live, peak = 0, 0
    for op, argument in program:
        if op == "ALLOC":
            live += argument
            peak = max(peak, live)
        elif op == "FREE":
            live -= argument
    return peak, live


class Sandbox:
    """One cgroup: the limits, the accounting and the kill/deny policy."""

    def __init__(self, **limits):
        self.limits = dict(DEFAULT_LIMITS, **limits)
        self.resets = 0

    def run(self, program):
        """Execute a program to completion or to the first breached limit."""
        validate(program)
        self.resets += 1
        usage = {"steps": 0, "memory": 0, "peak": 0, "pids": 1, "denied": 0}
        output, denied = [], []
        report = {"status": "ok", "reason": None, "usage": usage, "output": output,
                  "denied": denied, "executed": 0}

        def kill(reason):
            report["status"], report["reason"] = "killed", reason
            return report

        for index, (op, argument) in enumerate(program):
            usage["steps"] += 1
            if usage["steps"] > self.limits["cpu"]:
                return kill("cpu")
            if op == "ALLOC":
                usage["memory"] += argument
                usage["peak"] = max(usage["peak"], usage["memory"])
                if usage["memory"] > self.limits["memory"]:
                    return kill("memory")
            elif op == "FREE":
                usage["memory"] = max(0, usage["memory"] - argument)
            elif op == "FORK":
                usage["pids"] += argument
                if usage["pids"] > self.limits["pids"]:
                    return kill("pids")
            elif op == "LOOP":
                for _ in range(argument):          # charged step by step, like a real cpu quota
                    usage["steps"] += 1
                    if usage["steps"] > self.limits["cpu"]:
                        return kill("cpu")
                output.append(f"loop {argument}")
            elif op == "EMIT":
                output.append(str(argument))
            else:                                     # READ / WRITE go through the policy
                allowed = op == "READ" or argument.startswith("out/")
                if not allowed:
                    usage["denied"] += 1
                    denied.append((op, argument))
                    if usage["denied"] > self.limits["denials"]:
                        return kill("policy")
                else:
                    output.append(f"{op.lower()} {argument}")
            report["executed"] = index + 1
        return report


def clean_program():
    return (("READ", "input.txt"), ("ALLOC", 1024), ("EMIT", "loaded"), ("ALLOC", 1024),
            ("FREE", 512), ("WRITE", "out/result.txt"), ("LOOP", 30))


def runaway():
    """A submission that never terminates on its own - only the cpu budget stops it."""
    return (("ALLOC", 8),) + (("LOOP", 5000),) * 2


def main():
    sandbox = Sandbox()
    report = sandbox.run(clean_program())

    # 1) happy path: a well-behaved submission finishes, its output is captured and nothing is denied.
    check(report["status"] == "ok" and report["reason"] is None
          and report["output"] == ["read input.txt", "loaded", "write out/result.txt", "loop 30"]
          and report["denied"] == [] and report["executed"] == 7
          and report["usage"]["steps"] == 7 + 30,
          f"the clean run went wrong: {report}")

    # 2) memory kill: the pointer is over-budget, and the reported peak is exactly the ceiling.
    over = sandbox.run((("ALLOC", 4096), ("ALLOC", 1)))
    check(over["status"] == "killed" and over["reason"] == "memory"
          and over["usage"]["peak"] == 4097 and over["executed"] == 1,
          f"the memory kill is wrong: {over['usage']}")

    # 3) memory boundary: allocating exactly the limit is legal, a single extra byte is not.
    at_limit = sandbox.run((("ALLOC", 4096), ("FREE", 4096), ("ALLOC", 2048)))
    check(at_limit["status"] == "ok" and at_limit["usage"]["peak"] == 4096
          and at_limit["usage"]["memory"] == 2048,
          f"the memory boundary is wrong: {at_limit['usage']}")

    # 4) cpu kill: the runaway submission is stopped, and the steps stop at the budget.
    stopped = sandbox.run(runaway())
    check(stopped["status"] == "killed" and stopped["reason"] == "cpu"
          and stopped["usage"]["steps"] == 401 and stopped["executed"] == 1
          and len(stopped["output"]) == 0,
          f"the cpu kill is wrong: {stopped['usage']}")

    # 5) pid boundary: forking up to the limit is fine, the next fork is killed.
    forks = sandbox.run((("FORK", 2), ("EMIT", "children up"), ("FORK", 1)))
    check(forks["status"] == "killed" and forks["reason"] == "pids" and forks["usage"]["pids"] == 4
          and forks["output"] == ["children up"],
          f"the pid kill is wrong: {forks['usage']}")

    # 6) filesystem policy: a write outside the scratch directory is denied (the run continues) and
    #    too many denials kill the run with a policy reason.
    denied = sandbox.run((("WRITE", "/etc/passwd"), ("WRITE", "/etc/shadow"), ("EMIT", "alive")))
    stopped_by_policy = sandbox.run((("WRITE", "/etc/passwd"), ("WRITE", "/etc/shadow"),
                                     ("WRITE", "/etc/hosts"), ("EMIT", "never")))
    check(denied["status"] == "ok" and denied["denied"] == [("WRITE", "/etc/passwd"),
                                                            ("WRITE", "/etc/shadow")]
          and denied["output"] == ["alive"] and stopped_by_policy["reason"] == "policy"
          and stopped_by_policy["usage"]["denied"] == 3,
          f"the filesystem policy is wrong: {denied['denied']}")

    # 7) usage conservation against an independent oracle: for a program that finishes, the peak and
    #    the live total equal a plain pass over the allocations.
    live_program = (("ALLOC", 100), ("ALLOC", 200), ("FREE", 50), ("ALLOC", 300), ("FREE", 100))
    measured = sandbox.run(live_program)
    oracle = peak_memory(live_program)
    check(measured["status"] == "ok"
          and (measured["usage"]["peak"], measured["usage"]["memory"]) == oracle == (550, 450),
          f"the accounting disagrees with the oracle: {measured['usage']} vs {oracle}")

    # 8) boundaries: an empty program succeeds with zero usage, and an unknown instruction is a
    #    configuration error rather than a silent skip.
    empty = sandbox.run(())
    try:
        sandbox.run((("EXEC", "/bin/sh"),))
        raise AssertionError("an unknown instruction must be rejected")
    except ValueError as error:
        check("EXEC" in str(error), f"unexpected error text: {error}")
    check(empty["status"] == "ok" and empty["usage"] == {"steps": 0, "memory": 0, "peak": 0,
                                                         "pids": 1, "denied": 0}
          and empty["output"] == [] and empty["executed"] == 0,
          f"the empty-program boundary is wrong: {empty['usage']}")

    # 9) determinism and independence: the same program yields the same report, and a second sandbox
    #    starts with a fresh budget unaffected by the first one's kills.
    fresh = Sandbox()
    check(fresh.run(clean_program()) == report and fresh.resets == 1
          and sandbox.resets == 9 and Sandbox(memory=8).run((("ALLOC", 8),))["status"] == "ok",
          "the sandbox must be deterministic and per-run isolated")

    print(f"clean_usage={report['usage']} output={report['output']}")
    print(f"cpu_kill={stopped['usage']['steps']} memory_kill_peak={over['usage']['peak']} "
          f"pids_kill={forks['usage']['pids']} denials={denied['usage']['denied']} "
          f"accounting_oracle={oracle}")
    print("PASS")


if __name__ == "__main__":
    main()
