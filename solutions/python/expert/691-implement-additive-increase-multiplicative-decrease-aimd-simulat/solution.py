"""Task 691 - Implement additive-increase multiplicative-decrease (AIMD) simulation.  (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)

AIMD is the control law the Internet converged on, and here the reason shows up as exact algebra
rather than a slogan.  A window that grows by `a` per round trip and multiplies by `b < 1` at every
congestion event satisfies, for two flows sharing a link,

    w_1 - w_2  ->  b * (w_1 - w_2)      at every event (the additive terms are equal)
    w_i        =   a_i * k / ((1 - b) * rtt_i)   at equilibrium, k rounds between events

so the fairness gap shrinks geometrically while the additive term lifts every window equally, and the
equilibrium rate ratio is (a_1/rtt_1)/(a_2/rtt_2) — the well-known RTT bias of pure AIMD when the
rates are normalised by RTT.  The other three combinations of increase and decrease fail on purpose:
MIMD preserves the *ratio* exactly, AIAD preserves the *difference* exactly, and MIAD moves the ratio
strictly away from one at every event, settling at a stable but unfair point.  Each of those negative
results is checked as an identity or a strict monotonicity, which is what makes this simulation a
statement about the control law and not a curve that happens to look right.

Modelled per round: window updates, synchronous congestion events (a droptail bottleneck tells both
flows at once), the RTT bias through the increase term, and a memoryless per-packet loss process
driven by a fixed-seed splitmix64 PRNG.  Simplifications: no queue dynamics, no slow start, no SACK
or reordering effects, and the congestion signal is a schedule or a loss process rather than a real
buffer.  Verified: the sawtooth bounds W*b <= w <= W with an exactly known cycle average, throughput
against both the Mathis (1.22/sqrt(p)) and the AIMD SQRT (sqrt(3/(2p))) formulas, the geometric
decay of the fairness gap, Jain's index for converged pairs, the RTT bias, and the MIMD/AIAD/MIAD
counterexamples.
"""

import math
from collections import namedtuple

MIN_WINDOW = 1.0                                    # a window never drops below one packet
MI_FACTOR = 1.1                                     # the multiplicative-increase factor

Aimd = namedtuple("Aimd", "name add_up mul_up mul_down add_down")
LAWS = {
    "aimd": Aimd("aimd", 1.0, 0.0, 0.5, 0.0),       # +1 per RTT, halve on loss
    "mimd": Aimd("mimd", 0.0, MI_FACTOR, 0.9, 0.0),  # *1.1 per RTT, *0.9 on loss
    "aiad": Aimd("aiad", 1.0, 0.0, 0.0, 1.0),       # +1 per RTT, -1 on loss
    "miad": Aimd("miad", 0.0, 1.01, 0.0, 1.0),      # *1.01 per RTT, -1 on loss
}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def increase(window, law, rounds=1.0):
    if law.mul_up > 0.0:
        return window * law.mul_up ** rounds
    return window + law.add_up * rounds


def decrease(window, law):
    if law.mul_down > 0.0:
        return max(window * law.mul_down, MIN_WINDOW)
    return max(window - law.add_down, MIN_WINDOW)


def make_prng(seed):
    """splitmix64, so the loss process is identical on every run."""
    state = seed & 0xFFFFFFFFFFFFFFFF

    def nxt():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    return nxt


def sawtooth(a=1.0, b=0.5, window=1.0, period=20, cycles=10):
    """One flow: `period` rounds of additive increase, then one multiplicative decrease."""
    trace, peaks = [window], []
    for _ in range(cycles):
        for _ in range(period):
            window += a
            trace.append(window)
        peaks.append(window)
        window = max(window * b, MIN_WINDOW)
        trace.append(window)
    return trace, peaks


def mathis_throughput(loss_rate):
    """Mathis et al.: throughput ~ 1.22/sqrt(p) MSS per RTT."""
    return 1.22 / math.sqrt(loss_rate)


def sqrt_throughput(loss_rate):
    """The AIMD SQRT formula (Padhye et al.): sqrt(3/(2p)) MSS per RTT."""
    return math.sqrt(3.0 / (2.0 * loss_rate))


def jain_index(values):
    """Jain's fairness index: (sum x)^2 / (n * sum x^2); one means a perfectly fair split."""
    total = sum(values)
    squares = sum(value * value for value in values)
    return 1.0 if squares == 0 else total * total / (len(values) * squares)


def run_pair(law_one, law_two, windows, rounds, period, rtts=(1.0, 1.0)):
    """Two flows on one link, with a synchronous congestion event every `period` rounds.

    A flow with a longer RTT fits fewer additive steps into a round (its increase per round is
    a/rtt), and the event is delivered to both flows at the same instant — the droptail model."""
    laws = (law_one, law_two)
    state = [float(windows[0]), float(windows[1])]
    gap, ratio, snapshots = [], [], []
    for _ in range(rounds // period):
        for _ in range(period):
            state = [max(increase(state[i], laws[i], 1.0 / rtts[i]), MIN_WINDOW) for i in range(2)]
        gap.append(state[0] - state[1])
        ratio.append(state[0] / state[1])
        snapshots.append(tuple(state))
        state = [decrease(state[i], laws[i]) for i in range(2)]
    return {"gap": gap, "ratio": ratio, "windows": snapshots, "final": tuple(state)}


def main():
    check(LAWS["aimd"].add_up == 1.0 and LAWS["aimd"].mul_down == 0.5,
          "the reference law is the classic +1 / -50% AIMD")
    check(increase(10.0, LAWS["aimd"], 3.0) == 13.0, "additive increase is linear in the rounds")
    check(abs(increase(10.0, LAWS["mimd"], 2.0) - 10.0 * MI_FACTOR ** 2) < 1e-9,
          "multiplicative increase is exponential in the rounds")
    check(decrease(10.0, LAWS["aimd"]) == 5.0, "multiplicative decrease halves the window")
    check(decrease(0.5, LAWS["aiad"]) == MIN_WINDOW and decrease(1.0, LAWS["aiad"]) == MIN_WINDOW,
          "the window is floored at one packet")

    trace, peaks = sawtooth(period=20, cycles=10)
    check(len(trace) == 1 + 10 * 21, "the trace holds one window per round plus the initial value")
    check(all(MIN_WINDOW <= value <= peaks[-1] for value in trace),
          "the window stays between one packet and the largest peak")
    for index, peak in enumerate(peaks):
        base = 1 + index * 21
        ramp = trace[base:base + 20]
        check(ramp == sorted(ramp) and ramp[-1] == peak,
              "each congestion-avoidance ramp climbs monotonically to the peak")
        check(abs(trace[base + 20] - peak * 0.5) < 1e-9, "and the event halves it")
        if index:
            check(abs((ramp[0] - 1.0) - 0.5 * peaks[index - 1]) < 1e-9,
                  "so the next cycle starts exactly at half of the peak that caused the loss")
    late = peaks[-1]
    cycle_mean = sum(trace[1 + 9 * 21:1 + 9 * 21 + 20]) / 20.0
    check(abs(cycle_mean - (late - 9.5)) < 1e-9,
          "the mean of a twenty-round ramp from W-19 to W is exactly W - 9.5")
    check(all(abs(peaks[k + 1] - (0.5 * peaks[k] + 20.0)) < 1e-9 for k in range(len(peaks) - 1)),
          "the peaks follow the exact recurrence W_next = b*W + a*period")
    check(abs(late - 40.0) < 0.05,
          "and they converge to a*period/(1-b) = 40")
    check(abs(cycle_mean / late - 0.75) < 0.02,
          "which lands on the classic 0.75*W_max continuum result within two percent")
    check(sqrt_throughput(0.02) < mathis_throughput(0.02) * 1.01,
          "the SQRT and Mathis throughput formulas agree to within a percent")

    prng = make_prng(0x1A1D)
    for loss_rate in (0.002, 0.01, 0.05):
        window, total, samples = 10.0, 0.0, 0
        for _ in range(6000):
            # every packet of the window can be dropped, so the chance per RTT is p*cwnd
            if (prng() % 1000000) / 1000000.0 < min(1.0, loss_rate * window):
                window = decrease(window, LAWS["aimd"])
            else:
                window = increase(window, LAWS["aimd"])
            total += window
            samples += 1
            check(window >= MIN_WINDOW, "the random-loss run keeps the window floor")
        measured = total / samples
        reference = mathis_throughput(loss_rate)
        check(abs(measured - reference) / reference < 0.25,
              f"AIMD at p={loss_rate}: {measured:.2f} MSS/RTT vs Mathis {reference:.2f} "
              f"and SQRT {sqrt_throughput(loss_rate):.2f}")

    pair = run_pair(LAWS["aimd"], LAWS["aimd"], (10.0, 20.0), 800, period=20)
    gaps = pair["gap"]
    check(all(abs(gaps[k + 1] - 0.5 * gaps[k]) < 1e-9 for k in range(len(gaps) - 1)),
          "AIMD shrinks the window difference by exactly b = 0.5 at every event")
    check(abs(gaps[-1]) < 1e-9, "so a pair of AIMD flows reaches an equal split")
    check(jain_index(pair["windows"][-1]) > 0.999,
          f"the converged pair scores {jain_index(pair['windows'][-1]):.4f} on Jain's index")
    check(abs(pair["windows"][-1][0] - 40.0) < 1e-6,
          "and both windows sit at the equilibrium a*period/(1-b) = 40")

    mimd = run_pair(LAWS["mimd"], LAWS["mimd"], (10.0, 20.0), 200, period=20)
    check(max(abs(value / 0.5 - 1.0) for value in mimd["ratio"]) < 1e-12,
          "MIMD preserves the ratio exactly: no event moves the split toward fairness")
    check(abs(jain_index(mimd["windows"][-1]) - jain_index([10.0, 20.0])) < 1e-12,
          "and Jain's index is exactly where it started")

    aiad = run_pair(LAWS["aiad"], LAWS["aiad"], (10.0, 20.0), 800, period=20)
    check(max(abs(value + 10.0) for value in aiad["gap"]) < 1e-12,
          "AIAD preserves the window difference exactly, so the distance to fair never shrinks")

    miad = run_pair(LAWS["miad"], LAWS["miad"], (10.0, 20.0), 800, period=20)
    away = [abs(ratio - 1.0) for ratio in miad["ratio"]]
    check(all(away[k + 1] > away[k] for k in range(len(away) - 1)),
          "MIAD moves the split strictly away from fair at every single event")
    check(miad["ratio"][-1] < 0.9 * miad["ratio"][0],
          f"and it settles at a stable but unfair {miad['ratio'][-1]:.3f} instead of one")
    check(jain_index(miad["windows"][-1]) < 0.85,
          f"leaving Jain at {jain_index(miad['windows'][-1]):.3f} after forty events")
    check(aiad["windows"][-1][0] - aiad["windows"][-1][1] == -10.0,
          "AIAD's gap is the same after forty events as at the start")

    biased = run_pair(LAWS["aimd"], LAWS["aimd"], (10.0, 10.0), 800, period=20, rtts=(1.0, 2.0))
    windows = biased["windows"][-1]
    rates = [windows[0] / 1.0, windows[1] / 2.0]
    check(abs(windows[1] / windows[0] - 0.5) < 0.02,
          "the equilibrium windows are a_i*k/((1-b)*rtt_i), so the long-RTT flow holds half")
    check(abs(rates[0] / rates[1] - 4.0) < 0.2,
          f"AIMD is RTT-biased: the short-RTT flow gets {rates[0] / rates[1]:.2f}x the throughput")
    check(jain_index(rates) < 0.9, "which is the classic unfairness of pure AIMD")

    stable = run_pair(LAWS["aimd"], LAWS["aimd"], (30.0, 30.0), 120, period=20)
    check(all(window[0] == window[1] for window in stable["windows"]),
          "two identical flows stay identical round for round")
    check(jain_index(stable["windows"][-1]) == 1.0, "which is perfect fairness")
    slow_events = run_pair(LAWS["aimd"], LAWS["aimd"], (10.0, 20.0), 800, period=40)
    check(slow_events["windows"][-1][0] > pair["windows"][-1][0],
          "fewer congestion events leave both windows larger")
    check(len(pair["windows"]) == 800 // 20 and len(slow_events["windows"]) == 800 // 40,
          "the number of congestion events is the run length divided by the period")

    print(f"sawtooth: {len(peaks)} cycles, peak {peaks[-1]:.2f} (limit a*period/(1-b) = 40), "
          f"cycle mean {cycle_mean:.2f}")
    print(f"AIMD pair: gap {gaps[0]:.1f} -> {gaps[-1]:.1e} over {len(gaps)} events, "
          f"Jain {jain_index(pair['windows'][-1]):.4f}")
    print(f"RTT bias (rtt 1 vs 2): throughput ratio {rates[0] / rates[1]:.2f}, "
          f"Jain {jain_index(rates):.3f}")
    print("PASS")


if __name__ == "__main__":
    main()
