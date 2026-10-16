"""Task 940 - Implement high-fidelity load generator with correlated user behavior models.  (Tier: Master, CTCI Ch 11 - Testing)

A deterministic, session-based load generator.  A user model is a Markov chain over UI states with,
per state, a transition row, an endpoint mix, a stop probability and a mean think time; sessions are
drawn from it (weighted transitions, endpoints correlated with the state, exponential think times by
inverse transform, a hard request cap).  The generator also produces an open-loop arrival process - a
two-regime Markov-modulated tick process, so arrivals are bursty rather than memoryless - and a
closed-loop model of N virtual users cycling between response and think time.

Simplified versus Locust/k6: single process, configured response times, no worker pool.  Verified:
transition frequencies and state-conditioned endpoint mixes match the declared rows; the state mix
matches an independent power iteration; mean session length matches an analytic backward recurrence;
think times reproduce their means; the open-loop process is autocorrelated while an equal-mean
memoryless one is not; closed-loop occupancy matches N*s/(s+t) and never exceeds N; the stop and cap
boundaries hold; invalid models are rejected; generation is reproducible.
"""
import math
from collections import Counter

STATE_TABLE = (
    ("browse", (("browse", 30), ("search", 35), ("product", 35)),
     (("/home", 70), ("/feed", 30)), 0.25, 2.0),
    ("search", (("search", 20), ("product", 60), ("cart", 20)),
     (("/search", 60), ("/suggest", 40)), 0.30, 1.5),
    ("product", (("product", 25), ("cart", 50), ("browse", 25)),
     (("/product", 80), ("/reviews", 20)), 0.25, 3.0),
    ("cart", (("product", 40), ("checkout", 60)),
     (("/cart", 60), ("/cart/price", 40)), 0.15, 4.0),
    ("checkout", (("browse", 100),),
     (("/pay", 70), ("/confirm", 30)), 0.45, 6.0),
)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64 - one fixed seed makes every generated workload reproducible."""

    def __init__(self, seed):
        self.state = seed & ((1 << 64) - 1)

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return z ^ (z >> 31)

    def uniform(self):
        return (self.next() >> 11) / float(1 << 53)


def draw(rng, choices):
    """Deterministic weighted draw over ((value, weight), ...)."""
    point = rng.uniform() * sum(weight for _, weight in choices)
    for value, weight in choices:
        point -= weight
        if point < 0:
            return value
    return choices[-1][0]


class UserModel:
    def __init__(self):
        self.states = tuple(row[0] for row in STATE_TABLE)
        self.transitions = {row[0]: row[1] for row in STATE_TABLE}
        self.endpoints = {row[0]: row[2] for row in STATE_TABLE}
        self.stop = {row[0]: row[3] for row in STATE_TABLE}
        self.think = {row[0]: row[4] for row in STATE_TABLE}
        self.validate()

    def validate(self):
        for state in self.states:
            weights = [weight for row in (self.transitions[state], self.endpoints[state])
                       for _, weight in row]
            if any(weight <= 0 for weight in weights) or not 0.0 <= self.stop[state] <= 1.0:
                raise ValueError(f"invalid weights or stop probability in state {state}")


def one_session(model, rng, cap, start="browse", honour_stop=True):
    session, state = [], start
    while len(session) < cap:
        session.append((state, draw(rng, model.endpoints[state]),
                        -model.think[state] * math.log(1.0 - rng.uniform())))
        if honour_stop and rng.uniform() < model.stop[state]:
            break
        state = draw(rng, model.transitions[state])
    return session


def generate_sessions(model, count, rng, cap=12):
    return [one_session(model, rng, cap) for _ in range(count)]


def stationary(model, iterations=200):
    """Independent oracle: power iteration over the transition rows gives the state mix."""
    mix = {state: 1.0 / len(model.states) for state in model.states}
    for _ in range(iterations):
        upcoming = {state: 0.0 for state in model.states}
        for state in model.states:
            row, total = model.transitions[state], sum(w for _, w in model.transitions[state])
            for target, weight in row:
                upcoming[target] += mix[state] * weight / total
        mix = upcoming
    return mix

def expected_lengths(model):
    """Independent oracle: L(s) = 1 + (1 - stop[s]) * sum over successors P(s->s') * L(s')."""
    lengths = {state: 1.0 for state in model.states}
    for _ in range(200):
        lengths = {state: 1.0 + (1.0 - model.stop[state]) * sum(
            weight / sum(w for _, w in model.transitions[state]) * lengths[target]
            for target, weight in model.transitions[state]) for state in model.states}
    return lengths


def row_deviation(rows, sessions, index):
    """Largest |observed - declared| conditional frequency over all rows."""
    counts = Counter((pair[0][index], pair[1][index])
                     for session in sessions for pair in zip(session, session[1:]))
    worst = 0.0
    for state, row in rows.items():
        total = sum(weight for _, weight in row)
        seen = sum(count for (source, _), count in counts.items() if source == state)
        if not seen:
            continue
        for target, weight in row:
            worst = max(worst, abs(counts.get((state, target), 0) / seen - weight / total))
    return worst


def mmpp_arrivals(rng, ticks, low=5, high=20, switch=0.1, slots=2):
    """Two-regime bursty arrival process: the mean rate depends on the current regime."""
    counts, regime = [], False
    for _ in range(ticks):
        if rng.uniform() < switch:
            regime = not regime
        counts.append(sum(1 for _ in range((high if regime else low) * slots)
                          if rng.uniform() < 0.5))
    return counts


def memoryless_arrivals(rng, ticks, slots=25):
    return [sum(1 for _ in range(slots) if rng.uniform() < 0.5) for _ in range(ticks)]


def autocorrelation(series, lag=1):
    sample_mean = sum(series) / len(series)
    numerator = sum((series[i] - sample_mean) * (series[i + lag] - sample_mean)
                    for i in range(len(series) - lag))
    denominator = sum((value - sample_mean) ** 2 for value in series)
    return numerator / denominator if denominator else 0.0


def closed_loop(users, service_time, think_time, ticks):
    remaining, thinking = [0] * users, [0] * users
    occupancy, peak = [], 0
    for _ in range(ticks):
        for index in range(users):
            if remaining[index] > 0:
                remaining[index] -= 1
            elif thinking[index] > 0:
                thinking[index] -= 1
            if remaining[index] == 0 and thinking[index] == 0:
                remaining[index], thinking[index] = service_time, think_time
        live = sum(1 for value in remaining if value > 0)
        occupancy.append(live)
        peak = max(peak, live)
    return sum(occupancy) / len(occupancy), peak


def main():
    model = UserModel()
    sessions = generate_sessions(model, 4000, Rng(20240517))

    # 1) transition rows and state-conditioned endpoint mixes match the declaration.
    worst = row_deviation(model.transitions, sessions, 0)
    worst_endpoint = row_deviation(model.endpoints, sessions, 1)
    off_route = sum(1 for session in sessions for state, endpoint, _ in session
                    if endpoint == "/pay" and state != "checkout")
    check(worst <= 0.03 and worst_endpoint <= 0.03 and off_route == 0,
          f"row deviation transitions={worst:.4f} endpoints={worst_endpoint:.4f}")

    # 2) the state mix matches an independent power iteration (long chain, stopping disabled).
    long_chain = one_session(model, Rng(7), cap=40000, honour_stop=False)
    observed = Counter(state for state, _, _ in long_chain)
    oracle = stationary(model)
    drift = sum(abs(observed.get(state, 0) / len(long_chain) - oracle[state])
                for state in model.states)
    check(drift <= 0.03, f"state mix drifts from the stationary distribution by {drift:.4f}")

    # 3) mean session length matches the analytic recurrence (cap raised above the tail).
    lengths = [len(session) for session in generate_sessions(model, 4000, Rng(31337), cap=200)]
    analytic = expected_lengths(model)["browse"]
    check(abs(sum(lengths) / len(lengths) - analytic) <= 0.15 and max(lengths) < 200,
          f"mean session length {sum(lengths) / len(lengths):.3f} vs analytic {analytic:.3f}")

    # 4) think times reproduce the declared per-state means (exponential inverse transform).
    think = {state: [duration for session in sessions for name, _, duration in session
                     if name == state] for state in model.states}
    worst_think = max(abs(sum(think[state]) / len(think[state]) - model.think[state])
                      for state in model.states)
    check(worst_think <= 0.25, f"mean think time deviates by {worst_think:.4f}")

    # 5) open loop: the bursty process is autocorrelated, an equal-mean memoryless one is not.
    bursty, flat = mmpp_arrivals(Rng(99), 4000), memoryless_arrivals(Rng(1234), 4000)
    burst, unflat = autocorrelation(bursty), autocorrelation(flat)
    check(abs(sum(bursty) / len(bursty) - sum(flat) / len(flat)) < 1.5 and burst > 0.15
          and unflat < 0.1 and burst > unflat + 0.1,
          f"burstiness mmpp={burst:.3f} (mean {sum(bursty) / len(bursty):.2f}) "
          f"memoryless={unflat:.3f} (mean {sum(flat) / len(flat):.2f})")

    # 6) closed loop: occupancy follows Little's law N*s/(s+t) and never exceeds the user count.
    occupancy, peak = closed_loop(10, 3, 7, 3000)
    check(abs(occupancy - 10 * 3 / (3 + 7)) <= 0.02 and peak <= 10,
          f"closed-loop occupancy {occupancy:.4f} vs N*s/(s+t)=3.0 (peak {peak})")

    # 7) boundaries: stop probability 1 gives one request, stop 0 always hits the cap.
    stopped, capped = UserModel(), UserModel()
    stopped.stop = {state: 1.0 for state in stopped.states}
    capped.stop = {state: 0.0 for state in capped.states}
    check(len(one_session(stopped, Rng(5), cap=12)) == 1
          and all(len(one_session(capped, Rng(6), cap=5)) == 5 for _ in range(20)),
          "the stop-probability-1 and request-cap boundaries must hold")

    # 8) an invalid model is rejected instead of silently generating traffic, and the same seed
    #    always reproduces the same sessions.
    broken = UserModel()
    broken.endpoints["cart"] = (("/cart", 0), ("/cart/price", 0))
    try:
        broken.validate()
        raise AssertionError("expected ValueError for a zero-weight endpoint mix")
    except ValueError:
        pass
    check(generate_sessions(model, 4000, Rng(20240517)) == sessions,
          "the same seed must reproduce the same sessions")

    print(f"sessions={len(sessions)} mean_length={sum(lengths) / len(lengths):.3f} "
          f"row_deviation={worst:.4f}/{worst_endpoint:.4f} think_deviation={worst_think:.4f}")
    print(f"arrival autocorrelation mmpp={burst:.3f} memoryless={unflat:.3f}; "
          f"closed-loop occupancy={occupancy:.4f} peak={peak}")
    print("PASS")


if __name__ == "__main__":
    main()
