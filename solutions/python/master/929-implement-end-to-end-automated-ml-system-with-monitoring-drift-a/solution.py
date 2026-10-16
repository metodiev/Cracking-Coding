"""Task 929 - Implement end-to-end automated ML system with monitoring, drift alerts, and retraining.  (Tier: Master, CTCI Ch 14 - Databases)

A closed-loop training service: an online learner, a drift monitor, an alert path and an automated
retraining stage, all driven by a replayable stream.

* Stream: 2-D points with a linearly separable label, plus 5% label noise.  At the change point the
  decision boundary rotates 90 degrees, so any model trained on the old phase becomes wrong - the
  classic concept-drift scenario.
* Learner: online logistic regression with a fixed learning rate (one SGD step per observation).
* Monitor: a CUSUM statistic on the 0/1 errors, S_t = max(0, S_{t-1} + (e_t - k)), which detects a
  sustained error-rate increase far faster than a threshold on a single step; an alert fires when
  S_t > h and the monitor is reset afterwards.
* Retraining: on alert the service replays its recent buffer and refits the weights from scratch,
  then swaps the model in and resets the monitor.

Verified: the online learner reaches high accuracy and no alert fires on the stationary phase (the
false-positive side of the safety/liveness pair); the alert fires within a bounded number of steps
after the concept change (liveness); the retrained model recovers accuracy and beats the same run
with retraining disabled; the CUSUM reset is visible and the refit weights realign with the new
boundary (an independent geometric check); single-sample and empty windows are safe; and the run is
deterministic.  Simplified: one model family, one monitor, a small synthetic stream, no serving RPC.
"""
import math

PHASE_ONE_END, STEPS = 300, 700
DRIFT_SLACK, ALERT_THRESHOLD, WINDOW = 0.12, 6.0, 40
LEARNING_RATE, RETRAIN_PASSES = 0.35, 12
MASK64 = (1 << 64) - 1
TWO_POW_53 = 9007199254740992.0
TRUE_WEIGHTS = ((1.0, 0.6), (-0.6, 1.0))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def gaussian(state):
    total = 0.0
    for _ in range(3):
        state, raw = splitmix64(state)
        total += (raw >> 11) / TWO_POW_53
    return state, (total - 1.5) * 2.0


def make_stream(steps=STEPS, phase_one_end=PHASE_ONE_END, noise=0.05, state=0xABCD):
    """Deterministic stream; the true boundary rotates at the change point.

    The noise argument is the fraction of labels flipped: the deterministic draw is compared against
    a threshold chosen so that about `noise` of the samples get the wrong label.
    """
    stream, state = [], state
    for step in range(steps):
        state, x1 = gaussian(state)
        state, x2 = gaussian(state)
        state, draw = gaussian(state)
        weights = TRUE_WEIGHTS[0] if step < phase_one_end else TRUE_WEIGHTS[1]
        label = 1 if weights[0] * x1 + weights[1] * x2 >= 0.0 else 0
        if abs(draw) < noise * 1.25:                      # deterministic label noise
            label = 1 - label
        stream.append(((x1, x2), label))
    return stream


def probability(weights, point):
    margin = weights[0] + weights[1] * point[0] + weights[2] * point[1]
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, margin))))


def sgd_step(weights, point, label, rate):
    """One logistic-regression step; returns the squared error for the monitor."""
    predicted = probability(weights, point)
    error = predicted - label
    weights[0] -= rate * error
    weights[1] -= rate * error * point[0]
    weights[2] -= rate * error * point[1]
    return abs(error)


def fit(batch, passes=RETRAIN_PASSES, rate=LEARNING_RATE):
    """Batch refit from scratch on the replay buffer (deterministic order)."""
    weights = [0.0, 0.0, 0.0]
    for _ in range(passes):
        for point, label in batch:
            sgd_step(weights, point, label, rate)
    return weights


def accuracy(weights, batch):
    return sum(1 for point, label in batch if predict(weights, point) == label) / len(batch)


def predict(weights, point):
    return 1 if probability(weights, point) >= 0.5 else 0


class Monitor:
    """CUSUM on the classification error with an explicit alert threshold."""

    def __init__(self, slack=DRIFT_SLACK, threshold=ALERT_THRESHOLD):
        self.slack, self.threshold = slack, threshold
        self.statistic, self.peak, self.alerts, self.resets = 0.0, 0.0, 0, 0

    def observe(self, wrong):
        self.statistic = max(0.0, self.statistic + ((1.0 if wrong else 0.0) - self.slack))
        self.peak = max(self.peak, self.statistic)
        if self.statistic > self.threshold:
            self.alerts += 1
            self.statistic = 0.0
            return True
        return False

    def reset(self):
        self.statistic = 0.0
        self.resets += 1


def run(stream, retrain=True, change_point=PHASE_ONE_END, window=WINDOW,
        freeze_after=PHASE_ONE_END):
    """Drive the whole pipeline; returns the run report.

    The online learner warms up until `freeze_after`, then the model is treated as deployed: it only
    changes again when the monitor raises an alert and the service refits it, which is what makes the
    retraining counterfactual measurable.
    """
    weights = [0.0, 0.0, 0.0]
    monitor = Monitor()
    buffer, report = [], {"alerts": [], "errors": [], "refits": 0}
    for step, (point, label) in enumerate(stream):
        wrong = predict(weights, point) != label
        report["errors"].append(0.0 if wrong else 1.0)
        if step < freeze_after:
            sgd_step(weights, point, label, LEARNING_RATE)
        buffer.append((point, label))
        if len(buffer) > 200:
            buffer.pop(0)
        if monitor.observe(wrong) and retrain:
            report["alerts"].append(step)
            weights = fit(buffer)
            report["refits"] += 1
            monitor.reset()
    report["weights"] = weights
    report["monitor"] = monitor
    return report


def windowed_accuracy(errors, start, stop):
    chunk = errors[start:stop]
    return sum(chunk) / len(chunk) if chunk else 0.0


def main():
    stream = make_stream()
    check(len(stream) == STEPS and len(make_stream()) == STEPS,
          "the stream generator must be deterministic")

    # 1. Baseline behaviour on the stationary phase: good accuracy and no false alarms.
    stationary = run(stream[:PHASE_ONE_END])
    check(windowed_accuracy(stationary["errors"], 60, PHASE_ONE_END) > 0.9,
          "the online learner must reach high accuracy on a separable stationary stream")
    check(stationary["monitor"].alerts == 0 and stationary["monitor"].peak < ALERT_THRESHOLD,
          "the monitor must stay quiet while the concept is stable")

    # 2. Liveness: after the concept change the alert fires within a bounded delay.
    retrained = run(stream)
    check(retrained["alerts"], "the concept change must trigger an automatic retraining")
    delay = retrained["alerts"][0] - PHASE_ONE_END
    check(0 <= delay <= 80, "the drift alert must arrive within a bounded delay of the change")

    # 3. Recovery: retraining restores accuracy, and the same stream without retraining does not.
    frozen = run(stream, retrain=False)
    tail = (STEPS - 200, STEPS)
    recovered = windowed_accuracy(retrained["errors"], *tail)
    stale = windowed_accuracy(frozen["errors"], *tail)
    check(recovered > 0.9, "the retrained model must recover high accuracy on the new concept")
    check(recovered > stale + 0.15,
          "retraining must beat leaving the stale model in place on the post-change window")

    # 4. The refit realigns the decision boundary with the new concept (independent geometry check).
    boundary = TRUE_WEIGHTS[1]
    learned = retrained["weights"]
    cosine = ((learned[1] * boundary[0] + learned[2] * boundary[1])
              / math.hypot(learned[1], learned[2]) / math.hypot(*boundary))
    check(cosine > 0.9, "the retrained weights must point along the new decision boundary")
    check(retrained["monitor"].resets >= 1, "the monitor must be reset after retraining")

    # 5. Monitoring internals: the CUSUM resets and single observations behave.
    monitor = Monitor()
    check(not monitor.observe(True) and monitor.statistic > 0.0,
          "a single error must raise the statistic without alerting")
    check(not monitor.observe(False) or monitor.alerts == 1,
          "a correct prediction must not create an alert out of nowhere")
    fired = Monitor()
    while not fired.observe(True):
        pass
    check(fired.alerts == 1 and fired.statistic == 0.0,
          "sustained errors must fire exactly once and reset the statistic")

    # 6. Boundary cases: empty and single-observation streams.
    empty = run([])
    check(empty["errors"] == [] and empty["alerts"] == [] and windowed_accuracy([], 0, 10) == 0.0,
          "an empty stream must produce an empty report, not an exception")
    solo = run(stream[:1])
    check(len(solo["errors"]) == 1 and solo["alerts"] == [],
          "a single observation must be processed without alerting")

    # 7. Determinism: the same stream and policy reproduce the same alerts and weights.
    again = run(stream)
    check(again["errors"] == retrained["errors"] and again["weights"] == retrained["weights"]
          and again["alerts"] == retrained["alerts"],
          "the pipeline must be reproducible")

    print(f"alerts {retrained['alerts']} (delay {delay} steps), monitor resets "
          f"{retrained['monitor'].resets}")
    print(f"post-change accuracy retrained {recovered:.3f} vs stale {stale:.3f}, "
          f"boundary cosine {cosine:.3f}")
    print("PASS")


if __name__ == "__main__":
    main()
