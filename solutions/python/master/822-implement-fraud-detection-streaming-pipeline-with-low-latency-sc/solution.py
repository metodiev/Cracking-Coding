"""Task 822 - Implement fraud-detection streaming pipeline with low-latency scoring and feedback.  (Tier: Master, CTCI Beyond - Machine Learning & AI)

A fraud pipeline scores every payment event as it arrives under a fixed per-event budget of work, while
the labels that would tell it whether it was right arrive long afterwards.  The whole loop is modelled
deterministically:

  * `FraudPipeline.ingest` deduplicates by event id (exactly once), rejects anything older than the
    watermark (max timestamp seen so far minus the allowed lateness) and parks the rest in a small
    event-time buffer;
  * `FraudPipeline.drain` releases the buffered events whose timestamp has fallen behind the watermark,
    in (timestamp, id) order, and runs the three remaining stages on each of them: causal features from
    the card's sliding window (`raw_features` looks only at strictly earlier timestamps, so events that
    share a timestamp can never influence each other), a logistic score, and a threshold decision
    subject to a per-window alert cap.  Each stage is charged simulated cycles;
  * `batch_reference` recomputes every feature and decision with a completely different implementation -
    it filters the late events by the same watermark rule and then scans the accepted events in
    timestamp order with a two-pointer sliding window - and is the oracle the streaming pass must match;
  * `feedback` plays the delayed labels back into the model: one logistic-regression SGD step per label,
    remembering only the last few feature vectors, so the served model improves without any label ever
    reaching a decision it precedes - which is what the no-leakage check measures by perturbing the
    future of the stream and demanding that every earlier decision stays bit-identical;
  * `run_stream` drives arrivals, delayed labels and a distribution shift, and reports the measured
    latency profile (mean and p99 cycles against the budget), the alert-rate cap, the AUC/recall/
    precision before and after feedback, and the drift response.

Verified: every streaming feature, score and decision equal the batch reference, including when the
stream arrives out of order; the scoring order equals timestamp order; exactly-once scoring with a
complete accounting of processed / duplicate / late events; the latency budget for every decision (not
just the p99) with a bounded number of window lookups; no leakage; the alert cap; the model improving
from a cold start near chance to a useful AUC with measurable precision and recall; the drift detector
firing after the shift and the adapted threshold lowering the alert rate without wrecking the ranking;
determinism; and the degenerate streams (empty, single event, all-fraud, all-legit, duplicate flood).

Simplified: events are small records in one process instead of a Kafka/Flink topology, the cycle count
is a deterministic cost model rather than wall-clock latency (nothing sleeps) and the event-time delay
is counted in ticks, the model is a four-feature logistic regression trained by plain SGD rather than
gradient-boosted trees over a feature store, labels are synthesised from the same features with an
unobservable component so no model can reach AUC 1.0, and drift is a parameter shift in the generator.
The invariants verified - causal features, exactly-once scoring, bounded latency, no leakage, alert-rate
control and measurable learning from delayed feedback - are the ones a production fraud pipeline must
hold.
"""
import heapq

MASK64 = (1 << 64) - 1
FEATURES = 4
BUDGET_CYCLES = 40
INGEST_COST = 3
LOOKUP_COST = 1
SCORE_COST = 2
DECIDE_COST = 2
ALERT_CAP_PER_WINDOW = 6
WINDOW_SIZE = 8
ALLOWED_LATENESS = 5
FEEDBACK_MEMORY = 400
LEARNING_RATE = 0.35
MERCHANT_RISK = {0: 0.0, 1: 0.4, 2: 0.9}
E = 2.718281828459045


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
    """Fixed-seed splitmix64 stream: every run generates the same traffic."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def below(self, bound):
        self.state, value = splitmix64(self.state)
        return value % bound

    def frac(self):
        self.state, value = splitmix64(self.state)
        return (value >> 11) / float(1 << 52)


# ------------------------------------------------------------------ events
class Event:
    __slots__ = ("event_id", "timestamp", "card", "merchant", "amount", "label")

    def __init__(self, event_id, timestamp, card, merchant, amount, label=0):
        self.event_id = event_id
        self.timestamp = timestamp
        self.card = card
        self.merchant = merchant
        self.amount = amount
        self.label = label

    def key(self):
        return (self.timestamp, self.event_id)

    def __repr__(self):
        return f"E{self.event_id}@t{self.timestamp}"


def true_label(rng, amount, velocity, merchant):
    """The hidden rule: the log odds move with the amount, the card's window velocity and the
    merchant's risk, so the signal is learnable but not separable (there is an unobservable draw)."""
    log_odds = (6.0 * (amount / 500.0 - 0.5) + 4.0 * (min(velocity, 6) / 6.0 - 0.4) +
                2.5 * (MERCHANT_RISK[merchant] - 0.35) - 0.3)
    return 1 if rng.frac() < sigmoid(log_odds) else 0


def make_stream(seed, length, shift_at=None):
    """A deterministic stream, labelled from the same causal features the model can see."""
    rng = Rng(seed)
    events = []
    history = []
    for index in range(length):
        timestamp = index // 2
        card = rng.below(12)
        merchant = rng.below(3)
        amount = 20 + rng.below(400)
        if shift_at is not None and index >= shift_at:
            amount = 380 + rng.below(600)
            merchant = 2
        velocity = sum(1 for other in history
                       if timestamp - WINDOW_SIZE <= other[0] < timestamp and other[1] == card)
        events.append(Event(index, timestamp, card, merchant, amount,
                            true_label(rng, amount, velocity, merchant)))
        history.append((timestamp, card, amount, merchant))
    for index in range(0, length - 2, 7):
        events[index], events[index + 1] = events[index + 1], events[index]
    return events


# ------------------------------------------------------------------ features and model
def sigmoid(value):
    if value < -60:
        return 0.0
    if value > 60:
        return 1.0
    return 1.0 / (1.0 + pow(E, -value))


def raw_features(event, history):
    """Causal features from the card's own sliding window (strictly earlier timestamps only)."""
    lookups = 0
    velocity = 0
    for other in history:
        if other.card != event.card:
            continue
        if event.timestamp - WINDOW_SIZE <= other.timestamp < event.timestamp:
            lookups += 1
            velocity += 1
    return ([1.0, event.amount / 500.0, min(velocity, 6) / 6.0, MERCHANT_RISK[event.merchant]],
            lookups)


def score(weights, features):
    return sum(weight * value for weight, value in zip(weights, features))


class FraudPipeline:
    """Deduplicate, buffer by event time, featurise, score, decide - and learn from delayed labels."""

    def __init__(self, threshold=0.6, lateness=ALLOWED_LATENESS):
        self.threshold = threshold
        self.lateness = lateness
        self.weights = [0.0] * FEATURES
        self.seen_ids = set()
        self.buffer = []
        self.history = []
        self.memory = {}
        self.alerts = []
        self.scored_order = []
        self.cycle_history = []
        self.decision_ticks = []
        self.predictions = []
        self.cap_counters = {}
        self.max_timestamp = None
        self.clock = 0
        self.counters = {"processed": 0, "duplicate": 0, "late": 0, "alerted": 0, "allowed": 0,
                         "buffered": 0}

    def ingest(self, event):
        """Stage one: dedup and watermark filtering; the event is buffered, not yet scored."""
        self.clock += 1
        if event.event_id in self.seen_ids:
            self.counters["duplicate"] += 1
            self.cycle_history.append(INGEST_COST)
            return {"status": "duplicate", "event_id": event.event_id, "cycles": INGEST_COST}
        self.seen_ids.add(event.event_id)
        if self.max_timestamp is not None and event.timestamp < self.max_timestamp - self.lateness:
            self.counters["late"] += 1
            self.cycle_history.append(INGEST_COST)
            return {"status": "late", "event_id": event.event_id, "cycles": INGEST_COST}
        self.max_timestamp = event.timestamp if self.max_timestamp is None \
            else max(self.max_timestamp, event.timestamp)
        heapq.heappush(self.buffer, (event.timestamp, event.event_id, event))
        self.counters["buffered"] += 1
        self.cycle_history.append(INGEST_COST)
        return {"status": "buffered", "event_id": event.event_id, "cycles": INGEST_COST}

    def drain(self, force=False):
        """Release the events the watermark has passed, oldest timestamp first."""
        records = []
        while self.buffer:
            timestamp = self.buffer[0][0]
            if not force and self.max_timestamp is not None and \
                    timestamp > self.max_timestamp - self.lateness:
                break
            _, _, event = heapq.heappop(self.buffer)
            records.append(self._score_event(event))
        return records

    def _score_event(self, event):
        cycles = 0
        features, lookups = raw_features(event, self.history)
        cycles += lookups * LOOKUP_COST
        value = score(self.weights, features)
        cycles += FEATURES * SCORE_COST
        probability = sigmoid(value)
        window = event.timestamp // 10
        used = self.cap_counters.get(window, 0)
        alert = probability >= self.threshold and used < ALERT_CAP_PER_WINDOW
        cycles += DECIDE_COST
        if alert:
            self.cap_counters[window] = used + 1
            self.counters["alerted"] += 1
            self.alerts.append(event.event_id)
        else:
            self.counters["allowed"] += 1
        self.counters["processed"] += 1
        self.history.append(event)
        horizon = event.timestamp - WINDOW_SIZE - self.lateness
        if self.history and self.history[0].timestamp < horizon:
            self.history = [item for item in self.history if item.timestamp >= horizon]
        self.scored_order.append(event.key())
        self.memory[event.event_id] = features
        if len(self.memory) > FEEDBACK_MEMORY:
            del self.memory[min(self.memory)]
        self.predictions.append((probability, event.label))
        self.cycle_history.append(cycles)
        self.decision_ticks.append(self.max_timestamp - event.timestamp)
        return {"status": "scored", "event_id": event.event_id, "probability": probability,
                "alert": alert, "cycles": cycles, "lookups": lookups, "features": tuple(features),
                "tick": self.clock}

    def feedback(self, event_id, label):
        """One logistic-regression step for a label that has just arrived."""
        features = self.memory.get(event_id)
        if features is None:
            return False
        error = label - sigmoid(score(self.weights, features))
        for index in range(FEATURES):
            self.weights[index] += LEARNING_RATE * error * features[index]
        return True

    def p99(self):
        ordered = sorted(self.cycle_history)
        return ordered[int(0.99 * (len(ordered) - 1))] if ordered else 0

    def mean_cycles(self):
        return sum(self.cycle_history) / float(len(self.cycle_history)) if self.cycle_history else 0.0

    def max_delay(self):
        return max(self.decision_ticks) if self.decision_ticks else 0


# ------------------------------------------------------------------ oracle and metrics
def batch_reference(events, threshold, weights, lateness=ALLOWED_LATENESS):
    """Independent oracle: the same watermark filter, then a two-pointer scan over timestamp order."""
    accepted = []
    seen = set()
    latest = None
    for event in events:
        if event.event_id in seen:
            continue
        seen.add(event.event_id)
        if latest is not None and event.timestamp < latest - lateness:
            continue
        latest = event.timestamp if latest is None else max(latest, event.timestamp)
        accepted.append(event)
    accepted.sort(key=lambda item: item.key())
    results = {}
    cap_counters = {}
    start = 0
    for position, event in enumerate(accepted):
        while start < position and accepted[start].timestamp < event.timestamp - WINDOW_SIZE:
            start += 1
        velocity = 0
        lookups = 0
        index = start
        while index < position:
            if accepted[index].card == event.card and \
                    accepted[index].timestamp < event.timestamp:
                velocity += 1
                lookups += 1
            index += 1
        features = [1.0, event.amount / 500.0, min(velocity, 6) / 6.0, MERCHANT_RISK[event.merchant]]
        probability = sigmoid(score(weights, features))
        window = event.timestamp // 10
        used = cap_counters.get(window, 0)
        alert = probability >= threshold and used < ALERT_CAP_PER_WINDOW
        if alert:
            cap_counters[window] = used + 1
        results[event.event_id] = (probability, alert, tuple(features), lookups)
    return results


def auc(pairs):
    """The Mann-Whitney statistic: how often a fraud outranks a legitimate event."""
    positives = [value for value, label in pairs if label == 1]
    negatives = [value for value, label in pairs if label == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else (0.5 if positive == negative else 0.0)
    return wins / (len(positives) * len(negatives))


def evaluate_weights(events, weights, rate=0.2):
    """Score a held-out stream with fixed weights: AUC, plus precision/recall when the top `rate`
    share of the stream is flagged (an uncalibrated model is judged by its ranking, not by 0.5)."""
    history = []
    pairs = []
    for event in sorted(events, key=lambda item: item.key()):
        features, _ = raw_features(event, history)
        pairs.append((sigmoid(score(weights, features)), event.label))
        history.append(event)
    ranked = sorted(range(len(pairs)), key=lambda index: (-pairs[index][0], index))
    cutoff = max(1, int(rate * len(pairs)))
    flagged = [pairs[index][1] for index in ranked[:cutoff]]
    positives = sum(1 for _, label in pairs if label == 1)
    precision = sum(flagged) / float(len(flagged))
    recall = sum(flagged) / float(positives) if positives else 0.0
    return auc(pairs), precision, recall


def precision_recall(pairs, threshold=0.6):
    flagged = [(value, label) for value, label in pairs if value >= threshold]
    positives = sum(1 for _, label in pairs if label == 1)
    true_positives = sum(1 for _, label in flagged if label == 1)
    return (true_positives / float(len(flagged)) if flagged else 0.0,
            true_positives / float(positives) if positives else 0.0)


def run_stream(events, delay=8, adapt=False, threshold=0.6, lateness=ALLOWED_LATENESS, learn=True):
    """Ingest every arrival, drain the watermark after each one, play the labels back late."""
    pipeline = FraudPipeline(threshold, lateness)
    labels = {event.event_id: event.label for event in events}
    adaptations = 0
    decisions = {}
    alerts_by_window = {}
    pending = []
    for event in events:
        pipeline.ingest(event)
        for record in pipeline.drain():
            decisions[record["event_id"]] = record
            window = next(item.timestamp for item in events
                          if item.event_id == record["event_id"]) // 10
            alerts_by_window[window] = alerts_by_window.get(window, 0) + (1 if record["alert"] else 0)
            pending.append(record["event_id"])
            if len(pending) > delay:
                oldest = pending.pop(0)
                if learn:
                    pipeline.feedback(oldest, labels[oldest])
        if adapt and pipeline.clock % 40 == 39:
            recent = pipeline.predictions[-40:]
            if recent:
                hot = sum(1 for probability, _ in recent if probability >= pipeline.threshold)
                if hot > 0.25 * len(recent):
                    pipeline.threshold = min(0.9, pipeline.threshold + 0.05)
                    adaptations += 1
    for record in pipeline.drain(force=True):
        decisions[record["event_id"]] = record
        window = next(item.timestamp for item in events
                      if item.event_id == record["event_id"]) // 10
        alerts_by_window[window] = alerts_by_window.get(window, 0) + (1 if record["alert"] else 0)
        pending.append(record["event_id"])
    if learn:
        for event_id in pending:
            pipeline.feedback(event_id, labels[event_id])
    return {"pipeline": pipeline, "decisions": decisions, "alerts_by_window": alerts_by_window,
            "auc_first": auc(pipeline.predictions[:max(1, delay)]),
            "adaptations": adaptations,
            "auc_last": auc(pipeline.predictions[-80:]),
            "auc_all": auc(pipeline.predictions)}


# ----------------------------------------------------------------------- checks
def main():
    events = make_stream(0xF00D, length=800)

    # 1. out-of-order arrivals inside the watermark: the streaming pass equals the batch reference
    out_of_order = sum(1 for left, right in zip(events, events[1:])
                       if left.timestamp > right.timestamp)
    check(out_of_order > 0, "the fixture must contain out-of-order arrivals")
    streamed = run_stream(events, learn=False)
    pipeline = streamed["pipeline"]
    reference = batch_reference(events, 0.6, [0.0] * FEATURES)
    check(len(streamed["decisions"]) == len(reference),
          f"the streaming pass and the reference must score the same events: "
          f"{len(streamed['decisions'])} vs {len(reference)}")
    for event_id, (probability, alerted, features, lookups) in reference.items():
        record = streamed["decisions"][event_id]
        check(record["features"] == features, f"{event_id}: the features must match the reference")
        check(record["lookups"] == lookups, f"{event_id}: the lookups must match the reference")
        check(abs(record["probability"] - probability) < 1e-12,
              f"{event_id}: the probability must match the reference")
        check(record["alert"] == alerted, f"{event_id}: the alert must match the reference")
    check(pipeline.scored_order == sorted(pipeline.scored_order),
          "events must be scored in timestamp order even when they arrive out of order")

    # 2. exactly-once scoring and a complete accounting of every arrival
    counters = pipeline.counters
    check(counters["duplicate"] == 0, "a fresh stream has no duplicates")
    check(counters["late"] == 0, "the fixture stays inside the lateness window")
    check(counters["processed"] == len(streamed["decisions"]), "processed must match the decisions")
    check(counters["processed"] + counters["duplicate"] + counters["late"] == len(events),
          f"every event must be accounted for: {counters}")
    check(counters["alerted"] + counters["allowed"] == counters["processed"],
          "every scored event is either alerted or allowed")
    check(len(pipeline.alerts) == counters["alerted"] and
          len(set(pipeline.alerts)) == len(pipeline.alerts),
          "each alert must be raised exactly once")
    duplicate = Event(3, 100, 1, 1, 999, 0)
    check(pipeline.ingest(duplicate)["status"] == "duplicate", "a repeated id must be rejected")
    check(pipeline.ingest(duplicate)["status"] == "duplicate", "and again on the next arrival")
    check(pipeline.counters["processed"] + pipeline.counters["duplicate"] +
          pipeline.counters["late"] == len(events) + 2, "the counters must stay complete")

    # 3. late arrivals are rejected instead of corrupting a window
    late_stream = make_stream(0x1234, length=60)
    late_pipeline = FraudPipeline()
    for event in late_stream:
        late_pipeline.ingest(event)
        late_pipeline.drain()
    stale = Event(10 ** 6, late_pipeline.max_timestamp - ALLOWED_LATENESS - 1, 1, 1, 900, 1)
    check(late_pipeline.ingest(stale)["status"] == "late",
          "an event older than the watermark must be rejected")
    check(late_pipeline.counters["late"] == 1, "the late counter must advance")
    check(all(entry[2].event_id != stale.event_id for entry in late_pipeline.buffer),
          "the rejected event must not enter the buffer")
    check(len(late_pipeline.buffer) <= 2 * ALLOWED_LATENESS + 2,
          f"the event-time buffer must stay bounded, got {len(late_pipeline.buffer)}")

    # 4. no leakage: perturbing the future leaves every earlier decision untouched
    prefix = run_stream(events[:100])["decisions"]
    extended = run_stream(events[:100] + [Event(10 ** 6 + index, 200 + index, 5, 2, 900, 1)
                                          for index in range(12)])["decisions"]
    for event_id, record in prefix.items():
        check(abs(record["probability"] - extended[event_id]["probability"]) < 1e-12,
              f"{event_id}: a later event must not change an earlier decision")
        check(record["alert"] == extended[event_id]["alert"],
              f"{event_id}: a later event must not change an earlier alert")

    # 5. the latency budget holds for every decision, and the window lookups stay bounded
    check(pipeline.p99() <= BUDGET_CYCLES,
          f"the p99 cost {pipeline.p99()} must stay inside {BUDGET_CYCLES} cycles")
    check(pipeline.mean_cycles() <= BUDGET_CYCLES / 2.0,
          f"the mean cost {pipeline.mean_cycles():.1f} must leave headroom")
    check(all(cycles <= BUDGET_CYCLES for cycles in pipeline.cycle_history),
          "no single decision may exceed the latency budget")
    worst = max(record["lookups"] for record in streamed["decisions"].values())
    check(INGEST_COST + worst * LOOKUP_COST + FEATURES * SCORE_COST + DECIDE_COST <= BUDGET_CYCLES,
          "the worst case, not only the measured p99, must fit the budget")
    check(worst <= 2 * WINDOW_SIZE,
          f"the window lookups must stay bounded by the window, got {worst}")
    check(len(pipeline.history) <= 2 * (WINDOW_SIZE + ALLOWED_LATENESS) + 2,
          f"the feature state must be pruned to the window plus lateness, got "
          f"{len(pipeline.history)} entries")
    check(pipeline.max_delay() <= ALLOWED_LATENESS + 1,
          f"an event must be scored within the lateness window, worst delay "
          f"{pipeline.max_delay()} ticks")

    # 6. the alert cap binds and the per-window counts still sum to the alerts raised
    for window, alerts in sorted(streamed["alerts_by_window"].items()):
        check(alerts <= ALERT_CAP_PER_WINDOW, f"window {window} raised {alerts} alerts")
    check(sum(streamed["alerts_by_window"].values()) == counters["alerted"],
          "the per-window counts must sum to the alerts raised")
    shifted = make_stream(0xD21F7, length=240, shift_at=120)
    hot = run_stream(shifted)
    check(any(alerts == ALERT_CAP_PER_WINDOW for alerts in hot["alerts_by_window"].values()),
          "a hot window must hit the alert cap")
    check(all(alerts <= ALERT_CAP_PER_WINDOW for alerts in hot["alerts_by_window"].values()),
          "the cap must hold on the shifted traffic as well")

    # 7. delayed feedback turns a cold model into a useful one, measured on a held-out half
    train_events, test_events = events[:len(events) // 2], events[len(events) // 2:]
    cold_auc, cold_precision, _ = evaluate_weights(test_events, [0.0] * FEATURES)
    check(abs(cold_auc - 0.5) < 1e-9,
          f"before any label arrives every event scores identically, AUC {cold_auc}")
    trained = run_stream(train_events, delay=6)
    check(abs(trained["auc_first"] - 0.5) < 1e-9,
          f"the first predictions are made by the cold model, AUC {trained['auc_first']}")
    check(trained["pipeline"].weights != [0.0] * FEATURES, "the model must actually have learned")
    trained_auc, precision, recall = evaluate_weights(test_events, trained["pipeline"].weights)
    ceiling = evaluate_weights(events, [0.0, 6.0, 4.0, 2.5])[0]
    whole_auc = evaluate_weights(events, trained["pipeline"].weights)[0]
    check(ceiling > 0.75,
          f"the fixture must carry a learnable signal, oracle AUC {ceiling:.3f}")
    check(trained_auc > 0.65,
          f"feedback must lift the held-out AUC well above chance, got {trained_auc:.3f}")
    check(trained_auc > cold_auc + 0.15,
          f"the improvement must be large: {cold_auc:.3f} -> {trained_auc:.3f}")
    check(whole_auc > 0.7,
          f"the trained model must rank the traffic well, AUC {whole_auc:.3f}")
    check(whole_auc >= 0.9 * ceiling,
          f"the trained model must recover most of the oracle's ranking: {whole_auc:.3f} vs "
          f"{ceiling:.3f}")
    check(recall > 0.35, f"the trained model must find a share of the fraud, recall {recall:.3f}")
    check(precision > 0.2, f"and must not flag everything, precision {precision:.3f}")
    check(precision > cold_precision,
          f"the precision must improve: {cold_precision:.3f} -> {precision:.3f}")
    check(trained["auc_last"] > 0.55,
          f"the online AUC must move above chance, got {trained['auc_last']:.3f}")

    # 8. drift: the shifted traffic pushes the alert rate up, adaptation brings it back down
    naive = run_stream(shifted, adapt=False)
    adapted = run_stream(shifted, adapt=True)
    naive_rate = len(naive["pipeline"].alerts) / float(naive["pipeline"].counters["processed"])
    adapted_rate = len(adapted["pipeline"].alerts) / float(adapted["pipeline"].counters["processed"])
    check(naive_rate > 0.0, "the shifted traffic must trigger alerts")
    check(adapted_rate <= naive_rate + 1e-9,
          f"adaptation must not raise the alert rate: {adapted_rate:.3f} vs {naive_rate:.3f}")
    check(adapted["pipeline"].threshold > naive["pipeline"].threshold,
          f"the adaptive pipeline must have raised its threshold after the shift: "
          f"{adapted['pipeline'].threshold:.2f} vs {naive['pipeline'].threshold:.2f}")
    check(adapted["adaptations"] > 0, "the drift detector must have fired")
    check(adapted["auc_last"] >= naive["auc_last"] - 0.02,
          f"adaptation must keep the ranking quality: {adapted['auc_last']:.3f} vs "
          f"{naive['auc_last']:.3f}")

    # 9. determinism and the degenerate streams
    again = run_stream(events)
    twice = run_stream(events)
    check(again["pipeline"].alerts == twice["pipeline"].alerts,
          "two identical runs must alert identically")
    check(again["pipeline"].counters == twice["pipeline"].counters, "the counters must be identical")
    check(again["pipeline"].weights == twice["pipeline"].weights, "the models must be identical")
    check(len(pipeline.alerts) == pipeline.counters["alerted"],
          "the frozen run must account for its alerts")
    empty = run_stream([])
    check(empty["pipeline"].counters["processed"] == 0 and empty["pipeline"].p99() == 0,
          "an empty stream must do nothing")
    single = run_stream([Event(1, 0, 1, 2, 500, 1)])
    check(single["pipeline"].counters["processed"] == 1 and single["auc_last"] == 0.5,
          "a single event is scored and cannot support an AUC")
    all_fraud = [Event(index, index // 3, index % 4, 2, 400, 1) for index in range(60)]
    fraud_run = run_stream(all_fraud)
    check(fraud_run["pipeline"].counters["processed"] == 60, "an all-fraud stream must be scored")
    check(fraud_run["auc_last"] == 0.5, "an all-fraud stream has no ranking signal")
    legit = [Event(index, index // 3, index % 4, 0, 30, 0) for index in range(60)]
    check(run_stream(legit)["pipeline"].counters["processed"] == 60,
          "an all-legit stream must be scored")
    flood = [Event(7, index, 1, 1, 100, 0) for index in range(20)]
    flood_run = run_stream(flood)
    check(flood_run["pipeline"].counters["duplicate"] == 19,
          f"a duplicate flood must be rejected everywhere but once, got "
          f"{flood_run['pipeline'].counters['duplicate']}")
    check(flood_run["pipeline"].counters["processed"] == 1, "only the first copy may be scored")

    print(f"pipeline: {counters['processed']} scored, {counters['alerted']} alerts, mean "
          f"{pipeline.mean_cycles():.1f} / p99 {pipeline.p99()} cycles (budget {BUDGET_CYCLES}), "
          f"worst lookups {worst}, worst event-time delay {pipeline.max_delay()} ticks")
    print(f"feedback: cold AUC {cold_auc:.3f} -> held-out {trained_auc:.3f} / whole-stream "
          f"{whole_auc:.3f} (oracle ceiling {ceiling:.3f}); precision {cold_precision:.3f} -> "
          f"{precision:.3f}, recall {recall:.3f}")
    print(f"drift: alert rate {naive_rate:.3f} -> {adapted_rate:.3f}, threshold "
          f"{naive['pipeline'].threshold:.2f} -> {adapted['pipeline'].threshold:.2f}")
    print("PASS")


if __name__ == "__main__":
    main()
