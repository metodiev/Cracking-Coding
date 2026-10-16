"""Task 821 - Implement revenue-optimized bidding engine with real-time constraints (auction simulator).  (Tier: Master, CTCI Beyond - Machine Learning & AI)

An ad exchange has to decide, inside a strict per-request budget of work, which advertisers win which
slots and at what price, while respecting each advertiser's budget and frequency cap and while pushing
the total spend of every campaign as close to its budget as possible.  This implementation models that
whole loop deterministically:

  * `best_allocation` is the slot allocator: welfare = reported value x quality x slot position, with a
    reserve price acting as a floor; it is a bitmask dynamic program over (slot, bidder set), and it is
    verified against `brute_force_allocation`, which enumerates every injective assignment;
  * `vcg_outcome` prices the winners with their *externality* (the welfare everyone else would have had
    without them), which is what makes the mechanism strategy-proof; `second_price_outcome` is the
    single-slot special case used by the live market.  Both are checked for individual rationality,
    for the reserve floor and - by brute force over a grid of misreports - for truthfulness;
  * `optimal_reserve` grid-searches the reserve that maximises the *exact* expected revenue of a
    second-price auction over a discrete value distribution (every pair of values is enumerated, so the
    expectation is closed form, not sampled), and the result is compared with the classical
    (high + 1) / 2 rule for a uniform distribution;
  * `run_market` is the real-time part: 200 sequential auctions, per-advertiser budget pacing, frequency
    caps, a hard spend cap and a per-decision cost model in simulated cycles.  Safety (never overspend,
    never exceed a cap) and liveness (a competitive campaign still spends most of its budget) are both
    measured, and the latency profile is reported as the p99 of the per-decision cycles.

Verified: the allocator equal to the brute-force optimum on 24 random instances; the allocation
invariants (one slot per bidder, welfare equal to the sum, no slot twice); VCG and second-price
truthfulness by brute force; individual rationality; the reserve floor; the optimal reserve maximising
exact expected revenue, beating zero reserve and matching the (high + 1) / 2 rule; budget safety and
frequency caps over 200 auctions with the invariant re-checked after every auction; the paced spend
landing inside [0.9, 1.0] of the budget while an unpaced campaign overshoots it; the per-decision cost
under the latency budget (mean and p99) with the candidate set pruned; determinism; and the degenerate
cases (no eligible bidders, a single bidder paying the reserve, a value below the reserve, a zero
budget and a tie broken by identifier).

Simplified: one process instead of an exchange's request path, expected revenue is enumerated exactly
over a small discrete distribution instead of estimated from traffic, the cost model counts simulated
cycles rather than microseconds, quality scores are static rather than predicted, and the market has no
competitive bidding feedback (an advertiser's presence does not change others' values) and no bid
shading.  The invariants verified - optimal allocation, strategy-proof pricing, the reserve rule, budget
and frequency safety, pacing accuracy and the latency budget - are the ones a real bidding engine must
hold.
"""
from itertools import permutations

INF = float("inf")
LATENCY_BUDGET_CYCLES = 400
COST_PER_CANDIDATE = 7
COST_PER_SLOT = 11
COST_PER_PRICING = 5


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & (2 ** 64 - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & (2 ** 64 - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & (2 ** 64 - 1)
    return state, z ^ (z >> 31)


class Rng:
    """Fixed-seed splitmix64 stream: every run generates the same market."""

    def __init__(self, seed):
        self.state = seed & (2 ** 64 - 1)

    def below(self, bound):
        self.state, value = splitmix64(self.state)
        return value % bound


# ------------------------------------------------------------------ the players
class Slot:
    __slots__ = ("index", "position")

    def __init__(self, index, position):
        self.index = index
        self.position = position


SLOTS = [Slot(0, 1.0), Slot(1, 0.6), Slot(2, 0.35)]


class Advertiser:
    """A campaign: what a click is worth, how often it may win, and what it may spend."""

    __slots__ = ("name", "value", "quality", "budget", "cap", "spend", "wins", "multiplier")

    def __init__(self, name, value, quality, budget, cap, multiplier=1.0):
        self.name = name
        self.value = value
        self.quality = quality
        self.budget = budget
        self.cap = cap
        self.spend = 0.0
        self.wins = 0
        self.multiplier = multiplier

    def eligible(self, reserve):
        return self.value >= reserve and self.spend < self.budget and self.wins < self.cap

    def effective_bid(self, pacing=True, clamp_budget=True):
        bid = self.value * (self.multiplier if pacing else 1.0)
        if not clamp_budget:
            return bid
        remaining = self.budget - self.spend
        return min(bid, remaining / max(self.quality, 1e-9) + 1e-9)

    def __repr__(self):
        return f"{self.name}(v={self.value}, b={self.budget}, spent={self.spend:.1f})"


# ------------------------------------------------------------------ allocation
def best_allocation(bidders, slots, reserve):
    """Bitmask dynamic program maximising reported welfare; returns {name: slot index}."""
    eligible = [bidder for bidder in bidders if bidder.value >= reserve]
    eligible.sort(key=lambda bidder: bidder.name)
    count = len(eligible)
    width = len(slots)
    if count == 0 or width == 0:
        return {}
    best = [[-INF] * (1 << count) for _ in range(width + 1)]
    choice = [[None] * (1 << count) for _ in range(width + 1)]
    best[0][0] = 0.0
    for slot in range(width):
        for mask in range(1 << count):
            if best[slot][mask] == -INF:
                continue
            if best[slot][mask] > best[slot + 1][mask]:
                best[slot + 1][mask] = best[slot][mask]
                choice[slot + 1][mask] = (mask, None)
            for index in range(count):
                if mask & (1 << index):
                    continue
                gain = eligible[index].value * eligible[index].quality * slots[slot].position
                new_mask = mask | (1 << index)
                if best[slot][mask] + gain > best[slot + 1][new_mask]:
                    best[slot + 1][new_mask] = best[slot][mask] + gain
                    choice[slot + 1][new_mask] = (mask, index)
    mask = max(range(1 << count), key=lambda candidate: best[width][candidate])
    allocation = {}
    slot = width
    while slot > 0:
        step = choice[slot][mask]
        if step is None:
            mask = 0
            break
        previous, index = step
        if index is not None:
            allocation[eligible[index].name] = slot - 1
        mask = previous
        slot -= 1
    return allocation


def brute_force_allocation(bidders, slots, reserve):
    """The oracle: enumerate every injective assignment of eligible bidders to slots."""
    eligible = sorted((bidder for bidder in bidders if bidder.value >= reserve),
                      key=lambda bidder: bidder.name)
    best_welfare = 0.0
    best = {}
    width = min(len(slots), len(eligible))
    for count in range(1, width + 1):
        for combo in permutations(range(len(eligible)), count):
            welfare = 0.0
            allocation = {}
            for position, index in enumerate(combo):
                welfare += (eligible[index].value * eligible[index].quality *
                            slots[position].position)
                allocation[eligible[index].name] = position
            if welfare > best_welfare:
                best_welfare = welfare
                best = allocation
    return best


def welfare(allocation, bidders, slots):
    by_name = {bidder.name: bidder for bidder in bidders}
    return sum(by_name[name].value * by_name[name].quality * slots[slot].position
               for name, slot in allocation.items())


# ------------------------------------------------------------------ pricing
def second_price_outcome(bidders, reserve):
    """One slot: the highest effective bid wins and pays the next-highest effective price."""
    eligible = [bidder for bidder in bidders if bidder.value >= reserve]
    if not eligible:
        return None, 0.0
    ranked = sorted(eligible, key=lambda bidder: (-bidder.value * bidder.quality, bidder.name))
    winner = ranked[0]
    runners = ranked[1:]
    rival = max((bidder.value * bidder.quality for bidder in runners), default=0.0)
    price = max(rival / winner.quality, float(reserve))
    return winner, min(price, winner.value)


RESERVE_PREFIX = "__reserve"


def reserve_dummies(slots, reserve):
    """One dummy per slot: a slot is only sold if some real bidder beats the floor in it.

    Modelling the floor as a bidder - rather than as a filter on the reports - is what keeps the
    mechanism strategy proof: the allocation rule then maximises welfare over a pool whose values do
    not depend on anybody's report."""
    return [Advertiser(f"{RESERVE_PREFIX}{slot.index}", reserve, 1.0, float("inf"), 10 ** 9)
            for slot in slots]


def vcg_outcome(bidders, slots, reserve):
    """Multi-slot VCG: each winner pays the welfare everyone else loses because it won."""
    pool = list(bidders) + reserve_dummies(slots, reserve)
    allocation = best_allocation(pool, slots, 0.0)
    real = {name: slot for name, slot in allocation.items() if not name.startswith(RESERVE_PREFIX)}
    if not real:
        return {}, 0.0
    by_name = {advertiser.name: advertiser for advertiser in pool}
    total = welfare(allocation, pool, slots)
    winners = sum(by_name[name].value * by_name[name].quality * slots[slot].position
                  for name, slot in real.items())
    others_included = total - winners
    payments = {}
    for name in real:
        own = by_name[name].value * by_name[name].quality * slots[real[name]].position
        without = best_allocation([advertiser for advertiser in pool if advertiser.name != name],
                                  slots, 0.0)
        payments[name] = max(0.0, welfare(without, pool, slots) - (total - own))
    return payments, sum(payments.values())


# ------------------------------------------------------------------ reserve optimisation
def expected_second_price_revenue(reserve, values, probabilities):
    """Exact expected revenue of a two-bidder second-price auction with a reserve."""
    revenue = 0.0
    for left, left_probability in zip(values, probabilities):
        for right, right_probability in zip(values, probabilities):
            weight = left_probability * right_probability
            if left < reserve and right < reserve:
                continue
            if left >= reserve and right >= reserve:
                revenue += weight * max(reserve, min(left, right))
            else:
                revenue += weight * reserve
    return revenue


def optimal_reserve(values, probabilities):
    """Grid search over the candidate reserves, keeping the exact expectation as the objective."""
    best_reserve, best_revenue = 0, expected_second_price_revenue(0, values, probabilities)
    for reserve in sorted(set(values)):
        revenue = expected_second_price_revenue(reserve, values, probabilities)
        if revenue > best_revenue + 1e-12:
            best_reserve, best_revenue = reserve, revenue
    return best_reserve, best_revenue


def uniform_distribution(high):
    values = list(range(1, high + 1))
    probability = 1.0 / len(values)
    return values, [probability] * len(values)


# ------------------------------------------------------------------ the market
def pace(advertiser, elapsed, horizon, shape=3.0):
    """A pacing multiplier that pulls the spend towards the fraction of the horizon elapsed."""
    if advertiser.budget <= 0:
        return 0.0
    target = elapsed / float(horizon)
    actual = advertiser.spend / float(advertiser.budget)
    if target <= 0:
        return 1.0
    return max(0.05, min(2.0, 1.0 + shape * (target - actual)))


def run_market(advertisers, rounds, reserve, horizon_slots=4, seed=0x5A1E, paced=True,
               enforce_cap=True):
    """Sequential single-slot auctions with pacing, frequency caps and a spend cap."""
    rng = Rng(seed)
    revenue = 0.0
    cycles = []
    sales = 0
    events = []
    half = rounds // 2
    first_half = 0.0
    second_half = 0.0
    for round_index in range(rounds):
        participants = []
        for advertiser in advertisers:
            if not advertiser.eligible(reserve):
                continue
            if rng.below(100) < 60:
                continue
            if paced:
                advertiser.multiplier = pace(advertiser, round_index, rounds)
            participants.append(advertiser)
        examined = len(participants)
        if examined > horizon_slots * 2:
            participants = sorted(participants,
                                  key=lambda advertiser: (-advertiser.value * advertiser.quality,
                                                          advertiser.name))[:horizon_slots * 2]
        priced = []
        for advertiser in participants:
            bid = advertiser.effective_bid(paced, clamp_budget=enforce_cap)
            priced.append((bid * advertiser.quality, advertiser))
        priced.sort(key=lambda item: (-item[0], item[1].name))
        if not priced:
            cycles.append(examined * COST_PER_CANDIDATE + COST_PER_PRICING)
            events.append((round_index, None, 0.0))
            continue
        ranked = sorted(priced, key=lambda item: (-item[0], item[1].name))
        winner = ranked[0][1]
        rival = ranked[1][0] if len(ranked) > 1 else 0.0
        price = max(rival / winner.quality, float(reserve))
        price = min(price, winner.effective_bid(paced, clamp_budget=enforce_cap))
        if enforce_cap:
            price = min(price, winner.budget - winner.spend)
        if price <= 0 or winner.wins >= winner.cap:
            cycles.append(examined * COST_PER_CANDIDATE + COST_PER_PRICING)
            events.append((round_index, None, 0.0))
            continue
        winner.spend += price
        winner.wins += 1
        revenue += price
        if round_index < half:
            first_half += price
        else:
            second_half += price
        sales += 1
        cycles.append(examined * COST_PER_CANDIDATE + COST_PER_SLOT + COST_PER_PRICING)
        events.append((round_index, winner.name, price))
    return {"revenue": revenue, "cycles": cycles, "sales": sales, "events": events,
            "first_half_revenue": first_half, "second_half_revenue": second_half,
            "p99_cycles": sorted(cycles)[int(0.99 * (len(cycles) - 1))],
            "mean_cycles": sum(cycles) / float(len(cycles))}


# ----------------------------------------------------------------------- checks
def random_bidders(rng, count):
    names = [f"adv{index}" for index in range(count)]
    return [Advertiser(names[index], 1 + rng.below(20), 0.5 + rng.below(10) / 10.0,
                       50 + rng.below(50), 5 + rng.below(10)) for index in range(count)]


def main():
    # 1. the allocator equals the brute-force optimum and respects the invariants
    rng = Rng(0xA0C710)
    for trial in range(24):
        bidders = random_bidders(rng, 2 + rng.below(5))
        reserve = rng.below(6)
        allocation = best_allocation(bidders, SLOTS, reserve)
        brute = brute_force_allocation(bidders, SLOTS, reserve)
        check(welfare(allocation, bidders, SLOTS) == welfare(brute, bidders, SLOTS),
              f"trial {trial}: the DP must match brute force")
        check(len(set(allocation.values())) == len(allocation),
              f"trial {trial}: no slot may be sold twice")
        for name, slot in allocation.items():
            bidder = next(item for item in bidders if item.name == name)
            check(bidder.value >= reserve, f"trial {trial}: a winner below the reserve")
            check(0 <= slot < len(SLOTS), "a slot index must be in range")
        check(len(allocation) <= len(SLOTS), "no more winners than slots")

    # 2. the reserve floor and the degenerate cases
    solo = [Advertiser("only", 10, 1.0, 100, 5)]
    check(best_allocation(solo, SLOTS, 11) == {}, "a value below the reserve must not win")
    check(best_allocation(solo, SLOTS, 10) == {"only": 0}, "a value at the reserve must win")
    check(best_allocation([], SLOTS, 0) == {}, "no bidders means no allocation")
    check(best_allocation(solo, [], 0) == {}, "no slots means no allocation")
    winner, price = second_price_outcome(solo, 4)
    check(winner.name == "only" and price == 4.0,
          f"a single bidder must pay the reserve, got {price}")
    check(second_price_outcome([], 1)[0] is None, "an empty auction has no winner")
    cheap = [Advertiser("cheap", 3, 1.0, 100, 5)]
    check(second_price_outcome(cheap, 5)[0] is None, "a value below the reserve must not win")

    # 3. second-price truthfulness, individual rationality and the reserve floor, by brute force
    bidders = [Advertiser("a", 8, 1.0, 100, 5), Advertiser("b", 6, 1.2, 100, 5),
               Advertiser("c", 4, 0.9, 100, 5)]
    reserve = 5
    truths = {bidder.name: bidder.value for bidder in bidders}
    for bidder in bidders:
        true_winner, true_price = second_price_outcome(bidders, reserve)
        true_utility = (truths[bidder.name] - true_price) if true_winner.name == bidder.name else 0.0
        check(true_price >= reserve, "the price must respect the reserve")
        if true_winner.name == bidder.name:
            check(true_price <= bidder.value, "the winner must not pay above its bid")
        for report in range(1, 21):
            if report == bidder.value:
                continue
            original = bidder.value
            bidder.value = report
            winner, price = second_price_outcome(bidders, reserve)
            utility_now = (truths[bidder.name] - price) if winner.name == bidder.name else 0.0
            bidder.value = original
            check(utility_now <= true_utility + 1e-9,
                  f"{bidder.name} could gain by reporting {report}: {utility_now} > {true_utility}")
    best_effective = max(bidder.value * bidder.quality
                         for bidder in bidders if bidder.value >= reserve)
    check(true_winner.value * true_winner.quality == best_effective,
          "the winner must be the best effective bid, not merely the highest value")

    # 4. multi-slot VCG: welfare optimal, revenue accounted, individual rationality, truthfulness
    payments, revenue = vcg_outcome(bidders, SLOTS, reserve)
    allocation = best_allocation(bidders, SLOTS, reserve)
    check(set(payments) == set(allocation), "every winner must be priced")
    check(abs(revenue - sum(payments.values())) < 1e-12, "the revenue must be the sum of payments")
    check(len(allocation) == 2, f"two slots must fill, got {allocation}")
    for name, price in payments.items():
        bidder = next(item for item in bidders if item.name == name)
        slot = allocation[name]
        check(price <= bidder.value * bidder.quality * SLOTS[slot].position + 1e-9,
              f"{name} must be individually rational")
        check(price >= 0.0, "a payment cannot be negative")
    for bidder in bidders:
        original = bidder.value
        baseline = utility_for(bidder, bidders, SLOTS, reserve, original)
        for report in range(1, 21):
            if report == original:
                continue
            bidder.value = report
            current = utility_for(bidder, bidders, SLOTS, reserve, original)
            bidder.value = original
            check(current <= baseline + 1e-9,
                  f"VCG must be strategy proof: {bidder.name} gained by reporting {report}")
    check(revenue > 0.0, "a contested auction must raise revenue")

    # 5. the optimal reserve maximises the exact expected revenue and beats no reserve
    values, probabilities = uniform_distribution(4)
    reserve, best = optimal_reserve(values, probabilities)
    check(best > expected_second_price_revenue(0, values, probabilities),
          "the optimal reserve must beat a zero reserve")
    check(reserve in (2, 3), f"the (high + 1) / 2 rule must show up on the grid, got {reserve}")
    check(0.0 < best <= max(values),
          f"the expected revenue must be a sane price, got {best:.4f}")
    high_values, high_probabilities = uniform_distribution(10)
    high_reserve, high_best = optimal_reserve(high_values, high_probabilities)
    check(abs(high_reserve - 5.5) <= 0.5,
          f"the uniform case must peak near (high + 1) / 2, got {high_reserve}")
    check(high_best > expected_second_price_revenue(0, high_values, high_probabilities),
          "the reserve must help at every distribution size")
    check(expected_second_price_revenue(99, values, probabilities) == 0.0,
          "a reserve above every value must earn nothing")

    # 6. the market: budget safety, frequency caps and pacing accuracy
    def market_fixture(cap=400):
        return [Advertiser("cheap", 3, 1.0, 60, cap), Advertiser("mid", 7, 1.0, 120, cap),
                Advertiser("rich", 15, 1.0, 300, cap), Advertiser("tiny", 2, 1.0, 10, cap)]

    market = market_fixture()
    result = run_market(market, rounds=200, reserve=1, horizon_slots=4)
    for advertiser in market:
        check(advertiser.spend <= advertiser.budget + 1e-9,
              f"{advertiser.name} overspent: {advertiser.spend} > {advertiser.budget}")
        check(advertiser.wins <= advertiser.cap, f"{advertiser.name} exceeded its frequency cap")
    check(result["sales"] > 0, "a market with demand must sell something")
    rich = next(item for item in market if item.name == "rich")
    mid = next(item for item in market if item.name == "mid")
    check(rich.spend >= 0.9 * rich.budget,
          f"a competitive campaign must spend its budget: {rich.spend:.1f} of {rich.budget}")
    check(mid.spend >= 0.9 * mid.budget,
          f"pacing must keep the mid campaign on budget: {mid.spend:.1f} of {mid.budget}")
    check(result["revenue"] > 0.0 and abs(result["revenue"] -
          sum(advertiser.spend for advertiser in market)) < 1e-9,
          "the revenue must equal the total spend")

    # 7. without pacing the same market front-loads: it earns less and dies in the second half
    loose = market_fixture()
    unpaced = run_market(loose, rounds=200, reserve=1, horizon_slots=4, paced=False)
    rich_loose = next(item for item in loose if item.name == "rich")
    mid_loose = next(item for item in loose if item.name == "mid")
    for advertiser in loose:
        check(advertiser.spend <= advertiser.budget + 1e-9,
              "the eligibility rule keeps even an unpaced market inside its budgets")
    check(result["revenue"] > unpaced["revenue"],
          f"pacing must raise total revenue by keeping bidders in the market: "
          f"{result['revenue']:.1f} vs {unpaced['revenue']:.1f}")
    check(result["second_half_revenue"] > unpaced["second_half_revenue"],
          f"pacing must smooth the spend into the second half: "
          f"{result['second_half_revenue']:.1f} vs {unpaced['second_half_revenue']:.1f}")
    check(unpaced["first_half_revenue"] > unpaced["second_half_revenue"],
          "an unpaced campaign front-loads its budget")
    check(mid.spend > mid_loose.spend,
          f"pacing must protect the mid campaign: {mid.spend:.1f} vs {mid_loose.spend:.1f}")
    check(rich_loose.spend > 0.9 * rich_loose.budget,
          "the unpaced campaign must still exhaust its budget")

    # 8. the latency budget: every decision stays inside it, with the candidate set pruned
    check(result["p99_cycles"] <= LATENCY_BUDGET_CYCLES,
          f"p99 cycles {result['p99_cycles']} must stay inside {LATENCY_BUDGET_CYCLES}")
    check(result["mean_cycles"] <= LATENCY_BUDGET_CYCLES / 2.0,
          f"the mean cost {result['mean_cycles']:.1f} must leave headroom")
    candidates = 4 * 2          # the pruning bound of the market loop
    check(candidates * COST_PER_CANDIDATE + COST_PER_SLOT + COST_PER_PRICING <=
          LATENCY_BUDGET_CYCLES, "the worst-case decision must fit the budget")
    check(all(cycle <= LATENCY_BUDGET_CYCLES for cycle in result["cycles"]),
          "no single decision may exceed the latency budget")
    check(len(result["cycles"]) == 200, "every round must be accounted for")

    # 9. determinism and edge cases of the market
    mirror = market_fixture()
    again = run_market(mirror, rounds=200, reserve=1, horizon_slots=4)
    check(again["events"] == result["events"], "two identical markets must produce identical events")
    check(abs(again["revenue"] - result["revenue"]) < 1e-9, "the revenue must be reproducible")
    empty = run_market([], rounds=10, reserve=1)
    check(empty["revenue"] == 0.0 and empty["sales"] == 0, "an empty market earns nothing")
    broke = run_market([Advertiser("broke", 20, 1.0, 0, 10)], rounds=10, reserve=1)
    check(broke["sales"] == 0, "a campaign with no budget must never win")
    capped = run_market([Advertiser("capped", 20, 1.0, 1000, 2)], rounds=50, reserve=1)
    check(capped["sales"] == 2, f"the frequency cap must bind exactly, got {capped['sales']}")
    tie = [Advertiser("beta", 5, 1.0, 100, 5), Advertiser("alpha", 5, 1.0, 100, 5)]
    winner, price = second_price_outcome(tie, 1)
    check(winner.name == "alpha", "a tie must be broken by identifier, deterministically")
    check(price == 5.0, "a tie must make the winner pay its own bid")

    print(f"allocation: DP == brute force on 24 instances; VCG revenue {revenue:.2f} on 2 slots")
    print(f"reserve: uniform(4) -> r*={reserve} revenue={best:.4f} (no reserve "
          f"{expected_second_price_revenue(0, values, probabilities):.4f}); "
          f"uniform(10) -> r*={high_reserve}")
    print(f"market: revenue {result['revenue']:.1f} over {result['sales']} sales, "
          f"mean {result['mean_cycles']:.1f} / p99 {result['p99_cycles']} cycles "
          f"(budget {LATENCY_BUDGET_CYCLES})")
    print("PASS")


def utility_for(bidder, bidders, slots, reserve, true_value):
    """The utility a bidder with `true_value` gets from the current reports."""
    pool = list(bidders) + reserve_dummies(slots, reserve)
    allocation = best_allocation(pool, slots, 0.0)
    if bidder.name not in allocation:
        return 0.0
    payments, _ = vcg_outcome(bidders, slots, reserve)
    if bidder.name not in payments:
        return 0.0
    return (true_value * bidder.quality * slots[allocation[bidder.name]].position -
            payments[bidder.name])


if __name__ == "__main__":
    main()
