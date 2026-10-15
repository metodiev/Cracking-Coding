"""Task 355 - Implement epoch-based reclamation idea for concurrent data structure safe delete. (Tier: Intermediate, CTCI Ch 15)

Epoch-based reclamation (EBR) sketch: a global epoch counter, per-thread pinning
(pin() records the current epoch, unpin() clears it) and three retirement buckets keyed
by epoch % EPOCHS. Documented free rule: an object retired in epoch r may be freed only
when no thread is pinned in an epoch <= r - i.e. r <= min_pinned_epoch - 1, and with no
pinned threads every retired object is safe. collect() first tries to advance the epoch
(an advance requires every pinned thread to sit in the current epoch), then drains every
bucket entry the rule allows and keeps the rest on the deferred list. Real EBR frees the
bucket from two epochs back; this conservative rule is strictly safer. The self-test
simulates deterministic interleavings (scripted phases plus an LCG-driven script) and
asserts that nothing is ever freed while a thread pins its epoch, that deferred objects
stay put while old pins exist, and that after quiescence the deferred list drains fully.
"""
EPOCHS = 3
SCRIPT_STEPS = 120


class EbrManager:
    def __init__(self, num_threads):
        self.global_epoch = 0
        self.pin_epochs = [None] * num_threads
        self.buckets = [[] for _ in range(EPOCHS)]     # entries: (object_id, retirement_epoch)
        self.free_log = []                             # entries: (object_id, retirement_epoch, epoch_at_free)
        self.violations = 0
        self.deferred_entries = 0
        self.advances = 0

    def pin(self, thread):
        assert self.pin_epochs[thread] is None, "thread {} already pinned".format(thread)
        self.pin_epochs[thread] = self.global_epoch

    def unpin(self, thread):
        assert self.pin_epochs[thread] is not None, "thread {} is not pinned".format(thread)
        self.pin_epochs[thread] = None

    def is_pinned(self, thread):
        return self.pin_epochs[thread] is not None

    def retire(self, thread, object_id):
        assert self.pin_epochs[thread] is not None, "retire requires a pinned thread"
        self.buckets[self.global_epoch % EPOCHS].append((object_id, self.global_epoch))

    def try_advance(self):
        """Advance only when every pinned thread sits in the current epoch."""
        if all(epoch is None or epoch == self.global_epoch for epoch in self.pin_epochs):
            self.global_epoch += 1
            self.advances += 1
            return True
        return False

    def min_pinned_epoch(self):
        pinned = [epoch for epoch in self.pin_epochs if epoch is not None]
        return min(pinned) if pinned else None

    def collect(self):
        """Drain every safe deferred object; returns the objects freed by this call."""
        self.try_advance()
        min_pinned = self.min_pinned_epoch()
        safe_epoch = None if min_pinned is None else min_pinned - 1
        freed = []
        for bucket in self.buckets:
            kept = []
            for object_id, retired_epoch in bucket:
                if safe_epoch is None or retired_epoch <= safe_epoch:
                    self._record_free(object_id, retired_epoch)
                    freed.append(object_id)
                else:
                    kept.append((object_id, retired_epoch))
            self.deferred_entries += len(kept)
            bucket[:] = kept
        return freed

    def _record_free(self, object_id, retired_epoch):
        # The safety invariant, checked at the moment of every free:
        # no thread may be pinned in an epoch <= the object's retirement epoch.
        for epoch in self.pin_epochs:
            if epoch is not None and epoch <= retired_epoch:
                self.violations += 1
        self.free_log.append((object_id, retired_epoch, self.global_epoch))


def main():
    # Phase 1: retirement in epoch 0; the pin that predates an advance blocks the free.
    mgr = EbrManager(3)
    mgr.pin(0)
    mgr.retire(0, "A")                            # A retired in epoch 0
    assert mgr.collect() == [], "A must not be freed while thread 0 pins epoch 0"

    mgr.pin(1)                                    # thread 1 pins the current epoch 1
    assert mgr.collect() == [], "old pin (epoch 0) keeps A deferred"
    assert mgr.violations == 0, "no free while a thread pinned an old epoch"

    mgr.unpin(0)
    assert mgr.collect() == ["A"], "A frees once no thread pins an epoch <= 0"
    assert mgr.violations == 0, "freeing A respected the pin rule"

    # Phase 2: a thread pinned in the retirement epoch itself blocks the free.
    mgr.pin(0)                                    # thread 0 pins epoch 2
    mgr.retire(0, "B")                            # B retired in epoch 2
    assert mgr.collect() == [], "B deferred while thread 1 still pins epoch 1"
    mgr.unpin(1)
    assert mgr.collect() == [], "B still deferred: thread 0 pins its retirement epoch 2"
    mgr.unpin(0)
    assert mgr.collect() == ["B"], "B frees at quiescence"
    assert mgr.violations == 0, "freeing B respected the pin rule"
    assert all(retired_epoch <= epoch_at_free for _, retired_epoch, epoch_at_free in mgr.free_log), \
        "free log records a retirement epoch not in the future"

    # Phase 3: LCG-driven interleaving of pin / unpin / retire / collect across 3 threads.
    mgr = EbrManager(3)
    seed = 4242
    next_object = 100
    retired = []
    for step in range(SCRIPT_STEPS):
        seed = (seed * 75 + 74) % 65537
        thread = seed % 3
        seed = (seed * 75 + 74) % 65537
        action = seed % 4
        if action == 0 and not mgr.is_pinned(thread):
            mgr.pin(thread)
        elif action == 1 and mgr.is_pinned(thread):
            mgr.unpin(thread)
        elif action >= 2 and mgr.is_pinned(thread):
            mgr.retire(thread, next_object)
            retired.append(next_object)
            next_object += 1
        mgr.collect()
        assert mgr.violations == 0, "free-of-pinned violation at step {}".format(step)

    assert len(retired) >= 10, "the interleaving retired a useful number of objects"
    assert mgr.deferred_entries > 0, "old pins actually deferred some objects"

    # Quiescence: unpin everything, then the deferred list must drain completely.
    for thread in range(3):
        if mgr.is_pinned(thread):
            mgr.unpin(thread)
    mgr.collect()
    assert all(bucket == [] for bucket in mgr.buckets), "deferred lists drained"
    freed_ids = sorted(object_id for object_id, _, _ in mgr.free_log)
    assert freed_ids == sorted(retired), "every retired object freed exactly once"
    assert len(set(freed_ids)) == len(freed_ids), "no double frees"
    assert mgr.violations == 0, "the whole run respected the pin rule"
    assert mgr.advances >= 1, "epochs actually advanced"


if __name__ == "__main__":
    main()
    print("PASS")
