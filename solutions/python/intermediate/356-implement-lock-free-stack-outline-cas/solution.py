"""Task 356 - Implement lock-free stack outline (CAS). (Tier: Intermediate, CTCI Ch 15)

Michael-Scott style lock-free stack: push/pop are CAS retry loops on the single head
pointer. CPython needs a stand-in for the hardware instruction, so
compare_and_swap_head() performs compare+swap under a tiny lock - only that pair is
atomic, never the whole push/pop. pop() returns the documented sentinel None on an empty
stack (None is also the initial head, so "empty" and "no value" are the same distinguished
state; pushed values are ints and can never be None). The self-test checks LIFO order
single-threaded, then that 4 concurrent pusher threads add all 1000 nodes (exact count,
every value once) and 4 concurrent poppers drain them with no loss or duplication. The
ABA hazard of pointer CAS is documented but cannot occur here: nodes are never recycled
during a run, and any lost CAS is retried with a freshly read head.
"""
import threading

THREADS = 4
PUSHES_PER_THREAD = 250


class Node:
    __slots__ = ("value", "next")

    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


class LockFreeStack:
    def __init__(self):
        self._head = None
        self._cas_lock = threading.Lock()          # emulates the hardware CAS instruction
        self.cas_attempts = 0
        self.cas_failures = 0

    def compare_and_swap_head(self, expected, new_head):
        with self._cas_lock:
            self.cas_attempts += 1
            if self._head is expected:
                self._head = new_head
                return True
            self.cas_failures += 1
            return False

    def push(self, value):
        retries = 0
        while True:
            current = self._head
            node = Node(value, current)
            if self.compare_and_swap_head(current, node):
                return retries
            retries += 1

    def pop(self):
        """Pops the top value; returns None (the documented sentinel) when empty."""
        while True:
            current = self._head
            if current is None:
                return None
            if self.compare_and_swap_head(current, current.next):
                return current.value

    def is_empty(self):
        return self._head is None

    def snapshot_values(self):
        values = []
        node = self._head
        while node is not None:
            values.append(node.value)
            node = node.next
        return values


def main():
    # Single-threaded script: LIFO order and the empty sentinel.
    stack = LockFreeStack()
    assert stack.pop() is None, "pop on an empty stack returns the sentinel"
    for value in range(1, 6):
        stack.push(value)
    assert [stack.pop() for _ in range(5)] == [5, 4, 3, 2, 1], "LIFO order"
    assert stack.pop() is None, "empty again after draining"
    assert stack.cas_failures == 0, "single-threaded script never loses a CAS"

    # Concurrent pushes: no node is lost.
    stack = LockFreeStack()
    total = THREADS * PUSHES_PER_THREAD

    def pusher(thread_index):
        for offset in range(PUSHES_PER_THREAD):
            stack.push(thread_index * PUSHES_PER_THREAD + offset)

    pushers = [threading.Thread(target=pusher, args=(index,), daemon=True)
               for index in range(THREADS)]
    for thread in pushers:
        thread.start()
    for thread in pushers:
        thread.join(10)
        assert not thread.is_alive(), "pusher did not finish in time"

    values = stack.snapshot_values()
    assert len(values) == total, "no node lost under concurrent pushes"
    assert sorted(values) == list(range(total)), "every pushed value present exactly once"

    # Concurrent pops: no node is lost or duplicated.
    popped_per_thread = [[] for _ in range(THREADS)]

    def popper(thread_index):
        while True:
            value = stack.pop()
            if value is None:
                break
            popped_per_thread[thread_index].append(value)

    poppers = [threading.Thread(target=popper, args=(index,), daemon=True)
               for index in range(THREADS)]
    for thread in poppers:
        thread.start()
    for thread in poppers:
        thread.join(10)
        assert not thread.is_alive(), "popper did not finish in time"

    all_popped = [value for chunk in popped_per_thread for value in chunk]
    assert len(all_popped) == total, "no node lost under concurrent pops"
    assert sorted(all_popped) == list(range(total)), "every pushed value popped exactly once"
    assert stack.pop() is None, "stack is empty again"
    assert stack.is_empty(), "empty state after the concurrent phase"

    # Exact CAS accounting: every successful push/pop is one attempt, every retry one
    # failure. The two threads that stopped with the sentinel made no CAS attempt at all.
    assert stack.cas_failures >= 0, "CAS failures are non-negative (they may be 0)"
    assert stack.cas_attempts == 2 * total + stack.cas_failures, \
        "attempts == successful operations + failures"


if __name__ == "__main__":
    main()
    print("PASS")
