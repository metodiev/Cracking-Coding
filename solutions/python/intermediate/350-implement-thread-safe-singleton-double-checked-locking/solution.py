"""Task 350 - Thread-safe singleton via double-checked locking. (Tier: Intermediate, CTCI Ch 15)

Double-checked locking: an unlocked fast-path read, then the lock, then a
re-check.  The re-check inside the lock is what makes DCL correct; CPython needs
no volatile keyword because the GIL plus the lock's release/acquire pairing
publish the fully constructed object (Java and C# mark the field volatile - see
those files, and JavaScript uses the module cache instead).  The test releases 32
threads through one gate, asserts they all receive the same object and that the
constructor ran exactly once.
"""
import threading


class Singleton:
    _instance = None
    _lock = threading.Lock()
    _instantiations = 0

    def __init__(self):
        Singleton._instantiations += 1

    @classmethod
    def get_instance(cls):
        if cls._instance is None:                    # fast path, no lock
            with cls._lock:
                if cls._instance is None:            # re-check under the lock
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def instantiation_count(cls):
        return cls._instantiations


def main():
    assert Singleton.instantiation_count() == 0      # lazy: nothing constructed yet

    caller_count = 32
    gate = threading.Event()
    results = [None] * caller_count

    def caller(index):
        gate.wait(15)
        results[index] = Singleton.get_instance()

    threads = [threading.Thread(target=caller, args=(index,), daemon=True)
               for index in range(caller_count)]
    for thread in threads:
        thread.start()
    gate.set()                                       # release all callers at once
    for thread in threads:
        thread.join(15)
        assert not thread.is_alive(), "caller did not terminate"

    first = results[0]
    assert first is not None
    assert all(result is first for result in results), "every caller got the same instance"
    assert Singleton.instantiation_count() == 1, "the constructor ran exactly once"
    assert Singleton.get_instance() is first, "later callers reuse the instance"


if __name__ == "__main__":
    main()
    print("PASS")
