"""Bounded parallel work with ordered results; no ROS clients or robot actions."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor


def ordered_geometry_map(function, items, workers=1, stop=lambda: False):
    """Preserve candidate preference; completion order never picks the grasp.

    At most `workers` checks are in flight. Cancellation/deadlines stop adding
    work. All running checks finish before their captured scene can be replaced.
    Exceptions propagate so a failed check cannot turn into an accepted grasp.
    """
    workers = max(1, min(4, int(workers)))
    iterator = iter(items)
    if workers == 1:
        for item in iterator:
            if stop():
                return
            result = function(item)
            if stop():
                return
            yield result
        return
    pending = deque()
    exhausted = object()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='grasp_geometry') as pool:
        try:
            for _ in range(workers):
                if stop():
                    return
                item = next(iterator, exhausted)
                if item is exhausted:
                    break
                pending.append(pool.submit(function, item))
            while pending:
                if stop():
                    return
                result = pending.popleft().result()
                if stop():
                    return
                yield result
                if not stop():
                    item = next(iterator, exhausted)
                    if item is not exhausted:
                        pending.append(pool.submit(function, item))
        finally:
            for future in pending:
                future.cancel()
