"""Utility helpers used across the codebase.

Includes a tiny priority-lock helper and a simple chunk iterator for splitting text into fixed sizes.
"""

from asyncio import Future
from heapq import heappop, heappush
from typing import Iterator


class Tuple(tuple):
    """Small wrapper to make heap items comparable by priority (index 0)."""
    def __lt__(self, value: tuple) -> bool:
        return self[0] < value[0]


class PriorityLock(list[tuple[float, Future]]):
    """A lightweight priority-based semaphore.

    When `acquire` is called, if capacity exists (`n>0`) it consumes one slot
    and returns immediately. Otherwise the caller is enqueued and receives a
    Future that will be resolved when capacity is released.
    """
    def __init__(self, n=1):
        self.n = n

    def acquire(self, priority: float) -> None | Future:
        """Attempt to acquire a slot and return a Future if waiting is required."""
        future = Future()
        if self.n:
            self.n -= 1
        else:
            future = Future()
            heappush(self, Tuple((priority, future)))
            return future

    def release(self):
        """Release a slot and resolve the next waiter if any."""
        while self:
            _, future = heappop(self)
            if not future.done():
                return future.set_result(None)
        self.n += 1


def chunkit(s: str, n=4096) -> Iterator[str]:
    """Yield successive chunks of `s` with maximum length `n`."""
    while s:
        yield s[:n]
        s = s[n:]
