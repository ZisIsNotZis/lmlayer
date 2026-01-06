from asyncio import Future
from heapq import heappop, heappush
from typing import Iterator


class PriorityLock(list[tuple[float, Future]]):
    def __init__(self, n=1):
        self.n = n

    def acquire(self, priority: float) -> None | Future:
        future = Future()
        if self.n:
            self.n -= 1
        else:
            future = Future()
            heappush(self, (priority, future))
            return future

    def release(self):
        while self:
            _, future = heappop(self)
            if not future.done():
                return future.set_result(None)
        self.n += 1


def chunkit(s: str, n=4096) -> Iterator[str]:
    while s:
        yield s[:n]
        s = s[n:]
