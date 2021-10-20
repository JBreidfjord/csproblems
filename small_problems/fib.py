import cProfile
import pstats
from functools import cache


def fib_gen(n: int):
    yield 0
    if n > 0:
        yield 1
    last = 0
    next = 1
    for _ in range(1, n):
        last, next = next, (last + next)
        yield next


@cache
def fib_recursive(n: int) -> int:
    if n < 2:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)


if __name__ == "__main__":
    n = 500_000
    with cProfile.Profile() as p:
        for i in fib_gen(n):
            ...
            # print(i)

    stats = pstats.Stats(p)
    stats.sort_stats(pstats.SortKey.TIME).print_stats()

    with cProfile.Profile() as p:
        for i in range(n + 1):
            fib_recursive(i)
            # print(fib_recursive(i))

    stats = pstats.Stats(p)
    stats.sort_stats(pstats.SortKey.TIME).print_stats()
