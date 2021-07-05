import cProfile
import pstats


def fib(n: int):
    yield 0
    if n > 0:
        yield 1
    last = 0
    next = 1
    for _ in range(1, n):
        last, next = next, (last + next)
        yield next


if __name__ == "__main__":
    with cProfile.Profile() as p:
        for i in fib(10):
            print(i)

    stats = pstats.Stats(p)
    stats.sort_stats(pstats.SortKey.CALLS).print_stats()
