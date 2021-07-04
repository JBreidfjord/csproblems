import cProfile
import pstats


def fib(n: int) -> int:
    if n == 0:
        return n
    last: int = 0
    next: int = 1
    for _ in range(1, n):
        last, next = next, (last + next)
    return next


if __name__ == "__main__":
    with cProfile.Profile() as p:
        for i in range(11):
            print(fib(i))

    stats = pstats.Stats(p)
    stats.sort_stats(pstats.SortKey.CALLS).print_stats()
