import cProfile
import pstats


def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


if __name__ == "__main__":
    with cProfile.Profile() as p:
        for i in range(11):
            print(fib(i))

    stats = pstats.Stats(p)
    stats.sort_stats(pstats.SortKey.CALLS).print_stats()
