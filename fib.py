import cProfile
import pstats

memo: dict[int, int] = {0: 0, 1: 1}  # base cases already stored

# could alternatively use @lru_cache decorator
def fib(n: int) -> int:
    """Utilizes memoization to reduce number of recursive function calls"""
    if n not in memo:
        memo[n] = fib(n - 1) + fib(n - 2)
    return memo[n]


if __name__ == "__main__":
    with cProfile.Profile() as p:
        for i in range(11):
            print(fib(i))

    stats = pstats.Stats(p)
    stats.sort_stats(pstats.SortKey.CALLS).print_stats()
