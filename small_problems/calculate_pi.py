def calculate_pi(n_terms: int) -> float:
    """
    Uses Leibniz's formula to calculate pi
    using the specified number of terms for the series
    """
    num = 4.0
    den = 1.0
    mult = 1.0
    pi = 0.0
    for _ in range(n_terms):
        pi += mult * (num / den)
        mult *= -1
        den += 2.0
    return pi


if __name__ == "__main__":
    print(calculate_pi(10_000_000))
