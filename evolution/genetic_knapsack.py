from numpy import random

from knapsack_items import Item, item_list


def generate_initital_population(population_size: int, individual_size: int, mutation_rate: float):
    # Population could be randomly initialized individuals instead
    population = [mutate_individual([0] * individual_size, mutation_rate)] * population_size
    return population


def calculate_individual_fitness(
    individual: list[int], knapsack_items: list[Item], knapsack_capacity: int
):
    weight = 0
    value = 0
    for i, bit in enumerate(individual):
        if bit:
            weight += knapsack_items[i].weight
            value += knapsack_items[i].value

    return value if weight <= knapsack_capacity else 0


def select_population(
    population: list[list[int]],
    number_of_selections: int,
    knapsack_items: list[Item],
    knapsack_capacity: int,
) -> list[list[int]]:
    population_fitness = [
        calculate_individual_fitness(i, knapsack_items, knapsack_capacity) for i in population
    ]
    total_fitness = sum(population_fitness)
    selection_probabilities = [fitness / total_fitness for fitness in population_fitness]

    return rng.choice(population, number_of_selections, p=selection_probabilities)


def uniform_crossover(parent_a: list[int], parent_b: list[int]):
    mask = rng.integers(2, size=len(parent_a)).tolist()
    children = []

    child_1 = [a if mask[i] else b for i, (a, b) in enumerate(zip(parent_a, parent_b))]
    children.append(child_1)
    child_2 = [b if mask[i] else a for i, (a, b) in enumerate(zip(parent_a, parent_b))]
    children.append(child_2)

    return children


def mutate_individual(individual: list[int], mutation_rate: float):
    mutations = round(len(individual) * mutation_rate)
    mutation_indices = rng.integers(len(individual), size=mutations)
    for mutation_index in mutation_indices:
        individual[mutation_index] = 0 if individual[mutation_index] else 1

    return individual


def generate_new_population(
    current_population: list[list[int]],
    survival_rate: float,
    mutation_rate: float,
    knapsack_items: list[Item],
    knapsack_capacity: int,
):
    current_population.sort(
        key=lambda i: calculate_individual_fitness(i, knapsack_items, knapsack_capacity)
    )
    population_size = len(current_population)

    number_of_children = int(population_size - (population_size * survival_rate))
    reproducers = select_population(
        current_population, number_of_children, knapsack_items, knapsack_capacity
    )
    group_a = reproducers[: number_of_children // 2]
    group_b = reproducers[number_of_children // 2 :]
    children = []
    for a, b in zip(group_a, group_b):
        children.extend(uniform_crossover(a, b))

    children = [mutate_individual(child, mutation_rate) for child in children]

    new_population = current_population[number_of_children:] + children
    return new_population


def run_ga(
    population_size: int,
    survival_rate: float,
    mutation_rate: float,
    number_of_generations: int,
    knapsack_items: list,
    knapsack_capacity: int,
):
    population = generate_initital_population(population_size, len(knapsack_items), mutation_rate)
    global_best_fitness = max(
        [calculate_individual_fitness(i, knapsack_items, knapsack_capacity) for i in population]
    )
    print(f"{global_best_fitness = }")
    for i in range(number_of_generations):
        population = generate_new_population(
            population, survival_rate, mutation_rate, knapsack_items, knapsack_capacity
        )

        best_fitness = max(
            [calculate_individual_fitness(i, knapsack_items, knapsack_capacity) for i in population]
        )

        if best_fitness > global_best_fitness:
            global_best_fitness = best_fitness
            print(f"Generation {i}: Best fitness {best_fitness}")

    solution = max(
        population, key=lambda i: calculate_individual_fitness(i, knapsack_items, knapsack_capacity)
    )
    return [knapsack_items[idx].name for idx, i in enumerate(solution) if i]


if __name__ == "__main__":
    population_size = 16
    survival_rate = 0.15
    mutation_rate = 0.05
    number_of_generations = 10000
    knapsack_items = item_list
    knapsack_capacity = 6_404_180

    rng = random.default_rng()

    result = run_ga(
        population_size,
        survival_rate,
        mutation_rate,
        number_of_generations,
        knapsack_items,
        knapsack_capacity,
    )
    print(result)
