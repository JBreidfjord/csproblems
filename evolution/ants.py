from collections import namedtuple
from dataclasses import dataclass
from math import sqrt

import numpy as np

rng = np.random.default_rng()
Node = namedtuple("Node", ["x", "y"])


class Ant:
    def __init__(self, nodes: list[Node]):
        self.node_count = len(nodes)
        self.nodes = nodes

        self.visited: list[int] = []
        starting_node: int = rng.integers(0, self.node_count)
        self.visited.append(starting_node)

    def visit_node(self, node: int):
        self.visited.append(node)

    def visit_random_node(self):
        unvisited = list(set(range(self.node_count)) ^ set(self.visited))
        self.visited.append(rng.choice(unvisited))

    def get_visit_probabilities(
        self, pheromones: np.ndarray, distance_matrix: np.ndarray, alpha: int, beta: int,
    ) -> tuple[list, list]:
        """Calculates visit probabilities based on pheromones and the distance between nodes."""
        current_node = self.visited[-1]
        unvisited = set(range(self.node_count)) ^ set(self.visited)

        indices = []
        probabilities = []
        total_probability = 0

        for i in unvisited:
            indices.append(i)
            pheromones_on_path = pheromones[current_node, i] ** alpha
            heuristic_for_path = (1 / distance_matrix[current_node, i]) ** beta
            probability = pheromones_on_path * heuristic_for_path
            probabilities.append(probability)
            total_probability += probability

        probabilities = np.array(probabilities)
        probabilities /= total_probability

        return indices, probabilities.tolist()

    def roulette_wheel_selection(self, indices: list[int], probabilities: list[int]) -> int:
        slices = []
        total = 0
        for i in range(len(indices)):
            slice = [indices[i], total, total + probabilities[i]]
            slices.append(slice)
            total += probabilities[i]

        spin = rng.random()
        result = [slice for slice in slices if slice[1] < spin <= slice[2]]
        return result[0][0]

    def get_distance_travelled(self):
        total_distance = 0
        for i in range(1, len(self.visited)):
            node_a = self.nodes[self.visited[i - 1]]
            node_b = self.nodes[self.visited[i]]
            total_distance += distance(node_a, node_b)

        return total_distance

    def get_visited(self):
        return [self.nodes[i] for i in self.visited]

    def __repr__(self):
        return f"Ant({self.get_visited()})"


class Colony:
    def __init__(self, nodes: list[Node], ants_factor: float):
        """Generates an initial colony."""
        n_ants = round(len(nodes) * ants_factor)
        self.ants = [Ant(nodes) for _ in range(n_ants)]
        self.best_ant: Ant = rng.choice(self.ants)

    def __iter__(self):
        yield from self.ants

    def update_best_ant(self):
        for ant in self.ants:
            if ant.get_distance_travelled() < self.best_ant.get_distance_travelled():
                self.best_ant = ant


class ACO:
    def __init__(self, node_count: int, min: int, max: int):
        """Class containing functions to perform Ant Colony Optimization."""
        self.node_count = node_count
        self._generate_graph(min, max)
        self._init_pheromones()

    def _generate_graph(self, min: int, max: int):
        """Generates nodes and calculates a distance matrix for the set of nodes."""
        # Set prevents duplicate nodes
        nodes = set()
        while len(nodes) < self.node_count:
            x = rng.integers(min, max, endpoint=True)
            y = rng.integers(min, max, endpoint=True)
            nodes.add(Node(x, y))
        self.nodes = list(nodes)

        self.distance_matrix = np.zeros((self.node_count, self.node_count))
        for i, node_a in enumerate(self.nodes):
            for j, node_b in enumerate(self.nodes):
                self.distance_matrix[i, j] = distance(node_a, node_b)

    def _init_pheromones(self):
        """Initializes pheromone values as an array of ones."""
        self.pheromones = np.ones((self.node_count, self.node_count))

    def create_colony(self, ants_factor: float):
        """Initializes colony."""
        self.colony = Colony(self.nodes, ants_factor)

    def move_ants(self, random_factor: float, alpha: int, beta: int):
        """Move ants based on visit probabilities."""
        for ant in self.colony:
            if rng.random() < random_factor:
                ant.visit_random_node()
            else:
                indices, probabilities = ant.get_visit_probabilities(
                    self.pheromones, self.distance_matrix, alpha, beta
                )
                node = ant.roulette_wheel_selection(indices, probabilities)
                ant.visit_node(node)

    def update_pheromones(self, evaporation_rate: float):
        """Update pheromone trail values based on distance travelled."""
        for x in range(self.node_count):
            for y in range(self.node_count):
                self.pheromones[x, y] *= evaporation_rate
                for ant in self.colony:
                    self.pheromones[x, y] += 1 / ant.get_distance_travelled()
        return self.pheromones

    def solve(
        self,
        total_iterations: int,
        evaporation_rate: float = 0.5,
        ants_factor: float = 0.5,
        alpha: int = 1,
        beta: int = 1,
        random_factor: float = 0.3,
    ):
        """Runs ACO algorithm."""
        for _ in range(total_iterations):
            self.create_colony(ants_factor)
            for _ in range(self.node_count - 1):
                self.move_ants(random_factor, alpha, beta)
            self.update_pheromones(evaporation_rate)
            self.colony.update_best_ant()
        return self.colony.best_ant


@dataclass
class Config:
    node_count: int = 10
    min: int = 0
    max: int = 100
    total_iterations: int = 1000
    evaporation_rate: float = 0.5
    ants_factor: float = 0.5
    random_factor: float = 0.3
    alpha: int = 1
    beta: int = 1


def distance(node_a: Node, node_b: Node):
    """Calculates distance between points."""
    return sqrt((node_b.x - node_a.x) ** 2 + (node_b.y - node_a.y) ** 2)


if __name__ == "__main__":
    config = Config(
        node_count=10,
        max=100,
        total_iterations=500,
        evaporation_rate=0.5,
        ants_factor=0.5,
        random_factor=0.3,
        alpha=1,
        beta=1,
    )

    aco = ACO(config.node_count, config.min, config.max)
    solution = aco.solve(
        config.total_iterations,
        config.evaporation_rate,
        config.ants_factor,
        config.alpha,
        config.beta,
        config.random_factor,
    )
    print(aco.nodes)
    print(solution)
    print(solution.get_distance_travelled())
