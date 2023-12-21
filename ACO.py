import numpy as np
import matplotlib.pyplot as plt
import time

def initialize_pheromones(num_nodes, initial_pheromone):
    return np.full((num_nodes, num_nodes), initial_pheromone)

def update_pheromones(pheromones, delta_pheromones, evaporation_rate):
    return (1 - evaporation_rate) * pheromones + delta_pheromones

def calculate_probabilities(graph, pheromones, current_node, visited_nodes, alpha, beta):
    unvisited_nodes = [node for node in range(len(graph)) if node not in visited_nodes]
    probabilities = []

    for node in unvisited_nodes:
        pheromone = pheromones[current_node][node]
        visibility = 1 / graph[current_node][node]
        probability = (pheromone * alpha) * (visibility * beta)
        probabilities.append(probability)

    probabilities /= np.sum(probabilities)
    return probabilities

def select_next_node(probabilities):
    return np.random.choice(len(probabilities), p=probabilities)

def calculate_delta_pheromones(ant_paths, graph):
    delta_pheromones = np.zeros_like(graph, dtype=float)

    for path in ant_paths:
        for i in range(len(path) - 1):
            current_node, next_node = path[i], path[i + 1]
            delta_pheromones[current_node][next_node] += 1 / calculate_path_cost(path, graph)

    return delta_pheromones

def calculate_path_cost(path, graph):
    cost = 0
    for i in range(len(path) - 1):
        cost += graph[path[i]][path[i + 1]]
    return cost

def plot_graph(graph, ant_paths=None, title="Ant Colony Optimization"):
    plt.figure(figsize=(8, 8))
    plt.imshow(graph, cmap='viridis', interpolation='nearest')
    plt.title(title)

    for i in range(len(graph)):
        plt.text(i, i, str(i), ha='center', va='center', color='white', fontweight='bold')

    if ant_paths:
        for path in ant_paths:
            path_edges = list(zip(path[:-1], path[1:]))
            for edge in path_edges:
                plt.plot([edge[1], edge[0]], [edge[0], edge[1]], 'w-', lw=2)

                # Annotate edges with cost
                plt.text((edge[1] + edge[0]) / 2, (edge[0] + edge[1]) / 2,
                         f'{graph[edge[0]][edge[1]]:.2f}', color='white', ha='center', va='center', fontweight='bold')

    plt.colorbar(label='Pheromone Level')
    plt.xlabel('Node')
    plt.ylabel('Node')
    plt.show()

def aco_optimizer(graph, num_ants, alpha, beta, evaporation_rate, num_iterations):
    num_nodes = len(graph)
    pheromones = initialize_pheromones(num_nodes, initial_pheromone=1.0)

    best_path = None
    best_path_cost = float('inf')

    for iteration in range(num_iterations):
        ant_paths = []

        for ant in range(num_ants):
            current_node = np.random.randint(num_nodes)
            ant_path = [current_node]

            while len(ant_path) < num_nodes:
                probabilities = calculate_probabilities(graph, pheromones, current_node, ant_path, alpha, beta)
                next_node = select_next_node(probabilities)
                ant_path.append(next_node)
                current_node = next_node

            ant_paths.append(ant_path)

        delta_pheromones = calculate_delta_pheromones(ant_paths, graph)
        pheromones = update_pheromones(pheromones, delta_pheromones, evaporation_rate)

        # Find the best path among ant paths
        current_best_path = min(ant_paths, key=lambda path: calculate_path_cost(path, graph))
        current_best_path_cost = calculate_path_cost(current_best_path, graph)

        if current_best_path_cost < best_path_cost:
            best_path = current_best_path
            best_path_cost = current_best_path_cost

    plot_graph(graph, ant_paths=[best_path], title="Ant Colony Optimization - Best Path")
    return best_path, best_path_cost
def measure_aco_performance(graph, *args, **kwargs):
    start_time = time.time()
    best_path, best_path_cost = aco_optimizer(graph, *args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return best_path, best_path_cost, execution_time

# Example Usage for ACO:
num_nodes = 5
graph_aco = np.random.rand(num_nodes, num_nodes)  # Replace with your graph or distance matrix
num_ants_aco = 5
alpha_aco = 1.0
beta_aco = 2.0
evaporation_rate_aco = 0.5
num_iterations_aco = 50

best_path_aco, best_path_cost_aco, execution_time_aco = measure_aco_performance(graph_aco, num_ants_aco, alpha_aco, beta_aco,
                                                                                 evaporation_rate_aco, num_iterations_aco)

print("Best Path (ACO):", best_path_aco)
print("Best Path Cost (ACO):", best_path_cost_aco)
print("Execution Time (ACO):", execution_time_aco, "seconds")