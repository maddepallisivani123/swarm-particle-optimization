import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# PSO functions
from PSO import initialize_particles, update_velocity, update_position, plot_particles

# ACO functions
from ACO import initialize_pheromones, plot_graph, calculate_probabilities, select_next_node, \
    calculate_delta_pheromones, update_pheromones

def hybrid_optimizer(objective_function, graph, num_particles, num_dimensions, search_space,
                     num_ants, alpha, beta, evaporation_rate, num_iterations, cognitive_param, social_param, inertia_weight, show_animation=False):
    num_nodes = len(graph)
    pheromones = initialize_pheromones(num_nodes, initial_pheromone=1.0)
    
    # PSO initialization
    positions, velocities = initialize_particles(num_particles, num_dimensions, search_space)
    personal_best = positions.copy()

    global_best_index = np.argmin([objective_function(p) for p in positions])
    global_best = positions[global_best_index]

    if show_animation:
        fig, ax = plt.subplots()
        ax.set_xlim(search_space[0], search_space[1])
        ax.set_ylim(search_space[0], search_space[1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Hybrid PSO-ACO Optimization')

        scatter_pso = plot_particles(ax, positions)
        scatter_aco = plot_graph(graph, title="Ant Colony Optimization")

    def update(frame):
        nonlocal positions, velocities, personal_best, global_best, pheromones
        # PSO step
        for i in range(num_particles):
            if objective_function(positions[i]) < objective_function(personal_best[i]):
                personal_best[i] = positions[i]

        global_best_index = np.argmin([objective_function(p) for p in personal_best])
        global_best = personal_best[global_best_index]

        velocities = update_velocity(positions, velocities, personal_best, global_best, inertia_weight, cognitive_param, social_param)
        positions = update_position(positions, velocities)

        # ACO step
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

        # Visualization
        if show_animation:
            scatter_pso.set_offsets(positions)
            scatter_aco.set_paths([best_path])

    if show_animation:
        ani = FuncAnimation(fig, update, frames=num_iterations, repeat=False)
        plt.show()
    else:
        for _ in range(num_iterations):
            update(None)

    return global_best, objective_function(global_best)
def objective_function(x):
    # Replace this with your actual objective function
    return x[0]**2 + x[1]**2 
graph = [
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 1, 1, 1, 0]
]

def measure_performance(objective_function, *args, **kwargs):
    start_time = time.time()
    best_solution, best_value = hybrid_optimizer(objective_function, *args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return best_solution, best_value, execution_time

# Example Usage:
num_particles = 30
num_dimensions = 2
search_space = [-5, 5]
num_ants = 5
alpha = 1.0
beta = 2.0
evaporation_rate = 0.5
num_iterations = 20
cognitive_param = 1.5
social_param = 1.5
inertia_weight = 0.7

# Using measure_performance function
best_solution, best_value, execution_time = measure_performance(objective_function, graph, num_particles, num_dimensions, search_space,
                                            num_ants, alpha, beta, evaporation_rate, num_iterations, cognitive_param, social_param, inertia_weight, show_animation=True)

print("Best Solution:", best_solution)
print("Best Value:", best_value)
print("Execution Time:", execution_time, "seconds")