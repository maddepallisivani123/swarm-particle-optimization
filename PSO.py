import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

def objective_function(x):
    return sum(x**2)

def initialize_particles(num_particles, num_dimensions, search_space):
    positions = np.random.uniform(low=search_space[0], high=search_space[1], size=(num_particles, num_dimensions))
    velocities = np.random.rand(num_particles, num_dimensions)
    return positions, velocities

def update_velocity(positions, velocities, personal_best, global_best, inertia_weight, cognitive_param, social_param):
    inertia_term = inertia_weight * velocities
    cognitive_term = cognitive_param * np.random.rand() * (personal_best - positions)
    social_term = social_param * np.random.rand() * (global_best - positions)
    return inertia_term + cognitive_term + social_term

def update_position(positions, velocities):
    return positions + velocities

def plot_particles(ax, positions):
    return ax.scatter(positions[:, 0], positions[:, 1], c='b', marker='o', alpha=0.5)

def pso_optimizer(objective_function, num_particles, num_dimensions, search_space, num_iterations, cognitive_param, social_param, inertia_weight, show_animation=False):
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
        ax.set_title('Particle Swarm Optimization')

        scatter = plot_particles(ax, positions)

    def update(frame):
        nonlocal positions, velocities, personal_best, global_best
        for i in range(num_particles):
            if objective_function(positions[i]) < objective_function(personal_best[i]):
                personal_best[i] = positions[i]

        global_best_index = np.argmin([objective_function(p) for p in personal_best])
        global_best = personal_best[global_best_index]

        velocities = update_velocity(positions, velocities, personal_best, global_best, inertia_weight, cognitive_param, social_param)
        positions = update_position(positions, velocities)

        if show_animation:
            scatter.set_offsets(positions)

    if show_animation:
        ani = FuncAnimation(fig, update, frames=num_iterations, repeat=False)
        plt.show()
    else:
        for _ in range(num_iterations):
            update(None)

    return global_best, objective_function(global_best)
def measure_pso_performance(objective_function, *args, **kwargs):
    start_time = time.time()
    best_solution, best_value = pso_optimizer(objective_function, *args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return best_solution, best_value, execution_time

# Example Usage for PSO:
num_particles = 30
num_dimensions = 2
search_space = [-5, 5]
num_iterations = 100  # Increased iterations
cognitive_param = 1.5
social_param = 1.5
inertia_weight = 0.7

best_solution_pso, best_value_pso, execution_time_pso = measure_pso_performance(objective_function, num_particles, num_dimensions, search_space,
                                                                                   num_iterations, cognitive_param, social_param, inertia_weight, show_animation=True)

print("Best Solution (PSO):", best_solution_pso)
print("Best Value (PSO):", best_value_pso)
print("Execution Time (PSO):", execution_time_pso, "seconds")