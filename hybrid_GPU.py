import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Define PSO functions using TensorFlow
def initialize_particles_gpu(num_particles, num_dimensions, search_space):
    positions = tf.random.uniform((num_particles, num_dimensions), minval=search_space[0], maxval=search_space[1])
    velocities = tf.random.uniform((num_particles, num_dimensions))
    return positions, velocities

def update_velocity_gpu(positions, velocities, personal_best, global_best, inertia_weight, cognitive_param, social_param):
    inertia_term = inertia_weight * velocities
    cognitive_term = cognitive_param * tf.random.uniform(shape=positions.shape) * (personal_best - positions)
    social_term = social_param * tf.random.uniform(shape=positions.shape) * (global_best - positions)
    return inertia_term + cognitive_term + social_term

def update_position_gpu(positions, velocities):
    return positions + velocities

def pso_optimizer_gpu(objective_function, num_particles, num_dimensions, search_space, num_iterations, cognitive_param, social_param, inertia_weight, show_animation=False):
    positions, velocities = initialize_particles_gpu(num_particles, num_dimensions, search_space)
    personal_best = positions.numpy().copy()

    global_best_index = np.argmin([objective_function(p) for p in positions.numpy()])
    global_best = positions.numpy()[global_best_index]

    if show_animation:
        fig, ax = plt.subplots()
        ax.set_xlim(search_space[0], search_space[1])
        ax.set_ylim(search_space[0], search_space[1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Particle Swarm Optimization')

        scatter = ax.scatter(positions[:, 0].numpy(), positions[:, 1].numpy(), c='b', marker='o', alpha=0.5)

    start_time = time.time()
    def update(frame):
        nonlocal positions, velocities, personal_best, global_best
        for i in range(num_particles):
            if objective_function(positions.numpy()[i]) < objective_function(personal_best[i]):
                personal_best[i] = positions.numpy()[i]

        global_best_index = np.argmin([objective_function(p) for p in personal_best])
        global_best = personal_best[global_best_index]

        velocities = update_velocity_gpu(positions, velocities, personal_best, global_best, inertia_weight, cognitive_param, social_param)
        positions = update_position_gpu(positions, velocities)

        if show_animation:
            scatter.set_offsets(positions[:, :2].numpy())

    if show_animation:
        ani = FuncAnimation(fig, update, frames=num_iterations, repeat=False)
        plt.show()
    else:
        for _ in range(num_iterations):
            update(None)
    end_time = time.time()
    execution_time = end_time - start_time

    return global_best, objective_function(global_best), execution_time

# Usage on GPU
with tf.device('/GPU:0'):  # Choose the appropriate GPU device
    num_particles = 30
    num_dimensions = 2
    search_space = [-5, 5]
    num_iterations = 20
    cognitive_param = 1.5
    social_param = 1.5
    inertia_weight = 0.7

    best_solution_gpu, best_value_gpu, execution_time_gpu = pso_optimizer_gpu(objective_function, num_particles, num_dimensions, search_space,
                                                                              num_iterations, cognitive_param, social_param, inertia_weight, show_animation=True)

    print("Best Solution (GPU):", best_solution_gpu)
    print("Best Value (GPU):", best_value_gpu)
    print("Execution Time (GPU):", execution_time_gpu, "seconds")