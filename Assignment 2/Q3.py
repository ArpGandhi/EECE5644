import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def map_objective(pos,landmarks,measurements,sigma_i,sigma_x,sigma_y):
    x,y = pos
    likelihood_term = 0
    for i in range(len(landmarks)):
        xi,yi = landmarks[i]
        di = np.sqrt((x-xi)**2 + (y-yi)**2)
        likelihood_term+= (measurements[i]-di)**2/(sigma_i**2)
    prior_term = x**2/(sigma_x**2) + y**2/(sigma_y**2)
    return 0.5*(likelihood_term + prior_term)

def generate_landmarks(K):
    angles = np.linspace(0,2*np.pi,K,endpoint=False)
    landmarks = np.column_stack([np.cos(angles), np.sin(angles)])
    return landmarks

def generate_measurements(true_pos, landmarks, sigma_i):
    measurements = []
    for landmark in landmarks:
        while True:
            true_dist = np.linalg.norm(true_pos - landmark)
            noise = np.random.normal(0, sigma_i)
            measurement = true_dist + noise
            if measurement >= 0: 
                measurements.append(measurement)
                break
    return np.array(measurements)

np.random.seed(42)
true_position = np.array([0.3,0.4])
sigma_i = 0.3
sigma_x = sigma_y = 0.25

print(f"\nTrue vehicle position: {true_position}")
print(f"Noise \u03C3: {sigma_i}")
print(f"\u03C3x = \u03C3y: {sigma_x}")

fig, axes = plt.subplots(2,2,figsize=(16,14))
axes = axes.flatten()
K_values = [1,2,3,4]
contour_levels = np.linspace(0,20,15)

for idx, K in enumerate(K_values):
    print(f"K = {K} LANDMARKS")
    landmarks = generate_landmarks(K)
    measurements = generate_measurements(true_position, landmarks, sigma_i)
    
    print(f"\nLandmarks (on unit circle):")
    for i, lm in enumerate(landmarks):
        print(f"Landmark {i+1}: [{lm[0]:6.3f},{lm[1]:6.3f}]")

    for i, m in enumerate(measurements):
        true_dist = np.linalg.norm(true_position - landmarks[i])
        print(f"r{i+1} = {m:.4f} (true distance = {true_dist:.4f}, error = {m-true_dist:+.4f})")
    
    x_range = np.linspace(-2,2,200)
    y_range = np.linspace(-2,2,200)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    Z_grid = np.zeros_like(X_grid)

    for i in range(len(x_range)):
        for j in range(len(y_range)):
            pos = np.array([X_grid[j, i], Y_grid[j, i]])
            Z_grid[j, i] = map_objective(pos, landmarks, measurements, 
                                        sigma_i, sigma_x, sigma_y)
    
    initial_guess = np.array([0.0,0.0])
    result = minimize(map_objective, initial_guess, 
                     args=(landmarks, measurements, sigma_i, sigma_x, sigma_y),
                     method='BFGS')
    map_estimate = result.x
    
    error = np.linalg.norm(map_estimate - true_position)
    print(f"\nMAP estimate: [{map_estimate[0]:.4f}, {map_estimate[1]:.4f}]")
    print(f"True position: [{true_position[0]:.4f}, {true_position[1]:.4f}]")
    print(f"Localization error: {error:.4f}")
    print(f"Optimization success: {result.success}")
    
    ax = axes[idx]
    contour = ax.contour(X_grid, Y_grid, Z_grid, levels=contour_levels, 
                        cmap='viridis', linewidths=1.5)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    ax.plot(true_position[0], true_position[1], 'r+', markersize=20, 
           markeredgewidth=3, label='True Position', zorder=5)
    ax.plot(landmarks[:, 0], landmarks[:, 1], 'bo', markersize=12, 
           markeredgewidth=2, markerfacecolor='none', label='Landmarks', zorder=4)
    for i, lm in enumerate(landmarks):
        ax.annotate(f'L{i+1}', xy=(lm[0], lm[1]), 
                   xytext=(lm[0]*1.15, lm[1]*1.15),
                   fontsize=9, ha='center')
    ax.plot(map_estimate[0], map_estimate[1], 'g*', markersize=20,
           markeredgewidth=2, label='MAP Estimate', zorder=5)
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='gray', 
                       linestyle='--', linewidth=1.5, label='Unit Circle')
    ax.add_patch(circle)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$y$', fontsize=12)
    ax.set_title(f'K = {K} Landmarks (Error = {error:.4f})', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('Assignment 2/outputs/q3_map_localization.png', dpi=150, bbox_inches='tight')

n_trials = 20
errors = {K: [] for K in K_values}

np.random.seed(100)
for K in K_values:
    landmarks = generate_landmarks(K)
    for trial in range(n_trials):
        measurements = generate_measurements(true_position, landmarks, sigma_i)
        result = minimize(map_objective, np.array([0.0, 0.0]),
                         args=(landmarks, measurements, sigma_i, sigma_x, sigma_y),
                         method='BFGS')
        map_estimate = result.x
        error = np.linalg.norm(map_estimate - true_position)
        errors[K].append(error)

print(f"{'K':>3} | {'Mean Error':>12} | {'Std Dev':>10} | {'Min':>8} | {'Max':>8}")
for K in K_values:
    mean_error = np.mean(errors[K])
    std_error = np.std(errors[K])
    min_error = np.min(errors[K])
    max_error = np.max(errors[K])
    print(f"{K:3d} | {mean_error:12.4f} | {std_error:10.4f} | " +
          f"{min_error:8.4f} | {max_error:8.4f}")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
positions = range(1, len(K_values) + 1)
bp = ax.boxplot([errors[K] for K in K_values], positions=positions,
                widths=0.6, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

ax.set_xlabel('Number of Landmarks (K)',fontsize=13)
ax.set_ylabel('Localization Error',fontsize=13)
ax.set_title('MAP Estimate Accuracy vs Number of Landmarks',fontsize=14,fontweight='bold')
ax.set_xticks(positions)
ax.set_xticklabels(K_values)
ax.grid(True,alpha=0.3,axis='y')
plt.tight_layout()
plt.savefig('Assignment 2/outputs/q3_error_analysis.png',dpi=150,bbox_inches='tight')