import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from matplotlib.patches import Ellipse
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

true_weights = np.array([0.2,0.3,0.25,0.25])
true_means = [
    np.array([0.0,0.0]),    
    np.array([1.5,1.5]),      
    np.array([6.0,0.0]),  
    np.array([0.0,6.0])       
]
true_covs = [
    np.array([[1.5,0.5], [0.5,1.5]]),
    np.array([[1.2,-0.3], [-0.3,1.2]]),
    np.array([[0.8,0.2], [0.2,0.8]]),
    np.array([[1.0,-0.4], [-0.4,1.0]])
]

true_n_components = 4
dist_01 = np.linalg.norm(true_means[0]-true_means[1])
avg_eigenval_0 = np.mean(np.linalg.eigvals(true_covs[0]))
avg_eigenval_1 = np.mean(np.linalg.eigvals(true_covs[1]))
overlap_ratio = dist_01/(avg_eigenval_0+avg_eigenval_1)

def generate_gmm_data(n_samples, weights, means, covs):
    n_components = len(weights)
    n_dim = len(means[0])
    X = np.zeros((n_samples, n_dim))
    component_labels = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        component = np.random.choice(n_components, p=weights)
        component_labels[i] = component
        X[i] = np.random.multivariate_normal(means[component], covs[component])
    return X, component_labels

def cross_validate_gmm(X, n_components, n_folds=10, n_init=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=None)
    log_likelihoods = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                             max_iter=200, n_init=n_init, random_state=None)
        try:
            gmm.fit(X_train)
            log_likelihood = gmm.score(X_val)*len(X_val)
            log_likelihoods.append(log_likelihood)
        except:
            log_likelihoods.append(-np.inf)
    return np.mean(log_likelihoods)

def select_best_gmm_order(X, candidate_orders, n_folds=10):
    cv_scores = {}
    for n_comp in candidate_orders:
        avg_ll = cross_validate_gmm(X, n_comp, n_folds=n_folds)
        cv_scores[n_comp] = avg_ll
    best_order = max(cv_scores, key=cv_scores.get)
    return best_order, cv_scores

dataset_sizes = [10,100,1000]
candidate_orders = list(range(1, 11)) 
n_experiments = 100 
n_folds = 10

selection_counts = {
    size: {order: 0 for order in candidate_orders}
    for size in dataset_sizes
}

for exp in range(n_experiments):
    for size in dataset_sizes:
        X, _ = generate_gmm_data(size, true_weights, true_means, true_covs)
        best_order, cv_scores = select_best_gmm_order(X, candidate_orders, n_folds=n_folds)
        selection_counts[size][best_order] += 1

selection_rates = {
    size: {order: count/n_experiments for order, count in counts.items()}
    for size, counts in selection_counts.items()
}

for size in dataset_sizes:
    print(f"\nDataset Size: {size} samples")
    print(f"{'GMM Order':<12} {'Selection Count':<18} {'Selection Rate (%)':<20}")
    for order in candidate_orders:
        count = selection_counts[size][order]
        rate = selection_rates[size][order]
        marker = " <-- TRUE" if order == true_n_components else ""
        print(f"{order:<12} {count:<18} {rate*100:<20.2f}{marker}")
    most_selected = max(selection_counts[size], key=selection_counts[size].get)
    most_selected_rate = selection_rates[size][most_selected]
    print(f"Most frequently selected: {most_selected} components ({most_selected_rate*100:.1f}%)")
    print(f"True model selected: {selection_counts[size][true_n_components]} times ({selection_rates[size][true_n_components]*100:.1f}%)")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, size in enumerate(dataset_sizes):
    ax = axes[0, idx] if idx < 2 else axes[1, 0]
    orders = list(candidate_orders)
    rates = [selection_rates[size][order] * 100 for order in orders]
    bars = ax.bar(orders, rates, color='steelblue', alpha=0.7, edgecolor='black')
    true_idx = orders.index(true_n_components)
    bars[true_idx].set_color('red')
    bars[true_idx].set_alpha(0.8)
    ax.set_xlabel('Number of GMM Components', fontsize=11)
    ax.set_ylabel('Selection Rate (%)', fontsize=11)
    ax.set_title(f'Dataset Size: {size} samples', fontsize=12, fontweight='bold')
    ax.set_xticks(orders)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max(rates) * 1.2])

ax = axes[1, 1]
heatmap_data = np.zeros((len(candidate_orders), len(dataset_sizes)))
for i, order in enumerate(candidate_orders):
    for j, size in enumerate(dataset_sizes):
        heatmap_data[i, j] = selection_rates[size][order] * 100

im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(dataset_sizes)))
ax.set_yticks(range(len(candidate_orders)))
ax.set_xticklabels([f'{size}' for size in dataset_sizes])
ax.set_yticklabels(candidate_orders)
ax.set_xlabel('Dataset Size', fontsize=11)
ax.set_ylabel('GMM Order', fontsize=11)
ax.set_title('Selection Rate Heatmap (%)', fontsize=12, fontweight='bold')

for i in range(len(candidate_orders)):
    for j in range(len(dataset_sizes)):
        ax.text(j, i, f'{heatmap_data[i,j]:.0f}', ha="center", va="center", 
                color="black", fontsize=9)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Selection Rate (%)', fontsize=10)
ax.axhline(y=true_n_components-0.5, color='blue', linestyle='--', linewidth=2, alpha=0.7)

plt.tight_layout()
plt.savefig('Assignment 3/outputs/gmm_model_selection_results.png', dpi=300, bbox_inches='tight')
plt.show()

X_vis, labels_vis = generate_gmm_data(2000, true_weights, true_means, true_covs)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
colors = ['red', 'blue', 'green', 'orange']

for i in range(true_n_components):
    mask = labels_vis == i
    ax.scatter(X_vis[mask, 0], X_vis[mask, 1], c=colors[i], alpha=0.4, s=20, 
              label=f'Component {i}')

for i, mean in enumerate(true_means):
    ax.scatter(mean[0], mean[1], c=colors[i], marker='X', s=300, 
              edgecolors='black', linewidths=2, zorder=5)

for i in range(true_n_components):
    eigenvalues, eigenvectors = np.linalg.eig(true_covs[i])
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2*2*np.sqrt(eigenvalues)
    ellipse = Ellipse(true_means[i], width, height, angle=angle, facecolor='none', 
                     edgecolor=colors[i], linewidth=2, linestyle='--')
    ax.add_patch(ellipse)

ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title('True 4-Component GMM Distribution', fontsize=14, fontweight='bold')
legend = ax.legend(fontsize=10)
for handle in legend.legend_handles:
    handle.set_alpha(0.4)
ax.grid(alpha=0.3)
ax.axis('equal')

plt.tight_layout()
plt.savefig('Assignment 3/outputs/true_gmm_distribution.png', dpi=300, bbox_inches='tight')
plt.show()