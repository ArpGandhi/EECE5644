import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from PIL import Image

np.random.seed(42)
IMAGE_PATH = "Assignment 4/253027.jpg"

try:
    img = Image.open(IMAGE_PATH)
    image = np.array(img)
    print(f"Successfully loaded image")
except Exception as e:
    print(f"Failed to load image: {e}")

print(f"Image shape: {image.shape}")

if len(image.shape) == 2:
    image = np.stack([image] * 3, axis=-1)

def create_feature_vectors(image):
    height, width = image.shape[:2]
    row_indices, col_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    row_flat = row_indices.flatten()
    col_flat = col_indices.flatten()
    red_flat = image[:,:,0].flatten()
    green_flat = image[:,:,1].flatten()
    blue_flat = image[:,:,2].flatten()
    features = np.column_stack([row_flat, col_flat, red_flat, green_flat, blue_flat])
    features_normalized = np.zeros_like(features, dtype=np.float64)
    for i in range(features.shape[1]):
        min_val = features[:,i].min()
        max_val = features[:,i].max()
        if max_val > min_val:
            features_normalized[:, i] = (features[:, i] - min_val) / (max_val - min_val)
        else:
            features_normalized[:, i] = 0
    return features_normalized

features = create_feature_vectors(image)
print(f"Feature matrix shape: {features.shape}\n")

n_components_range = [2,3,4,5,6,7,8,10,12,15]
k_folds_range = [2,3,4,5,6,7,8,9,10]

all_results = {}

for k_fold in k_folds_range:
    print(f"\nTesting {k_fold}-fold:")
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    cv_scores = []
    cv_scores_std = []
    
    for n_components in n_components_range:
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(features)):
            X_train, X_val = features[train_idx], features[val_idx]
            gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                                random_state=42, max_iter=100)
            gmm.fit(X_train)
            val_log_likelihood = gmm.score(X_val)
            fold_scores.append(val_log_likelihood)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        cv_scores.append(mean_score)
        cv_scores_std.append(std_score)
        print(f"  {n_components:2d} components: {mean_score:.4f} ± {std_score:.4f}")
    
    best_idx = np.argmax(cv_scores)
    best_n_components = n_components_range[best_idx]
    best_cv_score = cv_scores[best_idx]
    
    all_results[k_fold] = {
        'cv_scores': cv_scores,
        'cv_scores_std': cv_scores_std,
        'best_n_components': best_n_components,
        'best_cv_score': best_cv_score
    }
    
    print(f"  Best: {best_n_components} components (score: {best_cv_score:.4f})")

print(f"\nK-fold Summary:")
for k in k_folds_range:
    result = all_results[k]
    print(f"  {k}-fold: {result['best_n_components']} components (score: {result['best_cv_score']:.4f})")

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, k in enumerate(k_folds_range):
    result = all_results[k]
    ax = axes[idx]
    
    ax.plot(n_components_range, result['cv_scores'], 'o-', linewidth=2, markersize=6)
    ax.axvline(result['best_n_components'], color='r', linestyle='--', linewidth=1.5,
               label=f"Best: {result['best_n_components']}")
    ax.set_xlabel('GMM Components')
    ax.set_ylabel('Log-Likelihood')
    ax.set_title(f'{k}-Fold CV')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('Assignment 4\outputs\gmm_cv_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

best_k = max(all_results.keys(), key=lambda k: all_results[k]['best_cv_score'])
best_n_components = all_results[best_k]['best_n_components']
best_cv_score = all_results[best_k]['best_cv_score']

print(f"\nBest overall: {best_k}-fold CV with {best_n_components} components")

final_gmm = GaussianMixture(n_components=best_n_components,covariance_type='full',
                            random_state=42,max_iter=200,verbose=1)
final_gmm.fit(features)

print(f"\nConverged: {final_gmm.converged_}")
print(f"Iterations: {final_gmm.n_iter_}")
print(f"Final log-likelihood: {final_gmm.score(features):.4f}\n")

labels = final_gmm.predict(features)
height, width = image.shape[:2]
label_image = labels.reshape(height, width)

unique_labels = np.unique(labels)
n_segments = len(unique_labels)

print(f"Segments: {n_segments}")
for seg_id in unique_labels:
    count = np.sum(labels == seg_id)
    percentage = (count/len(labels))*100
    print(f"  Segment {seg_id}: {count:6d} pixels ({percentage:5.2f}%)")

colors = np.random.randint(50, 255, size=(n_segments, 3), dtype=np.uint8)
segmented_color = np.zeros((height, width, 3), dtype=np.uint8)
for i, label in enumerate(unique_labels):
    mask = label_image == label
    segmented_color[mask] = colors[i]

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(image)
axes[0].set_title('Original Image', fontsize=16, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(segmented_color)
axes[1].set_title(f'GMM Segmentation ({best_n_components} components)', 
                  fontsize=16, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('Assignment 4\outputs\segmentation_result.png', dpi=300, bbox_inches='tight')
plt.show()

for label in unique_labels:
    mask = labels == label
    segment_features = features[mask]
    avg_row = segment_features[:, 0].mean() * (height - 1)
    avg_col = segment_features[:, 1].mean() * (width - 1)
    avg_red = segment_features[:, 2].mean() * 255
    avg_green = segment_features[:, 3].mean() * 255
    avg_blue = segment_features[:, 4].mean() * 255
    print(f"  Segment {label}: {np.sum(mask)} pixels, center ({avg_row:.0f},{avg_col:.0f}), RGB({avg_red:.0f},{avg_green:.0f},{avg_blue:.0f})")