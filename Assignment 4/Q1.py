import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import seaborn as sns

np.random.seed(42)
iters=2000

def generate_data(n_samples, r_neg=2, r_pos=4, sigma=1):
    X,y = [],[]
    for label, radius in [(-1, r_neg), (1, r_pos)]:
        n = n_samples // 2
        theta = np.random.uniform(-np.pi, np.pi, n)
        x_clean = radius * np.column_stack([np.cos(theta), np.sin(theta)])
        noise = np.random.normal(0, sigma, (n, 2))
        x_noisy = x_clean + noise
        X.append(x_noisy)
        y.append(np.full(n, label))
    X = np.vstack(X)
    y = np.hstack(y)
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

X_train, y_train = generate_data(1000)
X_test, y_test = generate_data(10000)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], 
            alpha=0.6, label='Class -1', s=30)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
            alpha=0.6, label='Class +1', s=30)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Training Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1], 
            alpha=0.3, label='Class -1', s=10)
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], 
            alpha=0.3, label='Class +1', s=10)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Test Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.tight_layout()
plt.savefig('Assignment 4\outputs\data_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

gamma_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
C_values = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
k_folds_range = [2,3,4,5,6,7,8,9,10]

svm_all_results = {}

for k_folds in k_folds_range:
    print(f"\nTesting SVM with {k_folds}-fold:")
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    svm_cv_results = np.zeros((len(gamma_values), len(C_values)))
    
    for i, gamma in enumerate(gamma_values):
        for j, C in enumerate(C_values):
            fold_accuracies = []
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                svm = SVC(kernel='rbf', gamma=gamma, C=C, random_state=42)
                svm.fit(X_tr, y_tr)
                y_pred = svm.predict(X_val)
                fold_accuracies.append(accuracy_score(y_val, y_pred))
            svm_cv_results[i, j] = np.mean(fold_accuracies)
    
    best_idx = np.unravel_index(np.argmax(svm_cv_results), svm_cv_results.shape)
    best_gamma = gamma_values[best_idx[0]]
    best_C = C_values[best_idx[1]]
    best_cv_acc = svm_cv_results[best_idx]
    
    svm_all_results[k_folds] = {
        'cv_results': svm_cv_results,
        'best_gamma': best_gamma,
        'best_C': best_C,
        'best_cv_acc': best_cv_acc
    }
    
    print(f"  Best: gamma={best_gamma}, C={best_C}, accuracy={best_cv_acc:.4f}")

print("\nSVM K-fold Summary:")
for k in k_folds_range:
    result = svm_all_results[k]
    print(f"  {k}-fold: gamma={result['best_gamma']}, C={result['best_C']}, acc={result['best_cv_acc']:.4f}")

fig, axes = plt.subplots(3,3,figsize=(14,12))
axes = axes.flatten()

for idx, k_folds in enumerate(k_folds_range):
    result = svm_all_results[k_folds]
    ax = axes[idx]
    sns.heatmap(result['cv_results'], annot=True, fmt='.3f', cmap='viridis',
                xticklabels=[f'{c:.1f}' for c in C_values],
                yticklabels=[f'{g:.2f}' for g in gamma_values],
                ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_xlabel('C (Box Constraint)')
    ax.set_ylabel('Gamma (Kernel Width)')
    ax.set_title(f'SVM {k_folds}-Fold CV (Best: \u03B3={result["best_gamma"]}, C={result["best_C"]})')

plt.tight_layout()
plt.savefig('Assignment 4\outputs\svm_cv_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

best_k = max(svm_all_results.keys(), key=lambda k: svm_all_results[k]['best_cv_acc'])
best_gamma = svm_all_results[best_k]['best_gamma']
best_C = svm_all_results[best_k]['best_C']
best_cv_acc = svm_all_results[best_k]['best_cv_acc']

print(f"\nBest overall SVM: {best_k}-fold with gamma={best_gamma}, C={best_C}")

final_svm = SVC(kernel='rbf', gamma=best_gamma, C=best_C, random_state=42)
final_svm.fit(X_train, y_train)

y_pred_svm = final_svm.predict(X_test)
svm_test_acc = accuracy_score(y_test, y_pred_svm)

print(f"SVM Test Accuracy: {svm_test_acc:.4f}")
print(f"SVM Test Error: {1-svm_test_acc:.4f}\n")

perceptron_range = [5,10,15,20,25,30,40,50]
mlp_all_results = {}

for k_folds in k_folds_range:
    print(f"\nTesting MLP with {k_folds}-fold CV:")
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    mlp_cv_results = []
    
    for perceptron in perceptron_range:
        fold_accuracies = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            mlp = MLPClassifier(hidden_layer_sizes=(perceptron,), activation='tanh',
                               solver='adam', max_iter=iters,
                               random_state=42, early_stopping=False)
            mlp.fit(X_tr, y_tr)
            y_pred = mlp.predict(X_val)
            fold_accuracies.append(accuracy_score(y_val, y_pred))
        avg_acc = np.mean(fold_accuracies)
        mlp_cv_results.append(avg_acc)
    
    best_idx = np.argmax(mlp_cv_results)
    best_perceptron = perceptron_range[best_idx]
    best_mlp_cv_acc = mlp_cv_results[best_idx]
    
    mlp_all_results[k_folds] = {
        'cv_results': mlp_cv_results,
        'best_perceptron': best_perceptron,
        'best_mlp_cv_acc': best_mlp_cv_acc
    }
    
    print(f"  Best: {best_perceptron} perceptrons, accuracy={best_mlp_cv_acc:.4f}")

print("\nMLP K-fold Summary:")
for k in k_folds_range:
    result = mlp_all_results[k]
    print(f"  {k}-fold: {result['best_perceptron']} perceptrons, acc={result['best_mlp_cv_acc']:.4f}")

fig, axes = plt.subplots(3,3, figsize=(14, 10))
axes = axes.flatten()

for idx, k_folds in enumerate(k_folds_range):
    result = mlp_all_results[k_folds]
    ax = axes[idx]
    ax.plot(perceptron_range, result['cv_results'], 'o-', linewidth=2, markersize=6)
    ax.axvline(result['best_perceptron'], color='r', linestyle='--',
               label=f"Best: {result['best_perceptron']}")
    ax.set_xlabel('Hidden Units')
    ax.set_ylabel('CV Accuracy')
    ax.set_title(f'MLP {k_folds}-Fold CV')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('Assignment 4\outputs\mlp_cv_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

best_k = max(mlp_all_results.keys(), key=lambda k: mlp_all_results[k]['best_mlp_cv_acc'])
best_perceptron = mlp_all_results[best_k]['best_perceptron']
best_mlp_cv_acc = mlp_all_results[best_k]['best_mlp_cv_acc']

print(f"\nBest overall MLP: {best_k}-fold with {best_perceptron} perceptrons")

final_mlp = MLPClassifier(hidden_layer_sizes=(best_perceptron,), activation='tanh',
                          solver='adam', max_iter=iters,
                          random_state=42, early_stopping=False)
final_mlp.fit(X_train, y_train)

y_pred_mlp = final_mlp.predict(X_test)
mlp_test_acc = accuracy_score(y_test, y_pred_mlp)

print(f"MLP Test Accuracy: {mlp_test_acc:.4f}")
print(f"MLP Test Error: {1-mlp_test_acc:.4f}\n")

def plot_decision_boundary(model, X, y, title, filepath):
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, levels=[-1.5, 0, 1.5], 
                 colors=['blue', 'red'])
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    sample_idx = np.random.choice(len(X), min(2000, len(X)), replace=False)
    plt.scatter(X[sample_idx][y[sample_idx] == -1, 0], 
                X[sample_idx][y[sample_idx] == -1, 1],
                c='blue', alpha=0.4, s=10, label='Class -1')
    plt.scatter(X[sample_idx][y[sample_idx] == 1, 0], 
                X[sample_idx][y[sample_idx] == 1, 1],
                c='red', alpha=0.4, s=10, label='Class +1')

    plt.xlabel('$x_1$', fontsize=12)
    plt.ylabel('$x_2$', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

plot_decision_boundary(final_svm, X_test, y_test,
                       f'SVM Decision Boundary (Test Error: {1-svm_test_acc:.4f})',
                       'Assignment 4\outputs\svm_decision_boundary.png')

plot_decision_boundary(final_mlp, X_test, y_test,
                       f'MLP Decision Boundary (Test Error: {1-mlp_test_acc:.4f})',
                       'Assignment 4\outputs\mlp_decision_boundary.png')

print(f"SVM: gamma={best_gamma}, C={best_C}, CV error={1-svm_all_results[best_k]['best_cv_acc']:.4f}, Test error={1-svm_test_acc:.4f}")
print(f"MLP: {best_perceptron} perceptrons, CV error={1-best_mlp_cv_acc:.4f}, Test error={1-mlp_test_acc:.4f}")