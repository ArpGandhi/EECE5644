import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)
priors = np.array([0.25,0.25,0.25,0.25])

means = [
    np.array([0,0,0]),
    np.array([3,3,0]),
    np.array([3,0,3]),
    np.array([0,3,3])
]

covs = [
    np.array([[2.0,0.5,0.3], 
              [0.5,2.0,0.5], 
              [0.3,0.5,2.0]]),
    np.array([[1.8,-0.3,0.2],  
              [-0.3,1.8,-0.3], 
              [0.2,-0.3,1.8]]),
    np.array([[1.9,0.6,-0.2],  
              [0.6,1.9,0.3], 
              [-0.2,0.3,1.9]]),
    np.array([[2.1,-0.4,0.4],  
              [-0.4,2.1,-0.4], 
              [0.4,-0.4,2.1]])
]

num_classes = 4

def generate_data(n_samples, means, covs, priors):
    n_dim = len(means[0])
    X = np.zeros((n_samples, n_dim))
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        class_idx = np.random.choice(num_classes, p=priors)
        X[i] = np.random.multivariate_normal(means[class_idx], covs[class_idx])
        y[i] = class_idx

    return X,y

def theoretical_map_classifier(X, means, covs, priors):
    n_samples = X.shape[0]
    posteriors = np.zeros((n_samples, num_classes))
    for c in range(num_classes):
        posteriors[:,c] = multivariate_normal.pdf(X,means[c],covs[c])*priors[c]
    predictions = np.argmax(posteriors,axis=1)

    return predictions

def train_mlp(X_train,y_train,hidden_dim,n_restarts=5):
    best_model = None
    best_score = -np.inf
    for restart in range(n_restarts):
        model = MLPClassifier(hidden_layer_sizes=(hidden_dim,),
            activation='tanh', 
            solver='adam',max_iter=1000,random_state=restart,
            alpha=0.0001,learning_rate_init=0.001)
        model.fit(X_train, y_train)
        score = model.score(X_train, y_train)
        if score > best_score:
            best_score = score
            best_model = model

    return best_model

def predict_mlp(model,X):
    return model.predict(X)

def evaluate_error(y_true, y_pred):
    return 1-accuracy_score(y_true, y_pred)

def cross_validate_perceptrons(X_train, y_train, perceptron_range, n_folds=10):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_errors = {p: [] for p in perceptron_range}
    for p in perceptron_range:
        print(f"Evaluating {p} perceptrons")
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            model = train_mlp(X_tr, y_tr, hidden_dim=p, n_restarts=3)
            y_pred = predict_mlp(model, X_val)
            error = evaluate_error(y_val, y_pred)
            cv_errors[p].append(error)
    avg_errors = {p: np.mean(cv_errors[p]) for p in perceptron_range}
    best_p = min(avg_errors, key=avg_errors.get)
    print(f"Cross-validation errors: {avg_errors}")
    print(f"Selected {best_p} perceptrons") 

    return best_p, avg_errors

train_sizes = [100,500,1000,5000,10000]
test_size = 100000
perceptron_range = [1,2,3,4,5,8,10,15,20]
X_test, y_test = generate_data(test_size, means, covs, priors)

y_pred_optimal = theoretical_map_classifier(X_test, means, covs, priors)
error_optimal = evaluate_error(y_test, y_pred_optimal)
print(f"Theoretical optimal error: {error_optimal:.4f}")

results = {
    'train_sizes': train_sizes,
    'optimal_perceptrons': [],
    'test_errors': [],
    'optimal_error': error_optimal
}

for n_train in train_sizes:
    print(f"Training with {n_train} samples")
    X_train, y_train = generate_data(n_train, means, covs, priors)
    best_p, cv_errors = cross_validate_perceptrons(X_train, y_train, perceptron_range)
    results['optimal_perceptrons'].append(best_p)
    print(f"\nTraining final model with {best_p} perceptrons")
    final_model = train_mlp(X_train, y_train, hidden_dim=best_p, n_restarts=5)
    y_pred_test = predict_mlp(final_model, X_test)
    test_error = evaluate_error(y_test, y_pred_test)
    results['test_errors'].append(test_error)
    print(f"Test error: {test_error:.4f}")

print(f"Output activation: {final_model.out_activation_}")
print(f"Number of outputs: {final_model.n_outputs_}")

print("MLP Architecture")
print(f"Number of layers: {final_model.n_layers_}")
print(f"Hidden layer sizes: {final_model.hidden_layer_sizes}")
print(f"Layer shapes: {[coef.shape for coef in final_model.coefs_]}")
print(f"Input features: {final_model.n_features_in_}")
print(f"Output classes: {final_model.n_outputs_}")

plt.figure(figsize=(10, 6))
plt.semilogx(results['train_sizes'], results['test_errors'], 
             'bo-', linewidth=2, markersize=8, label='MLP Classifier')

plt.axhline(y=results['optimal_error'], color='r', linestyle='--', 
            linewidth=2, label='Theoretical Optimal')

plt.xlabel('Number of Training Samples', fontsize=12)
plt.ylabel('Probability of Error', fontsize=12)
plt.title('MLP Classification Performance vs Training Set Size', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

for i, (n, p) in enumerate(zip(results['train_sizes'], results['optimal_perceptrons'])):
    plt.annotate(f'{p}P', (n, results['test_errors'][i]), 
                textcoords="offset points", xytext=(0,10), 
                ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('Assignment 3/outputs/mlp_performance.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"{'Train Size':<12} {'Optimal P':<12} {'Test Error':<15} {'Error %':<10}")
for n, p, err in zip(results['train_sizes'], 
                     results['optimal_perceptrons'], 
                     results['test_errors']):
    print(f"{n:<12} {p:<12} {err:<15.6f} {err*100:<10.2f}")
print(f"{'Theoretical':<12} {'-':<12} {error_optimal:<15.6f}")