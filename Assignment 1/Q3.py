#!Z:\ml\ml\Scripts\python.exe

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

np.random.seed(42)

# Wine Quality dataset
red_wine = pd.read_csv('wine_quality/winequality-red.csv', delimiter=';')
white_wine = pd.read_csv('wine_quality/winequality-white.csv', delimiter=';')

wine_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)
X_wine = wine_data.iloc[:,:-1].values
labels_wine = wine_data.iloc[:,-1].values
N, d = X_wine.shape
classes = np.unique(labels_wine)
C = len(classes)

print(f"Dataset: {N} samples, {d} features, {C} classes")
print(f"Classes (Quality Scores): {classes} \n")

mu_wine = np.zeros((C, d))
Cov_wine = []
prior_wine = np.zeros(C)
alpha = 0.001 

for i,c in enumerate(classes):
    Xi = X_wine[labels_wine == c]
    Ni = len(Xi)
    mu_wine[i] = np.mean(Xi, axis=0)
    Cov_sample = np.cov(Xi.T)
    lambda_reg = alpha * np.trace(Cov_sample) / np.linalg.matrix_rank(Cov_sample)
    Cov_reg = Cov_sample + lambda_reg * np.eye(d)
    Cov_wine.append(Cov_reg)
    prior_wine[i] = Ni/N
    print(f"Class {c}: {Ni} samples")
print()

posterior_wine = np.zeros((N,C))

for i,c in enumerate(classes):
    rv = multivariate_normal(mean=mu_wine[i], cov=Cov_wine[i], allow_singular=True)
    posterior_wine[:, i] = rv.pdf(X_wine) * prior_wine[i]

predictions_wine = classes[np.argmax(posterior_wine, axis=1)]
Conf_mat_wine = np.zeros((C,C))

for i in range(N):
    true_idx = np.where(classes == labels_wine[i])[0][0]
    pred_idx = np.where(classes == predictions_wine[i])[0][0]
    Conf_mat_wine[pred_idx, true_idx] += 1

Conf_mat_wine = Conf_mat_wine/Conf_mat_wine.sum(axis=0, keepdims=True)
print("Confusion Matrix P(D=i|L=j) for wine dataset:")
print("       ", end="")
for c in classes:
    print(f"L={c:d}      ", end="")
print()

for i,c_pred in enumerate(classes):
    print(f"D={c_pred:d}   ", end="")
    for j in range(C):
        print(f"{Conf_mat_wine[i,j]:7.4f}  ", end="")
    print()
print()

P_error_wine = np.mean(predictions_wine!= labels_wine)
print(f"Probability of Error: {P_error_wine:.4f} \n")
pca_wine = PCA(n_components=3)
X_pca_wine = pca_wine.fit_transform(X_wine)
explained_var = pca_wine.explained_variance_ratio_
print(f"Variance explained by first 2 PCs: {100*sum(explained_var[:2]):.2f}% \n")

fig, ax = plt.subplots(figsize=(12,8))
colors = plt.cm.tab10(np.linspace(0,1,C))
markers = ['o', 's', '^', 'd', 'v', '>', '<', 'p', 'h', '*']

for i,c in enumerate(classes):
    class_mask = labels_wine == c
    correct_mask = class_mask & (predictions_wine == labels_wine)
    incorrect_mask = class_mask & (predictions_wine!= labels_wine)

    if np.sum(correct_mask) > 0:
        ax.scatter(X_pca_wine[correct_mask, 0], X_pca_wine[correct_mask, 1],
                  c=[colors[i]], marker=markers[i % len(markers)], s=30,
                  edgecolors='k', linewidths=0.5, alpha=0.6,
                  label=f'Quality {c}')
    
    if np.sum(incorrect_mask) > 0:
        ax.scatter(X_pca_wine[incorrect_mask, 0], X_pca_wine[incorrect_mask, 1],
                  c=[colors[i]], marker=markers[i % len(markers)], s=80,
                  edgecolors='r', linewidths=2, alpha=0.8)

ax.set_xlabel('First Principal Component', fontsize=12)
ax.set_ylabel('Second Principal Component', fontsize=12)
ax.set_title(f'Wine Quality Classification (P(error)={P_error_wine:.4f})\n' +
             'Filled=Correct, Red Edge=Incorrect', fontsize=13)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Q3_ops/wine_quality_2d.png', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')

for i, c in enumerate(classes):
    class_mask = labels_wine == c
    ax.scatter(X_pca_wine[class_mask, 0], X_pca_wine[class_mask, 1], 
              X_pca_wine[class_mask, 2],
              c=[colors[i]], marker=markers[i % len(markers)], s=20,
              edgecolors='k', linewidths=0.5, alpha=0.6,
              label=f'Quality {c}')

ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_zlabel('PC3', fontsize=12)
ax.set_title('Wine Quality - 3D PCA Projection', fontsize=13)
ax.legend(bbox_to_anchor=(1.15,1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('Q3_ops/wine_quality_3d.png', dpi=300, bbox_inches='tight')
plt.show()

# Human Activity Recognition dataset
X_train = np.loadtxt('human_activity_ecognition_using_smartphones/UCI HAR Dataset/train/X_train.txt')
y_train = np.loadtxt('human_activity_ecognition_using_smartphones/UCI HAR Dataset/train/y_train.txt')
X_test = np.loadtxt('human_activity_ecognition_using_smartphones/UCI HAR Dataset/test/X_test.txt')
y_test = np.loadtxt('human_activity_ecognition_using_smartphones/UCI HAR Dataset/test/y_test.txt')

X_har = np.vstack([X_train,X_test])
labels_har = np.concatenate([y_train, y_test]).astype(int)
N_har, d_har = X_har.shape
classes_har = np.unique(labels_har)
C_har = len(classes_har)

print(f"Dataset: {N_har} samples, {d_har} features, {C_har} classes")
activity_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs',
                  'Sitting', 'Standing', 'Laying']
mu_har = np.zeros((C_har,d_har))
Cov_har = []
prior_har = np.zeros(C_har)
alpha_har = 0.001

for i, c in enumerate(classes_har):
    Xi = X_har[labels_har == c]
    Ni = len(Xi)
    mu_har[i] = np.mean(Xi, axis=0)
    Cov_sample = np.cov(Xi.T)
    lambda_reg = alpha_har * np.trace(Cov_sample)/np.linalg.matrix_rank(Cov_sample)
    Cov_reg = Cov_sample + lambda_reg * np.eye(d_har)
    Cov_har.append(Cov_reg)
    prior_har[i] = Ni/N_har
    if i < len(activity_names):
        print(f"Class {c} ({activity_names[i]}): {Ni}")
print("\n")
posterior_har = np.zeros((N_har,C_har))

for i, c in enumerate(classes_har):
    rv = multivariate_normal(mean=mu_har[i], cov=Cov_har[i], allow_singular=True)
    posterior_har[:, i] = rv.pdf(X_har) * prior_har[i]

predictions_har = classes_har[np.argmax(posterior_har, axis=1)]
Conf_mat_har = np.zeros((C_har,C_har))

for i in range(N_har):
    true_idx = np.where(classes_har == labels_har[i])[0][0]
    pred_idx = np.where(classes_har == predictions_har[i])[0][0]
    Conf_mat_har[pred_idx, true_idx]+= 1

Conf_mat_har = Conf_mat_har/Conf_mat_har.sum(axis=0, keepdims=True)

print("Confusion Matrix P(D=i|L=j) for human activity dataset:")
print("       ", end="")

for c in classes_har:
    print(f"L={c:d}      ", end="")
print()

for i, c_pred in enumerate(classes_har):
    print(f"D={c_pred:d}   ", end="")
    for j in range(C_har):
        print(f"{Conf_mat_har[i,j]:7.4f}  ", end="")
    print()
print()

P_error_har = np.mean(predictions_har!= labels_har)
print(f"Probability of Error: {P_error_har:.4f} \n")

pca_har = PCA(n_components=7)
X_pca_har = pca_har.fit_transform(X_har)
explained_var_har = pca_har.explained_variance_ratio_
print(f"Variance explained by first 6 PCs: {100*sum(explained_var_har[:6]):.2f}% \n")

fig, ax = plt.subplots(figsize=(12,8))
colors_har = plt.cm.tab10(np.linspace(0,1,C_har))

for i, c in enumerate(classes_har):
    class_mask = labels_har == c
    correct_mask = class_mask & (predictions_har == labels_har)
    incorrect_mask = class_mask & (predictions_har!= labels_har)
    
    if np.sum(correct_mask) > 0:
        ax.scatter(X_pca_har[correct_mask, 0], X_pca_har[correct_mask, 1],
                  c=[colors_har[i]], marker='o', s=20, alpha=0.6,
                  edgecolors='k', linewidths=0.5,
                  label=activity_names[i] if i < len(activity_names) else f'Class {c}')

    if np.sum(incorrect_mask) > 0:
        ax.scatter(X_pca_har[incorrect_mask, 0], X_pca_har[incorrect_mask, 1],
                  c=[colors_har[i]], marker='x', s=60, linewidths=2, alpha=0.8)

ax.set_xlabel('First Principal Component', fontsize=12)
ax.set_ylabel('Second Principal Component', fontsize=12)
ax.set_title(f'Human Activity Recognition (P(error)={P_error_har:.4f})\n' +
             'Circle=Correct, X=Incorrect', fontsize=13)
ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Q3_ops/har_2d.png', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')

for i, c in enumerate(classes_har):
    class_mask = labels_har == c
    ax.scatter(X_pca_har[class_mask, 0], X_pca_har[class_mask, 1],
              X_pca_har[class_mask, 2],
              c=[colors_har[i]], marker='o', s=15, alpha=0.6,
              edgecolors='k', linewidths=0.3,
              label=activity_names[i] if i < len(activity_names) else f'Class {c}')

ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_zlabel('PC3', fontsize=12)
ax.set_title('Human Activity Recognition - 3D PCA Projection', fontsize=13)
ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('Q3_ops/har_3d.png', dpi=300, bbox_inches='tight')
plt.show()
