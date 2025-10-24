import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generateDataFromGMM(N, gmmParameters):
#    Generates N vector samples from the specified mixture of Gaussians
#    Returns samples and their component labels
#    Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors']
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0]
    C = len(priors)
    x = np.zeros((n,N))
    labels = np.zeros((1,N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1,N))
    thresholds = np.zeros((1,C+1))
    thresholds[:,0:C] = np.cumsum(priors)
    thresholds[:,C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:, l]))
        Nl = len(indl[1])
        labels[indl] = (l+1)*1
        u[indl] = 1.1
        x[:,indl[1]] = np.transpose(
            np.random.multivariate_normal(meanVectors[:, l], covMatrices[:, :, l], Nl))
    
    return x, labels

def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3,.4,.3]
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10],[0, 0, 0],[10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:, :, 0] = np.array([[1, 0, -3],[0, 1, 0],[-3, 0, 15]])
    gmmParameters['covMatrices'][:, :, 1] = np.array([[8, 0, 0],[0, .5, 0],[0, 0, .5]])
    gmmParameters['covMatrices'][:, :, 2] = np.array([[1, 0, -3],[0, 1, 0],[-3, 0, 15]])
    x, labels = generateDataFromGMM(N, gmmParameters)
    return x

def hw2q2():
    Ntrain = 100
    data = generateData(Ntrain)
    xTrain = data[0:2,:]
    yTrain = data[2,:]
    Nvalidate = 1000
    data = generateData(Nvalidate)
    xValidate = data[0:2,:]
    yValidate = data[2,:]
    return xTrain, yTrain, xValidate, yValidate

def cubic_features(X):
    if X.ndim == 1:
        X = X.reshape(1,-1)
    elif X.shape[0] == 2:
        X = X.T

    x1 = X[:,0]
    x2 = X[:,1]
    features = np.column_stack([
        np.ones(len(X)),x1,x2,                  
        x1**2,x1*x2,x2**2,        
        x1**3,(x1**2)*x2,x1*(x2**2),x2**3
    ])
    return features

def ml_estimator(X,y):
    Z = cubic_features(X)
    w_ml = np.linalg.lstsq(Z,y,rcond=None)[0]
    return w_ml

def map_estimator(X,y,gamma,sigma_squared=1.0):
    Z = cubic_features(X)
    n_features = Z.shape[1]
    w_map = np.linalg.lstsq(Z.T@Z+(sigma_squared/gamma)*np.eye(n_features),Z.T@y,rcond=None)[0]
    return w_map

def evaluate_model(w,X,y):
    Z = cubic_features(X)
    y_pred = Z@w
    mse = np.mean((y-y_pred)**2)
    return mse

np.random.seed(42)
xTrain, yTrain, xValidate, yValidate = hw2q2()

print(f"\nTraining samples: {xTrain.shape[1]}")
print(f"Validation samples: {xValidate.shape[1]}")
sigma_squared = 1.0

w_ml = ml_estimator(xTrain, yTrain)
mse_ml_train = evaluate_model(w_ml, xTrain, yTrain)
mse_ml_val = evaluate_model(w_ml, xValidate, yValidate)

print(f"ML Training MSE: {mse_ml_train:.4f}")
print(f"ML Validation MSE: {mse_ml_val:.4f}")
print(f"ML Weight norm: {np.linalg.norm(w_ml):.4f}")

gamma_values = np.logspace(-6,6,100)
mse_map_train = []
mse_map_val = []
w_norms = []

for gamma in gamma_values:
    w_map = map_estimator(xTrain, yTrain, gamma, sigma_squared)
    mse_train = evaluate_model(w_map, xTrain, yTrain)
    mse_val = evaluate_model(w_map, xValidate, yValidate)
    mse_map_train.append(mse_train)
    mse_map_val.append(mse_val)
    w_norms.append(np.linalg.norm(w_map))

mse_map_train = np.array(mse_map_train)
mse_map_val = np.array(mse_map_val)
w_norms = np.array(w_norms)
best_idx = np.argmin(mse_map_val)
best_gamma = gamma_values[best_idx]
best_mse_val = mse_map_val[best_idx]
w_best = map_estimator(xTrain,yTrain,best_gamma,sigma_squared)

print(f"\nBest \u03B3: {best_gamma:.4e}")
print(f"Best MAP Validation MSE: {best_mse_val:.4f}")
print(f"ML Validation MSE: {mse_ml_val:.4f}")
print(f"Improvement: {((mse_ml_val-best_mse_val)/mse_ml_val*100):.2f}%")

fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(3, 3, 1, projection='3d')
ax1.scatter(xTrain[0, :], xTrain[1, :], yTrain, c='blue', marker='o', alpha=0.6, s=30)
ax1.set_xlabel(r'$x_1$', fontsize=10)
ax1.set_ylabel(r'$x_2$', fontsize=10)
ax1.set_zlabel('y', fontsize=10)
ax1.set_title('Training Data (N=100)', fontsize=11, fontweight='bold')

ax2 = fig.add_subplot(3, 3, 2, projection='3d')
ax2.scatter(xValidate[0, :], xValidate[1, :], yValidate,c='red', marker='o', alpha=0.3, s=20)
ax2.set_xlabel(r'$x_1$', fontsize=10)
ax2.set_ylabel(r'$x_2$', fontsize=10)
ax2.set_zlabel('y', fontsize=10)
ax2.set_title('Validation Data (N=1000)', fontsize=11, fontweight='bold')

ax3 = fig.add_subplot(3, 3, 3)
ax3.semilogx(gamma_values, mse_map_val, 'b-', linewidth=2.5, label='MAP Validation MSE')
ax3.axhline(y=mse_ml_val, color='r', linestyle='--', linewidth=2, label='ML Validation MSE')
ax3.axvline(x=best_gamma, color='g', linestyle=':', linewidth=2, 
           label=r'Optimal $\gamma$ = {best_gamma:.2e}')
ax3.scatter([best_gamma], [best_mse_val], color='g', s=100, zorder=5, marker='*')
ax3.set_xlabel(r'$\gamma$ (prior variance)', fontsize=11)
ax3.set_ylabel('Validation MSE', fontsize=11)
ax3.set_title(r'MAP Performance vs $\gamma$', fontsize=11, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(3, 3, 4)
ax4.semilogx(gamma_values, mse_map_train, 'b-', linewidth=2.5, label='MAP Training MSE')
ax4.axhline(y=mse_ml_train, color='r', linestyle='--', linewidth=2, label='ML Training MSE')
ax4.axvline(x=best_gamma, color='g', linestyle=':', linewidth=2)
ax4.set_xlabel(r'$\gamma$ (prior variance)', fontsize=11)
ax4.set_ylabel('Training MSE', fontsize=11)
ax4.set_title(r'Training Error vs $\gamma$', fontsize=11, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(3, 3, 5)
ax5.loglog(gamma_values, w_norms, 'b-', linewidth=2.5)
ax5.axvline(x=best_gamma, color='g', linestyle=':', linewidth=2, label=r'Optimal $\gamma$ = {best_gamma:.2e}')
ax5.axhline(y=np.linalg.norm(w_ml), color='r', linestyle='--',linewidth=2, label='ML weight norm')
ax5.set_xlabel(r'$\gamma$ (prior variance)', fontsize=11)
ax5.set_ylabel(r'$||w||_2$', fontsize=11)
ax5.set_title(r'Weight Norm vs $\gamma$', fontsize=11, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(3, 3, 6)
ax6.semilogx(gamma_values, mse_map_train, 'b-', linewidth=2, label='Training MSE')
ax6.semilogx(gamma_values, mse_map_val, 'r-', linewidth=2, label='Validation MSE')
ax6.axvline(x=best_gamma, color='g', linestyle=':', linewidth=2)
ax6.set_xlabel(r'$\gamma$ (prior variance)', fontsize=11)
ax6.set_ylabel('MSE', fontsize=11)
ax6.set_title('Overfitting Analysis', fontsize=11, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

ax7 = fig.add_subplot(3, 3, 7, projection='3d')
x1_grid = np.linspace(xTrain[0, :].min(), xTrain[0, :].max(), 30)
x2_grid = np.linspace(xTrain[1, :].min(), xTrain[1, :].max(), 30)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
X_grid = np.vstack([X1_grid.ravel(), X2_grid.ravel()])
Z_ml = cubic_features(X_grid) @ w_ml
Z_ml_grid = Z_ml.reshape(X1_grid.shape)
ax7.plot_surface(X1_grid, X2_grid, Z_ml_grid, alpha=0.6, cmap='viridis')
ax7.scatter(xTrain[0, :], xTrain[1, :], yTrain, c='red', marker='o', s=20)
ax7.set_xlabel(r'$x_1$', fontsize=10)
ax7.set_ylabel(r'$x_2$', fontsize=10)
ax7.set_zlabel('y', fontsize=10)
ax7.set_title('ML Model Fit', fontsize=11, fontweight='bold')

ax8 = fig.add_subplot(3, 3, 8, projection='3d')
Z_map = cubic_features(X_grid) @ w_best
Z_map_grid = Z_map.reshape(X1_grid.shape)
ax8.plot_surface(X1_grid, X2_grid, Z_map_grid, alpha=0.6, cmap='plasma')
ax8.scatter(xTrain[0, :], xTrain[1, :], yTrain, c='red', marker='o', s=20)
ax8.set_xlabel(r'$x_1$', fontsize=10)
ax8.set_ylabel(r'$x_2$', fontsize=10)
ax8.set_zlabel('y', fontsize=10)
ax8.set_title(rf'MAP Model Fit ($\gamma$={best_gamma:.2e})', fontsize=11, fontweight='bold')

ax9 = fig.add_subplot(3, 3, 9)
residuals_ml = yValidate - cubic_features(xValidate) @ w_ml
residuals_map = yValidate - cubic_features(xValidate) @ w_best
ax9.hist(residuals_ml, bins=30, alpha=0.5, label='ML Residuals', color='red')
ax9.hist(residuals_map, bins=30, alpha=0.5, label='MAP Residuals', color='blue')
ax9.set_xlabel('Residual', fontsize=11)
ax9.set_ylabel('Frequency', fontsize=11)
ax9.set_title('Residual Distribution on Validation Set', fontsize=11, fontweight='bold')
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Assignment 2/outputs/q2_complete_analysis.png', dpi=150, bbox_inches='tight')