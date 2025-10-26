import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from sklearn.metrics import roc_curve, auc

np.random.seed(42)

prior_L0 = 0.6
prior_L1 = 0.4
m01 = np.array([-0.9,-1.1])
m02 = np.array([0.8,0.75])
m11 = np.array([-1.1,0.9])
m12 = np.array([0.9,-0.75])
w01 = w02 = w11 = w12 = 0.5
C01 = C02 = C11 = C12 = np.array([[0.75,0],[0,1.25]])

def generate_data(n_samples):
    X = []
    labels = []
    for _ in range(n_samples):
        if np.random.rand()<prior_L0:
            label = 0
            if np.random.rand()<w01:
                x = np.random.multivariate_normal(m01,C01)
            else:
                x = np.random.multivariate_normal(m02,C02)
        else:
            label = 1
            if np.random.rand()<w11:
                x = np.random.multivariate_normal(m11,C11)
            else:
                x = np.random.multivariate_normal(m12,C12)
        X.append(x)
        labels.append(label)
    return np.array(X),np.array(labels)

def class_conditional_pdf(x,label):
    if label == 0:
        pdf1 = multivariate_normal.pdf(x,mean=m01,cov=C01)
        pdf2 = multivariate_normal.pdf(x,mean=m02,cov=C02)
        return w01*pdf1 + w02*pdf2
    else:
        pdf1 = multivariate_normal.pdf(x,mean=m11,cov=C11)
        pdf2 = multivariate_normal.pdf(x,mean=m12,cov=C12)
        return w11*pdf1 + w12*pdf2

def posterior_probability(x,label):
    prior = prior_L0 if label == 0 else prior_L1
    likelihood = class_conditional_pdf(x,label)
    p_x = prior_L0*class_conditional_pdf(x,0) + prior_L1*class_conditional_pdf(x,1)
    return (prior*likelihood)/p_x if p_x>0 else 0

def optimal_classifier(X):
    predictions = []
    scores = []
    for x in X:
        p0 = posterior_probability(x,0)
        p1 = posterior_probability(x,1)
        predictions.append(1 if p1>p0 else 0)
        scores.append(p1)
    return np.array(predictions),np.array(scores)

X_train_50,y_train_50 = generate_data(50)
X_train_500,y_train_500 = generate_data(500)
X_train_5000,y_train_5000 = generate_data(5000)
X_validate,y_validate = generate_data(10000)

print(f"Training sets: {len(X_train_50)}, {len(X_train_500)}, {len(X_train_5000)}")
print(f"Validation set: {len(X_validate)}\n")

y_pred_optimal,scores_optimal = optimal_classifier(X_validate)
error_optimal = np.mean(y_pred_optimal!=y_validate)
print(f"Optimal classifier error: {error_optimal:.4f}\n")

fpr,tpr,thresholds = roc_curve(y_validate,scores_optimal)
roc_auc = auc(fpr,tpr)
threshold_idx = np.argmin(np.abs(thresholds-0.5))
print(f"using tpr and fpr for min(p_err) = p(L=0)fpr+p(L=1)(1-tpr) = {prior_L0*fpr[threshold_idx] + prior_L1*(1-tpr[threshold_idx])}\n")

plt.figure(figsize=(10,8))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot(fpr[threshold_idx], tpr[threshold_idx], 'r*', markersize=15, 
         label='Min P(error) point')
plt.xlabel('False Positive Rate',fontsize=12)
plt.ylabel('True Positive Rate',fontsize=12)
plt.title('ROC Curve for Optimal Classifier',fontsize=14)
plt.legend(fontsize=10)
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('Assignment 2/outputs/q1_part1_roc.png', dpi=150, bbox_inches='tight')

# Logistic Regression

def sigmoid(z):
    return 1/(1+np.exp(-z))

def logistic_linear_features(X):
    return np.column_stack([np.ones(len(X)),X])

def logistic_quadratic_features(X):
    return np.column_stack([np.ones(len(X)), 
        X[:,0], X[:,1],
        X[:,0]**2,X[:,0]*X[:,1], X[:,1]**2
    ])

def log_likelihood(w,Z,y):
    predictions = sigmoid(Z@w)
    epsilon = 1e-15
    predictions = np.clip(predictions,epsilon,1-epsilon)
    # cost function of logistic regression
    return -np.sum(y*np.log(predictions)+(1-y)*np.log(1-predictions))

def train_logistic_model(X_train,y_train,feature_func):
    Z = feature_func(X_train)
    w_init = np.zeros(Z.shape[1])
    result = minimize(log_likelihood,w_init,args=(Z,y_train),
                     method='BFGS', options={'maxiter': 1000})
    return result.x

def evaluate_logistic_model(w,X,y,feature_func):
    Z = feature_func(X)
    predictions_prob = sigmoid(Z@w)
    predictions = (predictions_prob>=0.5).astype(int)
    error = np.mean(predictions!= y)
    return error,predictions_prob

w_linear_50 = train_logistic_model(X_train_50, y_train_50,logistic_linear_features)
w_linear_500 = train_logistic_model(X_train_500, y_train_500,logistic_linear_features)
w_linear_5000 = train_logistic_model(X_train_5000, y_train_5000,logistic_linear_features)
error_linear_50,_ = evaluate_logistic_model(w_linear_50,X_validate,y_validate,logistic_linear_features)
error_linear_500,_ = evaluate_logistic_model(w_linear_500,X_validate,y_validate,logistic_linear_features)
error_linear_5000,_ = evaluate_logistic_model(w_linear_5000,X_validate,y_validate,logistic_linear_features)
print(f"Linear model (N=50) error: {error_linear_50:.4f}")
print(f"Linear model (N=500) error: {error_linear_500:.4f}")
print(f"Linear model (N=5000) error: {error_linear_5000:.4f}\n")

w_quad_50 = train_logistic_model(X_train_50,y_train_50,logistic_quadratic_features)
w_quad_500 = train_logistic_model(X_train_500,y_train_500,logistic_quadratic_features)
w_quad_5000 = train_logistic_model(X_train_5000,y_train_5000,logistic_quadratic_features)
error_quad_50,_ = evaluate_logistic_model(w_quad_50,X_validate,y_validate,logistic_quadratic_features)
error_quad_500,_ = evaluate_logistic_model(w_quad_500,X_validate,y_validate,logistic_quadratic_features)
error_quad_5000,_ = evaluate_logistic_model(w_quad_5000,X_validate,y_validate,logistic_quadratic_features)
print(f"Quadratic model (N=50) error: {error_quad_50:.4f}")
print(f"Quadratic model (N=500) error: {error_quad_500:.4f}")
print(f"Quadratic model (N=5000) error: {error_quad_5000:.4f}")

def plot_decision_boundary(w,feature_func,X,y,title,filename):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,200),
                         np.linspace(y_min,y_max,200))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = feature_func(grid_points)
    Z_pred = sigmoid(Z@w).reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx,yy,Z_pred,levels=[0, 0.5, 1], alpha=0.3, colors=['blue','red'])
    plt.contour(xx,yy,Z_pred,levels=[0.5], colors='black', linewidths=2)
    plt.scatter(X[y==0,0], X[y==0,1], c='blue', marker='o', s=20, alpha=0.6, label='Class 0')
    plt.scatter(X[y==1,0], X[y==1,1], c='red', marker='s', s=20, alpha=0.6, label='Class 1')
    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename,dpi=150,bbox_inches='tight')

plot_decision_boundary(w_quad_5000, logistic_quadratic_features, 
                      X_validate[:1000], y_validate[:1000],
                      'Quadratic Model (N=5000) Decision Boundary',
                      'Assignment 2/outputs/q1_part2_quad_5000_boundary.png')