from sklearn_rvm.rvm import RVC
from sklearn.datasets import load_iris
from abc import ABCMeta, abstractmethod
from collections import deque

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.special import expit # check if we want to impor tmore stuff



def _calculate_statistics(K, alpha_values, included_cond, y, sigma):

    
STEP_MIN = 1/2e8
GRAD_MIN = 1e-6
MAX_ITS = 100 # how to allow to change this?

iris = load_iris()
X = iris.data[:100]
y = iris.target[:100]

X, y = check_X_y(X, y, y_numeric=True, ensure_min_samples=2)

n_samples = X.shape[0]
K = pairwise_kernels(X, metric='linear', filter_params=True) # linear?
K = np.hstack((np.ones((n_samples, 1)), K))

INFINITY = 1e20
EPSILON = 1e-9

alpha_values = np.zeros(n_samples + 1) + INFINITY
included_cond = np.zeros(n_samples + 1, dtype=bool)

selected_basis = 0
Phi = K[:, selected_basis] 
t_hat = 2*y - 1 # PseudoLinear Target {-1.1}
logout = (t_hat*0.9 + 1) / 2
if len(Phi.shape)==1:
    Phi = Phi[:,np.newaxis]
    logout = logout[:,np.newaxis]

mu, _, _, _ = np.linalg.lstsq(Phi, np.log(logout/(1-logout))) # TODO: least squares solution. np.log or math.log?
mask_mu = mu < EPSILON # TODO: confirm if correct


alpha_values[selected_basis] = 1 / (mu + mu[mask_mu])**2

included_cond[selected_basis] = True

#Sigma, mu, s, q, Phi = self._calculate_statistics(K, alpha_values, included_cond, y, mu)
####
mu_mp = mu

n_samples = y.shape[0]
error_log = []

A = np.diag(alpha_values[included_cond])
Phi = K[:, included_cond] # e o bias?
M = Phi.shape[1]

def DataErr( Phi, mu_mp, y):
    t_hat = expit(Phi @ mu_mp) # prediction of the output. Change? might be confusing
    t_hat0 = t_hat == 0
    t_hat1 = t_hat == 1

    if any(t_hat0[y>0]) or any(t_hat1[y<1]):
        data_err = INFINITY
    else:
        # error is calculated through cross-entropy
        data_err = - np.sum(y*np.log(t_hat+EPSILON)) #TODO: shouldnt I divide by n_trials?
    return t_hat, data_err

t_hat, data_err = DataErr(Phi, mu_mp, y)
reg = A.T @ mu_mp**2 / 2
total_err = data_err + reg
error_log.append(total_err) # Check if cant be scalar

for i in range(MAX_ITS):
    # Calculate the error of predictions and its gradient
    e = y[:,None] - t_hat
    g = Phi.T @ e - A * mu_mp
    # Calculate B - likelihoood dependent analogue of the noise precision 
    B = t_hat * (1-t_hat) # call it B?

    # Compute the Hessian
    tmp = Phi * (B * np.ones((1, M)))
    H = tmp.T @ Phi + A
    # Invert Hessian via Cholesky - lower triangular Cholesky factor of H.
    # Must be positive definite. Check exception
    U = np.linalg.cholesky(H)

    # Check for termination based on the Gradient
    if all(abs(g))<GRAD_MIN:
        break

    # Calculate Newton Step: H^-1 * g
    delta_mu = np.linalg.lstsq(U, np.linalg.lstsq(U.T, g)[0])[0]
    step = 1

    while step>STEP_MIN:
        mu_new = mu + step*delta_mu
        tmp = Phi @ mu_new
        t_hat, data_err = DataErr(Phi, mu_new, y)
        reg = A.T @ mu_new**2 / 2
        total_err = data_err + reg

        # Check if error increased
        if total_err >= error_log[-1]:
            step /= 2
        else:
            mu_mp = mu_new
            step = 0 # to leave the while loop

    # Compute covariance approximation
    Ui = np.linalg.inv(U)
    Sigma = Ui @ Ui.T

    # Update s and q
    tmp_1 = K.T @ (Phi * (B * np.ones((1,M))))
    S = (B.T @ K**2).T - np.sum((tmp_1 @ Ui)**2, axis=1)[:,np.newaxis]
    Q = K.T @ e

    s = S
    q = Q

    s[included_cond] = (A * S[included_cond]) / (A - S[included_cond])
    q[included_cond] = (A * Q[included_cond]) / (A - Q[included_cond])

# 3. Initialize Sigma, q and s for all bases

# Start updating the model iteratively
# Create queue with indices to select candidates for update
queue = deque(list(range(n_samples + 1)))
max_iter = 100
for epoch in range(max_iter):          
    # 4. Pick a candidate basis vector from the start of the queue and put it at the end
    basis_idx = queue.popleft()
    queue.append(basis_idx)

    # 5. Compute theta
    theta = q ** 2 - s


    current_alpha_values = np.copy(alpha_values)
    current_included_cond = np.copy(included_cond)

    # 6. Re-estimate included alpha
    if theta[basis_idx] > 0 and current_alpha_values[basis_idx] < INFINITY:
        alpha_values[basis_idx] = s[basis_idx] ** 2 / (q[basis_idx] ** 2 - s[basis_idx])

    # 7. Add basis function to the model with updated alpha
    elif theta[basis_idx] > 0 and current_alpha_values[basis_idx] >= INFINITY:
        alpha_values[basis_idx] = s[basis_idx] ** 2 / (q[basis_idx] ** 2 - s[basis_idx])
        included_cond[basis_idx] = True

    # 8. Delete theta basis function from model and set alpha to infinity
    elif theta[basis_idx] <= 0 and current_alpha_values[basis_idx] < INFINITY:
        alpha_values[basis_idx] = INFINITY
        included_cond[basis_idx] = False


    #Sigma, mu, s, q, Phi = self._calculate_statistics(K, alpha_values, included_cond, y, mu)
    mu_mp = mu

    n_samples = y.shape[0]
    error_log = []

    A = np.diag(alpha_values[included_cond])
    Phi = K[:, included_cond] # e o bias?
    M = Phi.shape[1]

    def DataErr( Phi, mu_mp, y):
        t_hat = expit(Phi @ mu_mp) # prediction of the output. Change? might be confusing
        t_hat0 = t_hat == 0
        t_hat1 = t_hat == 1

        if any(t_hat0[y>0]) or any(t_hat1[y<1]):
            data_err = INFINITY
        else:
            # error is calculated through cross-entropy
            data_err = - np.sum(y*np.log(t_hat+EPSILON)) #TODO: shouldnt I divide by n_trials?
        return t_hat, data_err

    t_hat, data_err = DataErr(Phi, mu_mp, y)
    reg = A.T @ mu_mp**2 / 2
    total_err = data_err + reg
    error_log.append(total_err) # Check if cant be scalar

    for i in range(MAX_ITS):
        # Calculate the error of predictions and its gradient
        e = y[:,None] - t_hat
        g = Phi.T @ e - A * mu_mp
        # Calculate B - likelihoood dependent analogue of the noise precision 
        B = t_hat * (1-t_hat) # call it B?

        # Compute the Hessian
        tmp = Phi * (B * np.ones((1, M)))
        H = tmp.T @ Phi + A
        # Invert Hessian via Cholesky - lower triangular Cholesky factor of H.
        # Must be positive definite. Check exception
        U = np.linalg.cholesky(H)

        # Check for termination based on the Gradient
        if all(abs(g))<GRAD_MIN:
            break

        # Calculate Newton Step: H^-1 * g
        delta_mu = np.linalg.lstsq(U, np.linalg.lstsq(U.T, g)[0])[0]
        step = 1

        while step>STEP_MIN:
            mu_new = mu + step*delta_mu
            tmp = Phi @ mu_new
            t_hat, data_err = DataErr(Phi, mu_new, y)
            reg = A.T @ mu_new**2 / 2
            total_err = data_err + reg

            # Check if error increased
            if total_err >= error_log[-1]:
                step /= 2
            else:
                mu_mp = mu_new
                step = 0 # to leave the while loop

        # Compute covariance approximation
        Ui = np.linalg.inv(U)
        Sigma = Ui @ Ui.T

        # Update s and q
        tmp_1 = K.T @ (Phi * (B * np.ones((1,M))))
        S = (B.T @ K**2).T - np.sum((tmp_1 @ Ui)**2, axis=1)[:, np.newaxis]
        Q = K.T @ e

        s = S
        q = Q

        s[included_cond] = (A * S[included_cond]) / (A - S[included_cond])
        q[included_cond] = (A * Q[included_cond]) / (A - Q[included_cond])













