import numpy as np
import scipy as sp
from scipy import optimize


class KernelSVC :

    def __init__(self, C, kernel, epsilon=1e-3, tol = 1e-2) :
        self.type = 'non-linear'
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
        self.beyond_margin = None
        self.tol = tol
        self.mean_score = 0
        self.std_score = 1


    def fit(self, X, y, verbose = False, class_weights = None, precomputed_K = None) :
        #### You might define here any variable needed for the rest of the code
        N = len(y)
        d = np.diag(y)
        if precomputed_K is None:
            K = self.kernel(X, X)
        else :
            K = precomputed_K
        print('Kernel computed')
        # Constraints variables for the inequality constraint
        A = np.kron(np.array([[-1], [1]]), np.eye(N))
        s = np.kron(np.array([self.C, 0]), np.ones(N))
        if class_weights is not None :
            s[:N] *= (y == 1)*class_weights[0] + (y == -1)*class_weights[1]
        print("s[:N] contains ", np.unique(s[:N]))

        # Lagrange dual problem
        def loss(alpha) :
            loss_ = 1 / 2 * alpha.T @ d @ K @ d @ alpha - np.sum(alpha)
            return loss_

        # Partial derivate of Ld on alpha
        def grad_loss(alpha) :
            grad = d @ K @ d @ alpha - np.ones_like(alpha)
            return grad

        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  s - A*alpha >= 0

        fun_eq = lambda \
            alpha : y.T @ alpha  # '''----------------function defining the equality constraint------------------'''
        jac_eq = lambda \
            alpha : y  # '''----------------jacobian wrt alpha of the  equality constraint------------------'''
        fun_ineq = lambda \
            alpha : s + A @ alpha  # '''---------------function defining the ineequality constraint-------------------'''
        jac_ineq = lambda \
            alpha : A  # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''

        constraints = ({'type' : 'eq', 'fun' : fun_eq, 'jac' : jac_eq},
                       {'type' : 'ineq',
                        'fun' : fun_ineq,
                        'jac' : jac_ineq})
        print('Optimisation starting')
        optRes = optimize.minimize(fun=lambda alpha : loss(alpha),
                                   x0=np.ones(N),
                                   method='SLSQP',
                                   jac=lambda alpha : grad_loss(alpha),
                                   constraints=constraints,
                                   tol=self.tol,
                                   options={'disp' : verbose, 'iprint' : 2})
        print('End optimisation ', optRes)

        self.alpha = d @ optRes.x
        ## Assign the required attributes
        supportIndices = np.argwhere(
            (np.abs(self.alpha) > self.epsilon) * (np.abs(self.alpha) < self.C - self.epsilon)).flatten()
        marginIndices = np.argwhere((np.abs(self.alpha) > self.epsilon)).flatten()

        self.support = X[
            supportIndices]  # '''------------------- A matrix with each row corresponding to a support vector ------------------'''
        self.b = np.mean(y[supportIndices] - (K @ self.alpha)[
            supportIndices])  # ''' -----------------offset of the classifier------------------ '''
        self.norm_f = self.alpha.T @ K @ self.alpha  # '''------------------------RKHS norm of the function f ------------------------------'''

        # Only keep the indices where alpha is not zero
        self.beyond_margin = X[marginIndices]
        self.alpha = self.alpha[marginIndices]



    ### Implementation of the separting function f
    def separating_function(self, x) :
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return self.kernel(x, self.beyond_margin) @ self.alpha + self.b

    def predict(self, X) :
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d > 0) - 1