import numpy as np
from numpy.linalg import inv, eigh

class KernelFDA:
    def __init__(self, n_components, kernel):
        self.kernel = kernel
        self.p = n_components

    def fit(self, X, y):
        self.X_fit = X

        n = len(y)
        n1 = np.sum(y)
        n0 = n - n1
        
        # Compute kernel matrix
        hXX = self.kernel(X, X)

        # Compute mean vectors
        M1 = 1/n1 * (hXX @ y)
        M0 = 1/n0 * (hXX @ (np.ones(n) - y))

        # Compute between-class scatter matrix
        M = np.outer((M1 - M0), (M1 - M0))

        # Compute within-class scatter matrix
        K1 = hXX[:, y == 1]
        K0 = hXX[:, y == 0]
        H1 = np.eye(n1) - 1/n1 * np.ones((n1, n1))
        H0 = np.eye(n0) - 1/n0 * np.ones((n0, n0))
        N = K1.dot(H1).dot(K1.T) + K0.dot(H0).dot(K0.T)

        epsilon = 0.0001
        # Add a small epsilon to prevent singularity of matrix N
        sigm = inv(N + epsilon * np.eye(X.shape[0])).dot(M)

        # Compute eigenvectors and eigenvalues
        eig_val, eig_vec = eigh(sigm)
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]

        # Extract the first p eigenvectors
        self.A = eig_vec[:, :self.p]

    def transform(self, X):
        # Compute transformed data using kernel matrix and eigenvectors
        K = self.kernel(X, self.X_fit)
        return K @ self.A

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        # Make predictions using transformed data
        # Note: This implementation assumes n_components = 1 for simplicity
        y_pred = np.where(self.transform(X) > 0, 1, 0)
        return y_pred
    