import numpy as np


def get_covariance_matrix(N, tau, noise):

    # This function helps you obtain a symmetric positive definite
    # covariance matrix which can be used with your CG Algorithm.
    # It will also produce a rhs vector b

    # Inputs
    # N = Size of matrix A you wish to obtain
    # tau = Matern kernel hyper-parameter
    # noise = additional added noise

    x = np.linspace(0, N, N)
    X, Y = np.meshgrid(x, x)
    A = (1 + np.sqrt(3) * np.abs(X - Y) / tau) * np.exp(-np.sqrt(3) * np.abs(X - Y) / tau)
    A = (A > 10 * np.finfo(float).eps) * A
    A = A + noise * np.eye(N)
    N = A.shape[0]
    b = np.random.randn(N)

    return A, b