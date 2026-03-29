import numpy as np


def conjugate_gradient(A, b, tol, max_iters=1000):
    
    n = len(b)

    # Initial guess
    x = np.zeros(n)

    # Compute initial residual
    r = b - A @ x
    # Compute search direction
    p = r.copy()
    # Compute initial r^Tr
    rr = np.dot(r, r)
    # Compute norm of b
    b_norm = np.linalg.norm(b)

    # For tracking relative residual norm at each iteration
    res_history = [np.sqrt(rr) / b_norm]

    i = 0
    while np.sqrt(rr) > tol and i < max_iters:

        # Compute A times search direction p
        Ap = A @ p

        # Compute step size
        alpha = rr / np.dot(Ap, p)

        # Update solution x
        x = x + alpha * p

        # Update residual r
        r = r - alpha * Ap

        # Compute beta
        rr_new = np.dot(r, r)
        beta = rr_new / rr

        # Update search direction
        p = r + beta * p

        # Setup for next iteration
        rr = rr_new
        i += 1

        # Record value of residual
        res_history.append(np.sqrt(rr) / b_norm)

    
    if i == max_iters:
        print(f"CG did not converge: reached max iterations of ({max_iters})")
    else:
        print(f"CG converged in {i} iterations")
        
    return x, i, res_history

    