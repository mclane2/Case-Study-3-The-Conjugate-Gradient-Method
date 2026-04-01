function [x, i, res_history] = conjugate_gradient(A, b, tol, max_iters)
    if nargin < 4
        max_iters = 1000;
    end

    n = length(b);
    x = zeros(n, 1);
    r = b - A * x;
    p = r;
    rr = dot(r, r);
    b_norm = norm(b);
    res_history = [sqrt(rr) / b_norm];
    i = 0;

    while sqrt(rr) > tol && i < max_iters
        Ap = A * p;
        alpha = rr / dot(Ap, p);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rr_new = dot(r, r);
        beta = rr_new / rr;
        p = r + beta * p;
        rr = rr_new;
        i = i + 1;
        res_history = [res_history; sqrt(rr) / b_norm];
    end

    if i == max_iters
        fprintf('CG did not converge: reached max iterations of (%d)\n', max_iters);
    else
        fprintf('CG converged in %d iterations\n', i);
    end
end