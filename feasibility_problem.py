import numpy as np
import cvxpy as cp


def functionFeasibilityProblem_cvx(
    H,
    gamma,
    Pmax,
    sigma2=1
):
    """
    SINR Feasibility Problem using CVXPY
    """

    # Dimensions
    N, K = H.shape

    # Beamforming variable
    W = cp.Variable((N, K), complex=True)

    constraints = []

    # SINR constraints
    for k in range(K):

        hk = H[:, k]

        # Desired signal
        desired_signal = cp.real(
            hk.conj().T @ W[:, k]
        )

        # Interference vector
        interference_terms = []

        for i in range(K):

            if i != k:

                interference_terms.append(
                    hk.conj().T @ W[:, i]
                )

        # Interference + noise
        if len(interference_terms) > 0:

            interference_vector = cp.hstack(
                interference_terms
            )

            rhs = cp.norm(
                cp.hstack([
                    interference_vector,
                    np.sqrt(sigma2)
                ]),
                2
            )

        else:

            rhs = np.sqrt(sigma2)

        # SINR SOC constraint
        constraints.append(
            desired_signal >=
            np.sqrt(gamma[k]) * rhs
        )

    # Total transmit power constraint
    constraints.append(
        cp.sum_squares(cp.abs(W)) <= Pmax
    )

    # Optimization problem
    problem = cp.Problem(
        cp.Minimize(0),
        constraints
    )

    # Solve problem
    try:

        problem.solve(
            solver=cp.SCS,
            verbose=False,
            eps=1e-4,
            max_iters=5000
        )

    except Exception as e:

        print("Solver Error:", e)

        return False, None

    # Feasibility check
    feasible = (
        problem.status == cp.OPTIMAL
        or
        problem.status == cp.OPTIMAL_INACCURATE
    )

    # Return results
    if feasible:

        return feasible, W.value

    else:

        return False, None
