

import numpy as np
import cvxpy as cp


def functionFeasibilityProblem_cvx(H, gamma, Pmax, sigma2=1):
    """
    SINR Feasibility Problem using CVXPY

    Parameters
    ----------
    H : ndarray (N x K)
        Channel matrix

    gamma : ndarray
        Target SINR values

    Pmax : float
        Maximum transmit power

    sigma2 : float
        Noise variance

    Returns
    -------
    feasible : bool
        True if feasible

    W_opt : ndarray
        Optimal beamforming matrix
    """

    # Dimensions
    N, K = H.shape

    # Complex beamforming matrix
    W = cp.Variable((N, K), complex=True)

    constraints = []

    # SINR constraints
    for k in range(K):

        hk = H[:, k]

        # Desired signal
        desired_signal = cp.real(
            hk.conj().T @ W[:, k]
        )

        # Interference terms
        interference_terms = []

        for i in range(K):

            if i != k:

                interference_terms.append(
                    hk.conj().T @ W[:, i]
                )

        # Stack interference + noise
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

        # SOC SINR constraint
        constraints.append(
            desired_signal >= np.sqrt(gamma[k]) * rhs
        )

    # Total transmit power constraint
    constraints.append(
        cp.sum(cp.abs(W) ** 2) <= Pmax
    )

    # Feasibility optimization
    problem = cp.Problem(
        cp.Minimize(0),
        constraints
    )

    # Solve using CLARABEL
    try:

        problem.solve(
            solver=cp.CLARABEL,
            verbose=False
        )

    except:

        # Fallback solver
        problem.solve(
            solver=cp.SCS,
            verbose=False
        )

    # Check feasibility
    feasible = problem.status in [
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE
    ]

    # Return solution
    if feasible:
        return feasible, W.value
    else:
        return feasible, None
