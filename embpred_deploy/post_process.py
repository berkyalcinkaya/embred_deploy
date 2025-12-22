import numpy as np


def monotonic_decoding(probabilities, loss='NLL'):
    """
    Enforces progressive class labeling across T timepoints with constraints:
    - Rule 1: From class k at timepoint t, valid classes at t+1 are {k, k+1}
    - Rule 2: Timepoint 0 is always class 0
    
    Uses dynamic programming to find the optimal path that maximizes probability
    while respecting these constraints.

    Parameters
    ----------
    probabilities : np.ndarray of shape (T, k)
        probabilities[t, i] = predicted probability of class i at timepoint t
    loss : str, optional
        Either 'NLL' (Negative Log Likelihood) or 'EM' (Earth Mover/Wasserstein-1).
        Defaults to 'NLL'.

    Returns
    -------
    path : np.ndarray of shape (T,)
        The optimal class index at each timepoint (0-indexed).
        Class indices are in [0, k-1].
    """
    # Basic validation
    if loss not in ['NLL', 'EM']:
        raise ValueError("loss must be either 'NLL' or 'EM'")
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array of shape (T, k)")

    T, k = probabilities.shape

    # Precompute costs: cost[t, i] = negative log probability of class i at timepoint t
    cost = np.zeros((T, k), dtype=np.float64)
    eps = 1e-12

    for t in range(T):
        for i in range(k):
            if loss == 'NLL':
                p = probabilities[t, i]
                cost[t, i] = -np.log(max(p, eps))
            else:  # EM loss
                distances = np.abs(np.arange(k) - i)
                cost[t, i] = np.sum(probabilities[t] * distances)

    # DP table: dp[t, i] = minimum total cost to reach class i at timepoint t
    dp = np.full((T, k), np.inf, dtype=np.float64)
    backptr = np.full((T, k), -1, dtype=np.int32)

    # Rule 2: Timepoint 0 is always class 0
    dp[0, 0] = cost[0, 0]
    backptr[0, 0] = 0

    # Fill DP table
    # Rule 1: From class j at t-1, can only go to {j, j+1} at t
    # So to reach class i at t, we must have come from {i-1, i} at t-1
    for t in range(1, T):
        for i in range(k):
            # Valid previous states: i-1 (if exists) and i
            candidates = []
            if i > 0:  # Can come from class i-1
                candidates.append((dp[t-1, i-1], i-1))
            # Can come from same class i
            candidates.append((dp[t-1, i], i))
            
            # Choose the minimum cost path
            min_val = np.inf
            min_j = -1
            for val, j in candidates:
                if val < min_val:
                    min_val = val
                    min_j = j
            
            if min_j >= 0:  # Valid path exists
                dp[t, i] = cost[t, i] + min_val
                backptr[t, i] = min_j

    # Find optimal final class
    final_class = np.argmin(dp[T-1, :])

    # Backtrack to retrieve the optimal path
    path = np.zeros(T, dtype=np.int32)
    path[T-1] = final_class
    for t in range(T-2, -1, -1):
        path[t] = backptr[t+1, path[t+1]]

    return path
