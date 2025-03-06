import numpy as np


def monotonic_decoding(probabilities, loss='NLL'):
    """
    Enforces a non-decreasing (monotonic) labeling across T frames, 
    each having a probability distribution over k classes.

    Parameters
    ----------
    probabilities : np.ndarray of shape (T, k)
        probabilities[t, i] = predicted probability of class i at frame t
    loss : str, optional
        Either 'NLL' (Negative Log Likelihood) or 'EM' (Earth Mover/Wasserstein-1).
        Defaults to 'NLL'.

    Returns
    -------
    path : list of length T
        The optimal (monotonically non-decreasing) class index at each frame.
        Class indices are in [0, k-1].
    """
    print(probabilities.shape)
    # Basic checks
    if loss not in ['NLL', 'EM']:
        raise ValueError("loss must be either 'NLL' or 'EM'")
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array of shape (T, k)")

    T, k = probabilities.shape

    # Precompute the per-frame cost for each class, to speed up DP loops.
    # cost[t, i] = e(i, probabilities[t]) for whichever loss is chosen
    cost = np.zeros((T, k), dtype=np.float64)

    # A small epsilon to avoid log(0). Alternatively, one can handle zero-prob as inf cost.
    eps = 1e-12

    for t in range(T):
        for i in range(k):
            if loss == 'NLL':
                # -log p(t,i)
                p = probabilities[t, i]
                if p < eps:
                    cost[t, i] = -np.log(eps)  # or float('inf') if you prefer
                else:
                    cost[t, i] = -np.log(p)
            else:
                # Earth Mover distance to class i
                # sum_{m=0..k-1} p[t, m] * |i - m|
                # Here, we treat i, m as 0-based indices
                distances = np.abs(np.arange(k) - i)
                cost[t, i] = np.sum(probabilities[t] * distances)

    # Initialize DP table and back-pointer
    # dp[t, i] = minimum total cost up to frame t if we choose class i at frame t
    dp = np.full((T, k), np.inf, dtype=np.float64)
    backptr = np.full((T, k), -1, dtype=np.int32)

    # Base case: for t = 0, dp[0, i] = cost of choosing i at first frame
    dp[0, :] = cost[0, :]

    # Fill DP table
    for t in range(1, T):
        for i in range(k):
            # We want y^{t} = i, with y^{t-1} <= i
            # => we pick the best j in [0..i]
            # dp[t, i] = cost[t, i] + min_{j in [0..i]} dp[t-1, j]
            # We'll do a simple linear scan for j <= i
            min_val = np.inf
            min_j = -1
            for j in range(i+1):  # j goes from 0 to i
                val = dp[t-1, j]
                if val < min_val:
                    min_val = val
                    min_j = j
            dp[t, i] = cost[t, i] + min_val
            backptr[t, i] = min_j

    # Now find the minimal cost in the last frame
    final_class = np.argmin(dp[T-1, :])
    min_cost = dp[T-1, final_class]

    # Backtrack to retrieve the path
    path = [0] * T
    path[T-1] = final_class
    for t in range(T-2, -1, -1):
        path[t] = backptr[t+1, path[t+1]]

    return path
