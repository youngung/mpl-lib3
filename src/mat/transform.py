import numpy as np

def trans_2ndtensor(aold: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Transform a second-rank tensor using coordiante transformation matrix (r)

    Parameters
    ----------
    aold : np.ndarray
        The original 2nd-rank tensor (shape: (n, n)).
    r : np.ndarray
        The transformation matrix (shape: (n, n)).

    Returns
    -------
    np.ndarray
        The transformed tensor (shape: (n, n)), computed as:
        anew[i, j] = r[i, k] * r[j, l] * aold[k, l]
    """
    # Validate input shapes
    if aold.ndim != 2 or r.ndim != 2:
        raise ValueError("Both aold and r must be 2D arrays.")
    if aold.shape != r.shape:
        raise ValueError("aold and r must have the same shape.")

    # Perform the tensor transformation
    return r @ aold @ r.T


def trans_4thtensor(aold: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Transform a fourth-rank tensor using coordinate transformation matrix (r)

    Parameters
    ----------
    aold : np.ndarray
        The original 4th-rank tensor (shape: (n, n, n, n)).
    r : np.ndarray
        The transformation matrix (shape: (n, n)).

    Returns
    -------
    np.ndarray
        The transformed tensor (shape: (n, n, n, n)), computed as:
        anew[i, j, k, l] = r[i, p] * r[j, q] * r[k, r_] * r[l, s] * aold[p, q, r_, s]
    """
    if aold.ndim != 4 or r.ndim != 2:
        raise ValueError("aold must be 4D and r must be 2D arrays.")
    n = r.shape[0]
    if aold.shape != (n, n, n, n) or r.shape != (n, n):
        raise ValueError("aold must have shape (n, n, n, n) and r must have shape (n, n).")

    # Use einsum for efficient tensor transformation
    return np.einsum('ip,jq,kr,ls,pqrs->ijkl', r, r, r, r, aold)