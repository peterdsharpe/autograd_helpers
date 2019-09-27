import autograd.numpy as np

def index_update(x, idx, y):
    """Pure equivalent of :code:`x[idx] = y`.

    Returns the value of `x` that would result from the
    NumPy-style :mod:`indexed assignment <numpy.doc.indexing>`::
      x[idx] = y

    Example usage for vector: index_update(v, [2], 5)
    Example usage for matrix: index_update(A, [2,3], 5)

    Note the `index_update` operator is pure; `x` itself is
    not modified, instead the new value that `x` would have taken is returned.

    Unlike NumPy's :code:`x[idx] = y`, if multiple indices refer to the same
    location it is undefined which update is chosen; JAX may choose the order of
    updates arbitrarily and nondeterministically (e.g., due to concurrent
    updates on some hardware platforms).

    Args:
      x: an array with the values to be updated.
      idx: a Numpy-style index, consisting of `None`, integers, `slice` objects,
        ellipses, ndarrays with integer dtypes, or a tuple of the above. A
        convenient syntactic sugar for forming indices is via the
        :data:`jax.ops.index` object. Must be an iterable.
      y: the array of updates. `y` must be broadcastable to the shape of the
        array that would be returned by `x[idx]`.

    Returns:
      An array.
      """
    flattened_idx = np.ravel_multi_index(idx, x.shape)
    np.put(x, flattened_idx, y, mode='raise')