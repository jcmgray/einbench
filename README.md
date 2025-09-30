# einbench

This is a random pairwise tensor contraction generator for benchmarking
and verifying einsum implementations. It includes two datasets:

1. `contractions_benchmark.txt`: A set of ~1000 random contractions with
   operation count varying between $\sim10^0$ and $\sim10^9$. Half include
   batch indices and half do not (are matrix multiplication equivalents).

2. `contractions_verify.txt`: A set of ~1000 random contractions with
   significantly lower operation count (up to $\sim10^6$) for verifying
   correctness of implementations. These include a variety of edge cases such
   as trivially summed, traced, or diagonal indices.

Both are generated with the same code, which is included in this repository
as `einbench.py`. The main function is:

```python
def random_pairwise_contraction(
    cost,
    seed=None,
    p_lkept=1.0,
    p_rkept=1.0,
    p_con=1.0,
    allow_scalar=True,
    allow_outer=True,
    allow_batch=True,
    p_batch=0.25,
    allow_sum=False,
    p_sum=0.25,
    allow_trace=False,
    p_trace=0.05,
    num_indices_center=4,
    num_indices_concentration=1.5,
    num_indices=None,
):
    """Generate a random pairwise einsum contraction with an approximate
    target computational cost. The process is as follows:

    1. Choose the number of distinct indices to use in the contraction
       (between 1 and log2(cost)). The distribution is controlled by
       `num_indices_center` and `num_indices_concentration` which define
       a beta distribution over the range.
    2. Randomly distribute the total cost among the indices, to define
       their sizes.
    3. Adjust the sizes downwards until the actual cost is <= target cost.
    4. Randomly assign each index to one of several types, which define
       whether they appear on the left, right, and/or output.
    5. Adjust to disallow scalars or pure outer products if required.
    6. Randomly permute the indices on each term.
    7. Return the einsum input and output strings, and the index size dict.

    Parameters
    ----------
    cost : float or int
        Target computational cost (in scalar operations) for the contraction.
    seed : int, optional
        Random seed for reproducibility, supplied to numpy RandomState.
    p_lkept : float, optional
        Relative probability of choosing an index that appears on the left
        and in the output (default: 1.0).
    p_rkept : float, optional
        Relative probability of choosing an index that appears on the right
        and in the output (default: 1.0).
    p_con : float, optional
        Relative probability of choosing an index that appears on both
        left and right and is contracted (default: 1.0).
    allow_scalar : bool, optional
        Whether to allow contractions where one term is a scalar
        (default: True).
    allow_outer : bool, optional
        Whether to allow contractions that are pure outer products.
    allow_batch : bool, optional
        Whether to allow batch indices (default: True).
    p_batch : float, optional
        Relative probability of choosing an index that appears on both
        left and right and in the output (default: 0.25). Only used if
        `allow_batch` is True.
    allow_sum : bool, optional
        Whether to allow indices that appear only on one side (default: False).
        Sum indices can always be trivially removed (summed over) prior to
        contraction.
    p_sum : float, optional
        Relative probability of choosing an index that appears only on
        the on one of the inputs (default: 0.25). Only used if `allow_sum`.
    allow_trace : bool, optional
        Whether to allow trace indices that appear twice on one side
        (default: False). Trace indices can always be trivially removed
        (summed over or diagonal taken) prior to contraction.
    p_trace : float, optional
        Absolute probability that any index is repeated, if `allow_trace`
        is True (default: 0.05).
    num_indices_center : float, optional
        Peak of the distribution for the number of distinct indices
        to use in the contraction (default: 4).
    num_indices_concentration : float, optional
        Concentration of the distribution for the number of distinct indices
        to use in the contraction (default: 1.5). Higher values lead to
        more concentration around `num_indices_center`. 1.0 gives a flat
        distribution.
    num_indices : int, optional
        Number of distinct indices to use in the contraction (default: None).
        If None, this is chosen randomly between 1 and log2(cost), using a
        beta distribution with center and concentration as specified above.

    Returns
    -------
    inputs : tuple[tuple[str, ...], tuple[str, ...]]
        The einsum input strings (lhs, rhs).
    output : tuple[str, ...]
        The einsum output string.
    size_dict : dict[str, int]
        Dictionary mapping index labels to their sizes.
    """
    ...
```