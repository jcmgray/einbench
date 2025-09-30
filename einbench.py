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
    target computational cost.

    Parameters
    ----------
    cost : float|int
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
    import string
    import numpy as np

    # for consistency across versions/platforms

    if seed is not None:
        seed = np.array([
            seed,
            cost,
            allow_scalar,
            allow_outer,
            allow_batch,
            allow_sum,
            allow_trace,
        ], dtype="int32")

    rng = np.random.RandomState(seed)

    # choose how many indices to use in our contraction
    # assume smallest indices will have size at least 2
    mn = 1
    mx = int(np.log2(cost))
    if num_indices is None:
        if mx <= 2:
            # tiny contraction - choose uniformly
            num_indices = rng.randint(mn, mx + 1)
        else:
            # turn on the concentration slowly for very small
            # contractions where there are not many potential indices
            if mx <= 2 * num_indices_center:
                # flat
                num_indices_concentration = min(num_indices_concentration, 1)
            elif mx <= 3 * num_indices_center:
                # half way to flat
                num_indices_concentration = min(
                    num_indices_concentration, (num_indices_concentration + 1) / 2
                )

            # sample from a beta distribution
            center = min(max(0.0, (num_indices_center - 1) / (mx - mn)), 1.0)
            a = 1 + center * (2 * num_indices_concentration - 2)
            b = 1 + (1 - center) * (2 * num_indices_concentration - 2)
            num_indices = round(mn + (mx - mn) * rng.beta(a=a, b=b))
    else:
        # validate
        num_indices = int(num_indices)
        if not (mn <= num_indices <= mx):
            raise ValueError(
                f"num_indices must be between {mn} and {mx} for this cost: {cost}."
            )

    # randomly distribute the total ops among index sizes
    # splits = rng.beta(3.0, 3.0, size=num_indices)
    splits = rng.uniform(size=num_indices)
    splits /= splits.sum()
    sizes = [max(2, round(cost**s)) for s in splits]

    # make contraction closer to `cost` by incrementally
    # reducing size of any large enough indices
    cost_actual = np.prod(sizes)
    ratio = cost_actual / cost
    while ratio > 1.0:
        possibly_big = [i for i, d in enumerate(sizes) if d > 2]
        if not possibly_big:
            # couldn't reach target cost somehow
            raise ValueError(
                "Could not reach target cost with given parameters:"
                f" cost={cost}, actual={cost_actual}, ratio={ratio},"
                f" sizes={sizes}, num_indices={num_indices}."
            )
        i = rng.choice(possibly_big)
        # deincrement size of index i
        old = sizes[i]

        # if index is very big we need to take big steps
        dincr = max(1, round(0.1 * old))

        new = old - dincr
        sizes[i] = new
        cost_actual = new * (cost_actual // old)
        ratio = cost_actual / cost

    # make the dictionary of index sizes
    size_dict = dict(zip(string.ascii_letters, sizes))

    # now we distribute the indices according
    # to one of the size following types:
    index_probs = {
        "lkept": p_lkept,  # L + O
        "rkept": p_rkept,  # R + O
        "con": p_con,  # L + R
        "batch": p_batch,  # L + R + O
        "lonly": p_sum / 2,  # L
        "ronly": p_sum / 2,  # R
    }

    if not allow_batch:
        index_probs.pop("batch")
    if not allow_sum:
        index_probs.pop("lonly")
        index_probs.pop("ronly")

    options = np.array(list(index_probs.keys()))
    weights = np.array(list(index_probs.values()))
    weights /= weights.sum()

    types = {}
    for ix in size_dict:
        o = rng.choice(options, p=weights)
        types[ix] = o

    if not allow_scalar:
        # ensure both sides have at least one index

        def has_left():
            return any(
                o in ["lkept", "con", "batch", "lonly", "ltrace"]
                for o in types.values()
            )

        def has_right():
            return any(
                o in ["rkept", "con", "batch", "ronly", "rtrace"]
                for o in types.values()
            )

        while not (has_left() and has_right()):
            ix = rng.choice(list(types.keys()))
            o = rng.choice(options, p=weights)
            types[ix] = o

    if not allow_outer:

        def all_outer():
            return all(o in ["lkept", "rkept"] for o in types.values())

        while all_outer():
            ix = rng.choice(list(types.keys()))
            o = rng.choice(options, p=weights)
            types[ix] = o

    # now we build the terms
    lhs = []
    rhs = []
    out = []
    for ix, o in types.items():
        if o in ["lkept", "con", "batch", "lonly"]:
            lhs.append(ix)
            if allow_trace and rng.rand() < p_trace:
                lhs.append(ix)  # trace or diagonal index
        if o in ["rkept", "con", "batch", "ronly"]:
            rhs.append(ix)
            if allow_trace and rng.rand() < p_trace:
                rhs.append(ix)  # trace or diagonal index
        if o in ["lkept", "rkept", "batch"]:
            out.append(ix)

    # finally shuffle each term
    rng.shuffle(lhs)
    rng.shuffle(rhs)
    rng.shuffle(out)

    # turn into einsum equation
    inputs = (tuple(lhs), tuple(rhs))
    output = tuple(out)

    return inputs, output, size_dict


def compute_flops_and_size(inputs, output, size_dict, dtype=None):
    import math

    # number of scalar operations
    flops = float(math.prod(list(size_dict.values())))
    size = float(
        math.prod([size_dict[ix] for ix in inputs[0]])
        + math.prod([size_dict[ix] for ix in inputs[1]])
        + math.prod([size_dict[ix] for ix in output])
    )

    if dtype is not None:
        if "complex" in dtype:
            flops *= 8
        else:
            flops *= 2

    return flops, size


def timeitsimple(
    fn,
    min_reps=1,
    min_time=0.2,
    max_reps=100,
    max_time=1.0,
):
    """A simple timing function that runs fn() multiple times until
    either min_reps or min_time is reached, but no more than max_reps
    or max_time. Returns the minimum time of all runs.
    """
    import time

    ttot = 0.0
    r = 0
    ts = []
    while (r < min_reps) or (ttot < min_time):
        ti = time.perf_counter()
        fn()
        tf = time.perf_counter()
        t = tf - ti
        ts.append(t)
        r += 1
        ttot += t
        if (r >= max_reps) or (ttot >= max_time):
            break
    return min(ts)


def bench(
    cost_target,
    seed,
    method,
    dtype="float64",
    allow_batch=True,
    **kwargs,
):
    import cotengra as ctg
    import numpy as np

    inputs, output, size_dict = random_pairwise_contraction(
        cost_target,
        seed=seed,
        allow_batch=allow_batch,
        **kwargs,
    )

    flops, size = compute_flops_and_size(inputs, output, size_dict, dtype=dtype)

    arrays = ctg.utils.make_arrays_from_inputs(
        inputs,
        size_dict=size_dict,
        seed=seed,
        dtype=dtype,
    )
    eq = ctg.utils.inputs_output_to_eq(inputs, output)

    if method == "einsum":
        t = timeitsimple(lambda: np.einsum(eq, *arrays, optimize=False))
    elif method == "bmm":
        # test we are in the correct new version of numpy
        from numpy._core.einsumfunc import _parse_eq_to_batch_matmul  # noqa

        t = timeitsimple(lambda: np.einsum(eq, *arrays, optimize=True))
    elif method == "dot":
        try:
            # test we are in the correct *old* version of numpy
            from numpy._core.einsumfunc import _parse_eq_to_batch_matmul  # noqa

            raise ValueError(
                "This version of numpy already has bmm optimization."
            )
        except ImportError:
            pass

        t = timeitsimple(lambda: np.einsum(eq, *arrays, optimize=True))
    else:
        raise ValueError(
            f"Unknown method {method} is not one of 'einsum', 'bmm', or 'dot'."
        )

    return {
        "time": t,
        "flops": flops,
        "size": size,
    }
