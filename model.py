"""
Information Theory from Scratch in NumPy: Entropy, KL and Mutual Information

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - entropy
import numpy as np

def entropy(p, base=2):
    """
    Calculate the Shannon entropy of a probability distribution.

    H(p) = -sum(p_k * log(p_k))

    Parameters:
        p: Probability distribution (list or NumPy array of any shape).
        base: Logarithm base. Default is 2 (bits).

    Returns:
        Python float representing the entropy.
    """
    # Validate the logarithm base.
    if not np.isfinite(base) or base <= 0 or base == 1:
        raise ValueError("The logarithm base must be positive and different from 1.")

    # Convert the input into a flattened NumPy array.
    p = np.asarray(p, dtype=float).ravel()

    # Reject invalid probability values.
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("Probabilities must be finite and non-negative.")

    # Ignore zero probabilities to avoid log(0) and numerical warnings.
    p_nonzero = p[p > 0]

    # Calculate entropy using the specified logarithm base.
    H = -np.sum(p_nonzero * np.log(p_nonzero) / np.log(base))

    return float(H)


def cross_entropy(p, q, base=2):
    """
    Calculate the cross-entropy between two probability distributions.

    H(p, q) = -sum(p_k * log(q_k)), considering only p_k > 0.

    Parameters:
        p: True probability distribution (list or NumPy array).
        q: Predicted probability distribution (list or NumPy array).
        base: Logarithm base. Default is 2 (bits).

    Returns:
        Python float representing the cross-entropy.
        Returns infinity if q_k = 0 where p_k > 0.
    """
    # Validate the logarithm base.
    if not np.isfinite(base) or base <= 0 or base == 1:
        raise ValueError("The logarithm base must be positive and different from 1.")

    # Convert both inputs into flattened NumPy arrays.
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()

    # Ensure both distributions contain the same number of states.
    if p.size != q.size:
        raise ValueError("Both distributions must have the same number of elements.")

    # Reject invalid probability values.
    if (not np.all(np.isfinite(p)) or
            not np.all(np.isfinite(q)) or
            np.any(p < 0) or np.any(q < 0)):
        raise ValueError("Probabilities must be finite and non-negative.")

    # Only consider states where the true probability is positive.
    mask = p > 0

    # A positive true probability with zero predicted probability
    # results in infinite cross-entropy.
    if np.any(q[mask] == 0):
        return float("inf")

    # Calculate cross-entropy without evaluating log(0).
    H_cross = -np.sum(p[mask] * np.log(q[mask]) / np.log(base))

    return float(H_cross)


def perplexity(p):
    """
    Calculate the perplexity of a probability distribution.

    Perplexity(p) = 2 ** H(p)

    Entropy is measured in bits.

    Parameters:
        p: Probability distribution (list or NumPy array).

    Returns:
        Python float representing the perplexity.
    """
    # Calculate entropy in bits.
    H = entropy(p, base=2)

    # Convert entropy into perplexity.
    return float(2.0 ** H)

# Step 2 - huffman_lengths
import heapq


def huffman_lengths(p):
    """
    Compute optimal prefix-codeword lengths using Huffman coding.

    Parameters:
        p: Probability distribution (list or NumPy array).

    Returns:
        NumPy array containing the optimal codeword length
        for each symbol. Symbols with zero probability
        receive a codeword length of 0.
    """
    # Convert probabilities into a flattened NumPy array.
    p = np.asarray(p, dtype=float).ravel()

    # Validate the probability values.
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("Probabilities must be finite and non-negative.")

    # Initialize codeword lengths to zero.
    lengths = np.zeros(len(p), dtype=int)

    # Build a min-heap containing only positive-probability symbols.
    # Each entry stores (probability, unique_id, symbol_indices).
    # The unique ID prevents heapq from comparing member lists
    # when two groups have identical probabilities.
    heap = []
    unique_id = 0

    for i, prob in enumerate(p):
        if prob > 0:
            heapq.heappush(heap, (float(prob), unique_id, [i]))
            unique_id += 1

    # If there are no positive-probability symbols, return zero lengths.
    if len(heap) == 0:
        return lengths

    # A single positive-probability symbol still requires one bit.
    if len(heap) == 1:
        lengths[heap[0][2][0]] = 1
        return lengths

    # Repeatedly merge the two groups with the smallest probabilities.
    while len(heap) > 1:
        # Extract the two least probable groups.
        prob1, _, members1 = heapq.heappop(heap)
        prob2, _, members2 = heapq.heappop(heap)

        # Every symbol in these groups gains one additional bit.
        for index in members1:
            lengths[index] += 1

        for index in members2:
            lengths[index] += 1

        # Merge the two groups and insert the combined group into the heap.
        merged_prob = prob1 + prob2
        merged_members = members1 + members2

        heapq.heappush(
            heap,
            (merged_prob, unique_id, merged_members)
        )

        unique_id += 1

    return lengths


def expected_code_length(p, lengths):
    """
    Calculate the expected codeword length of a distribution.

    L = sum(p_k * l_k)

    Parameters:
        p: Probability distribution (list or NumPy array).
        lengths: Codeword lengths corresponding to each symbol.

    Returns:
        Python float representing the expected code length.
        Returns infinity if a positive-probability symbol
        has a codeword length of zero.
    """
    # Convert inputs into flattened NumPy arrays.
    p = np.asarray(p, dtype=float).ravel()
    lengths = np.asarray(lengths, dtype=float).ravel()

    # Ensure both arrays contain the same number of elements.
    if p.size != lengths.size:
        raise ValueError("Probabilities and code lengths must have the same size.")

    # Validate probability values.
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("Probabilities must be finite and non-negative.")

    # Validate codeword lengths.
    if not np.all(np.isfinite(lengths)) or np.any(lengths < 0):
        raise ValueError("Codeword lengths must be finite and non-negative.")

    # Consider only symbols with positive probability.
    mask = p > 0

    # A positive-probability symbol without a codeword
    # makes the expected code length infinite.
    if np.any(lengths[mask] == 0):
        return float("inf")

    # Calculate the expected codeword length.
    L = np.sum(p[mask] * lengths[mask])

    return float(L)

