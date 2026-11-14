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

# Step 3 - load_names
import dml_data


ALPHABET = ".abcdefghijklmnopqrstuvwxyz"


def load_names(seed=0):
    """
    Load and encode the first-names dataset, then split it into
    training, validation, and test sets.

    Parameters:
        seed: Random seed for reproducible shuffling. Default is 0.

    Returns:
        Dictionary containing:
            'train': Training set (80%).
            'val': Validation set (10%).
            'test': Test set (remaining 10%).

        Each set is a list of NumPy arrays with dtype np.int64.
        Every encoded name ends with 0, representing the end-of-name symbol.
    """
    # Load the first-names dataset.
    names = dml_data.load("names").names

    # Get the total number of names.
    n = len(names)

    # Generate a reproducible random permutation of the dataset indices.
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    # Create a mapping from each character to its index in ALPHABET.
    char_to_index = {char: i for i, char in enumerate(ALPHABET)}

    # Encode each shuffled name as integer character indices.
    # Append 0 to represent the end-of-name symbol.
    encoded_names = []

    for i in order:
        name = names[i].lower()

        # Convert each character into its corresponding integer index.
        encoded = [char_to_index[char] for char in name]

        # Append the end-of-name symbol.
        encoded.append(0)

        # Store the encoded name as a NumPy int64 array.
        encoded_names.append(np.array(encoded, dtype=np.int64))

    # Calculate the training and validation split boundaries.
    n_train = int(0.8 * n)
    n_val = int(0.9 * n)

    # Split the shuffled and encoded names into three datasets.
    train = encoded_names[:n_train]
    val = encoded_names[n_train:n_val]
    test = encoded_names[n_val:]

    # Return the datasets in the required dictionary format.
    return {
        "train": train,
        "val": val,
        "test": test
    }

# Step 4 - fit_ngram
def ngram_events(seqs, order):
    """
    Generate context-target events from encoded sequences.

    Each row contains:
        - The preceding (order - 1) symbols.
        - The current symbol to predict.

    Missing context symbols at the beginning of a sequence
    are padded with zeros.

    Parameters:
        seqs: List of encoded NumPy arrays.
        order: Order of the n-gram model.

    Returns:
        NumPy integer array of shape (N, order), where N is
        the total number of symbols across all sequences.
    """
    # Validate the n-gram order.
    if not isinstance(order, (int, np.integer)) or isinstance(order, (bool, np.bool_)) or order < 1:
        raise ValueError("Order must be a positive integer.")

    events_list = []

    # Process each encoded sequence independently.
    for seq in seqs:
        seq = np.asarray(seq, dtype=np.int64).ravel()

        # Validate symbol indices.
        if np.any((seq < 0) | (seq >= 27)):
            raise ValueError("Symbol indices must be between 0 and 26.")

        # Allocate an event array for the current sequence.
        events = np.zeros((len(seq), order), dtype=np.int64)

        # Generate one context-target event per symbol.
        for i, symbol in enumerate(seq):

            # The current symbol is always the prediction target.
            events[i, -1] = symbol

            # Identify the available preceding context.
            start = max(0, i - order + 1)
            context = seq[start:i]

            # Place the context immediately before the target.
            # Any remaining positions on the left stay zero-padded.
            if len(context) > 0:
                events[i, order - 1 - len(context):-1] = context

        # Store events generated from this sequence.
        if len(events) > 0:
            events_list.append(events)

    # Combine events from all sequences.
    # Return the correct empty shape when there are no events.
    if not events_list:
        return np.empty((0, order), dtype=np.int64)

    return np.concatenate(events_list, axis=0)


def fit_ngram(seqs, order, alpha=0.0):
    """
    Fit an n-gram language model using conditional probabilities.

    The probability table estimates:

        P(next_symbol | context)

    Counts are accumulated for every context-target event.
    Add-alpha smoothing is applied to every cell.

    Parameters:
        seqs: List of encoded NumPy arrays.
        order: Order of the n-gram model.
        alpha: Additive smoothing parameter. Default is 0.0.

    Returns:
        NumPy float array of shape (27,) * order containing
        conditional probability distributions.
    """
    # Validate the smoothing parameter.
    alpha = float(alpha)

    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Alpha must be finite and non-negative.")

    # Generate context-target events from the input sequences.
    events = ngram_events(seqs, order)

    # Initialize the count table.
    # Its shape is (27,) repeated 'order' times.
    counts = np.zeros((27,) * order, dtype=float)

    # Accumulate the number of occurrences of each event.
    if len(events) > 0:
        np.add.at(counts, tuple(events.T), 1.0)

    # Apply add-alpha smoothing to every cell.
    counts += alpha

    # Calculate the total count for each context.
    # Normalization is performed along the final (target) axis.
    totals = np.sum(counts, axis=-1, keepdims=True)

    # Initialize with a uniform distribution.
    # This handles contexts with zero total count when alpha = 0.
    probabilities = np.full_like(counts, 1.0 / 27.0)

    # Normalize counts to obtain conditional probabilities.
    # Unobserved contexts retain their uniform distribution.
    np.divide(
        counts,
        totals,
        out=probabilities,
        where=totals > 0
    )

    return probabilities

# Step 5 - model_perplexity
def model_perplexity(table, seqs):
    """
    Evaluate a language model using cross-entropy and perplexity.

    Parameters:
        table: Conditional probability table of the language model.
               Its number of dimensions determines the n-gram order.
        seqs: List of encoded sequences to evaluate.

    Returns:
        Tuple containing:
            bits_per_symbol: Average negative log2 probability per event.
            perplexity: 2 raised to the bits per symbol.

        Both values are Python floats.

        Returns (inf, inf) if any observed event has zero probability.
    """
    # Convert the probability table into a NumPy array.
    table = np.asarray(table, dtype=float)

    # Determine the n-gram order from the number of table dimensions.
    order = table.ndim

    # Generate context-target events using the previously implemented function.
    events = ngram_events(seqs, order)

    # An empty dataset has no events over which to calculate a mean.
    if len(events) == 0:
        raise ValueError("Cannot calculate perplexity for an empty dataset.")

    # Retrieve the model probability assigned to each observed event.
    # Each row of events provides the corresponding multidimensional index.
    probabilities = table[tuple(events.T)]

    # Check that the retrieved values are valid probabilities.
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("Model probabilities must be finite and non-negative.")

    # If any observed event has zero probability,
    # the cross-entropy and perplexity are both infinite.
    if np.any(probabilities == 0):
        return float("inf"), float("inf")

    # Calculate the average negative log2 probability per symbol.
    bits_per_symbol = float(-np.mean(np.log2(probabilities)))

    # Calculate perplexity from the average cross-entropy in bits.
    perplexity = float(2.0 ** bits_per_symbol)

    return bits_per_symbol, perplexity

