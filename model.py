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

# Step 6 - ngram_ladder
def ngram_ladder(splits, max_order=4, alphas=(0.001, 0.01, 0.03, 0.1, 0.3, 1.0)):
    """
    Evaluate n-gram language models across different model orders.

    For each order, select the best smoothing parameter using
    validation cross-entropy, then evaluate training and test perplexity.

    Parameters:
        splits: Dictionary containing 'train', 'val', and 'test' sequences.
        max_order: Maximum n-gram model order to evaluate. Default is 4.
        alphas: Tuple of smoothing parameters to evaluate.

    Returns:
        List of dictionaries containing:
            'order': N-gram model order.
            'alpha': Selected smoothing parameter.
            'train': Training perplexity using alpha=0.0.
            'test': Test perplexity using the selected alpha.
    """

    # Initialize the results with the uniform baseline.
    rows = [
        {
            "order": 0,
            "alpha": None,
            "train": 27.0,
            "test": 27.0
        }
    ]

    # Ensure that at least one smoothing parameter is provided.
    alphas = tuple(alphas)

    if len(alphas) == 0:
        raise ValueError("At least one alpha value must be provided.")

    # Retrieve the training, validation, and test datasets.
    train_seqs = splits["train"]
    val_seqs = splits["val"]
    test_seqs = splits["test"]

    # Evaluate n-gram models from order 1 to max_order.
    for order in range(1, max_order + 1):

        # Initialize variables for selecting the best alpha.
        best_alpha = None
        best_val_entropy = float("inf")
        best_table = None

        # Find the smoothing parameter with the lowest
        # cross-entropy on the validation dataset.
        for alpha in alphas:

            # Fit the model using the current alpha.
            table = fit_ngram(
                train_seqs,
                order,
                alpha=alpha
            )

            # Evaluate cross-entropy on the validation dataset.
            val_entropy, _ = model_perplexity(
                table,
                val_seqs
            )

            # Update the best model if validation entropy improves.
            # Using '<' preserves the earlier alpha when values tie.
            if best_alpha is None or val_entropy < best_val_entropy:
                best_alpha = alpha
                best_val_entropy = val_entropy
                best_table = table

        # Fit a maximum likelihood model without smoothing.
        mle_table = fit_ngram(
            train_seqs,
            order,
            alpha=0.0
        )

        # Calculate training perplexity using the maximum
        # likelihood model.
        _, train_perplexity = model_perplexity(
            mle_table,
            train_seqs
        )

        # Calculate test perplexity using the model
        # with the best validation-selected alpha.
        _, test_perplexity = model_perplexity(
            best_table,
            test_seqs
        )

        # Store the results for the current n-gram order.
        rows.append(
            {
                "order": order,
                "alpha": best_alpha,
                "train": float(train_perplexity),
                "test": float(test_perplexity)
            }
        )

    return rows

# Step 7 - compress_names
def compress_names(splits, alpha=0.01):
    """
    Compress held-out names using context-dependent Huffman codes.

    For unigram, bigram, and trigram models:
        1. Fit an n-gram probability model on training sequences.
        2. Build a Huffman code for every possible context.
        3. Calculate the mean actual codeword length on test sequences.
        4. Calculate the theoretical cross-entropy on test sequences.

    Parameters:
        splits: Dictionary containing 'train', 'val', and 'test' sequences.
        alpha: Additive smoothing parameter. Default is 0.01.

    Returns:
        A flat dictionary containing:
            unigram_bits, unigram_ce,
            bigram_bits, bigram_ce,
            trigram_bits, trigram_ce,
            fixed_bits, log2_27.

        All dictionary values are Python floats.
    """

    # Initialize the results with fixed-length coding baselines.
    results = {
        "fixed_bits": 5.0,
        "log2_27": float(np.log2(27))
    }

    # Evaluate unigram, bigram, and trigram language models.
    models = [
        ("unigram", 1),
        ("bigram", 2),
        ("trigram", 3)
    ]

    # Process each language model independently.
    for name, order in models:

        # Step 1: Fit the n-gram model using the training dataset.
        table = fit_ngram(
            splits["train"],
            order,
            alpha=alpha
        )

        # Step 2: Initialize an array to store Huffman codeword lengths.
        # Its shape is identical to the conditional probability table.
        lengths = np.zeros(table.shape, dtype=np.int64)

        # Generate every possible context index.
        # The final axis represents the symbol being predicted.
        for context in np.ndindex(table.shape[:-1]):

            # Build an optimal Huffman code for the current context.
            lengths[context] = huffman_lengths(table[context])

        # Step 3: Generate context-target events from held-out names.
        events = ngram_events(
            splits["test"],
            order
        )

        # Retrieve the actual Huffman codeword length for each
        # observed symbol using its corresponding context.
        event_lengths = lengths[tuple(events.T)]

        # Calculate the average actual number of bits per symbol.
        results[f"{name}_bits"] = float(np.mean(event_lengths))

        # Step 4: Calculate theoretical cross-entropy in bits
        # using the previously implemented model evaluation function.
        bits_per_symbol, _ = model_perplexity(
            table,
            splits["test"]
        )

        # Store the theoretical cross-entropy.
        results[f"{name}_ce"] = float(bits_per_symbol)

    # Return all compression measurements in one flat dictionary.
    return results

# Step 8 - entropy_table
def entropy_table(P, base=2):
    """
    Compute entropy quantities from a two-variable joint
    probability distribution.

    Parameters:
        P: 2-D array representing the joint distribution P(X, Y).
        base: Logarithm base. Default is 2 (bits).

    Returns:
        Dictionary containing marginal, joint, conditional
        and mutual information quantities.
    """

    # Convert the joint probability table into a NumPy array.
    P = np.asarray(P, dtype=float)

    if P.ndim != 2:
        raise ValueError("P must be a 2-dimensional joint probability table.")

    # Calculate marginal probability distributions.
    p_x = np.sum(P, axis=1)
    p_y = np.sum(P, axis=0)

    # Calculate marginal and joint entropies.
    # max(0.0, ...) also ensures that negative zero becomes positive zero.
    H_X = float(max(0.0, entropy(p_x, base=base)))
    H_Y = float(max(0.0, entropy(p_y, base=base)))
    H_XY = float(max(0.0, entropy(P, base=base)))

    # Calculate conditional entropies using the chain rule.
    H_X_given_Y = float(max(0.0, H_XY - H_Y))
    H_Y_given_X = float(max(0.0, H_XY - H_X))

    # Calculate mutual information.
    I_XY = float(max(0.0, H_X + H_Y - H_XY))

    # Calculate normalized mutual information.
    denominator = min(H_X, H_Y)

    if denominator > 0.0:
        NMI = float(I_XY / denominator)
    else:
        NMI = 0.0

    # Calculate conditional entropy H(X | Y = y) for every column.
    # Initialize with positive zeros, including zero-probability columns.
    H_X_given_y = np.zeros(P.shape[1], dtype=float)

    for j in range(P.shape[1]):

        # Only calculate conditional entropy for positive-probability columns.
        if p_y[j] > 0.0:

            # Calculate the conditional probability distribution P(X | Y=y).
            conditional_distribution = P[:, j] / p_y[j]

            # Calculate entropy and eliminate negative zero.
            H_X_given_y[j] = max(
                0.0,
                entropy(conditional_distribution, base=base)
            )

    # Return all quantities in the required format.
    return {
        "H_X": H_X,
        "H_Y": H_Y,
        "H_XY": H_XY,
        "H_X_given_Y": H_X_given_Y,
        "H_Y_given_X": H_Y_given_X,
        "I_XY": I_XY,
        "NMI": NMI,
        "H_X_given_y": H_X_given_y
    }

# Step 9 - chain_rule_entropy
def chain_rule_entropy(P, base=2):
    """
    Apply the chain rule of entropy to a joint distribution
    over any number of random variables.

    Parameters:
        P: N-dimensional joint probability array, with one
           axis per random variable (X1, X2, ..., Xn).
        base: Logarithm base. Default is 2 (bits).

    Returns:
        A tuple containing:
            H_joint: Joint entropy H(X1, X2, ..., Xn).
            terms: List of conditional entropies:
                   H(X1), H(X2 | X1), ...,
                   H(Xn | X1, ..., Xn-1).
    """

    # Convert the input into a NumPy array.
    P = np.asarray(P, dtype=float)

    # Ensure that the joint distribution has at least one dimension.
    if P.ndim < 1:
        raise ValueError("P must have at least one dimension.")

    # Validate the probability values.
    if not np.all(np.isfinite(P)) or np.any(P < 0):
        raise ValueError("Probabilities must be finite and non-negative.")

    # Get the total number of random variables.
    n = P.ndim

    # Store the entropy of consecutive joint marginals.
    marginal_entropies = []

    # Calculate H(X1), H(X1, X2), ..., H(X1, ..., Xn).
    for i in range(1, n + 1):

        # Sum over all remaining axes to obtain
        # the marginal distribution of the first i variables.
        if i < n:
            marginal = np.sum(P, axis=tuple(range(i, n)))
        else:
            marginal = P

        # Calculate the entropy of the current marginal.
        H = entropy(marginal, base=base)

        # Remove possible negative zero caused by numerical rounding.
        H = float(max(0.0, H))

        # Store the current joint marginal entropy.
        marginal_entropies.append(H)

    # Calculate conditional entropies using consecutive differences.
    terms = []
    previous_entropy = 0.0

    for current_entropy in marginal_entropies:

        # H(Xi | X1, ..., Xi-1) = H(X1, ..., Xi)
        #                            - H(X1, ..., Xi-1)
        conditional_entropy = max(
            0.0,
            current_entropy - previous_entropy
        )

        # Store the conditional entropy as a Python float.
        terms.append(float(conditional_entropy))

        # Update the previous marginal entropy.
        previous_entropy = current_entropy

    # The final marginal entropy is the full joint entropy.
    H_joint = float(max(0.0, marginal_entropies[-1]))

    return H_joint, terms

# Step 10 - sequence_logo
MOTIF_THETA = np.array([
    [0.25, 0.25, 0.25, 0.25],
    [0.40, 0.20, 0.20, 0.20],
    [0.10, 0.10, 0.70, 0.10],
    [0.05, 0.05, 0.05, 0.85],
    [0.90, 0.03, 0.03, 0.04],
    [0.02, 0.02, 0.02, 0.94],
    [1.00, 0.00, 0.00, 0.00],
    [0.96, 0.01, 0.01, 0.02],
    [0.60, 0.05, 0.05, 0.30],
    [0.85, 0.05, 0.05, 0.05],
    [0.45, 0.05, 0.45, 0.05],
    [0.20, 0.40, 0.20, 0.20],
    [0.25, 0.25, 0.25, 0.25],
    [0.30, 0.20, 0.30, 0.20],
    [0.25, 0.25, 0.25, 0.25],
])


def make_motif_alignment(n, seed):
    """
    Generate aligned DNA sequences from the known motif distribution.

    Parameters:
        n: Number of DNA sequences to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of aligned DNA sequences.
    """
    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Sample one nucleotide for each position using its
    # corresponding probability distribution.
    idx = np.stack([
        rng.choice(4, size=n, p=row)
        for row in MOTIF_THETA
    ], axis=1)

    # Convert sampled nucleotide indices into DNA sequences.
    return [
        "".join(row)
        for row in np.array(list("ACGT"))[idx]
    ]


def sequence_logo(seqs):
    """
    Convert aligned DNA sequences into a position weight matrix,
    information content values, and a consensus sequence.

    Parameters:
        seqs: List of equal-length DNA sequences containing
              only A, C, G, and T.

    Returns:
        Tuple containing:
            pwm: Empirical nucleotide frequencies (T, 4).
            info_bits: Information content at each position.
            consensus: Most frequent nucleotide at each position.
    """

    # Define the nucleotide order.
    nucleotides = "ACGT"

    # Convert sequences into a list.
    seqs = list(seqs)

    # Validate the input.
    if len(seqs) == 0:
        raise ValueError("At least one DNA sequence is required.")

    if not all(isinstance(seq, str) for seq in seqs):
        raise ValueError("All sequences must be strings.")

    # Get the number of sequences and their length.
    N = len(seqs)
    T = len(seqs[0])

    # Ensure all sequences have equal length.
    if any(len(seq) != T for seq in seqs):
        raise ValueError("All sequences must have equal length.")

    # Ensure all characters belong to the DNA alphabet.
    if any(base not in nucleotides for seq in seqs for base in seq):
        raise ValueError("Sequences must contain only A, C, G, and T.")

    # Create a mapping from nucleotides to integer indices.
    nucleotide_to_index = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3
    }

    # Encode the sequences as integer arrays.
    encoded = np.array([
        [nucleotide_to_index[base] for base in seq]
        for seq in seqs
    ], dtype=np.int64)

    # Initialize the position weight matrix.
    pwm = np.zeros((T, 4), dtype=float)

    # Calculate empirical nucleotide frequencies at each position.
    for t in range(T):
        counts = np.bincount(encoded[:, t], minlength=4)
        pwm[t] = counts / N

    # Calculate information content for every position.
    # Maximum entropy for four nucleotides is 2 bits.
    info_bits = np.array([
        max(0.0, 2.0 - entropy(pwm[t], base=2))
        for t in range(T)
    ], dtype=float)

    # Construct the consensus sequence.
    # np.argmax resolves ties using the earliest nucleotide in ACGT order.
    consensus = "".join(
        nucleotides[i]
        for i in np.argmax(pwm, axis=1)
    )

    return pwm, info_bits, consensus

# Step 11 - entropy_estimates
def entropy_estimates(counts, base=2):
    """
    Estimate entropy using the plug-in estimator and
    the Miller-Madow bias correction.

    Parameters:
        counts: Array of non-negative observed counts over the states.
        base: Logarithm base. Default is 2 (bits).

    Returns:
        Tuple containing:
            plugin: Plug-in entropy of the empirical distribution.
            miller_madow: Bias-corrected entropy estimate.

        Both values are returned as Python floats.
    """

    # Validate the logarithm base.
    if not np.isfinite(base) or base <= 0 or base == 1:
        raise ValueError("The logarithm base must be positive and different from 1.")

    # Convert counts into a flattened NumPy array.
    counts = np.asarray(counts, dtype=float).ravel()

    # Validate the observed counts.
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("Counts must be finite and non-negative.")

    # Calculate the total number of observations.
    N = float(np.sum(counts))

    # Handle the case where there are no observations.
    if N == 0:
        return 0.0, 0.0

    # Convert observed counts into empirical probabilities.
    probabilities = counts / N

    # Calculate the number of states with non-zero observed counts.
    K = int(np.count_nonzero(counts > 0))

    # Calculate plug-in entropy using the existing entropy function.
    plugin = float(entropy(probabilities, base=base))

    # Calculate the Miller-Madow first-order bias correction.
    # The correction is initially expressed in nats and then
    # converted into the requested logarithm base.
    correction = (K - 1) / (2.0 * N * np.log(base))

    # Add the bias correction to the plug-in entropy.
    miller_madow = float(plugin + correction)

    return plugin, miller_madow

# Step 12 - kl_divergence
def kl_divergence(p, q, base=2):
    """
    Calculate the Kullback-Leibler (KL) divergence between
    two discrete probability distributions.

    D_KL(p || q) = sum(p_k * log(p_k / q_k))

    Parameters:
        p: True probability distribution (list or NumPy array).
        q: Approximate probability distribution (list or NumPy array).
        base: Logarithm base. Default is 2 (bits).

    Returns:
        Python float representing the KL divergence.

        Returns infinity if q_k = 0 where p_k > 0.
        States with p_k = 0 contribute nothing.
    """

    # Validate the logarithm base.
    if not np.isfinite(base) or base <= 0 or base == 1:
        raise ValueError("The logarithm base must be positive and different from 1.")

    # Convert both inputs into flattened NumPy arrays.
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()

    # Ensure both distributions have the same number of elements.
    if p.size != q.size:
        raise ValueError("Both distributions must have the same number of elements.")

    # Validate probability values.
    if (not np.all(np.isfinite(p)) or
            not np.all(np.isfinite(q)) or
            np.any(p < 0) or np.any(q < 0)):
        raise ValueError("Probabilities must be finite and non-negative.")

    # Only consider states where p_k > 0.
    mask = p > 0

    # KL divergence is infinite if p_k > 0 and q_k = 0.
    if np.any(q[mask] == 0):
        return float("inf")

    # Calculate KL divergence using logarithms.
    # Subtracting logarithms avoids unnecessary division.
    divergence = np.sum(
        p[mask] * (np.log(p[mask]) - np.log(q[mask]))
    ) / np.log(base)

    return float(divergence)

