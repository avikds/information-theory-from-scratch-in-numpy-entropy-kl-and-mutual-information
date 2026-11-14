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

# Step 13 - quantized_entropy
def gaussian_entropy(Sigma, base=np.e):
    """
    Calculate the differential entropy of a Gaussian distribution.

    h(X) = 1/2 * ln((2 * pi * e)^d * |Sigma|)

    Parameters:
        Sigma: Covariance matrix or scalar variance.
        base: Logarithm base. Default is np.e (nats).

    Returns:
        Python float representing Gaussian differential entropy.
    """

    # Validate the logarithm base.
    if not np.isfinite(base) or base <= 0 or base == 1:
        raise ValueError("The logarithm base must be positive and different from 1.")

    # Convert the covariance input into a NumPy array.
    Sigma = np.asarray(Sigma, dtype=float)

    # Ensure covariance values are finite.
    if not np.all(np.isfinite(Sigma)):
        raise ValueError("Covariance values must be finite.")

    # Handle the scalar variance case.
    if Sigma.ndim == 0:

        # A variance must be strictly positive.
        if Sigma <= 0:
            raise ValueError("Variance must be strictly positive.")

        # For a scalar variance, the dimensionality is one.
        d = 1

        # Calculate the logarithm of the determinant.
        log_det = np.log(float(Sigma))

    else:

        # A covariance matrix must be two-dimensional and square.
        if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
            raise ValueError("Sigma must be a square covariance matrix or scalar variance.")

        # Determine the dimensionality of the Gaussian distribution.
        d = Sigma.shape[0]

        if d == 0:
            raise ValueError("Covariance matrix must not be empty.")

        # Check that the covariance matrix is symmetric.
        if not np.allclose(Sigma, Sigma.T):
            raise ValueError("Covariance matrix must be symmetric.")

        # Use Cholesky decomposition to verify positive definiteness.
        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix must be positive definite.")

        # Calculate log(det(Sigma)) using the Cholesky factor.
        # This is more numerically stable than computing the determinant directly.
        log_det = 2.0 * np.sum(np.log(np.diag(L)))

    # Calculate differential entropy in nats.
    h_nats = 0.5 * (
        d * np.log(2.0 * np.pi * np.e) + log_det
    )

    # Convert entropy into the requested logarithm base.
    result = h_nats / np.log(base)

    return float(result)


def quantized_entropy(x, n_bits):
    """
    Calculate the empirical entropy of quantized continuous samples.

    Samples are quantized using:

        codes = floor(x * 2**n_bits)

    The entropy is calculated in bits.

    Parameters:
        x: Continuous samples (list or NumPy array).
        n_bits: Number of binary fractional bits used for quantization.

    Returns:
        Python float representing the entropy of the empirical
        distribution of the resulting integer codes.
    """

    # Validate the number of quantization bits.
    if not isinstance(n_bits, (int, np.integer)) or isinstance(n_bits, (bool, np.bool_)) or n_bits < 0:
        raise ValueError("n_bits must be a non-negative integer.")

    # Convert the samples into a flattened NumPy array.
    x = np.asarray(x, dtype=float).ravel()

    # Reject empty input.
    if x.size == 0:
        raise ValueError("At least one sample is required.")

    # Ensure that all samples are finite.
    if not np.all(np.isfinite(x)):
        raise ValueError("Samples must contain only finite values.")

    # Calculate the quantization scale.
    scale = 2.0 ** n_bits

    # Quantize samples into integer codes.
    codes = np.floor(x * scale)

    # Check that the resulting codes can be safely represented
    # as NumPy int64 values.
    if not np.all(np.isfinite(codes)) or np.any(codes < np.iinfo(np.int64).min) or np.any(codes > np.iinfo(np.int64).max):
        raise ValueError("Quantized codes exceed the supported integer range.")

    codes = codes.astype(np.int64)

    # Count the frequency of each unique integer code.
    _, counts = np.unique(codes, return_counts=True)

    # Convert the counts into empirical probabilities.
    probabilities = counts / counts.sum()

    # Calculate Shannon entropy in bits.
    return float(entropy(probabilities, base=2))

# Step 14 - gaussian_kl
def gaussian_kl(mu1, Sigma1, mu2, Sigma2):
    """
    Calculate the KL divergence between two multivariate Gaussian
    distributions:

        D_KL(N(mu1, Sigma1) || N(mu2, Sigma2))

    The divergence is calculated in nats.

    Parameters:
        mu1: Mean of the first Gaussian (scalar or vector).
        Sigma1: Covariance matrix or scalar variance of the first Gaussian.
        mu2: Mean of the second Gaussian (scalar or vector).
        Sigma2: Covariance matrix or scalar variance of the second Gaussian.

    Returns:
        Python float representing the KL divergence in nats.
    """

    # Convert means into one-dimensional NumPy arrays.
    mu1 = np.asarray(mu1, dtype=float).reshape(-1)
    mu2 = np.asarray(mu2, dtype=float).reshape(-1)

    # Ensure that both mean vectors have the same dimensions.
    if mu1.size != mu2.size:
        raise ValueError("Both mean vectors must have the same dimension.")

    # Validate the mean values.
    if not np.all(np.isfinite(mu1)) or not np.all(np.isfinite(mu2)):
        raise ValueError("Mean values must be finite.")

    # Convert scalar variances into 1x1 covariance matrices.
    Sigma1 = np.asarray(Sigma1, dtype=float)
    Sigma2 = np.asarray(Sigma2, dtype=float)

    if Sigma1.ndim == 0:
        Sigma1 = Sigma1.reshape(1, 1)

    if Sigma2.ndim == 0:
        Sigma2 = Sigma2.reshape(1, 1)

    # Ensure covariance matrices are square.
    if (Sigma1.ndim != 2 or Sigma2.ndim != 2 or
            Sigma1.shape[0] != Sigma1.shape[1] or
            Sigma2.shape[0] != Sigma2.shape[1]):
        raise ValueError("Covariance matrices must be square.")

    # Determine the dimensionality of the distributions.
    D = mu1.size

    # Ensure covariance dimensions match the mean vectors.
    if Sigma1.shape != (D, D) or Sigma2.shape != (D, D):
        raise ValueError("Covariance dimensions must match the mean vectors.")

    # Validate covariance matrix values.
    if not np.all(np.isfinite(Sigma1)) or not np.all(np.isfinite(Sigma2)):
        raise ValueError("Covariance matrices must contain finite values.")

    # Covariance matrices must be symmetric.
    if not np.allclose(Sigma1, Sigma1.T) or not np.allclose(Sigma2, Sigma2.T):
        raise ValueError("Covariance matrices must be symmetric.")

    # Verify positive definiteness using Cholesky decomposition.
    try:
        np.linalg.cholesky(Sigma1)
        np.linalg.cholesky(Sigma2)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrices must be positive definite.")

    # Calculate the log determinants using slogdet.
    # This avoids directly computing potentially unstable determinants.
    sign1, logdet1 = np.linalg.slogdet(Sigma1)
    sign2, logdet2 = np.linalg.slogdet(Sigma2)

    if sign1 <= 0 or sign2 <= 0:
        raise ValueError("Covariance determinants must be positive.")

    # Calculate the trace term using solve instead of an explicit inverse.
    trace_term = np.trace(
        np.linalg.solve(Sigma2, Sigma1)
    )

    # Calculate the difference between the mean vectors.
    diff = mu2 - mu1

    # Calculate the quadratic term using a linear system solver.
    quadratic_term = diff @ np.linalg.solve(Sigma2, diff)

    # Calculate the logarithm of the determinant ratio.
    logdet_term = logdet2 - logdet1

    # Apply the multivariate Gaussian KL divergence formula.
    divergence = 0.5 * (
        trace_term
        + quadratic_term
        - D
        + logdet_term
    )

    # Eliminate tiny negative values caused by floating-point rounding.
    return float(max(0.0, divergence))

# Step 15 - fit_forward_gaussian
# Parameters of the two-component Gaussian mixture.
BIMODAL_MEANS = np.array([
    [-2.5, -2.5],
    [2.5, 2.5]
])

BIMODAL_COV = np.array([
    [1.0, -0.5],
    [-0.5, 1.0]
])


def bimodal_sample(n, seed):
    """
    Generate samples from an equal-weight two-component
    Gaussian mixture.

    Parameters:
        n: Number of samples.
        seed: Random seed for reproducibility.

    Returns:
        NumPy array of shape (n, 2).
    """
    rng = np.random.default_rng(seed)

    # Randomly select one of the two Gaussian components.
    which = rng.integers(0, 2, size=n)

    # Generate standard normal samples.
    z = rng.standard_normal((n, 2))

    # Transform samples using the covariance matrix.
    return BIMODAL_MEANS[which] + z @ np.linalg.cholesky(BIMODAL_COV).T


def gaussian_logpdf(X, mu, Sigma):
    """
    Calculate the log probability density of a multivariate
    Gaussian distribution for each observation.

    Parameters:
        X: Observations of shape (n, d), or a single point of shape (d,).
        mu: Mean vector of shape (d,).
        Sigma: Covariance matrix of shape (d, d).
               A scalar variance is also accepted for d = 1.

    Returns:
        NumPy array of shape (n,) containing log probability densities.
    """

    # Convert observations and mean into NumPy arrays.
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float).reshape(-1)

    # Treat a single observation as a batch containing one point.
    X = np.atleast_2d(X)

    # Ensure observations have the correct dimensions.
    if X.ndim != 2 or X.shape[1] != mu.size:
        raise ValueError("Observation dimensions must match the mean vector.")

    # Validate the input values.
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(mu)):
        raise ValueError("Observations and mean must contain finite values.")

    # Convert scalar variance into a 1x1 covariance matrix.
    Sigma = np.asarray(Sigma, dtype=float)

    if Sigma.ndim == 0:
        Sigma = Sigma.reshape(1, 1)

    d = mu.size

    # Validate the covariance matrix dimensions.
    if Sigma.shape != (d, d):
        raise ValueError("Covariance matrix dimensions must match the data.")

    if not np.all(np.isfinite(Sigma)):
        raise ValueError("Covariance matrix must contain finite values.")

    # Verify that the covariance matrix is symmetric and positive definite.
    if not np.allclose(Sigma, Sigma.T):
        raise ValueError("Covariance matrix must be symmetric.")

    try:
        np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix must be positive definite.")

    # Calculate the logarithm of the covariance determinant.
    sign, logdet = np.linalg.slogdet(Sigma)

    if sign <= 0:
        raise ValueError("Covariance determinant must be positive.")

    # Center the observations around the Gaussian mean.
    centered = X - mu

    # Solve Sigma * z = centered.T instead of computing
    # the explicit inverse of the covariance matrix.
    solved = np.linalg.solve(Sigma, centered.T).T

    # Calculate the Mahalanobis distance for each observation.
    quadratic = np.sum(centered * solved, axis=1)

    # Calculate the Gaussian log probability density.
    logpdf = -0.5 * (
        d * np.log(2.0 * np.pi)
        + logdet
        + quadratic
    )

    return np.asarray(logpdf, dtype=float)


def bimodal_logpdf(X):
    """
    Calculate the log probability density of the target
    equal-weight two-component Gaussian mixture.

    Parameters:
        X: Observations of shape (n, 2), or a single point
           of shape (2,).

    Returns:
        NumPy array containing the log density of each observation.
    """

    # Calculate the log density under the first Gaussian component.
    logp0 = gaussian_logpdf(
        X,
        BIMODAL_MEANS[0],
        BIMODAL_COV
    )

    # Calculate the log density under the second Gaussian component.
    logp1 = gaussian_logpdf(
        X,
        BIMODAL_MEANS[1],
        BIMODAL_COV
    )

    # Combine both component densities using log-sum-exp.
    # Each component has an equal mixture weight of 0.5.
    log_density = np.logaddexp(logp0, logp1) - np.log(2.0)

    return log_density


def fit_forward_gaussian(samples):
    """
    Fit a single Gaussian distribution to samples using maximum
    likelihood estimation.

    The forward KL-optimal Gaussian has:
        Mean = Sample mean
        Covariance = Maximum likelihood sample covariance

    The covariance is divided by n, not n - 1.

    Parameters:
        samples: Array of observations with shape (n, d).

    Returns:
        Tuple containing:
            mu: Estimated mean vector.
            cov: Estimated maximum likelihood covariance matrix.
    """

    # Convert samples into a NumPy array.
    samples = np.asarray(samples, dtype=float)

    # Interpret a one-dimensional input as n observations of one variable.
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    # Ensure that the input has the correct dimensions.
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("Samples must be a non-empty two-dimensional array.")

    # Validate sample values.
    if not np.all(np.isfinite(samples)):
        raise ValueError("Samples must contain finite values.")

    # Calculate the maximum likelihood estimate of the mean.
    mu = np.mean(samples, axis=0)

    # Center the observations around the estimated mean.
    centered = samples - mu

    # Calculate the maximum likelihood covariance matrix.
    # Divide by n rather than n - 1.
    n = samples.shape[0]
    cov = (centered.T @ centered) / n

    return mu, cov

# Step 16 - fit_reverse_gaussian
def bimodal_grad(X):
    """
    Calculate the gradient of the log density of the
    bimodal Gaussian mixture with respect to each observation.

    Parameters:
        X: Observations of shape (n, 2), or a single point
           of shape (2,).

    Returns:
        NumPy array of shape (n, 2) containing the gradients.
    """

    # Convert observations into a NumPy array.
    X = np.asarray(X, dtype=float)

    # Treat a single observation as a batch of one.
    X = np.atleast_2d(X)

    # Validate the observation dimensions.
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape (n, 2) or (2,).")

    if not np.all(np.isfinite(X)):
        raise ValueError("Observations must contain finite values.")

    # Calculate the log densities under both Gaussian components.
    logp0 = gaussian_logpdf(
        X,
        BIMODAL_MEANS[0],
        BIMODAL_COV
    )

    logp1 = gaussian_logpdf(
        X,
        BIMODAL_MEANS[1],
        BIMODAL_COV
    )

    # Calculate the log of the sum of component densities.
    # This ensures numerical stability when calculating responsibilities.
    log_norm = np.logaddexp(logp0, logp1)

    # Calculate the posterior responsibilities of both components.
    r0 = np.exp(logp0 - log_norm)
    r1 = np.exp(logp1 - log_norm)

    # Calculate the responsibility-weighted difference between
    # each component mean and the observations.
    weighted_delta = (
        r0[:, None] * (BIMODAL_MEANS[0] - X)
        + r1[:, None] * (BIMODAL_MEANS[1] - X)
    )

    # Calculate the gradient using the inverse covariance
    # through a linear system solver rather than an explicit inverse.
    grad = np.linalg.solve(
        BIMODAL_COV,
        weighted_delta.T
    ).T

    return grad


def fit_reverse_gaussian(mu0, n_steps=400, lr=0.05, n_mc=1024, seed=0):
    """
    Fit a Gaussian distribution to a bimodal target by minimizing
    the reverse KL divergence D_KL(q || p).

    The approximating Gaussian is parameterized as:

        q = N(mu, L @ L.T)

    where L is a lower-triangular matrix.

    Parameters:
        mu0: Initial mean vector of shape (2,).
        n_steps: Number of gradient descent iterations.
        lr: Learning rate.
        n_mc: Number of fixed Monte Carlo samples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple containing:
            mu: Optimized mean vector.
            cov: Optimized covariance matrix.
    """

    # Validate optimization parameters.
    if not isinstance(n_steps, (int, np.integer)) or isinstance(n_steps, (bool, np.bool_)) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer.")

    if not isinstance(n_mc, (int, np.integer)) or isinstance(n_mc, (bool, np.bool_)) or n_mc <= 0:
        raise ValueError("n_mc must be a positive integer.")

    if not np.isfinite(lr) or lr < 0:
        raise ValueError("Learning rate must be finite and non-negative.")

    # Convert the initial mean into a NumPy array.
    mu = np.asarray(mu0, dtype=float).reshape(-1).copy()

    if mu.shape != (2,) or not np.all(np.isfinite(mu)):
        raise ValueError("mu0 must be a finite vector of shape (2,).")

    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Generate standard normal Monte Carlo samples ONCE.
    # These same samples are reused throughout gradient descent.
    z = rng.standard_normal((n_mc, 2))

    # Initialize the lower-triangular covariance factor.
    L = np.eye(2)

    # Perform gradient descent to minimize reverse KL divergence.
    for _ in range(n_steps):

        # Reparameterize the Gaussian samples:
        # x = mu + L @ z, expressed in row-wise array form.
        x = mu + z @ L.T

        # Calculate the gradient of the target log density.
        g = bimodal_grad(x)

        # Calculate the gradient with respect to the mean.
        grad_mu = -np.mean(g, axis=0)

        # Calculate the inverse transpose of L using a linear solver.
        # This avoids explicitly calculating the matrix inverse.
        L_inv_T = np.linalg.solve(
            L.T,
            np.eye(2)
        )

        # Calculate the gradient with respect to L.
        # Include the entropy contribution -L^(-T).
        grad_L = -g.T @ z / n_mc - L_inv_T

        # Keep the covariance factor lower triangular.
        grad_L = np.tril(grad_L)

        # Update both parameters using gradients computed
        # from the same set of Monte Carlo samples.
        mu -= lr * grad_mu
        L -= lr * grad_L

    # Recover the covariance matrix from the optimized factor.
    cov = L @ L.T

    return mu, cov

# Step 17 - kl_both_ways
def kl_both_ways(n=100000, seed=0):
    """
    Compare forward and reverse KL divergence for two Gaussian
    approximations of a bimodal target distribution.

    The function:
        1. Generates samples from the bimodal target.
        2. Fits a forward Gaussian using maximum likelihood.
        3. Fits a reverse Gaussian using reverse KL minimization.
        4. Uses shared standard normal samples to generate
           samples from both fitted Gaussian distributions.
        5. Estimates both KL divergence directions using
           Monte Carlo averages.

    Parameters:
        n: Number of Monte Carlo samples. Default is 100000.
        seed: Random seed for reproducibility. Default is 0.

    Returns:
        Dictionary containing two entries:
            'forward': Results for the forward Gaussian fit.
            'reverse': Results for the reverse Gaussian fit.

        Each entry contains:
            'mu': Fitted mean vector.
            'cov': Fitted covariance matrix.
            'kl_p_q': Estimated KL(p || q) in nats.
            'kl_q_p': Estimated KL(q || p) in nats.
    """

    # Step 1: Generate samples from the target bimodal distribution.
    xp = bimodal_sample(n, seed)

    # Calculate the target log density for these samples.
    logp_xp = bimodal_logpdf(xp)

    # Step 2: Fit the forward Gaussian using maximum likelihood.
    # Use seed + 1 to generate an independent training sample.
    forward_samples = bimodal_sample(n, seed + 1)

    mu_f, cov_f = fit_forward_gaussian(forward_samples)

    # Fit the reverse Gaussian by minimizing reverse KL divergence.
    mu_r, cov_r = fit_reverse_gaussian(
        [1.0, 0.5],
        seed=seed
    )

    # Step 3: Generate standard normal samples ONCE.
    # Reuse the same random samples for both Gaussian approximations.
    rng = np.random.default_rng(seed + 2)
    z = rng.standard_normal((n, 2))

    # Store the two fitted Gaussian models.
    models = {
        "forward": (mu_f, cov_f),
        "reverse": (mu_r, cov_r)
    }

    # Initialize the output dictionary.
    results = {}

    # Step 4: Evaluate both KL directions for each fitted Gaussian.
    for name, (mu, cov) in models.items():

        # Generate samples from the fitted Gaussian using
        # the shared standard normal samples.
        L = np.linalg.cholesky(cov)
        xq = mu + z @ L.T

        # Calculate log q(x) for samples from the target distribution.
        logq_xp = gaussian_logpdf(xp, mu, cov)

        # Calculate log q(x) for samples from the fitted Gaussian.
        logq_xq = gaussian_logpdf(xq, mu, cov)

        # Calculate log p(x) for samples from the fitted Gaussian.
        logp_xq = bimodal_logpdf(xq)

        # Estimate forward KL divergence:
        # D_KL(p || q) = E_p[log p(x) - log q(x)].
        kl_p_q = float(np.mean(logp_xp - logq_xp))

        # Estimate reverse KL divergence:
        # D_KL(q || p) = E_q[log q(x) - log p(x)].
        kl_q_p = float(np.mean(logq_xq - logp_xq))

        # Store the fitted parameters and KL estimates.
        results[name] = {
            "mu": mu,
            "cov": cov,
            "kl_p_q": kl_p_q,
            "kl_q_p": kl_q_p
        }

    return results

# Step 18 - mean_field_reverse_kl
def mean_field_reverse_kl(P, seed=0, n_iters=200, eps=1e-8):
    """
    Approximate a joint probability distribution using a
    factorized mean-field distribution by minimizing reverse KL.

    Parameters:
        P: Two-dimensional joint probability table P(x, y).
        seed: Random seed for reproducibility. Default is 0.
        n_iters: Number of coordinate descent iterations. Default is 200.
        eps: Small smoothing constant. Default is 1e-8.

    Returns:
        Tuple containing:
            qx: Optimized marginal approximation for X.
            qy: Optimized marginal approximation for Y.
            kl: KL divergence between the factorized approximation
                and the original joint distribution, measured in bits.
    """

    # Convert the input into a two-dimensional NumPy array.
    P = np.asarray(P, dtype=float)

    # Validate the joint probability table.
    if P.ndim != 2 or P.shape[0] == 0 or P.shape[1] == 0:
        raise ValueError("P must be a non-empty two-dimensional array.")

    if not np.all(np.isfinite(P)) or np.any(P < 0):
        raise ValueError("P must contain finite, non-negative values.")

    if P.sum() <= 0:
        raise ValueError("P must have a positive total probability.")

    # Validate the iteration count and smoothing parameter.
    if not isinstance(n_iters, (int, np.integer)) or isinstance(n_iters, (bool, np.bool_)) or n_iters < 0:
        raise ValueError("n_iters must be a non-negative integer.")

    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite value.")

    # Step 1: Smooth the joint probability table and take its logarithm.
    # Smoothing prevents log(0) and numerical underflow.
    logp = np.log((P + eps) / np.sum(P + eps))

    # Step 2: Initialize qx and qy using random Dirichlet distributions.
    rng = np.random.default_rng(seed)

    qx = rng.dirichlet(np.ones(P.shape[0]))
    qy = rng.dirichlet(np.ones(P.shape[1]))

    # Step 3: Perform alternating coordinate descent updates.
    for _ in range(n_iters):

        # Update qx while keeping qy fixed:
        # qx(x) is proportional to exp(sum_y qy(y) * log P(x, y)).
        log_qx = np.sum(qy[None, :] * logp, axis=1)

        # Subtract the largest exponent for numerical stability.
        log_qx -= np.max(log_qx)

        # Convert log probabilities into normalized probabilities.
        qx = np.exp(log_qx)
        qx /= np.sum(qx)

        # Update qy using the newly updated qx:
        # qy(y) is proportional to exp(sum_x qx(x) * log P(x, y)).
        log_qy = np.sum(qx[:, None] * logp, axis=0)

        # Subtract the largest exponent for numerical stability.
        log_qy -= np.max(log_qy)

        # Convert log probabilities into normalized probabilities.
        qy = np.exp(log_qy)
        qy /= np.sum(qy)

    # Step 4: Remove very small probabilities.
    # Values below 1e-3 are set to zero.
    qx[qx < 1e-3] = 0.0
    qy[qy < 1e-3] = 0.0

    # Renormalize qx after thresholding.
    if qx.sum() > 0:
        qx /= qx.sum()
    else:
        qx[np.argmax(qx)] = 1.0

    # Renormalize qy after thresholding.
    if qy.sum() > 0:
        qy /= qy.sum()
    else:
        qy[np.argmax(qy)] = 1.0

    # Step 5: Construct the factorized joint approximation.
    q_joint = np.outer(qx, qy)

    # Calculate KL divergence against the original joint table.
    # The divergence is measured in bits.
    kl = float(kl_divergence(q_joint, P, base=2))

    return qx, qy, kl

# Step 19 - gaussian_mi
def gaussian_mi(rho):
    """
    Calculate mutual information between two jointly Gaussian
    random variables using their correlation coefficient.

    I(X;Y) = -0.5 * ln(1 - rho^2)

    Parameters:
        rho: Correlation coefficient between X and Y.

    Returns:
        Python float representing mutual information in nats.
        Returns infinity when abs(rho) >= 1.
    """

    # Convert the correlation coefficient into a Python float.
    rho = float(rho)

    # Validate the correlation coefficient.
    if not np.isfinite(rho):
        raise ValueError("Correlation coefficient must be finite.")

    # Perfect correlation results in infinite mutual information.
    if abs(rho) >= 1:
        return float("inf")

    # Calculate Gaussian mutual information in nats.
    mi = -0.5 * np.log(1.0 - rho ** 2)

    return float(mi)


def gaussian_block_mi(Sigma, dx):
    """
    Calculate mutual information between two blocks of jointly
    Gaussian variables using their joint covariance matrix.

    X consists of the first dx coordinates.
    Y consists of the remaining coordinates.

    I(X;Y) = h(X) + h(Y) - h(X,Y)

    Parameters:
        Sigma: Joint covariance matrix of shape (d, d).
        dx: Number of coordinates belonging to X.

    Returns:
        Python float representing mutual information in nats.
    """

    # Convert the covariance matrix into a NumPy array.
    Sigma = np.asarray(Sigma, dtype=float)

    # Validate the covariance matrix dimensions.
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be a square covariance matrix.")

    # Determine the total number of variables.
    d = Sigma.shape[0]

    # Validate the partition dimension.
    if not isinstance(dx, (int, np.integer)) or isinstance(dx, (bool, np.bool_)) or not 0 < dx < d:
        raise ValueError("dx must be an integer between 0 and d.")

    # Check that the covariance matrix is symmetric.
    if not np.allclose(Sigma, Sigma.T):
        raise ValueError("Sigma must be symmetric.")

    # Extract the covariance matrix of X (first dx coordinates).
    Sigma_x = Sigma[:dx, :dx]

    # Extract the covariance matrix of Y (remaining coordinates).
    Sigma_y = Sigma[dx:, dx:]

    # Calculate the marginal differential entropies.
    h_x = gaussian_entropy(Sigma_x)
    h_y = gaussian_entropy(Sigma_y)

    # Calculate the joint differential entropy.
    h_xy = gaussian_entropy(Sigma)

    # Calculate mutual information using the entropy identity.
    mi = h_x + h_y - h_xy

    # Protect against tiny negative values caused by floating-point rounding.
    return float(max(0.0, mi))


def sample_gaussian_pair(n, rho, seed):
    """
    Generate two correlated standard Gaussian random variables.

    X ~ N(0, 1)
    Y = rho * X + sqrt(1 - rho^2) * Z

    where Z is an independent standard normal random variable.

    Parameters:
        n: Number of samples.
        rho: Desired correlation coefficient.
        seed: Random seed for reproducibility.

    Returns:
        Tuple containing:
            x: NumPy array of length n.
            y: NumPy array of length n.

        Both variables have unit variance and correlation rho.
    """

    # Validate the sample count.
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 0:
        raise ValueError("n must be a non-negative integer.")

    # Validate the correlation coefficient.
    rho = float(rho)

    if not np.isfinite(rho) or abs(rho) > 1:
        raise ValueError("rho must be between -1 and 1.")

    # Create a reproducible random number generator.
    rng = np.random.default_rng(seed)

    # Generate the first standard normal variable.
    x = rng.standard_normal(n)

    # Generate independent standard normal noise.
    noise = rng.standard_normal(n)

    # Construct the second Gaussian variable with correlation rho.
    y = rho * x + np.sqrt(1.0 - rho ** 2) * noise

    return x, y

# Step 20 - histogram_mi
def scott_bins(x):
    """
    Calculate the number of histogram bins using Scott's rule.

    B = N^(1/3) * (max(x) - min(x)) / (3.5 * std(x))

    Parameters:
        x: One-dimensional array of observations.

    Returns:
        Integer representing the recommended number of bins.
        The result is always at least 1.
    """

    # Convert observations into a flattened NumPy array.
    x = np.asarray(x, dtype=float).ravel()

    # Validate the input.
    if x.size == 0:
        raise ValueError("At least one observation is required.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Observations must be finite.")

    # Calculate the number of observations, range, and standard deviation.
    N = x.size
    data_range = np.max(x) - np.min(x)
    sigma = np.std(x)

    # Constant-valued data requires only one bin.
    if data_range == 0 or sigma == 0:
        return 1

    # Calculate the recommended number of bins using Scott's rule.
    B = (N ** (1.0 / 3.0) * data_range) / (3.5 * sigma)

    # Round upward and ensure at least one bin.
    return int(max(1, np.ceil(B)))


def bin_index(x, bins, strategy="width"):
    """
    Convert observations into integer histogram bin indices.

    Parameters:
        x: Array of observations.
        bins: Number of bins.
        strategy: Binning strategy:
            "width": Equal-width intervals.
            "quantile": Approximately equal-frequency bins.

    Returns:
        Integer NumPy array containing bin indices
        between 0 and bins - 1.
    """

    # Convert observations into a flattened NumPy array.
    x = np.asarray(x, dtype=float)

    original_shape = x.shape
    x = x.ravel()

    # Validate the input.
    if x.size == 0:
        raise ValueError("At least one observation is required.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Observations must be finite.")

    if not isinstance(bins, (int, np.integer)) or isinstance(bins, (bool, np.bool_)) or bins < 1:
        raise ValueError("bins must be a positive integer.")

    if strategy not in ("width", "quantile"):
        raise ValueError("strategy must be 'width' or 'quantile'.")

    # Equal-width binning.
    if strategy == "width":

        # Calculate the minimum and maximum observations.
        xmin = np.min(x)
        xmax = np.max(x)

        # Handle constant-valued observations.
        if xmax == xmin:
            indices = np.zeros(x.size, dtype=np.int64)

        else:
            # Assign observations to equal-width intervals.
            indices = np.floor(
                (x - xmin) / (xmax - xmin) * bins
            ).astype(np.int64)

            # Ensure the maximum observation belongs to the last bin.
            indices = np.clip(indices, 0, bins - 1)

    # Quantile binning.
    else:

        N = x.size

        # Obtain stable ranks, breaking ties by original position.
        order = np.argsort(x, kind="stable")

        ranks = np.empty(N, dtype=np.int64)
        ranks[order] = np.arange(N)

        # Assign ranks to approximately equal-frequency bins.
        indices = ranks * bins // N

        # Ensure all indices are within the valid range.
        indices = np.clip(indices, 0, bins - 1)

    # Restore the original input shape.
    return indices.reshape(original_shape)


def joint_table(a, b, na=None, nb=None):
    """
    Construct a normalized joint contingency table.

    Parameters:
        a: Integer bin indices for the first variable.
        b: Integer bin indices for the second variable.
        na: Number of possible states for the first variable.
            Defaults to max(a) + 1.
        nb: Number of possible states for the second variable.
            Defaults to max(b) + 1.

    Returns:
        Normalized joint probability table of shape (na, nb).
    """

    # Convert both inputs into flattened arrays.
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    # Ensure both arrays have the same number of observations.
    if a.size != b.size:
        raise ValueError("Both arrays must have the same number of elements.")

    if a.size == 0:
        raise ValueError("At least one observation is required.")

    # Validate that the indices are finite, non-negative integers.
    if (not np.issubdtype(a.dtype, np.integer) or
            not np.issubdtype(b.dtype, np.integer) or
            np.any(a < 0) or np.any(b < 0)):
        raise ValueError("Bin indices must be non-negative integers.")

    # Determine the number of states when not explicitly provided.
    if na is None:
        na = int(np.max(a)) + 1

    if nb is None:
        nb = int(np.max(b)) + 1

    # Validate the table dimensions.
    if (not isinstance(na, (int, np.integer)) or
            isinstance(na, (bool, np.bool_)) or na < 1):
        raise ValueError("na must be a positive integer.")

    if (not isinstance(nb, (int, np.integer)) or
            isinstance(nb, (bool, np.bool_)) or nb < 1):
        raise ValueError("nb must be a positive integer.")

    # Ensure all indices fit within the requested table dimensions.
    if np.any(a >= na) or np.any(b >= nb):
        raise ValueError("Bin indices exceed the specified table dimensions.")

    # Initialize the joint count table.
    counts = np.zeros((na, nb), dtype=float)

    # Count the occurrences of each pair of states.
    np.add.at(counts, (a, b), 1.0)

    # Normalize counts to obtain joint probabilities.
    return counts / a.size


def histogram_mi(x, y, bins_x=None, bins_y=None,
                 strategy="width", correct=False):
    """
    Estimate mutual information using a histogram plug-in estimator.

    Parameters:
        x: Observations of the first variable.
        y: Observations of the second variable.
        bins_x: Number of bins for X. None uses Scott's rule.
        bins_y: Number of bins for Y. None uses Scott's rule.
        strategy: Binning strategy ('width' or 'quantile').
        correct: Whether to apply the Miller-Madow bias correction.

    Returns:
        Python float representing estimated mutual information in nats.
    """

    # Convert observations into flattened NumPy arrays.
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # Validate the observations.
    if x.size != y.size:
        raise ValueError("Both variables must have the same number of observations.")

    if x.size == 0:
        raise ValueError("At least one observation is required.")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Observations must be finite.")

    if strategy not in ("width", "quantile"):
        raise ValueError("strategy must be 'width' or 'quantile'.")

    # Use Scott's rule whenever the number of bins is not specified.
    if bins_x is None:
        bins_x = scott_bins(x)

    if bins_y is None:
        bins_y = scott_bins(y)

    # Discretize both variables using the selected strategy.
    a = bin_index(x, bins_x, strategy)
    b = bin_index(y, bins_y, strategy)

    # Construct the normalized joint probability table.
    # Explicit dimensions preserve all requested histogram bins.
    joint = joint_table(
        a,
        b,
        na=bins_x,
        nb=bins_y
    )

    # Calculate plug-in mutual information in nats.
    mi = float(entropy_table(joint, base=np.e)["I_XY"])

    # Optionally apply the Miller-Madow bias correction.
    if correct:

        # Count non-empty cells in the joint table.
        K_xy = np.count_nonzero(joint)

        # Count non-empty states in each marginal distribution.
        K_x = np.count_nonzero(np.sum(joint, axis=1))
        K_y = np.count_nonzero(np.sum(joint, axis=0))

        # Calculate the number of observations.
        N = x.size

        # Calculate and subtract the Miller-Madow bias.
        correction = (K_xy - K_x - K_y + 1) / (2.0 * N)

        mi -= correction

    return float(mi)

# Step 21 - ksg_mi
EULER_GAMMA = 0.5772156649015329


def digamma_table(n_max):
    """
    Precompute digamma function values for integer arguments.

    Uses the recurrence:

        psi(1) = -gamma
        psi(n) = psi(n-1) + 1/(n-1)

    Parameters:
        n_max: Maximum integer argument.

    Returns:
        Float NumPy array of length n_max + 1.
        Index 0 is initialized to 0 and is not used.
    """

    # Validate the maximum index.
    if (not isinstance(n_max, (int, np.integer)) or
            isinstance(n_max, (bool, np.bool_)) or n_max < 0):
        raise ValueError("n_max must be a non-negative integer.")

    # Initialize the digamma lookup table.
    psi = np.zeros(n_max + 1, dtype=float)

    # Calculate the first digamma value.
    if n_max >= 1:
        psi[1] = -EULER_GAMMA

    # Calculate remaining values using the recurrence relation.
    for n in range(2, n_max + 1):
        psi[n] = psi[n - 1] + 1.0 / (n - 1)

    return psi


def ksg_mi(x, y, k=3):
    """
    Estimate mutual information using the KSG k-nearest-neighbour
    estimator (maximum-norm variant).

    Parameters:
        x: First variable, shape (n,) or (n, dx).
        y: Second variable, shape (n,) or (n, dy).
        k: Number of nearest neighbours. Default is 3.

    Returns:
        Python float representing estimated mutual information in nats.

    Notes:
        - Uses the maximum (Chebyshev) norm.
        - Excludes the observation itself when finding neighbours.
        - Counts marginal neighbours strictly inside the joint radius.
        - Does not clip small negative estimates.
    """

    # Convert both inputs into NumPy arrays.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Convert one-dimensional inputs into column vectors.
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Validate the input dimensions.
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Inputs must have shape (n,) or (n, d).")

    if x.shape[1] == 0 or y.shape[1] == 0:
        raise ValueError("Each variable must have at least one dimension.")

    # Both variables must contain the same number of observations.
    if x.shape[0] != y.shape[0]:
        raise ValueError("Both variables must have the same number of observations.")

    # Validate the observations.
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Observations must contain only finite values.")

    # Determine the number of observations.
    n = x.shape[0]

    # Validate the number of neighbours.
    if (not isinstance(k, (int, np.integer)) or
            isinstance(k, (bool, np.bool_)) or k < 1 or k >= n):
        raise ValueError("k must be a positive integer smaller than n.")

    # Precompute the digamma values required by the estimator.
    psi = digamma_table(n)

    # Initialize arrays for marginal neighbour counts.
    nx = np.zeros(n, dtype=np.int64)
    ny = np.zeros(n, dtype=np.int64)

    # Create an index array to exclude each observation itself.
    not_self = np.arange(n)

    # Calculate neighbourhood distances for every observation.
    for i in range(n):

        # Calculate maximum-norm distances in the X space.
        dx = np.max(
            np.abs(x - x[i]),
            axis=1
        )

        # Calculate maximum-norm distances in the Y space.
        dy = np.max(
            np.abs(y - y[i]),
            axis=1
        )

        # Calculate joint-space distances using the maximum norm.
        # d((xi, yi), (xj, yj)) = max(dx, dy).
        d_joint = np.maximum(dx, dy)

        # Exclude the observation itself from the neighbour search.
        d_joint[i] = np.inf

        # Find the distance to the k-th nearest neighbour.
        # np.partition avoids sorting the entire distance array.
        epsilon = np.partition(d_joint, k - 1)[k - 1]

        # Count other observations strictly inside the joint radius
        # in the X space.
        nx[i] = np.count_nonzero(
            (dx < epsilon) & (not_self != i)
        )

        # Count other observations strictly inside the joint radius
        # in the Y space.
        ny[i] = np.count_nonzero(
            (dy < epsilon) & (not_self != i)
        )

    # Apply the KSG mutual information estimator.
    # Digamma values are retrieved from the precomputed lookup table.
    mi = (
        psi[k]
        + psi[n]
        - np.mean(psi[nx + 1] + psi[ny + 1])
    )

    # Return the estimate without clipping negative values.
    return float(mi)

# Step 22 - estimator_showdown
def additive_noise_mi(f, sigma, n_x=4001, n_y=8001):
    """
    Numerically calculate mutual information for:

        X ~ N(0, 1)
        Y = f(X) + sigma * noise

    Computes I(X;Y) = h(Y) - h(Y|X).
    """

    # Create a grid for the standard normal variable X.
    xs = np.linspace(-6.0, 6.0, n_x)

    # Calculate the standard normal probability density.
    px = np.exp(-0.5 * xs ** 2) / np.sqrt(2.0 * np.pi)

    dx = xs[1] - xs[0]

    # Apply the nonlinear transformation.
    fx = f(xs)

    # Construct the integration grid for Y.
    ys = np.linspace(
        fx.min() - 8.0 * sigma,
        fx.max() + 8.0 * sigma,
        n_y
    )

    dy = ys[1] - ys[0]

    # Initialize the marginal probability density of Y.
    py = np.zeros(n_y)

    # Numerically integrate over X in chunks.
    for i in range(0, n_x, 500):
        chunk = fx[i:i + 500]

        py += np.sum(
            px[i:i + 500, None]
            * np.exp(
                -0.5 * ((ys[None, :] - chunk[:, None]) / sigma) ** 2
            ),
            axis=0
        )

    # Normalize the conditional Gaussian densities.
    py *= dx / (sigma * np.sqrt(2.0 * np.pi))

    # Calculate differential entropy h(Y).
    h_y = -np.sum(
        py[py > 0] * np.log(py[py > 0])
    ) * dy

    # Calculate conditional entropy h(Y | X).
    h_y_given_x = gaussian_entropy(sigma ** 2)

    return float(h_y - h_y_given_x)


def estimator_showdown(n=1000, rhos=(0.0, 0.3, 0.6, 0.9, 0.99),
                       n_repeats=5, seed=0):
    """
    Compare six mutual information estimators against known
    theoretical values across Gaussian, warped, and nonlinear data.

    Returns:
        Dictionary containing:
            rows: Mean estimates for every scenario.
            mae: Mean absolute errors grouped by family and method.
    """

    # Define the six estimators in the required order.
    methods = [
        "fixed10",
        "scott",
        "scott_mm",
        "quantile10",
        "ksg3",
        "gauss_formula"
    ]

    # Initialize the experimental scenarios.
    scenarios = []

    # --------------------------------------------------
    # Family 1: Gaussian relationships.
    # --------------------------------------------------
    for rho in rhos:
        scenarios.append({
            "name": f"gauss {rho}",
            "family": "gaussian",
            "type": "gaussian",
            "rho": rho,
            # Remove negative zero from the theoretical truth.
            "truth": float(max(0.0, gaussian_mi(rho)))
        })

    # --------------------------------------------------
    # Family 2: Warped Gaussian relationships.
    # --------------------------------------------------
    for rho in rhos:
        scenarios.append({
            "name": f"warped {rho}",
            "family": "warped",
            "type": "warped",
            "rho": rho,
            # Mutual information is preserved by invertible transformations.
            "truth": float(max(0.0, gaussian_mi(rho)))
        })

    # --------------------------------------------------
    # Family 3: Nonlinear relationships.
    # --------------------------------------------------
    nonlinear_functions = [
        ("x^2", lambda t: t ** 2, 0.3),
        ("sin 3x", lambda t: np.sin(3 * t), 0.2),
        ("|x|", np.abs, 0.1)
    ]

    for name, f, sigma in nonlinear_functions:

        # Calculate true mutual information numerically.
        truth = additive_noise_mi(f, sigma)

        scenarios.append({
            "name": name,
            "family": "nonlinear",
            "type": "nonlinear",
            "function": f,
            "sigma": sigma,
            "truth": float(truth)
        })

    # Initialize result storage.
    rows = []

    # Initialize error storage by family and estimator.
    errors = {
        family: {method: [] for method in methods}
        for family in ("gaussian", "warped", "nonlinear")
    }

    # --------------------------------------------------
    # Evaluate all experimental scenarios.
    # --------------------------------------------------
    for i, scenario in enumerate(scenarios):

        truth = scenario["truth"]

        # Store estimates from each repetition.
        estimates = {
            method: [] for method in methods
        }

        for r in range(n_repeats):

            # Generate the required reproducible seed.
            s = seed + 1000 * r + 17 * i

            # Generate data according to the scenario.
            if scenario["type"] == "gaussian":

                x, y = sample_gaussian_pair(
                    n,
                    scenario["rho"],
                    s
                )

            elif scenario["type"] == "warped":

                x, y = sample_gaussian_pair(
                    n,
                    scenario["rho"],
                    s
                )

                # Apply invertible transformations.
                x = np.exp(x)
                y = y ** 3

            else:

                # Generate nonlinear data with Gaussian noise.
                rng = np.random.default_rng(s)

                x = rng.standard_normal(n)

                y = (
                    scenario["function"](x)
                    + scenario["sigma"] * rng.standard_normal(n)
                )

            # Method 1: Fixed 10-bin histogram estimator.
            estimates["fixed10"].append(
                histogram_mi(x, y, 10, 10)
            )

            # Method 2: Scott's rule histogram estimator.
            estimates["scott"].append(
                histogram_mi(x, y)
            )

            # Method 3: Scott's rule with Miller-Madow correction.
            estimates["scott_mm"].append(
                histogram_mi(x, y, correct=True)
            )

            # Method 4: Quantile histogram estimator.
            estimates["quantile10"].append(
                histogram_mi(
                    x, y, 10, 10, strategy="quantile"
                )
            )

            # Method 5: KSG estimator with k = 3.
            estimates["ksg3"].append(
                ksg_mi(x, y, 3)
            )

            # Method 6: Gaussian closed-form estimator.
            sample_rho = float(
                np.corrcoef(x, y)[0, 1]
            )

            estimates["gauss_formula"].append(
                gaussian_mi(sample_rho)
            )

        # --------------------------------------------------
        # Calculate mean estimates and errors.
        # --------------------------------------------------
        row = {
            "name": scenario["name"],
            "family": scenario["family"],
            "truth": float(truth)
        }

        for method in methods:

            values = np.asarray(estimates[method], dtype=float)

            # Calculate the average estimate across repetitions.
            row[method] = float(np.mean(values))

            # Calculate mean absolute error.
            scenario_error = float(
                np.mean(np.abs(values - truth))
            )

            errors[scenario["family"]][method].append(
                scenario_error
            )

        rows.append(row)

    # --------------------------------------------------
    # Calculate mean absolute error for each family.
    # --------------------------------------------------
    mae = {}

    for family in errors:

        mae[family] = {}

        for method in methods:
            mae[family][method] = float(
                np.mean(errors[family][method])
            )

    return {
        "rows": rows,
        "mae": mae
    }

# Step 23 - optimal_x_partition
def optimal_x_partition(x, y_labels, k_max, clumps_per_col=5):
    """
    Find optimal vertical grid partitions that maximize mutual
    information between grid columns and categorical labels.

    Uses dynamic programming to minimize conditional entropy.

    Parameters:
        x: Continuous observations.
        y_labels: Categorical labels corresponding to observations.
        k_max: Maximum number of grid columns.
        clumps_per_col: Maximum number of clumps per column.

    Returns:
        NumPy float array of length k_max + 1.

        out[k] contains the maximum mutual information in bits
        achievable using k contiguous grid columns.
        out[0] is always zero.
    """

    # Convert inputs into flattened NumPy arrays.
    x = np.asarray(x, dtype=float).ravel()
    y_labels = np.asarray(y_labels).ravel()

    # Validate the observations.
    if x.size == 0:
        raise ValueError("At least one observation is required.")

    if x.size != y_labels.size:
        raise ValueError("x and y_labels must have the same length.")

    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain finite values.")

    # Validate the grid parameters.
    if (not isinstance(k_max, (int, np.integer)) or
            isinstance(k_max, (bool, np.bool_)) or k_max < 1):
        raise ValueError("k_max must be a positive integer.")

    if (not isinstance(clumps_per_col, (int, np.integer)) or
            isinstance(clumps_per_col, (bool, np.bool_)) or
            clumps_per_col < 1):
        raise ValueError("clumps_per_col must be a positive integer.")

    # Encode arbitrary categorical labels as consecutive integers.
    # This supports binary, three-class, and multiclass problems.
    _, y_encoded = np.unique(
        y_labels,
        return_inverse=True
    )

    n = x.size

    # Determine the number of distinct label categories.
    n_classes = int(np.max(y_encoded)) + 1

    # Sort observations by x using stable sorting.
    order = np.argsort(x, kind="stable")

    y_sorted = y_encoded[order]

    # Determine the number of clumps.
    m = min(n, clumps_per_col * k_max)

    # Assign each sorted observation to a clump.
    clump_ids = np.arange(n) * m // n

    # Count observations belonging to every class in each clump.
    counts = np.zeros((m, n_classes), dtype=float)

    np.add.at(
        counts,
        (clump_ids, y_sorted),
        1.0
    )

    # Calculate cumulative class counts for efficient interval queries.
    prefix = np.vstack((
        np.zeros((1, n_classes)),
        np.cumsum(counts, axis=0)
    ))

    # Calculate the overall entropy of the target labels.
    overall_counts = np.sum(counts, axis=0)

    H_Y = entropy(
        overall_counts / n,
        base=2
    )

    # --------------------------------------------------
    # Precompute the weighted conditional entropy cost
    # for every possible consecutive interval of clumps.
    # --------------------------------------------------

    cost = np.zeros((m, m + 1), dtype=float)

    for a in range(m):

        for b in range(a + 1, m + 1):

            # Retrieve class counts for clumps a through b - 1.
            interval_counts = prefix[b] - prefix[a]

            # Calculate the number of observations in the interval.
            n_ab = np.sum(interval_counts)

            if n_ab == 0:
                cost[a, b] = 0.0
                continue

            # Calculate the conditional class distribution.
            probabilities = interval_counts / n_ab

            # Calculate weighted conditional entropy.
            cost[a, b] = (
                (n_ab / n)
                * entropy(probabilities, base=2)
            )

    # --------------------------------------------------
    # Dynamic programming to find optimal partitions.
    # dp[k, b] = minimum conditional entropy when the first
    # b clumps are partitioned into k non-empty columns.
    # --------------------------------------------------

    max_k = min(k_max, m)

    dp = np.full(
        (max_k + 1, m + 1),
        np.inf,
        dtype=float
    )

    # Base case: zero clumps have zero cost.
    dp[0, 0] = 0.0

    # Evaluate every possible number of columns.
    for k in range(1, max_k + 1):

        for b in range(k, m + 1):

            best_cost = np.inf

            # Try every valid position for the last partition boundary.
            for a in range(k - 1, b):

                candidate = (
                    dp[k - 1, a] + cost[a, b]
                )

                if candidate < best_cost:
                    best_cost = candidate

            dp[k, b] = best_cost

    # --------------------------------------------------
    # Convert minimum conditional entropy into mutual information.
    # I(X_grid; Y) = H(Y) - H(Y | X_grid).
    # --------------------------------------------------

    out = np.zeros(k_max + 1, dtype=float)

    for k in range(1, k_max + 1):

        out[k] = max(
            0.0,
            H_Y - dp[k, m]
        )

    # Preserve the required output length when k_max > n.
    result = np.zeros(k_max + 1, dtype=float)
    result[:max_k + 1] = out

    return result

# Step 24 - mic
def mic(x, y, alpha=0.6, clumps_per_col=5):
    """
    Calculate the Maximal Information Coefficient (MIC)
    and its characteristic matrix.

    Parameters:
        x: First continuous variable.
        y: Second continuous variable.
        alpha: Controls the maximum grid resolution. Default is 0.6.
        clumps_per_col: Maximum number of clumps per column. Default is 5.

    Returns:
        Tuple containing:
            value: Maximum normalized mutual information (Python float).
            M: Characteristic matrix containing normalized scores.
    """

    # Convert both variables into flattened NumPy arrays.
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # Validate the input observations.
    if x.size == 0 or y.size == 0:
        raise ValueError("Both variables must contain observations.")

    if x.size != y.size:
        raise ValueError("Both variables must have the same number of observations.")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Observations must contain finite values.")

    # Validate alpha and clumps_per_col.
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite.")

    if (not isinstance(clumps_per_col, (int, np.integer)) or
            isinstance(clumps_per_col, (bool, np.bool_)) or
            clumps_per_col < 1):
        raise ValueError("clumps_per_col must be a positive integer.")

    # Determine the maximum grid resolution.
    n = x.size
    B = max(n ** alpha, 4.0)

    # Determine the characteristic matrix dimensions.
    size = int(B // 2) + 1

    # Initialize the characteristic matrix.
    M = np.zeros((size, size), dtype=float)

    # Perform two passes:
    # Pass 1: Optimize the X-axis and quantile-bin Y.
    # Pass 2: Optimize the Y-axis and quantile-bin X.
    for pass_idx, (a, b) in enumerate(((x, y), (y, x))):

        # Iterate through possible numbers of rows.
        for rows in range(2, size):

            # Determine the maximum allowed number of columns.
            k_max = min(int(B // rows), size - 1)

            # Stop this pass when fewer than two columns are allowed.
            if k_max < 2:
                break

            # Equiquantile binning of the other variable.
            labels = bin_index(
                b,
                rows,
                strategy="quantile"
            )

            # Find the optimal vertical partition using dynamic programming.
            best = optimal_x_partition(
                a,
                labels,
                k_max,
                clumps_per_col
            )

            # Evaluate normalized mutual information for every grid size.
            for k in range(2, k_max + 1):

                # Normalize by the logarithm of the smaller grid dimension.
                normalizer = np.log2(min(k, rows))

                score = float(best[k] / normalizer)

                # Update the characteristic matrix.
                # The second pass fills the transposed position.
                if pass_idx == 0:
                    M[k, rows] = max(M[k, rows], score)
                else:
                    M[rows, k] = max(M[rows, k], score)

    # Return the maximum MIC score and characteristic matrix.
    return float(np.max(M)), M

# Step 25 - relationship_zoo
# Define the five functional relationships.
ZOO = {
    "line": lambda t: t,
    "parabola": lambda t: t ** 2,
    "cubic": lambda t: 4.0 * t ** 3 - 3.0 * t,
    "sine": lambda t: np.sin(4.0 * np.pi * t),
    "step": lambda t: np.sign(t) * (np.abs(t) > 0.3),
}


def zoo_sample(name, n, seed, noise=0.0):
    """
    Generate samples for a specified relationship.

    Parameters:
        name: Name of the relationship.
        n: Number of observations.
        seed: Random seed for reproducibility.
        noise: Relative noise level.

    Returns:
        Tuple containing the generated X and Y arrays.
    """

    rng = np.random.default_rng(seed)

    # Generate circular data.
    if name == "circle":
        t = rng.uniform(0.0, 2.0 * np.pi, n)
        r = 1.0 + noise * rng.standard_normal(n)

        return r * np.cos(t), r * np.sin(t)

    # Generate X-shaped data.
    if name == "x_shape":
        t = rng.uniform(-1.0, 1.0, n)
        s = rng.choice([-1.0, 1.0], n)

        return t, s * t + noise * rng.standard_normal(n)

    # Generate independent data.
    if name == "independent":
        return (
            rng.uniform(-1.0, 1.0, n),
            rng.uniform(-1.0, 1.0, n)
        )

    # Generate samples for the five functional relationships.
    t = rng.uniform(-1.0, 1.0, n)

    fx = ZOO[name](t)

    # Add Gaussian noise proportional to the standard deviation
    # of the noiseless transformed output.
    return (
        t,
        fx + noise * fx.std() * rng.standard_normal(n)
    )


def relationship_zoo(n=500, seed=0, noise=0.0):
    """
    Compare squared Pearson correlation and MIC across
    eight different types of relationships.

    Parameters:
        n: Number of observations per relationship. Default is 500.
        seed: Random seed for reproducibility. Default is 0.
        noise: Noise level for the generated relationships. Default is 0.0.

    Returns:
        Dictionary mapping each relationship name to another
        dictionary containing:

            r2: Squared sample correlation (Python float).
            mic: Maximal information coefficient (Python float).
    """

    # Define all eight relationships in the required order.
    names = [
        "line",
        "parabola",
        "cubic",
        "sine",
        "step",
        "circle",
        "x_shape",
        "independent"
    ]

    # Initialize the result dictionary.
    results = {}

    # Evaluate every relationship.
    for name in names:

        # Generate reproducible samples for the current relationship.
        x, y = zoo_sample(
            name,
            n,
            seed,
            noise
        )

        # Calculate the squared sample correlation.
        correlation = np.corrcoef(x, y)[0, 1]
        r2 = float(correlation ** 2)

        # Calculate the maximal information coefficient.
        mic_value, _ = mic(x, y)

        # Store both measurements.
        results[name] = {
            "r2": r2,
            "mic": float(mic_value)
        }

    return results

# Step 26 - conditional_mi
def bsc(eps):
    """
    Construct the transition matrix of a Binary Symmetric Channel.

    Parameters:
        eps: Probability of flipping the input bit.

    Returns:
        A 2x2 NumPy array representing the channel transition matrix.
    """

    # Validate the channel error probability.
    if not np.isfinite(eps) or not 0.0 <= eps <= 1.0:
        raise ValueError("eps must be between 0 and 1.")

    # Construct the Binary Symmetric Channel transition matrix.
    return np.array([
        [1.0 - eps, eps],
        [eps, 1.0 - eps]
    ], dtype=float)


def channel_chain(p_x, A, B):
    """
    Construct the joint distribution P(X, Y, Z)
    for the Markov chain X -> Y -> Z.

    Parameters:
        p_x: Probability distribution of X.
        A: Transition matrix P(Y | X).
        B: Transition matrix P(Z | Y).

    Returns:
        Three-dimensional NumPy array P[x, y, z].
    """

    # Convert inputs into NumPy arrays.
    p_x = np.asarray(p_x, dtype=float).ravel()
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    # Validate the dimensions of the input distributions.
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be two-dimensional transition matrices.")

    if A.shape[0] != p_x.size:
        raise ValueError("The rows of A must correspond to the states of X.")

    if A.shape[1] != B.shape[0]:
        raise ValueError("The states of Y must match between A and B.")

    # Validate probability values.
    if (not np.all(np.isfinite(p_x)) or
            not np.all(np.isfinite(A)) or
            not np.all(np.isfinite(B)) or
            np.any(p_x < 0) or np.any(A < 0) or np.any(B < 0)):
        raise ValueError("All probabilities must be finite and non-negative.")

    # Ensure p_x is a valid probability distribution.
    if not np.isclose(np.sum(p_x), 1.0):
        raise ValueError("p_x must sum to 1.")

    # Ensure each row of the transition matrices is normalized.
    if not np.allclose(np.sum(A, axis=1), 1.0):
        raise ValueError("Each row of A must sum to 1.")

    if not np.allclose(np.sum(B, axis=1), 1.0):
        raise ValueError("Each row of B must sum to 1.")

    # Construct the joint distribution:
    # P[x, y, z] = p(x) * A[x, y] * B[y, z].
    P = (
        p_x[:, None, None]
        * A[:, :, None]
        * B[None, :, :]
    )

    return P


def conditional_mi(P, base=2):
    """
    Calculate conditional mutual information I(X; Y | Z)
    from a three-dimensional joint probability table.

    I(X; Y | Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)

    Parameters:
        P: Three-dimensional joint probability table P[x, y, z].
        base: Logarithm base. Default is 2 (bits).

    Returns:
        Python float representing conditional mutual information.
    """

    # Convert the joint probability table into a NumPy array.
    P = np.asarray(P, dtype=float)

    # Ensure the input is three-dimensional.
    if P.ndim != 3:
        raise ValueError("P must be a three-dimensional joint probability table.")

    # Marginalize over Y to obtain P(X, Z).
    P_xz = np.sum(P, axis=1)

    # Marginalize over X to obtain P(Y, Z).
    P_yz = np.sum(P, axis=0)

    # Marginalize over X and Y to obtain P(Z).
    P_z = np.sum(P, axis=(0, 1))

    # Calculate the required entropy quantities.
    H_xz = entropy(P_xz, base=base)
    H_yz = entropy(P_yz, base=base)
    H_z = entropy(P_z, base=base)
    H_xyz = entropy(P, base=base)

    # Calculate conditional mutual information.
    I_xyz = H_xz + H_yz - H_z - H_xyz

    # Eliminate small negative values caused by floating-point rounding.
    return float(max(0.0, I_xyz))

