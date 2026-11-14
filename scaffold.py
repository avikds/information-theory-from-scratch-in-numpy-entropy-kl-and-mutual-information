"""
Information Theory from Scratch in NumPy: Entropy, KL and Mutual Information scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Information theory from scratch: entropy, KL and mutual information.

Chapter 6 of Murphy, Probabilistic Machine Learning, run on real data. Entropy becomes actual bits
when you compress names with your own Huffman codes. KL divergence becomes the extra cost of the
wrong code, and two different one-Gaussian fits to the same bimodal target. Mutual information gets
a closed form, three estimators, and an exam with known answers. Then the data processing
inequality, sufficient statistics and Fano's bound close the chapter on tumour features and digits.
"""
import bz2
import lzma
import zlib

import numpy as np
import dml_data


def main() -> None:
    splits = load_names(0)

    print("1. Perplexity of names (section 6.1.5): the effective number of next characters")
    print("   order  alpha   train perplexity  test perplexity")
    for r in ngram_ladder(splits):
        print(f"   {r['order']:>5}  {str(r['alpha']):>5}   {r['train']:>16.2f}  {r['test']:>15.2f}")
    print("   Each symbol of context helps by less than the one before, and the train-test gap keeps growing.")

    print("\n2. Real bits per character on held-out names (section 6.1.2)")
    c = compress_names(splits)
    for name in ("unigram", "bigram", "trigram"):
        print(f"   {name:8s} Huffman {c[name + '_bits']:.3f} bits, model cross entropy {c[name + '_ce']:.3f} bits")
    names = dml_data.load("names").names
    order = np.random.default_rng(0).permutation(len(names))
    text = "\n".join(names[i] for i in order[int(0.9 * len(names)):]).encode()
    sizes = {"zlib": zlib.compress(text, 9), "lzma": lzma.compress(text, preset=9), "bz2": bz2.compress(text, 9)}
    print("   general-purpose compressors: " + ", ".join(f"{k} {len(v) * 8 / len(text):.2f}" for k, v in sizes.items()))
    print(f"   fixed-length code: {c['fixed_bits']:.0f} bits. The model tables are trained in advance and not counted here.")

    print("\n3. Joint tables (sections 6.1.3, 6.1.4, 6.3.3)")
    t = entropy_table(np.array([[1 / 8, 3 / 8], [3 / 8, 1 / 8]]))
    print(f"   even/prime: H(X,Y) {t['H_XY']:.2f}, H(Y|X) {t['H_Y_given_X']:.2f}, I(X;Y) {t['I_XY']:.2f} bits")
    ex = np.array([[1 / 8, 1 / 16, 1 / 32, 1 / 32], [1 / 16, 1 / 8, 1 / 32, 1 / 32], [1 / 16, 1 / 16, 1 / 16, 1 / 16], [1 / 4, 0, 0, 0]]).T
    t = entropy_table(ex)
    print(f"   Exercise 6.3: H(X) {t['H_X']:.2f}, H(X|Y=y) {t['H_X_given_y'].tolist()}: observing y = 3 raises the entropy of X")
    ev = ngram_events(splits["train"], 2)
    J = np.zeros((27, 27))
    np.add.at(J, (ev[:, 0], ev[:, 1]), 1.0)
    J /= J.sum()
    t = entropy_table(J)
    per_prev = entropy_table(J.T)["H_X_given_y"]
    print(f"   consecutive letters in names: H(next) {t['H_Y']:.2f}, H(next|prev) {t['H_Y_given_X']:.2f}, I {t['I_XY']:.2f} bits")
    print(f"   after q: {per_prev[ALPHABET.index('q')]:.2f} bits. At the start of a name: {per_prev[0]:.2f} bits, above the average.")

    print("\n4. Sequence logos and the cost of estimating entropy (sections 6.1.1.1, 6.1.1.2)")
    _, info, cons = sequence_logo(make_motif_alignment(10000, 1))
    print(f"   consensus {cons}, column heights {np.round(info, 2).tolist()}, total {info.sum():.2f} bits")
    z = 1.0 / np.arange(1, 20001)
    z /= z.sum()
    rng = np.random.default_rng(0)
    print(f"   Zipf over 20,000 words has {entropy(z):.2f} bits. Plug-in / Miller-Madow estimates by sample size:")
    for n in (100, 1000, 10000, 100000, 1000000):
        plug, mm = np.mean([entropy_estimates(rng.multinomial(n, z)) for _ in range(5)], axis=0)
        print(f"   N = {n:>7}: {plug:.2f} / {mm:.2f}")

    print("\n5. Forward vs reverse KL on a bimodal target (section 6.2.6, Figure 6.3)")
    r = kl_both_ways()
    print(f"   forward fit: KL(p||q) {r['forward']['kl_p_q']:.2f} nats, KL(q||p) {r['forward']['kl_q_p']:.2f} nats, mean {(np.round(r['forward']['mu'], 1) + 0.0).tolist()}")
    print(f"   reverse fit: KL(p||q) {r['reverse']['kl_p_q']:.1f} nats, KL(q||p) {r['reverse']['kl_q_p']:.2f} nats, mean {(np.round(r['reverse']['mu'], 1) + 0.0).tolist()}")
    print(f"   log 2 = {np.log(2):.3f}: the cost of giving up on one of two equal modes")
    ex64 = np.array([[1 / 8, 1 / 8, 0, 0], [1 / 8, 1 / 8, 0, 0], [0, 0, 1 / 4, 0], [0, 0, 0, 1 / 4]])
    found = {}
    for s in range(100):
        qx, qy, kl = mean_field_reverse_kl(ex64, seed=s)
        key = (tuple(np.flatnonzero(qx).tolist()), round(kl, 3))
        found[key] = found.get(key, 0) + 1
    print(f"   Exercise 6.4, 100 random starts, (support, reverse KL in bits): count {found}")
    print(f"   at the product of marginals: reverse KL {kl_divergence(np.outer(ex64.sum(1), ex64.sum(0)), ex64)}, forward KL = I(X;Y) = {kl_divergence(ex64, np.outer(ex64.sum(1), ex64.sum(0))):.2f} bits")

    print("\n6. Mutual information estimators against known truths (section 6.3.5)")
    x, y = sample_gaussian_pair(500, 0.6, 1)
    print(f"   one sample, rho 0.6, truth {gaussian_mi(0.6):.3f} nats. Equal-width bins: " + ", ".join(f"{b}: {histogram_mi(x, y, b, b):.2f}" for b in (2, 4, 8, 16, 32, 64)) + f"; KSG {ksg_mi(x, y):.3f}")
    sd = estimator_showdown()
    print("   mean absolute error in nats      " + "  ".join(f"{m:>13s}" for m in sd["mae"]["gaussian"]))
    for family, row in sd["mae"].items():
        print(f"   {family:30s}  " + "  ".join(f"{v:>13.3f}" for v in row.values()))

    print("\n7. Correlation vs MIC at n = 500 (section 6.3.7, Figure 6.6)")
    zoo = relationship_zoo(500, 0)
    for name, row in zoo.items():
        print(f"   {name:12s} R^2 {row['r2']:.3f}   MIC {row['mic']:.3f}")
    noisy = relationship_zoo(500, 1, 1.0)
    vals = [noisy[k]["mic"] for k in ("line", "parabola", "cubic", "sine", "step")]
    print(f"   equal noise (R^2 of the fit 0.5): MIC from {min(vals):.2f} to {max(vals):.2f} across the functional shapes on this one sample; over 30 seeds the means sit between 0.41 and 0.44")

    print("\n8. Data processing and sufficiency (sections 6.3.8, 6.3.9)")
    P = channel_chain([0.5, 0.5], bsc(0.1), bsc(0.2))
    ixy, ixz = entropy_table(P.sum(axis=2))["I_XY"], entropy_table(P.sum(axis=1))["I_XY"]
    print(f"   X -> Y -> Z: I(X;Y) {ixy:.3f} >= I(X;Z) {ixz:.3f}, gap = I(X;Y|Z) = {conditional_mi(P):.3f} bits")
    thetas = np.linspace(0.1, 0.9, 9)
    stats = [("the full sequence", None), ("number of heads", lambda s: int(s.sum())), ("first half", lambda s: int(s[:5].sum())),
             ("majority vote", lambda s: int(s.sum() > 5)), ("first flip", lambda s: int(s[0])), ("parity", lambda s: int(s.sum()) % 2)]
    for name, f in stats:
        print(f"   I(theta; {name}) = {sufficient_statistic_mi(10, f, thetas):.4f} bits")

    print("\n9. Feature selection and Fano's bound (section 6.3.10)")
    bc = dml_data.load("breast-cancer")
    order_bc, mi_bc = mi_rank(bc.data, bc.target)
    print("   breast cancer, top by I(X;Y): " + ", ".join(f"{bc.feature_names[i]} {v:.3f}" for i, v in zip(order_bc[:3], mi_bc[:3])))
    Q = np.stack([bin_index(bc.data[:, j], 4, "quantile") for j in range(30)], axis=1).astype(np.int64)
    perm = np.random.default_rng(0).permutation(len(bc.target))
    tr, te = perm[:400], perm[400:]
    greedy = fano_curve(Q[tr], bc.target[tr], Q[te], bc.target[te], 3, 4)[-1]
    top3 = fano_curve(Q[tr], bc.target[tr], Q[te], bc.target[te], 3, 4, fixed_order=mi_rank(bc.data[tr], bc.target[tr], 4)[0][:3])[-1]
    print(f"   three features, H(Y|X) and test error: top-ranked {top3['h_y_given_x']:.3f}, {top3['test_error']:.3f}; greedy {greedy['h_y_given_x']:.3f}, {greedy['test_error']:.3f}")
    print(f"   two classes: the weak Fano bound is {greedy['weak']:.1f} for every feature set, and the tight one still says something ({top3['tight']:.3f})")
    dg = dml_data.load("sklearn-digits")
    X = (np.asarray(dg.data) > 7).astype(np.int64)
    y = np.asarray(dg.target).astype(np.int64)
    perm = np.random.default_rng(0).permutation(len(y))
    print("   digits, binarized pixels:  k  H(Y|X)  weak  tight  train err  test err")
    for r in fano_curve(X[perm[:1200]], y[perm[:1200]], X[perm[1200:]], y[perm[1200:]], 14):
        print(f"   {r['k']:>26}  {r['h_y_given_x']:.3f}  {r['weak']:.3f}  {r['tight']:.3f}  {r['train_error']:.3f}  {r['test_error']:.3f}")
    print("   The bound holds at every k. Past k = 9 the plug-in H(Y|X) keeps falling while test error rises: the table is memorizing.")


if __name__ == "__main__":
    main()

