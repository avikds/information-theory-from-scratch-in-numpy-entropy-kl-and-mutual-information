# Information Theory from Scratch in NumPy: Entropy, KL and Mutual Information

Murphy's Probabilistic Machine Learning builds a whole toolkit from one idea, the entropy of a distribution. This project builds it in NumPy and runs every piece on something real. You start with 32,033 first names: entropy, cross entropy and perplexity, n-gram language models, and your own Huffman codes, so that a bit is an actual bit. From there you work through joint and conditional entropy tables (the book's even and prime numbers, a book exercise, letter pairs in names), DNA sequence logos and how badly plug-in entropy undershoots on a long tail, KL divergence as the extra cost of the wrong code, differential entropy read through quantization, and forward versus reverse KL fits to a bimodal target, including the three minima of a book exercise. Mutual information gets a Gaussian closed form, histogram estimators with Scott's rule and the KSG k-nearest-neighbour estimator, all graded against known truths, plus a dynamic-programming MIC that reruns the book's MIC experiment. You finish with the data processing inequality on a noisy channel chain, sufficient statistics measured exactly, and mutual-information feature selection with Fano's bound on breast-cancer and digits data. Headline numbers: test perplexity falls from 27 (uniform) to 16.6, 11.6, 9.2 and 8.2 for orders 1 to 4; a trigram table with Huffman codes pays 3.24 bits per character on held-out names, against 3.68 for bz2 and 4.03 for zlib (the tables are trained in advance); a forward Gaussian fit scores 0.94 nats forward and 3.73 reverse, while a reverse fit scores 23.5 and 0.70; KSG's mean absolute error stays at 0.03 to 0.05 nats across Gaussian, warped and nonlinear data, where the Gaussian formula goes from 0.014 to 1.32; MIC scores a parabola at 0.99 where R squared is 0.006, but a noise-free circle only 0.63; and on digits, plug-in H(Y|X) keeps falling while test error turns around at nine pixels. Fano's closed-form bound turns out to be vacuous for every two-class problem, so the project uses the tighter form as well.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** entropy
- [x] **2.** huffman_lengths
- [x] **3.** load_names
- [x] **4.** fit_ngram
- [x] **5.** model_perplexity

---

Built on Deep-ML.
