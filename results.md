```
Downloading original names dataset...
Loaded 32033 original names.
1. Perplexity of names (section 6.1.5): the effective number of next characters
   order  alpha   train perplexity  test perplexity
       0   None              27.00            27.00
       1  0.001              16.85            16.63
       2    0.3              11.64            11.59
       3    0.3               8.86             9.22
       4    0.1               6.47             8.16
   Each symbol of context helps by less than the one before, and the train-test gap keeps growing.

2. Real bits per character on held-out names (section 6.1.2)
   unigram  Huffman 4.088 bits, model cross entropy 4.056 bits
   bigram   Huffman 3.580 bits, model cross entropy 3.535 bits
   trigram  Huffman 3.245 bits, model cross entropy 3.218 bits
Loaded 32033 original names.
   general-purpose compressors: zlib 4.04, lzma 3.91, bz2 3.68
   fixed-length code: 5 bits. The model tables are trained in advance and not counted here.

3. Joint tables (sections 6.1.3, 6.1.4, 6.3.3)
   even/prime: H(X,Y) 1.81, H(Y|X) 0.81, I(X;Y) 0.19 bits
   Exercise 6.3: H(X) 1.75, H(X|Y=y) [1.75, 1.75, 2.0, 0.0]: observing y = 3 raises the entropy of X
   consecutive letters in names: H(next) 4.07, H(next|prev) 3.54, I 0.53 bits
   after q: 1.46 bits. At the start of a name: 4.25 bits, above the average.

4. Sequence logos and the cost of estimating entropy (sections 6.1.1.1, 6.1.1.2)
   consensus TAGTATAAAAGCGGA, column heights [0.0, 0.09, 0.65, 1.17, 1.36, 1.57, 2.0, 1.71, 0.6, 1.17, 0.51, 0.08, 0.0, 0.03, 0.0], total 10.93 bits
   Zipf over 20,000 words has 10.13 bits. Plug-in / Miller-Madow estimates by sample size:
   N =     100: 6.03 / 6.59
   N =    1000: 8.02 / 8.42
   N =   10000: 9.32 / 9.55
   N =  100000: 9.97 / 10.06
   N = 1000000: 10.12 / 10.13

5. Forward vs reverse KL on a bimodal target (section 6.2.6, Figure 6.3)
   forward fit: KL(p||q) 0.94 nats, KL(q||p) 3.73 nats, mean [0.0, 0.0]
   reverse fit: KL(p||q) 23.5 nats, KL(q||p) 0.70 nats, mean [2.5, 2.5]
   log 2 = 0.693: the cost of giving up on one of two equal modes
   Exercise 6.4, 100 random starts, (support, reverse KL in bits): count {((0, 1), 1.0): 65, ((2,), 2.0): 19, ((3,), 2.0): 16}
   at the product of marginals: reverse KL inf, forward KL = I(X;Y) = 1.50 bits

6. Mutual information estimators against known truths (section 6.3.5)
   one sample, rho 0.6, truth 0.223 nats. Equal-width bins: 2: 0.05, 4: 0.12, 8: 0.17, 16: 0.30, 32: 0.64, 64: 1.34; KSG 0.173
   mean absolute error in nats            fixed10          scott       scott_mm     quantile10           ksg3  gauss_formula
   gaussian                                0.169          0.138          0.093          0.126          0.027          0.014
   warped                                  0.504          0.319          0.308          0.119          0.042          0.427
   nonlinear                               0.612          0.278          0.298          0.441          0.050          1.324

7. Correlation vs MIC at n = 500 (section 6.3.7, Figure 6.6)
   line         R^2 1.000   MIC 1.000
   parabola     R^2 0.006   MIC 0.986
   cubic        R^2 0.221   MIC 0.973
   sine         R^2 0.026   MIC 0.952
   step         R^2 0.889   MIC 0.885
   circle       R^2 0.000   MIC 0.629
   x_shape      R^2 0.000   MIC 0.611
   independent  R^2 0.004   MIC 0.116
   equal noise (R^2 of the fit 0.5): MIC from 0.33 to 0.45 across the functional shapes on this one sample; over 30 seeds the means sit between 0.41 and 0.44

8. Data processing and sufficiency (sections 6.3.8, 6.3.9)
   X -> Y -> Z: I(X;Y) 0.531 >= I(X;Z) 0.173, gap = I(X;Y|Z) = 0.358 bits
   I(theta; the full sequence) = 1.0345 bits
   I(theta; number of heads) = 1.0345 bits
   I(theta; first half) = 0.7027 bits
   I(theta; majority vote) = 0.5791 bits
   I(theta; first flip) = 0.2126 bits
   I(theta; parity) = 0.0014 bits

9. Feature selection and Fano's bound (section 6.3.10)
   breast cancer, top by I(X;Y): worst perimeter 0.656, worst radius 0.646, worst area 0.645
   three features, H(Y|X) and test error: top-ranked 0.292, 0.148; greedy 0.084, 0.083
   two classes: the weak Fano bound is 0.0 for every feature set, and the tight one still says something (0.042)
   digits, binarized pixels:  k  H(Y|X)  weak  tight  train err  test err
                            1  2.858  0.559  0.563  0.800  0.814
                            2  2.417  0.427  0.431  0.679  0.700
                            3  2.018  0.306  0.332  0.516  0.506
                            4  1.662  0.199  0.254  0.438  0.487
                            5  1.304  0.092  0.185  0.345  0.377
                            6  1.006  0.002  0.133  0.283  0.322
                            7  0.741  0.000  0.091  0.228  0.278
                            8  0.494  0.000  0.056  0.154  0.253
                            9  0.340  0.000  0.036  0.114  0.226
                           10  0.229  0.000  0.022  0.083  0.250
                           11  0.155  0.000  0.014  0.057  0.255
                           12  0.101  0.000  0.009  0.037  0.305
                           13  0.065  0.000  0.005  0.026  0.365
                           14  0.038  0.000  0.003  0.016  0.402
   The bound holds at every k. Past k = 9 the plug-in H(Y|X) keeps falling while test error rises: the table is memorizing.
```
