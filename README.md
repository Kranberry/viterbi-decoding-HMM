# HMM Gene Sequence Tagger

This project implements a gene sequence tagger, that identifies gene names in  biological context. There is only type of entity here: I-GENE. All other words are tagged as O. 

There are a couple of things implemented here -
1. A baseline tagger, which just selects the tag with maximum emission probability for every word
2. Baseline tagger with better word classes for rare words
3. Trigram HMM tagger with Viterbi decoding
4. Trigramm HMM tagger with better word classes for rare words
5. Smoothing (Linear Interpolation) to improve the HMM tagger.

To run the main script:
```python count_freqs.py gene.train```
This will train the model on the input training data file gene.train, and will finally output the F1 score. There are fields inside the script to change, based on what setting you want to run (whether 1,2,3,4, or 5 above).


