#! /usr/bin/python

import sys
from collections import defaultdict
import math
import numpy as np
import re
from math import log
from eval_gene_tagger import corpus_iterator, Evaluator
from viterbi_decoder import Viterbi

"""
Count n-gram frequencies in a data file and write counts to
stdout. 
"""

def simple_conll_corpus_iterator(corpus_file):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()
    while l:
        line = l.strip()
        if line: # Nonempty line
            # Extract information from line.
            # Each line has the format
            # word pos_tag phrase_tag ne_tag
            fields = line.split(" ")
            ne_tag = fields[-1]
            #phrase_tag = fields[-2] #Unused
            #pos_tag = fields[-3] #Unused
            word = " ".join(fields[:-1])
            # word = word.lower() #TODO: needed?
            yield word, ne_tag
        else: # Empty line
            yield (None, None)                        
        l = corpus_file.readline()

def sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, ne_tag) tuples.
    """
    current_sentence = [] #Buffer for the current sentence
    for l in corpus_iterator:        
            if l==(None, None):
                if current_sentence:  #Reached the end of a sentence
                    yield current_sentence
                    current_sentence = [] #Reset buffer
                else: # Got empty input stream
                    sys.stderr.write("WARNING: Got empty input file/stream.\n")
                    raise StopIteration
            else:
                current_sentence.append(l) #Add token to the buffer

    if current_sentence: # If the last line was blank, we're done
        yield current_sentence  #Otherwise when there is no more token
                                # in the stream return the last sentence.

def get_ngrams(sent_iterator, n):
    """
    Get a generator that returns n-grams over the entire corpus,
    respecting sentence boundaries and inserting boundary tokens.
    Sent_iterator is a generator object whose elements are lists
    of tokens.
    """
    for sent in sent_iterator:
         #Add boundary symbols to the sentence
         w_boundary = (n-1) * [(None, "*")]
         w_boundary.extend(sent)
         w_boundary.append((None, "STOP"))
         #Then extract n-grams
         ngrams = (tuple(w_boundary[i:i+n]) for i in range(len(w_boundary)-n+1))
         for n_gram in ngrams: #Return one n-gram at a time
            yield n_gram        


class Hmm(object):
    """
    Stores counts for n-grams and emissions. 
    """

    def __init__(self, n=3):
        assert n>=2, "Expecting n>=2."
        self.n = n
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in range(self.n)]
        self.word_counts = defaultdict(int)
        self.all_states = set()

    def train(self, corpus_file):
        """
        Count n-gram frequencies and emission probabilities from a corpus file.
        """
        ngram_iterator = \
            get_ngrams(sentence_iterator(simple_conll_corpus_iterator(corpus_file)), self.n)

        for ngram in ngram_iterator:
            #Sanity check: n-gram we get from the corpus stream needs to have the right length
            assert len(ngram) == self.n, "ngram in stream is %i, expected %i" % (len(ngram, self.n))

            tagsonly = tuple([ne_tag for word, ne_tag in ngram]) #retrieve only the tags            
            for i in range(2, self.n+1): #Count NE-tag 2-grams..n-grams
                self.ngram_counts[i-1][tagsonly[-i:]] += 1
            
            if ngram[-1][0] is not None: # If this is not the last word in a sentence
                self.ngram_counts[0][tagsonly[-1:]] += 1 # count 1-gram
                self.emission_counts[ngram[-1]] += 1 # and emission frequencies
                # self.word_counts[ngram[-1][0]] += 1 #update count of the word

            # Need to count a single n-1-gram of sentence start symbols per sentence
            if ngram[-2][0] is None: # this is the first n-gram in a sentence
                self.ngram_counts[self.n - 2][tuple((self.n - 1) * ["*"])] += 1

            self.all_states.update(tagsonly)
        self.all_states.remove("STOP")
        self.all_states.remove("*")

    def compute_emission_params(self, word, tag):
        # return log(self.emission_counts[(word, tag)], 2) - log(self.ngram_counts[0][tuple([tag])], 2)
        return self.emission_counts[(word, tag)]/ self.ngram_counts[0][tuple([tag])]

    def compute_transition_params(self, prev_prev_tag, prev_tag, current_tag,
                                  smoothing=False,
                                  lambda1=0.6,
                                  lambda2=0.1,
                                  lambda3=0.3):
        trigram_num = tuple([prev_prev_tag, prev_tag, current_tag])
        trigram_denom = tuple([prev_prev_tag, prev_tag])
        trigram_estimate = self.ngram_counts[2][trigram_num] / self.ngram_counts[1][trigram_denom]
        if not smoothing:
            return trigram_estimate

        if prev_tag=="*":
            bigram_denom = self.ngram_counts[1][tuple(['*', '*'])]
            #this because, count of unigram(*) is equal to count of bigram(*,*)
        else:
            bigram_denom = self.ngram_counts[0][tuple([prev_tag])]
        bigram_estimate = self.ngram_counts[1][tuple([prev_tag, current_tag])] / bigram_denom


        unigram_estimate = self.ngram_counts[0][tuple([current_tag])] / sum([self.ngram_counts[0][unig] for unig in self.ngram_counts[0]])

        return lambda3*unigram_estimate + lambda2*bigram_estimate + lambda1*trigram_estimate


    def form_word_counts(self, corpus_file):
        self.word_counts = defaultdict(int)
        for l in corpus_file:
            line = l.strip()
            fields = line.strip().split(" ")
            ne_tag = fields[-1]
            word = " ".join(fields[:-1])
            self.word_counts[word] += 1


    def find_infrequent_words(self, threshold = 5):
        self.infrequent = []
        for word in self.word_counts:
            if self.word_counts[word] < threshold:
                self.infrequent.append(word)
                # print(word,':::',self.word_counts[word])
        # print(len(self.infrequent))


    def check_if_infrequent_word(self, word):
        return word in self.infrequent

    def has_only_num(self, word):
        return bool(re.search(r'^[\d]+$', word))

    # def check_how_many_only_nums(self):
    #     o = 0
    #     i = 0
    #     for word_tag in self.emission_counts:
    #         word=word_tag[0]
    #         tag=word_tag[1]
    #         if (word in self.infrequent) (self.has_only_num(word)):
    #             if tag=="O":
    #                 o+=1
    #             elif tag=="I-GENE":
    #                 i+=1
    #     return o,i

    def has_both_letters_and_num(self, word):
        return bool(re.search(r'(?:[A-Za-z].*?\d|\d.*?[A-Za-z])', word))

    def has_only_caps(self,word):
        return bool(re.search(r'^[A-Z]+$', word[1:])) #check if the word has caps, but only after first letterends_in

    def is_punctuation(self, word):
        return bool(re.search(r'^[.,!?+;:\\-]+$', word))

    def starts_with_caps(self, word):
        return bool(re.search(r'^[A-Z]', word))

    def ends_in_gene_suffixes(self,word):
        suffixes = ["ase", "ases","ate", "tide", "ogen", "lin", "min", "osis"]
        if len(word) >= 6:
            for suffix in suffixes:
                if word.endswith(suffix):
                    return True
        return False


    def replace_word_class_in_training_file(self,
                                            corpus_file,
                                            output_file,
                                            check_conditions_list,
                                            word_classes_list):
        c=0
        for l in corpus_file:
            c+=1
            line = l.strip()
            fields = line.strip().split(" ")
            ne_tag = fields[-1]
            word = " ".join(fields[:-1])
            if word in self.infrequent:
                any_one=False
                for i,check_condition in enumerate(check_conditions_list):
                    if check_condition(word):
                        line = " ".join([word_classes_list[i], ne_tag])
                        any_one=True
                        continue
                if not any_one:
                    line = " ".join(["_RARE_", ne_tag])
            output_file.write(line.strip() + "\n")
            if c%10000==0:
                print("%s lines done" % c)

        output_file.close()
        corpus_file.close()

    def write_counts(self, output, printngrams=[1,2,3]):
        """
        Writes counts to the output file object.
        Format:

        """
        # First write counts for emissions
        for word, ne_tag in self.emission_counts:
            output.write("%i WORDTAG %s %s\n" % (self.emission_counts[(word, ne_tag)], ne_tag, word))


        # Then write counts for all ngrams
        for n in printngrams:            
            for ngram in self.ngram_counts[n-1]:
                ngramstr = " ".join(ngram)
                output.write("%i %i-GRAM %s\n" %(self.ngram_counts[n-1][ngram], n, ngramstr))

    def read_counts(self, corpusfile):

        self.n = 3
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in range(self.n)]
        self.all_states = set()

        for line in corpusfile:
            parts = line.strip().split(" ")
            count = float(parts[0])
            if parts[1] == "WORDTAG":
                ne_tag = parts[2]
                word = parts[3]
                self.emission_counts[(word, ne_tag)] = count
                self.all_states.add(ne_tag)
            elif parts[1].endswith("GRAM"):
                n = int(parts[1].replace("-GRAM",""))
                ngram = tuple(parts[2:])
                self.ngram_counts[n-1][ngram] = count

    def baseline_tagger_predictions(self, counts_file, test_file, output_preds_file):
        self.read_counts(counts_file)
        all_states_list = list(self.all_states)
        c=0
        for l in test_file:
            preds_all_tags = []
            word = l.strip()
            orig_word=word
            if word == "":
                output_preds_file.write("\n")
                continue
            if (word not in self.word_counts) or (word in self.infrequent):
                word = "_RARE_"
            for tag in all_states_list:
                preds_all_tags.append(self.compute_emission_params(word,tag))
            pred = all_states_list[np.argmax(preds_all_tags)]
            output_preds_file.write(" ".join([orig_word,pred]) + "\n")
        output_preds_file.close()


    def baseline_tagger_predictions_informative_wordclasses(self, counts_file,
                                                            test_file,
                                                            output_preds_file,
                                                            check_conditions,
                                                            word_classes):
        self.read_counts(counts_file)
        all_states_list = list(self.all_states)
        c=0
        for l in test_file:
            preds_all_tags = []
            word = l.strip()
            orig_word = word
            if word == "":
                output_preds_file.write("\n")
                continue
            if (word not in self.word_counts) or (word in self.infrequent):
                any_one=False
                for i,check_condition in enumerate(check_conditions):
                    if check_condition(word):
                        word = word_classes[i]
                        any_one=True
                        continue
                if not any_one:
                    word="_RARE_"

            for tag in all_states_list:
                preds_all_tags.append(self.compute_emission_params(word,tag))
            pred = all_states_list[np.argmax(preds_all_tags)]
            output_preds_file.write(" ".join([orig_word,pred]) + "\n")
        output_preds_file.close()


    def viterbi_predictions(self, counts_file, test_file, output_preds_file, smoothing=False, lambda1=0.6,lambda2=0.1,
                            lambda3=0.3):
        self.read_counts(counts_file)
        viterbi = Viterbi(self, smoothing, lambda1, lambda2, lambda3)
        all_states_list = list(self.all_states)
        c=0
        sentence = []
        orig_sentence = []
        for l in test_file:
            word = l.strip()
            orig_word = word
            if (word not in self.word_counts) or (word in self.infrequent):
                word = "_RARE_"
            sentence.append(word)
            orig_sentence.append(orig_word)
            if word == "":
                sentence = sentence[:-1] #remove the last space added
                orig_sentence = orig_sentence[:-1]  # remove the last space added
                outputs = viterbi.predict_tags_for_sentence(sentence)
                for w,t in zip(orig_sentence,outputs):
                    output_preds_file.write(w+" "+t+"\n")
                output_preds_file.write("\n")
                sentence = []
                orig_sentence = []
        output_preds_file.close()


    def viterbi_predictions_informative_wordclasses(self, counts_file,
                                                    test_file,
                                                    output_preds_file,
                                                    check_conditions,
                                                    word_classes ):
        self.read_counts(counts_file)
        viterbi = Viterbi(self)
        all_states_list = list(self.all_states)
        c=0
        sentence = []
        orig_sentence = []
        for l in test_file:
            word = l.strip()
            orig_word = word
            if (word not in self.word_counts) or (word in self.infrequent):
                any_one=False
                for i,check_condition in enumerate(check_conditions):
                    if check_condition(word):
                        word = word_classes[i]
                        any_one=True
                        continue
                if not any_one:
                    word="_RARE_"
            sentence.append(word)
            orig_sentence.append(orig_word)
            if word == "":
                sentence = sentence[:-1] #remove the last space added
                orig_sentence = orig_sentence[:-1]  # remove the last space added
                outputs = viterbi.predict_tags_for_sentence(sentence)
                for w,t in zip(orig_sentence,outputs):
                    output_preds_file.write(w+" "+t+"\n")
                output_preds_file.write("\n")
                sentence = []
                orig_sentence = []
        output_preds_file.close()


def usage():
    print ("""
    python count_freqs.py [input_file] > [output_file]
        Read in a gene tagged training input file and produce counts.
    """)

if __name__ == "__main__":

    if len(sys.argv)!=2: # Expect exactly one argument: the training data file
        usage()
        sys.exit(2)

    try:
        input = open(sys.argv[1],"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % input)
        sys.exit(1)


    # mode = "baseline"
    # mode = "baseline+word_classes"
    # mode = "viterbi"
    mode = "viterbi+word_classes"
    # mode = "viterbi_smoothing"

    # Initialize a trigram counter
    counter = Hmm(3)

    if mode in ("baseline", "viterbi", "viterbi_smoothing"):
        train_output_file = "gene.train.rare"
        gene_counts_file = "gene.counts.rare"
        if mode == "baseline":
            prediction_output_file = "gene_dev.p1.out"
        elif mode == "viterbi":
            prediction_output_file = "gene_dev.p1.vit.out"
        else:
            prediction_output_file = "gene_dev.p1.vit.smoothing.out"


    elif mode in ("baseline+word_classes", "viterbi+word_classes"):
        train_output_file = "gene.train.word_classes"
        gene_counts_file = "gene.counts.word_classes"
        if mode == "baseline+word_classes":
            prediction_output_file = "gene_dev.p1.informative_wc.out"
        elif mode == "viterbi+word_classes":
            prediction_output_file = "gene_dev.p1.vit.informative_wc.out"




    # Replace infrequent words in training file
    counter.form_word_counts(open(sys.argv[1], "r"))
    counter.find_infrequent_words(5)


    check_conditions = [
                        counter.has_only_num,
                        # counter.starts_with_caps,
                         counter.has_only_caps,
                         # counter.has_both_letters_and_num,
                         # counter.is_punctuation,
                         # counter.ends_in_gene_suffixes
                        ]
    word_classes = [
                    "onlyNum",
                    # "initCaps",
                    "onlyCaps",
                    # "bothLettersAndNum",
                    # "punct",
                    # "endsInSuffixes"
                    ]

    print("Features = ",word_classes)

    print("Replacing infrequent words in training file")
    counter.replace_word_class_in_training_file(open(sys.argv[1], "r"),
                                                open(train_output_file, 'w'),
                                                check_conditions,
                                                word_classes)

    print("Training")
    # Collect counts
    counter.train(open(train_output_file, "r"))


    #
    # # prediction_output_file = "gene_dev.p1.out"
    # prediction_output_file = "gene_dev.p1.informative_wc.out"
    # # prediction_output_file = "gene_dev.p1.vit.out"
    # # prediction_output_file = "gene_dev.p1.vit.smoothing.out"



    print("Writing counts")
    # Write the counts
    # counter.write_counts(sys.stdout)
    counter.write_counts(open(gene_counts_file, "w"))



    print("Calculating predictions and writing them to file")
    # #predict
    if mode == "baseline":
        counter.baseline_tagger_predictions(open(gene_counts_file, "r"),
                                            open("gene.dev", "r"),
                                            open(prediction_output_file, "w"))

    if mode == "baseline+word_classes":
        counter.baseline_tagger_predictions_informative_wordclasses(open(gene_counts_file, "r"),
                                        open("gene.dev", "r"),
                                        open(prediction_output_file, "w"),
                                        check_conditions,
                                        word_classes)

    if mode == "viterbi":
        counter.viterbi_predictions(open(gene_counts_file, "r"),
                                    open("gene.dev", "r"),
                                    open(prediction_output_file, "w"))

    if mode == "viterbi+word_classes":
        counter.viterbi_predictions_informative_wordclasses(open(gene_counts_file, "r"),
                                    open("gene.dev", "r"),
                                    open(prediction_output_file, "w"),
                                    check_conditions,
                                    word_classes)

    if mode == "viterbi_smoothing":
        l1,l2,l3=[0.6,0.1,0.3]
        counter.viterbi_predictions(open(gene_counts_file, "r"),
                                    open("gene.dev", "r"),
                                    open(prediction_output_file, "w"), True, l1, l2, l3)



    # #Hyperparameter tuning
    # prediction_output_file = 'gene_dev.p1.vit.out_hyptune'
    # # #predict
    # lambda1=[0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    # lambda2=[0.1,0.2,0.3,0.4,0.5]
    # best_fscore=0
    # best_lambdas = []
    # for l2 in lambda2:
    #     for l1 in lambda1:
    #         l3 = 1 - l2 - l1
    #         if l3<0:
    #             continue
    #         print(l1,l2,l3)
    #         counter.viterbi_predictions(open(gene_counts_file, "r"),
    #                                     open("gene.dev", "r"),
    #                                     open(prediction_output_file, "w"), l1, l2, l3)
    #         gs_iterator = corpus_iterator(open("gene.key"))
    #         pred_iterator = corpus_iterator(open(prediction_output_file), with_logprob=False)
    #         evaluator = Evaluator()
    #         evaluator.compare(gs_iterator, pred_iterator)
    #         fscore = evaluator.print_scores()
    #         if fscore>best_fscore:
    #             best_fscore=fscore
    #             best_lambdas=[l1,l2,l3]
    #         print("Best lambdas till now=", best_lambdas)
    #         print("Best F score till now =", best_fscore)
    #         print()
    # print("Best lambdas =", best_lambdas)
    # print("Best F score =", best_fscore)


    print("Evaluating")

    gs_iterator = corpus_iterator(open("gene.key"))
    pred_iterator = corpus_iterator(open(prediction_output_file), with_logprob = False)
    evaluator = Evaluator()
    evaluator.compare(gs_iterator, pred_iterator)
    evaluator.print_scores()











