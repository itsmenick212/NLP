import sys
from collections import defaultdict
import math
import random
import os
import os.path
import numpy as np
"""
COMS W4705 - Natural Language Processing - Summer 2019 
Homework 1 - Trigram Language Models
Daniel Bauer

Student:    Nick Gupta 
UNI:        ng2528
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """

    begin_padding = ['START'] * (n-1) if n > 2 else ['START']
    end_padding = ['STOP']
    sequence = begin_padding + sequence + end_padding
    padded_ngrams = []
    for ran in range(len(sequence) - n + 1):
        padded_ngrams.append(tuple(sequence[ran:ran+n]))
    return padded_ngrams



class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 

        ##Your code here
        
        for sentence in corpus:
            trigram_count = get_ngrams(sentence, 3)
            bigram_count = get_ngrams(sentence, 2)
            unigram_count = get_ngrams(sentence, 1)
            for trigram in trigram_count:
                self.trigramcounts[trigram] += 1
            for bigram in bigram_count:
                self.bigramcounts[bigram] += 1
            for unigram in unigram_count:
                self.unigramcounts[unigram] += 1
            if trigram[:2] == ('START', 'START'):
                self.bigramcounts[('START', 'START')] += 1
                
        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if self.bigramcounts[trigram[:2]] != 0 :
            return self.trigramcounts[trigram]/self.bigramcounts[trigram[:2]]
        else:
            return 0.0

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if self.unigramcounts[bigram[:1]] != 0:
            return self.bigramcounts[bigram]/self.unigramcounts[bigram[:1]]
        else:
            return 0.0
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it. 
        
        if not hasattr(self, 'domain_id'):
        #https://www.programiz.com/python-programming/methods/built-in/hasattr
        #https://stackoverflow.com/questions/33821320/using-hasattr-and-not-hasattr-in-python
            self.domain_id = sum(self.unigramcounts.values())
            self.domain_id -= - self.unigramcounts[('START',)] + self.unigramcounts[('STOP',)]
        
        return 0.0

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        trigram_outcomes = (None, 'START', 'START')
        result = list()
        i = 0
        while trigram_outcomes[2] != 'STOP' and i < t:
            FirstWord = trigram_outcomes[1]
            SecondWord = trigram_outcomes[2]
            outcomes = [trigram for trigram in self.trigramcounts.keys() if trigram[:2] == (FirstWord, SecondWord)]
            prob_sum = [self.raw_trigram_probability(trigram) for trigram in outcomes]
            r = np.random.random([candidate[2] for candidate in outcomes], 1, probs = prob_sum)[0]
            trigram_outcomes = (FirstWord, SecondWord, r)
            result.append(r)
            i += 1
        
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        
        interpol = 0.0
        interpol += lambda1 * self.raw_trigram_probability(trigram)
        interpol += lambda2 * self.raw_bigram_probability(trigram[1:])
        interpol += lambda3 * self.raw_unigram_probability(trigram[2:])
        
        return interpol
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        
        trigram_count = get_ngrams(sentence, 3)
        
        probabilities = [self.smoothed_trigram_probability(trigram) 
        for trigram in trigram_count]
        return float(sum(math.log2(prob) for prob in probabilities))

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        entropy = 0
        M = 0
        for sequence in corpus:
            entropy += self.sentence_logprob(sequence)
            M += len(sequence)
        entropy /= M
        return float(2**(-entropy))


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            # .. 
            total += 1
            correct += pp
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # .. 
            correct += pp
            total +=1 
        
        return (correct/total)

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)
    print(model.generate_sentence())
