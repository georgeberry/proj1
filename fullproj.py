from __future__ import division

'''
@George Berry (geb97)
@2/21/2014
@proj1 for NLP
'''

import re
from random import choice
import random
import sys
import time
from functools import wraps
from itertools import islice
import math

##global functions##

#fast sliding window function
def window(iterable, size):
    it = iter(iterable)

    #this works because we knock out the first n elements
    #i.e.: you can only go through an iterator once
    result = tuple(islice(it, size))

    if len(result) == size:
        yield result
    for item in it:
        result = result[1:] + (item,)
        yield result

#times functions easily
def timer(f):
    @wraps(f)
    def wrapper(*args,**kwargs):
        tic = time.time()
        result = f(*args, **kwargs)
        print(f.__name__ + " took " + str(time.time() - tic) + " seconds")
        return result
    return wrapper


def clean_up(text, n):

    with open(text, 'rb') as f:
        text = f.read()

    '''
    simply cleans text with regex subs.
    assumptions: we want punctuation as its own word; everything lowercase
    assume sentences end with 1+ repetitions of .?!
    keep ' the way it is: i.e. we don't separate contractions
    '''
    #remove xml tags
    text = re.sub(r'<.*>','',text)

    #remove verse numbers/hotel review numbers
    text = re.sub(r'[0-9]+[:,][0-9]+(,*)\s*','', text)

    #replace newlines with spaces (we do a space split below)
    text = re.sub(r'[\n]+',' ', text)

    #puts space around punctuation except sentence enders
    text = re.sub(r'[^\w\s\'\.!\?]', ' \g<0> ', text)

    #puts an end sentence token after each sentence along with whitespace.
    text = re.sub(r'(\.+)|(!+)|(\?+)', ' \g<0> </s> ', text) 

    #put n-1 <s> tokens before every </s> token
    for x in range(n-1):
        text = re.sub('</s>', '</s> <s>', text)

    #lowercase, remove leading/trailing spaces
    text = text.lower().strip()

    #split on 1+ spaces
    text = re.split(r' +', text)

    for x in range(n-1):
        text.insert(0, '<s>')

    return text


##classes##

class gram:
    '''
    holds information for a single tuple of length n-1
    think of this as the thing conditioned on in P(w_n|w_n-1,w_n-2)
    can add w_n's 1 at a time
        the total distribution is stored in this object

    can call object method to get a probabalistically-determined next word

    data structure:
        holds a dictionary of next words, keyed by next word 
        with values as empirical counts in the training set

    '''
    ##class gram setup methods ##

    def __init__(self, input_tuple, word):

        self.prev_words = input_tuple
        #total number of observed subsequent words
        self.cum_sum = None
        #unsmoothed next words
        self.next_words = {word : 1}

        #good turing elements
        #smoothed counts of seen words
        self.gt_next_counts = {}
        #sum of smoothed seen words
        self.gt_seen_sum = None
        #sum of smoothed seen words plus U * P(U) (num unseen times prob unseen)
        self.gt_sum = None
        #the smoothed count of one unseen word
        self.c_0 = None


    def add_next(self, word):
        '''
        call to add a word to the self.next_words dict

        should be done all at once
        '''
        if not isinstance(word, str):
            return 'must input string'
        
        try:
            self.next_words[word] += 1
        except KeyError:
            self.next_words[word] = 1


    ## class gram output methods ##

    def gt_random(self, vocab_list, freq_of_freq_dict, k_cutoff):

        '''
        Implements good-turing smoothing with a cutoff.

        develops a count of c_0, by computing all possible n_grams (vocab^n)
            subtracts the total observed

        this is a kind of ``global'' good-turing: we take the counts over all
            n_grams, rather than implementing good turing seperately for each conditional probability

        '''

        self.gt_setup(vocab_list, freq_of_freq_dict, k_cutoff)

        #create a random uniform on the real line between 0 and the smoothed wordcounts
        r = random.uniform(0, self.gt_sum)

        #pick unseen word with P(B) = C(unseen)/C(total)
        if r > self.gt_seen_sum/self.gt_sum:
            #all unseen words have same probability
            return random.choice(list(set(vocab_list) - set(self.gt_next_counts.keys())))

        #pick seen with prob 1-B, then weighted by adjusted freq counts
        else:
            r = random.uniform(0, self.gt_seen_sum)
            left_point = 0
            for c, w in self.gt_next_counts.iteritems():
                if left_point + w >= r:
                    return c
                left_point += w
            assert False, "error"


    #idea for implementation from here: http://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
    def random_next(self):
        '''
        given a conditional probability word_dict, 
        returns a random word
        '''
        if not self.cum_sum:
            self.cum_sum = sum(w for c, w in self.next_words.iteritems())
        r = random.uniform(0, self.cum_sum)
        left_point = 0
        for c, w in self.next_words.iteritems():
            if left_point + w >= r:
                return c
            left_point += w
        assert False, "error"


    def return_prob(self, current_word, vocab_list, freq_of_freq_dict, k_cutoff):
        '''
        give this a word
            will return the conditional probability of seeing the word
            given previous n-1
        '''

        self.gt_setup(vocab_list, freq_of_freq_dict, k_cutoff)

        #count of current word
        #divide by all words in the row
        if current_word in self.gt_next_counts:
            return self.gt_next_counts[current_word]/self.gt_sum
        else:
            return self.c_0/self.gt_sum

    ## class gram helper methods ##

    @staticmethod
    def gt_counts(c, k, ffd):
        '''
        returns smoothed count
        '''
        if c > k:
            return c
        else:
            #c* equaiton from page 103
            return ( ( (c+1)*(ffd[c+1]/ffd[c]) ) - ( c*( (k + 1)*ffd[k+1] )/ffd[1] ) )/(1 - ( (k+1)*(ffd[k+1])/ffd[1] ) )

    def num_times_seen(self):
        '''
        returns a list of the number of times each n_gram is seen
        '''
        return self.next_words.values()

    def gt_setup(self, vocab_list, freq_of_freq_dict, k_cutoff):
        '''
        if c_0 and gt_counts/gt_next_counts/gt_sum/gt_seen_sum aren't initialized

        compute & store in the class object

        trying to follow DRY here
        '''
        if self.c_0 == None:

            N_1_counts = freq_of_freq_dict[1]

            #unique total ngrams - unique seen ngrams
            N_0_counts = (len(vocab_list)**(len(self.prev_words) + 1)) - sum(freq_of_freq_dict.values())

            if len(self.prev_words) > 0:
                self.c_0 = N_1_counts/N_0_counts
            else:
                #for unigrams, there are no unseen words
                self.c_0 = 0.

            for k, v in self.next_words.iteritems():
                self.gt_next_counts[k] = self.gt_counts(v, k_cutoff, freq_of_freq_dict)

            #fancier way: need to check
            #self.gt_next_counts = {k: self.gt_counts(v, k_cutoff, freq_of_freq_dict) for k, v in self.next_words.iteritems()}

            self.gt_seen_sum = sum(w for c, w in self.gt_next_counts.iteritems())

            #add smoothed counts with implied 0 counts
            #we have seen len(gt_next_counts.keys()) following words
            #give the rest the default probability self.c_0 and add it that many times to self.gt_sum
            self.gt_sum = self.gt_seen_sum + (len(vocab_list) - len(self.gt_next_counts.keys()))*(self.c_0)

    def __repr__(self):
        return str(self.next_words)


class ngrams:
    '''
    give this cleaned text as a list of words
    does a little jiggering for start words
    can call to spit out sentences

    the meat of this:
        makes a dictionary at self.conditionals keyed by n-1 grams \
            with values as instances of the "gram" class
        to generate sentences, "ask" the appropriate "gram" instance for a next word
    '''
    def __init__(self, n, text_as_list):
        self.n = n
        self.text = text_as_list

        self.k_cutoff = 20

        #this should be converted to a dictionary for fast lookup
        self.vocab = {}

        #stores class object for every n-1 gram
        #classes hold next words, return random next words, and return probabilities
        self.conditionals = {}

        #records number of times different N_grams appear a number of times
        self.freq_of_freqs = {}
        self.unique_words = None

        #creates gram class instances
        #initializes freq_of_freqs dict
        self.gt_setup()

    @timer
    def gt_setup(self):

        #replaces first occurances with unknown words
        #skips the start/end sentence tokens
        self.text, self.vocab = self.handle_unknowns(self.text)

        self.unique_words = len(self.vocab.keys())

        for gram_tuple in window(self.text, self.n):
            prev_n_minus_one = gram_tuple[:-1]
            current_word = gram_tuple[-1]

            try:
                #increment class instance
                self.conditionals[prev_n_minus_one].add_next(current_word)
            except KeyError:
                #initialize class instance
                self.conditionals[prev_n_minus_one] = gram(prev_n_minus_one, current_word)

        #make freq of freq dict
        for v in self.conditionals.values():

            for count in v.num_times_seen():
                try:
                    self.freq_of_freqs[count] += 1
                except KeyError:
                    self.freq_of_freqs[count] = 1
    

    @timer
    def gen(self, smoothing = None):
        if self.n > 1:
            sentence = ['<s>']*(self.n-1)
            #current keeps track of preceding n-1 words
            current = sentence[-self.n+1:]
        elif self.n == 1:
            sentence = ['<s>']

        while sentence[-1] != '</s>':
            if self.n > 1:
                if tuple(current) in self.conditionals:
                    if smoothing == 'turing':
                        s = self.conditionals[tuple(current)].gt_random(self.vocab.keys(), self.freq_of_freqs, self.k_cutoff)
                    else:
                        s = self.conditionals[tuple(current)].random_next()
                else:
                    if smoothing == 'turing':
                        s = random.choice(self.vocab.keys())

                sentence.append(s)

                current = sentence[-self.n+1:]

            elif self.n == 1:
                if smoothing == 'turing':
                    s = self.conditionals[()].gt_random(self.vocab.keys(), self.freq_of_freqs, self.k_cutoff) #for unigram
                else:
                    s = self.conditionals[()].random_next()
                
                sentence.append(s)

        return ' '.join(sentence)



    @timer
    def gt_gen(self):
        ''' 
        function to generate a sentence from unsmoothed counts

        relies on calls to the appropriate gram instance in self.conditionals

        start with n-1 <s> and then keep going until we see </s>
        '''

        if self.n > 1:
            sentence = ['<s>']*(self.n-1)

            #current keeps track of preceding n-1 words
            current = sentence[-self.n+1:]

        elif self.n == 1:
            sentence = ['<s>']

        while sentence[-1] != '</s>':
            if self.n > 1:
                if tuple(current) in self.conditionals:
                    s = self.conditionals[tuple(current)].gt_random(self.vocab.keys(), self.freq_of_freqs, self.k_cutoff)
                else:
                    s = random.choice(self.vocab.keys())

                sentence.append(s)

                current = sentence[-self.n+1:]

            elif self.n == 1:
                s = self.conditionals[()].gt_random(self.vocab.keys(), self.freq_of_freqs, self.k_cutoff) #for unigram
                
                sentence.append(s)

        return ' '.join(sentence)


    @staticmethod
    @timer
    def handle_unknowns(text):

        seen_words = {'<s>':1, '</s>':1}
        vocab = {}

        for word in xrange(len(text)):
            if text[word] not in seen_words:
                seen_words[text[word]] = 1
                text[word] = '<unk>'
                try:
                    vocab[text[word]] += 1
                except KeyError:
                    vocab[text[word]] = 1
            else:
                try:
                    vocab[text[word]] += 1
                except KeyError:
                    vocab[text[word]] = 1

        return text, vocab


    @timer
    def perplexity(self, corpus_as_list):
        '''
        should be called after gt_process

        uses log base 2 for perplexity
        '''

        #sub for unknowns
        for word in xrange(len(corpus_as_list)):
            if corpus_as_list[word] not in self.vocab:
                corpus_as_list[word] = '<unk>'

        log_prob = 0.

        for gram_tuple in window(corpus_as_list, self.n):

            if self.n == 1:
                prev_n_minus_one = tuple()
                current_word = gram_tuple[0]

            elif self.n > 1:
                prev_n_minus_one = gram_tuple[:-1]
                current_word = gram_tuple[-1]

            if prev_n_minus_one in self.conditionals:
                current_gram = self.conditionals[prev_n_minus_one]
                log_prob += math.log(current_gram.return_prob(current_word, self.vocab.keys(), self.freq_of_freqs, self.k_cutoff), 2)

            else:
                #if we haven't seen the previous 2 words in order,
                #then we just have interpolated counts of c_0 for all elements of that row
                #so the probability is 1/V
                log_prob += math.log((1/self.unique_words), 2)

        
        log_perplexity = (-1./len(self.text))*log_prob
        return 2**log_perplexity


##run program from command line ##

if __name__ == "__main__":

    #n for n gram
    n = int(sys.argv[1])

    #file
    f = sys.argv[2]

    #sentences to gen
    s = int(sys.argv[3])

    #perplexity corpus
    p = sys.argv[4]

    t = clean_up(f, n)
    p = clean_up(p, n)

    tt = ngrams(n, t)

    for each in xrange(s):
        print tt.gen()
        #print tt.gen(smoothing='turing')

    print tt.perplexity(p)

