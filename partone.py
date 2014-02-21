'''
George Berry (geb97) and Antonio Sirianni (ads334) submission

code written for python 2.7.6 (works well with pypy2)

intended to be called from unixlike command line like so: 
    $python partone.py n_gram filename num_sentences

    should specify: 
        n_gram size with argv[1]
        filename with argv[2]
        number of sentences to make with argv[3]
'''

import re
from random import choice
import random
import sys
import time
from functools import wraps
from itertools import islice


def window(iterable, size):
    it = iter(iterable)
    result = tuple(islice(it, size))

    if len(result) == size:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


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

    #replace verse numbers (beginning of sentence) with <s> tag
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


#classes

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
    def __init__(self, input_tuple, word):
        if not isinstance(input_tuple, tuple):
            if isinstance(input_tuple, str):
                input_tuple = (input_tuple,)
            else:
                return 'intput not convertable to tuple'

        self.prev_words = input_tuple

        #total number of subsequent words
        self.cum_sum = 0

        self.next_words = {word : 1}

    def add_next(self, word):
        '''
        call to add a word to the self.next_words dict
        '''
        if not isinstance(word, str):
            return 'must input string'
        
        #self.next_words[word] = self.next_words.get(word, 0) + 1

        try:
            self.next_words[word] += 1
        except:
            self.next_words[word] = 1

    def random_next(self):
        '''
        given a conditional probability word_dict, 
        returns a random word
        '''
        self.cum_sum = sum(w for c, w in self.next_words.iteritems())
        r = random.uniform(0, self.cum_sum)
        left_point = 0
        for c, w in self.next_words.iteritems():
            if left_point + w >= r:
                return c
            left_point += w
        assert False, "error"

    def good_turing_next(self, words_in_vocab, n):
        '''


        '''

    def __repr__(self):
        return 'gram for previous words: ' + str(self.prev_words)


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
        self.vocab = list(set(text_as_list))
        self.conditionals = {}

        self.process()

    @timer
    def process(self):
        word_range = range(self.n)

        for gram_tuple in window(self.text, self.n):
            #prev_n_minus_one = gram_tuple[:-1]
            #current_word = gram_tuple[-1]

            try:
                #increment class instance
                self.conditionals[tuple(gram_tuple[:-1])].add_next(gram_tuple[-1])
            except:
                #initialize class instance
                self.conditionals[tuple(gram_tuple[:-1])] = gram(tuple(gram_tuple[:-1]), gram_tuple[-1])

    def gen(self):
        ''' 
        function to generate a sentence

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
                s = self.conditionals[tuple(current)].random_next()

                sentence.append(s)

                current = sentence[-self.n+1:]

            elif self.n == 1:
                s = self.conditionals[()].random_next() #for unigram
                
                sentence.append(s)
                
        return ' '.join(sentence)


if __name__ == "__main__":
    n = int(sys.argv[1])
    f = sys.argv[2]
    k = int(sys.argv[3])

    t = clean_up(f, n)

    tt = ngrams(n, t)

    print(len(set(tt.text)))

    for each in range(k):
        print tt.gen() + '\n' 
