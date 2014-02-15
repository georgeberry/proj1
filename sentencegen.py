'''
@by George Berry
@geb97@cornell.edu
@created 2/8/2014
'''

import re, time
from functools import wraps
from random import choice
import random

#open file
biblefile = 'filepath'

hotelfile = 'filepath'

dostoevsky = 'filepath'

with open(biblefile, 'rb') as f:
    bible = f.read()

with open(hotelfile, 'rb') as f:
    hotel = f.read()

with open(dostoevsky, 'rb') as f:
    dos = f.read()

def timer(f):
    @wraps(f)
    def wrapper(*args,**kwargs):
        tic = time.time()
        result = f(*args, **kwargs)
        print(f.__name__ + " took " + str(time.time() - tic) + " seconds")
        return result
    return wrapper


@timer
def clean_up(text):
    '''
    simply cleans text with regex subs.
    assumptions: we want punctuation as its own word; everything lowercase; sentence start tokens but no end tokens (.?!) are end tokens.
    except '
    e.g. we preserve contractions
    '''

    #remove xml tags
    text = re.sub(r'<.*>','',text)
    #replace verse numbers (beginning of sentence) with <s> tag
    text = re.sub(r'[0-9]+(:|,)[0-9]+\s*','', text)
    #replace newlines with spaces (we do a space split below)
    text = re.sub(r'[\n]+',' ', text)
    #replace punctuation with a space then that punctuation
    text = re.sub(r"[^\w\s\']+", " \g<0>", text) #puts space before punctuation
    text = re.sub(r'\.', '. <s>', text)
    text = text.lower().strip()
    text = re.split(r' +', text)
    #text.append('</s>') #just need this
    #we automatically put <s> at the start with make_ngrams function
    return text


def lookup_default(list, index, default):
    '''
    for large number of calls to lists that may or may not have items
    allows easy specification of a default
    '''
    try:
        return list[index]
    except:
        return default


#classes

class gram:
    '''
    holds information for a single tuple of n-1 previous words:
    condidtional distribution; smoothing; probabilistic next word

    data structure:
        holds a dictionary of next words, keyed by word with values as empirical counts in the training set
    '''
    def __init__(self, input_tuple, word):
        if not isinstance(input_tuple, tuple):
            if isinstance(input_tuple, str):
                input_tuple = (input_tuple,)
            else:
                return 'intput not convertable to tuple'

        self.prev_words = input_tuple

        self.cum_sum = 0

        self.next_words = {word : 1}

    def add_next(self, word):
        '''
        call to add a word to the self.next_words dict
        '''
        if not isinstance(word, str):
            return 'must input string'
        try:
            self.next_words[word] += 1
        except:
            self.next_words[word] = 1

    def next(self, word_dict):
        '''
        given a conditional probability word_dict, 
        returns a random word
        swears if something goes horribly wrong
        '''
        self.cum_sum = sum(w for c, w in word_dict.iteritems())
        r = random.uniform(0, self.cum_sum)
        left_point = 0
        for c, w in word_dict.iteritems():
            if left_point + w > r:
                return c
            left_point += w
        assert False, "wtf"

    def random_next(self):
        '''
        easily call an unsmoothed next word
        '''
        return self.next(self.next_words)

    @staticmethod
    def seen_words(word_dict):
        return set(c for c, w in word_dict.iteritems())

    @staticmethod
    def count(word_dict):
        return sum(w for c, w in word_dict.iteritems())


    def laplace_next(self, words_in_vocab):
        '''
        idea here:
        there are two groups, A and B:
            we have seen empirically words in A. 
            we are smoothing for words in B.

            the empirical words have a chunk of the total smoothed probability
                this is given by: (W + #W)/(W + V). W is unsmoothed counts of seen words; #W is unique # seen words; V is # words in vocab

            choose number uniform at random between (0, W + V)
            if num > (W + #W)/(W + V), pick a random word from class B

            otherwise, pick one with weighted probability (given by smoothed counts) from class A
        '''
        laplace = self.next_words

        for key, val in laplace.iteritems():
            laplace[key] += 1

        cut_point = self.count(laplace)/(self.count(self.next_words) + len(words_in_vocab))

        r = random.uniform(0, self.count(self.next_words) + len(words_in_vocab))

        if r < cut_point:
            return self.next(laplace)
        else: 
            return random.choice(list(set(words_in_vocab) - self.seen_words(laplace)))


    def phrase_probability(self, phrase_as_tuple):
        if not isinstance(phrase_as_tuple, tuple):
            return 'nope'

    def __repr__(self):
        return 'gram for previous words: ' + str(self.prev_words)


class ngrams:
    '''
    give this cleaned text as a list of words
    does a little jiggering for start words
    can call to spit out sentences

    the meat of this:
        makes a dictionary keyed by n-1 grams with values as instances of the "gram" class
        to generate sentences, etc., we can just quickly "ask" the appropriate "gram" instance for a next word
        this works quickly because the dict lookup is basically free
        then, the determinig of the next word is relatively quick (see gram.gen)

    only loops through data once for setup
    '''
    def __init__(self, n, text_as_list):
        self.n = n
        self.text = text_as_list
        self.vocab = set(text_as_list)
        self.conditionals = {}

        self.process()

    @timer
    def process(self):
        word_range = range(self.n-1, -1, -1)

        #make unsmoothed
        for token_num in range(len(self.text)):
            temp = list()

            for countdown in word_range:
                #iterates up
                temp.append(lookup_default(self.text, token_num - countdown, '<s>'))

            current_word = temp.pop()

            try: 
                self.conditionals[tuple(temp)].add_next(current_word)
            except:
                self.conditionals[tuple(temp)] = gram(tuple(temp), current_word)

    @timer
    def gen(self, laplace_smoothing = False):
        ''' 
        function to generate a sentence

        extremely straightfoward. 

        start items need to be fixed: right now we just pick a random n-1 tuple that begins with <s>

        then, run a while loop stopping if the prev character was .!?

        note for laplace smoothing: we run into the possibility of the appropriate (n-1) gram not being in the text
            this is the try/except block on line 243-5

            in this case, we just pick a random word
        '''


        sentence = []

        if self.n > 1:
            begin_candidates = []

            for k, v in self.conditionals.iteritems():
                if k[0] == '<s>':
                    begin_candidates.append(k)

            current = random.choice(begin_candidates)
            for word in current:
                sentence.append(word)
        else:
            sentence.append(self.conditionals[()].random_next())


        while sentence[-1] != '.' and sentence[-1] != '?' and sentence[-1] != '!':
            if self.n > 1:
                if laplace_smoothing == False:
                    s = self.conditionals[tuple(current)].random_next()
                else:
                    try:
                        s = self.conditionals[tuple(current)].laplace_next(self.vocab)
                    except: #maybe our n-1 gram isn't in the text. assume all words are equally likely
                        #reduces to 1/V, or a random choice from the vocab words
                        s = random.choice(list(self.vocab))

                sentence.append(s)
                current = sentence[-self.n+1:]
            else:
                if laplace_smoothing == False:
                    s = self.conditionals[()].random_next() #for unigram
                else:
                    s = self.conditionals[()].laplace_next(self.vocab)
                sentence.append(s)
                current = tuple()
                
        return ' '.join(sentence)


    def perplexity(self, corpus_as_list):


        pass



if __name__ == "__main__":
    t = clean_up(dos)

    #tt = ngrams(1, t)

    #for each in range(3):
    #    print('~~~~~')
    #    print(tt.gen() + '\n')

    tt = ngrams(2, t)
    for each in range(1):
        print(tt.gen(laplace_smoothing = False) + '\n')

    tt = ngrams(3, t)

    for each in range(1):
        print(tt.gen(laplace_smoothing=False) + '\n')


'''
1) change line 39 to interpolate n-1 "<s>" tokens at the start of the text and between each sentence
2)
'''
