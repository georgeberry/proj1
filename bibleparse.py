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
biblefile = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/proj1/bible_corpus 2/kjbible.train'

with open(biblefile, 'r') as f:
    bible = f.read()


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
    #remove xml tags
    text = re.sub(r'<.*>','',text)
    #replace verse numbers (beginning of sentence) with <s> tag
    text = re.sub(r'[0-9]+:[0-9]+\s*','', text)
    #replace newlines with spaces (we do a space split below)
    text = re.sub(r'[\n]+',' ', text)
    #replace punctuation with a space then that punctuation
    text = re.sub(r"[^\w\s\']+", " \g<0>", text)
    text = re.sub(r'\.', '. </s> <s>', text)
    text = text.lower().strip()
    text = re.split(r' +', text)
    text.append('</s>') #just need this
    #we automatically put <s> at the start with make_ngrams function
    return text


#helpers
def lookup_default(list, index, default):
    try:
        return list[index]
    except:
        return default


#classes

class gram:
    '''
    holds a tuple of n-1 previous words
    holds a dictionary of next words, keyed by word with values as counts
    has a function to return a random next word
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
        if not isinstance(word, str):
            return 'must input string'
        try:
            self.next_words[word] += 1
        except:
            self.next_words[word] = 1

    def next(self, word_dict):
        self.cum_sum = sum(w for c, w in word_dict.iteritems())
        r = random.uniform(0, self.cum_sum)
        left_point = 0
        for c, w in word_dict.iteritems():
            if left_point + w > r:
                return c
            left_point += w
        assert False, "wtf"

    def random_next(self):
        return self.next(self.next_words)

    @staticmethod
    def seen_words(word_dict):
        return set(c for c, w in word_dict.iteritems())

    @staticmethod
    def count(word_dict):
        return sum(w for c, w in word_dict.iteritems())

    #smooth
    #idea here: w_n-1 has occured a times
    #we can get this from summing the total occurances of following words, w
    #then, we make a choice:
    #pick a word in the "seen words" dict with probability, a+set(elements of a)/(a+V), then choose word normally
    #OR, with prob 1 - a+set(elements of a)/(a+V), pick a word uniformly at random from the words we HAVEN'T seen
    def laplace_next(self, words_in_vocab):
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

        #print begin_candidates
        #print self.conditionals
        #if self.n == 1:
        #    current = '<s>'

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
    t = clean_up(bible)
    tt = ngrams(1, t)

    for each in range(3):
        print('~~~~~')
        print(tt.gen() + '\n')

    tt = ngrams(2, t)
    for each in range(3):
        print(tt.gen(laplace_smoothing = True) + '\n')

    tt = ngrams(2, t)
    for each in range(3):
        print(tt.gen() + '\n')



'''
checks

a = gram('dude')
for each in range(10):
    a.add_next('bro')
for each in range(90):
    a.add_next('brodude')

counts = {}
for each in range(10000):
    temp =a.random_next()
    try:
        counts[temp] += 1
    except:
        counts[temp] = 1

print sum(counts.values())
print counts
'''