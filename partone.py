'''
@by George Berry
@geb97@cornell.edu
@created 2/8/2014
'''

import re, time
from random import choice
import random
import sys


#open file
#biblefile = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/proj1/bible_corpus 2/kjbible.train'

#hotelfile = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/proj1/reviews.train'


#with open(biblefile, 'rb') as f:
#    bible = f.read()

#with open(hotelfile, 'rb') as f:
#    hotel = f.read()


def clean_up(text, n):

    with open(text, 'rb') as f:
        text = f.read()

    '''
    simply cleans text with regex subs.
    assumptions: we want punctuation as its own word; everything lowercase; sentence start tokens but no end tokens (.?!) are end tokens.
    except '
    e.g. we preserve contractions
    '''

    '''
    need to change this to avoid splitting up things like e.e. and etc.

    maybe only split on periods if there is a space then a capital letter???

    '''

    #remove xml tags
    text = re.sub(r'<.*>','',text)
    #replace verse numbers (beginning of sentence) with <s> tag
    text = re.sub(r'[0-9]+[:,][0-9]+(,*)\s*','', text)
    #replace newlines with spaces (we do a space split below)
    text = re.sub(r'[\n]+',' ', text)
    #replace punctuation with a space then that punctuation

    text = re.sub(r'[^\w\s\'\.!\?]', ' \g<0> ', text) #puts space around punctuation except sentence enders

    text = re.sub(r'(\.+)|(!+)|(\?+)', ' \g<0> </s> ', text) #puts an end sentence token after each sentence along with whitespace.

    #specificially for slashes, which can cause some trouble
    #text = re.sub(r'(\w|\s)(/+)(\w|\s)', '\g<1> \g<2> \g<3>', text)

    for x in range(n-1):
        text = re.sub('</s>', '</s> <s>', text)

    text = text.lower().strip()
    text = re.split(r' +', text)

    for x in range(n-1):
        text.insert(0, '<s>')

    return text


def lookup_default(L, index, default):
    '''
    for large number of calls to lists that may or may not have items
    allows easy specification of a default
    '''
    try:
        return L[index]
    except:
        return default


#classes

class gram:
    '''
    holds information for a single tuple of length n-1
    think of this as the thing conditioned on in P(w_n|w_n-1,w_n-2)
    can add w_n's 1 at a time
        the total distribution is stored in this object

    can call the object to get a probabalistically-determined next word

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

        #total number of subsequent words
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
        self.vocab = list(set(text_as_list))
        self.conditionals = {}

        self.process()

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

    def gen(self):
        ''' 
        function to generate a sentence

        extremely straightfoward. 

        start items need to be fixed: right now we just pick a random n-1 tuple that begins with <s>

        then, run a while loop stopping if the prev character was .!?

        note for laplace smoothing: we run into the possibility of the appropriate (n-1) gram not being in the text
            this is the try/except block on line 243-5

            in this case, we just pick a random word
        '''

        if self.n > 1:
            sentence = ['<s>']*(self.n-1)
        else:
            sentence = ['<s>']

        current = sentence[-self.n+1:]

        while sentence[-1] != '</s>':
            if self.n > 1:
                s = self.conditionals[tuple(current)].random_next()

                sentence.append(s)

                current = sentence[-self.n+1:]
            else:
                s = self.conditionals[()].random_next() #for unigram
                
                sentence.append(s)

                current = tuple()
                
        return ' '.join(sentence)


if __name__ == "__main__":
    n = int(sys.argv[1])
    f = sys.argv[2]

    t = clean_up(f, n)

    tt = ngrams(n, t)

    for each in range(5):
        print(tt.gen() + '\n')