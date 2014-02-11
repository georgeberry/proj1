'''
@by George Berry
@geb97@cornell.edu
@created 2/8/2014
'''

import re, time
from functools import wraps

#open file
biblefile = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/proj1/bible_corpus 2/kjbible.train'

with open(biblefile, 'rb') as f:
    bible = f.read()


def timer(f):
    @wraps(f)
    def wrapper(*args,**kwargs):
        tic = time.time()
        result = f(*args, **kwargs)
        print f.__name__ + " took " + str(time.time() - tic) + " seconds"
        return result
    return wrapper


class n_grams:
    def __init__(self, text_as_string, n, prob = True):
        self.text = text_as_string
        self.n = n
        self.n_grams_dict = {}
        self.prob = {}
        self.total_words = 0

        #kinda hacky, just run this at the beginning
        self.clean_up()
        self.make_n_grams()
        if prob == True:
            self.calc_prob(text_as_string)


    #sub for <WORDS>
    #sub for 1:1
    #get rid of newlines
    #lowercase-ize
    @timer
    def clean_up(self):
        #remove xml tags
        self.text = re.sub(r'<.*>','',self.text)
        #replace verse numbers (beginning of sentence) with <s> tag
        self.text = re.sub(r'[0-9]+:[0-9]+\s*','', self.text)
        #replace newlines with spaces (we do a space split below)
        self.text = re.sub(r'[\n]+',' ', self.text)
        #replace punctuation with a space then that punctuation
        self.text = re.sub(r"[^\w\s\']+", " \g<0>", self.text)
        self.text = re.sub(r'\.', '. </s> <s>', self.text)
        self.text = self.text.lower().strip()
        self.text = re.split(r' +', self.text)
        self.text.append('</s>') #just need this
        #we automatically put <s> at the start with make_ngrams function


    ##make ngrams
    @timer
    def make_n_grams(self):

        range_n = range((self.n-1), -1, -1)
        self.total_words = len(self.text)

        #initialize with repeated <s> at the beginning
        for token_num in range(len(self.text)):
            n_gram = []

            #count backwards from n-1 to 0
            #gives the word at position token_num, along with the previous n-1 words
            for countdown in range_n:
                try:
                    n_gram.append(self.text[token_num - countdown])
                except:
                    n_gram.append('<s>')
                #n_gram.append(word_or_start_token(text_list, token_num - countdown))
            try:
                self.n_grams_dict[tuple(n_gram)] += 1
            except:
                self.n_grams_dict[tuple(n_gram)] = 1


    #ideally, we want to do this in 2 passes for all
    @timer
    def calc_prob(self, text_as_string):
        #if n == 1, simple division
        if self.n == 1:
            self.prob = self.n_grams_dict
            for k, v in self.prob.iteritems():
                self.prob[k] = float(v)/float(self.total_words)
        else:
            self.prob = self.n_grams_dict
            n_minus_one_dict = n_grams(text_as_string, self.n - 1, prob = False)
            n_minus_one_dict = n_minus_one_dict.n_grams_dict

            for n_gram in self.prob.keys():
                n_minus_one_gram = n_gram[1:]
                self.prob[n_gram] = float(self.prob[n_gram])/float(n_minus_one_dict[n_minus_one_gram])


            




'''def word_or_start_token(text_list, token_num):
    if token_num < 0:
        return '<s>'
    else:
        return text_list[token_num]'''




if __name__ == "__main__":
    a = n_grams(bible,1)
    b = n_grams(bible,2)
    c = n_grams(bible,3)
    d = n_grams(bible,4)
